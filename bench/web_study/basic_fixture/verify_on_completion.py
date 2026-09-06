"""Observe a live host turn and immediately verify its frozen fixture state.

Read-only toward the browser and rollout. Never infer completion from assistant
text, replay a completed trial as live, or count an unsuccessful oracle as a win.
"""
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import subprocess
import time


def utc():
    return dt.datetime.now(dt.timezone.utc)


def timestamp(text):
    value = dt.datetime.fromisoformat(text.replace('Z', '+00:00'))
    if value.tzinfo is None:
        raise ValueError('host timestamp must contain a timezone')
    return value


def boundaries(raw, turn_id):
    """Only record-owned host events count; ignore partial final JSONL lines."""
    found = {}
    for number, line in enumerate(raw.splitlines(keepends=True), 1):
        if not line.endswith(b'\n'):
            continue
        record = json.loads(line)
        payload = record.get('payload', {})
        if not isinstance(payload, dict) or payload.get('turn_id') != turn_id:
            continue
        kind = payload.get('type') if record.get('type') == 'event_msg' else record.get('type')
        if kind not in ('task_started', 'task_complete', 'turn_context'):
            continue
        if kind in found:
            raise ValueError('duplicate owned host boundary: ' + kind)
        found[kind] = dict(timestamp=record['timestamp'], line=number)
        timestamp(record['timestamp'])
        if kind == 'turn_context':
            found[kind].update(model=payload.get('model'), effort=payload.get('effort'))
    return found


def watch(series, position, source, turn_id, evidence, poll_seconds=0.1, timeout=900,
          max_lag_seconds=2, python='/usr/bin/python3'):
    if not 0.01 <= poll_seconds <= 1 or timeout <= 0 or max_lag_seconds <= 0:
        raise ValueError('invalid observation bounds')
    series, source, evidence = map(Path, (series, source, evidence))
    plan_raw = (series / 'plan.json').read_bytes()
    plan = json.loads(plan_raw)
    if plan.get("verification", {}).get("mode") != "host_completion_observer_v1":
        raise ValueError("live verification must be preregistered; no historical backfill")
    for key, value in (("poll_seconds", poll_seconds), ("max_lag_seconds", max_lag_seconds),
                       ("max_clock_drift_seconds", 0.25)):
        if plan["verification"].get(key, value) != value:
            raise ValueError("observer setting differs from preregistration: " + key)
    pins = plan.get("fixture", [])
    if not pins or not any(pin["path"] == "server.py" for pin in pins):
        raise ValueError("frozen fixture hashes are required")
    def fixture_matches():
        root = (series / "fixture").resolve()
        for pin in pins:
            path = (root / pin["path"]).resolve()
            if root not in path.parents or hashlib.sha256(path.read_bytes()).hexdigest() != pin["sha256"]:
                return False
        return True
    if not fixture_matches():
        raise ValueError("frozen fixture changed before observation")
    matches = [t for t in plan['trials'] if t['position'] == position]
    if len(matches) != 1:
        raise ValueError('require exactly one preregistered trial')
    trial = matches[0]
    raw = source.read_bytes()
    offset = len(raw)
    prefix_hash = hashlib.sha256(raw)
    complete_prefix, separator, pending = raw.rpartition(b"\n")
    complete_prefix = complete_prefix + separator
    line_offset = complete_prefix.count(b"\n")
    initial = boundaries(complete_prefix, turn_id)
    if 'task_started' not in initial or 'task_complete' in initial:
        raise ValueError('observer requires a started, not already completed host turn')
    evidence.mkdir(parents=True, exist_ok=False)
    state_path = series / 'runs' / (trial['run_id'] + '.json')
    verifier = series / 'fixture' / 'server.py'
    verifier_hash = hashlib.sha256(verifier.read_bytes()).hexdigest()
    source_identity = (source.stat().st_dev, source.stat().st_ino)
    begin_utc, begin_mono = utc(), time.monotonic()
    terminal = dict(schema='greppy.web-study.live-verification.v1', position=position,
                    run_id=trial['run_id'], turn_id=turn_id, source=str(source.resolve()),
                    plan_sha256=hashlib.sha256(plan_raw).hexdigest(),
                    observer_started=begin_utc.isoformat(), poll_seconds=poll_seconds,
                    verifier_sha256=verifier_hash, timing_valid=False,
                    end_to_end_verified_seconds=None, efficiency_acceptance=False,
                    time_scope='Host task_started through independent oracle completion; includes polling and verifier overhead, excludes pre-task dispatch queue.',
                    limitations=['No runtime/model load control is established by this observer.',
                                 'Host timestamp delivery lag is measured and gated; no latency is subtracted.'])
    (evidence / 'ready.json').write_text(json.dumps(terminal, indent=2) + '\n')
    events = dict(initial)
    del raw
    try:
        while True:
            current = source.stat()
            if (current.st_dev, current.st_ino) != source_identity or current.st_size < offset:
                raise ValueError('rollout replaced or truncated during observation')
            with source.open('rb') as stream:
                stream.seek(offset)
                appended = stream.read()
            offset += len(appended)
            prefix_hash.update(appended)
            complete_prefix, separator, pending = (pending + appended).rpartition(b'\n')
            complete_prefix += separator
            added = boundaries(complete_prefix, turn_id)
            if events.keys() & added.keys():
                raise ValueError('duplicate owned host boundary')
            events.update({kind: dict(event, line=event['line'] + line_offset)
                           for kind, event in added.items()})
            line_offset += complete_prefix.count(b'\n')

            if 'task_complete' in events:
                detected = utc()
                break
            if time.monotonic() - begin_mono >= timeout:
                raise TimeoutError('host turn still incomplete; no success inferred')
            time.sleep(poll_seconds)
        context = events.get('turn_context', {})
        if context.get('model') != plan['model'] or context.get('effort') != plan['effort']:
            raise ValueError('host model or reasoning effort differs from preregistration')
        started = timestamp(events['task_started']['timestamp'])
        completed = timestamp(events['task_complete']['timestamp'])
        lag = (detected - completed).total_seconds()
        if not started <= completed <= detected:
            raise ValueError('host boundary timestamps are out of order')
        terminal.update(boundaries=events, completion_detected=detected.isoformat(),
                        completion_detection_lag_seconds=lag,
                        observed_source_prefix_bytes=offset,
                        observed_source_prefix_sha256=prefix_hash.hexdigest())
        snapshot = state_path.read_bytes()
        snapshot_dir = evidence / 'state'
        snapshot_dir.mkdir()
        (snapshot_dir / state_path.name).write_bytes(snapshot)
        argv = [python, str(verifier), 'verify-run', trial['run_id'], '--run-dir', str(snapshot_dir)]
        verify_start = time.monotonic()
        result = subprocess.run(argv, capture_output=True, text=True, timeout=30)
        verified = utc()
        terminal.update(verifier_argv=argv, oracle_exit_code=result.returncode,
                        oracle_stdout=result.stdout, oracle_stderr=result.stderr,
                        verifier_seconds=time.monotonic() - verify_start,
                        verified_at=verified.isoformat(),
                        verified_state_sha256=hashlib.sha256(snapshot).hexdigest())
        oracle = json.loads(result.stdout)
        if not isinstance(oracle.get('ok'), bool) or oracle.get('run_id') != trial['run_id']:
            raise ValueError('invalid independent oracle response')
        if result.returncode != (0 if oracle['ok'] else 1):
            raise ValueError('oracle exit status and result disagree')
        terminal['oracle'] = oracle
        terminal['fixture_state_unchanged_during_verification'] = snapshot == state_path.read_bytes()
        terminal['verifier_unchanged'] = verifier_hash == hashlib.sha256(verifier.read_bytes()).hexdigest()
        terminal['plan_unchanged'] = plan_raw == (series / 'plan.json').read_bytes()
        terminal['frozen_fixture_unchanged'] = fixture_matches()
        # Compare both clocks at the same endpoint. Integrity hashes above can
        # take time after `verified`; that work is not a wall-clock step.
        # Keep the oracle completion boundary and drift threshold unchanged.
        clock_check_utc, clock_check_mono = utc(), time.monotonic()
        drift = abs((clock_check_utc - begin_utc).total_seconds()
                    - (clock_check_mono - begin_mono))
        terminal['observer_clock_drift_seconds'] = drift
        terminal['timing_valid'] = (lag <= max_lag_seconds and drift <= 0.25 and
            terminal['fixture_state_unchanged_during_verification'] and terminal['verifier_unchanged'] and
            terminal['plan_unchanged'] and terminal['frozen_fixture_unchanged'])
        terminal['elapsed_to_oracle_seconds'] = (verified - started).total_seconds()
        if terminal['timing_valid']:
            terminal['end_to_end_verified_seconds'] = terminal['elapsed_to_oracle_seconds']
    except Exception as error:
        terminal['error'] = repr(error)
    finally:
        terminal['observer_wall_seconds'] = time.monotonic() - begin_mono
        with (evidence / 'terminal.json').open('x') as output:
            json.dump(terminal, output, indent=2)
            output.write('\n')
    return terminal


def bind_timing(receipt_path, trial_record):
    """Bind a prospective timing receipt to the later full trace/oracle export."""
    receipt = json.loads(Path(receipt_path).read_text())
    if receipt.get('schema') != 'greppy.web-study.live-verification.v1':
        raise ValueError('unknown live verification schema')
    for key in ('position', 'run_id', 'turn_id', 'plan_sha256'):
        if receipt.get(key) != trial_record.get(key):
            raise ValueError('live verification binding mismatch: ' + key)
    binding = dict(path=str(Path(receipt_path).resolve()), timing_valid=False,
                   end_to_end_verified_seconds=None, time_scope=receipt['time_scope'])
    if receipt.get('timing_valid') is not True:
        binding['reason'] = receipt.get('error', 'observation quality gate did not pass')
        return binding
    for key in ('verified_state_sha256', 'oracle', 'oracle_exit_code'):
        if receipt.get(key) != trial_record.get(key):
            raise ValueError('live verification state differs from exported trial: ' + key)
    metadata = json.loads(Path(trial_record['artifacts']['metadata']).read_text())
    source = Path(metadata['source'])
    if source.resolve() != Path(receipt['source']).resolve():
        raise ValueError('live verification rollout differs from exported trace')
    size = receipt['observed_source_prefix_bytes']
    if not isinstance(size, int) or size <= 0 or size > source.stat().st_size:
        raise ValueError('invalid observed rollout prefix length')
    with source.open('rb') as stream:
        digest = hashlib.sha256(stream.read(size)).hexdigest()
    if digest != receipt['observed_source_prefix_sha256']:
        raise ValueError('observed rollout prefix changed')
    seconds = receipt['end_to_end_verified_seconds']
    if not isinstance(seconds, (float, int)) or isinstance(seconds, bool) or not 0 <= seconds < float('inf'):
        raise ValueError('invalid verified duration')
    binding.update(timing_valid=True, end_to_end_verified_seconds=seconds,
                   completion_detection_lag_seconds=receipt['completion_detection_lag_seconds'],
                   verifier_seconds=receipt['verifier_seconds'])
    return binding


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('series', type=Path)
    p.add_argument('position', type=int)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--turn-id', required=True)
    p.add_argument('--evidence', type=Path, required=True)
    p.add_argument('--timeout', type=float, default=900)
    a = p.parse_args()
    result = watch(a.series, a.position, a.source, a.turn_id, a.evidence, timeout=a.timeout)
    print(json.dumps({k: result.get(k) for k in ('timing_valid', 'oracle', 'error', 'end_to_end_verified_seconds')}))
    raise SystemExit(0 if result['timing_valid'] else 1)
