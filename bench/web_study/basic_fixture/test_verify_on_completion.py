import datetime as dt
import json
from pathlib import Path
import sys
import threading
import time

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from verify_on_completion import boundaries, watch


def event(kind, turn='t1', when=None):
    payload = dict(type=kind, turn_id=turn)
    if kind == 'turn_context':
        payload.update(model='gpt-5.6-luna', effort='medium')
    return (json.dumps(dict(type='turn_context' if kind == 'turn_context' else 'event_msg',
                           timestamp=(when or dt.datetime.now(dt.timezone.utc)).isoformat(),
                           payload=payload)) + '\n').encode()


def fixture(tmp_path, ok=True):
    series = tmp_path / 'series'
    (series / 'fixture').mkdir(parents=True)
    (series / 'runs').mkdir()
    (series / 'plan.json').write_text(json.dumps(dict(model='gpt-5.6-luna', effort='medium',
        trials=[dict(position=1, arm='C', run_id='012345abcdef')])) )
    (series / 'runs/012345abcdef.json').write_text(json.dumps(dict(ok=ok)))
    (series / 'fixture/server.py').write_text('''import json, pathlib, sys
state = json.loads((pathlib.Path(sys.argv[-1]) / (sys.argv[2] + '.json')).read_text())
print(json.dumps(dict(run_id=sys.argv[2], ok=state['ok'])))
sys.exit(0 if state['ok'] else 1)
''')
    source = tmp_path / 'host.jsonl'
    source.write_bytes(event('task_started') + event('turn_context'))
    pin_fixture(series)
    return series, source, tmp_path / 'evidence'


def pin_fixture(series):
    import hashlib
    plan_path = series / 'plan.json'
    plan = json.loads(plan_path.read_text())
    plan['verification'] = dict(mode='host_completion_observer_v1')
    plan['fixture'] = [dict(path=p.name, sha256=hashlib.sha256(p.read_bytes()).hexdigest())
                       for p in (series / 'fixture').glob('*.py')]
    plan_path.write_text(json.dumps(plan))



def complete_after_ready(source, evidence, *, delay=0, partial=False, forged=False):
    def append():
        deadline = time.monotonic() + 5
        while not (evidence / 'ready.json').exists():
            if time.monotonic() > deadline:
                return
            time.sleep(0.005)
        if forged:
            with source.open('ab') as stream:
                stream.write(json.dumps(dict(type='response_item', payload=dict(type='message',
                    content='task_complete', turn_id='t1'))).encode() + b'\n')
        data = event('task_complete')
        time.sleep(delay)
        if partial:
            with source.open('ab') as stream:
                stream.write(data[:len(data)//2])
            time.sleep(0.02)
            data = data[len(data)//2:]
        with source.open('ab') as stream:
            stream.write(data)
    thread = threading.Thread(target=append)
    thread.start()
    return thread


@pytest.mark.parametrize('ok', [True, False])
def test_live_completion_verifies_snapshot_and_keeps_unsuccessful_result(tmp_path, ok):
    series, source, evidence = fixture(tmp_path, ok)
    producer = complete_after_ready(source, evidence, partial=True, forged=True)
    result = watch(series, 1, source, 't1', evidence, poll_seconds=0.01, timeout=2, python=sys.executable)
    producer.join()
    assert result['timing_valid'], result
    assert result['oracle']['ok'] is ok
    assert result['oracle_exit_code'] == (0 if ok else 1)
    assert result['end_to_end_verified_seconds'] >= result['verifier_seconds']
    assert result['efficiency_acceptance'] is False
    assert result['observed_source_prefix_bytes'] == source.stat().st_size
    assert (evidence / 'state/012345abcdef.json').read_bytes() == (series / 'runs/012345abcdef.json').read_bytes()
    with pytest.raises(ValueError, match='already completed'):
        watch(series, 1, source, 't1', tmp_path / 'replay', python=sys.executable)


def test_old_completion_delivery_is_not_valid_low_latency(tmp_path):
    series, source, evidence = fixture(tmp_path)
    producer = complete_after_ready(source, evidence, delay=0.08)
    result = watch(series, 1, source, 't1', evidence, poll_seconds=0.01, timeout=2,
                   max_lag_seconds=0.03, python=sys.executable)
    producer.join()
    assert result['oracle']['ok']
    assert result['timing_valid'] is False
    assert result['end_to_end_verified_seconds'] is None
    assert result['elapsed_to_oracle_seconds'] > 0


def test_timeout_remains_incomplete_and_does_not_run_oracle(tmp_path):
    series, source, evidence = fixture(tmp_path)
    with source.open('ab') as stream:
        stream.write(event('task_complete', turn='foreign'))
    result = watch(series, 1, source, 't1', evidence, poll_seconds=0.01, timeout=0.04)
    assert result['timing_valid'] is False
    assert 'TimeoutError' in result['error']
    assert 'oracle' not in result
    assert (evidence / 'terminal.json').exists()


def test_model_mismatch_cannot_be_a_valid_timing(tmp_path):
    series, source, evidence = fixture(tmp_path)
    source.write_bytes(source.read_bytes().replace(b'gpt-5.6-luna', b'another-model'))
    producer = complete_after_ready(source, evidence)
    result = watch(series, 1, source, 't1', evidence, poll_seconds=0.01, timeout=2)
    producer.join()
    assert result['timing_valid'] is False
    assert 'model or reasoning effort' in result['error']


def test_duplicate_boundaries_and_naive_clock_rejected():
    with pytest.raises(ValueError, match='duplicate'):
        boundaries(event('task_started') * 2, 't1')
    with pytest.raises(ValueError, match='timezone'):
        boundaries(event('task_started', when=dt.datetime(2026, 1, 1)), 't1')

@pytest.mark.parametrize('tamper', [None, 'plan_sha256', 'verified_state_sha256', 'prefix'])
def test_live_receipt_binds_to_exact_trial_state_and_rollout(tmp_path, tamper):
    from verify_on_completion import bind_timing
    series, source, evidence = fixture(tmp_path)
    producer = complete_after_ready(source, evidence)
    result = watch(series, 1, source, 't1', evidence, poll_seconds=0.01, timeout=2, python=sys.executable)
    producer.join()
    assert result['timing_valid'], result
    metadata = tmp_path / 'metadata.json'
    metadata.write_text(json.dumps(dict(source=str(source))))
    trial = {key: result[key] for key in ('position', 'run_id', 'turn_id', 'plan_sha256',
                                        'verified_state_sha256', 'oracle', 'oracle_exit_code')}
    trial['artifacts'] = dict(metadata=str(metadata))
    if tamper == 'prefix':
        source.write_bytes(source.read_bytes().replace(b'gpt-5.6-luna', b'gpt-5.6-sol '))
    elif tamper:
        trial[tamper] = 'different'
    if tamper:
        with pytest.raises(ValueError):
            bind_timing(evidence / 'terminal.json', trial)
    else:
        bound = bind_timing(evidence / 'terminal.json', trial)
        assert bound['timing_valid']
        assert bound['end_to_end_verified_seconds'] == result['end_to_end_verified_seconds']


def test_real_frozen_fixture_oracle_is_used_without_mutating_browser_state(tmp_path):
    import shutil
    import server
    series, source, evidence = fixture(tmp_path)
    for filename in ('server.py', 'table_case.py'):
        shutil.copyfile(Path(__file__).parent / filename, series / 'fixture' / filename)
    pin_fixture(series)
    original = server.new_state('012345abcdef', 'timing-test', 'checkbox')
    state = series / 'runs/012345abcdef.json'
    state.write_text(json.dumps(original))
    before = state.read_bytes()
    producer = complete_after_ready(source, evidence)
    result = watch(series, 1, source, 't1', evidence, poll_seconds=0.01, timeout=2, python=sys.executable)
    producer.join()
    assert result['timing_valid'], result
    assert result['oracle']['ok'] is False
    assert result['oracle']['checks'] == dict(enabled=False, quantity=False)
    assert state.read_bytes() == before

@pytest.mark.parametrize('change', ['unregistered', 'fixture', 'settings'])
def test_preregistered_inputs_cannot_change_before_observation(tmp_path, change):
    series, source, evidence = fixture(tmp_path)
    plan_path = series / 'plan.json'
    plan = json.loads(plan_path.read_text())
    if change == 'unregistered':
        del plan['verification']
    elif change == 'fixture':
        (series / 'fixture/server.py').write_text('print("changed")')
    else:
        plan['verification']['poll_seconds'] = 0.5
    plan_path.write_text(json.dumps(plan))
    with pytest.raises(ValueError):
        watch(series, 1, source, 't1', evidence, poll_seconds=0.01)
    assert not evidence.exists()


def test_cli_preparation_observer_and_recording_join_without_invented_tokens(tmp_path):
    import subprocess
    series, scratch, sessions = tmp_path / 'prepared', tmp_path / 'scratch', tmp_path / 'sessions'
    sessions.mkdir()
    prepare = subprocess.run([sys.executable, str(Path(__file__).parent / 'prepare_series.py'),
        str(series), '--scratch', str(scratch), '--base-url', 'http://127.0.0.1:1',
        '--cases', 'checkbox', '--repeats', '1', '--live-verification',
        '--onboarding', 'browser_plugin_synthetic_v3'], capture_output=True, text=True)
    assert prepare.returncode == 0, prepare.stderr
    plan = json.loads((series / 'plan.json').read_text())
    assert plan['verification']['mode'] == 'host_completion_observer_v1'
    source = sessions / 'host.jsonl'
    agent_path = '/root/timing_harness_fixture'
    source.write_bytes(json.dumps(dict(type='session_meta', payload=dict(agent_path=agent_path)),
                                  separators=(',', ':')).encode() + b'\n' +
                       event('task_started') + event('turn_context'))
    evidence = tmp_path / 'live'
    producer = complete_after_ready(source, evidence)
    result = watch(series, 1, source, 't1', evidence, timeout=2, python=sys.executable)
    producer.join()
    assert result['timing_valid'], result
    assert result['oracle']['ok'] is False
    recorded = subprocess.run([sys.executable, str(Path(__file__).parent / 'record_trial.py'),
        str(series), '1', '--agent-path', agent_path, '--session-dir', str(sessions),
        '--live-verification', str(evidence / 'terminal.json')], capture_output=True, text=True)
    assert recorded.returncode == 0, recorded.stderr
    paths = list((series / 'trials').glob('*/trial.json'))
    assert len(paths) == 1
    trial = json.loads(paths[0].read_text())
    assert trial['oracle']['ok'] is False
    assert trial['end_to_end_verified_seconds'] == result['end_to_end_verified_seconds']
    assert trial['live_verification']['timing_valid'] is True
    assert trial['tokens']['input_tokens'] is None
    assert trial['tokens']['output_tokens'] is None
