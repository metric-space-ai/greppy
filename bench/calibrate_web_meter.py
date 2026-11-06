#!/usr/bin/env python3
"""Bounded Linux calibration of the actual web-efficiency meter.

No browser/build/install. Run only in an admitted benchmark lane. Raw receipts,
a schedule fixed before execution and every sample are retained. Qualification
covers the declared fixture, never unsampled peaks or arbitrary agent workloads.
"""
import argparse
from contextlib import nullcontext
import hashlib
import importlib.util
import itertools
import json
import math
import mmap
import os
from pathlib import Path
import random
import select
import secrets
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
METER = ROOT / 'crates/web-runtime/scripts/measure-web-efficiency.py'
MIB = 1024 * 1024
OPERATIONS = 160000
HOLD_SECONDS = 0.75


class CalibrationFailure(RuntimeError):
    pass


def require(condition, name, detail):
    if not condition:
        raise CalibrationFailure(name + ': ' + detail)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def work():
    token = b'greppy-meter-calibration-v1'
    block = b'x' * 1024
    for _ in range(OPERATIONS):
        token = hashlib.sha256(token + block).digest()
    return token.hex()


def fixture(mode):
    """One known child, no descendants; physical private pages are touched."""
    began_cpu = time.process_time_ns()
    phases = []
    allocations = []
    checksum = work()

    def phase(name, seconds):
        start = time.monotonic_ns()
        time.sleep(seconds)
        phases.append({'name': name, 'start_ns': start, 'end_ns': time.monotonic_ns()})

    def allocate(size):
        block = mmap.mmap(-1, size, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
        for offset in range(0, size, mmap.PAGESIZE):
            block[offset] = 1
        allocations.append(block)

    burst = None
    try:
        if mode == 'cost':
            phase('cost_hold', 0.6)
        else:
            phase('baseline', HOLD_SECONDS)
            if mode == 'steps':
                allocate(8 * MIB)
                phase('held_8_mib', HOLD_SECONDS)
                allocate(24 * MIB)
                phase('held_32_mib', HOLD_SECONDS)
            else:
                burst_start = time.monotonic_ns()
                allocate(32 * MIB)
                phase('held_burst', 0.02)
                burst = {'allocation_started_ns': burst_start,
                         'fully_touched_ns': phases[-1]['start_ns']}
            for block in allocations:
                block.close()
            allocations.clear()
            if burst is not None:
                burst['released_ns'] = time.monotonic_ns()
            phase('released', HOLD_SECONDS)
    finally:
        for block in allocations:
            block.close()
    print(json.dumps({'schema': 'greppy.web-meter-fixture.v1', 'mode': mode,
                      'operations': OPERATIONS, 'checksum': checksum,
                      'self_cpu_ns': time.process_time_ns() - began_cpu,
                      'phases': phases, 'burst': burst}), flush=True)


def load_meter():
    spec = importlib.util.spec_from_file_location('calibrated_web_meter', METER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parsed_fixture(text, expected_checksum, mode):
    value = json.loads(text)
    require(value.get('schema') == 'greppy.web-meter-fixture.v1'
            and value.get('mode') == mode
            and value.get('operations') == OPERATIONS
            and value.get('checksum') == expected_checksum,
            'calibration.fixture_identity', 'fixture work or result changed')
    require(type(value.get('self_cpu_ns')) is int and value['self_cpu_ns'] > 0,
            'calibration.fixture_cpu', 'fixture CPU telemetry is missing')
    return value


def execute_case(meter, directory, mode, observed, expected_checksum, deadline, kernel_parent=None):
    remaining = deadline - time.monotonic()
    require(remaining > 0, 'calibration.deadline', 'total calibration deadline reached')
    timeout = min(12, remaining)
    directory.mkdir()
    command = [sys.executable, '-B', str(Path(__file__).resolve()), '--fixture', mode]
    if observed:
        with (meter._memory_module.OwnedKernelGroup(kernel_parent, meter.cgroup_v2_envelope())
              if kernel_parent else nullcontext(None)) as resource_group:
            measured = meter.execute(command, directory, timeout, resource_group=resource_group)
            (directory / 'measured.json').write_text(json.dumps(measured, indent=2) + '\n')
        if resource_group is not None:
            measured['kernel_resource_accounting']['group_removed'] = resource_group.closed
        (directory / 'measured.json').write_text(json.dumps(measured, indent=2) + '\n')
        require(measured['exit_code'] == 0 and not measured['timed_out'],
                'calibration.fixture_exit', 'metered fixture failed or timed out')
        require(measured['process_custody']['supported']
                and measured['process_custody']['all_reaped'],
                'calibration.fixture_custody', 'Linux owned-child custody/reap was not verified')
        target = parsed_fixture((directory / 'stdout.log').read_text(), expected_checksum, mode)
        samples = [json.loads(line) for line in (directory / 'process-memory.jsonl').read_text().splitlines()]
        cpu = measured['user_cpu_seconds'] + measured['system_cpu_seconds']
        require(cpu + 0.005 >= target['self_cpu_ns'] / 1e9,
                'calibration.cpu_accounting', 'meter omitted real fixture CPU time')
        result = {'wall_ms': measured['process_cold_total_ms'], 'fixture': target,
                  'measured': measured, 'samples': samples}
    else:
        # This controlled fixture creates no descendants. A pidfd and wait reap
        # the actual child; bounded timeout cleanup never targets foreign PIDs.
        with ((directory / 'stdout.log').open('wb') as stdout,
              (directory / 'stderr.log').open('wb') as stderr):
            began = time.monotonic_ns()
            process = subprocess.Popen(command, stdout=stdout, stderr=stderr, start_new_session=True)
            descriptor = None
            try:
                # Same spawn/session/log destinations as execute(), with one
                # kernel exit notification instead of polling or Smaps scans.
                descriptor = os.pidfd_open(process.pid)
                require(bool(select.select([descriptor], [], [], timeout)[0]),
                        'calibration.reference_wait', 'reference fixture exceeded its deadline')
                process.wait()
                elapsed = (time.monotonic_ns() - began) / 1e6
            finally:
                if descriptor is not None:
                    os.close(descriptor)
                if process.poll() is None:
                    process.kill()
                    process.communicate(timeout=5)
        require(process.returncode == 0, 'calibration.fixture_exit', 'reference fixture failed')
        target = parsed_fixture((directory / 'stdout.log').read_text(), expected_checksum, mode)
        result = {'wall_ms': elapsed, 'fixture': target, 'measured': None, 'samples': []}
    (directory / 'receipt.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def phase_median(case, name, metric):
    phase = next(item for item in case['fixture']['phases'] if item['name'] == name)
    values = [sample[metric] for sample in case['samples']
              if sample.get('complete_for_observed_processes')
              and phase['start_ns'] <= sample['sample_started_ns']
              and sample['sample_finished_ns'] <= phase['end_ns']
              and sample.get(metric) is not None]
    require(bool(values), 'calibration.phase_coverage',
            'no complete sample wholly inside phase ' + name)
    return statistics.median(values)


def memory_contract(case):
    report = {}
    for metric, name in [('private_bytes', 'private'), ('pss_bytes', 'pss')]:
        baseline = phase_median(case, 'baseline', metric)
        steps = {}
        for phase, expected in [('held_8_mib', 8 * MIB), ('held_32_mib', 32 * MIB)]:
            delta = phase_median(case, phase, metric) - baseline
            require(abs(delta - expected) <= 2 * MIB, 'calibration.' + name + '_step',
                    f'{phase}: expected {expected} +/- {2 * MIB}, observed {delta}')
            steps[phase] = delta
        released = phase_median(case, 'released', metric) - baseline
        require(abs(released) <= 2 * MIB, 'calibration.' + name + '_release',
                f'closed mappings did not return near baseline: {released}')
        peak = case['measured']['process_family_memory']['peak_sampled_' + name + '_bytes']
        require(peak is not None and peak >= baseline + 30 * MIB,
                'calibration.' + name + '_peak_export', 'sampled peak export lost the held mapping')
        report[name] = {'baseline_bytes': baseline, 'step_deltas_bytes': steps,
                        'released_delta_bytes': released, 'peak_sampled_bytes': peak}
    return report


def kernel_delegation_contract(meter, parent):
    """A real pre-existing neighbor must prevent ownership of the delegation."""
    children_before = {path.name for path in parent.iterdir() if path.is_dir()}
    controls_before = (parent / 'cgroup.subtree_control').read_text()
    neighbor = subprocess.Popen([sys.executable, '-B', '-c', 'import sys; sys.stdin.buffer.read()'],
                                stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL)
    accepted = None
    try:
        try:
            accepted = meter._memory_module.OwnedKernelGroup(parent, meter.cgroup_v2_envelope())
        except RuntimeError as error:
            require('refuses a shared or foreign-populated' in str(error),
                    'calibration.kernel_foreign_monitor', 'unexpected refusal: ' + str(error))
        else:
            accepted.close()
            raise CalibrationFailure('calibration.kernel_foreign_monitor: '
                                     'accepted a monitor leaf containing a real neighbor')
    finally:
        if accepted is not None:
            accepted.close()
        try:
            neighbor.communicate(timeout=5)
        finally:
            if neighbor.poll() is None:
                neighbor.kill()
                neighbor.communicate(timeout=5)
    require(neighbor.returncode == 0
            and {path.name for path in parent.iterdir() if path.is_dir()} == children_before
            and (parent / 'cgroup.subtree_control').read_text() == controls_before,
            'calibration.kernel_foreign_release', 'refusal changed delegation or failed to release neighbor')
    return {'real_preexisting_neighbor_refused': True, 'neighbor_reaped': True,
            'delegation_unchanged': True}


def kernel_contract(case):
    kernel = case['measured'].get('kernel_resource_accounting')
    require(kernel is not None and kernel['target_entry_observed'],
            'calibration.kernel_entry', 'target never entered the fresh owned cgroup')
    require(kernel.get('group_removed') and not kernel['populated'],
            'calibration.kernel_release', 'owned cgroup was not emptied and removed')
    require(kernel['cpu_usage_seconds'] + 0.005 >= case['fixture']['self_cpu_ns'] / 1e9,
            'calibration.kernel_cpu', 'kernel cgroup omitted real fixture CPU')
    require(kernel['memory_peak_bytes'] >= 32 * MIB,
            'calibration.kernel_peak', 'kernel peak lost the touched 32 MiB private mapping')
    require(kernel['memory_events'].get('oom_kill', 0) == 0,
            'calibration.kernel_oom', 'bounded fixture was killed by OOM')
    return kernel


def paired_effect(pairs):
    """Exact randomized paired test plus a distribution-free median interval."""
    ratios = [pair['metered']['wall_ms'] / pair['reference']['wall_ms'] for pair in pairs]
    changes = [math.log(ratio) for ratio in ratios]
    observed = abs(sum(changes))
    extreme = sum(abs(sum(sign * change for sign, change in zip(signs, changes))) + 1e-12 >= observed
                  for signs in itertools.product((-1, 1), repeat=len(changes)))
    ordered = sorted(ratios)
    # Exact sign/order-statistic interval for the median under independent,
    # identically distributed continuous pairs. No small-sample p95 fiction.
    lower_index = 0
    coverage = 1 - 2 / (2 ** len(ratios))
    for k in range(1, (len(ratios) + 1) // 2 + 1):
        candidate = 1 - 2 * sum(math.comb(len(ratios), j) for j in range(k)) / (2 ** len(ratios))
        if candidate >= 0.95:
            lower_index, coverage = k - 1, candidate
    return {'independent_pairs': len(pairs),
            'median_metered_over_reference_ratio': statistics.median(ratios),
            'median_ratio_interval': [ordered[lower_index], ordered[-1 - lower_index]],
            'interval_coverage_under_stated_assumptions': coverage,
            'exact_randomization_p_two_sided_zero_effect': extreme / (2 ** len(changes)),
            'pair_changes_percent': [(ratio - 1) * 100 for ratio in ratios],
            'interpretation': 'a non-significant difference does not establish negligible overhead',
            'assumptions': 'fixed fixture; independently randomized within-pair order; '
                           'median interval additionally assumes independent identically distributed continuous pairs; '
                           'does not certify arbitrary workloads or other hosts'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--pairs', type=int, default=6)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--contracts-only', action='store_true')
    parser.add_argument('--kernel-cgroup-parent', type=Path,
                        help='contracts only: explicit delegated parent of this sole monitor leaf')
    parser.add_argument('--fixture', choices=('cost', 'steps', 'burst'), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not sys.platform.startswith('linux'):
        parser.error('Linux /proc calibration required; unsupported is not a successful calibration')
    if args.fixture:
        fixture(args.fixture)
        return 0
    if not args.output or not args.output.is_absolute() or not 6 <= args.pairs <= 12:
        parser.error('an absolute new --output and 6..12 independent pairs are required')
    if args.kernel_cgroup_parent and (not args.contracts_only
                                      or not args.kernel_cgroup_parent.is_absolute()):
        parser.error('--kernel-cgroup-parent requires --contracts-only and an absolute delegated directory')
    runs = args.output.with_name(args.output.name + '.runs')
    if args.output.exists() or runs.exists():
        parser.error('output or raw-run directory exists; preserve prior evidence')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    runs.mkdir()
    seed = args.seed if args.seed is not None else secrets.randbits(64)
    generator = random.Random(seed)
    schedule = [['metered', 'reference'] if generator.getrandbits(1) else ['reference', 'metered']
                for _ in range(args.pairs)]
    protocol = {'schema': 'greppy.web-meter-calibration-protocol.v1', 'seed': seed,
                'schedule': schedule, 'contracts_only': args.contracts_only,
                'operations': OPERATIONS, 'hold_seconds': HOLD_SECONDS,
                'meter_sha256': digest(METER), 'sampler_sha256': digest(ROOT / 'bench/process_memory.py'),
                'calibrator_sha256': digest(Path(__file__).resolve())}
    # Publish the immutable schedule before observing any result. No retries or
    # result-dependent stopping; an inconclusive result is retained as such.
    (runs / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    deadline = time.monotonic() + 90
    report = {'schema': 'greppy.web-meter-calibration.v1', 'protocol': protocol,
              'scope': 'one controlled Linux process; actual production meter; no browser/CLI/LLM',
              'exclusions': ['unsampled transient peaks', 'kernel/GPU/unmapped cache',
                             'arbitrary workload or host generalization'],
              'unsampled_peak_coverage_certified': False}
    try:
        meter = load_meter()
        report['host_before'] = meter.host_envelope()
        expected_checksum = work()
        if args.kernel_cgroup_parent:
            report['delegation_contract'] = kernel_delegation_contract(meter, args.kernel_cgroup_parent)
        held = execute_case(meter, runs / 'held-steps', 'steps', True, expected_checksum, deadline,
                            kernel_parent=args.kernel_cgroup_parent)
        report['memory_contract'] = memory_contract(held)
        report['cpu_contract'] = {'reported_fixture_cpu_seconds': held['fixture']['self_cpu_ns'] / 1e9,
                                  'meter_cpu_seconds': held['measured']['user_cpu_seconds']
                                                       + held['measured']['system_cpu_seconds']}
        if args.kernel_cgroup_parent:
            report['held_kernel_contract'] = kernel_contract(held)
        if not args.contracts_only or args.kernel_cgroup_parent:
            burst = execute_case(meter, runs / 'short-burst', 'burst', True, expected_checksum, deadline,
                                 kernel_parent=args.kernel_cgroup_parent)
            baseline = phase_median(burst, 'baseline', 'private_bytes')
            largest = max((sample.get('private_bytes') or 0 for sample in burst['samples']), default=0)
            report['short_burst'] = {'known_touched_private_bytes': 32 * MIB,
                                     'fixture_window': burst['fixture']['burst'],
                                     'observed_delta_bytes': largest - baseline,
                                     'detected_at_declared_tolerance': largest - baseline >= 30 * MIB,
                                     'interpretation': 'detection of this burst does not certify all transient peaks'}
            if args.kernel_cgroup_parent:
                report['burst_kernel_contract'] = kernel_contract(burst)
        if not args.contracts_only:
            pairs = []
            for index, order in enumerate(schedule):
                pair = {'index': index, 'order': order}
                pairs.append(pair)
                report['observer_pairs'] = pairs
                for variant in order:
                    pair[variant] = execute_case(meter, runs / f'pair-{index}-{variant}', 'cost',
                                                 variant == 'metered', expected_checksum, deadline)
            report['observer_effect'] = paired_effect(pairs)
        report['host_after'] = meter.host_envelope()
        report['source_provenance'] = meter.source_identity()
        require(digest(METER) == protocol['meter_sha256']
                and digest(ROOT / 'bench/process_memory.py') == protocol['sampler_sha256']
                and digest(Path(__file__).resolve()) == protocol['calibrator_sha256'],
                'calibration.source_identity', 'measurement sources changed during calibration')
        report['status'] = 'declared_contracts_verified'
        print('WEB_METER_CALIBRATION_OK ' + json.dumps({
            'memory_contract': report['memory_contract'], 'cpu_contract': report['cpu_contract'],
            'observer_effect': report.get('observer_effect'), 'short_burst': report.get('short_burst'),
            'burst_kernel_contract': report.get('burst_kernel_contract')}))
        return_code = 0
    except Exception as error:
        report['status'] = 'failed'
        report['error'] = type(error).__name__ + ': ' + str(error)
        print('WEB_METER_CALIBRATION_FAILED ' + report['error'], file=sys.stderr)
        return_code = 1
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    return return_code


if __name__ == '__main__':
    raise SystemExit(main())
