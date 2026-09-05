"""Reproduce token-cost vectors from pinned host telemetry, without token prices."""
from __future__ import annotations
import argparse
import hashlib
import json
import statistics
from pathlib import Path


def token_totals(metadata):
    records = metadata.get('token_usage_records', [])
    reasons = []
    if not records:
        reasons.append('missing response-level token telemetry')
    if metadata.get('token_usage_conflicts'):
        reasons.append('conflicting response-level token telemetry')
    if metadata.get('token_usage_records_without_response_id'):
        reasons.append('token telemetry without response identity')
    ids = [r.get('response_id') for r in records]
    if len(set(ids)) != len(ids) or any(not isinstance(i, str) for i in ids):
        reasons.append('missing or duplicate response identities')
    totals = {}
    for key in ('input_tokens', 'output_tokens', 'cached_input_tokens'):
        values = [r.get('usage', {}).get(key) if isinstance(r.get('usage'), dict) else None for r in records]
        valid = bool(values) and all(type(v) is int and v >= 0 for v in values)
        totals[key] = sum(values) if valid and not reasons else None
    if totals['input_tokens'] is None or totals['output_tokens'] is None:
        reasons.append('incomplete input/output counts')
    cache_valid = all(isinstance(r.get('usage'), dict)
                      and type(r['usage'].get('cached_input_tokens')) is int
                      and type(r['usage'].get('input_tokens')) is int
                      and 0 <= r['usage']['cached_input_tokens'] <= r['usage']['input_tokens']
                      for r in records)
    if not cache_valid:
        totals['cached_input_tokens'] = None
    totals['uncached_input_tokens'] = (totals['input_tokens'] - totals['cached_input_tokens']
                                      if totals['input_tokens'] is not None and totals['cached_input_tokens'] is not None else None)
    return totals, reasons


def load_trial(directory):
    trial_path = directory / 'trial.json'
    trial_raw = trial_path.read_bytes()
    trial = json.loads(trial_raw)
    manifest_path = Path(trial['artifacts']['manifest'])
    manifest_raw = manifest_path.read_bytes()
    manifest = json.loads(manifest_raw)
    trace_path = Path(trial['artifacts']['trace'])
    trace_raw = trace_path.read_bytes()
    trace_sha = hashlib.sha256(trace_raw).hexdigest()
    if trace_sha != manifest['sha256'] or len(trace_raw) != manifest['byte_length']:
        raise ValueError(f'trace integrity failure: {directory.name}')
    metadata_path = Path(trial['artifacts']['metadata'])
    metadata_raw = metadata_path.read_bytes()
    metadata = json.loads(metadata_raw)
    if metadata['turn_id'] != trial['turn_id'] or manifest['turn_id'] != trial['turn_id']:
        raise ValueError(f'turn identity mismatch: {directory.name}')
    tokens, telemetry_limits = token_totals(metadata)
    token_differences = {k: {'recorded': (trial.get('tokens') or {}).get(k), 'recomputed': v}
                         for k, v in tokens.items() if k != 'uncached_input_tokens'
                         and (trial.get('tokens') or {}).get(k) != v}
    complete = metadata.get('completion_boundary', {}).get('task_complete_present') is True
    matched = not metadata.get('tool_response_status', {}).get('unmatched_request_call_ids', [])
    context = metadata.get('turn_context', {})
    return {
        'trial': directory.name,
        'task_success': trial.get('oracle', {}).get('ok') is True,
        'completed_turn': complete,
        'matched_tool_calls': matched,
        'model': context.get('model'), 'effort': context.get('effort'),
        'tokens': tokens, 'telemetry_limits': telemetry_limits,
        'recorded_token_differences': token_differences,
        'agent_turn_seconds': trial.get('agent_turn_wall_seconds'),
        'end_to_end_verified_seconds': trial.get('end_to_end_verified_seconds'),
        'host_tool_envelopes': metadata.get('tool_response_status', {}).get('request_count'),
        'result_json_bytes': sum(c.get('result_json_bytes', 0) for c in metadata.get('tool_calls', []) if c.get('kind') == 'response'),
        'provenance': {'trial_sha256': hashlib.sha256(trial_raw).hexdigest(),
                       'manifest_sha256': hashlib.sha256(manifest_raw).hexdigest(),
                       'metadata_sha256': hashlib.sha256(metadata_raw).hexdigest(),
                       'trace_sha256': trace_sha},
    }


def median_complete(values):
    return statistics.median(values) if values and all(type(v) in (float, int) for v in values) else None


def summarize_groups(groups):
    summary = {}
    for name, rows in groups.items():
        # Failures remain in the observed distribution and counts. There is no
        # interpretation of their shorter duration as verified task efficiency.
        summary[name] = {
            'n': len(rows), 'successes': sum(r['task_success'] for r in rows),
            'all_tasks_successful': bool(rows) and all(r['task_success'] for r in rows),
            'model_effort_pairs': sorted({(r['model'], r['effort']) for r in rows}, key=str),
            'median_tokens': {k: median_complete([r['tokens'][k] for r in rows])
                              for k in ('input_tokens', 'output_tokens', 'cached_input_tokens', 'uncached_input_tokens')},
            'median_agent_turn_seconds': median_complete([r['agent_turn_seconds'] for r in rows]),
            'median_end_to_end_verified_seconds': median_complete([r['end_to_end_verified_seconds'] for r in rows]),
            'median_host_tool_envelopes': median_complete([r['host_tool_envelopes'] for r in rows]),
            'median_result_json_bytes': median_complete([r['result_json_bytes'] for r in rows]),
            'trials': rows,
        }
    return {'schema': 'greppy.web-study.cost-vector.v1', 'groups': summary,
            'interpretation': ['Input includes cached input; do not add cache a second time.',
                               'Each token dimension is reported separately; no currency prices or byte-to-token conversion.',
                               'Reported marginal medians do not add up to a median monetary cost.',
                               'Tokens cover the recorded participant turn including recovery and final response.',
                               'No causal attribution to individual fixes; E was a later exploratory series.',
                               'A completed turn and a post-hoc oracle do not provide end-to-end verified latency.',
                               'Failed runs remain present; their duration is not a successful task efficiency result.'],
            'main_study_acceptance_evaluated': False}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root', type=Path)
    p.add_argument('--group', action='append', required=True, help='NAME=trial,trial,...; explicit frozen membership')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    groups = {}
    membership = set()
    for spec in args.group:
        name, members = spec.split('=', 1)
        if name in groups:
            raise ValueError('duplicate group')
        names = members.split(',')
        if any(Path(n).name != n or n in ('', '.', '..') for n in names):
            raise ValueError('trial names must be direct directory names')
        if len(set(names)) != len(names) or membership.intersection(names):
            raise ValueError('duplicate trial membership')
        membership.update(names)
        groups[name] = [load_trial(args.root / n) for n in names]
    result = summarize_groups(groups)
    with args.output.open('x', encoding='utf-8') as destination:
        json.dump(result, destination, indent=2)
        destination.write('\n')
    print(json.dumps({name: {k:v for k,v in value.items() if k != 'trials'} for name,value in result['groups'].items()}, indent=2))


if __name__ == '__main__':
    main()
