"""Summarize paired Basic trials without replacing tokens with byte estimates."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

METRICS = ('input_tokens', 'output_tokens', 'cached_input_tokens', 'uncached_input_tokens')


def summarize(series):
    plan_bytes = (series / 'plan.json').read_bytes()
    plan = json.loads(plan_bytes)
    records = [json.loads(p.read_text()) for p in sorted((series / 'trials').glob('*/trial.json'))]
    expected = {t['position']: t for t in plan['trials']}
    actual = {t['position']: t for t in records}
    if len(actual) != len(records) or set(actual) != set(expected):
        raise ValueError('Require every preregistered trial exactly once, including failures')
    for r in records:
        if any(r[k] != expected[r['position']][k] for k in ('arm', 'case', 'repeat', 'run_id')):
            raise ValueError('Trial does not match preregistration')
        if r['context']['model'] != plan['model'] or r['context']['effort'] != plan['effort']:
            raise ValueError('Model or effort mismatch')
        if r['plan_sha256'] != hashlib.sha256(plan_bytes).hexdigest():
            raise ValueError('Plan changed after a trial was recorded')
    medians = {}
    for arm in ('A', 'C'):
        arm_records = [r for r in records if r['arm'] == arm]
        medians[arm] = {'runs': len(arm_records), 'passed': sum(r['oracle']['ok'] for r in arm_records)}
        for metric in METRICS:
            values = [r['tokens'].get(metric) for r in arm_records]
            medians[arm][metric] = statistics.median(values) if values and all(v is not None for v in values) else None
        medians[arm]['tool_calls'] = statistics.median(r['host_tool_envelopes']['request_count'] for r in arm_records)
    pairs = []
    for case, repeat in sorted({(r['case'], r['repeat']) for r in records}):
        pair = {r['arm']: r for r in records if (r['case'], r['repeat']) == (case, repeat)}
        if set(pair) != {'A', 'C'}:
            raise ValueError('Incomplete pair')
        p = {'case': case, 'repeat': repeat, 'both_passed': all(r['oracle']['ok'] for r in pair.values())}
        for metric in METRICS:
            a, c = (pair[arm]['tokens'].get(metric) for arm in ('A', 'C'))
            p[metric + '_change_percent'] = (c / a - 1) * 100 if a is not None and a > 0 and c is not None else None
        pairs.append(p)
    changes = {}
    for metric in METRICS:
        values = [p[metric + '_change_percent'] for p in pairs]
        changes[metric] = statistics.median(values) if all(v is not None for v in values) else None
    traces = []
    for r in records:
        metadata = json.loads(Path(r['artifacts']['metadata']).read_text())
        calls = metadata['tool_calls']
        requests = [t for t in calls if t['kind'] == 'request']
        responses = [t for t in calls if t['kind'] == 'response']
        traces.append({
            'position': r['position'], 'arm': r['arm'],
            'metadata': r['artifacts']['metadata'],
            'request_count': len(requests),
            'open_requests': sum(' web open ' in t.get('arguments', '') for t in requests),
            'observe_requests': sum(' web observe' in t.get('arguments', '') for t in requests),
            'click_requests': sum(' web click ' in t.get('arguments', '') for t in requests),
            'shell_nomatch_responses': sum('zsh:1: no matches found:' in json.dumps(t.get('result')) for t in responses),
            'tool_response_json_bytes': sum(t['result_json_bytes'] for t in responses),
        })
    token_win = all(changes[m] is not None and changes[m] < 0 for m in ('input_tokens', 'output_tokens'))
    integrity = all(r.get('candidate_integrity', {}).get('ok') is True
                    for r in records if r['arm'] == 'C') if plan.get('candidate_integrity_required') else None
    token_win = token_win and (integrity is True if plan.get('candidate_integrity_required') else True)
    return {
        'schema': 'greppy.basic-paired-summary.v1', 'plan_sha256': hashlib.sha256(plan_bytes).hexdigest(),
        'condition': plan.get('harness_condition'), 'medians': medians, 'pairs': pairs,
        'candidate_integrity': integrity,
        'median_paired_change_percent': changes, 'traces': traces,
        'token_gate': 'passes this development block only' if token_win and all(p['both_passed'] for p in pairs) else 'failed_or_unproven',
        'acceptance': 'not established; one development case, no arm B or controlled end-to-end latency',
        'time_acceptance': None, 'token_source': 'provider-reported usage, all completed trials retained',
    }


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('series', type=Path)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    result = summarize(a.series)
    with a.output.open('x') as f:
        json.dump(result, f, indent=2)
    print(json.dumps({k: result[k] for k in ('condition', 'medians', 'median_paired_change_percent', 'token_gate', 'acceptance')}))
