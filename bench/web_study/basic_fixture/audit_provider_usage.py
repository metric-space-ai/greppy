"""Reconcile per-response provider usage with recorded totals; public metadata only."""
import argparse
import json
from pathlib import Path
import subprocess

METRICS = ('input_tokens', 'output_tokens', 'cached_input_tokens')


def read_json(path):
    raw = subprocess.check_output(['greppy', 'rg', '--no-filename', '.*', str(path)], text=True)
    return json.loads(raw)


def audit(series):
    plan = read_json(series / 'plan.json')
    rows = []
    for spec in plan['trials']:
        folder = series / 'trials' / f"{spec['position']:02d}-{spec['case']}-{spec['repeat']}-{spec['arm']}"
        trial = read_json(folder / 'trial.json')
        meta = read_json(Path(trial['artifacts']['metadata']))
        usage = meta['token_usage_records']
        identifiers = [entry['response_id'] for entry in usage]
        assert usage and len(identifiers) == len(set(identifiers))
        assert not meta['token_usage_conflicts']
        assert not meta['token_usage_records_without_response_id']
        sums = {metric: sum(entry['usage'][metric] for entry in usage) for metric in METRICS}
        cumulative = meta['cumulative_turn_token_usage']['value']
        for metric in METRICS:
            assert sums[metric] == trial['tokens'][metric] == cumulative[metric], (spec['position'], metric)
        assert all(0 <= entry['usage']['cached_input_tokens'] <= entry['usage']['input_tokens'] for entry in usage)
        assert trial['context']['model'] == plan['model'] and trial['context']['effort'] == plan['effort']
        requests = [entry for entry in meta['tool_calls'] if entry['kind'] == 'request']
        responses = [entry for entry in meta['tool_calls'] if entry['kind'] == 'response']
        errors = [{"call_id":entry['call_id'], "line":entry['source_line'],
                   "markers":[marker for marker in ('STALE_REF', 'TIMEOUT', 'NO_SESSION', 'unexpected argument')
                              if marker in json.dumps(entry['result'])]}
                  for entry in responses]
        rows.append(dict(position=spec['position'], arm=spec['arm'], repeat=spec['repeat'],
                         response_count=len(usage), tokens=sums, totals_reconciled=True,
                         oracle_passed=trial['oracle']['ok'], metadata=trial['artifacts']['metadata'],
                         workflow_request_ids=[entry['call_id'] for entry in requests if 'web do --native' in entry.get('arguments', '')],
                         error_marker_calls=[entry for entry in errors if entry['markers']],
                         marker_scope='Public text markers are evidence locators, not automatic product-bug classifications.'))
    assert len(rows) == len(plan['trials'])
    return dict(schema='greppy.provider-usage-audit.v1', all_reconciled=True,
                private_reasoning_inspected=False, rows=rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('series', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.series)
    with args.output.open('x') as output:
        json.dump(result, output, indent=2)
    print(json.dumps(dict(all_reconciled=True, trials=len(result['rows']), output=str(args.output))))
