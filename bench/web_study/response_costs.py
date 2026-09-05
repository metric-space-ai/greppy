"""Compare provider-reported response costs using exported telemetry only.

No reasoning text is read. Non-reasoning output includes tool-call syntax and
visible prose; these data cannot assign its tokens to individual strings.
"""
import argparse
import json
from pathlib import Path
import statistics
from summarize_series import token_totals

METRICS = ('model_responses', 'input_tokens', 'output_tokens',
           'reasoning_output_tokens', 'non_reasoning_output_tokens')


def costs(metadata):
    totals, limits = token_totals(metadata)
    entries = metadata['token_usage_records']
    result = {'model_responses': len(entries), **totals,
              'reasoning_output_tokens': None, 'non_reasoning_output_tokens': None}
    reasoning = [e.get('usage', {}).get('reasoning_output_tokens') if e.get('usage') else None for e in entries]
    outputs = [e.get('usage', {}).get('output_tokens') if e.get('usage') else None for e in entries]
    if (not limits and entries and all(isinstance(r, int) and not isinstance(r, bool)
                                      and isinstance(o, int) and 0 <= r <= o
                                      for r, o in zip(reasoning, outputs))
            and sum(outputs) == totals.get('output_tokens')):
        result['reasoning_output_tokens'] = sum(reasoning)
        result['non_reasoning_output_tokens'] = sum(outputs) - sum(reasoning)
    else:
        limits = [*limits, 'Reasoning/non-reasoning output decomposition unavailable or inconsistent']
    return result, limits


def analyze(series, cases):
    plan = json.loads((series / 'plan.json').read_text())
    records = []
    for path in sorted((series / 'trials').glob('*/trial.json')):
        trial = json.loads(path.read_text())
        if trial['case'] not in cases:
            continue
        metadata = json.loads(Path(trial['artifacts']['metadata']).read_text())
        measured, limits = costs(metadata)
        records.append({'position': trial['position'], 'case': trial['case'], 'repeat': trial['repeat'],
                        'arm': trial['arm'], 'oracle_ok': trial['oracle']['ok'],
                        'metadata': trial['artifacts']['metadata'], 'costs': measured, 'limits': limits})
    blocks = {}
    for case in cases:
        block = [r for r in records if r['case'] == case]
        medians = {}
        for arm in ('A', 'C'):
            arm_rows = [r for r in block if r['arm'] == arm]
            medians[arm] = {'runs': len(arm_rows)}
            for metric in METRICS:
                values = [r['costs'][metric] for r in arm_rows]
                medians[arm][metric] = statistics.median(values) if values and all(v is not None for v in values) else None
        paired = {metric: [] for metric in METRICS}
        for repeat in sorted({r['repeat'] for r in block}):
            pair = {r['arm']: r for r in block if r['repeat'] == repeat}
            if set(pair) != {'A', 'C'}:
                continue
            for metric in METRICS:
                a, c = (pair[arm]['costs'][metric] for arm in ('A', 'C'))
                paired[metric].append((c/a-1)*100 if a is not None and a > 0 and c is not None else None)
        blocks[case] = {'medians': medians, 'median_paired_change_percent': {
            m: statistics.median(v) if v and all(x is not None for x in v) else None for m, v in paired.items()}}
    return {'schema': 'greppy.web-study.response-costs.v1', 'series': str(series), 'blocks': blocks,
            'planned_selected': sum(t['case'] in cases for t in plan['trials']),
            'recorded_selected': len(records), 'records': records,
            'limits': ['Provider token counts only; no token estimates from bytes.',
                       'Component medians do not add to a total median.',
                       'No private reasoning text or inference about mental trust.',
                       'Non-reasoning output includes calls and prose, not exclusively commands.',
                       'Descriptive attribution, not a causal ablation or acceptance result.']}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('series', type=Path)
    p.add_argument('--cases', nargs='+', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    result = analyze(a.series, a.cases)
    with a.output.open('x') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps({'output': str(a.output), 'blocks': result['blocks'],
                      'planned': result['planned_selected'], 'recorded': result['recorded_selected']}))
