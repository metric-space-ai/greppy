"""Compare complete paired workflow costs without granting model release eligibility."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

from contracts import canonical, strict_json
from study_corpus import index_trial

METRICS = ('provider_input_tokens', 'provider_output_tokens', 'tool_result_json_bytes',
           'tool_calls', 'end_to_end_seconds')


def compare_runs(runs, *, baseline, candidate, comparison_kind):
    if not baseline or not candidate or baseline == candidate:
        raise ValueError('two distinct arms are required')
    if comparison_kind not in ('development_tool_comparison', 'heads_on_off'):
        raise ValueError('explicit comparison kind required')
    groups = {}; identities = set()
    for run in runs:
        if run['arm'] not in (baseline, candidate) or not run.get('pair_id') or not run.get('run_id'):
            raise ValueError('invalid paired run identity')
        if run['run_id'] in identities:
            raise ValueError('duplicate run identity')
        identities.add(run['run_id'])
        pair = groups.setdefault(run['pair_id'], {})
        if run['arm'] in pair:
            raise ValueError('duplicate arm in pair')
        if type(run.get('success')) is not bool:
            raise ValueError('explicit independent outcome required')
        for metric in METRICS:
            value = run['metrics'].get(metric)
            if value is not None and (type(value) not in (int, float) or not math.isfinite(value) or value < 0):
                raise ValueError('invalid workflow cost')
        pair[run['arm']] = run
    if not groups:
        raise ValueError('no paired workflows')
    pairs = []
    for key, group in sorted(groups.items()):
        if set(group) != {baseline, candidate}:
            raise ValueError('incomplete pair; failed and missing runs must not be dropped')
        a, b = group[baseline], group[candidate]
        if (not a.get('agent_configuration') or a['agent_configuration'] != b.get('agent_configuration')):
            raise ValueError('paired agent configurations differ or are missing')
        if comparison_kind == 'heads_on_off':
            release = a.get('release_sha256')
            if (not isinstance(release, str) or len(release) != 64
                    or any(c not in '0123456789abcdef' for c in release)):
                raise ValueError('heads comparison requires the same release checksum')
            if (not a.get('release_sha256') or a['release_sha256'] != b.get('release_sha256')
                    or a.get('heads_enabled') is not False or b.get('heads_enabled') is not True):
                raise ValueError('heads comparison requires the same release with explicit off/on states')
            if a.get('backend') not in ('cpu', 'metal', 'cuda') or a['backend'] != b.get('backend'):
                raise ValueError('heads comparison requires the same explicit backend')
        costs = {}
        for metric in METRICS:
            av, bv = a['metrics'].get(metric), b['metrics'].get(metric)
            difference = bv - av if av is not None and bv is not None else None
            costs[metric] = {'baseline': av, 'candidate': bv, 'difference': difference,
                             'change_percent': 100 * difference / av if difference is not None and av > 0 else None}
        def decreased(metric):
            value = costs[metric]['difference']
            return value is not None and value < 0
        def increased(metric):
            value = costs[metric]['difference']
            return value is not None and value > 0
        smaller = decreased('tool_result_json_bytes')
        pairs.append({'pair_id': key, 'baseline_run': a['run_id'], 'candidate_run': b['run_id'],
                      'baseline_success': a['success'], 'candidate_success': b['success'],
                      'additional_task_failure': a['success'] and not b['success'], 'costs': costs,
                      'smaller_results_more_calls': smaller and increased('tool_calls'),
                      'smaller_results_more_provider_input': smaller and increased('provider_input_tokens'),
                      'smaller_results_more_provider_output': smaller and increased('provider_output_tokens')})
    aggregate = {}
    for metric in METRICS:
        differences = [p['costs'][metric]['difference'] for p in pairs if p['costs'][metric]['difference'] is not None]
        percentages = [p['costs'][metric]['change_percent'] for p in pairs if p['costs'][metric]['change_percent'] is not None]
        aggregate[metric] = {'pairs_with_values': len(differences), 'pairs_with_percentages': len(percentages),
                             'missing_pairs': len(pairs) - len(differences),
                             'median_paired_difference': statistics.median(differences) if differences else None,
                             'median_paired_change_percent': statistics.median(percentages) if percentages else None}
    return {'schema': 'greppy.heads.workflow-cost-diagnostic.v1', 'comparison_kind': comparison_kind,
            'baseline_arm': baseline, 'candidate_arm': candidate, 'pair_count': len(pairs),
            'pairs': pairs, 'aggregate': aggregate, 'production_eligible': False,
            'scope': 'Descriptive paired costs only. No causal head effect, equivalence or production acceptance.'}


def metadata_costs(metadata):
    requests = [x for x in metadata['tool_calls'] if x['kind'] == 'request']
    responses = [x for x in metadata['tool_calls'] if x['kind'] == 'response']
    request_ids = {x['call_id'] for x in requests}
    response_ids = {x['call_id'] for x in responses}
    if request_ids != response_ids or len(request_ids) != len(requests) or len(response_ids) != len(responses):
        raise ValueError('incomplete or duplicate tool-call coverage')
    response_bytes = 0
    for response in responses:
        measured = len(json.dumps(response['result'], ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode())
        if measured != response['result_json_bytes']:
            raise ValueError('declared response byte count differs from actual JSON result')
        response_bytes += measured
    if (metadata.get('token_usage_status', {}).get('usage_missing') is not False
            or metadata.get('token_usage_conflicts')):
        raise ValueError('missing or conflicting provider usage must not be omitted')
    usage = metadata.get('cumulative_turn_token_usage', {}).get('value', {})
    tokens = {}
    for metric in ('input_tokens', 'output_tokens'):
        value = usage.get(metric)
        if type(value) is not int or value < 0:
            raise ValueError('complete provider token counters required')
        tokens[metric] = value
    return {'tool_calls': len(requests), 'tool_result_json_bytes': response_bytes, 'tokens': tokens}


def read(path):
    raw = Path(path).read_bytes()
    return strict_json(raw), hashlib.sha256(raw).hexdigest()


def import_basic_summary(path):
    path = Path(path).resolve()
    summary, summary_hash = read(path)
    if summary.get('schema') != 'greppy.basic-paired-summary.v1':
        raise ValueError('unsupported study summary')
    plan, plan_hash = read(path.parent / 'plan.json')
    if plan_hash != summary['plan_sha256']:
        raise ValueError('study summary plan checksum mismatch')
    runs = []; sources = []; trace_pairs = set(); completed_trials = set()
    for trace in summary['traces']:
        metadata_path = Path(trace['metadata']).resolve()
        trial_path = metadata_path.parent / 'trial.json'
        trial, trial_hash = read(trial_path)
        if (trial.get('schema') != 'greppy.web-study.basic.v1' or trial['plan_sha256'] != plan_hash
                or trial['arm'] != trace['arm'] or trial['position'] != trace['position']
                or Path(trial['artifacts']['metadata']).resolve() != metadata_path):
            raise ValueError('trace/trial identity mismatch')
        planned = [x for x in plan['trials'] if x['position'] == trial['position']]
        if len(planned) != 1 or any(planned[0][k] != trial[k] for k in ('case', 'repeat', 'arm', 'run_id')):
            raise ValueError('trial differs from preregistered pair')
        completed_trials.add((trial['position'], trial['arm'], trial['run_id']))
        # Verify export bytes and original tool pointers; do not parse reasoning.
        index_trial(trial_path, family=trial['case'])
        metadata, metadata_hash = read(metadata_path)
        measured = metadata_costs(metadata)
        if (measured['tool_calls'] != trace['request_count']
                or measured['tool_calls'] != trial['host_tool_envelopes']['request_count']):
            raise ValueError('tool-call coverage mismatch')
        if any(trial['tokens'].get(k) != v for k, v in measured['tokens'].items()):
            raise ValueError('trial token counters differ from complete provider totals')
        response_bytes = measured['tool_result_json_bytes']
        if response_bytes != trace['tool_response_json_bytes']:
            raise ValueError('summary response bytes differ from complete metadata counters')
        key = (trial['case'], trial['repeat'])
        trace_pairs.add(key)
        runs.append({'run_id': trial['run_id'], 'pair_id': canonical([plan_hash, *key]), 'arm': trial['arm'],
                     'agent_configuration': {k: trial['context'][k] for k in ('model', 'effort')},
                     'success': trial['oracle']['ok'],
                     'metrics': {'provider_input_tokens': trial['tokens'].get('input_tokens'),
                                 'provider_output_tokens': trial['tokens'].get('output_tokens'),
                                 'tool_result_json_bytes': response_bytes, 'tool_calls': measured['tool_calls'],
                                 'end_to_end_seconds': trial.get('end_to_end_verified_seconds')}})
        sources.append({'trial': str(trial_path), 'trial_sha256': trial_hash,
                        'metadata': str(metadata_path), 'metadata_sha256': metadata_hash})
    declared = {(x['case'], x['repeat']) for x in summary['pairs']}
    expected_trials = {(x['position'], x['arm'], x['run_id']) for x in plan['trials'] if x['arm'] in ('A', 'C')}
    if completed_trials != expected_trials or len(completed_trials) != len(summary['traces']):
        raise ValueError('planned runs are missing or duplicated; do not drop incomplete or failed workflows')
    if declared != trace_pairs or len(declared) != len(summary['pairs']):
        raise ValueError('summary pair coverage mismatch')
    report = compare_runs(runs, baseline='A', candidate='C', comparison_kind='development_tool_comparison')
    for metric, source_metric in [('provider_input_tokens', 'input_tokens'), ('provider_output_tokens', 'output_tokens')]:
        observed = report['aggregate'][metric]['median_paired_change_percent']
        expected = summary['median_paired_change_percent'][source_metric]
        if observed is None or not math.isclose(observed, expected, abs_tol=1e-9, rel_tol=1e-12):
            raise ValueError('paired change does not reproduce the source summary')
    report.update(source_summary={'path': str(path), 'sha256': summary_hash}, plan_sha256=plan_hash,
                  sources=sources, source_token_gate=summary['token_gate'],
                  source_acceptance=summary['acceptance'],
                  note='Provider totals include retries and recovery. JSON response bytes are a separate metric, not provider input tokens.')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study-summary', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    report = import_basic_summary(args.study_summary)
    with args.out.open('x') as stream:
        stream.write(canonical(report) + '\n')
    print(canonical({k: report[k] for k in ('pair_count', 'aggregate', 'production_eligible')}))
