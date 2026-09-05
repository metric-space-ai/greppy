import json
from pathlib import Path
import pytest
from summarize_series import load_trial, summarize_groups, token_totals
from export_codex_trace import export_turn


def telemetry(usage):
    return {'token_usage_records': [{'response_id': 'r1', 'usage': usage}]}


def test_cache_is_a_subset_and_missing_cache_is_not_zero():
    totals, issues = token_totals(telemetry({'input_tokens': 100, 'output_tokens': 9, 'cached_input_tokens': 80}))
    assert not issues
    assert totals == dict(input_tokens=100, output_tokens=9, cached_input_tokens=80, uncached_input_tokens=20)
    missing, issues = token_totals(telemetry({'input_tokens': 100, 'output_tokens': 9}))
    assert missing['input_tokens'] == 100
    assert missing['cached_input_tokens'] is None
    assert missing['uncached_input_tokens'] is None


def test_conflicting_or_invalid_telemetry_cannot_become_a_small_cost():
    m = telemetry({'input_tokens': 100, 'output_tokens': 9, 'cached_input_tokens': 80})
    m['token_usage_conflicts'] = [{'response_id': 'r1'}]
    totals, issues = token_totals(m)
    assert all(v is None for v in totals.values())
    assert issues
    totals, issues = token_totals(telemetry({'input_tokens': True, 'output_tokens': 9}))
    assert totals['input_tokens'] is None
    assert issues
    totals, _ = token_totals(telemetry({'input_tokens': 100, 'output_tokens': 9, 'cached_input_tokens': 101}))
    assert totals['uncached_input_tokens'] is None


def test_failure_is_retained_and_does_not_prove_efficiency():
    rows = [dict(task_success=passed, model='luna', effort='medium',
                 tokens=dict(input_tokens=100, output_tokens=10, cached_input_tokens=None, uncached_input_tokens=None),
                 agent_turn_seconds=seconds, end_to_end_verified_seconds=None,
                 host_tool_envelopes=2, result_json_bytes=90)
            for passed, seconds in [(True, 100), (False, 1)]]
    report = summarize_groups({'C': rows})
    assert report['groups']['C']['n'] == 2
    assert report['groups']['C']['successes'] == 1
    assert report['groups']['C']['all_tasks_successful'] is False
    assert report['groups']['C']['median_end_to_end_verified_seconds'] is None
    assert report['main_study_acceptance_evaluated'] is False


def test_changed_trace_is_rejected(tmp_path):
    source = tmp_path / 'source.jsonl'
    records = [dict(type='event_msg', payload=dict(type='task_started', turn_id='t1')),
               dict(type='turn_context', payload=dict(turn_id='t1', model='luna', effort='medium')),
               dict(type='event_msg', payload=dict(type='token_usage_record', response_id='r1', usage=dict(input_tokens=100, output_tokens=9, cached_input_tokens=80))),
               dict(type='event_msg', payload=dict(type='task_complete', turn_id='t1'))]
    source.write_text(''.join(json.dumps(r)+'\n' for r in records))
    directory = tmp_path / 'trial'
    artifacts = export_turn(source, directory, 't1')
    trial = dict(turn_id='t1', artifacts={k:str(v) for k,v in artifacts.items()},
                 tokens=None, oracle=dict(ok=True), agent_turn_wall_seconds=1)
    (directory / 'trial.json').write_text(json.dumps(trial))
    assert load_trial(directory)['tokens']['uncached_input_tokens'] == 20
    with Path(artifacts['trace']).open('a') as out:
        out.write('{}\n')
    with pytest.raises(ValueError, match='integrity'):
        load_trial(directory)
