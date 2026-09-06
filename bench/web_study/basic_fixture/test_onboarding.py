"""Both participants get the same business scope; no task solution is injected."""
import json
import pytest
from dispatch import TASKS, task_goal
from onboarding import (participant_message, prepare_messages,
                        SYNTHETIC_BROWSER_CONDITION, COORDINATED_BROWSER_CONDITION)


def trial(arm, case='table'):
    value = dict(position=1 if arm == 'A' else 2, run_id='abcdef123456', case=case, repeat=1,
                 arm=arm, url='http://127.0.0.1:9019/?run_id=abcdef123456',
                 cli_context={'command': 'gw-abcdef123456'})
    value['task_goal'] = task_goal(value)
    return value


@pytest.mark.parametrize('case', sorted(TASKS))
def test_same_goal_disposable_authorization_and_reporting_scope(case):
    messages = [participant_message(trial(arm, case), COORDINATED_BROWSER_CONDITION)
                for arm in ('A', 'C')]
    common = [message.split('Use only ')[0] for message in messages]
    assert common[0] == common[1]
    for message in messages:
        assert TASKS[case] in message
        assert 'does not place a real order' in message
        assert 'Do not message other tasks.' in message
        assert 'coordinator records traces and forwards bug reports' in message
        assert 'Do not read fixture source' in message
        assert 'Finish with one short factual sentence.' in message
        for solution in ('#inventory-region', '#reservation-dialog', 'Ember', 'ascending'):
            assert solution not in message


@pytest.mark.parametrize('arm', ['A', 'C'])
def test_prior_condition_is_unchanged_except_explicit_new_shared_paragraph(arm):
    before = participant_message(trial(arm), SYNTHETIC_BROWSER_CONDITION)
    after = participant_message(trial(arm), COORDINATED_BROWSER_CONDITION)
    start = after.index('Do not message other tasks.')
    end = after.index('\n\n', start) + 2
    assert after[:start] + after[end:] == before
    if arm == 'A':
        assert 'mcp__node_repl__js' in after and 'emit its complete documentation' in after
    else:
        assert 'text(await tools.exec_command' in after and 'write_stdin until terminal' in after


def test_exact_future_dispatches_are_frozen_without_claiming_delivery(tmp_path):
    source = dict(onboarding_condition=COORDINATED_BROWSER_CONDITION,
                  model='gpt-5.6-luna', effort='medium', trials=[trial('A'), trial('C')])
    (tmp_path / 'plan.json').write_text(json.dumps(source))
    records = prepare_messages(tmp_path, 'paired')
    assert len(records) == 2
    assert all(r['model'] == 'gpt-5.6-luna' and r['reasoning_effort'] == 'medium' for r in records)
    assert all(r['fork_turns'] == 'none' and r['delivery_evidence'] is None for r in records)
    for record, spec in zip(records, source['trials']):
        assert record['message'] == participant_message(spec, COORDINATED_BROWSER_CONDITION)
    with pytest.raises(FileExistsError):
        prepare_messages(tmp_path, 'different')


@pytest.mark.parametrize('arm', ['A', 'C'])
def test_native_wait_documentation_is_generic_and_preserves_previous_conditions(arm):
    from onboarding import NATIVE_WAIT_BROWSER_CONDITION
    before = participant_message(trial(arm), COORDINATED_BROWSER_CONDITION)
    after = participant_message(trial(arm), NATIVE_WAIT_BROWSER_CONDITION)
    if arm == 'A':
        assert after == before
    else:
        assert after.startswith(before + '\n\nAvailable Greppy capabilities:')
        assert 'web wait QUERY --native' in after
        assert 'Choose your own targets and conditions.' in after
        for solution in ('#inventory', '#reservation', 'Ember', 'ascending', '--absent'):
            assert solution not in after


@pytest.mark.parametrize('case', sorted(TASKS))
def test_workflow_condition_preserves_standard_arm_and_no_solution(case):
    from onboarding import WORKFLOW_BROWSER_CONDITION
    assert participant_message(trial('A', case), WORKFLOW_BROWSER_CONDITION) == participant_message(
        trial('A', case), COORDINATED_BROWSER_CONDITION)
    message = participant_message(trial('C', case), WORKFLOW_BROWSER_CONDITION)
    assert message.startswith(participant_message(trial('C', case), COORDINATED_BROWSER_CONDITION))
    for fact in ('web do --native', '--expect QUERY', 'retains earlier effects',
                 'not visibility', 'web do --json --native', 'full archived state'):
        assert fact in message
    for solution in ('#inventory', '#reservation', 'Ember', 'ascending', 'css=dialog[open]'):
        assert solution not in message
