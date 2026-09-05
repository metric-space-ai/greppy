import hashlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from dispatch import TASKS, freeze_dispatch, task_goal


def plan(tmp_path, case='table'):
    trials = [{'position': n, 'case': case, 'arm': arm, 'run_id': str(n),
               'url': f'http://127.0.0.1:1234/?run_id={n}'}
              for n, arm in enumerate(('A', 'C'), 1)]
    for trial in trials:
        trial['task_goal'] = task_goal(trial)
    (tmp_path / 'plan.json').write_text(json.dumps({
        'model': 'gpt-5.6-luna', 'effort': 'medium', 'trials': trials}))
    return trials


@pytest.mark.parametrize('case', sorted(TASKS))
def test_business_goal_identical_across_arms_and_has_no_trial_solution(tmp_path, case):
    a, c = plan(tmp_path, case)
    assert a['task_goal'] == c['task_goal']
    assert a['url'] not in a['task_goal']['text']
    assert c['url'] not in c['task_goal']['text']


def test_preserves_exact_dispatch_without_claiming_delivery_or_overwriting(tmp_path):
    trial = plan(tmp_path)[0]
    message = f"{trial['task_goal']['text']}\n\nOpen {trial['url']}\nUse browser tools."
    record = freeze_dispatch(tmp_path, 1, 'table_a1', message)
    saved = json.loads((tmp_path / 'prepared-dispatches/01.json').read_text())
    assert saved == record
    assert saved['message'] == message
    assert saved['message_sha256'] == hashlib.sha256(message.encode()).hexdigest()
    assert saved['state'] == 'prepared_not_sent'
    assert saved['delivery_evidence'] is None
    with pytest.raises(FileExistsError):
        freeze_dispatch(tmp_path, 1, 'table_a1_retry', message)
    assert json.loads((tmp_path / 'prepared-dispatches/01.json').read_text()) == saved


@pytest.mark.parametrize('missing', ['goal', 'url', 'historical_goal'])
def test_refuses_incomplete_binding_or_historical_backfill(tmp_path, missing):
    trial = plan(tmp_path)[0]
    message = f"{trial['task_goal']['text']} {trial['url']}"
    if missing == 'historical_goal':
        source = json.loads((tmp_path / 'plan.json').read_text())
        del source['trials'][0]['task_goal']
        (tmp_path / 'plan.json').write_text(json.dumps(source))
    elif missing == 'goal':
        message = trial['url']
    else:
        message = trial['task_goal']['text']
    with pytest.raises(ValueError):
        freeze_dispatch(tmp_path, 1, 'table_a1', message)
    assert not (tmp_path / 'prepared-dispatches').exists()
