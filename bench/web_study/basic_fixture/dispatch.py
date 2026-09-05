"""Prospective, synthetic task provenance; no inference from private traces."""
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path

TASKS = {
    'text': 'Replace the existing note with Ready for review and save it.',
    'checkbox': 'Enable the checkbox and set the now-enabled quantity to 3.',
    'address': 'Choose Germany, Berlin and postcode 10115, then confirm visible validation.',
    'dialog': 'Open Complete basic task and use the Save button inside that dialog.',
    'table': 'Filter to EU and at least 3 available, sort unit price low to high, and reserve 3 of the cheapest matching item in its confirmation dialog. Reload and confirm that the reservation persists.',
}


def task_goal(trial):
    text = TASKS[trial['case']]
    return {'schema': 'greppy.web-study.task-goal.v1', 'case': trial['case'],
            'text': text, 'sha256': hashlib.sha256(text.encode()).hexdigest(),
            'scope': 'synthetic business objective, identical across arms; no action relevance labels'}


def freeze_dispatch(series, position, task_name, message):
    """Call BEFORE spawn; return the exact message to pass to spawn_agent.

    This artifact records intent, not delivery. Save the actual tool receipt
    separately; neither a completed trial nor temporal proximity proves that
    this message was sent unchanged.
    """
    series = Path(series)
    plan_raw = (series / 'plan.json').read_bytes()
    plan = json.loads(plan_raw)
    trial = next(t for t in plan['trials'] if t['position'] == position)
    goal = trial.get('task_goal')
    if goal != task_goal(trial):
        raise ValueError('requires a prospectively prepared task goal; no historical backfill')
    if goal['text'] not in message or trial['url'] not in message:
        raise ValueError('dispatch must include the exact business goal and trial URL')
    if not task_name.strip():
        raise ValueError('task_name is required')
    record = {'schema': 'greppy.web-study.prepared-dispatch.v1',
              'state': 'prepared_not_sent', 'created_at': dt.datetime.now(dt.timezone.utc).isoformat(),
              'position': position, 'run_id': trial['run_id'], 'arm': trial['arm'],
              'task_goal': goal, 'task_name': task_name,
              'model': plan['model'], 'reasoning_effort': plan['effort'], 'fork_turns': 'none',
              'message': message, 'message_sha256': hashlib.sha256(message.encode()).hexdigest(),
              'plan_sha256': hashlib.sha256(plan_raw).hexdigest(), 'delivery_evidence': None}
    folder = series / 'prepared-dispatches'
    folder.mkdir(exist_ok=True)
    with (folder / f'{position:02d}.json').open('x', encoding='utf-8') as output:
        json.dump(record, output, ensure_ascii=False, indent=2)
        output.write('\n')
    return record


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('series', type=Path)
    p.add_argument('position', type=int)
    p.add_argument('--task-name', required=True)
    p.add_argument('--message-file', type=Path, required=True,
                   help='Full synthetic participant prompt; do not include credentials or private traces')
    a = p.parse_args()
    print(json.dumps(freeze_dispatch(a.series, a.position, a.task_name,
                                    a.message_file.read_text(encoding='utf-8')), ensure_ascii=False))
