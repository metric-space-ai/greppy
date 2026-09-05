"""Validate prospective study goals without promoting dispatch intent to delivery."""
from pathlib import Path
import hashlib

from contracts import digest, strict_json


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def prospective_goal(dispatch_path, plan_path, *, run_id, position, arm, case):
    dispatch_path = Path(dispatch_path)
    plan_path = Path(plan_path)
    dispatch_raw = dispatch_path.read_bytes()
    plan_raw = plan_path.read_bytes()
    prepared = strict_json(dispatch_raw)
    plan = strict_json(plan_raw)
    if (prepared.get('schema') != 'greppy.web-study.prepared-dispatch.v1'
            or prepared.get('state') != 'prepared_not_sent'
            or prepared.get('delivery_evidence') is not None):
        raise ValueError('expected an undelivered prepared-dispatch artifact')
    if prepared.get('plan_sha256') != sha(plan_raw):
        raise ValueError('prepared dispatch plan checksum mismatch')
    if (type(position) is not int or position < 1 or prepared.get('position') != position
            or prepared.get('run_id') != run_id or prepared.get('arm') != arm):
        raise ValueError('prepared dispatch trial identity mismatch')
    trials = [x for x in plan['trials'] if x.get('position') == position]
    if len(trials) != 1:
        raise ValueError('prepared dispatch requires one exact planned trial')
    trial = trials[0]
    if (trial.get('run_id') != run_id or trial.get('arm') != arm or trial.get('case') != case):
        raise ValueError('planned trial identity mismatch')
    goal = prepared.get('task_goal')
    if (not isinstance(goal, dict) or goal.get('schema') != 'greppy.web-study.task-goal.v1'
            or goal.get('case') != case or trial.get('task_goal') != goal
            or not isinstance(goal.get('text'), str) or not goal['text'].strip()
            or goal.get('sha256') != sha(goal['text'].encode())):
        raise ValueError('prospective task goal is absent or inconsistent')
    if any(x.get('case') == case and x.get('task_goal') != goal for x in plan['trials']):
        raise ValueError('task goal must be identical across planned arms and repeats')
    message = prepared.get('message')
    if (not isinstance(message, str) or prepared.get('message_sha256') != sha(message.encode())
            or goal['text'] not in message or not isinstance(trial.get('url'), str)
            or not trial['url'] or trial['url'] not in message):
        raise ValueError('prepared message does not bind the exact goal and trial URL')
    if (prepared.get('model') != plan.get('model')
            or prepared.get('reasoning_effort') != plan.get('effort')
            or prepared.get('fork_turns') != 'none'
            or not isinstance(prepared.get('task_name'), str) or not prepared['task_name'].strip()):
        raise ValueError('prepared dispatch execution configuration mismatch')
    # Return hashes and pointers only. The full dispatch can include harness
    # instructions and is not an allowed teacher field. Preparation is not proof
    # of when or whether the actual agent received this message.
    return {'schema': 'greppy.heads.prospective-goal-binding.v1',
            'state': 'prepared_not_sent', 'delivery_verified': False,
            'run_id': run_id, 'position': position, 'arm': arm, 'case': case,
            'goal_sha256': goal['sha256'], 'goal_contract_sha256': digest(goal),
            'goal_pointer': '/task_goal/text', 'dispatch_path': str(dispatch_path.resolve()),
            'dispatch_sha256': sha(dispatch_raw), 'message_sha256': prepared['message_sha256'],
            'plan_path': str(plan_path.resolve()), 'plan_sha256': sha(plan_raw),
            'admission': 'held', 'hold_reason': 'actual_dispatch_delivery_evidence_missing'}
