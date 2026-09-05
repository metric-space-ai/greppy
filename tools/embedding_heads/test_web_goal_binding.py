import copy
from pathlib import Path
import tempfile
import unittest

from contracts import canonical
from web_goal_binding import prospective_goal, sha


def artifacts(root):
    goal = {'schema': 'greppy.web-study.task-goal.v1', 'case': 'text',
            'text': 'SYNTHETIC_GOAL Save the note.', 'sha256': sha(b'SYNTHETIC_GOAL Save the note.'),
            'scope': 'synthetic objective'}
    trial = {'position': 1, 'case': 'text', 'run_id': 'run-a', 'arm': 'A',
             'url': 'http://fixture/?run_id=run-a', 'task_goal': goal}
    plan = {'model': 'test-model', 'effort': 'medium', 'trials': [trial]}
    plan_path = root / 'plan.json'
    plan_path.write_text(canonical(plan))
    message = goal['text'] + ' ' + trial['url'] + ' PRIVATE_HARNESS_SENTINEL'
    prepared = {'schema': 'greppy.web-study.prepared-dispatch.v1', 'state': 'prepared_not_sent',
                'position': 1, 'run_id': 'run-a', 'arm': 'A', 'task_goal': goal,
                'task_name': 'task-a', 'model': 'test-model', 'reasoning_effort': 'medium',
                'fork_turns': 'none', 'message': message, 'message_sha256': sha(message.encode()),
                'plan_sha256': sha(plan_path.read_bytes()), 'delivery_evidence': None}
    return plan, prepared


class GoalBindingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.plan_path = self.root / 'plan.json'
        self.dispatch_path = self.root / 'dispatch.json'
        self.plan, self.prepared = artifacts(self.root)

    def validate(self, **kwargs):
        self.dispatch_path.write_text(canonical(self.prepared))
        return prospective_goal(self.dispatch_path, self.plan_path,
                                run_id=kwargs.get('run_id', 'run-a'), position=1, arm='A', case='text')

    def test_prepared_is_held_and_does_not_export_prompt_or_goal_text(self):
        value = self.validate()
        self.assertEqual(value['state'], 'prepared_not_sent')
        self.assertFalse(value['delivery_verified'])
        self.assertEqual(value['admission'], 'held')
        self.assertNotIn('PRIVATE_HARNESS_SENTINEL', canonical(value))
        self.assertNotIn('SYNTHETIC_GOAL', canonical(value))

    def test_different_run_rejected(self):
        with self.assertRaisesRegex(ValueError, 'identity mismatch'):
            self.validate(run_id='run-b')

    def test_modified_prompt_and_plan_rejected(self):
        self.prepared['message'] += ' changed'
        with self.assertRaisesRegex(ValueError, 'message'):
            self.validate()
        self.prepared['message_sha256'] = sha(self.prepared['message'].encode())
        self.plan_path.write_text(canonical(self.plan) + '\n')
        with self.assertRaisesRegex(ValueError, 'checksum'):
            self.validate()

    def test_historical_plan_without_goal_cannot_be_backfilled(self):
        del self.plan['trials'][0]['task_goal']
        self.plan_path.write_text(canonical(self.plan))
        self.prepared['plan_sha256'] = sha(self.plan_path.read_bytes())
        with self.assertRaisesRegex(ValueError, 'goal is absent'):
            self.validate()

    def test_claimed_delivery_not_accepted_as_prepared_artifact(self):
        self.prepared['delivery_evidence'] = {'sent': True}
        with self.assertRaisesRegex(ValueError, 'undelivered'):
            self.validate()

    def test_exact_goal_must_be_in_message(self):
        self.prepared['message'] = 'http://fixture/?run_id=run-a'
        self.prepared['message_sha256'] = sha(self.prepared['message'].encode())
        with self.assertRaisesRegex(ValueError, 'exact goal'):
            self.validate()
    def test_different_goal_in_other_arm_rejected(self):
        other = copy.deepcopy(self.plan['trials'][0])
        other.update(position=2, arm='C')
        other['task_goal']['text'] = 'Different goal'
        other['task_goal']['sha256'] = sha(b'Different goal')
        self.plan['trials'].append(other)
        self.plan_path.write_text(canonical(self.plan))
        self.prepared['plan_sha256'] = sha(self.plan_path.read_bytes())
        with self.assertRaisesRegex(ValueError, 'identical across'):
            self.validate()



if __name__ == '__main__':
    unittest.main()
