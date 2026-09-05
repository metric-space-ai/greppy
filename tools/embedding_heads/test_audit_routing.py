import json
from pathlib import Path
import tempfile
import unittest

from audit_sampling import plan_audit
from contracts import canonical, digest, sanitized_example
from queue_store import QueueStore
from teacher_queue import enqueue_audit_file
from test_teacher_pipeline import example, labels
from admission import audit_queue


class AuditRoutingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.store = QueueStore(self.root / 'queue.sqlite')
        self.examples = []
        self.sources = []
        for i in range(10):
            item = example(); item.update(id=f'e{i}', source_id=f's{i}', group_key=f'g{i}')
            item['records'][0]['text'] += ' password=hunter22'
            self.examples.append(item)
            self.sources.append({'id': item['source_id'], 'sha256': digest(['source', i]),
                                 'domain': 'log', 'family': 'build', 'length_bin': 'long',
                                 'split': 'train', 'stage': 'broad',
                                 'examples': {item['id']: digest(sanitized_example(item))}})
        self.path = self.root / 'examples.jsonl'
        self.path.write_text(''.join(canonical(x) + '\n' for x in self.examples))

    def route(self, plan):
        return enqueue_audit_file(self.store, self.path, ['minimax', 'grok'], 48000, plan, self.sources)

    def test_routes_all_m3_and_only_selected_grok_with_exact_sanitized_hash(self):
        plan = plan_audit(self.sources, seed=17)
        result = self.route(plan)
        self.assertEqual(result['jobs_by_provider'], {'minimax': 10, 'grok': 1})
        with self.store.connect() as db:
            for job in db.execute('SELECT payload FROM jobs'):
                self.assertNotIn('hunter22', job['payload'])
                safe = json.loads(job['payload'])[0]
                self.assertEqual(digest(safe), self.sources[int(safe['id'][1:])]['examples'][safe['id']])
                self.assertGreater(safe['redaction_count'], 0)

    def test_problem_expansion_reuses_completed_jobs(self):
        plan = plan_audit(self.sources, seed=17)
        first = self.route(plan)
        job = self.store.claim('minimax')
        eid = job['examples'][0]['id']
        answer = labels()
        for row in answer['annotations']:
            row['example_id'] = eid
        self.store.complete(job, answer)
        target = next(x['example_id'] for x in plan['assignments'] if not x['grok_required'])
        extended = plan_audit(self.sources, seed=17, problem_examples=[target])
        second = self.route(extended)
        self.assertTrue(set(first['ids']).issubset(second['ids']))
        with self.store.connect() as db:
            self.assertEqual(db.execute('SELECT COUNT(*) FROM jobs').fetchone()[0], 12)
            self.assertEqual(db.execute('SELECT status FROM jobs WHERE id=?', (job['id'],)).fetchone()[0], 'done')

    def test_changed_input_cannot_use_old_audit_assignment(self):
        plan = plan_audit(self.sources, seed=17)
        self.examples[0]['task'] = 'A different task'
        self.path.write_text(canonical(self.examples[0]) + '\n')
        with self.assertRaisesRegex(ValueError, 'differs from frozen'):
            self.route(plan)
        with self.store.connect() as db:
            self.assertEqual(db.execute('SELECT COUNT(*) FROM jobs').fetchone()[0], 0)


    def complete_provider(self, provider, ambiguous_id=None):
        while (job := self.store.claim(provider)) is not None:
            eid = job['examples'][0]['id']
            answer = labels()
            for row in answer['annotations']:
                row['example_id'] = eid
                row['ambiguous'] = eid == ambiguous_id
            self.store.complete(job, answer)

    def receipts(self):
        return {x['id']: {'example_sha256': digest(sanitized_example(x)), 'status': 'pass',
                         'evidence_artifact_sha256': digest(['synthetic oracle', x['id']]),
                         'complete_capture': True, 'privacy_admitted': True} for x in self.examples}

    def test_broad_review_requires_sampled_judge_and_independent_evidence(self):
        plan = plan_audit(self.sources, seed=17)
        self.route(plan)
        self.complete_provider('minimax')
        report = audit_queue(self.store, self.receipts(), audit_plan=plan, source_roster=self.sources)
        self.assertEqual(report['counts'], {'review_complete': 9, 'held': 1})
        self.assertFalse(report['all_planned_examples_review_complete'])
        self.complete_provider('grok')
        report = audit_queue(self.store, self.receipts(), audit_plan=plan, source_roster=self.sources)
        self.assertEqual(report['counts'], {'review_complete': 10, 'held': 0})
        self.assertTrue(report['all_planned_examples_review_complete'])
        no_evidence = audit_queue(self.store, {}, audit_plan=plan, source_roster=self.sources)
        self.assertEqual(no_evidence['counts']['held'], 10)

    def test_unsampled_uncertainty_never_passes(self):
        plan = plan_audit(self.sources, seed=17)
        target = next(x['example_id'] for x in plan['assignments'] if not x['grok_required'])
        self.route(plan)
        self.complete_provider('minimax', ambiguous_id=target)
        self.complete_provider('grok')
        report = audit_queue(self.store, self.receipts(), audit_plan=plan, source_roster=self.sources)
        held = next(x for x in report['examples'] if x['example_id'] == target)
        self.assertEqual(held['status'], 'held')
        self.assertIn('audit_plan_missing_uncertainty_escalation', held['reasons'])

    def test_incomplete_queue_reports_missing_population(self):
        plan = plan_audit(self.sources, seed=17)
        self.path.write_text(canonical(self.examples[0]) + '\n')
        self.route(plan)
        self.complete_provider('minimax'); self.complete_provider('grok')
        report = audit_queue(self.store, self.receipts(), audit_plan=plan, source_roster=self.sources)
        self.assertEqual(len(report['missing_planned_examples']), 9)
        self.assertFalse(report['all_planned_examples_review_complete'])


if __name__ == '__main__':
    unittest.main()
