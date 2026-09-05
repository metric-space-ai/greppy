"""Blind-teacher agreement is a review input, never sufficient training admission."""
import argparse
import json
from pathlib import Path
from contracts import RUBRIC_VERSION, canonical, digest, validate_annotations
from queue_store import QueueStore


def audit_queue(store, mechanical, *, rubric=RUBRIC_VERSION, selected_jobs=None):
    """mechanical maps example IDs to an independently produced evidence receipt.

    Receipt: example_sha256 (the exact sanitized teacher input), status pass/fail/
    unknown, evidence_artifact_sha256, complete_capture and privacy_admitted.
    The caller must independently verify the artifact; this function cannot certify
    an oracle simply from its self-reported receipt.
    """
    grouped = {}
    with store.connect() as db:
        jobs = list(db.execute("SELECT id,provider,model,rubric,payload,result,status FROM jobs"))
    if selected_jobs is not None:
        if not selected_jobs or len(selected_jobs) != len(set(selected_jobs)):
            raise ValueError('job selection must be nonempty and unique')
        if set(selected_jobs) - {job['id'] for job in jobs}:
            raise ValueError('job selection contains unknown jobs')
        jobs = [job for job in jobs if job['id'] in set(selected_jobs)]
    for job in jobs:
        if job['rubric'] != rubric:
            continue
        examples = json.loads(job['payload'])
        if job['status'] == 'done':
            result = json.loads(job['result'])
            validate_annotations(result, examples)
            rows = {}
            for row in result['annotations']:
                rows.setdefault(row['example_id'], []).append(row)
        else:
            rows = {}
        for example in examples:
            key = digest(example)
            item = grouped.setdefault(key, {'example': example, 'providers': {}})
            item['providers'].setdefault(job['provider'], []).append({
                'job_id': job['id'], 'model': job['model'], 'status': job['status'],
                'annotations': rows.get(example['id'], []),
            })
    report = []
    for key, item in sorted(grouped.items()):
        example = item['example']; reasons = []
        selected = {}
        for provider, model in [('minimax','MiniMax-M3'), ('grok','grok-4.6')]:
            jobs = item['providers'].get(provider, [])
            if len(jobs) != 1:
                reasons.append(provider + ('_missing' if not jobs else '_multiple_versions'))
            elif jobs[0]['model'] != model or jobs[0]['status'] != 'done':
                reasons.append(provider + '_not_complete_with_required_model')
            else:
                selected[provider] = {r['record_id']:r for r in jobs[0]['annotations']}
        conflicts = []
        if len(selected) == 2:
            for record in example['records']:
                rid = record['id']; a = selected['minimax'][rid]; b = selected['grok'][rid]
                different = [f for f in ('severity','relevance') if a[f] != b[f]]
                if set(a['evidence_ids']) != set(b['evidence_ids']):
                    different.append('evidence_ids')
                if a['ambiguous'] or b['ambiguous']:
                    different.append('ambiguous')
                if different:
                    conflicts.append({'record_id': rid, 'fields': different})
            if conflicts: reasons.append('teacher_conflict_or_uncertainty')
        receipt = mechanical.get(example['id'])
        if not receipt:
            reasons.append('independent_evidence_missing')
        else:
            evidence = receipt.get('evidence_artifact_sha256')
            if receipt.get('example_sha256') != key:
                reasons.append('independent_evidence_input_mismatch')
            if receipt.get('status') != 'pass':
                reasons.append('independent_evidence_not_passed')
            if not isinstance(evidence, str) or len(evidence) != 64 or any(c not in '0123456789abcdef' for c in evidence):
                reasons.append('independent_evidence_artifact_missing')
            if receipt.get('complete_capture') is not True:
                reasons.append('complete_capture_unverified')
            if receipt.get('privacy_admitted') is not True:
                reasons.append('privacy_unverified')
        report.append({'example_id': example['id'], 'example_sha256': key,
                       'source_id': example['source_id'], 'domain': example['domain'],
                       'split': example['split'], 'status': 'review_complete' if not reasons else 'held',
                       'reasons': reasons, 'conflicts': conflicts,
                       'teacher_jobs': {p:[j['job_id'] for j in js] for p,js in item['providers'].items()}})
    return {'schema': 'greppy.heads.admission-review.v1', 'rubric': rubric,
            'note': 'Review receipts require artifact verification; this report is not production acceptance.',
            'counts': {status: sum(r['status']==status for r in report) for status in ('review_complete','held')},
            'examples': report}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--db',type=Path,required=True)
    parser.add_argument('--mechanical',type=Path)
    parser.add_argument('--job-selection',type=Path,help='Explicit JSON array of immutable teacher job IDs; never choose a version silently')
    parser.add_argument('--out',type=Path,required=True)
    args = parser.parse_args()
    receipts = json.loads(args.mechanical.read_text()) if args.mechanical else {}
    selection = json.loads(args.job_selection.read_text()) if args.job_selection else None
    report = audit_queue(QueueStore(args.db), receipts, selected_jobs=selection)
    with args.out.open('x') as f:
        f.write(canonical(report)+'\n')
    print(canonical(report['counts']))


if __name__ == '__main__': main()
