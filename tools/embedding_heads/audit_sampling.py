"""Freeze source-level blind audit sampling before teacher labels are available."""
import argparse
import math
from pathlib import Path

from contracts import canonical, digest, strict_json

SCHEMA = 'greppy.heads.audit-sampling.v1'
SOURCE_KEYS = {'id', 'sha256', 'domain', 'family', 'length_bin', 'split', 'stage', 'examples'}


def hash_value(value):
    return isinstance(value, str) and len(value) == 64 and all(x in '0123456789abcdef' for x in value)


def plan_audit(sources, *, seed, problem_examples=()):
    """Source examples map IDs to hashes of exact sanitized teacher inputs.

    Sampling never uses predicted classes, agreement or outcomes. Problem flags
    are added afterwards and cannot change membership of the random cohort.
    """
    if type(seed) is not int or not 0 <= seed < 2 ** 63:
        raise ValueError('a fixed nonnegative 63-bit sampling seed is required')
    sources = sorted(sources, key=lambda x: x['id'])
    if not sources or len({x['id'] for x in sources}) != len(sources):
        raise ValueError('source population must be nonempty and unique')
    examples = {}; strata = {}; source_hash_owners = {}
    for source in sources:
        if set(source) != SOURCE_KEYS:
            raise ValueError('audit roster must contain only the declared source fields')
        if (any(not isinstance(source[k], str) or not source[k] for k in ('id', 'family', 'length_bin'))
                or not hash_value(source['sha256']) or source['domain'] not in ('log', 'web')
                or source['split'] not in ('train', 'development', 'final')
                or source['stage'] not in ('pilot', 'broad', 'final')):
            raise ValueError('invalid source audit stratum')
        if (source['stage'] == 'final') != (source['split'] == 'final'):
            raise ValueError('final audit stage must match final split')
        content_key = (source['domain'], source['sha256'])
        if content_key in source_hash_owners:
            raise ValueError('duplicate source content must be deduplicated before sampling')
        source_hash_owners[content_key] = source['id']
        if not isinstance(source['examples'], dict) or not source['examples']:
            raise ValueError('complete source example roster is required')
        for eid, value in source['examples'].items():
            if not isinstance(eid, str) or not eid or not hash_value(value) or eid in examples:
                raise ValueError('invalid or duplicate example identity')
            examples[eid] = {'sha256': value, 'source_id': source['id']}
        if source['stage'] == 'broad':
            stratum = (source['domain'], source['family'], source['length_bin'], source['split'])
            strata.setdefault(stratum, []).append(source['id'])
    problem_examples = set(problem_examples)
    if problem_examples - set(examples):
        raise ValueError('problem selection references unknown examples')
    population_hash = digest(sources)
    random_sources = set(); random_strata = []
    for stratum, members in sorted(strata.items()):
        count = math.ceil(len(members) / 10)
        ranked = sorted(members, key=lambda sid: digest([SCHEMA, seed, list(stratum), sid]))
        selected = sorted(ranked[:count])
        random_sources.update(selected)
        random_strata.append({'stratum': dict(zip(('domain', 'family', 'length_bin', 'split'), stratum)),
                              'population_sources': len(members), 'selected_sources': selected,
                              'inclusion_probability': count / len(members)})
    assignments = []
    for source in sources:
        for eid, value in sorted(source['examples'].items()):
            reasons = []
            if source['stage'] in ('pilot', 'final'):
                reasons.append(source['stage'] + '_full_review')
            if source['id'] in random_sources:
                reasons.append('stratified_random_10_percent')
            if eid in problem_examples:
                reasons.append('targeted_conflict_or_uncertainty')
            assignments.append({'example_id': eid, 'example_sha256': value,
                                'source_id': source['id'], 'domain': source['domain'],
                                'split': source['split'], 'minimax_required': True,
                                'grok_required': bool(reasons), 'review_reasons': reasons,
                                'random_cohort': source['id'] in random_sources,
                                'targeted_cohort': eid in problem_examples})
    return {'schema': SCHEMA, 'population_sha256': population_hash, 'seed': seed,
            'sampling_unit': 'complete_source', 'fraction': 0.1, 'rounding': 'ceil_per_stratum',
            'random_selection_sha256': digest({'population': population_hash, 'seed': seed,
                                              'strata': random_strata}),
            'random_strata': random_strata, 'assignments': assignments,
            'note': 'Audit routing only; no annotation, admission or production acceptance is granted.'}


def verify_plan(plan, sources):
    if plan.get('schema') != SCHEMA:
        raise ValueError('unsupported audit sampling plan')
    targeted = [x['example_id'] for x in plan['assignments'] if x['targeted_cohort']]
    expected = plan_audit(sources, seed=plan['seed'], problem_examples=targeted)
    if canonical(expected) != canonical(plan):
        raise ValueError('audit plan differs from the frozen source population or sampling rule')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sources', type=Path, required=True, help='JSON array of source roster entries')
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--problems', type=Path, help='JSON array of conflict/uncertainty example IDs')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    sources = strict_json(args.sources.read_bytes())
    problems = strict_json(args.problems.read_bytes()) if args.problems else []
    report = plan_audit(sources, seed=args.seed, problem_examples=problems)
    with args.out.open('x') as stream:
        stream.write(canonical(report) + '\n')
    print(canonical({'sources': len(sources), 'examples': len(report['assignments']),
                     'grok_required': sum(x['grok_required'] for x in report['assignments']),
                     'random_selection_sha256': report['random_selection_sha256']}))


if __name__ == '__main__':
    main()
