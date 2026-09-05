"""Stratified audit coverage; point estimates never imply a release gate.

TP/FP/FN/TN describe model predictions crossed with ORIGINAL reference labels.
Each audited_errors count instead uses the independent audited reference.
"""
import argparse
import json
import math

STRATA = ('TP', 'FP', 'FN', 'TN')


def summarize(data):
    strata = data['strata']
    if set(strata) != set(STRATA):
        raise ValueError('Provide all four original-label strata, including unjudged ones')
    errors = {}
    for name in STRATA:
        values = [strata[name][key] for key in ('population', 'judged', 'audited_errors')]
        if any(type(value) is not int for value in values):
            raise ValueError('Counts must be integers (not booleans)')
        population, judged, audited_errors = values
        if not 0 <= audited_errors <= judged <= population:
            raise ValueError('Require 0 <= audited_errors <= judged <= population')
        errors[name] = population * audited_errors / judged if judged else (0.0 if population == 0 else None)
    missing = [name for name in STRATA if errors[name] is None]
    predicted = strata['TP']['population'] + strata['FP']['population']
    tp = errors['TP'] + errors['FP'] if all(errors[s] is not None for s in ('TP', 'FP')) else None
    fn = errors['FN'] + errors['TN'] if all(errors[s] is not None for s in ('FN', 'TN')) else None
    result = {
        'schema': 'greppy.heads.audit-coverage.v1',
        'missing_strata': missing,
        'all_strata_covered': not missing,
        'fully_audited': all(strata[s]['judged'] == strata[s]['population'] for s in STRATA),
        'estimated_error_counts': errors,
        'error_precision_point_estimate': tp / predicted if tp is not None and predicted else None,
        'error_recall_point_estimate': tp / (tp + fn) if tp is not None and fn is not None and tp + fn else None,
        'release_gate': 'not_evaluated',
        'limitations': [
            'Point estimates require random sampling within each stratum.',
            'Sampling uncertainty and dependence within outputs are not estimated.',
            'Coverage does not establish label correctness or absence of leakage.',
        ],
    }
    if missing == ['TN'] and tp is not None:
        known_fn = errors['FN']
        first_failure = max(0, math.floor(tp / .90 - tp - known_fn + 1e-10) + 1)
        result['unaudited_tn_sensitivity'] = {
            'recall_assuming_zero_tn_errors': tp / (tp + known_fn) if tp + known_fn else None,
            'recall_assuming_all_tn_errors': tp / (tp + known_fn + strata['TN']['population']),
            'tn_errors_to_fall_below_90_percent': first_failure,
            'possible_within_tn_population': first_failure <= strata['TN']['population'],
            'note': 'Conditional on point estimates in other strata; not confidence bounds.',
        }
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input')
    args = parser.parse_args()
    with open(args.input, encoding='utf-8') as stream:
        print(json.dumps(summarize(json.load(stream)), indent=2, allow_nan=False))
