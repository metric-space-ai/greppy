"""Audit manually identified public retry rounds against real provider counters.

This associates measured generations with visible actions. It does not estimate
counterfactual savings, split shared generations, or inspect private reasoning.
"""
import argparse
import hashlib
import json
from pathlib import Path

# IDs are evidence annotations, not commands to execute or inferred model motives.
REPEATED_SUBMISSION = {
    '05-table-3-A': [
        ('call_wgr7HDBickgdiTOaWWCqf2cq', 'tab.ax.click(9)'),
        ('call_MEbnkLmskoiG6tKz9hEq1nFs', 'tab.ax.pressKey')],
    '03-table-2-C': [
        ('call_8G8VsXgq4TbShCHeGaJEX9GT', 'web click @1002'),
        ('call_whTG1iS87dcsoYRZWxoI6lZQ', 'web click @1002'),
        ('call_9Vyabuk6By5OFRIJ4IleKaTK', 'write_stdin')],
    '06-table-3-C': [
        ('call_ct4Ruq2qysTromcuUddSUmgI', 'Confirm reservation'),
        ('call_3VTqssTWah1mKedOI0hZTb65', 'Confirm reservation'),
        ('call_ltrvfIhtzZxveCBE11se5aHD', 'write_stdin')],
    '07-table-4-C': [
        ('call_YIOAYydPqxrMlRPybt6dhdt0', 'Confirm reservation')],
    '10-table-5-C': [
        ('call_G7njYtlEJ5TxEeoZgDbwn9GK', 'web click @1002'),
        ('call_VC10cQnHcQoqzaFt3mgqdqVS', 'write_stdin')],
}
KEYS = ('input_tokens', 'output_tokens', 'cached_input_tokens')


def audit(series):
    trials = []
    sources = {}

    def load(path):
        raw = path.read_bytes()
        sources[str(path)] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)

    timelines = sorted((series / 'usage-timelines').glob('*.json'))
    assert len(timelines) == 10, 'Expected the complete frozen ten-trial S08 block'
    for path in timelines:
        timeline = load(path)
        assert timeline['complete'] and not timeline['problems'], path
        metadata_paths = list((series / 'trials' / path.stem).glob('*.metadata.json'))
        assert len(metadata_paths) == 1, path
        metadata = load(metadata_paths[0])
        public_calls = {c['call_id']: c for c in metadata['tool_calls']
                        if c.get('kind') == 'request'}
        rounds = timeline['responses']
        assert all(sum(r['tokens'][k] for r in rounds) == timeline['total'][k]
                   for k in KEYS), path
        annotations = REPEATED_SUBMISSION.get(path.stem, [])
        ids = {call_id for call_id, _ in annotations}
        for call_id, expected in annotations:
            call = public_calls[call_id]
            assert expected in call['arguments'], (path, call_id)
            assert sum(any(c['call_id'] == call_id for c in r['calls'])
                       for r in rounds) == 1, (path, call_id)
        selected = [r for r in rounds if any(c['call_id'] in ids for c in r['calls'])]
        # Full generations stay intact even if multiple public calls share them.
        measured = {k: sum(r['tokens'][k] for r in selected) for k in KEYS}
        trials.append(dict(trial=path.stem, arm=path.stem[-1],
                           provider_total=timeline['total'], total_rounds=len(rounds),
                           annotated_call_ids=sorted(ids), selected_rounds=selected,
                           measured_selected_generations=measured,
                           percent_of_trial={k: 100 * measured[k] / timeline['total'][k]
                                             if timeline['total'][k] else None for k in KEYS}))
    return dict(schema='greppy.feedback-round-costs.v1', trials=trials,
                source_sha256=sources,
                interpretation='Observed generations associated with repeated submission attempts and their polling; NOT causal savings or removable token estimates.',
                limits=['Manual public-call annotation, not an exhaustive failure classifier.',
                        'Standard A retries are retained; A4 non-submission failure remains in the series.',
                        'Ordinary outcome observations and reloads are not classified as avoidable.',
                        'Cached input remains input. No bytes-to-tokens conversion.',
                        'No model motive or private reasoning is inferred.'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--series', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.series)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2)
    for trial in report['trials']:
        print(json.dumps({k: v for k, v in trial.items() if k != 'selected_rounds'}))
