"""Offline analysis of captured producer envelopes, not a product adapter.

Normalization is an explicit investigator intervention. It must not be presented
as existing CLI behavior or as an observed reduction of provider tokens.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess

PATHS = {
    'observe': ('result', 'actionables'),
    'find': ('result', 'value', 'nodes'),
    'extract': ('result', 'value', 'rows'),
}


def recover_rows(label, envelope):
    """Accept only complete, successful, known fixture producer shapes."""
    if envelope.get('schema') != 'greppy.web-runtime.v1':
        raise ValueError('unrecognized runtime schema')
    if envelope.get('status') != 'ok' or envelope.get('error'):
        raise ValueError('producer did not succeed; not a no-match')
    if label not in PATHS:
        raise ValueError('unknown producer')
    result = envelope.get('result', {})
    if result.get('refs_truncated') or result.get('truncated'):
        raise ValueError('incomplete result; no complete-selection claim')
    scope = result.get('observation_scope') or {}
    if any(value is True for key, value in scope.items() if key.endswith('_truncated')):
        raise ValueError('incomplete scope')
    node = envelope
    try:
        for key in PATHS[label]:
            node = node[key]
    except (KeyError, TypeError) as error:
        raise ValueError('producer shape differs from captured contract') from error
    if not isinstance(node, list) or any(not isinstance(row, dict) for row in node):
        raise ValueError('expected object records')
    if label == 'observe':
        count = result.get('ref_count')
    else:
        count = result.get('value', {}).get('count')
    if count is not None and count != len(node):
        raise ValueError('reported count differs from returned records')
    # Original envelope is separately preserved; nothing is inferred or joined.
    return node


def analyze(calls, cli):
    by_label = {call['label']: call for call in calls}
    producers = {}
    controls = []
    def match(label, rows, query):
        payload = ''.join(json.dumps(row, separators=(',', ':')) + '\n' for row in rows)
        argv = [str(cli), 'web', 'match', query]
        p = subprocess.run(argv, input=payload, text=True, capture_output=True, timeout=15)
        control = dict(label=label, argv=argv, stdin=payload, exit_code=p.returncode,
                       stdout=p.stdout, stderr=p.stderr)
        controls.append(control)
        if p.returncode:
            raise RuntimeError(f'{label} filter failed: {p.returncode}')
        return [json.loads(line) for line in p.stdout.splitlines() if line.strip()]
    for label, path in PATHS.items():
        captured = by_label[label]
        if captured.get('exit_code') != 0 or captured.get('timeout'):
            raise ValueError(f'{label}: producer capture failed')
        envelope = json.loads(captured['stdout'])
        rows = recover_rows(label, envelope)
        economy = match(label + '-recovered-text', rows, 'text~/Economy/')
        if len(economy) != 1 or economy[0].get('text') != 'Economy delivery':
            raise AssertionError(f'{label}: expected exactly Economy after explicit recovery')
        direct = by_label[label + '-direct-match']
        producers[label] = {
            'path': '.'.join(path), 'row_count': len(rows),
            'fields_common_to_all_rows': sorted(set.intersection(*(set(r) for r in rows))),
            'direct_match_exit': direct['exit_code'],
            'direct_match_bytes': len(direct['stdout'].encode()),
            'recovered_match_count': len(economy),
            'fixture_economy_row': economy[0],
            'has_ref_on_every_row': all('ref' in row for row in rows),
            'has_visibility_on_every_row': all('visible' in row for row in rows),
            'has_price_on_every_row': all('attr:data-price' in row for row in rows),
        }
        if label == 'find':
            visible = match('find-visible-deliveries', rows, 'text~/delivery/ visible=true')
            if len(visible) != 3 or any(r['text'] == 'Hidden delivery' for r in visible):
                raise AssertionError('visibility control did not preserve three visible options')
        if label == 'extract':
            cheap = match('extract-price-only', rows, 'attr:data-price<10')
            if {r['text'] for r in cheap} != {'Economy delivery', 'Hidden delivery'}:
                raise AssertionError('numeric-string control differs from captured contract')
            visible_cheap = match('extract-visible-price', rows, 'attr:data-price<10 visible=true')
            if visible_cheap:
                raise AssertionError('missing visibility unexpectedly matched true')
    return {
        'schema': 'greppy.composition-replay.v1',
        'source': 'recorded real browser envelopes plus explicit offline row recovery',
        'agent_token_acceptance': False, 'provider_tokens': None,
        'producers': producers, 'controls': controls,
        'conclusion': 'row recovery repairs regex composition; price/visibility/ref facts remain split',
    }

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--calls', required=True, type=Path)
    p.add_argument('--cli', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError('refusing to overwrite an existing analysis')
    result = analyze(json.loads(a.calls.read_text()), a.cli)
    a.output.write_text(json.dumps(result, indent=2) + '\n')
    print('PROBE recovered regex matches: observe=1 find=1 extract=1')
    print('PROBE visibility control: 3 visible; price-only: 2 including hidden; price+visible: 0 (field absent)')
