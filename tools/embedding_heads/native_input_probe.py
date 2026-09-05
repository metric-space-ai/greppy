"""Exercise shared native inputs on controlled compiler logs; no training labels."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import socket
import subprocess
import time

from contracts import canonical, strict_json


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def request(source_id, text, spans, *, head='log_classifier', task=None, max_tokens=128):
    raw = text.encode()
    candidates = []
    for i, span in enumerate(spans):
        candidates.append({'id': f'{source_id}-target-{i}', 'head': head,
                           'target': {'start': span[0], 'end': span[1]}, 'context': [],
                           'task': task, 'observation_id': None, 'goal_version': None,
                           'last_action': None})
    return {'source': {'id': source_id, 'sha256': sha(raw), 'text': text},
            'candidates': candidates,
            'limits': {'max_tokens': max_tokens, 'max_target_bytes': 2048,
                       'max_context_bytes': 1024, 'max_parts': 4096}}


def verify_rows(requests, rows, *, vectors):
    expected = {(r['source']['id'], c['id']): (r, c) for r in requests for c in r['candidates']}
    groups = {}; ids = set()
    for row in rows:
        if row['schema'] != 'greppy.heads.native-feature.v1' or row['production_eligible'] is not False:
            raise AssertionError('incorrect native probe envelope')
        item = row['input']; key = (item['source_id'], item['candidate_id'])
        if key not in expected or item['id'] in ids:
            raise AssertionError('invented or duplicate input identity')
        ids.add(item['id']); groups.setdefault(key, []).append(item)
        req, candidate = expected[key]; source = req['source']; raw = source['text'].encode()
        span = item['target']; piece = raw[span['start']:span['end']]
        if (item['source_sha256'] != source['sha256'] or item['target_sha256'] != sha(piece)
                or item['original_target'] != candidate['target']
                or item['input_sha256'] != sha(item['prompt'].encode())
                or not 0 < item['token_count'] <= req['limits']['max_tokens']):
            raise AssertionError('source, target, input hash or token boundary mismatch')
        prefix = 'task: classification | query: '
        body = strict_json(item['prompt'][len(prefix):])
        if not item['prompt'].startswith(prefix) or body['target'] != piece.decode() or body['task'] != candidate['task']:
            raise AssertionError('prepared prompt changed target or task')
        if body['last_action'] != candidate['last_action'] or body['head'] != candidate['head']:
            raise AssertionError('prepared prompt changed action or head')
        if vectors:
            vector = row['vector']
            if len(vector) != 768 or any(not math.isfinite(x) for x in vector):
                raise AssertionError('invalid native vector')
            if abs(sum(x*x for x in vector) - 1) > 1e-3:
                raise AssertionError('native representation is not unit-normalized')
        elif row['vector'] is not None or row['backend'] is not None:
            raise AssertionError('prepare-only unexpectedly performed model inference')
    if set(groups) != set(expected):
        raise AssertionError('missing candidate coverage')
    for key, parts in groups.items():
        _, c = expected[key]; end = c['target']['start']
        for part in parts:
            if part['target']['start'] != end:
                raise AssertionError('gap, overlap or reordered target pieces')
            end = part['target']['end']
        if end != c['target']['end']:
            raise AssertionError('target tail lost')
    return {'candidates': len(expected), 'parts': len(rows), 'full_target_coverage': True}


def invoke(binary, tokenizer, model, mode, requests, output):
    command = [str(binary), str(tokenizer), mode] + ([] if mode == 'prepare' else [str(model)])
    started = time.monotonic()
    completed = subprocess.run(command, input=''.join(canonical(r)+'\n' for r in requests),
                               capture_output=True, text=True, timeout=600, check=False)
    output.with_suffix('.stderr').write_text(completed.stderr)
    output.with_suffix('.status.json').write_text(canonical({'exit_code': completed.returncode})+'\n')
    if completed.returncode:
        raise RuntimeError(f'native probe failed with exit {completed.returncode}: {completed.stderr[-2000:]}')
    with output.open('x') as stream:
        stream.write(completed.stdout)
    rows = [strict_json(line) for line in completed.stdout.splitlines()]
    checks = verify_rows(requests, rows, vectors=mode != 'prepare')
    checks.update(mode=mode, cold_process_seconds=time.monotonic()-started,
                  artifact_sha256=file_sha(output))
    return rows, checks


def run(args):
    if socket.gethostname() != 'gpu3-a4500':
        raise ValueError('this probe is reserved for GPU3; no training is performed')
    args.out.mkdir(parents=True, exist_ok=False)
    requests = []; captures = []
    fixtures = [('warning', 'int main(void) { int unused = 7; return 0; }\n', 0),
                ('error', 'int main(void) { return missing_symbol; }\n', 1)]
    for name, source, expected_exit in fixtures:
        path = args.out / (name+'.c'); path.write_text(source)
        command = ['gcc', '-Wall', '-Wextra', '-fsyntax-only', path.name]
        result = subprocess.run(command, cwd=args.out, capture_output=True, check=False)
        if result.returncode != expected_exit or not result.stderr:
            raise AssertionError('controlled compiler case did not produce the expected outcome')
        text = result.stderr.decode('utf-8', errors='strict')
        (args.out/(name+'.stderr')).write_bytes(result.stderr)
        captures.append({'name': name, 'argv': command, 'exit_code': result.returncode,
                         'source_sha256': sha(source.encode()), 'stdout_sha256': sha(result.stdout),
                         'stderr_sha256': sha(result.stderr), 'controlled_fixture': True,
                         'complete_capture': True, 'annotation_admitted': False})
        requests.append(request(name, text, [(0, len(result.stderr))]))
    long_text = 'progress unchanged\n' * 100_000 + 'error: unique tail cause\n'
    offset = len(long_text.encode()) - len(b'error: unique tail cause\n')
    requests.append(request('long-tail', long_text, [(offset, len(long_text.encode()))]))
    # Same source span, different goals: labels may legitimately differ.
    log = requests[1]['source']['text']
    ranked = request('ranked-error', log, [(0, len(log.encode()))],
                     head='log_ranker', task='Identify the compiler failure cause.')
    other = dict(ranked['candidates'][0], id='ranked-error-other-goal',
                 task='Find whether unused-variable warnings occurred.')
    ranked['candidates'].append(other)
    requests.append(ranked)
    # Synthetic typed record is explicitly diagnostic, not a real Web episode.
    record = canonical({'kind': 'control', 'ref': '@7', 'role': 'button',
                        'name': 'Submit', 'disabled': True})
    web = request('web-record-fixture', record, [(0, len(record.encode()))],
                  head='web_ranker', task='Submit the completed form.')
    web['candidates'][0].update(observation_id='obs-fixture-1', goal_version=1,
                               last_action={'kind': 'fill', 'status': 'dispatched'})
    web['candidates'].append(dict(web['candidates'][0], id='web-goal-version-2',
                                  goal_version=2))
    requests.append(web)
    # A deliberately oversized target tests exact UTF-8 splitting and tail coverage.
    unicode_text = 'diagnostic ä🦀 "quoted"\r\n' * 80
    oversized = request('oversized', unicode_text, [(0, len(unicode_text.encode()))])
    prep_requests = requests + [oversized]
    input_file = args.out / 'preparation-inputs.jsonl'
    input_file.write_text(''.join(canonical(r)+'\n' for r in prep_requests))
    prepared, prep_checks = invoke(args.binary, args.tokenizer, args.model, 'prepare', prep_requests, args.out/'prepared.jsonl')
    native, native_checks = invoke(args.binary, args.tokenizer, args.model, args.backend, requests, args.out/'native.jsonl')
    repeated, repeat_checks = invoke(args.binary, args.tokenizer, args.model, args.backend, requests, args.out/'repeated.jsonl')
    prep_subset = [x for x in prepared if x['input']['source_id'] != 'oversized']
    if [x['input'] for x in prep_subset] != [x['input'] for x in native] or [x['input'] for x in native] != [x['input'] for x in repeated]:
        raise AssertionError('preparation and inference did not use identical input contracts')
    drift = max(abs(a-b) for x,y in zip(native,repeated) for a,b in zip(x['vector'],y['vector']))
    if drift > 5e-5:
        raise AssertionError('native replay exceeds declared 5e-5 tolerance')
    if any(x['backend'] != args.backend for x in native):
        raise AssertionError('requested backend silently changed')
    web_rows = [x['input'] for x in native if x['input']['source_id'] == 'web-record-fixture']
    if (len(web_rows) != 2 or web_rows[0]['input_sha256'] != web_rows[1]['input_sha256']
            or web_rows[0]['conditioning_sha256'] == web_rows[1]['conditioning_sha256']
            or any('checked' in x['prompt'] for x in web_rows)):
        raise AssertionError('Web version invalidation or unknown-field preservation failed')
    log_rows = [x['input'] for x in native if x['input']['source_id'] == 'ranked-error']
    by_candidate = {}
    for item in log_rows:
        by_candidate.setdefault(item['candidate_id'], []).append(item['input_sha256'])
    if len(by_candidate) != 2 or len({tuple(v) for v in by_candidate.values()}) != 2:
        raise AssertionError('log ranking goals did not change prepared representations')
    report = {'schema': 'greppy.heads.native-input-probe.v1', 'host': socket.gethostname(),
              'binary_sha256': file_sha(args.binary), 'tokenizer_sha256': file_sha(args.tokenizer),
              'model_sha256': file_sha(args.model), 'probe_sha256': file_sha(__file__),
              'input_sha256': file_sha(input_file), 'captures': captures,
              'preparation': prep_checks, 'native': native_checks, 'replay': repeat_checks,
              'max_replay_vector_difference': drift, 'production_eligible': False,
              'limits': ['Controlled compiler fixtures and a synthetic 100001-line source.',
                         'The Web record is synthetic; no browser workflow is measured.',
                         'Only the explicit tail target is embedded in the long source; this is not a whole-output latency test.',
                         'Cold process timing includes startup and hashing, not the warm head latency gate.',
                         'No teacher labels, model training, calibration or task acceptance.']}
    with (args.out/'report.json').open('x') as stream:
        stream.write(canonical(report)+'\n')
    print(canonical(report))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--tokenizer', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--backend', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--out', type=Path, required=True)
    run(parser.parse_args())
