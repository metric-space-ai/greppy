import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from candidate import capture, verify
from prepare_context import prepare
from summarize_trials import summarize


def executables(tmp_path):
    cli = tmp_path / 'fake cli'
    cli.write_text('''#!/usr/bin/python3
import json, os, sys
keys = ['GREPPY_WEB_RUNTIME', 'GREPPY_WEB_SESSION', 'GREPPY_WEB_TAB', 'GREPPY_WEB_AGENT', 'GREPPY_WEB_VIEW', 'GREPPY_WEB_CHAIN_VIEW']
print(json.dumps({'argv':sys.argv[1:], 'env':{k:os.environ.get(k) for k in keys}}))
sys.exit(17)
''')
    runtime = tmp_path / 'fake runtime'
    runtime.write_text('#!/bin/sh\nexit 0\n')
    cli.chmod(0o700)
    runtime.chmod(0o700)
    return cli, runtime


@pytest.mark.parametrize('view,chain', [('default', 'default'), ('compact', 'default'), ('default', 'compact')])
def test_wrapper_sets_candidate_and_options_without_changing_argv(tmp_path, view, chain):
    cli, runtime = executables(tmp_path)
    aliases = tmp_path / 'aliases'
    aliases.mkdir()
    context = prepare(tmp_path / 'scratch', cli, aliases, 'test', 'test-runtime',
                      runtime=runtime, view=view, chain_view=chain)
    dirty = {**os.environ, **{k: 'inherited-wrong-value' for k in (
        'GREPPY_WEB_RUNTIME', 'GREPPY_WEB_SESSION', 'GREPPY_WEB_TAB', 'GREPPY_WEB_AGENT',
        'GREPPY_WEB_VIEW', 'GREPPY_WEB_CHAIN_VIEW')}}
    argv = ['web', 'open', 'http://example.invalid/?a=1&b=2', 'literal $(not-a-command)']
    result = subprocess.run([context['alias'], *argv], env=dirty, capture_output=True, text=True)
    assert result.returncode == 17
    output = json.loads(result.stdout)
    assert output['argv'] == argv
    assert output['env'] == {'GREPPY_WEB_RUNTIME': str(runtime.resolve()),
        'GREPPY_WEB_SESSION': None, 'GREPPY_WEB_TAB': None, 'GREPPY_WEB_AGENT': None,
        'GREPPY_WEB_VIEW': 'compact' if view == 'compact' else None,
        'GREPPY_WEB_CHAIN_VIEW': 'compact' if chain == 'compact' else None}
    assert verify(context['candidate'])['ok']


@pytest.mark.parametrize('changed', ['cli', 'runtime'])
def test_detects_replaced_executable_even_with_same_size(tmp_path, changed):
    cli, runtime = executables(tmp_path)
    candidate = capture(cli, runtime)
    path = Path(candidate[changed]['path'])
    content = path.read_bytes()
    path.write_bytes(content[:-1] + b'X')
    result = verify(candidate)
    assert not result['ok']
    assert result['errors'] == [changed + ': executable identity changed']


def test_unusable_runtime_fails_before_creating_context(tmp_path):
    cli, _ = executables(tmp_path)
    with pytest.raises(FileNotFoundError):
        prepare(tmp_path / 'scratch', cli, tmp_path, 'test', 'runtime', runtime=tmp_path / 'missing')
    assert not (tmp_path / 'scratch').exists()


@pytest.mark.parametrize('integrity', [True, False, None])
def test_token_win_requires_candidate_integrity_when_preregistered(tmp_path, integrity):
    trials = [{'position': n, 'arm': arm, 'case': 'dialog', 'repeat': 1, 'run_id': arm}
              for n, arm in enumerate(('A', 'C'), 1)]
    plan = {'trials': trials, 'model': 'gpt-5.6-luna', 'effort': 'medium',
            'candidate_integrity_required': True}
    raw = json.dumps(plan).encode()
    (tmp_path / 'plan.json').write_bytes(raw)
    metadata = tmp_path / 'metadata.json'
    metadata.write_text('{"tool_calls": []}')
    for trial in trials:
        out = tmp_path / 'trials' / trial['arm']
        out.mkdir(parents=True)
        tokens = 100 if trial['arm'] == 'A' else 50
        row = {**trial, 'context': {'model': plan['model'], 'effort': plan['effort']},
               'plan_sha256': hashlib.sha256(raw).hexdigest(), 'oracle': {'ok': True},
               'tokens': {'input_tokens': tokens, 'output_tokens': tokens},
               'host_tool_envelopes': {'request_count': 1},
               'artifacts': {'metadata': str(metadata)}}
        if integrity is not None:
            row['candidate_integrity'] = {'ok': integrity}
        (out / 'trial.json').write_text(json.dumps(row))
    result = summarize(tmp_path)
    assert result['candidate_integrity'] is (integrity is True)
    assert (result['token_gate'] == 'passes this development block only') is (integrity is True)
    assert result['medians']['C']['runs'] == 1  # Invalid provenance never drops the run.
