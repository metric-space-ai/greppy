"""Prepare an isolated CLI context without prescribing a browser solution.

The short alias forwards argv unchanged. `web open` may create its own session;
no browser action or hidden verification runs here. Record this as a harness
condition, never as a product performance improvement.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
from candidate import capture


def prepare(scratch: Path, cli: Path, alias_dir: Path, trial_id: str, runtime_id: str,
            *, runtime: Path, view='default', chain_view='default'):
    if not re.fullmatch(r'[a-z0-9-]{1,40}', trial_id):
        raise ValueError('trial_id must be 1–40 lowercase letters, digits or hyphens')
    if not re.fullmatch(r'[a-z0-9-]{1,60}', runtime_id):
        raise ValueError('runtime_id must be 1–60 lowercase letters, digits or hyphens')
    if not Path('/Volumes/tmp').is_mount() or not scratch.resolve().is_relative_to(Path('/Volumes/tmp')):
        raise ValueError('scratch must be on mounted /Volumes/tmp')
    cli = cli.resolve(strict=True)
    if not cli.is_file() or not os.access(cli, os.X_OK):
        raise ValueError('cli must be an executable file')
    if view not in ('default', 'compact') or chain_view not in ('default', 'compact'):
        raise ValueError('view and chain_view must be default or compact')
    candidate = capture(cli, runtime)
    runtime_executable = candidate['runtime']['path']
    alias_dir = alias_dir.resolve(strict=True)
    trial = scratch.resolve() / trial_id
    alias = alias_dir / ('gw-' + trial_id)
    if trial.exists() or alias.exists() or alias.is_symlink():
        raise FileExistsError('refusing to reuse an existing participant context or alias')
    trial.mkdir(parents=True)
    workspace = trial / 'workspace'
    workspace.mkdir()
    temporary = scratch.resolve() / 'tmp'
    temporary.mkdir(exist_ok=True)
    runtime = scratch.resolve() / 'runtime'
    runtime.mkdir(exist_ok=True)
    q = shlex.quote
    source = '\n'.join([
        '#!/bin/sh',
        '# Study transport only; argv and exit status are unchanged.',
        'unset GREPPY_WEB_SESSION GREPPY_WEB_TAB GREPPY_WEB_AGENT GREPPY_WEB_VIEW GREPPY_WEB_CHAIN_VIEW GREPPY_WEB_RUNTIME_DIST',
        'export TMPDIR=' + q(str(temporary)),
        'export GREPPY_RUN_ID=' + q('study-' + runtime_id),
        'export GREPPY_WEB_RUNTIME_DIR=' + q(str(runtime)),
        'export GREPPY_WEB_RUNTIME=' + q(runtime_executable),
        *(['export GREPPY_WEB_VIEW=compact'] if view == 'compact' else []),
        *(['export GREPPY_WEB_CHAIN_VIEW=compact'] if chain_view == 'compact' else []),
        'cd ' + q(str(workspace)) + ' || exit 125',
        'exec ' + q(str(cli)) + ' "$@"',
        '',
    ])
    wrapper = trial / 'gw'
    wrapper.write_text(source)
    wrapper.chmod(0o700)
    alias.symlink_to(wrapper)
    record = {
        'schema': 'greppy.web-study.context.v1',
        'condition': 'short_alias_isolated_cwd_native_open',
        'trial_id': trial_id, 'alias': str(alias), 'command': alias.name,
        'workspace': str(workspace), 'wrapper': str(wrapper), 'cli': str(cli),
        'candidate': candidate,
        'rendering': {'view': view, 'chain_view': chain_view},
        'wrapper_sha256': hashlib.sha256(source.encode()).hexdigest(),
        'runtime_id': 'study-' + runtime_id,
        'browser_actions_during_setup': 0,
        'session_setup': 'participant invokes native web open or explicit session create',
        'cost_attribution': 'harness correction; no product efficiency claim',
    }
    (trial / 'context.json').write_text(json.dumps(record, indent=2) + '\n')
    return record


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scratch', required=True, type=Path)
    p.add_argument('--cli', required=True, type=Path)
    p.add_argument('--runtime', required=True, type=Path)
    p.add_argument('--view', choices=('default', 'compact'), default='default')
    p.add_argument('--chain-view', choices=('default', 'compact'), default='default')
    p.add_argument('--alias-dir', required=True, type=Path)
    p.add_argument('--trial-id', required=True)
    p.add_argument('--runtime-id', required=True)
    a = p.parse_args()
    print(json.dumps(prepare(a.scratch, a.cli, a.alias_dir, a.trial_id, a.runtime_id,
                            runtime=a.runtime, view=a.view, chain_view=a.chain_view)))


if __name__ == '__main__':
    main()
