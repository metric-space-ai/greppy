"""Build and preserve a bounded debug CLI candidate with source and disk guards."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import tempfile
import time

SOURCE_FILES = ['Cargo.toml', 'Cargo.lock', 'crates/cli/src/web/common.rs',
                'crates/cli/src/web/expect.rs', 'crates/cli/src/web/runtimes.rs',
                'crates/cli/src/web/see.rs', 'crates/cli/src/web/view.rs', 'crates/cli/src/web/view_scope.rs',
                'crates/cli/src/web/chain.rs', 'crates/cli/src/web/chain_output.rs',
                'crates/cli/src/web/act.rs', 'crates/web-client/src/protocol.rs',
                'crates/web-client/src/lib.rs', 'crates/web-client/src/describe-node.js',
                'crates/cli/src/web/mod.rs', 'crates/cli/src/web/nav.rs',
                'crates/cli/src/web/workflow.rs', 'crates/web-client/src/workflow.rs',
                'crates/web-client/src/workflow-condition.js', 'crates/web-client/src/workflow-preflight.js',
                'bench/web_study/basic_fixture/build_cli_candidate.py']


def build(root, artifacts, timeout, test_web=False, test_client=False):
    assert Path('/Volumes/tmp').is_mount() and artifacts.resolve().is_relative_to(Path('/Volumes/tmp'))
    assert timeout > 0
    evidence = Path(tempfile.mkdtemp(prefix='cli-guarded-', dir=artifacts))
    def hashes():
        return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in SOURCE_FILES}
    before = hashes()
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    env = os.environ.copy()
    env.update(CI='1', CARGO_BUILD_JOBS='1', CARGO_INCREMENTAL='0',
               CARGO_TARGET_DIR=str(artifacts / 'target'), TMPDIR=str(artifacts / 'tmp'))
    argv = ['cargo', 'build', '-p', 'greppy', '--features', 'ci-test-assets', '--bin', 'greppy', '-j', '1']
    if test_web:
        argv = ['cargo', 'test', '-p', 'greppy', '--features', 'ci-test-assets', '--lib',
                '-j', '1', 'web::', '--', '--test-threads=1']
    if test_client:
        argv = ['cargo', 'test', '-p', 'greppy-web-client', '--lib', '-j', '1', 'workflow::', '--', '--test-threads=1']
    receipt = dict(argv=argv, cwd=str(root), head=head, source_before=before,
                   kind='client_unit_tests' if test_client else ('web_unit_tests' if test_web else 'cli_build'),
                   evidence=str(evidence), guard_seconds=timeout, guard_free_bytes=2 * 1024**3,
                   scope='debug CLI test-assets candidate; listed source hashes and HEAD; not all external assets/libraries or native runtime acceptance')
    disk_paths = (artifacts, root, Path(env['TMPDIR']))
    receipt['checked_disk_paths'] = [str(path) for path in disk_paths]
    assert min(shutil.disk_usage(path).free for path in disk_paths) >= 3 * 1024**3, 'artifact, source and temporary volumes each require 3 GiB free'
    print('Build evidence: ' + str(evidence), flush=True)
    start = time.monotonic()
    stopped = None
    with (evidence / 'build.log').open('w') as log:
        process = subprocess.Popen(argv, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        receipt['pid'] = process.pid
        (evidence / 'started.json').write_text(json.dumps(receipt, indent=2) + '\n')
        while process.poll() is None:
            if min(shutil.disk_usage(path).free for path in disk_paths) < receipt['guard_free_bytes']:
                stopped = 'disk guard'
            elif time.monotonic() - start > timeout:
                stopped = 'time guard'
            if stopped:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                break
            time.sleep(1)
        code = process.wait()
    receipt.update(exit_code=code, stopped=stopped, wall_seconds=time.monotonic() - start,
                   source_unchanged=before == hashes())
    if code == 0 and receipt['source_unchanged'] and not test_web and not test_client:
        binary = artifacts / 'target/debug/greppy'
        frozen = evidence / 'greppy'
        shutil.copy2(binary, frozen)
        frozen.chmod(0o555)
        receipt['candidate'] = dict(path=str(frozen), sha256=hashlib.sha256(frozen.read_bytes()).hexdigest(), bytes=frozen.stat().st_size)
        assert receipt['candidate']['sha256'] == hashlib.sha256(binary.read_bytes()).hexdigest()
    (evidence / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print((evidence / 'build.log').read_text())
    print(json.dumps(receipt), flush=True)
    return code if code > 0 else (0 if code == 0 and receipt['source_unchanged'] else 125)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--artifacts', type=Path, required=True)
    p.add_argument('--timeout', type=int, default=240)
    p.add_argument('--test-web', action='store_true', help='Run compiled CLI web unit tests; do not publish a CLI candidate')
    p.add_argument("--test-client", action="store_true", help="Run shared declarative workflow contract tests")
    a = p.parse_args()
    raise SystemExit(build(a.root.resolve(), a.artifacts.resolve(), a.timeout, a.test_web, a.test_client))
