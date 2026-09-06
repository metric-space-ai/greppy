"""Source-bound native regression capture, with bounded owned process group.

Evidence tooling, not a release gate. The selected source set includes dirty
and untracked inputs; it does not claim full toolchain/dependency provenance.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time

root = Path(subprocess.check_output(['git', 'rev-parse', '--show-toplevel'], text=True).strip())
sources = [
    'crates/web-runtime/runtime/src/' + name for name in (
        'content_worker.rs', 'daemon.rs', 'lib.rs', 'supervisor.rs',
        'controller_worker.rs', 'protocol.rs', 'session.rs', 'observed_refs.rs',
        'locator_diagnostics.rs', 'wait_contract.rs', 'observed-ref-registry.js',
        'observed-ref-condition.js', 'native-label-text.js', 'observed-working-scope.js',
    )
] + [
    'crates/web-runtime/runtime/' + name for name in (
        'tests/session-daemon.rs', 'js/select-option-runtime.js',
        'tests/select-option-runtime.cjs', 'tests/select-choices-runtime.cjs',
        'tests/observed-ref-condition.cjs', 'fixtures/select-option-contract.mjs',
        'js/wait-for-function-runtime.js', 'tests/wait-for-function-runtime.cjs',
        'fixtures/wait-for-function-value.mjs', 'js/playwright.mjs', 'Cargo.toml',
    )
] + [
    'crates/web-client/src/lib.rs', 'crates/web-client/src/select-choices.js',
    'crates/web-client/src/describe-node.js', 'crates/web-client/Cargo.toml',
    'crates/web-runtime/Cargo.toml', 'crates/web-runtime/Cargo.lock',
    'contracts/web-runtime/page-state-v1.md',
    'contracts/web-runtime/internal-boolean-wait.md',
    'contracts/web-runtime/select-choices-v1.md',
    'crates/cli/src/web/expect.rs', 'crates/cli/src/web/common.rs',
    'crates/cli/src/web/runtimes.rs',
]

sources += ["crates/web-runtime/runtime/src/daemon_workflow.rs", "crates/web-runtime/runtime/tests/cases/workflow_cases.rs", "crates/web-client/src/workflow.rs", "crates/web-client/src/workflow-condition.js", "crates/web-client/src/workflow-preflight.js", "crates/web-client/src/protocol.rs"]

sources += ["crates/cli/src/web/" + name + ".rs" for name in ("act", "chain", "mod", "nav", "workflow")]

sources += ["crates/cli/src/web/view.rs", "crates/cli/src/web/view_scope.rs"]

def hashes():
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in sources}

def executable_hashes():
    result = {}
    for name in filter(None, os.environ.get('GREPPY_CAPTURE_EXECUTABLES', '').split(os.pathsep)):
        digest = hashlib.sha256()
        with open(name, 'rb') as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b''):
                digest.update(chunk)
        result[name] = digest.hexdigest()
    return result

def diff():
    return subprocess.check_output([
        'git', 'diff', '--binary', 'HEAD', '--', 'crates/web-runtime',
        'crates/web-client', 'crates/cli/src/web', 'contracts/web-runtime',
    ], cwd=root)

command = sys.argv[1:]
if not command:
    raise SystemExit('supply a command')
budget_s = int(os.environ.get('GREPPY_CAPTURE_BUDGET_S', '240'))
if not 1 <= budget_s <= 600:
    raise SystemExit('capture budget must be between 1 and 600 seconds')
artifact_root = Path('/Volumes/tmp/dev-artifacts/greppy')
checked_paths = [artifact_root, Path(os.environ.get("TMPDIR") or tempfile.gettempdir()), root]
if min(shutil.disk_usage(path).free for path in checked_paths) < 3 * 1024**3:
    raise SystemExit('refusing capture: artifact, source and temporary volumes each require at least 3 GiB free')
evidence = Path(tempfile.mkdtemp(prefix='native-capture-v3-run-', dir=artifact_root))
before = diff()
(evidence / 'source-before.patch').write_bytes(before)
for name in sources:
    target = evidence / 'source-before' / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes((root / name).read_bytes())
(evidence / 'capture.py').write_bytes(Path(__file__).read_bytes())
record = {
    'schema': 'greppy.native-regression-capture.v3', 'cwd': os.getcwd(),
    'source_root': str(root), 'command': command, 'started': time.time(),
    'budget_s': budget_s, 'checked_disk_paths': [str(path) for path in checked_paths], 'head': subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
    'source_hashes_before': hashes(),
    'executable_hashes_before': executable_hashes(),
    'capture_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'environment': {name: os.environ.get(name) for name in (
        'CARGO_TARGET_DIR', 'CARGO_BUILD_JOBS', 'CARGO_INCREMENTAL', 'TMPDIR',
        'RUSTUP_TOOLCHAIN', 'GREPPY_WEB_TRACE_PHASE', 'GREPPY_WORKFLOW_TEST_RUNTIME',
    )},
    'scope': 'Selected source bytes and command output; not signed-package acceptance.',
}
def write_record(name):
    (evidence / name).write_text(json.dumps(record, indent=2) + '\n')

write_record('invocation.json')
print('EVIDENCE_DIR=' + str(evidence), flush=True)
with (evidence / 'raw.log').open('wb') as output:
    child = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT,
                             start_new_session=True)
    record['child_pid'] = child.pid
    write_record('invocation.json')
    print('CAPTURE_CHILD_PID=' + str(child.pid), flush=True)
    deadline = time.monotonic() + budget_s
    while child.poll() is None:
        free = min(shutil.disk_usage(path).free for path in checked_paths)
        if free < 2 * 1024**3 or time.monotonic() >= deadline:
            record['guard'] = {'reason': 'disk' if free < 2 * 1024**3 else 'deadline',
                               'free_bytes': free, 'time': time.time()}
            try:
                os.killpg(child.pid, signal.SIGTERM)
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
            except ProcessLookupError:
                child.wait()
            break
        # Poll only our child, with a short observation interval.
        try:
            child.wait(timeout=0.25)
        except subprocess.TimeoutExpired:
            pass
    code = child.wait()
after = diff()
(evidence / 'source-after.patch').write_bytes(after)
record.update(exit_code=code, finished=time.time(), source_hashes_after=hashes(),
              executable_hashes_after=executable_hashes())
record['sources_unchanged'] = before == after and record['source_hashes_before'] == record['source_hashes_after']
record['executables_unchanged'] = record['executable_hashes_before'] == record['executable_hashes_after']
record['capture_exit_code'] = code if code else (124 if 'guard' in record else (0 if record['sources_unchanged'] and record['executables_unchanged'] else 86))
write_record('result.json')
print(json.dumps({key: record[key] for key in ('exit_code', 'capture_exit_code', 'sources_unchanged')}), flush=True)
sys.exit(record['capture_exit_code'])
