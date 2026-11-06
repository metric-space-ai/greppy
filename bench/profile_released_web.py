#!/usr/bin/env python3
"""Collect one real released Greppy CLI/browser baseline in the existing Linux CI.

No Greppy package install, model download or fallback engine. Only CLI/WebSDK
are extracted from the checksum-verified release. Explicit ephemeral CI may
provision the package's declared system dependencies before measured work. A failed target stays
a failed measurement; collection success is not a performance/quality verdict.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = 'metric-space-ai/greppy'


def sha256(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def extract_selected(package, destination, evidence):
    total = 0
    with (evidence / 'extract-stderr.log').open('wb') as errors:
        process = subprocess.Popen(
            ['timeout', '--kill-after=5s', '90s', 'dpkg-deb', '--threads-max=2',
             '--fsys-tarfile', str(package)], stdout=subprocess.PIPE, stderr=errors)
        try:
            with tarfile.open(fileobj=process.stdout, mode='r|') as archive:
                for member in archive:
                    name = member.name.removeprefix('./')
                    path = PurePosixPath(name)
                    if path.is_absolute() or '..' in path.parts:
                        raise ValueError('unsafe package member')
                    selected = (name == 'usr/lib/greppy/bin/greppy'
                                or name.startswith('usr/lib/greppy/bin/web-runtime/'))
                    if not selected or member.isdir():
                        continue
                    if not member.isfile():
                        raise ValueError('selected package entry is not a regular file: ' + name)
                    total += member.size
                    if total > 2 * 1024 ** 3:
                        raise ValueError('selected release exceeds the bounded extraction budget')
                    target = destination / path
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with archive.extractfile(member) as incoming, target.open('xb') as outgoing:
                        shutil.copyfileobj(incoming, outgoing, 1024 * 1024)
                    if target.stat().st_size != member.size:
                        raise ValueError('truncated package member')
                    target.chmod(member.mode & 0o755)
            if process.wait(timeout=10) != 0:
                raise RuntimeError('release extraction failed; see preserved stderr')
        finally:
            process.stdout.close()
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)
    return total


def inspect_environment(package, cli):
    """Record declared dependencies and isolated host loader probes before a task."""
    control = subprocess.run(['dpkg-deb', '-f', str(package), 'Package', 'Version', 'Depends'],
                             capture_output=True, text=True, timeout=15, check=True)
    version = subprocess.run([str(cli), '--version'], capture_output=True, text=True, timeout=15)
    if version.returncode != 0:
        raise RuntimeError('released CLI version probe failed')
    # Run in separate processes: do not retain graphics libraries in the meter.
    # This is preparation, not a measured cold-start or sandbox validation.
    probes = {}
    for library in ('libEGL.so.1', 'libEGL.so'):
        result = subprocess.run(
            [sys.executable, '-B', '-c',
             'import ctypes, sys; ctypes.CDLL(sys.argv[1])', library],
            capture_output=True, text=True, timeout=10)
        probes[library] = {'exit_code': result.returncode, 'stderr': result.stderr[-4096:]}
    return {'package_control': control.stdout.strip(), 'cli_version': version.stdout.strip(),
            'host_egl_loader_probes': probes,
            'scope': 'preparation outside timed worker; no OS-cache-cold or sandbox claim',
            'egl_loadable_on_host': any(p['exit_code'] == 0 for p in probes.values())}


def provision_dependencies(package, evidence):
    """Honor the verified package's simple Debian dependency list in opt-in CI."""
    depends = subprocess.run(['dpkg-deb', '-f', str(package), 'Depends'],
                             capture_output=True, text=True, timeout=15, check=True)
    packages = [name.strip() for name in depends.stdout.split(',')]
    if not packages or any(not re.fullmatch(r'[a-z][a-z0-9+.-]*', name) for name in packages):
        raise ValueError('dependency provisioning requires plain Debian package names')
    started = time.monotonic()
    # Only the explicitly requested hosted-CI preparation installs system packages.
    # GNU timeout bounds the whole apt process group, outside measured task costs.
    with (evidence / 'dependency-provisioning.log').open('wb') as log:
        for limit, arguments in ((90, ['update']),
                                 (180, ['install', '--no-install-recommends', '-y', '--', *packages])):
            subprocess.run(['sudo', '-n', 'timeout', '--kill-after=5s', str(limit) + 's',
                            'env', 'DEBIAN_FRONTEND=noninteractive', 'apt-get', *arguments],
                           stdout=log, stderr=subprocess.STDOUT, timeout=limit + 15, check=True)
    installed = subprocess.run(['dpkg-query', '-W', '-f=${Package}=${Version}\\n', *packages],
                               capture_output=True, text=True, timeout=15, check=True)
    return {'declared_packages': packages, 'installed_versions': installed.stdout.splitlines(),
            'wall_seconds': time.monotonic() - started,
            'scope': 'ephemeral CI preparation outside measured worker; no Greppy package/model installation'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', required=True)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--cpu-percent', type=int, default=200)
    parser.add_argument('--memory-mib', type=int, default=2048)
    parser.add_argument('--provision-dependencies', action='store_true',
                        help='Install declared system dependencies in explicit ephemeral CI preparation')
    args = parser.parse_args()
    if (not sys.platform.startswith('linux') or not args.output.is_absolute()
            or args.output.exists() or not re.fullmatch(r'v[0-9]+\.[0-9]+\.[0-9]+', args.release)
            or not 50 <= args.cpu_percent <= 400 or not 256 <= args.memory_mib <= 4096):
        parser.error('Linux, a new absolute output, release tag vN.N.N and bounded CPU/RAM are required')
    args.output.mkdir(parents=True)
    evidence = args.output / 'evidence'
    evidence.mkdir()
    report = {'schema': 'greppy.released-web-profile.v1', 'release': args.release,
              'repository': REPOSITORY, 'profile_source_sha256': sha256(Path(__file__)),
              'status': 'collecting', 'scope': 'real public CLI web do/session/run and owned runtime stop; '
              'excludes LLM/providers and deployment/download cost; no OS-cache-cold claim',
              'requested_worker_budget': {'cpu_percent': args.cpu_percent, 'memory_mib': args.memory_mib},
              'is_before_after_optimization_result': False}
    (evidence / 'protocol.json').write_text(json.dumps(report, indent=2) + '\n')
    code = 1
    try:
        if shutil.disk_usage(args.output).free < 5 * 1024 ** 3:
            raise RuntimeError('insufficient CI scratch capacity for the bounded release staging')
        download = args.output / 'download'
        download.mkdir()
        package = download / 'greppy-linux-x86_64.deb'
        subprocess.run(['gh', 'release', 'download', args.release, '--repo', REPOSITORY,
                        '--pattern', package.name, '--pattern', package.name + '.sha256',
                        '--dir', str(download)], check=True, timeout=180)
        expected = (download / (package.name + '.sha256')).read_text().split()
        if len(expected) != 2 or expected[1].lstrip('*') != package.name:
            raise ValueError('invalid official package checksum record')
        actual = sha256(package)
        if actual != expected[0].lower():
            raise ValueError('official release package checksum mismatch')
        report['package_sha256'] = actual
        sdk = args.output / 'sdk'
        sdk.mkdir()
        report['extracted_bytes'] = extract_selected(package, sdk, evidence)
        cli = sdk / 'usr/lib/greppy/bin/greppy'
        runtime_root = sdk / 'usr/lib/greppy/bin/web-runtime'
        runtime = runtime_root / 'bin/web-runtime'
        if not cli.is_file() or not runtime.is_file():
            raise RuntimeError('official release does not contain both required executables')
        report['cli_sha256'], report['runtime_sha256'] = sha256(cli), sha256(runtime)
        report['sdk_metadata'] = {}
        for name in ('provenance.json', 'coverage-manifest.json', 'benchmark-receipt.json'):
            path = runtime_root / name
            if path.is_file() and path.stat().st_size <= 128 * 1024:
                report['sdk_metadata'][name] = json.loads(path.read_text())
        report['environment_before_provisioning'] = inspect_environment(package, cli)
        if args.provision_dependencies:
            report['dependency_provisioning'] = provision_dependencies(package, evidence)
        report['environment'] = inspect_environment(package, cli)
        if not report['environment']['egl_loadable_on_host']:
            report['status'] = 'environment_preflight_failed'
            report['quality_verified'] = False
            report['quality_error'] = 'host cannot load either standard libEGL library; no workload was started'
            (evidence / 'profile.json').write_text(json.dumps(report, indent=2) + chr(10))
            print('RELEASE_PROFILE ' + json.dumps(report))
            return 0
        # The normal meter creates its own private state below this short parent.
        # Keep state separate from public evidence; never archive owner tokens.
        state_parent = Path(tempfile.mkdtemp(prefix='gwp-', dir='/tmp'))
        report['private_state_parent'] = str(state_parent)
        xdg = state_parent / 'xdg'
        xdg.mkdir(mode=0o700)
        report['private_xdg_runtime_dir'] = str(xdg)
        unit = 'greppy-release-profile-' + str(os.getpid()) + '-' + str(time.time_ns())
        parent = '/sys/fs/cgroup/system.slice/' + unit + '.service'
        report['unit'] = {'name': unit + '.service', 'runtime_max_seconds': 150,
                          'stop_condition': 'synchronous process exit; systemd --wait/--collect',
                          'owner': os.environ.get('GITHUB_RUN_ID', 'explicit-local-caller')}
        measurement = evidence / 'measurement.json'
        command = ['sudo', '-n', 'systemd-run', '--unit=' + unit, '--wait', '--collect', '--pipe',
                   '--service-type=exec', '--uid=' + str(os.getuid()), '--gid=' + str(os.getgid()),
                   '--property=Delegate=memory', '--property=DelegateSubgroup=monitor',
                   '--property=MemoryMax=' + str(args.memory_mib) + 'M',
                   '--property=CPUQuota=' + str(args.cpu_percent) + '%',
                   '--property=TasksMax=64', '--property=RuntimeMaxSec=150', '--expand-environment=no',
                   '--setenv=XDG_RUNTIME_DIR=' + str(xdg),
                   '--working-directory=' + str(ROOT), sys.executable, '-B',
                   str(ROOT / 'crates/web-runtime/scripts/measure-web-efficiency.py'),
                   '--engine', 'cli', '--cli', str(cli), '--runtime', str(runtime),
                   '--state-parent', str(state_parent), '--kernel-cgroup-parent', parent,
                   '--capture-owned-runtime-stderr',
                   '--output', str(measurement), '--rounds', '3', '--tasks', '2',
                   '--concurrency', '1', '--nodes', '100', '--timeout', '90']
        with (evidence / 'worker-stdout.log').open('wb') as stdout, (evidence / 'worker-stderr.log').open('wb') as stderr:
            result = subprocess.run(command, stdout=stdout, stderr=stderr, timeout=170)
        report['worker_exit_code'] = result.returncode
        if measurement.is_file():
            data = json.loads(measurement.read_text())
            report['status'] = 'measurement_collected'
            report['quality_verified'] = data['quality_verified']
            report['quality_error'] = data['quality_error']
            report['measurement_sha256'] = sha256(measurement)
            match = re.search(r'startup stderr retained at ([^;\n]+)',
                              data['metrics'].get('stdout_tail', ''))
            if match:
                startup = Path(match[1])
                if (startup.is_file() and not startup.is_symlink()
                        and startup.resolve().is_relative_to(state_parent.resolve())
                        and startup.stat().st_size <= 64 * 1024):
                    raw = startup.read_bytes()
                    text = raw.decode('utf-8', errors='replace')
                    text = re.sub(r'[A-Za-z0-9_+/=-]{32,}', '<opaque>', text)
                    (evidence / 'startup-stderr.redacted.log').write_text(text)
                    report['startup_diagnostic'] = {
                        'redacted_artifact': 'startup-stderr.redacted.log',
                        'original_sha256': hashlib.sha256(raw).hexdigest(),
                        'scope': 'only the explicitly reported log inside our new private state'}
                else:
                    report['startup_diagnostic'] = {'status': 'unavailable_or_outside_private_state'}
            report['whole_cli_sampled_surface_verified'] = data['metrics']['process_family_memory'].get(
                'complete_for_execution_surface', False)
        else:
            report['status'] = 'measurement_unavailable'
            report['quality_verified'] = None
            report['quality_error'] = 'worker produced no final receipt; see bounded unit logs'
        # A correctly collected failed workload is data, never a fabricated win.
        code = 0 if measurement.is_file() else 1
    except Exception as error:
        report['status'] = 'collection_failed'
        report['error'] = type(error).__name__ + ': ' + str(error)
    (evidence / 'profile.json').write_text(json.dumps(report, indent=2) + '\n')
    print('RELEASE_PROFILE ' + json.dumps({key: report.get(key) for key in
          ('status', 'release', 'quality_verified', 'quality_error', 'worker_exit_code', 'error')}))
    return code


if __name__ == '__main__':
    raise SystemExit(main())
