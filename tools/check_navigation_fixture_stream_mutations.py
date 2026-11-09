#!/usr/bin/env python3
"""Prove real accepted-socket flags and fragmented-header fixture calibration."""
from pathlib import Path
import subprocess

source = Path('crates/web-runtime/runtime/tests/support/navigation_fixture_stream.rs')
original = source.read_text()
name = 'accepted_navigation_fixture_stream_reads_complete_headers'
command = ['cargo', 'test', '--locked', '--jobs', '2', '--manifest-path',
           'crates/web-runtime/Cargo.toml', '-p', 'web-runtime', '--lib',
           '--no-default-features', name, '--', '--test-threads=1', '--nocapture']


def run_positive():
    result = subprocess.run(command, capture_output=True, text=True, timeout=300)
    print(result.stdout, result.stderr)
    assert result.returncode == 0 and name + ' ... ok' in result.stdout


run_positive()
cases = [
    ('    stream.set_nonblocking(false)?; // normalize accepted fixture socket',
     '    // fault: accepted socket retains its inherited nonblocking flag',
     'accepted lifecycle socket retained nonblocking mode'),
    ('        if request.windows(4).any(|window| window == b"\\r\\n\\r\\n") {',
     '        if true { // fault: accept only the first TCP read',
     'lifecycle fixture accepted incomplete HTTP headers'),
]
for old, new, marker in cases:
    assert original.count(old) == 1, old
    try:
        source.write_text(original.replace(old, new))
        result = subprocess.run(command, capture_output=True, text=True, timeout=300)
        print(result.stdout, result.stderr)
        assert result.returncode == 101 and marker in result.stdout + result.stderr
        print('mutation: navigation fixture transport rejected', marker)
    finally:
        source.write_text(original)
    run_positive()
subprocess.run(['git', 'diff', '--exit-code', '--', str(source)], check=True)
