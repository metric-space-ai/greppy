"""Content identity checks outside measured agent turns; never a speed metric."""
import hashlib
import os
from pathlib import Path


def executable(path):
    path = Path(path).resolve(strict=True)
    if not path.is_file() or not os.access(path, os.X_OK):
        raise ValueError('candidate must be an executable file')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return {'path': str(path), 'sha256': digest.hexdigest(), 'bytes': path.stat().st_size}


def capture(cli, runtime):
    return {'schema': 'greppy.web-study.candidate.v1',
            'cli': executable(cli), 'runtime': executable(runtime),
            'scope': 'CLI and runtime executable bytes; not all libraries/assets or source provenance'}


def verify(candidate):
    errors = []
    if candidate.get('schema') != 'greppy.web-study.candidate.v1':
        return {'ok': False, 'errors': ['unsupported candidate schema']}
    for name in ('cli', 'runtime'):
        try:
            expected = candidate[name]
            actual = executable(expected['path'])
            if actual != expected:
                errors.append(name + ': executable identity changed')
        except (OSError, ValueError, KeyError) as error:
            errors.append(name + ': ' + str(error))
    return {'ok': not errors, 'errors': errors,
            'scope': 'post-run byte check against preregistration; does not freeze files'}
