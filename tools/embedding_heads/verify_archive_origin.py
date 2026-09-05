"""Verify archived bytes against an anonymously downloadable dataset artifact."""
import argparse
import json
import os
from pathlib import Path, PurePosixPath
import re

from contracts import canonical
from experiments import file_sha


def verify_origin(repo, filename, expected_sha256, cache):
    if not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', repo):
        raise ValueError('invalid dataset repository')
    relative = PurePosixPath(filename)
    if relative.is_absolute() or '..' in relative.parts or not relative.parts:
        raise ValueError('invalid dataset artifact path')
    if not re.fullmatch(r'[0-9a-f]{64}', expected_sha256):
        raise ValueError('invalid expected checksum')
    # An existing token or offline cache must not stand in for anonymous origin
    # verification. This is artifact acquisition, not a browser page extraction.
    os.environ['HF_HUB_OFFLINE'] = '0'
    os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN'] = '1'
    # Avoid shared Xet chunk-cache failures; use the client's supported HTTP path.
    os.environ['HF_HUB_DISABLE_XET'] = '1'
    os.environ.setdefault('HF_HUB_ETAG_TIMEOUT', '20')
    os.environ.setdefault('HF_HUB_DOWNLOAD_TIMEOUT', '60')
    from huggingface_hub import hf_hub_download, __version__
    path = Path(hf_hub_download(repo_id=repo, repo_type='dataset', filename=filename,
                               token=False, cache_dir=str(cache), force_download=True))
    parts = path.relative_to(cache).parts
    if len(parts) < 4 or parts[1] != 'snapshots' or not re.fullmatch(r'[0-9a-f]{40}', parts[2]):
        raise ValueError('download did not provide a pinned dataset revision')
    revision = parts[2]
    actual = file_sha(path)
    return {'schema': 'greppy.heads.archive-origin.v1', 'repo_id': repo,
            'repo_type': 'dataset', 'revision': revision, 'filename': filename,
            'anonymous_download_completed': True, 'client_version': __version__,
            'transport': 'http', 'verifier_sha256': file_sha(__file__),
            'expected_sha256': expected_sha256, 'downloaded_sha256': actual,
            'source_origin_verified': actual == expected_sha256,
            'downloaded_bytes': path.stat().st_size, 'cache_path': str(path),
            'immutable_url': f'https://huggingface.co/datasets/{repo}/resolve/{revision}/{filename}',
            'privacy_admitted': False, 'complete_build_output_verified': False,
            'final_test_eligible': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', required=True)
    parser.add_argument('--filename', required=True)
    parser.add_argument('--expected-sha256', required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    report = verify_origin(args.repo, args.filename, args.expected_sha256, args.cache.resolve())
    with args.out.open('x') as stream:
        stream.write(canonical(report) + '\n')
    print(json.dumps(report, indent=2))
    if not report['source_origin_verified']:
        raise SystemExit('downloaded bytes differ; historical archive origin remains unverified')
