"""Inventory complete extracted archive members without admitting their contents.

No source text is emitted. CRC/size/hash integrity does not prove public origin,
successful log retrieval, privacy, independence or final-test freshness.
"""
import argparse
import codecs
from collections import Counter
import hashlib
import json
from pathlib import Path
import zlib

from contracts import canonical
from experiments import file_sha


def classify_payload(prefix, size):
    if size == 0:
        return 'empty'
    # Complete small JSON API failures are acquisition failures for a build-log
    # archive, not evidence of a failed build. Their text may be retained only as
    # a separately labelled network diagnostic after its own admission review.
    if size <= len(prefix):
        try:
            value = json.loads(prefix.decode('utf-8'))
        except (ValueError, UnicodeError):
            value = None
        if isinstance(value, dict) and isinstance(value.get('message'), str):
            status = value.get('status')
            if str(status) in ('400', '401', '403', '404', '410', '422', '429', '500', '502', '503'):
                return 'retrieval_error_json'
            if 'rate limit exceeded' in value['message'].lower():
                return 'retrieval_error_json'
    stripped = prefix.lstrip().lower()
    if stripped.startswith((b'<!doctype html', b'<html')):
        return 'html_response_requires_review'
    return 'text_capture_requires_review'


def inspect_member(root, name):
    relative = Path(name)
    path = root / relative
    if relative.is_absolute() or '..' in relative.parts or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError('archive member escapes extraction directory')
    child = root
    for component in relative.parts:
        child = child / component
        if child.is_symlink():
            raise ValueError('symlink in archive member path')
    if not path.is_file():
        raise ValueError('missing extracted archive member')
    h = hashlib.sha256()
    crc = 0
    size = 0
    lines = 0
    tail = b''
    prefix = bytearray()
    valid_utf8 = True
    decoder = codecs.getincrementaldecoder('utf-8')('strict')
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
            crc = zlib.crc32(block, crc)
            size += len(block)
            lines += block.count(b'\n')
            tail = block[-1:]
            if len(prefix) < 65536:
                prefix.extend(block[:65536 - len(prefix)])
            if valid_utf8:
                try:
                    decoder.decode(block)
                except UnicodeError:
                    valid_utf8 = False
    if valid_utf8:
        try:
            decoder.decode(b'', final=True)
        except UnicodeError:
            valid_utf8 = False
    lines += bool(size and tail != b'\n')
    return {'member': name, 'sha256': h.hexdigest(), 'bytes': size, 'lines': lines,
            'crc32': crc & 0xffffffff, 'valid_utf8': valid_utf8,
            'capture_kind': classify_payload(bytes(prefix), size),
            'privacy_admitted': False, 'complete_build_output_verified': False,
            'final_test_eligible': False}


def audit_archive(archive, extracted):
    import py7zr
    archive, extracted = Path(archive), Path(extracted)
    with py7zr.SevenZipFile(archive, mode='r') as stream:
        members = stream.list()
    seen = set()
    results = []
    for member in members:
        if member.filename in seen:
            raise ValueError('duplicate archive member')
        seen.add(member.filename)
        if member.is_directory:
            continue
        if not member.is_file or member.is_symlink:
            raise ValueError('unsupported archive member type')
        row = inspect_member(extracted, member.filename)
        if row['bytes'] != member.uncompressed or member.crc32 is None or row['crc32'] != member.crc32:
            raise ValueError('extracted member does not match archive size/CRC')
        row['archive_member_matches'] = True
        results.append(row)
    with py7zr.SevenZipFile(archive, mode='r') as stream:
        bad_member = stream.testzip()
        if bad_member is not None:
            raise ValueError('archive decompression CRC failed')
    actual = {str(p.relative_to(extracted)) for p in extracted.rglob('*') if p.is_file() or p.is_symlink()}
    expected = {row['member'] for row in results}
    extras = sorted(actual - expected)
    return {'schema': 'greppy.heads.log-archive-audit.v1', 'archive_path': str(archive),
            'archive_sha256': file_sha(archive), 'archive_bytes': archive.stat().st_size,
            'extraction_path': str(extracted), 'py7zr_version': py7zr.__version__,
            'archive_decompression_crc_passed': True, 'member_count': len(results),
            'member_bytes': sum(row['bytes'] for row in results),
            'capture_kind_counts': dict(Counter(row['capture_kind'] for row in results)),
            'max_lines': max((row['lines'] for row in results), default=0),
            'at_least_100000_lines': sum(row['lines'] >= 100000 for row in results),
            'invalid_utf8_members': sum(not row['valid_utf8'] for row in results),
            'extra_extracted_files_not_admitted': extras, 'members': results,
            'source_origin_verified': False, 'admitted_build_outputs': 0,
            'note': 'Archive integrity only. Public provenance, privacy, full build capture and source lineage still need independent verification.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--extracted', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    report = audit_archive(args.archive, args.extracted)
    with args.out.open('x') as stream:
        stream.write(canonical(report) + '\n')
    print(canonical({k: v for k, v in report.items() if k not in ('members', 'extra_extracted_files_not_admitted')}))
