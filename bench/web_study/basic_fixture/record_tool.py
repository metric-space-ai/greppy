"""Opt-in research command: web records QUERY --fields data-* --where PREDICATES.

All browser work stays in the delegated Greppy context. This Python orchestration
is measured as part of the experimental system, not claimed as a native feature.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

class RecordError(Exception):
    def __init__(self, code, message, source=None):
        super().__init__(message)
        self.code, self.source = code, source


def checked(payload):
    if payload.get('schema') != 'greppy.web-runtime.v1' or payload.get('status') != 'ok':
        raise RecordError('UPSTREAM_RESULT', 'The upstream result is not a successful known envelope.', payload)
    return payload.get('result', {})


def collect(delegate, query, fields, where, call=None):
    start = time.monotonic()
    calls = []
    def invoke(*args, stdin=None):
        argv = [str(delegate), 'web', *args]
        if call:
            code, output, stderr = call(argv, stdin)
        else:
            p = subprocess.run(argv, input=stdin, text=True, capture_output=True, timeout=60)
            code, output, stderr = p.returncode, p.stdout, p.stderr
        calls.append({'argv': argv, 'exit_code': code, 'stdout_bytes': len(output.encode())})
        if code:
            raise RecordError('UPSTREAM_EXIT', 'A delegated command failed.',
                              {'argv': argv, 'exit_code': code, 'stdout': output, 'stderr': stderr})
        return output
    rows = []
    context = {'scope_query': query, 'frame_scope': 'current_document_only'}
    try:
        if any(not re.fullmatch(r'data-[a-zA-Z0-9_-]+', field) for field in fields):
            raise RecordError('FIELDS', 'Experimental attribute fields are named data-* attributes.')
        envelope = json.loads(invoke('observe', query, '--json'))
        result = checked(envelope)
        context.update(request_id=envelope.get('request_id'), url=result.get('url'))
        scope = result.get('observation_scope') or {}
        if result.get('refs_truncated') or any(v is True for k, v in scope.items() if k.endswith('_truncated')):
            raise RecordError('INCOMPLETE', 'The source observation is truncated.', envelope)
        source_rows = result.get('actionables')
        if not isinstance(source_rows, list):
            raise RecordError('SHAPE', 'The observation has no actionable record array.', envelope)
        refs = [r.get('ref') for r in source_rows]
        if any(not isinstance(ref, str) or not re.fullmatch(r'@[0-9]+', ref) for ref in refs) or len(set(refs)) != len(refs):
            raise RecordError('REFS', 'Observation references are missing or non-unique.', envelope)
        if result.get('ref_count') != len(refs):
            raise RecordError('INCOMPLETE', 'Returned records and reference count differ.', envelope)
        for source_row in source_rows:
            inspection = json.loads(invoke('inspect', source_row['ref'], '--attrs', '--json'))
            detail = checked(inspection)
            session = detail.get('session_id')
            if context.get('session_id') and context['session_id'] != session:
                raise RecordError('CONTEXT', 'The delegated session changed during collection.', inspection)
            context['session_id'] = session
            node = detail.get('value', {}).get('node')
            if not isinstance(node, dict) or not isinstance(node.get('visible'), bool):
                raise RecordError('SHAPE', 'Inspection has no explicit node visibility.', inspection)
            attrs = node.get('attrs')
            if not isinstance(attrs, dict):
                raise RecordError('SHAPE', 'Inspection has no attribute object.', inspection)
            row = {k: source_row[k] for k in ('ref', 'role', 'name', 'disabled', 'checked', 'selected', 'value')
                   if k in source_row and source_row[k] is not None}
            row['visible'] = node['visible']
            row['inspection_request_id'] = inspection.get('request_id')
            for field in fields:
                # Missing remains null, not zero/false or an inferred value.
                row[field] = attrs.get(field)
            rows.append(row)
        matched = rows
        if where:
            lines = ''.join(json.dumps(r, separators=(',', ':')) + '\n' for r in rows)
            output = invoke('match', where, stdin=lines)
            matched = [json.loads(line) for line in output.splitlines() if line.strip()]
            if any(row not in rows for row in matched):
                raise RecordError('FILTER_SHAPE', 'Filtered records differ from the source records.')
        return {'schema': 'greppy.web.records-experiment.v1', 'status': 'ok',
                'untrusted_content_boundary': 'UNTRUSTED_PAGE_CONTENT', 'context': context,
                'counts': {'observed': len(rows), 'matched': len(matched)}, 'records': matched,
                'consistency': 'sequential_native_reads; references revalidated by delegated actions',
                'processing': {'native_commands': len(calls), 'seconds': time.monotonic()-start}}, 0
    except (RecordError, ValueError, TypeError, subprocess.TimeoutExpired) as error:
        return {'schema': 'greppy.web.records-experiment.v1', 'status': 'error',
                'untrusted_content_boundary': 'UNTRUSTED_PAGE_CONTENT', 'context': context,
                'error': {'code': getattr(error, 'code', 'DECODE_OR_TIMEOUT'), 'message': str(error),
                          'source': getattr(error, 'source', None)},
                'counts': {'inspected_before_error': len(rows)}, 'records': [],
                'processing': {'native_commands': len(calls), 'seconds': time.monotonic()-start}}, 1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--delegate', required=True, type=Path)
    p.add_argument('command', nargs=argparse.REMAINDER)
    a = p.parse_args()
    command = a.command
    if command[:2] != ['web', 'records']:
        os.execv(str(a.delegate), [str(a.delegate), *command])
    rp = argparse.ArgumentParser(prog='web records', description=__doc__)
    rp.add_argument('query')
    rp.add_argument('--fields', default='')
    rp.add_argument('--where', default='')
    r = rp.parse_args(command[2:])
    payload, code = collect(a.delegate, r.query, [f for f in r.fields.split(',') if f], r.where)
    print(json.dumps(payload, separators=(',', ':')))
    return code

if __name__ == '__main__':
    sys.exit(main())
