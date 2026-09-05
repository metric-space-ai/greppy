"""Index verified visible native Web results; never export private trace context."""
import argparse
import hashlib
from pathlib import Path

from contracts import canonical, digest, strict_json

TRIAL_SCHEMAS = {'greppy.web-study.pilot.v1', 'greppy.web-study.basic.v1'}
ACTIONS = {'web.goto', 'web.back', 'web.forward', 'web.reload', 'web.click',
           'web.fill', 'web.type', 'web.clear', 'web.select', 'web.check',
           'web.uncheck', 'web.press', 'web.hover', 'web.scroll', 'web.upload'}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def load(path):
    raw = Path(path).read_bytes()
    return strict_json(raw), sha(raw)


def index_trial(path, *, family, exclusion=None):
    path = Path(path).resolve()
    trial, trial_hash = load(path)
    if trial.get('schema') not in TRIAL_SCHEMAS:
        raise ValueError('unsupported completed study trial')
    artifacts = trial['artifacts']
    trace_path = Path(artifacts['trace']).resolve()
    metadata_path = Path(artifacts['metadata']).resolve()
    manifest_path = Path(artifacts['manifest']).resolve()
    manifest, manifest_hash = load(manifest_path)
    metadata, metadata_hash = load(metadata_path)
    if (manifest.get('schema_version') != 'codex-trace-manifest.v1'
            or metadata.get('schema_version') != 'codex-trace-export.v1'):
        raise ValueError('unsupported trace manifest or metadata')
    if (Path(manifest['export']).resolve() != trace_path
            or Path(manifest['metadata']).resolve() != metadata_path
            or manifest['source'] != metadata['source']):
        raise ValueError('artifact path/source mismatch')
    if not (trial['turn_id'] == manifest['turn_id'] == metadata['turn_id']):
        raise ValueError('trial/trace turn mismatch')
    raw = trace_path.read_bytes()
    if sha(raw) != manifest['sha256'] or len(raw) != manifest['byte_length']:
        raise ValueError('trace checksum or byte length mismatch')
    lines = raw.splitlines(keepends=True)
    bounds = manifest['line_boundaries']
    byte_bounds = manifest['byte_boundaries']
    if (bounds != metadata['line_boundaries'] or byte_bounds != metadata['byte_boundaries']
            or bounds['first'] < 1 or bounds['last'] - bounds['first'] + 1 != len(lines)
            or bounds['record_count'] != len(lines)
            or byte_bounds['end_exclusive'] - byte_bounds['start'] != len(raw)):
        raise ValueError('trace boundary mismatch')
    # Parse only tool records referenced by metadata. Reasoning and arbitrary
    # messages in the contiguous export are neither parsed nor copied.
    requests = {}; seen_lines = set(); responses = set(); events = []; observations = []
    last_actions = {}
    episode_id = digest(['study episode', manifest['sha256'], trial['turn_id']])
    for tool in metadata['tool_calls']:
        line = tool['source_line']
        if type(line) is not int or not bounds['first'] <= line <= bounds['last'] or line in seen_lines:
            raise ValueError('invalid or duplicate tool source line')
        if seen_lines and line <= max(seen_lines):
            raise ValueError('tool chronology is not strictly increasing')
        seen_lines.add(line)
        record = strict_json(lines[line - bounds['first']])
        payload = record.get('payload', {})
        call = tool['call_id']
        if record.get('type') != 'response_item' or payload.get('call_id') != call:
            raise ValueError('tool source pointer mismatch')
        if tool['kind'] == 'request':
            if call in requests or payload.get('type') not in ('custom_tool_call', 'function_call'):
                raise ValueError('duplicate or invalid tool request')
            argument = payload.get('input') if payload['type'] == 'custom_tool_call' else payload.get('arguments')
            if payload.get('name') != tool['name'] or argument != tool['arguments']:
                raise ValueError('metadata request differs from original tool record')
            requests[call] = line
            continue
        if (tool['kind'] != 'response' or call not in requests or call in responses
                or payload.get('type') not in ('custom_tool_call_output', 'function_call_output')
                or payload.get('output') != tool['result']):
            raise ValueError('metadata response differs from original tool record')
        responses.add(call)
        # Accept only complete JSON text blocks, never JSON guessed from prose,
        # nested page strings, code arguments, reasoning or assistant assertions.
        output = tool['result']
        blocks = output if isinstance(output, list) else [{'type': 'input_text', 'text': output}]
        for index, block in enumerate(blocks):
            if not isinstance(block, dict) or block.get('type') not in ('input_text', 'text'):
                continue
            text = block.get('text')
            if not isinstance(text, str):
                continue
            try:
                envelope = strict_json(text)
            except (ValueError, TypeError):
                continue
            if not isinstance(envelope, dict):
                continue
            adapter = None
            if envelope.get('schema') == 'greppy.web-study.action-observe.v1':
                if (type(envelope.get('action_exit_code')) is not int or envelope['action_exit_code'] != 0
                        or type(envelope.get('observation_exit_code')) is not int
                        or envelope.get('subprocess_count') != 2
                        or envelope.get('task_success') != 'not_evaluated'
                        or not isinstance(envelope.get('action'), dict)
                        or not isinstance(envelope.get('observation'), dict)):
                    raise ValueError('invalid explicit action-observation adapter envelope')
                adapter = {'container_sha256': digest(envelope),
                           'container_schema': envelope['schema'],
                           'decoded_json_pointer': '/observation',
                           'action': {'source_sha256': digest(envelope['action']),
                                      'decoded_json_pointer': '/action', 'exit_code': 0,
                                      'operation': envelope['action'].get('operation'),
                                      'request_id': envelope['action'].get('request_id'),
                                      'session_id': envelope['action'].get('session_id'),
                                      'task_success': 'not_evaluated'},
                           'observation_exit_code': envelope['observation_exit_code']}
                envelope = envelope['observation']
                if envelope.get('schema') == 'greppy.web-runtime.v1':
                    if (envelope.get('status') == 'ok') != (adapter['observation_exit_code'] == 0):
                        raise ValueError('adapter exit code contradicts native observation status')
                elif adapter['observation_exit_code'] != 0:
                    # Keep no successful observation from a failed result-only
                    # capture; its full source remains available for diagnosis.
                    continue
            # Some study adapters emitted the native observation result only.
            # Record its shape explicitly; do not fabricate a runtime envelope,
            # request ID, operation, status, session or action attribution.
            result_only = (envelope.get('untrusted_content_boundary') == 'UNTRUSTED_PAGE_CONTENT'
                           and all(isinstance(envelope.get(k), list) for k in ('actionables', 'headings', 'links'))
                           and type(envelope.get('ref_count')) is int
                           and type(envelope.get('refs_truncated')) is bool
                           and all(isinstance(envelope.get(k), str) for k in ('text', 'title', 'url')))
            pointer = '/payload/output' + (f'/{index}/text' if isinstance(output, list) else '')
            if envelope.get('schema') != 'greppy.web-runtime.v1':
                if result_only:
                    event_id = digest([episode_id, line, pointer])
                    events.append({'id': event_id, 'source_line': line, 'adapter': adapter,
                                   'export_line': line - bounds['first'] + 1, 'json_pointer': pointer,
                                   'source_text_sha256': sha(text.encode()), 'snapshot_sha256': digest(envelope),
                                   'format': 'observation_result_only', 'operation': None,
                                   'request_id': None, 'status': None, 'session_id': None})
                    observations.append({'id': event_id, 'event_id': event_id,
                                         'snapshot_sha256': digest(envelope), 'format': 'observation_result_only',
                                         'last_action': adapter['action'] if adapter else None,
                                         'action_context_status': 'explicit_adapter_pair' if adapter else 'unknown',
                                         'goal': None, 'privacy_review': None, 'admission': 'held'})
                continue
            operation = envelope.get('operation')
            unknown_error = operation is None and envelope.get('status') == 'error'
            if ((not unknown_error and
                 (not isinstance(operation, str) or not operation.startswith('web.')))
                    or envelope.get('status') not in ('ok', 'error')
                    or not isinstance(envelope.get('request_id'), str)):
                raise ValueError(f'malformed native Web envelope at {trace_path}:{line}: operation={operation!r}, status={envelope.get("status")!r}')
            result = envelope.get('result')
            session = result.get('session_id') if isinstance(result, dict) else None
            pointer = '/payload/output' + (f'/{index}/text' if isinstance(output, list) else '')
            event = {'id': digest([episode_id, line, pointer]), 'source_line': line,
                     'adapter': adapter,
                     'export_line': line - bounds['first'] + 1, 'json_pointer': pointer,
                     'source_text_sha256': sha(text.encode()), 'envelope_sha256': digest(envelope),
                     'format': 'native_envelope',
                     'operation': operation, 'request_id': envelope['request_id'],
                     'status': envelope['status'], 'session_id': session}
            events.append(event)
            if unknown_error:
                last_actions.clear()
            if operation in ACTIONS:
                if isinstance(session, str) and session:
                    last_actions[session] = {'operation': operation, 'request_id': envelope['request_id'],
                                             'outcome': envelope['status'], 'event_id': event['id']}
                else:
                    last_actions.clear()  # Unknown scope may affect any active session.
            if operation == 'web.observe' and envelope['status'] == 'ok' and isinstance(result, dict):
                action = last_actions.get(session) if isinstance(session, str) else None
                if adapter:
                    action = adapter['action']
                observations.append({'id': event['id'], 'event_id': event['id'],
                                     'format': 'native_envelope',
                                     'snapshot_sha256': digest(envelope), 'last_action': action,
                                     'action_context_status': ('explicit_adapter_pair' if adapter else
                                                               'observed_same_session' if action else 'unknown'),
                                     'goal': None, 'privacy_review': None, 'admission': 'held'})
    oracle = trial.get('oracle')
    if not isinstance(oracle, dict) or type(oracle.get('ok')) is not bool:
        raise ValueError('trial lacks an explicit independent oracle outcome')
    return {'schema': 'greppy.heads.study-episode-index.v1', 'episode_id': episode_id,
            'family': family, 'group_key': family, 'split': 'development',
            'previously_exposed': True, 'final_eligible': False, 'production_eligible': False,
            'exclusion': exclusion, 'source': {'trial': str(path), 'trial_sha256': trial_hash,
                'trace': str(trace_path), 'trace_sha256': manifest['sha256'],
                'metadata': str(metadata_path), 'metadata_sha256': metadata_hash,
                'manifest': str(manifest_path), 'manifest_sha256': manifest_hash},
            'oracle': {'ok': oracle['ok'], 'sha256': digest(oracle), 'teacher_context': False},
            'events': events, 'observations': observations,
            'hold_reasons': ['explicit_goal_binding_required', 'privacy_review_required',
                             'annotation_and_independent_evidence_required'],
            'note': 'Contains source pointers and technical receipts only. No trace messages or page text.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True,
                        help='explicit list of trial paths, source families and exclusions')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    source, source_hash = load(args.manifest)
    if source.get('schema') != 'greppy.heads.study-import.v1':
        raise ValueError('unsupported study import manifest')
    episodes = [index_trial(item['trial'], family=item['family'], exclusion=item.get('exclusion'))
                for item in source['trials']]
    if len({item['episode_id'] for item in episodes}) != len(episodes):
        raise ValueError('duplicate source episodes')
    report = {'schema': 'greppy.heads.study-corpus-index.v1', 'import_manifest_sha256': source_hash,
              'episode_count': len(episodes), 'observation_count': sum(len(x['observations']) for x in episodes),
              'excluded_episodes': sum(x['exclusion'] is not None for x in episodes),
              'oracle_failed_episodes': sum(not x['oracle']['ok'] for x in episodes),
              'admitted_episodes': 0, 'final_eligible_episodes': 0, 'episodes': episodes}
    with args.out.open('x') as stream:
        stream.write(canonical(report) + '\n')
    print(canonical({k: v for k, v in report.items() if k != 'episodes'}))


if __name__ == '__main__':
    main()
