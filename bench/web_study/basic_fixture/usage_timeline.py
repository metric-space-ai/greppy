"""Audit provider usage beside public calls; never estimate tokens or read reasoning."""
import argparse
import hashlib
import json
from pathlib import Path

FIELDS = ('input_tokens', 'output_tokens', 'cached_input_tokens')


def timeline(records):
    previous = None
    pending = []
    rows = []
    problems = []
    for line, record in enumerate(records, 1):
        payload = record.get('payload', {})
        if record.get('type') == 'response_item' and payload.get('type') in ('function_call', 'custom_tool_call'):
            pending.append({'line': line, 'call_id': payload.get('call_id'), 'name': payload.get('name')})
        if record.get('type') != 'event_msg' or payload.get('type') != 'token_count':
            continue
        info = payload.get('info') or {}
        total = info.get('total_token_usage')
        last = info.get('last_token_usage')
        if not isinstance(total, dict) or not isinstance(last, dict):
            problems.append({'line': line, 'reason': 'missing provider usage'})
            continue
        if any(type(total.get(k)) is not int or type(last.get(k)) is not int for k in FIELDS):
            problems.append({'line': line, 'reason': 'missing or noninteger counters'})
            continue
        current = {k: total[k] for k in FIELDS}
        if previous == current:
            continue  # Repeated cumulative telemetry is not another model response.
        delta = {k: current[k] - (previous[k] if previous is not None else 0) for k in FIELDS}
        valid = all(delta[k] >= 0 and delta[k] == last[k] for k in FIELDS)
        if not valid:
            problems.append({'line': line, 'reason': 'counter reset, missing response or nonzero prior-turn baseline'})
        if not 0 <= delta['cached_input_tokens'] <= delta['input_tokens']:
            valid = False
            problems.append({'line': line, 'reason': 'invalid cache counters'})
        rows.append({'usage_line': line, 'calls': pending, 'tokens': delta if valid else None,
                     'attribution': 'generation with these calls; not causal savings' if pending else 'no public call since preceding usage'})
        pending = []
        previous = current
    if pending:
        problems.append({'reason': 'calls without following provider usage', 'calls': pending})
    if not rows:
        problems.append({'reason': 'no provider usage'})
    complete = not problems
    return {'schema': 'greppy.web.provider-usage-timeline.v1', 'complete': complete,
            'total': previous, 'responses': rows, 'problems': problems,
            'limits': ['Calls in one generation share usage; per-call tokens are not fabricated.',
                       'Temporal association does not prove avoidable or causal token cost.',
                       'Cached input remains input; it is reported separately, never subtracted from acceptance.',
                       'Only public call metadata and provider counters are inspected.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trace', type=Path, help='One bounded participant-turn JSONL export')
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    raw = args.trace.read_bytes()
    result = timeline(json.loads(line) for line in raw.splitlines() if line.strip())
    result['source'] = {'path': str(args.trace.resolve()), 'sha256': hashlib.sha256(raw).hexdigest()}
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps({'output': str(args.output), 'complete': result['complete'],
                      'responses': len(result['responses']), 'total': result['total']}))
    raise SystemExit(0 if result['complete'] else 1)


if __name__ == '__main__':
    main()
