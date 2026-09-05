"""Conservative source grouping and byte-exact log target spans.

These helpers never infer complete capture, privacy approval, or final-test freshness
from public dataset names. Those are separately verified admission facts.
"""
import hashlib
import re
from contracts import digest, redact_text

GROUPING_VERSION = 'source-template-groups-v1'


def source_spans(text, source_id):
    """One target per physical line, including its original terminator and byte range."""
    if not isinstance(text, str) or not text:
        raise ValueError('source text must be nonempty')
    offset = 0
    result = []
    for number, line in enumerate(text.splitlines(keepends=True), 1):
        raw = line.encode('utf-8')
        end = offset + len(raw)
        result.append({'id': digest([source_id, offset, end, hashlib.sha256(raw).hexdigest()]),
                       'text': line, 'byte_start': offset, 'byte_end': end,
                       'line': number, 'sha256': hashlib.sha256(raw).hexdigest()})
        offset = end
    assert offset == len(text.encode('utf-8'))
    return result


def verify_spans(text, records):
    raw = text.encode('utf-8')
    previous = 0
    for record in records:
        start, end = record['byte_start'], record['byte_end']
        if type(start) is not int or type(end) is not int or start != previous or not start < end <= len(raw):
            raise ValueError('source spans must partition all original bytes in order')
        part = raw[start:end]
        if part.decode('utf-8') != record['text'] or hashlib.sha256(part).hexdigest() != record['sha256']:
            raise ValueError('source span is not byte-exact')
        previous = end
    if previous != len(raw):
        raise ValueError('uncovered source bytes')


def template_hash(text):
    # Conservative extra grouping, never a proof that distinct hashes are unrelated.
    # Overgrouping costs data efficiency; undergrouping could leak templates.
    text, _ = redact_text(text)
    text = re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]', '', text)
    text = re.sub(r'\b[0-9a-fA-F]{8,}\b', '<HEX>', text)
    text = re.sub(r'\d+(?:\.\d+)*', '<NUMBER>', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return digest(text)


def grouped_splits(sources, *, seed, frozen=None):
    """Union every declared relation before splitting; frozen assignments fail closed.

    sources: id, text_hash, template_hash, relation_keys, optional requested_split.
    A relation key should cover the whole repository/session/template lineage.
    This does not discover semantic near-duplicates; their review remains required.
    """
    sources = list(sources)
    by_id = {}
    parent = {}
    for source in sources:
        sid = source['id']
        if sid in by_id:
            raise ValueError('duplicate source ID')
        if not source.get('relation_keys') or any(not isinstance(x, str) or not x for x in source['relation_keys']):
            raise ValueError('explicit source lineage is required')
        if source.get('requested_split') not in (None, 'train', 'development', 'final', 'diagnostic'):
            raise ValueError('invalid requested split')
        by_id[sid] = source
        parent[sid] = sid

    def root(sid):
        while parent[sid] != sid:
            parent[sid] = parent[parent[sid]]
            sid = parent[sid]
        return sid

    owners = {}
    for sid, source in by_id.items():
        keys = [('content', source['text_hash']), ('template', source['template_hash'])]
        keys += [('lineage', k) for k in source['relation_keys']]
        for key in keys:
            if key in owners:
                a, b = root(sid), root(owners[key])
                parent[max(a,b)] = min(a,b)
            else:
                owners[key] = sid
    groups = {}
    for sid in by_id:
        groups.setdefault(root(sid), []).append(sid)
    frozen = frozen or {}
    if set(frozen) - set(by_id):
        raise ValueError('frozen registry sources cannot be dropped when extending it')
    result = {}
    for members in groups.values():
        members.sort()
        fixed = {frozen[s]['split'] for s in members if s in frozen}
        fixed |= {by_id[s]['requested_split'] for s in members if by_id[s].get('requested_split')}
        if len(fixed) > 1:
            raise ValueError('related sources cross frozen/requested splits')
        group = digest({'version': GROUPING_VERSION, 'members': members})
        split = next(iter(fixed)) if fixed else ('development' if int(digest([seed, group])[:8],16) % 5 == 0 else 'train')
        # A known old source may never be promoted into a new final test.
        if split == 'final' and any(by_id[s].get('previously_exposed', True) for s in members):
            raise ValueError('final test requires independently verified fresh sources')
        for sid in members:
            stable_groups = sorted({frozen[s]['group_key'] for s in members if s in frozen and 'group_key' in frozen[s]})
            assigned_group = frozen.get(sid, {}).get('group_key', stable_groups[0] if stable_groups else group)
            result[sid] = {'split': split, 'group_key': assigned_group, 'component_sha256': group}
    return result


def target_examples(source, *, max_targets=32, max_chars=40000):
    """Bound targets without dropping context. Oversized full outputs are held.

    A future validated hierarchical context policy is needed for larger outputs;
    this function deliberately does not hide their tail or relabel pooled blocks.
    """
    from contracts import SCHEMA
    if source.get('capture_complete') is not True:
        raise ValueError('complete source capture has not been verified')
    if source.get('privacy_review') not in ('public-redacted', 'synthetic'):
        raise ValueError('source privacy has not been admitted')
    if type(max_targets) is not int or not 1 <= max_targets <= 48:
        raise ValueError('invalid target batch size')
    records = source_spans(source['text'], source['id'])
    examples = []
    for start in range(0, len(records), max_targets):
        targets = records[start:start+max_targets]
        ids = {r['id'] for r in targets}
        example = {
            'schema': SCHEMA, 'id': digest([source['id'], source['task'], [r['id'] for r in targets]]),
            'source_id': source['id'], 'family': source['family'], 'domain': 'log',
            'split': source['split'], 'group_key': source['group_key'], 'task': source['task'],
            'privacy_review': source['privacy_review'], 'records': targets,
            'context': [r for r in records if r['id'] not in ids],
            'source_sha256': hashlib.sha256(source['text'].encode()).hexdigest(),
        }
        from contracts import prompt_for
        if len(prompt_for([example])) > max_chars:
            raise ValueError('full context exceeds teacher budget; source held without truncation')
        examples.append(example)
    return examples
