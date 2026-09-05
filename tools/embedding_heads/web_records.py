"""Convert observed native Web records to teacher candidates without inventing state."""
from contracts import SCHEMA, canonical, digest, prompt_for


def observation_examples(envelope, *, source_id, goal, goal_version, last_action,
                         group_key, split='development', privacy_review=None,
                         max_targets=32, max_chars=48000):
    if not goal:
        return []  # No task-conditioned ranking without an explicit agent goal.
    if not isinstance(goal,str) or type(goal_version) is not int or goal_version < 1:
        raise ValueError('explicit versioned goal required')
    if not isinstance(last_action,str) or not last_action:
        raise ValueError('last action must be recorded explicitly, not inferred from page state')
    if envelope.get('schema') != 'greppy.web-runtime.v1' or not isinstance(envelope.get('result'),dict):
        raise ValueError('unsupported typed Web observation')
    if privacy_review not in ('synthetic','public-redacted'):
        raise ValueError('Web source privacy must be admitted')
    if type(max_targets) is not int or not 1 <= max_targets <= 48:
        raise ValueError('invalid target batch size')
    snapshot = digest(envelope); rows=[]
    result = envelope['result']

    def add(pointer, value, protected):
        rows.append({'id':digest([source_id,snapshot,pointer]),'text':canonical(value),
                     'protected':protected,'json_pointer':pointer,'snapshot_sha256':snapshot})

    for field, value in result.items():
        pointer='/result/'+field.replace('~','~0').replace('/','~1')
        if field in ('actionables','headings','links') and isinstance(value,list):
            for i, item in enumerate(value):
                # Identity and known/unknown state remain in the original record.
                add(pointer+'/'+str(i),item,field=='actionables')
        else:
            # Unknown fields are retained and protected, never defaulted to false.
            add(pointer,value,True)
    if not rows:
        raise ValueError('observation contains no source records')
    # Envelope status/scope/errors are evidence too. They cannot be removed by ranking.
    envelope_context = [{'id':digest([source_id,snapshot,'envelope']),
                         'text':canonical({k:v for k,v in envelope.items() if k!='result'}),
                         'protected':True}]
    examples=[]
    for start in range(0,len(rows),max_targets):
        target=rows[start:start+max_targets]; target_ids={r['id'] for r in target}
        example={'schema':SCHEMA,
                 'id':digest([source_id,snapshot,goal,goal_version,last_action,[r['id'] for r in target]]),
                 'source_id':source_id,'group_key':group_key,'domain':'web',
                 'family':'typed-web-observation','split':split,'task':goal,
                 'goal_version':goal_version,'last_action':last_action,
                 'privacy_review':privacy_review,'records':target,
                 'context':envelope_context+[r for r in rows if r['id'] not in target_ids],
                 'snapshot_sha256':snapshot}
        if len(prompt_for([example]))>max_chars:
            raise ValueError('full observation exceeds teacher budget; held without truncation')
        examples.append(example)
    return examples
