"""Audit public coordinator dispatch records and frozen executable/source bytes."""
import argparse,hashlib,json,sys
from pathlib import Path
from candidate import capture

def audit(folder,lines):
    plan=json.loads((folder/'plan.json').read_text());observed=[]
    expected={t['id']:t for t in plan['trials']}
    for line in lines:
        r=json.loads(line);p=r.get('payload',{})
        if p.get('type') not in ('function_call','custom_tool_call') or not str(p.get('name','')).endswith('spawn_agent'):continue
        args=p.get('arguments',p.get('input',{}))
        if isinstance(args,str):args=json.loads(args)
        name=args.get('task_name','')
        if not name.startswith('records_'):continue
        ident=name.removeprefix('records_').replace('_','-')
        if ident not in expected: raise ValueError('unplanned records participant')
        trial=expected[ident]
        observed.append({'trial':ident,'timestamp':r.get('timestamp'),'stored_message_sha256':hashlib.sha256(args['message'].encode()).hexdigest(),'message_matches':(None if args['message'].startswith('gAAAA') else hashlib.sha256(args['message'].encode()).hexdigest()==trial['message_sha256']),'message_check':('opaque_host_value' if args['message'].startswith('gAAAA') else 'plaintext_comparison'),'model_matches':args.get('model')==plan['model'],'effort_matches':args.get('reasoning_effort')==plan['effort'],'fresh_context':args.get('fork_turns')=='none'})
    sources={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()==value for name,value in plan['source_sha256'].items()}
    context=next(t['context'] for t in plan['trials'] if 'context' in t)
    actual=capture(Path(context['candidate']['cli']['path']),Path(context['candidate']['runtime']['path']))
    binary_matches={kind:actual[kind]['sha256']==context['candidate'][kind]['sha256'] for kind in ('cli','runtime')}
    candidate_consistency=all(t['context']['candidate']==context['candidate'] for t in plan['trials'] if 'context' in t)
    order=[r['trial'] for r in observed]==[t['id'] for t in plan['trials']]
    return {'schema':'greppy.web.records-dispatch-audit.v1','plan_sha256':hashlib.sha256((folder/'plan.json').read_bytes()).hexdigest(),'observed_dispatches':observed,'frozen_order_matches':order,'message_checks_unavailable':sum(r['message_matches'] is None for r in observed),'non_message_checks_ok':order and all(sources.values()) and all(binary_matches.values()) and candidate_consistency and all(all(r[k] for k in ('model_matches','effort_matches','fresh_context')) for r in observed),'sources_unchanged':sources,'binaries_unchanged':binary_matches,'all_candidates_same':candidate_consistency,'ok':order and all(sources.values()) and all(binary_matches.values()) and candidate_consistency and all(all(r[k] for k in ('message_matches','model_matches','effort_matches','fresh_context')) for r in observed),'scope':'Public coordinator spawn requests and executable/source bytes; not dynamic libraries, platform scheduling or proof of complete browser storage isolation. Operator stop and cleanup interventions are reported separately.'}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=audit(a.folder,sys.stdin)
    with a.output.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k!='observed_dispatches'}))
    sys.exit(0 if result['ok'] else 1)
