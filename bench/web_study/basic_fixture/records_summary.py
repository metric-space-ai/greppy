"""Paired exploratory statistics; no per-run token superiority requirement."""
import argparse,datetime,hashlib,json,math,random,statistics,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from summarize_series import token_totals

def compare(pairs):
    baseline=[a for a,b in pairs]; variant=[b for a,b in pairs]
    if not pairs or any(not isinstance(x,(int,float)) or x<=0 for x in baseline+variant):
        return {'available':False}
    changes=[100*(b/a-1) for a,b in pairs]
    logs=[math.log(b/a) for a,b in pairs]
    rng=random.Random(20260906)
    samples=sorted(statistics.mean(rng.choices(logs,k=len(logs))) for _ in range(10000))
    return {'available':True,'pairs':len(pairs),'aggregate_change_percent':100*(sum(variant)/sum(baseline)-1),'median_paired_change_percent':statistics.median(changes),'paired_changes_percent':changes,'lower_cost_pairs':sum(b<a for a,b in pairs),'mean_log_ratio_bootstrap_95_percent':([100*math.expm1(samples[249]),100*math.expm1(samples[9749])] if len(pairs)>=5 else None),'interval_caveat':'exploratory paired percentile bootstrap, five seeds of one case, not population coverage or a general-superiority test'}

def summarize(folder):
    plan=json.loads((folder/'plan.json').read_text());rows=[]
    for trial in plan['trials']:
        path=folder/trial['id']/'result.json'
        if not path.exists():
            rows.append({'trial':trial['id'],'arm':trial['arm'],'repeat':trial['repeat'],'missing':True});continue
        result=json.loads(path.read_text());paths=result['trace']
        metadata=json.loads(Path(paths['metadata']).read_text());manifest=json.loads(Path(paths['manifest']).read_text())
        raw=Path(paths['trace']).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=manifest['sha256']: raise ValueError('trace hash mismatch')
        tokens,limits=token_totals(metadata)
        cumulative=metadata.get('cumulative_turn_token_usage')
        if cumulative:
            for k in ('input_tokens','output_tokens','cached_input_tokens'):
                if tokens[k] is not None and cumulative['value'].get(k)!=tokens[k]: limits.append('cumulative mismatch: '+k)
        bounds=metadata['time_boundaries'];duration=None
        if bounds['started'] and bounds['ended']:
            duration=(datetime.datetime.fromisoformat(bounds['ended']['value'].replace('Z','+00:00'))-datetime.datetime.fromisoformat(bounds['started']['value'].replace('Z','+00:00'))).total_seconds()
        requests=[c for c in metadata['tool_calls'] if c['kind']=='request']
        rows.append({'trial':trial['id'],'arm':trial['arm'],'repeat':trial['repeat'],'correct':result['correct'],'operator_stop':(folder/f"{trial['id']}.operator-stop.json").exists(),'oracle':result['oracle'],'tokens':tokens,'telemetry_limits':limits,'context':metadata['turn_context'],'responses':result['responses'],'host_tool_calls':len(requests),'agent_turn_seconds':duration,'record_command_requests':sum(' web records ' in str(c.get('arguments','')) for c in requests),'unmatched_calls':metadata['tool_response_status']['unmatched_request_call_ids'],'completed':metadata['completion_boundary']['task_complete_present'],'trace_sha256':manifest['sha256']})
    comparisons={}
    for a,b in (('A','C'),('C','E'),('A','E')):
        pairs=[]
        for repeat in range(1,6):
            first=next(r for r in rows if r['arm']==a and r['repeat']==repeat)
            second=next(r for r in rows if r['arm']==b and r['repeat']==repeat)
            if first.get('missing') or second.get('missing'):continue
            pairs.append((first,second))
        comparisons[f'{b}_versus_{a}']={'all_paired_tasks_correct':bool(pairs) and all(x['correct'] and y['correct'] and not x['operator_stop'] and not y['operator_stop'] for x,y in pairs),'metrics':{k:compare([(x['tokens'].get(k),y['tokens'].get(k)) for x,y in pairs]) for k in ('input_tokens','output_tokens','uncached_input_tokens')},'exploratory_turn_time':compare([(x['agent_turn_seconds'],y['agent_turn_seconds']) for x,y in pairs])}
    return {'schema':'greppy.web.records-summary.v1','rows':rows,'comparisons':comparisons,'missing_trials':[r['trial'] for r in rows if r.get('missing')],'acceptance':'No every-run gate. Failed tasks stay present; their costs do not demonstrate successful efficiency. Optional E uptake must be reviewed before attributing differences to records. No p95 or full release acceptance.'}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=summarize(a.folder)
    with a.output.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps({'missing_trials':result['missing_trials'],'comparisons':result['comparisons']}))
