"""Record a completed Basic participant using host traces and frozen oracle."""
import argparse, datetime as dt, hashlib, json, re, subprocess, sys, time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from export_codex_trace import export_turn
from summarize_series import token_totals

p=argparse.ArgumentParser()
p.add_argument('series',type=Path);p.add_argument('position',type=int)
p.add_argument('--agent-path',required=True);p.add_argument('--session-dir',type=Path,required=True)
a=p.parse_args()
plan=json.loads((a.series/'plan.json').read_text())
trial=next(t for t in plan['trials'] if t['position']==a.position)
out=a.series/'trials'/f"{a.position:02d}-{trial['case']}-{trial['repeat']}-{trial['arm']}"
if out.exists():raise ValueError('Trial output already exists; never overwrite evidence')
# Verify and snapshot the oracle state immediately, before expensive trace export.
started=time.perf_counter()
raw=(a.series/'runs'/(trial['run_id']+'.json')).read_bytes()
proc=subprocess.run(['/usr/bin/python3',str(a.series/'fixture/server.py'),'verify-run',trial['run_id'],'--run-dir',str(a.series/'runs')],capture_output=True,text=True,check=False)
verifier_seconds=time.perf_counter()-started
verified_at=dt.datetime.now(dt.timezone.utc)
oracle=json.loads(proc.stdout)
found=subprocess.run(['greppy','rg','-l','--fixed-strings','"agent_path":'+json.dumps(a.agent_path),str(a.session_dir)],capture_output=True,text=True,check=True)
paths=found.stdout.splitlines()
if len(paths)!=1:raise ValueError(f'Expected one participant rollout, found {len(paths)}')
source=Path(paths[0]); records=[json.loads(x) for x in source.read_bytes().splitlines()]
contexts=[r['payload'] for r in records if r.get('type')=='turn_context']
if len(contexts)!=1:raise ValueError('Require exactly one participant turn')
turn_id=contexts[0]['turn_id']
if not any(r.get('type')=='event_msg' and r.get('payload',{}).get('type')=='task_complete' and r['payload'].get('turn_id')==turn_id for r in records):raise ValueError('Participant turn is not complete yet')
exported=export_turn(source,out,turn_id)
metadata=json.loads(exported['metadata'].read_text())
session_ids=sorted(set(re.findall(r'\bwrs_[0-9a-f]+\b',json.dumps(metadata['tool_calls']))))
prior_ids=set()
for previous in (a.series/'trials').glob('*/trial.json'):
    prior_ids.update(json.loads(previous.read_text()).get('observed_session_ids',[]))
session_isolation={'observed':bool(session_ids),'reused_ids':sorted(set(session_ids)&prior_ids),
                   'fresh_vs_prior':bool(session_ids) and not set(session_ids)&prior_ids if trial['arm']=='C' else None,
                   'limit':'Trace ID uniqueness; not a proof of complete storage/profile isolation.'}
tokens,limits=token_totals(metadata)
bounds=metadata['time_boundaries']
duration=None
if bounds['started'] and bounds['ended']:
    duration=(dt.datetime.fromisoformat(bounds['ended']['value'].replace('Z','+00:00'))-dt.datetime.fromisoformat(bounds['started']['value'].replace('Z','+00:00'))).total_seconds()
(out/'verified-state.json').write_bytes(raw)
result={'schema':'greppy.web-study.basic.v1',**trial,'agent_path':a.agent_path,'turn_id':turn_id,'context':metadata['turn_context'],'oracle':oracle,'oracle_exit_code':proc.returncode,'verified_at':verified_at.isoformat(),'verifier_seconds':verifier_seconds,'agent_turn_wall_seconds':duration,'end_to_end_verified_seconds':None,'latency_limit':'Independent post-hoc verification. No controlled completion-to-oracle dispatch latency; no main study end-to-end claim.','tokens':tokens,'telemetry_limits':limits,'host_tool_envelopes':metadata['tool_response_status'],'completion':metadata['completion_boundary'],'artifacts':{k:str(v) for k,v in exported.items()},'verified_state_sha256':hashlib.sha256(raw).hexdigest(),'plan_sha256':hashlib.sha256((a.series/'plan.json').read_bytes()).hexdigest()}
result.update(observed_session_ids=session_ids,session_isolation=session_isolation)
(out/'trial.json').write_text(json.dumps(result,indent=2))
print(json.dumps({'trial':str(out),'oracle':oracle,'context':metadata['turn_context'],'tokens':tokens,'agent_turn_seconds':duration,'tool_calls':metadata['tool_response_status']['request_count']}))
