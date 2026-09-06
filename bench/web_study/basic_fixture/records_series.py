"""Freeze isolated development trials and collect independent, durable evidence."""
import argparse,datetime,hashlib,json,os,shlex,sqlite3,subprocess,sys
from pathlib import Path
from prepare_context import prepare
from records_fixture import initial,TASK,oracle
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from export_codex_trace import export_turn

def save(path,value):
    with path.open('x') as f: json.dump(value,f,indent=2)

def freeze(folder,scratch,cli,runtime,port):
    folder.mkdir(parents=True,exist_ok=False)
    trials=[]
    for repeat,order in enumerate(('ACE','CEA','EAC','AEC','CAE'),1):
        for arm in order:
            ident=f'r{repeat}-{arm.lower()}'
            state=scratch/'states'/f'{ident}.json'
            save(state,initial(ident,repeat))
            trial={'position':len(trials)+1,'id':ident,'arm':arm,'repeat':repeat,'url':f'http://127.0.0.1:{port}/trial/{ident}','state':str(state)}
            message=f"Complete this browser task on {trial['url']}\n\n{TASK}\n\nWork only through the browser UI and documented browser APIs. Do not read fixture source, host state files or application APIs. Discover the page yourself. Batch actions when the next targets are known. Verify the visible result. This is a synthetic local test: confirmation creates only a disposable test record, no real order, payment or contract. Do not message other tasks or write reports. Finish with one short factual sentence.\n\n"
            if arm=='A':
                message+='Use only the standard Codex in-app Browser through mcp__node_repl__js. This study explicitly authorizes the standard browser despite repository defaults. Read and follow /Users/michaelwelsch/.codex/plugins/cache/openai-bundled/browser/26.901.22334/skills/control-in-app-browser/SKILL.md (shell allowed solely to read the skill). Initialize its browser-client runtime, select agent.browsers.get("iab"), emit its complete documentation, and create your own background tab. Use documented AX or Playwright APIs. Do not use Greppy Web, application APIs or another browser. Close your own tab after verification.'
            else:
                context=prepare(scratch/'contexts',cli,scratch/'aliases',ident,ident,runtime=runtime)
                trial['context']=context
                command=context['alias']
                if arm=='E':
                    wrapper=scratch/'aliases'/f'gwe-{ident}'
                    prototype=Path(__file__).with_name('record_tool.py').resolve()
                    wrapper.write_text('#!/bin/sh\nexport PYTHONDONTWRITEBYTECODE=1\nexec python3 '+shlex.quote(str(prototype))+' --delegate '+shlex.quote(command)+' "$@"\n')
                    wrapper.chmod(0o700);command=str(wrapper)
                trial['command']=command
                message+=f'Use only Greppy Web through {command}; it selects your isolated context. Command shape: {command} web COMMAND ARGUMENTS. First create your session with `web session create --profile project --json`, then open the task URL with `web open URL`. Forward the complete exec_command result with text(await tools.exec_command(...)). Poll any running session_id using write_stdin until terminal; empty output is not completion. Use returned page state for decisions and --help when needed. `web do ACTION ... :: ACTION ...` chains known steps; it retains completed effects if a later step fails. Do not replay completed mutations. Use native browser actions; no other browser or application API. Stop your own runtime with `web runtime stop` after verification.'
                if arm=='E':
                    message+='\nAdditional experimental capability: `web records QUERY --fields data-ATTR,data-OTHER --where PREDICATES` returns native actionable refs, names, states, explicit visibility and requested data attributes filtered by native web match predicates. It is limited to the current document, uses sequential native inspections and fails explicitly on missing/truncated source evidence. Choose whether it is useful; its use is not mandatory. Other commands are unchanged.'
            trial['message']=message
            trial['message_sha256']=hashlib.sha256(message.encode()).hexdigest()
            save(folder/f'{ident}.dispatch.json',trial)
            trials.append(trial)
    sources={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in ('record_tool.py','records_fixture.py','records_series.py')}
    plan={'schema':'greppy.web.records-series.v1','frozen_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'model':'gpt-5.6-luna','effort':'medium','source_sha256':sources,'trials':trials,'scope':'one exploratory development case, five repeated seeds; E optional capability, not mandatory usage','startup':'fresh browser contexts, startup and onboarding within participant turn; C/E explicit project profile','acceptance':'paired aggregate costs, distribution and uncertainty; no every-run token gate; correctness separate; failures retained'}
    save(folder/'plan.json',plan)
    print(json.dumps({'trials':len(trials),'first_message':trials[0]['message']}))

def collect(folder,ident,thread):
    trial=json.loads((folder/f'{ident}.dispatch.json').read_text())
    db=sqlite3.connect('file:/Users/michaelwelsch/.codex/state_5.sqlite?mode=ro',uri=True)
    row=db.execute('select rollout_path from threads where id=?',(thread,)).fetchone()
    if row is None: raise ValueError('participant thread absent')
    raw=Path(row[0]); turns=[]
    for line in raw.read_text().splitlines():
        record=json.loads(line);p=record.get('payload',{})
        if p.get('type')=='task_complete': turns.append(p['turn_id'])
    if len(turns)!=1: raise ValueError(f'expected one completed participant turn, got {len(turns)}')
    paths=export_turn(raw,folder/ident,turns[0])
    metadata=json.loads(paths['metadata'].read_text())
    state=json.loads(Path(trial['state']).read_text());checks=oracle(state)
    save(folder/ident/'oracle-state.json',state)
    totals={}
    usages=[r['usage'] for r in metadata['token_usage_records']]
    if usages and all(isinstance(u,dict) for u in usages) and not metadata['token_usage_conflicts']:
        for k in ('input_tokens','output_tokens','cached_input_tokens','total_tokens'):
            if all(isinstance(u.get(k),int) for u in usages): totals[k]=sum(u[k] for u in usages)
    result={'trial':ident,'thread':thread,'turn':turns[0],'oracle':checks,'correct':all(checks.values()),'tokens':totals,'responses':len(usages),'context':metadata['turn_context'],'trace':{k:str(v) for k,v in paths.items()},'collected_at':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    save(folder/ident/'result.json',result)
    print(json.dumps(result))

if __name__=='__main__':
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='op',required=True)
    f=sub.add_parser('freeze');f.add_argument('folder',type=Path);f.add_argument('--scratch',required=True,type=Path);f.add_argument('--cli',required=True,type=Path);f.add_argument('--runtime',required=True,type=Path);f.add_argument('--port',type=int,required=True)
    c=sub.add_parser('collect');c.add_argument('folder',type=Path);c.add_argument('ident');c.add_argument('thread')
    a=vars(p.parse_args());op=a.pop('op');globals()[op](**a)
