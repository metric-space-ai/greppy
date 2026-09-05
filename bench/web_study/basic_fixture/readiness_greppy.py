"""Prepared UI smoke, not an agent trial. Raw command evidence and host oracle."""
import argparse, json, os, subprocess, time
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('cli'); p.add_argument('base_url'); p.add_argument('run_dir',type=Path)
a=p.parse_args()
out=a.run_dir/'greppy-readiness'
out.mkdir(exist_ok=False)
env={**os.environ,'GREPPY_WEB_RUNTIME_DIR':str(a.run_dir/'runtime')}
runs=json.loads((a.run_dir/'runs.json').read_text())
commands=[]
def call(label,args):
    argv=[a.cli,'web',*args,'--json']
    started=time.monotonic(); proc=subprocess.run(argv,env=env,capture_output=True,text=True,timeout=45)
    row={'label':label,'argv':argv,'cwd':os.getcwd(),'exit_code':proc.returncode,'seconds':time.monotonic()-started,'stdout':proc.stdout,'stderr':proc.stderr}
    commands.append(row); (out/'commands.json').write_text(json.dumps(commands,indent=2))
    if proc.returncode: raise RuntimeError(f'{label}: exit {proc.returncode}; inspect saved output')
    return json.loads(proc.stdout)

outcomes=[]
for case,rid in runs.items():
    try:
        created=call(case+'-session',['session','create','--profile','project'])
        session=created['result']['session_id']
        def run(label,*args): return call(case+'-'+label,[*args,'--session',session])
        run('open','open',a.base_url+'/?run_id='+rid)
        run('ready','wait','css=#task input, #task button, #task select','--timeout','10000')
        run('before','observe')
        if case=='text':
            run('fill','fill','css=#note','Ready for review')
            run('save','click','text=Save note')
            revision=1
        elif case=='checkbox':
            run('enable','check','css=#enabled')
            run('enabled','wait','css=#quantity:not(:disabled)','--timeout','5000')
            run('fill','fill','css=#quantity','3')
            run('blur','press','Tab')
            revision=2
        elif case=='address':
            run('country','select','css=#country','Germany')
            run('cities','wait','css=#city option[value="Berlin"], #city option:nth-child(2)','--timeout','5000')
            run('city','select','css=#city','Berlin')
            run('fill','fill','css=#postcode','10115')
            run('blur','press','Tab')
            revision=3
        else:
            run('dialog','click','text=Open Complete basic task dialog')
            run('save','click','css=dialog[open] button:first-of-type')
            revision=1
        run('persisted','wait','text=Revision events: '+str(revision),'--timeout','5000')
        run('after','observe')
        proc=subprocess.run(['/usr/bin/python3',str(Path(__file__).with_name('server.py')),'verify-run',rid,'--run-dir',str(a.run_dir)],capture_output=True,text=True)
        outcomes.append({'case':case,'run_id':rid,'oracle_exit_code':proc.returncode,'oracle':json.loads(proc.stdout)})
    except Exception as e:
        outcomes.append({'case':case,'run_id':rid,'error':str(e)})
    (out/'outcomes.json').write_text(json.dumps(outcomes,indent=2))
print(json.dumps(outcomes))
raise SystemExit(0 if all(x.get('oracle',{}).get('ok') for x in outcomes) else 1)
