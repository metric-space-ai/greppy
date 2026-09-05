"""Freeze Basic inputs and preregister balanced A/C development trials."""
import argparse, hashlib, json, secrets
from pathlib import Path
import server
from prepare_context import prepare
from dispatch import task_goal
from onboarding import CONDITIONS

p=argparse.ArgumentParser()
p.add_argument('output',type=Path)
p.add_argument('--scratch',type=Path,required=True,help='Disposable fixture/state/runtime root on /Volumes/tmp')
p.add_argument('--base-url',required=True)
p.add_argument('--repeats',type=int,default=5)
p.add_argument('--cases', nargs='+', choices=['text','checkbox','address','dialog','table'], default=['text','checkbox','address','dialog'])
p.add_argument('--isolated-cli', type=Path, help='Opt into short aliases and isolated cwd; native open handles session creation')
p.add_argument('--alias-dir', type=Path)
p.add_argument('--runtime', type=Path, help='Explicit candidate runtime; do not use a shared mutable build target')
p.add_argument('--view', choices=('default','compact'), default='default')
p.add_argument('--chain-view', choices=('default','compact'), default='default')
p.add_argument('--runtime-id')
p.add_argument('--latency-limitation')
p.add_argument('--onboarding', choices=('legacy', *CONDITIONS), default='legacy')
a=p.parse_args()
if a.repeats < 1: p.error('repeats must be positive')
if a.isolated_cli and not (a.alias_dir and a.runtime_id and a.runtime):
    p.error('--isolated-cli requires --alias-dir, --runtime-id and --runtime')
if len(set(a.cases)) != len(a.cases): p.error('cases must be unique')
a.output.mkdir(parents=True,exist_ok=False)
if not Path('/Volumes/tmp').is_mount() or not a.scratch.resolve().is_relative_to(Path('/Volumes/tmp')):
    raise ValueError('scratch must be on mounted /Volumes/tmp; no system-disk fallback')
a.scratch.mkdir(parents=True,exist_ok=False)
for name in ('fixture','runs','runtime'):
    (a.scratch/name).mkdir()
    (a.output/name).symlink_to((a.scratch/name).resolve(),target_is_directory=True)
freeze=a.output/'fixture'
source=Path(__file__).parent
pins=[]
for name in ['server.py','table_case.py','static/index.html','static/app.js','static/table.js','static/styles.css','test_server.py','test_table_case.py','README.md']:
    raw=(source/name).read_bytes(); target=freeze/name; target.parent.mkdir(exist_ok=True,parents=True); target.write_bytes(raw)
    pins.append({'path':name,'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)})
server.set_dir(a.output/'runs')
trials=[]
for case_index,case in enumerate(a.cases):
    for repeat in range(1,a.repeats+1):
        order=['A','C'] if (case_index+repeat)%2 else ['C','A']
        seed=f'basic-development-{case}-{repeat}'
        for arm in order:
            rid=secrets.token_hex(6); server.write_state(server.new_state(rid,seed,case))
            trial={'position':len(trials)+1,'case':case,'repeat':repeat,'arm':arm,'seed':seed,'run_id':rid,'url':a.base_url.rstrip('/')+'/?run_id='+rid}
            trial['task_goal'] = task_goal(trial)
            if arm == 'C' and a.isolated_cli:
                trial['cli_context']=prepare(a.scratch/'contexts',a.isolated_cli,a.alias_dir,rid,a.runtime_id,runtime=a.runtime,view=a.view,chain_view=a.chain_view)
            trials.append(trial)
manifest={'schema':'greppy.basic-series.plan.v1','purpose':'development, never held-out acceptance or training final evaluation','model':'gpt-5.6-luna','effort':'medium','repeats':a.repeats,'state':'prepared_not_run','fixture':pins,'trials':trials,'constraints':['fresh contexts, isolated runs, serial execution','C must create a fresh explicit project session per participant; session identities audited','same frozen facts within each pair','all errors/recovery and failures retained','no solution scripts or host-state access by participants','B is separate; A/C does not claim actual Greppy-agent comparison','record provider usage; missing data remains null','do not infer verified end-to-end time from participant turn duration']}
manifest['harness_condition']='short_alias_isolated_cwd_native_open' if a.isolated_cli else 'explicit_session_long_wrapper'
if a.isolated_cli:
    manifest['candidate_integrity_required']=True
    manifest['rendering']={'view':a.view,'chain_view':a.chain_view}
    manifest['constraints'][1]='C has its own empty workspace and short CLI alias; native open may create the session; actual session identities audited'
    manifest['constraints'].append('Harness change, not a product optimization; compare separately with earlier series')
if a.latency_limitation: manifest['constraints'].append(a.latency_limitation)
if a.onboarding != 'legacy': manifest['onboarding_condition'] = a.onboarding
(a.output/'plan.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps({'plan':str(a.output/'plan.json'),'trial_count':len(trials),'first':trials[0]}))
