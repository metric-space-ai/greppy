"""Prepared full-UI smoke for the record experiment; not a timed agent trial."""
import argparse,json,subprocess
from pathlib import Path
from records_fixture import oracle


def run(command,url,state_path,output):
    calls=[];result={'calls':calls,'agent_token_acceptance':False,'completed':False}
    def persist(): output.write_text(json.dumps(result,indent=2))
    def call(*args):
        argv=[command,'web',*args]
        try:
            proc=subprocess.run(argv,capture_output=True,text=True,timeout=90)
        except subprocess.TimeoutExpired as error:
            calls.append({'argv':argv,'timeout_seconds':90,'stdout':repr(error.stdout),'stderr':repr(error.stderr)})
            persist();raise
        entry={'argv':argv,'exit_code':proc.returncode,'stdout':proc.stdout,'stderr':proc.stderr}
        calls.append(entry);persist()
        assert proc.returncode==0,entry
        documents = [json.loads(line) for line in proc.stdout.splitlines() if line.strip()]
        assert documents, entry
        failures = [d for d in documents if d.get("status") != "ok" and d.get("ok") is not True]
        assert not failures, {"failed_documents": len(failures), "recorded_call": len(calls)}
        return documents[-1]
    try:
        opened=call('open',url,'--json')
        snapshot=opened['result']['page_state']['snapshot']
        refs={n['name']:n['ref'] for n in snapshot['actionables']}
        records=call('records','css=#offers','--fields','data-price,data-days,data-region','--where','visible=true disabled=false data-region=EU data-days<=3')
        assert records['status']=='ok' and records['records'],records
        selected=min(records['records'],key=lambda r:float(r['data-price']))
        assert selected['visible'] and not selected['disabled']
        print('SMOKE record query returned '+str(len(records['records']))+' eligible offers',flush=True)
        call('click',selected['ref'],'--json')
        call('do','--json','fill',refs['Delivery note'],'Leave with reception','::','check',refs['Accept delivery conditions'],'::','click',refs['Save booking'])
        observed=call('observe','--json')
        confirm=[n for n in observed['result']['actionables'] if n['name']=='Confirm booking']
        assert len(confirm)==1,observed
        call('click',confirm[0]['ref'],'--json')
        call('wait','css=#confirmation:not([open])','--json')
        call('reload','--json')
        state=json.loads(state_path.read_text());checks=oracle(state)
        assert all(checks.values()),checks
        result.update(oracle=checks,selected_record=selected,prototype_processing=records['processing'],completed=True)
        print('SMOKE complete '+json.dumps(checks))
    except BaseException as error:
        result['failure']=repr(error);raise
    finally:
        try:
            stop=call('runtime','stop','--json')
            result['runtime_stopped']=stop['result']['running'] is False
        except BaseException as error:
            result['cleanup_failure']=repr(error);result['runtime_stopped']=False
        persist()
    assert result['runtime_stopped'],result

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--command',required=True);p.add_argument('--url',required=True)
    p.add_argument('--state',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists(): raise FileExistsError('output already exists')
    run(a.command,a.url,a.state,a.output)
