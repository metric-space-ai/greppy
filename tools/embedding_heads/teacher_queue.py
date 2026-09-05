"""Run resumable M3/Grok annotation jobs; stdout contains status, never source text."""
import argparse
import json
import time
from pathlib import Path

from contracts import RUBRIC_VERSION, canonical, prompt_for
from queue_store import QueueStore
from teachers import ProviderFailure, grok, minimax


def work_one(store, provider):
    job = store.claim(provider)
    if job is None:
        return False
    try:
        response,usage = (minimax(job) if provider == 'minimax' else grok(job))
        store.complete(job,response,usage)
        print(canonical({'job':job['id'],'provider':provider,'status':'done','usage':usage}),flush=True)
    except ProviderFailure as error:
        store.fail(job,error.kind,retry_after=error.retry_after,diagnostic=error.diagnostic,usage=error.usage)
        print(canonical({'job':job['id'],'provider':provider,'status':error.kind,'diagnostic':error.diagnostic}),flush=True)
    except (ValueError,TypeError,KeyError) as error:
        # Validator messages are a fixed allowlist; never print rejected source/model text.
        known = {'unexpected or duplicate annotation ID','incomplete annotation coverage',
                 'invented or duplicate evidence','invalid annotation fields','invalid observable reason',
                 'Web severity must be null','invalid log severity','invalid relevance',
                 'evidence must be nonempty source IDs','ambiguous must be boolean',
                 'response must contain only annotations array','annotation IDs must be strings'}
        diagnostic = str(error) if str(error) in known else 'invalid_response_shape'
        store.fail(job,'schema',diagnostic=diagnostic,usage=locals().get('usage',{}))
        print(canonical({'job':job['id'],'provider':provider,'status':'schema','diagnostic':diagnostic}),flush=True)
    return True


def enqueue_file(store, path, providers, max_chars):
    batches=[]; current=[]
    for raw in open(path,encoding='utf-8'):
        example=json.loads(raw)
        store.register_source(example['source_id'],example['split'],example['group_key'])
        proposed=current+[example]
        # Conservative bounded serialized input size. Teacher token usage is measured
        # from responses; a fixed count of variable-length outputs is not a budget.
        if len(prompt_for(proposed))>max_chars or sum(len(x['records']) for x in proposed)>48:
            if not current:
                raise ValueError('example exceeds batch input limit; split target spans with context first')
            batches.append(current); current=[example]
            if len(prompt_for(current))>max_chars or len(example['records'])>48:
                raise ValueError('example exceeds batch input limit')
        else: current=proposed
    if current: batches.append(current)
    ids=[]
    for provider in providers:
        model='MiniMax-M3' if provider=='minimax' else 'grok-4.6'
        for batch in batches:
            ids.append(store.enqueue(provider,model,batch))
    return {'jobs':len(ids),'ids':ids,'batches_per_provider':len(batches),'rubric':RUBRIC_VERSION}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--db',type=Path,required=True)
    commands=parser.add_subparsers(dest='command',required=True)
    enqueue=commands.add_parser('enqueue'); enqueue.add_argument('input',type=Path)
    enqueue.add_argument('--provider',choices=['minimax','grok','both'],default='both')
    enqueue.add_argument('--max-input-chars',type=int,default=48000)
    worker=commands.add_parser('work'); worker.add_argument('--provider',choices=['minimax','grok'],required=True)
    worker.add_argument('--max-jobs',type=int,default=0); worker.add_argument('--watch',action='store_true')
    commands.add_parser('status')
    resume=commands.add_parser('resume-provider'); resume.add_argument('provider',choices=['minimax','grok'])
    args=parser.parse_args(); store=QueueStore(args.db)
    if args.command=='enqueue':
        providers=['minimax','grok'] if args.provider=='both' else [args.provider]
        print(canonical(enqueue_file(store,args.input,providers,args.max_input_chars)))
    elif args.command=='status': print(canonical(store.status()))
    elif args.command=='resume-provider': store.resume_provider(args.provider); print(canonical(store.status()))
    else:
        count=0
        while not args.max_jobs or count<args.max_jobs:
            if work_one(store,args.provider): count+=1
            elif args.watch: time.sleep(30)
            else: break
        print(canonical(store.status()))


if __name__=='__main__': main()
