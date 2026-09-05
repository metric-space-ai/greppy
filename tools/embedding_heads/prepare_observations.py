"""Prepare source-pinned observation examples from an explicitly authored manifest."""
import argparse
import hashlib
import json
from pathlib import Path
from contracts import canonical, digest, strict_json
from web_records import observation_examples


def prepare(manifest_path, output):
    raw_manifest=manifest_path.read_bytes()
    manifest=strict_json(raw_manifest)
    if manifest.get('schema')!='greppy.heads.web-source-manifest.v1':
        raise ValueError('unsupported observation manifest')
    examples=[]; source_receipts=[]
    for source in manifest['observations']:
        path=Path(source['path']); raw=path.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=source['sha256']:
            raise ValueError('source checksum mismatch')
        envelope=strict_json(raw)
        items=observation_examples(envelope, source_id=manifest['episode_id'],
                goal=manifest['goal'],goal_version=manifest['goal_version'],
                last_action=source['last_action'],group_key=manifest['group_key'],
                split=manifest['split'],privacy_review=manifest['privacy_review'])
        examples.extend(items)
        source_receipts.append({'path':str(path),'sha256':source['sha256'],
                                'snapshot_sha256':digest(envelope),
                                'examples':[x['id'] for x in items]})
    if len({x['id'] for x in examples})!=len(examples):
        raise ValueError('duplicate observation examples')
    output.mkdir(parents=True,exist_ok=False)
    with (output/'examples.jsonl').open('x') as f:
        for x in examples: f.write(canonical(x)+'\n')
    report={'schema':'greppy.heads.web-preparation.v1',
            'source_manifest_sha256':hashlib.sha256(raw_manifest).hexdigest(),
            'example_count':len(examples),'candidate_count':sum(len(x['records']) for x in examples),
            'episode_count':1,'split':manifest['split'],'sources':source_receipts,
            'limitation':'Recorded observations; preparation is not independent outcome or production acceptance.'}
    with (output/'manifest.json').open('x') as f: f.write(canonical(report)+'\n')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest',type=Path); parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args(); report=prepare(args.manifest,args.out)
    print(canonical({k:report[k] for k in ('example_count','candidate_count','episode_count','split')}))


if __name__=='__main__': main()
