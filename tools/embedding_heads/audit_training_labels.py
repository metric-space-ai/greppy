"""Reproduce R5 majority projection and audit the later 180 rubric reversals.

Inputs are immutable. Outputs contain IDs/labels only and are diagnostic overlays,
not a claim that every original training example has been relabeled.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from reproduce_r5 import sha

LABELS = ('error','warning','progress','text')


def main():
    import numpy as np
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--traces',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    t = args.traces
    paths = {'corrections':t/'label-audit-grok/merged/corrections.jsonl',
             'reversals':t/'warning-rubric/reversals.jsonl',
             'base':t/'block-head/train.blocks.jsonl',
             'extra':t/'block-head/corpus-extra.blocks.jsonl'}
    reversals = {}
    for raw in open(paths['reversals']):
        row = json.loads(raw); key = (row['output_id'],int(row['block_idx']))
        if key in reversals or row['rubric_label'] not in LABELS:
            raise ValueError('Duplicate reversal or invalid label')
        reversals[key] = row
    by = defaultdict(list); metadata = {}; output_order = {}; output_sets = {}
    for ds in ('base','extra'):
        rows = []
        for i,raw in enumerate(open(paths[ds])):
            r = json.loads(raw)
            row = {'id':r['id'],'output_id':r['output_id'],'start':int(r['start']),'end':int(r['end']),'label':r['label']}
            rows.append(row); by[row['output_id']].append((row['start'],row['end'],ds,i))
            if ds == 'base': output_order.setdefault(row['output_id'],len(output_order))
        metadata[ds] = rows; output_sets[ds] = {r['output_id'] for r in rows}
    for spans in by.values(): spans.sort()
    old_votes = defaultdict(list); new_votes = defaultdict(list)
    adoption = Counter(); seen = set(); mapped_reversals = []; unmapped_reversals = []; mapped = 0
    for raw in open(paths['corrections']):
        c = json.loads(raw); key = (c['output_id'],int(c['block_idx']))
        label = c['grok_label']; updated = label
        if key in reversals:
            if key in seen: raise ValueError('Reversal maps to duplicate judgments')
            seen.add(key); rev = reversals[key]
            if label == rev['applied_label']: adoption['still_old_label'] += 1
            elif label == rev['rubric_label']: adoption['already_final_rubric'] += 1
            else: raise ValueError(f'Unexpected intervening label at {key}')
            updated = rev['rubric_label']
        found = None
        for start,end,ds,i in by.get(c['output_id'],[]):
            if start <= key[1] < end:
                found = (ds,i); break
        if found is None:
            if key in reversals: unmapped_reversals.append(key)
            continue
        mapped += 1; old_votes[found].append(label); new_votes[found].append(updated)
        if key in reversals: mapped_reversals.append(key)
    if set(reversals) != seen: raise ValueError('Some reversals have no source judgment')
    def winner(votes):
        counts = Counter(votes)
        return sorted(counts,key=lambda x:(-counts[x],x))[0]
    rng = np.random.default_rng(20260803)
    ncal = round(len(output_order)*.2)
    calibration = set(rng.permutation(len(output_order))[:ncal].tolist())
    cal_outputs = {oid for oid,index in output_order.items() if index in calibration}
    overlay = []; directions = Counter(); splits = Counter(); conflict_counts = Counter()
    for (ds,i),votes in old_votes.items():
        row = metadata[ds][i]; before = winner(votes); after = winner(new_votes[(ds,i)])
        if len(set(votes))>1: conflict_counts['old_majority_conflicts'] += 1
        if len(set(new_votes[(ds,i)]))>1: conflict_counts['rubric_majority_conflicts'] += 1
        split = 'calibration' if ds=='base' and row['output_id'] in cal_outputs else 'training'
        if before != after: directions[before+'->'+after] += 1; splits[split] += 1
        overlay.append({'dataset':ds,'index':i,'id':row['id'],'output_id':row['output_id'],'source_label':row['label'],'r5_projected_label':before,'rubric_projected_label':after,'split':split,'rubric_votes':dict(Counter(new_votes[(ds,i)]))})
    source_specs = {
        'extra':paths['extra'],
        'loghub':t/'warn-corpus/prepared/loghub.blocks.jsonl',
        'pkgforge-cargo':t/'warn-corpus/prepared/pkgforge-cargo.blocks.jsonl',
        'pkgforge-go':t/'warn-corpus/prepared/pkgforge-go.blocks.jsonl',
        'implicit':t/'diversity-5/implicit.blocks.jsonl',
        'terminal':t/'diversity-5/terminal.blocks.jsonl',
    }
    overlaps = {}
    for name,path in source_specs.items():
        ids = {json.loads(raw)['output_id'] for raw in open(path)}
        overlaps[name] = len(ids & cal_outputs)
    report = {'schema':'greppy.heads.rubric-projection-audit.v1','reversals':len(reversals),'adoption':dict(adoption),'mapped_judgments':mapped,'mapped_reversals':len(mapped_reversals),'unmapped_reversals':len(unmapped_reversals),'represented_blocks':len(overlay),'changed_pooled_block_labels':sum(directions.values()),'change_directions':dict(directions),'changed_by_split':dict(splits),'projection_conflicts':dict(conflict_counts),'calibration_outputs':len(cal_outputs),'calibration_output_id_overlap_with_fitting_sources':overlaps,'inputs':{name:{'path':str(path),'sha256':sha(path)} for name,path in {**paths,**source_specs}.items()},'script_sha256':sha(__file__),'limitations':['Reproduces original majority vote with alphabetical ties; does not establish that aggregation is semantically correct.','ID disjointness does not establish source, template, or content disjointness.','Only the 180 adjudicated reversals change; other labels remain unaudited.']}
    args.output.mkdir(parents=True,exist_ok=False)
    with open(args.output/'report.json','x') as stream: json.dump(report,stream,indent=2)
    with open(args.output/'projection.jsonl','x') as stream:
        for row in overlay: stream.write(json.dumps(row)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='inputs'},indent=2))


if __name__=='__main__':
    main()
