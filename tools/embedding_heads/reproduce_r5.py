"""Read-only historical R5 reproduction on native cached CUDA Q4_K vectors.

Does not train, calibrate, or open a new holdout. Outputs are created exclusively.
Requires numpy and torch; the checkpoint hash is checked before deserialization.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import struct
import time

from audit_metrics import summarize

PINS = {
    'head1-classifier.pt': '6ac254d427e077ce9a67bde5888bc88a8eba39e9629413f58537973f9e92803b',
    'head2-ranker.pt': '998462f914cabed24f2fcc86cab1dff4d24e931bcf41280b135e6cb62e7cbff4',
    'frozen-thresholds.json': 'f154cde4f2958dd578ff859b13963b9ed026ade6c3b899fc2b21205bfe5aae0a',
    'classifier-v1.f32le': '523c23339149d0cae8f30d15c422d10d82b7be5fcc9935aba3bcc805790a6a1c',
    'holdout-fresh.blocks.jsonl': 'ddd8ac6d106e258e947316069f386b23b0761a38b07bb304c2ddcf966e336e1b',
    'holdout-fresh.vectors.f32.npy': '7f41b8f5991b51d79ac8291072dadb8d45dcffb41b0f42dd11571c02aa3f9533',
}
LABELS = ['error','warning','progress','text']


def sha(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def binary_metric(truth, prediction):
    tp = int((truth & prediction).sum())
    fp = int((~truth & prediction).sum())
    fn = int((truth & ~prediction).sum())
    tn = int((~truth & ~prediction).sum())
    return {'tp':tp,'fp':fp,'fn':fn,'tn':tn,
            'precision': tp/(tp+fp) if tp+fp else None,
            'recall': tp/(tp+fn) if tp+fn else None,
            'f1': 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None}


def main():
    import numpy as np
    import torch
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--r5', type=Path, required=True)
    parser.add_argument('--asset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    r5 = args.r5
    inputs = {name:r5/name for name in PINS if name != 'classifier-v1.f32le'}
    inputs['classifier-v1.f32le'] = args.asset
    inputs.update({name:r5/name for name in ('holdout-fresh.blocks.jsonl','holdout-fresh.vectors.f32.npy','holdout-once-results.json','holdout-audit/corrected-metrics.json')})
    hashes = {name:sha(path) for name,path in inputs.items()}
    for name,pin in PINS.items():
        if hashes[name] != pin:
            raise ValueError(f'Hash mismatch: {name}')
    # These historical checkpoints contain NumPy scaler arrays. Only the pinned,
    # explicitly trusted artifact is eligible for the legacy pickle loader.
    ck = torch.load(inputs['head1-classifier.pt'], map_location='cpu', weights_only=False)
    if ck['labels'] != LABELS:
        raise ValueError('Unexpected class order')
    torch.set_num_threads(4)
    model = torch.nn.Sequential(torch.nn.Linear(768,256),torch.nn.GELU(),torch.nn.Linear(256,4)).eval()
    model.load_state_dict(ck['model_state_dict'])
    vectors = np.load(inputs['holdout-fresh.vectors.f32.npy'], mmap_mode='r', allow_pickle=False)
    rows = [json.loads(line) for line in open(inputs['holdout-fresh.blocks.jsonl'])]
    if vectors.shape != (len(rows),768) or not np.isfinite(vectors).all():
        raise ValueError('Invalid vector dimensions or values')
    mean, scale = np.asarray(ck['scaler']['mean']), np.asarray(ck['scaler']['scale'])
    started = time.perf_counter()
    with torch.no_grad():
        x = ((np.asarray(vectors)-mean)/scale).astype(np.float32)
        logits = model(torch.from_numpy(x))
        probs = logits.softmax(1).numpy()
    head_ms = (time.perf_counter()-started)*1000
    frozen = json.load(open(inputs['frozen-thresholds.json']))
    y = np.array([LABELS.index(row['label']) for row in rows])
    thresholds = frozen['classifier_decode']
    error = binary_metric(y==0, probs[:,0]>=thresholds['error_threshold'])
    warning = binary_metric(y==1, probs[:,1]>=thresholds['warning_threshold'])
    old = json.load(open(inputs['holdout-once-results.json']))
    for name, result in [('error',error),('warning',warning)]:
        expected = old[name]
        if (result['tp'], result['tp']+result['fp'], result['tp']+result['fn']) != (expected['tp'],expected['predicted'],expected['true']):
            raise AssertionError(f'Historical counts differ for {name}: {result}')
    pred = probs.argmax(1)
    confusion = np.zeros((4,4),dtype=np.int64)
    np.add.at(confusion,(y,pred),1)
    per_class = {name:binary_metric(y==i,pred==i) for i,name in enumerate(LABELS)}
    # Compare exported float32 parameters against the original float64 scaler.
    raw = args.asset.read_bytes()
    if raw[:8] != b'GRPYR5H1' or len(raw) != 797840:
        raise ValueError('Invalid pinned export')
    values = np.frombuffer(raw,dtype='<f4',offset=128).copy()
    cursor = 0
    arrays = []
    for shape in [(768,),(768,),(256,768),(256,),(4,256),(4,)]:
        size = int(np.prod(shape)); arrays.append(torch.from_numpy(values[cursor:cursor+size].reshape(shape))); cursor += size
    mu, sigma, w1, b1, w2, b2 = arrays
    with torch.no_grad():
        h = torch.nn.functional.gelu(torch.nn.functional.linear((torch.from_numpy(np.array(vectors))-mu)/sigma,w1,b1))
        exported = torch.nn.functional.linear(h,w2,b2).softmax(1).numpy()
    audit = json.load(open(inputs['holdout-audit/corrected-metrics.json']))
    original = audit['holdout']['raw_m3_metrics']
    judgments = audit['judgments']
    audited_errors = {'TP':judgments['TP_confirmed'],'FP':judgments['FP_actually_correct'],'FN':judgments['FN_judged']-judgments['FN_actually_wrong'],'TN':0}
    strata = {s:{'population':original[s.lower()],'judged':judgments.get(s+'_judged',0),'audited_errors':audited_errors[s]} for s in ('TP','FP','FN','TN')}
    report = {
        'schema':'greppy.heads.r5-reproduction.v1',
        'created_utc':datetime.now(timezone.utc).isoformat(),
        'evaluation_role':'historical_diagnostic_only',
        'labels':LABELS,'blocks':len(rows),'outputs':len({r['output_id'] for r in rows}),
        'reference_label_counts':dict(Counter(r['label'] for r in rows)),
        'missing_reference_classes':[name for name in LABELS if name not in {r['label'] for r in rows}],
        'threshold_metrics':{'error':error,'warning':warning},
        'argmax_confusion_rows_truth_columns_prediction':confusion.tolist(),
        'argmax_per_class':per_class,
        'historical_raw_counts_reproduced':True,
        'export_parameter_parity':{'max_probability_difference':float(np.abs(exported-probs).max()),'error_decision_disagreements':int(((exported[:,0]>=thresholds['error_threshold'])!=(probs[:,0]>=thresholds['error_threshold'])).sum()),'scope':'PyTorch arithmetic on same native cached vectors; does not establish current native encoder or Rust parity'},
        'historic_audit_coverage':summarize({'strata':strata}),
        'head_only_cpu_ms':head_ms,
        'runtime':{'python':platform.python_version(),'numpy':np.__version__,'torch':torch.__version__},
        'inputs':{name:{'path':str(inputs[name]),'sha256':digest} for name,digest in hashes.items()},
        'script_sha256':sha(__file__),
        'release_gate':'not_evaluated',
    }
    args.output.mkdir(parents=True,exist_ok=False)
    with open(args.output/'report.json','x') as stream:
        json.dump(report,stream,indent=2,allow_nan=False)
    # Private local audit source, not a training set. No source text is exported.
    with open(args.output/'predictions.jsonl','x') as stream:
        for row,p in zip(rows,probs):
            json.dump({'id':row['id'],'output_id':row['output_id'],'reference_label':row['label'],'probabilities':[float(v) for v in p]},stream); stream.write('\n')
    print(json.dumps({k:report[k] for k in ('blocks','reference_label_counts','threshold_metrics','export_parameter_parity','historic_audit_coverage')},indent=2))


if __name__ == '__main__':
    main()
