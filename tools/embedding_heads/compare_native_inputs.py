"""Compare hash-bound raw-text native probes; report drift without granting release."""
import argparse
import math
from pathlib import Path
from contracts import canonical, strict_json
from native_input_probe import file_sha


def compare(left, right):
    left, right = Path(left), Path(right)
    reports = [strict_json((p/'report.json').read_text()) for p in (left, right)]
    for report in reports:
        if report['schema'] != 'greppy.heads.native-input-probe.v1' or report['production_eligible'] is not False:
            raise ValueError('not an experimental native input probe')
    for key in ('input_sha256', 'binary_sha256', 'tokenizer_sha256', 'model_sha256', 'probe_sha256'):
        if reports[0][key] != reports[1][key]:
            raise ValueError('probe provenance differs: '+key)
    batches = []
    for folder, report in zip((left, right), reports):
        path = folder/'native.jsonl'
        if file_sha(path) != report['native']['artifact_sha256']:
            raise ValueError('native artifact checksum mismatch')
        batches.append([strict_json(line) for line in path.read_text().splitlines()])
    if not batches[0] or len(batches[0]) != len(batches[1]):
        raise ValueError('empty or different native row counts')
    comparisons = []
    ids = set()
    for a, b in zip(*batches):
        if a['input'] != b['input'] or a['input']['id'] in ids:
            raise ValueError('different or duplicate prepared input')
        ids.add(a['input']['id'])
        for row, report in zip((a,b), reports):
            if (row['schema'] != 'greppy.heads.native-feature.v1'
                    or row['production_eligible'] is not False
                    or row['backend'] != report['native']['mode']
                    or any(row[k] != report[k] for k in ('binary_sha256','tokenizer_sha256','model_sha256'))):
                raise ValueError('native row provenance differs')
        x, y = a['vector'], b['vector']
        if len(x) != 768 or len(y) != 768 or any(not math.isfinite(v) for v in x+y):
            raise ValueError('invalid native vector')
        nx, ny = sum(v*v for v in x), sum(v*v for v in y)
        if not nx or not ny:
            raise ValueError('zero native vector')
        comparisons.append({'input_id': a['input']['id'], 'head': a['input']['head'],
                            'max_absolute_difference': max(abs(v-w) for v,w in zip(x,y)),
                            'cosine_similarity': sum(v*w for v,w in zip(x,y))/math.sqrt(nx*ny)})
    return {'schema':'greppy.heads.native-backend-comparison.v1',
            'reports_sha256':[file_sha(p/'report.json') for p in (left,right)],
            'backends':[r['native']['mode'] for r in reports], 'rows':len(comparisons),
            'input_identity_equal':True,
            'maximum_absolute_vector_difference':max(r['max_absolute_difference'] for r in comparisons),
            'minimum_cosine_similarity':min(r['cosine_similarity'] for r in comparisons),
            'per_input':comparisons, 'production_eligible':False,
            'limits':['Small controlled probe, not a representative backend acceptance set.',
                      'No calibrated head decisions or agent task outcomes were compared.']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('left', type=Path)
    parser.add_argument('right', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.left, args.right)
    with args.out.open('x') as stream:
        stream.write(canonical(result)+'\n')
    print(canonical(result))
