"""Export uncalibrated experimental heads and same-vector PyTorch golden cases."""
import argparse
import json
from pathlib import Path
import struct
import zipfile

import numpy as np
import torch

from contracts import canonical, digest
from experiments import HEADS, DIMENSION, bound_file, file_sha, load_bundle, make_model, read_json


def export_one(result, bundle_path, manifest, data, output):
    identity = result['identity']
    if digest(identity) != result['run_id'] or file_sha(bundle_path) != identity['bundle_sha256']:
        raise ValueError('experiment identity or bundle mismatch')
    if (result['release_gate'] != 'not_evaluated' or result['validated_backends']
            or result['calibration'] is not None):
        raise ValueError('candidate exporter does not support release/calibration claims')
    cfg = identity['configuration']
    head, objective, hidden = cfg['head'], cfg['objective'], cfg['hidden']
    if head not in HEADS or hidden not in (0, 128, 256):
        raise ValueError('unsupported candidate architecture')
    if (head == 'log_classifier') != (objective == 'classification') or objective not in ('classification', 'ordinal', 'pairwise'):
        raise ValueError('head/objective mismatch')
    if identity['input_contract'] != manifest['input_contracts'][head] or identity['representation'] != manifest['representation']:
        raise ValueError('feature representation mismatch')
    run = Path(result['_directory'])
    weights_path = bound_file(run, result['assets']['weights'])
    with zipfile.ZipFile(weights_path) as archive:
        if sum(item.file_size for item in archive.infolist()) > 2 * 1024 * 1024:
            raise ValueError('candidate weights exceed size limit')
        if len(archive.namelist()) != len(set(archive.namelist())):
            raise ValueError('duplicate weight entries')
    outputs = 4 if objective == 'classification' else 1
    layout = [('scaler_mean', (DIMENSION,)), ('scaler_scale', (DIMENSION,))]
    if hidden == 0:
        layout += [('network.weight', (outputs, DIMENSION)), ('network.bias', (outputs,))]
    else:
        layout += [('network.0.weight', (hidden, DIMENSION)), ('network.0.bias', (hidden,)),
                   ('network.2.weight', (outputs, hidden)), ('network.2.bias', (outputs,))]
    parameters = layout + ([('cutpoint_base', ()), ('cutpoint_gaps', (2,))] if objective == 'ordinal' else [])
    with np.load(weights_path, allow_pickle=False) as archive:
        if set(archive.files) != {name for name, _ in parameters}:
            raise ValueError('unexpected candidate parameter set')
        values = {name: archive[name].copy() for name, _ in parameters}
    for name, shape in parameters:
        if values[name].shape != shape or values[name].dtype != np.float32 or not np.isfinite(values[name]).all():
            raise ValueError('invalid candidate parameter shape/dtype/value')
    if not (values['scaler_scale'] > 0).all():
        raise ValueError('invalid candidate normalization scale')
    torch.set_num_threads(4)
    model = make_model(hidden, objective).eval()
    model.load_state_dict({name: torch.from_numpy(value) for name, value in values.items()
                           if name not in ('scaler_mean', 'scaler_scale')})
    payload = [values[name].reshape(-1) for name, _ in layout]
    if objective == 'ordinal':
        with torch.no_grad():
            cuts = model.cutpoints().numpy()
        if not np.isfinite(cuts).all() or not (np.diff(cuts) > 0).all():
            raise ValueError('invalid ordered cutpoints')
        payload.append(cuts)
    floats = np.concatenate(payload).astype('<f4')
    header = bytearray(64)
    header[:8] = b'GRPYHD01'
    struct.pack_into('<6IQ', header, 8, 1, DIMENSION, hidden, outputs,
                     ('classification', 'ordinal', 'pairwise').index(objective), HEADS.index(head), len(floats))
    raw = bytes(header) + floats.tobytes(order='C')
    vectors = np.asarray(data[head]['development'][0][:16]).copy()
    with torch.no_grad():
        x = torch.from_numpy(((vectors - values['scaler_mean']) / values['scaler_scale']).astype(np.float32))
        predictions = model(x)
        if objective == 'classification':
            predictions = predictions.softmax(1)
        elif objective == 'ordinal':
            predictions = (predictions - model.cutpoints()).sigmoid().sum(1, keepdim=True)
        predictions = predictions.numpy()
    if not np.isfinite(predictions).all():
        raise ValueError('non-finite golden output')
    cases = []
    for vector, prediction in zip(vectors, predictions):
        expected = ({'kind': 'classification', 'probabilities': prediction.tolist()} if objective == 'classification'
                    else {'kind': 'relevance', 'score': float(prediction[0])})
        cases.append({'vector': vector.tolist(), 'expected': expected})
    golden = {'schema': 'greppy.heads.candidate-golden.v1', 'cases': cases,
              'scope': 'same-vector head arithmetic only; not native raw-text or workflow acceptance'}
    output.mkdir(parents=True, exist_ok=False)
    (output / 'weights.f32le').write_bytes(raw)
    (output / 'golden.json').write_text(canonical(golden) + '\n')
    exported = {'schema': 'greppy.heads.candidate.v1', 'role': identity['role'],
                'head': head, 'objective': objective, 'input_dimension': DIMENSION,
                'hidden_dimension': hidden,
                'input_contract_sha256': identity['input_contract']['sha256'],
                'representation_sha256': digest(identity['representation']),
                'source_run_id': result['run_id'], 'weights_sha256': file_sha(output / 'weights.f32le'),
                'golden_sha256': file_sha(output / 'golden.json'),
                'validated_backends': [], 'calibration': None}
    (output / 'manifest.json').write_text(canonical(exported) + '\n')
    return {'run_id': result['run_id'], 'head': head, 'objective': objective, 'hidden': hidden,
            'seed': cfg['seed'], 'golden_cases': len(cases), 'weights_bytes': len(raw),
            'manifest_sha256': file_sha(output / 'manifest.json')}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--runs', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    manifest, data = load_bundle(args.bundle)
    results = []
    for path in sorted(args.runs.glob('*/result.json')):
        result = read_json(path)
        result['_directory'] = str(path.parent)
        results.append(export_one(result, args.bundle, manifest, data, args.out / result['run_id']))
    if not results:
        raise ValueError('no completed experiments to export')
    report = {'schema': 'greppy.heads.candidate-export.v1', 'exporter_sha256': file_sha(__file__),
              'exports': results, 'candidates': len(results),
              'golden_cases': sum(r['golden_cases'] for r in results), 'production_eligible': False}
    (args.out / 'export-report.json').write_text(canonical(report) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'exports'}, indent=2))
