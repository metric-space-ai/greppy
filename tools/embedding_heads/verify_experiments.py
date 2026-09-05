"""Verify synthetic GPU3 experiments, including interruption/restart equivalence."""
import argparse
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np
import torch

import experiments
from contracts import canonical


def verify(root):
    root = Path(root)
    device = experiments.require_gpu3('cuda:0')
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(4)
    bundle = root / 'bundle' / 'manifest.json'
    manifest, data = experiments.load_bundle(bundle)
    if manifest['role'] != 'synthetic_pipeline_test':
        raise ValueError('this verifier must not train on production data')
    results = [experiments.read_json(p) for p in sorted((root / 'runs').glob('*/result.json'))]
    expected = {(head, objective, hidden, seed)
                for head in experiments.HEADS
                for objective in (('classification',) if head == 'log_classifier' else ('ordinal', 'pairwise'))
                for hidden in (0, 128, 256) for seed in (17, 43, 101)}
    actual = {(r['identity']['configuration']['head'], r['identity']['configuration']['objective'],
               r['identity']['configuration']['hidden'], r['identity']['configuration']['seed']) for r in results}
    if actual != expected or len(results) != 45:
        raise AssertionError('expected exactly 45 completed experiment variants')
    unchanged = 0
    for result in results:
        run_dir = root / 'runs' / result['run_id']
        paths = [run_dir / 'result.json', run_dir / 'manifest.json', run_dir / 'weights.npz', run_dir / 'checkpoint.pt']
        before = [(experiments.file_sha(p), p.stat().st_mtime_ns) for p in paths]
        configuration = result['identity']['configuration']
        with redirect_stdout(io.StringIO()):
            replay = experiments.run_one(bundle, manifest, data[configuration['head']],
                                         root / 'runs', configuration, device)
        if replay != result or before != [(experiments.file_sha(p), p.stat().st_mtime_ns) for p in paths]:
            raise AssertionError('completed replay changed artifacts')
        if not result['epoch_losses'][-1] < result['epoch_losses'][0]:
            raise AssertionError('synthetic training loss did not decrease')
        if result['validated_backends'] or result['release_gate'] != 'not_evaluated':
            raise AssertionError('smoke test claimed production acceptance')
        unchanged += 1
    resumed = []
    real_replace = experiments.os.replace
    # Test each objective/head with the nonlinear architecture and optimizer state.
    selected = [r for r in results if r['identity']['configuration']['hidden'] == 256
                and r['identity']['configuration']['seed'] == 17]
    with tempfile.TemporaryDirectory(prefix='resume-check-', dir=root) as scratch:
        for reference in selected:
            configuration = reference['identity']['configuration']
            run_root = Path(scratch)
            interrupted = False

            def interrupt_after_commit(source, target):
                nonlocal interrupted
                real_replace(source, target)
                if Path(target).name == 'checkpoint.pt' and not interrupted:
                    interrupted = True
                    raise InterruptedError('injected after first durable epoch')

            with redirect_stdout(io.StringIO()):
                try:
                    with patch.object(experiments.os, 'replace', side_effect=interrupt_after_commit):
                        experiments.run_one(bundle, manifest, data[configuration['head']], run_root, configuration, device)
                except InterruptedError:
                    pass
                else:
                    raise AssertionError('interruption was not exercised')
                result = experiments.run_one(bundle, manifest, data[configuration['head']], run_root, configuration, device)
            if result['resume_from_epoch'] != 1:
                raise AssertionError('did not resume the last committed epoch')
            if result['epoch_losses'] != reference['epoch_losses'] or result['development_metrics'] != reference['development_metrics']:
                raise AssertionError('resumed losses or predictions differ')
            with np.load(root / 'runs' / reference['run_id'] / 'weights.npz', allow_pickle=False) as expected_weights:
                with np.load(run_root / result['run_id'] / 'weights.npz', allow_pickle=False) as actual_weights:
                    if set(expected_weights.files) != set(actual_weights.files):
                        raise AssertionError('weight names differ')
                    for name in expected_weights.files:
                        np.testing.assert_array_equal(expected_weights[name], actual_weights[name])
            resumed.append({'head': configuration['head'], 'objective': configuration['objective'],
                            'hidden': 256, 'resume_from_epoch': 1, 'weights_bit_identical': True})
    return {'schema': 'greppy.heads.synthetic-experiment-check.v1', 'host': experiments.GPU_HOST,
            'device': 'cuda:0', 'complete_variants': len(results), 'immutable_replays': unchanged,
            'resume_checks': resumed, 'all_training_losses_decreased': True,
            'bundle_sha256': experiments.file_sha(bundle), 'trainer_sha256': experiments.file_sha(experiments.__file__),
            'verifier_sha256': experiments.file_sha(__file__), 'production_eligible': False,
            'note': 'Synthetic vectors verify pipeline mechanics only; no native encoder or task acceptance.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    import os
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    report = verify(args.root)
    with args.report.open('x') as stream:
        stream.write(canonical(report) + '\n')
    print(json.dumps(report, indent=2))
