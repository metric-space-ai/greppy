"""GPU3-only reproducible head experiments; never grants a production release.

Consumes hash-bound, already prepared vectors. Native extraction and independent
label admission are prerequisites, not inferred from this experiment's metrics.
"""
import argparse
from collections import defaultdict
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import socket
import time

from contracts import canonical, digest, strict_json

LABELS = ['error', 'warning', 'progress', 'text']
HEADS = ('log_classifier', 'log_ranker', 'web_ranker')
SCHEMA = 'greppy.heads.feature-bundle.v1'
DIMENSION = 768
GPU_HOST = 'gpu3-a4500'


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(8 << 20), b''):
            h.update(part)
    return h.hexdigest()


def require_hash(value):
    if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError('invalid SHA256')
    return value


def bound_file(root, item):
    relative = Path(item['path'])
    path = (root / relative).resolve()
    if relative.is_absolute() or not path.is_relative_to(root.resolve()):
        raise ValueError('bundle file escapes its root')
    if file_sha(path) != require_hash(item['sha256']):
        raise ValueError('bundle file hash mismatch')
    return path


def read_json(path):
    return strict_json(Path(path).read_text())


def load_bundle(path):
    """Reject final/diagnostic leakage and stale feature/label combinations."""
    import numpy as np
    path = Path(path)
    manifest = read_json(path)
    if manifest.get('schema') != SCHEMA or set(manifest.get('heads', {})) != set(HEADS):
        raise ValueError('unsupported feature bundle')
    role = manifest.get('role')
    if role not in ('synthetic_pipeline_test', 'development_candidate'):
        raise ValueError('unsupported experiment role')
    representation = manifest['representation']
    if representation.get('dimension') != DIMENSION:
        raise ValueError('expected 768 native dimensions')
    if role == 'synthetic_pipeline_test':
        if representation.get('kind') != 'synthetic_test':
            raise ValueError('synthetic tests must not claim native encoder provenance')
    else:
        if (representation.get('kind') != 'native_q4_k'
                or representation.get('encoder') != 'EmbeddingGemma-300M'
                or representation.get('frozen') is not True
                or representation.get('quantization') != 'Q4_K'
                or representation.get('backend') not in ('cpu', 'metal', 'cuda')):
            raise ValueError('native frozen Q4_K representation required')
        for key in ('encoder_sha256', 'tokenizer_sha256', 'binary_sha256'):
            require_hash(representation[key])
    provenance = manifest.get('label_provenance', {})
    if role == 'development_candidate':
        if not isinstance(provenance.get('rubric'), str) or not provenance['rubric']:
            raise ValueError('missing label rubric')
        for name in ('source_registry', 'split_manifest', 'annotations', 'admission_review', 'teacher_configurations'):
            bound_file(path.parent, provenance['artifacts'][name])
    contracts = manifest['input_contracts']
    if set(contracts) != set(HEADS):
        raise ValueError('all three input contracts are required')
    for head, contract in contracts.items():
        require_hash(contract['sha256'])
        if digest(contract['value']) != contract['sha256']:
            raise ValueError('input contract hash mismatch')
        value = contract['value']
        if (value.get('head') != head or not value.get('pooling')
                or not value.get('normalization') or not value.get('layer')
                or type(value.get('token_limit')) is not int or value['token_limit'] <= 0):
            raise ValueError('incomplete representation input contract')
        require_hash(value['preprocessor_sha256'])
        require_hash(value['prompt_sha256'])
        if head != 'log_classifier' and value.get('task_conditioned') is not True:
            raise ValueError('ranking requires task-conditioned inputs')
    # Label metadata is inspected before opening any vector arrays. Final-test
    # artifacts are never a legal partition of an experiment bundle.
    seen_groups = {}
    seen_sources = {}
    seen_inputs = {}
    data = {}
    for head in HEADS:
        item = manifest['heads'][head]
        if set(item) != {'train', 'development'}:
            raise ValueError('only train/development partitions are permitted')
        data[head] = {}
        for split, files in item.items():
            rows_path = bound_file(path.parent, files['rows'])
            rows = [strict_json(line) for line in rows_path.read_text().splitlines()]
            if not rows:
                raise ValueError('empty feature partition')
            seen_rows = set()
            for row in rows:
                if row.get('split') != split or row.get('head') != head:
                    raise ValueError('row head/split mismatch')
                for key in ('candidate_id', 'source_id', 'group_key', 'comparison_id'):
                    if not isinstance(row.get(key), str) or not row[key]:
                        raise ValueError('missing row identity')
                row_key = (row['candidate_id'], row.get('task_sha256'))
                if row_key in seen_rows:
                    raise ValueError('duplicate candidate/task row')
                seen_rows.add(row_key)
                for key in ('input_sha256', 'annotation_sha256', 'evidence_sha256'):
                    require_hash(row[key])
                if head != 'log_classifier':
                    require_hash(row['task_sha256'])
                if row.get('input_contract_sha256') != contracts[head]['sha256']:
                    raise ValueError('row input contract mismatch')
                expected_status = 'synthetic_test' if role == 'synthetic_pipeline_test' else 'review_complete'
                if row.get('admission') != expected_status:
                    raise ValueError('unadmitted row')
                for owners, key in ((seen_groups, row['group_key']),
                                    (seen_sources, row['source_id']),
                                    (seen_inputs, row['input_sha256'])):
                    previous = owners.setdefault(key, split)
                    if previous != split:
                        raise ValueError('related sources or identical inputs cross splits')
                y = row['label']
                if type(y) is not int or not 0 <= y < 4:
                    raise ValueError('invalid class/ordinal label')
            vectors_path = bound_file(path.parent, files['vectors'])
            vectors = np.load(vectors_path, allow_pickle=False, mmap_mode='r')
            if vectors.dtype != np.dtype('float32') or vectors.shape != (len(rows), DIMENSION):
                raise ValueError('invalid vector shape or dtype')
            for start in range(0, len(rows), 4096):
                if not np.isfinite(vectors[start:start + 4096]).all():
                    raise ValueError('non-finite features')
            if head == 'log_classifier' and split == 'train' and {r['label'] for r in rows} != set(range(4)):
                raise ValueError('classifier training requires all four classes')
            # A comparison scope cannot silently mix outputs, goals or lineages.
            scopes = {}
            for row in rows:
                scope = (row['source_id'], row['group_key'], row.get('task_sha256'))
                if scopes.setdefault(row['comparison_id'], scope) != scope:
                    raise ValueError('comparison scope mixes sources or tasks')
            data[head][split] = (vectors, rows)
    return manifest, data


def fit_scaler(vectors):
    """Streaming parallel variance; peak temporary memory is one feature batch."""
    import numpy as np
    count = 0
    mean = np.zeros(DIMENSION, dtype=np.float64)
    m2 = np.zeros(DIMENSION, dtype=np.float64)
    for start in range(0, len(vectors), 4096):
        block = np.asarray(vectors[start:start + 4096], dtype=np.float64)
        size = len(block)
        block_mean = block.mean(0)
        delta = block_mean - mean
        total = count + size
        m2 += ((block - block_mean) ** 2).sum(0) + delta ** 2 * count * size / total
        mean += delta * size / total
        count = total
    scale = np.sqrt(m2 / count).astype(np.float32)
    mean = mean.astype(np.float32)
    if not np.isfinite(mean).all() or not np.isfinite(scale).all():
        raise ValueError('non-finite scaler')
    scale[scale < 1e-6] = 1.0
    return mean, scale


class ScaledFeatures:
    def __init__(self, vectors, mean, scale):
        self.vectors, self.mean, self.scale = vectors, mean, scale

    def __getitem__(self, indexes):
        import numpy as np
        import torch
        values = ((np.asarray(self.vectors[indexes]) - self.mean) / self.scale).astype(np.float32)
        if not np.isfinite(values).all():
            raise ValueError('normalization produced non-finite inputs')
        return torch.from_numpy(values)


def require_gpu3(device):
    import torch
    if socket.gethostname().split('.')[0] != GPU_HOST:
        raise RuntimeError('training is restricted to the GPU3 host gpu3-a4500')
    if not device.startswith('cuda:') or not torch.cuda.is_available():
        raise RuntimeError('training requires an explicit CUDA device on GPU3')
    ordinal = int(device.split(':', 1)[1])
    if not 0 <= ordinal < torch.cuda.device_count():
        raise RuntimeError('CUDA device is unavailable')
    return torch.device(device)


def make_model(hidden, objective):
    import torch
    if hidden not in (0, 128, 256) or objective not in ('classification', 'ordinal', 'pairwise'):
        raise ValueError('unsupported head architecture')

    class Head(torch.nn.Module):
        def __init__(self):
            super().__init__()
            outputs = 4 if objective == 'classification' else 1
            self.network = (torch.nn.Linear(DIMENSION, outputs) if hidden == 0 else
                            torch.nn.Sequential(torch.nn.Linear(DIMENSION, hidden),
                                                torch.nn.GELU(approximate='none'),
                                                torch.nn.Linear(hidden, outputs)))
            if objective == 'ordinal':
                self.cutpoint_base = torch.nn.Parameter(torch.tensor(-1.0))
                self.cutpoint_gaps = torch.nn.Parameter(torch.zeros(2))

        def cutpoints(self):
            return torch.cat((self.cutpoint_base.reshape(1),
                              self.cutpoint_base + torch.nn.functional.softplus(self.cutpoint_gaps).cumsum(0)))

        def forward(self, x):
            return self.network(x)

    return Head()


def preference_pairs(rows, seed, per_scope=256):
    """Bound sampling within each output+task; never compare unrelated outputs."""
    import numpy as np
    if type(per_scope) is not int or per_scope < 1:
        raise ValueError('positive pair budget required')
    groups = defaultdict(lambda: defaultdict(list))
    for index, row in enumerate(rows):
        groups[row['comparison_id']][row['label']].append(index)
    rng = np.random.default_rng(seed)
    pairs = []
    for scope in sorted(groups):
        grades = groups[scope]
        strata = [(hi, lo) for hi in sorted(grades) for lo in sorted(grades) if hi > lo]
        if not strata:
            continue
        # Equal grade-pair strata, sampling rows uniformly with replacement.
        # Record the budget in the run manifest; no N-squared allocation.
        for offset in range(per_scope):
            hi, lo = strata[offset % len(strata)]
            pairs.append((int(rng.choice(grades[hi])), int(rng.choice(grades[lo]))))
    return np.asarray(pairs, dtype=np.int64).reshape(-1, 2)


def evaluate(model, x, rows, objective, device, batch_size):
    import numpy as np
    import torch
    model.eval()
    chunks = []
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            chunk = model(x[start:start + batch_size].to(device))
            if objective == 'classification':
                chunk = chunk.softmax(1)
            elif objective == 'ordinal':
                chunk = (chunk - model.cutpoints()).sigmoid().sum(1, keepdim=True)
            chunks.append(chunk.cpu().numpy())
    predictions = np.concatenate(chunks)
    if not np.isfinite(predictions).all():
        raise ValueError('non-finite head outputs')
    labels = np.array([r['label'] for r in rows])
    if objective == 'classification':
        confusion = np.zeros((4, 4), dtype=np.int64)
        top = predictions.argmax(1)
        np.add.at(confusion, (labels, top), 1)
        per_class = {}
        for k, name in enumerate(LABELS):
            tp = int(confusion[k, k]); fn = int(confusion[k].sum()) - tp
            fp = int(confusion[:, k].sum()) - tp
            tn = len(rows) - tp - fn - fp
            per_class[name] = {'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
                               'precision': tp / (tp + fp) if tp + fp else None,
                               'recall': tp / (tp + fn) if tp + fn else None}
        confidence = predictions.max(1)
        bins = np.minimum((confidence * 15).astype(int), 14)
        ece = sum(float((bins == b).mean()) * abs(float((top[bins == b] == labels[bins == b]).mean())
                  - float(confidence[bins == b].mean())) for b in range(15) if (bins == b).any())
        return {'rows': len(rows), 'confusion': confusion.tolist(), 'per_class': per_class,
                'nll': float(-np.log(np.maximum(predictions[np.arange(len(rows)), labels], 1e-30)).mean()),
                'brier': float(((predictions - np.eye(4)[labels]) ** 2).sum(1).mean()),
                'ece_15_bins': ece, 'calibration': 'unfitted; separate backend validation required'}
    scopes = defaultdict(list)
    for i, row in enumerate(rows):
        scopes[row['comparison_id']].append(i)
    results = []
    for scope in sorted(scopes):
        indexes = scopes[scope]
        order = sorted(indexes, key=lambda i: (-float(predictions[i, 0]), rows[i]['candidate_id']))
        ideal = sorted(indexes, key=lambda i: -int(labels[i]))
        gains = lambda ids: sum((2 ** int(labels[i]) - 1) / math.log2(j + 2) for j, i in enumerate(ids))
        denominator = gains(ideal)
        required = {i for i in indexes if labels[i] == 3}
        recalls = {str(k): len(required.intersection(order[:k])) / len(required) if required else None
                   for k in (1, 3, 5, 10)}
        results.append({'comparison_id': scope, 'candidates': len(indexes),
                        'ndcg': gains(order) / denominator if denominator else None,
                        'required_recall_at_k': recalls})
    available = [r['ndcg'] for r in results if r['ndcg'] is not None]
    return {'rows': len(rows), 'scopes': len(results), 'mean_ndcg': sum(available) / len(available) if available else None,
            'per_scope': results, 'note': 'Ranking diagnostics do not prove evidence retention or task success.'}


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.pending')
    with temporary.open('w') as stream:
        stream.write(canonical(value) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


@contextmanager
def exclusive_run(path):
    path.mkdir(parents=True, exist_ok=True)
    with (path / 'run.lock').open('a') as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('experiment already running') from error
        yield


def run_one(bundle_path, manifest, partitions, root, configuration, device):
    import numpy as np
    import torch
    device = require_gpu3(str(device))
    identity = {'schema': 'greppy.heads.experiment.v1', 'bundle_sha256': file_sha(bundle_path),
                'trainer_sha256': file_sha(__file__), 'contracts_code_sha256': file_sha(Path(__file__).with_name('contracts.py')),
                'configuration': configuration, 'python': platform.python_version(),
                'torch': torch.__version__, 'numpy': np.__version__,
                'cuda': torch.version.cuda, 'device_name': torch.cuda.get_device_name(device),
                'device_capability': list(torch.cuda.get_device_capability(device)),
                'representation': manifest['representation'],
                'input_contract': manifest['input_contracts'][configuration['head']],
                'label_provenance': manifest.get('label_provenance'),
                'role': manifest['role']}
    run_id = digest(identity)
    directory = root / run_id
    with exclusive_run(directory):
        completed = directory / 'result.json'
        if completed.exists():
            result = read_json(completed)
            if result['run_id'] != run_id or result['identity'] != identity:
                raise ValueError('completed run identity mismatch')
            for asset in result['assets'].values():
                bound_file(directory, asset)
            print(canonical({'run_id': run_id, 'status': 'already_complete'}), flush=True)
            return result
        atomic_json(directory / 'manifest.json', {'run_id': run_id, 'identity': identity,
                                                'release_gate': 'not_evaluated'})
        seed = configuration['seed']
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        train_vectors, train_rows = partitions['train']
        dev_vectors, dev_rows = partitions['development']
        # Fit normalization only on training, in float64; persist float32 values
        # used both by this runner and the eventual portable candidate loader.
        mean, scale = fit_scaler(train_vectors)
        x = ScaledFeatures(train_vectors, mean, scale)
        dx = ScaledFeatures(dev_vectors, mean, scale)
        y = torch.tensor([r['label'] for r in train_rows], dtype=torch.long)
        objective = configuration['objective']
        model = make_model(configuration['hidden'], objective).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=configuration['learning_rate'],
                                      weight_decay=configuration['weight_decay'])
        checkpoint = directory / 'checkpoint.pt'
        start_epoch = 0
        epoch_losses = []
        if checkpoint.exists():
            state = torch.load(checkpoint, map_location=device, weights_only=True)
            if state['run_id'] != run_id:
                raise ValueError('checkpoint identity mismatch')
            model.load_state_dict(state['model'])
            optimizer.load_state_dict(state['optimizer'])
            start_epoch = state['completed_epochs']
            epoch_losses = state['epoch_losses']
            if not 0 <= start_epoch <= configuration['epochs']:
                raise ValueError('invalid checkpoint epoch')
        started = time.perf_counter()
        for epoch in range(start_epoch, configuration['epochs']):
            model.train()
            rng = np.random.default_rng(seed + epoch * 1000003)
            pairs = preference_pairs(train_rows, seed + epoch, configuration['pairs_per_scope']) if objective == 'pairwise' else None
            if pairs is not None and not len(pairs):
                raise ValueError('pairwise training has no within-output preferences')
            count = len(pairs) if pairs is not None else len(train_rows)
            order = rng.permutation(count)
            loss_sum = 0.0
            for start in range(0, count, configuration['batch_size']):
                indexes = order[start:start + configuration['batch_size']]
                optimizer.zero_grad(set_to_none=True)
                if pairs is not None:
                    selected = pairs[indexes]
                    hi = model(x[selected[:, 0]].to(device)).reshape(-1)
                    lo = model(x[selected[:, 1]].to(device)).reshape(-1)
                    loss = torch.nn.functional.softplus(-(hi - lo)).mean()
                else:
                    prediction = model(x[indexes].to(device))
                    truth = y[indexes].to(device)
                    if objective == 'classification':
                        loss = torch.nn.functional.cross_entropy(prediction, truth)
                    else:
                        targets = (truth[:, None] > torch.arange(3, device=device)).float()
                        loss = torch.nn.functional.binary_cross_entropy_with_logits(prediction - model.cutpoints(), targets)
                if not torch.isfinite(loss):
                    raise ValueError('non-finite training loss')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0, error_if_nonfinite=True)
                optimizer.step()
                loss_sum += float(loss.detach()) * len(indexes)
            epoch_losses.append(loss_sum / count)
            temporary = checkpoint.with_suffix('.pt.pending')
            torch.save({'run_id': run_id, 'completed_epochs': epoch + 1,
                        'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                        'epoch_losses': epoch_losses}, temporary)
            # Checkpoints contain no dropout/RNG-dependent operations. Each epoch's
            # sampling is a pure function of seed+epoch, including after restart.
            with temporary.open('rb') as stream:
                os.fsync(stream.fileno())
            os.replace(temporary, checkpoint)
            print(canonical({'run_id': run_id, 'epoch': epoch + 1,
                             'loss': epoch_losses[-1]}), flush=True)
        metrics = evaluate(model, dx, dev_rows, objective, device, configuration['batch_size'])
        weights = {'scaler_mean': mean, 'scaler_scale': scale}
        weights.update({name: value.detach().cpu().numpy() for name, value in model.state_dict().items()})
        if any(not np.isfinite(value).all() for value in weights.values()):
            raise ValueError('non-finite candidate weights')
        weights_path = directory / 'weights.npz'
        with weights_path.with_suffix('.npz.pending').open('wb') as stream:
            np.savez(stream, **weights)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(weights_path.with_suffix('.npz.pending'), weights_path)
        result = {'run_id': run_id, 'identity': identity, 'epoch_losses': epoch_losses,
                  'development_metrics': metrics, 'resume_from_epoch': start_epoch,
                  'active_seconds_this_invocation': time.perf_counter() - started,
                  'assets': {'weights': {'path': 'weights.npz', 'sha256': file_sha(weights_path)},
                             'checkpoint': {'path': 'checkpoint.pt', 'sha256': file_sha(checkpoint)}},
                  'release_gate': 'not_evaluated', 'validated_backends': [],
                  'calibration': None}
        atomic_json(completed, result)
        print(canonical({'run_id': run_id, 'status': 'complete', 'head': configuration['head'],
                         'objective': objective, 'hidden': configuration['hidden'], 'seed': seed}), flush=True)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--weight-decay', type=float, default=0.0001)
    parser.add_argument('--pairs-per-scope', type=int, default=256)
    parser.add_argument('--seeds', type=int, nargs=3, default=[17, 43, 101])
    args = parser.parse_args()
    if (min(args.epochs, args.batch_size, args.pairs_per_scope) < 1
            or len(set(args.seeds)) != 3 or any(seed < 0 or seed >= 2 ** 31 for seed in args.seeds)
            or not math.isfinite(args.learning_rate) or args.learning_rate <= 0
            or not math.isfinite(args.weight_decay) or args.weight_decay < 0):
        parser.error('invalid experiment configuration')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import torch
    device = require_gpu3(args.device)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(4)
    manifest, data = load_bundle(args.bundle)
    for head in HEADS:
        objectives = ('classification',) if head == 'log_classifier' else ('ordinal', 'pairwise')
        for objective in objectives:
            for hidden in (0, 128, 256):
                for seed in args.seeds:
                    configuration = {'head': head, 'objective': objective, 'hidden': hidden, 'seed': seed,
                                     'epochs': args.epochs, 'batch_size': args.batch_size,
                                     'learning_rate': args.learning_rate, 'weight_decay': args.weight_decay,
                                     'pairs_per_scope': args.pairs_per_scope, 'normalization': 'train_only_standard_scaler',
                                     'calibration': 'not_fitted', 'device': args.device,
                                     'deterministic_algorithms': True, 'allow_tf32': False}
                    run_one(args.bundle, manifest, data[head], args.out, configuration, device)


if __name__ == '__main__':
    main()
