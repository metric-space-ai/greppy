"""Create visibly synthetic feature bundles for runner tests, never native evidence."""
import argparse
import json
from pathlib import Path

from contracts import canonical, digest
from experiments import DIMENSION, HEADS, SCHEMA, file_sha


def create_fixture(root):
    import numpy as np
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    manifest = {'schema': SCHEMA, 'role': 'synthetic_pipeline_test',
                'representation': {'dimension': DIMENSION, 'kind': 'synthetic_test'},
                'label_provenance': {'kind': 'programmed_test_labels', 'production_eligible': False},
                'input_contracts': {}, 'heads': {}}
    for head in HEADS:
        value = {'head': head, 'pooling': 'synthetic_test', 'normalization': 'none',
                 'layer': 'synthetic_test', 'token_limit': 128,
                 'preprocessor_sha256': file_sha(__file__),
                 'prompt_sha256': digest('synthetic_test'), 'task_conditioned': head != 'log_classifier'}
        contract_hash = digest(value)
        manifest['input_contracts'][head] = {'sha256': contract_hash, 'value': value}
        manifest['heads'][head] = {}
        for split, count in [('train', 32), ('development', 16)]:
            rows = []
            rng = np.random.default_rng(23 if split == 'train' else 71)
            vectors = rng.normal(0, 0.01, (count, DIMENSION)).astype(np.float32)
            for index in range(count):
                label = index % 4
                vectors[index, label] += 1.0
                source = f'{split}-output-{index // 4}'
                if head == 'web_ranker':
                    source = 'web-' + source
                rows.append({'head': head, 'split': split, 'candidate_id': f'{source}-record-{label}',
                             'source_id': source, 'group_key': source, 'comparison_id': source,
                             'task_sha256': digest('test task'),
                             'input_sha256': digest([head, split, index]),
                             'input_contract_sha256': contract_hash,
                             'annotation_sha256': digest(['synthetic', label]),
                             'evidence_sha256': digest(['synthetic', source, index]),
                             'admission': 'synthetic_test', 'label': label})
            for row in rows:
                row['source_sha256'] = digest(['synthetic source', row['source_id']])
                if head != 'log_classifier':
                    row['conditioning_sha256'] = digest({'task': 'test task', 'action': None})
                if head == 'web_ranker':
                    row['observation_id'] = row['source_id'] + '-observation'
                    row['action_sha256'] = digest(None)
            row_path = root / f'{head}-{split}.jsonl'
            row_path.write_text(''.join(canonical(row) + '\n' for row in rows))
            vector_path = root / f'{head}-{split}.npy'
            np.save(vector_path, vectors, allow_pickle=False)
            manifest['heads'][head][split] = {
                'rows': {'path': row_path.name, 'sha256': file_sha(row_path)},
                'vectors': {'path': vector_path.name, 'sha256': file_sha(vector_path)}}
    path = root / 'manifest.json'
    path.write_text(canonical(manifest) + '\n')
    return path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps({'manifest': str(create_fixture(args.out)), 'production_eligible': False}))
