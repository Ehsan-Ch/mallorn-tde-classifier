"""Crash-safe full-data refit of the unchanged cycle-2 recipe; local output only."""
from __future__ import annotations

import importlib.metadata
import json
import os
from pathlib import Path
import sys
import tempfile

import joblib
import numpy as np
import pandas as pd

from mallorn.checkpointing import atomic_json, atomic_model, exclusive_lock, freeze_json
from mallorn.data import sha256

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'artifacts/cycle2/deployment'


def atomic_csv(path, frame):
    fd, tmp = tempfile.mkstemp(prefix=path.name, suffix='.pending', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as handle:
            frame.to_csv(handle, index=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def run():
    # Fail before importing native CatBoost in an incompatible Linux runtime.
    if sys.platform.startswith('linux'):
        with open('/proc/self/statm') as handle:
            handle.read(100)
    from cycle2 import inputs
    from improve import fit_candidate, predict_candidate
    frozen = ROOT/'artifacts/cycle2/frozen_comparison.json'
    record = json.loads(frozen.read_text())
    if sha256(ROOT/'scripts/cycle2.py') != record['code_sha256']:
        raise ValueError('Frozen cycle-2 implementation changed')
    recipe = record['recipes'][record['chosen']]
    template = pd.read_csv(ROOT/'data/mallorn/sample_submission.csv')
    assert template.columns.tolist() == ['object_id', 'prediction']
    assert len(template) == 7135 and template.object_id.is_unique
    with exclusive_lock(OUT/'refit.lock'):
        source_files = ['scripts/refit_frozen.py', 'src/mallorn/checkpointing.py',
                        'data/mallorn/train_log.csv', 'data/mallorn/sample_submission.csv']
        for name in recipe['members']:
            x, meta, spec, train_path = inputs(name)
            config = record['component_configs'][name]
            assert sha256(train_path) == config['features_sha256']
            assert sha256(ROOT/'data/mallorn/train_log.csv') == config['metadata_sha256']
            assert sha256(ROOT/'scripts/improve.py') == config['fit_code_sha256']
            source_files += [str(train_path.relative_to(ROOT)), str(train_path.relative_to(ROOT)).replace('_train.csv', '_test.csv')]
        spec_record = {'frozen_recipe_sha256': sha256(frozen),
                       'files': {f: sha256(ROOT/f) for f in source_files},
                       'packages': {k: importlib.metadata.version(k) for k in
                                    ['catboost', 'numpy', 'pandas', 'scikit-learn', 'joblib']},
                       'threads': int(os.environ.get('MALLORN_THREADS', 6)),
                       'threshold': recipe['threshold'], 'seeds': record['deployment_seeds']}
        freeze_json(OUT/'resumable_manifest.json', spec_record)
        signature = sha256(OUT/'resumable_manifest.json')
        all_probabilities = []
        for name in recipe['members']:
            x, meta, spec, train_path = inputs(name)
            test_x = pd.read_csv(Path(str(train_path).replace('_train.csv', '_test.csv')),
                                 index_col='object_id').loc[template.object_id]
            assert x.columns.equals(test_x.columns)
            for seed in record['deployment_seeds']:
                path = OUT/f'checkpoints/{name}_{seed}.joblib'
                if path.exists():
                    checkpoint = joblib.load(path)
                    if checkpoint['manifest_sha256'] != signature:
                        raise ValueError('Refit checkpoint configuration differs')
                    fitted = checkpoint['model']
                    p = predict_candidate(fitted, test_x)
                    if not np.allclose(p, checkpoint['probabilities'], rtol=0, atol=1e-12):
                        raise ValueError('Saved-model replay differs')
                else:
                    fitted = fit_candidate(x, meta, spec, seed)
                    p = predict_candidate(fitted, test_x)
                    checkpoint = {'manifest_sha256': signature, 'model': fitted, 'probabilities': p}
                    atomic_model(path, checkpoint)
                assert p.shape == (len(template),) and np.isfinite(p).all()
                assert ((p >= 0) & (p <= 1)).all()
                all_probabilities.append(p)
                print('CHECKPOINTED', name, seed, flush=True)
        p = np.mean(all_probabilities, axis=0)
        result = template.copy()
        result.prediction = (p >= recipe['threshold']).astype(int)
        atomic_csv(OUT/'submission.csv', result)
        atomic_model(OUT/'probabilities.joblib', p)
        atomic_json(OUT/'manifest.json', {'recipe': record['chosen'],
            'frozen_comparison_sha256': sha256(frozen),
            'resumable_manifest_sha256': signature,
            'submission_sha256': sha256(OUT/'submission.csv'),
            'threshold': recipe['threshold'], 'positive_predictions': int(result.prediction.sum()),
            'objects': len(result), 'models': len(all_probabilities), 'submitted_to_kaggle': False})
        print('LOCAL_REFIT_COMPLETE; no upload performed', flush=True)


if __name__ == '__main__':
    run()
