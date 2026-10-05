"""Fixed LightGBM diversification experiment, driven by resume_research.py."""
from __future__ import annotations

from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import time

import joblib
from lightgbm import LGBMClassifier
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from mallorn.checkpointing import atomic_json, atomic_model, freeze_json
from mallorn.data import sha256
from mallorn.evaluation import best_threshold, score

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/cycle3'
COMPONENTS = {
    'lgb_physics_aux': ('physics', 'aux'),
    'lgb_physical_hierarchy': ('physical_view', 'hierarchy'),
    'lgb_complete_binary': ('complete', 'binary'),
}
PARAMS = dict(n_estimators=900, learning_rate=.025, num_leaves=9,
              min_child_samples=30, reg_lambda=10., colsample_bytree=.8,
              subsample=.85, subsample_freq=1, deterministic=True,
              force_col_wise=True, verbosity=-1)
FORBIDDEN = {'target', 'SpecType', 'Z_err', 'object_id'}


def now():
    return datetime.now(timezone.utc).isoformat()


def initialize():
    files = ['data/mallorn/train_log.csv', 'scripts/cycle3.py',
             'src/mallorn/checkpointing.py', 'src/mallorn/evaluation.py',
             'docs/THIRD_CYCLE_PROTOCOL.md', 'artifacts/cycle2/frozen_comparison.json']
    files += [f'artifacts/improvement/{view}_train.csv' for view, _ in COMPONENTS.values()]
    reference = json.loads((ROOT / files[5]).read_text())
    receipt = json.loads((ROOT/'artifacts/cycle2/freeze_receipt.json').read_text())
    assert sha256(ROOT/files[5]) == receipt['frozen_comparison_sha256']
    assert reference['chosen'] == 'diverse'
    for name in reference['recipes']['diverse']['members']:
        for repeat in range(2):
            files.append(f'artifacts/cycle2/{name}/oof_repeat{repeat}.npy')
    manifest = {'schema_version': 1, 'components': COMPONENTS, 'params': PARAMS,
                'split_seeds': [20261014, 20261015], 'folds': 5,
                'threads': int(os.environ.get('MALLORN_THREADS', 4)),
                'packages': {k: importlib.metadata.version(k) for k in
                             ['lightgbm', 'numpy', 'pandas', 'scikit-learn', 'joblib']},
                'files': {f: sha256(ROOT/f) for f in files},
                'publishing_allowed': False}
    # Normalize tuples to their on-disk JSON representation before comparison.
    manifest = json.loads(json.dumps(manifest))
    freeze_json(OUT/'manifest.json', manifest)
    if not (OUT/'start_receipt.json').exists():
        atomic_json(OUT/'start_receipt.json', {'started_at': now(),
                    'manifest_sha256': sha256(OUT/'manifest.json')})
    return manifest


def inputs(name):
    view, _ = COMPONENTS[name]
    meta = pd.read_csv(ROOT/'data/mallorn/train_log.csv').set_index('object_id')
    x = pd.read_csv(ROOT/f'artifacts/improvement/{view}_train.csv', index_col='object_id')
    assert x.index.is_unique and set(x.index) == set(meta.index)
    x = x.loc[meta.index]
    assert not FORBIDDEN & set(x.columns) and not np.isinf(x.to_numpy()).any()
    return x, meta


def fit(x, meta, kind, seed, threads):
    def model(weights):
        return LGBMClassifier(**PARAMS, random_state=seed, n_jobs=threads, class_weight=weights)
    y = meta.target.to_numpy()
    if kind == 'aux':
        labels = np.where(meta.SpecType.eq('AGN'), 0, np.where(y == 1, 2, 1))
        return {'model': model({0: 1, 1: 1, 2: 3}).fit(x, labels), 'positive': 2}
    if kind == 'hierarchy':
        transient = ~meta.SpecType.eq('AGN').to_numpy()
        return {'gate': model({0: 1, 1: 1}).fit(x, transient.astype(int)),
                'model': model({0: 1, 1: 3}).fit(x.loc[transient], y[transient]), 'positive': 1}
    return {'model': model({0: 1, 1: 3}).fit(x, y), 'positive': 1}


def predict(fitted, x):
    model = fitted['model']
    p = model.predict_proba(x)[:, list(model.classes_).index(fitted['positive'])]
    if 'gate' in fitted:
        gate = fitted['gate']
        p *= gate.predict_proba(x)[:, list(gate.classes_).index(1)]
    return p


def fold_path(name, repeat, fold):
    return OUT/name/f'repeat{repeat}_fold{fold}.joblib'


def verify_fold(record, signature, ids, a, b):
    if record['manifest_sha256'] != signature:
        raise ValueError('Checkpoint belongs to a different experiment')
    if record['fitting_ids'] != ids[a].tolist() or record['validation_ids'] != ids[b].tolist():
        raise ValueError('Checkpoint fold assignment differs')
    if set(record['fitting_ids']) & set(record['validation_ids']):
        raise ValueError('Fitting/validation overlap')
    p = np.asarray(record['probabilities'])
    if p.shape != (len(b),) or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError('Invalid checkpoint probabilities')
    return p


def run_folds(manifest, deadline=float('inf'), max_new=None):
    signature = sha256(OUT/'manifest.json')
    completed = 0
    for name, (_, kind) in COMPONENTS.items():
        x, meta = inputs(name)
        y = meta.target.to_numpy()
        for repeat, split_seed in enumerate(manifest['split_seeds']):
            cv = StratifiedKFold(5, shuffle=True, random_state=split_seed)
            for fold, (a, b) in enumerate(cv.split(x, y)):
                path = fold_path(name, repeat, fold)
                if path.exists():
                    record = joblib.load(path)
                    verify_fold(record, signature, x.index, a, b)
                    continue
                if time.monotonic() >= deadline or (max_new is not None and completed >= max_new):
                    return False
                start = time.monotonic()
                seed = 42 + 100*repeat + fold
                atomic_json(OUT/'progress.json', {'state': 'training', 'component': name,
                    'repeat': repeat, 'fold': fold, 'started_at': now()})
                fitted = fit(x.iloc[a], meta.iloc[a], kind, seed, manifest['threads'])
                p = predict(fitted, x.iloc[b])
                record = {'manifest_sha256': signature, 'fitting_ids': x.index[a].tolist(),
                          'validation_ids': x.index[b].tolist(), 'probabilities': p,
                          'model': fitted, 'columns': x.columns.tolist(), 'seed': seed,
                          'seconds': time.monotonic()-start, 'completed_at': now()}
                verify_fold(record, signature, x.index, a, b)
                atomic_model(path, record)
                completed += 1
                print(f'{name} repeat={repeat} fold={fold} committed ({record["seconds"]:.1f}s)', flush=True)
    return True


def stable_threshold(y, p):
    rng = np.random.default_rng(20261016)
    classes = [np.flatnonzero(y == k) for k in [0, 1]]
    values = []
    for _ in range(500):
        idx = np.concatenate([rng.choice(a, len(a), replace=True) for a in classes])
        values.append(best_threshold(y[idx], p[idx])[0])
    return float(np.median(values))


def paired_interval(y, decisions, reference):
    rng = np.random.default_rng(20261017)
    classes = [np.flatnonzero(y == k) for k in [0, 1]]
    def f1(a, b):
        return 2*np.sum(a & b)/max(int(a.sum()+b.sum()), 1)
    differences = []
    for _ in range(2000):
        idx = np.concatenate([rng.choice(a, len(a), replace=True) for a in classes])
        differences.append(f1(y[idx], decisions[idx])-f1(y[idx], reference[idx]))
    return np.quantile(differences, [.025, .975]).tolist()


def compare(manifest):
    _, meta = inputs('lgb_physics_aux')
    y = meta.target.to_numpy()
    signature = sha256(OUT/'manifest.json')
    predictions = {}
    replay = {}
    for name in COMPONENTS:
        x, _ = inputs(name)
        repeated = np.full((2, len(y)), np.nan)
        for repeat, seed in enumerate(manifest['split_seeds']):
            seen = np.zeros(len(y), dtype=int)
            for fold, (a, b) in enumerate(StratifiedKFold(5, shuffle=True, random_state=seed).split(x, y)):
                record = joblib.load(fold_path(name, repeat, fold))
                p = verify_fold(record, signature, x.index, a, b)
                assert record['columns'] == x.columns.tolist()
                replay_p = predict(record['model'], x.iloc[b])
                delta = float(np.max(np.abs(replay_p-p)))
                if delta > 1e-12:
                    raise ValueError('Saved model replay mismatch')
                replay[f'{name}/{repeat}/{fold}'] = delta
                repeated[repeat, b] = p
                seen[b] += 1
            assert np.all(seen == 1)
        predictions[name] = repeated
    old = json.loads((ROOT/'artifacts/cycle2/frozen_comparison.json').read_text())
    members = old['recipes']['diverse']['members']
    ref = np.mean([np.stack([np.load(ROOT/f'artifacts/cycle2/{m}/oof_repeat{r}.npy')
                            for r in range(2)]) for m in members], axis=0)
    assert ref.shape == (2, len(y)) and np.isfinite(ref).all()
    recipes = {'reference': ref}
    for name, p in predictions.items():
        recipes['blend_'+name] = .5*(ref+p)
    lgb_mean = np.mean(list(predictions.values()), axis=0)
    recipes['lgb_ensemble'] = lgb_mean
    recipes['blend_lgb_ensemble'] = .5*(ref+lgb_mean)
    results = {}
    for name, repeated in recipes.items():
        p = repeated.mean(axis=0)
        threshold = stable_threshold(y, p)
        results[name] = {'threshold': threshold, 'combined': score(y, p, p >= threshold),
            'repeat_scores': [score(y, q, q >= threshold) for q in repeated]}
    reference = results['reference']
    assert np.isclose(reference['threshold'], old['recipes']['diverse']['threshold'])
    assert np.isclose(reference['combined']['f1'], old['recipes']['diverse']['combined']['f1'])
    eligible = [name for name, r in results.items() if name != 'reference'
                and r['combined']['f1'] >= reference['combined']['f1']+.005
                and r['combined']['average_precision'] >= reference['combined']['average_precision']
                and all(a['f1'] >= b['f1']-.005 for a, b in zip(r['repeat_scores'], reference['repeat_scores']))]
    chosen = max(eligible, key=lambda name: results[name]['combined']['f1']) if eligible else 'reference'
    ref_decisions = ref.mean(axis=0) >= reference['threshold']
    for name in results:
        decisions = recipes[name].mean(axis=0) >= results[name]['threshold']
        results[name]['paired_f1_difference_95pct_descriptive'] = paired_interval(y, decisions, ref_decisions)
    component_scores = {}
    for name, repeated in predictions.items():
        p = repeated.mean(axis=0)
        threshold = stable_threshold(y, p)
        component_scores[name] = score(y, p, p >= threshold)
    record = {'manifest_sha256': signature, 'recipes': results, 'chosen': chosen,
              'component_diagnostics': component_scores, 'model_replay_max_errors': replay,
              'scope': 'Repeated development selection; not an independent test',
              'github_upload': False, 'kaggle_upload': False}
    freeze_json(OUT/'comparison.json', record)
    atomic_json(OUT/'progress.json', {'state': 'complete', 'completed_at': now(), 'chosen': chosen})
    print(json.dumps({k: v['combined'] for k,v in results.items()}, indent=2), flush=True)
    return record
