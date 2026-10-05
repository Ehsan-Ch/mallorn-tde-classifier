"""Recovered cycle-5 method: training-only selection and three-seed averaging.

Reconstructed from the recorded implementation after a workspace rollback.
The original fitted cycle-5 checkpoints are not included.
"""
import numpy as np
from lightgbm import LGBMClassifier

PARAMS = dict(n_estimators=900, learning_rate=.025, num_leaves=9,
              min_child_samples=30, reg_lambda=10., colsample_bytree=.8,
              subsample=.85, subsample_freq=1, deterministic=True,
              force_col_wise=True, verbosity=-1)
FORBIDDEN = {'target', 'SpecType', 'Z_err', 'object_id'}


def labels_weights(meta, kind):
    y = meta.target.to_numpy()
    types = meta.SpecType.astype(str)
    if kind == 'fine5':
        labels = np.select([y == 1, types.eq('AGN'), types.eq('SN IIn'), types.str.startswith('SLSN')],
                           [4, 0, 2, 3], default=1)
        return labels, {0: 1, 1: 1, 2: 1, 3: 1, 4: 3}, 4
    if kind == 'aux':
        return np.where(y == 1, 2, np.where(types.eq('AGN'), 0, 1)), {0: 1, 1: 1, 2: 3}, 2
    raise ValueError('Unknown auxiliary target')


def select_columns(x, gain, limit=125, correlation=.95):
    if FORBIDDEN & set(x.columns):
        raise ValueError('Forbidden feature column')
    gain = np.asarray(gain, dtype=float)
    if gain.shape != (x.shape[1],) or not np.isfinite(gain).all() or (gain < 0).any():
        raise ValueError('Invalid importance vector')
    available = x.columns[x.nunique(dropna=True) > 1]
    if not len(available):
        raise ValueError('No variable fitting-fold features')
    values = x[available].fillna(x[available].median()).to_numpy(dtype=float)
    centered = values-values.mean(axis=0)
    unit = centered/np.maximum(np.linalg.norm(centered, axis=0), 1e-100)
    positions = {name: i for i, name in enumerate(available)}
    importance = dict(zip(x.columns, gain))
    selected, indices = [], []
    for name in sorted(available, key=lambda c: (-importance[c], c)):
        i = positions[name]
        if indices and np.any(np.abs(unit[:, indices].T @ unit[:, i]) >= correlation):
            continue
        selected.append(name)
        indices.append(i)
        if len(selected) >= limit:
            break
    return selected


def fit(x, meta, kind, seed, threads):
    if FORBIDDEN & set(x.columns) or not x.index.equals(meta.index):
        raise ValueError('Invalid training features or row alignment')
    labels, weights, positive = labels_weights(meta, kind)
    gains = []
    for offset in (10000, 20000):
        selector = LGBMClassifier(**dict(PARAMS, n_estimators=600,
            random_state=seed+offset, n_jobs=threads, class_weight=weights, importance_type='gain'))
        gain = selector.fit(x, labels).feature_importances_.astype(float)
        gains.append(gain/max(gain.sum(), 1e-100))
    columns = select_columns(x, np.mean(gains, axis=0))
    models = [LGBMClassifier(**dict(PARAMS, random_state=seed+offset,
        n_jobs=threads, class_weight=weights)).fit(x[columns], labels) for offset in (0, 1000, 2000)]
    return {'models': models, 'selected_columns': columns, 'positive': positive}


def predict(fitted, x):
    return np.mean([model.predict_proba(x[fitted['selected_columns']])[:,
        list(model.classes_).index(fitted['positive'])] for model in fitted['models']], axis=0)
