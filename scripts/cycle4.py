"""Confusing-negative experiments; reuse the tested checkpoint engine unchanged."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
from lightgbm import LGBMClassifier

from mallorn.checkpointing import atomic_json, freeze_json
from mallorn.data import sha256
import cycle3

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'artifacts/cycle4'
COMPONENTS = {'fine5': ('complete', 'fine5'),
              'hard_negative_binary': ('complete', 'hard_negative_binary')}


def fit(x, meta, kind, seed, threads):
    y = meta.target.to_numpy()
    types = meta.SpecType.astype(str)
    params = dict(cycle3.PARAMS, random_state=seed, n_jobs=threads)
    if kind == 'fine5':
        # Target takes precedence; all mappings use only this fitting fold.
        labels = np.select([y == 1, types.eq('AGN'), types.eq('SN IIn'), types.str.startswith('SLSN')],
                           [4, 0, 2, 3], default=1)
        model = LGBMClassifier(**params, class_weight={0: 1, 1: 1, 2: 1, 3: 1, 4: 3})
        return {'model': model.fit(x, labels), 'positive': 4}
    weights = np.where(y == 1, 3., np.where(types.eq('SN IIn') | types.str.startswith('SLSN'), 2., 1.))
    model = LGBMClassifier(**params)
    return {'model': model.fit(x, y, sample_weight=weights), 'positive': 1}


def engine():
    """An isolated module instance keeps archived cycle-3 globals unchanged.

    The same validated fold/checkpoint/comparison implementation is configured
    for a new experiment. Its source hash is included in this cycle's manifest.
    """
    spec = importlib.util.spec_from_file_location('mallorn_cycle4_engine', ROOT/'scripts/cycle3.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.OUT = OUT
    module.COMPONENTS = COMPONENTS
    module.fit = fit
    # Both candidates (including the engine's initial metadata read) use the
    # complete feature view; preserve cycle-3 input validation and row ordering.
    module.inputs = lambda name: cycle3.inputs('lgb_complete_binary')
    return module


def initialize(base_manifest):
    # Do not call the cycle-3 initializer with different globals: it owns a
    # distinct frozen protocol. Include every dependency of the reused engine.
    manifest = json.loads(json.dumps(base_manifest))
    manifest['experiment'] = 'cycle4'
    manifest['components'] = {k: list(v) for k,v in COMPONENTS.items()}
    manifest['files']['scripts/cycle4.py'] = sha256(Path(__file__))
    manifest['files']['docs/FOURTH_CYCLE_PROTOCOL.md'] = sha256(ROOT/'docs/FOURTH_CYCLE_PROTOCOL.md')
    manifest['files']['artifacts/cycle3/comparison.json'] = sha256(ROOT/'artifacts/cycle3/comparison.json')
    freeze_json(OUT/'manifest.json', manifest)
    if not (OUT/'start_receipt.json').exists():
        atomic_json(OUT/'start_receipt.json', {'started_at': cycle3.now(),
                    'manifest_sha256': sha256(OUT/'manifest.json')})
    return manifest
