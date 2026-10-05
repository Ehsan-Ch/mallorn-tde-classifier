"""Freeze a development-selected recipe, then run its one-time reserved audit.

Separate commands make the freeze reviewable before outcome inspection. A
started audit can resume only with the identical recipe and implementation.
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from improve import fit_candidate, predict_candidate
from mallorn.data import sha256
from mallorn.modeling import score
from mallorn.reporting import bootstrap_interval, write_json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/improvement'
FEATURES = {
    'cached_features': 'artifacts/official_001/features.csv',
    'enhanced_features': 'artifacts/improvement/enhanced_train.csv',
    'physics_only': 'artifacts/improvement/physics_train.csv',
    'physics_advanced': 'artifacts/improvement/physics_train.csv',
    'complete_advanced': 'artifacts/improvement/complete_train.csv',
    'complete_selected': 'artifacts/improvement/complete_train.csv',
    'physical_focused': 'artifacts/improvement/physical_view_train.csv',
}


def now():
    return datetime.now(timezone.utc).isoformat()


def implementation_hashes():
    return {str(path.relative_to(ROOT)): sha256(path) for path in [
        ROOT / 'scripts/improve.py', Path(__file__).resolve(),
        ROOT / 'src/mallorn/modeling.py', ROOT / 'src/mallorn/reporting.py']}


def freeze(comparison_path, destination):
    if (OUT / 'audit/audit_started.json').exists():
        raise ValueError('The audit has started; no further recipe selection is allowed')
    comparison = json.loads(comparison_path.read_text())
    chosen = comparison['results'][0]
    components = []
    for member in chosen['members']:
        label, candidate = member.split('/')
        config = comparison['sources'][label]['config']
        path = FEATURES[label]
        if sha256(ROOT / path) != config['features_sha256']:
            raise ValueError('Feature data changed after the screening run')
        components.append({'name': member, 'features': path,
                           'features_sha256': config['features_sha256'],
                           'spec': config['candidates'][candidate], 'seed': 42})
    recipe = {
        'frozen_at': now(), 'selection_rule': 'Highest development F1; AP tie-break',
        'comparison_sha256': sha256(comparison_path), 'development_score': chosen,
        'threshold': chosen['threshold'], 'components': components,
        'ensemble': 'Equal arithmetic mean of component probabilities',
        'training': 'Each component refitted on all development objects, seed 42',
        'split_sha256': sha256(OUT / 'split.json'),
        'metadata_sha256': sha256(ROOT / 'data/mallorn/train_log.csv'),
        'implementation_sha256': implementation_hashes(),
        'limitations': [
            'Development selection scores are optimistic after model and threshold search.',
            'The audit was reserved for this cycle but all labels were used by the old baseline.',
            'No genuine source-template groups are available.',
            'Audit F1 cannot establish equivalence to the Kaggle private leaderboard.'],
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('x') as handle:
        json.dump(recipe, handle, indent=2)
        handle.write('\n')
    print('FROZEN', destination, sha256(destination))


def audit(recipe_path):
    recipe = json.loads(recipe_path.read_text())
    if recipe['implementation_sha256'] != implementation_hashes():
        raise ValueError('Implementation changed after the recipe freeze')
    if recipe['split_sha256'] != sha256(OUT / 'split.json'):
        raise ValueError('Reserved split changed')
    if recipe['metadata_sha256'] != sha256(ROOT / 'data/mallorn/train_log.csv'):
        raise ValueError('Training metadata changed')
    folder = OUT / 'audit'
    folder.mkdir(exist_ok=True)
    if (folder / 'metrics.json').exists():
        raise ValueError('Audit already completed; read the recorded result without re-tuning')
    receipt = folder / 'audit_started.json'
    if receipt.exists():
        if json.loads(receipt.read_text())['recipe_sha256'] != sha256(recipe_path):
            raise ValueError('A different recipe has already started the audit')
    else:
        with receipt.open('x') as handle:
            json.dump({'started_at': now(), 'recipe_sha256': sha256(recipe_path)}, handle, indent=2)
    split = json.loads((OUT / 'split.json').read_text())
    dev, held = split['development'], split['audit']
    if set(dev) & set(held):
        raise ValueError('Development/audit overlap')
    meta = pd.read_csv(ROOT / 'data/mallorn/train_log.csv').set_index('object_id')
    probabilities = []
    for i, component in enumerate(recipe['components']):
        path = ROOT / component['features']
        if sha256(path) != component['features_sha256']:
            raise ValueError('Frozen feature bytes changed')
        x = pd.read_csv(path, index_col='object_id')
        if any(c in x for c in ('target', 'SpecType', 'Z_err')):
            raise ValueError('Forbidden predictor column')
        saved = folder / f'component_{i}.joblib'
        if saved.exists():
            fitted = joblib.load(saved)
        else:
            fitted = fit_candidate(x.loc[dev], meta.loc[dev], component['spec'], component['seed'])
            joblib.dump(fitted, saved, compress=3)
        p = predict_candidate(fitted, x.loc[held])
        np.save(folder / f'component_{i}_audit.npy', p)
        probabilities.append(p)
        print('FITTED', component['name'], flush=True)
    p = np.mean(probabilities, axis=0)
    decisions = p >= recipe['threshold']
    # First access to reserved outcomes occurs only after every prediction is fixed.
    y = meta.loc[held, 'target'].to_numpy()
    result = score(y, p, decisions)
    interval = bootstrap_interval(y, decisions, seed=20261005, repeats=5000)
    interval['scope'] = 'Conditional on fixed audit decisions; no refitting or source-group uncertainty'
    metrics = {'completed_at': now(), 'recipe_sha256': sha256(recipe_path),
               'threshold': recipe['threshold'], 'audit': result,
               'conditional_f1_interval': interval, 'limitations': recipe['limitations']}
    pd.DataFrame({'object_id': held, 'target': y, 'probability': p,
                  'prediction': decisions.astype(int)}).to_csv(folder / 'predictions.csv', index=False)
    write_json(folder / 'metrics.json', metrics)
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    f = commands.add_parser('freeze')
    f.add_argument('--comparison', type=Path, required=True)
    f.add_argument('--output', type=Path, default=OUT / 'frozen_recipe.json')
    a = commands.add_parser('audit')
    a.add_argument('--recipe', type=Path, default=OUT / 'frozen_recipe.json')
    args = parser.parse_args()
    if args.command == 'freeze':
        freeze(args.comparison, args.output)
    else:
        audit(args.recipe)
