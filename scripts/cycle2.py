"""Fixed second-cycle candidates with repeated, resumable object-level CV."""
import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from improve import candidate_specs, fit_candidate, predict_candidate
from mallorn.data import sha256
from mallorn.modeling import best_threshold, score
from mallorn.reporting import write_json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/cycle2'
COMPONENTS = {
    'physics_aux': ('physics', candidate_specs('advanced')['cat_aux3_d5']),
    'selected_binary': ('complete', candidate_specs('selected')['cat_d4_w3_top125']),
    'selected_aux': ('complete', candidate_specs('selected')['cat_aux3_d5_top125']),
    'hierarchical': ('physical_view', candidate_specs('physical')['cat_hierarchy_d4']),
}
RECIPES = {
    'reference': ['physics_aux', 'selected_binary'],
    'selected_class': ['selected_binary', 'selected_aux'],
    'diverse': list(COMPONENTS),
}


def inputs(name):
    view, spec = COMPONENTS[name]
    path = ROOT / f'artifacts/improvement/{view}_train.csv'
    meta = pd.read_csv(ROOT / 'data/mallorn/train_log.csv').set_index('object_id')
    x = pd.read_csv(path, index_col='object_id').loc[meta.index]
    assert not set(x.columns) & {'target', 'SpecType', 'Z_err', 'object_id'}
    return x, meta, spec, path


def screen(name):
    OUT.mkdir(exist_ok=True)
    folder = OUT / name
    folder.mkdir(exist_ok=True)
    x, meta, spec, path = inputs(name)
    config = {'name': name, 'spec': spec, 'features_sha256': sha256(path),
              'metadata_sha256': sha256(ROOT / 'data/mallorn/train_log.csv'),
              'fit_code_sha256': sha256(ROOT / 'scripts/improve.py'),
              'split_seeds': [20261014, 20261015], 'folds': 5, 'objects': len(x)}
    saved_config = folder / 'config.json'
    if saved_config.exists() and json.loads(saved_config.read_text()) != config:
        raise ValueError('Configuration changed; use a separate experiment')
    write_json(saved_config, config)
    y = meta.target.to_numpy()
    for repeat, seed in enumerate(config['split_seeds']):
        file = folder / f'oof_repeat{repeat}.npy'
        p = np.load(file) if file.exists() else np.full(len(y), np.nan)
        start = time.monotonic()
        for fold, (a, b) in enumerate(StratifiedKFold(5, shuffle=True, random_state=seed).split(x, y)):
            if np.isfinite(p[b]).all():
                continue
            fitted = fit_candidate(x.iloc[a], meta.iloc[a], spec, 42 + 100 * repeat + fold)
            p[b] = predict_candidate(fitted, x.iloc[b])
            write_json(folder / f'repeat{repeat}_fold{fold}.json', {
                'fitting_ids': x.index[a].tolist(), 'validation_ids': x.index[b].tolist(),
                'columns': fitted['columns'], 'seed': 42 + 100 * repeat + fold})
            temp = folder / 'checkpoint.npy'
            np.save(temp, p)
            temp.replace(file)
            print(name, 'repeat', repeat, 'fold', fold, 'seconds', round(time.monotonic()-start, 1), flush=True)
        assert np.isfinite(p).all()
        threshold, _ = best_threshold(y, p)
        print(name, 'repeat', repeat, json.dumps(score(y, p, p >= threshold)), flush=True)


def stable_threshold(y, p):
    rng = np.random.default_rng(20261016)
    classes = [np.flatnonzero(y == k) for k in [0, 1]]
    thresholds = []
    for _ in range(500):
        idx = np.concatenate([rng.choice(a, len(a), replace=True) for a in classes])
        thresholds.append(best_threshold(y[idx], p[idx])[0])
    return float(np.median(thresholds)), np.quantile(thresholds, [.1, .9]).tolist()


def compare():
    _, meta, _, _ = inputs('physics_aux')
    y = meta.target.to_numpy()
    ids = set(meta.index)
    verification = {}
    for name in COMPONENTS:
        config = json.loads((OUT/name/'config.json').read_text())
        assert config['metadata_sha256'] == sha256(ROOT/'data/mallorn/train_log.csv')
        assert config['fit_code_sha256'] == sha256(ROOT/'scripts/improve.py')
        for repeat in range(2):
            seen = []
            for fold in range(5):
                audit = json.loads((OUT/name/f'repeat{repeat}_fold{fold}.json').read_text())
                a, b = set(audit['fitting_ids']), set(audit['validation_ids'])
                assert not a & b and a | b == ids
                assert not set(audit['columns']) & {'target', 'SpecType', 'Z_err', 'object_id'}
                seen.extend(audit['validation_ids'])
            assert len(seen) == len(ids) and set(seen) == ids
        verification[name] = {'repetitions': 2, 'folds_per_repetition': 5,
                              'validation_predictions_per_object': 2,
                              'fitting_validation_overlap': 0}
    write_json(OUT/'verification.json', verification)
    predictions = {name: np.stack([np.load(OUT/name/f'oof_repeat{r}.npy') for r in range(2)])
                   for name in COMPONENTS}
    assert all(p.shape == (2, len(y)) and np.isfinite(p).all() for p in predictions.values())
    result = {}
    for name, members in RECIPES.items():
        repeated = np.mean([predictions[m] for m in members], axis=0)
        p = repeated.mean(axis=0)
        threshold, interval = stable_threshold(y, p)
        tuned_threshold, tuned_f1 = best_threshold(y, p)
        result[name] = {'members': members, 'threshold': threshold,
                        'bootstrap_threshold_10_90': interval,
                        'combined': score(y, p, p >= threshold),
                        'repeat_scores_at_fixed_threshold': [score(y, q, q >= threshold) for q in repeated],
                        'unsmoothed_tuning_threshold': tuned_threshold,
                        'unsmoothed_tuning_f1': tuned_f1}
    reference = result['reference']['combined']
    old_threshold = json.loads((ROOT/'artifacts/improvement/frozen_recipe.json').read_text())['threshold']
    old_recipe_repeated = np.mean([predictions[m] for m in RECIPES['reference']], axis=0)
    reference_at_old_threshold = {
        'threshold': old_threshold,
        'combined': score(y, old_recipe_repeated.mean(axis=0),
                          old_recipe_repeated.mean(axis=0) >= old_threshold),
        'individual_repetitions': [score(y, p, p >= old_threshold) for p in old_recipe_repeated],
    }
    eligible = [k for k in result if result[k]['combined']['f1'] >= reference['f1'] + .01
                and result[k]['combined']['average_precision'] >= reference['average_precision'] - .005]
    chosen = max(eligible, key=lambda k: result[k]['combined']['f1']) if eligible else 'reference'
    record = {'scope': 'Repeated local CV selection; all historical labels reused, no pristine audit',
              'recipes': result, 'chosen': chosen, 'deployment_seeds': [42, 142],
              'reference_at_previously_submitted_threshold': reference_at_old_threshold,
              'code_sha256': sha256(Path(__file__)),
              'component_configs': {k: json.loads((OUT/k/'config.json').read_text()) for k in COMPONENTS}}
    path = OUT / 'frozen_comparison.json'
    if path.exists() and json.loads(path.read_text()) != record:
        raise ValueError('Frozen comparison already exists with different values')
    write_json(path, record)
    receipt = OUT/'freeze_receipt.json'
    if not receipt.exists():
        write_json(receipt, {'frozen_at': datetime.now(timezone.utc).isoformat(),
                            'frozen_comparison_sha256': sha256(path),
                            'kaggle_score_observed_for_this_recipe': False})
    print(json.dumps(record, indent=2))


def deploy():
    file = OUT / 'frozen_comparison.json'
    record = json.loads(file.read_text())
    if record['code_sha256'] != sha256(Path(__file__)):
        raise ValueError('Implementation changed after freeze')
    recipe = record['recipes'][record['chosen']]
    template = pd.read_csv(ROOT / 'data/mallorn/sample_submission.csv')
    assert template.columns.tolist() == ['object_id', 'prediction']
    folder = OUT / 'deployment'
    folder.mkdir(exist_ok=True)
    all_p = []
    for name in recipe['members']:
        x, meta, spec, train_path = inputs(name)
        if sha256(train_path) != record['component_configs'][name]['features_sha256']:
            raise ValueError('Feature data changed')
        test_path = Path(str(train_path).replace('_train.csv', '_test.csv'))
        test_x = pd.read_csv(test_path, index_col='object_id').loc[template.object_id]
        assert x.columns.equals(test_x.columns)
        for seed in record['deployment_seeds']:
            path = folder / f'{name}_{seed}.joblib'
            if path.exists():
                fitted = joblib.load(path)
            else:
                fitted = fit_candidate(x, meta, spec, seed)
                joblib.dump(fitted, path, compress=3)
            p = predict_candidate(fitted, test_x)
            np.save(folder / f'{name}_{seed}.npy', p)
            all_p.append(p)
            print('FITTED', name, seed, flush=True)
    p = np.mean(all_p, axis=0)
    assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
    result = template.copy()
    result.prediction = (p >= recipe['threshold']).astype(int)
    assert len(result) == 7135 and result.object_id.is_unique
    result.to_csv(folder / 'submission.csv', index=False)
    np.save(folder / 'probabilities.npy', p)
    write_json(folder / 'manifest.json', {'recipe': record['chosen'],
        'frozen_comparison_sha256': sha256(file), 'submission_sha256': sha256(folder/'submission.csv'),
        'threshold': recipe['threshold'], 'positive_predictions': int(result.prediction.sum()),
        'objects': len(result), 'submitted_to_kaggle': False})
    print((folder / 'manifest.json').read_text())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['screen', 'compare', 'deploy'])
    parser.add_argument('--component', choices=list(COMPONENTS))
    args = parser.parse_args()
    if args.action == 'screen':
        if args.component is None:
            parser.error('--component is required for screening')
        screen(args.component)
    elif args.action == 'compare':
        compare()
    else:
        deploy()
