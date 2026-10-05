"""Compare fixed, equal-weight ensembles on development OOF only.

This is candidate selection, not an independent evaluation. Every tried pair
is retained; no audit predictions or audit labels are read.
"""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from mallorn.data import sha256
from mallorn.modeling import best_threshold, score
from mallorn.reporting import write_json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/improvement'


def compare(labels, name):
    split = json.loads((OUT / 'split.json').read_text())
    predictions, sources, target = {}, {}, None
    for label in labels:
        folder = OUT / label
        config = json.loads((folder / 'config.json').read_text())
        if config['fold_seed'] != 20261004 or config['folds'] != 5:
            raise ValueError('Incompatible OOF folds')
        frame = pd.read_csv(folder / 'oof.csv').set_index('object_id')
        if set(frame.index) != set(split['development']):
            raise ValueError('Only the exact development objects are allowed')
        frame = frame.loc[split['development']]
        y = frame.target.to_numpy()
        if target is None:
            target = y
        np.testing.assert_array_equal(y, target)
        sources[label] = {'oof_sha256': sha256(folder / 'oof.csv'), 'config': config}
        for column in config['candidates']:
            p = frame[column].to_numpy()
            if not (np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()):
                raise ValueError('Invalid probabilities')
            predictions[f'{label}/{column}'] = p
    results = []
    # Deliberately use equal weights, with no continuous weight optimizer.
    recipes = [[key] for key in predictions]
    recipes += list(itertools.combinations(predictions, 2))
    recipes += [[key for key in predictions if key.startswith(label + '/')]
                for label in labels]
    recipes += [list(predictions)]
    for members in recipes:
        p = np.mean([predictions[key] for key in members], axis=0)
        threshold, _ = best_threshold(target, p)
        results.append({'members': list(members), 'threshold': threshold,
                        **score(target, p, p >= threshold)})
    results.sort(key=lambda row: (row['f1'], row['average_precision']), reverse=True)
    folder = OUT / name
    folder.mkdir(exist_ok=True)
    write_json(folder / 'comparison.json', {
        'scope': 'Development OOF candidate and threshold selection; optimistic tuning scores',
        'sources': sources, 'candidates_evaluated': len(results), 'results': results})
    print(json.dumps({'candidates_evaluated': len(results), 'top': results[:8]}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--labels', nargs='+', required=True)
    parser.add_argument('--name', required=True)
    args = parser.parse_args()
    compare(args.labels, args.name)
