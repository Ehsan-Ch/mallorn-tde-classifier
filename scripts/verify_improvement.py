"""Check improvement artifacts and replay a selected model's first OOF fold."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from improve import fit_candidate, predict_candidate
from mallorn.data import sha256
from mallorn.reporting import write_json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/improvement'


def verify(replay):
    split = json.loads((OUT / 'split.json').read_text())
    dev = split['development']
    assert len(dev) == len(set(dev)) == 2282
    assert len(split['audit']) == len(set(split['audit'])) == 761
    assert not set(dev) & set(split['audit'])
    details = {'development_objects': len(dev), 'audit_objects': len(split['audit']),
               'audit_outcomes_read': False, 'schemas': {}, 'oof': {}}
    for view in ['physics', 'enhanced', 'complete', 'physical_view']:
        a = pd.read_csv(OUT / f'{view}_train.csv', index_col='object_id')
        b = pd.read_csv(OUT / f'{view}_test.csv', index_col='object_id')
        assert a.columns.equals(b.columns), view
        assert len(a) == 3043 and len(b) == 7135
        assert a.index.is_unique and b.index.is_unique
        assert not set(a.index) & set(b.index)
        assert not set(a.columns) & {'SpecType', 'target', 'Z_err', 'split', 'object_id'}
        assert not np.isinf(a.to_numpy()).any() and not np.isinf(b.to_numpy()).any()
        details['schemas'][view] = {'features': a.shape[1], 'test_objects': len(b)}
    for path in OUT.glob('*/oof.csv'):
        config = json.loads((path.parent / 'config.json').read_text())
        frame = pd.read_csv(path).set_index('object_id')
        assert frame.index.tolist() == dev
        p = frame[list(config['candidates'])].to_numpy()
        assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
        details['oof'][path.parent.name] = {'models': p.shape[1], 'sha256': sha256(path)}
    if replay:
        name = 'cat_aux3_d5_top125'
        folder = OUT / 'complete_selected'
        config = json.loads((folder / 'config.json').read_text())
        x = pd.read_csv(OUT / 'complete_train.csv', index_col='object_id').loc[dev]
        # Only development metadata is supplied to the fit.
        meta = pd.read_csv(ROOT / 'data/mallorn/train_log.csv').set_index('object_id').loc[dev]
        a, b = next(StratifiedKFold(5, shuffle=True, random_state=20261004).split(x, meta.target))
        fitted = fit_candidate(x.iloc[a], meta.iloc[a], config['candidates'][name], 42)
        expected_columns = json.loads((folder / f'{name}_fold0_columns.json').read_text())
        assert fitted['columns'] == expected_columns
        p = predict_candidate(fitted, x.iloc[b])
        expected = np.load(folder / f'{name}.npy')[b]
        np.testing.assert_allclose(p, expected, rtol=1e-10, atol=1e-12)
        details['replay'] = {'candidate': name, 'fold': 0, 'objects': len(b),
                             'max_probability_error': float(np.max(np.abs(p - expected)))}
    write_json(OUT / 'verification.json', details)
    print(json.dumps(details, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replay', action='store_true')
    verify(parser.parse_args().replay)
