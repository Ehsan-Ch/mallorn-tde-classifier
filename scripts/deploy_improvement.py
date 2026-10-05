"""Refit an audited recipe and prepare local predictions; never upload them."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from audit_improvement import implementation_hashes
from improve import fit_candidate, predict_candidate
from mallorn.data import sha256
from mallorn.reporting import write_json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/improvement'


def deploy(recipe_path, predict_only):
    recipe = json.loads(recipe_path.read_text())
    audit = json.loads((OUT / 'audit/metrics.json').read_text())
    if audit['recipe_sha256'] != sha256(recipe_path):
        raise ValueError('Audit does not belong to this frozen recipe')
    if implementation_hashes() != recipe['implementation_sha256']:
        raise ValueError('Fitting implementation changed after the audit')
    folder = OUT / 'deployment'
    folder.mkdir(exist_ok=True)
    receipt = {'recipe_sha256': sha256(recipe_path),
               'training_scope': 'All 3043 labeled competition objects after the one-time audit'}
    saved_receipt = folder / 'recipe.json'
    if saved_receipt.exists() and json.loads(saved_receipt.read_text()) != receipt:
        raise ValueError('Deployment directory already belongs to a different recipe')
    write_json(saved_receipt, receipt)
    meta = pd.read_csv(ROOT / 'data/mallorn/train_log.csv').set_index('object_id')
    test = pd.read_csv(ROOT / 'data/mallorn/test_log.csv').set_index('object_id')
    template = pd.read_csv(ROOT / 'data/mallorn/sample_submission.csv')
    if template.columns.tolist() != ['object_id', 'prediction']:
        raise ValueError('Unexpected official submission schema')
    probabilities, artifacts = [], []
    for i, component in enumerate(recipe['components']):
        source = ROOT / component['features']
        if sha256(source) != component['features_sha256']:
            raise ValueError('Training features changed after the recipe freeze')
        test_source = Path(str(source).replace('_train.csv', '_test.csv'))
        train_x = pd.read_csv(source, index_col='object_id').loc[meta.index]
        test_x = pd.read_csv(test_source, index_col='object_id').loc[test.index]
        if not train_x.columns.equals(test_x.columns):
            raise ValueError('Train and inference feature schemas differ')
        model_path = folder / f'component_{i}.joblib'
        if model_path.exists():
            fitted = joblib.load(model_path)
        elif predict_only:
            raise ValueError('Predict-only requires all fitted component artifacts')
        else:
            fitted = fit_candidate(train_x, meta, component['spec'], component['seed'])
            joblib.dump(fitted, model_path, compress=3)
        probabilities.append(predict_candidate(fitted, test_x))
        artifacts.append({'component': component['name'], 'model_sha256': sha256(model_path),
                          'test_features_sha256': sha256(test_source),
                          'features_used': len(fitted['columns'])})
        print('READY', component['name'], flush=True)
    p = np.mean(probabilities, axis=0)
    assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
    predictions = (p >= recipe['threshold']).astype(int)
    submission = pd.DataFrame({'object_id': test.index, 'prediction': predictions})
    submission = submission.set_index('object_id').loc[template.object_id].reset_index()
    assert len(submission) == 7135 and submission.object_id.is_unique
    path = folder / 'submission.csv'
    if predict_only:
        pd.testing.assert_frame_equal(pd.read_csv(path), submission)
        np.testing.assert_allclose(np.load(folder / 'probabilities.npy'), p, rtol=0, atol=1e-12)
    else:
        submission.to_csv(path, index=False)
        np.save(folder / 'probabilities.npy', p)
    result = {**receipt, 'threshold': recipe['threshold'], 'test_objects': len(submission),
              'predicted_positive': int(predictions.sum()), 'components': artifacts,
              'submission_sha256': sha256(path), 'prediction_replay_checked': predict_only,
              'published_to_github': False, 'submitted_to_kaggle': False,
              'notice': 'Local submission candidate; no organizer test-set score is available'}
    write_json(folder / 'manifest.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recipe', type=Path, default=OUT / 'frozen_recipe.json')
    parser.add_argument('--predict-only', action='store_true')
    args = parser.parse_args()
    deploy(args.recipe, args.predict_only)
