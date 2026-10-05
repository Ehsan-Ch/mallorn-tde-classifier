"""Portable synthetic feature/model/replay check; no competition score claim."""
import argparse
from pathlib import Path
import tempfile
import joblib
from lightgbm import LGBMClassifier
import numpy as np
import pandas as pd
from mallorn.fixtures import make_fixture
from mallorn.data import load_dataset, validate_submission
from mallorn.features import extract_features
from mallorn.checkpointing import atomic_json, atomic_model


def run(output):
    if output.exists() and any(output.iterdir()):
        raise ValueError('Choose an empty output directory for this software check')
    output.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory() as temporary:
        root=Path(temporary);make_fixture(root)
        train,test,photo,_=load_dataset(root)
        x=extract_features(pd.concat([train,test],ignore_index=True),photo)
        assert not {'target','SpecType','Z_err'} & set(x.columns)
        model=LGBMClassifier(n_estimators=12,num_leaves=5,min_child_samples=5,
            n_jobs=1,random_state=42,verbosity=-1,deterministic=True,force_col_wise=True)
        model.fit(x.loc[train.object_id],train.target)
        p=model.predict_proba(x.loc[test.object_id])[:,1]
        assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
        atomic_model(output/'synthetic_model.joblib',model)
        np.testing.assert_array_equal(p,joblib.load(output/'synthetic_model.joblib').predict_proba(x.loc[test.object_id])[:,1])
        frame=pd.DataFrame({'object_id':test.object_id,'prediction':(p>=.5).astype(int)})
        validate_submission(frame,test.object_id)
        frame.to_csv(output/'synthetic_submission.csv',index=False)
        atomic_json(output/'software_check.json',{'data_kind':'synthetic_fixture',
            'not_competition_performance':True,'training_objects':len(train),'test_objects':len(test),
            'feature_count':x.shape[1],'model_replay_exact':True,'submission_schema_checked':True})
        print('Synthetic feature extraction, LightGBM fit, saved-model replay and schema check passed.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args().output)
