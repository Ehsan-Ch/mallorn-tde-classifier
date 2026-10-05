import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import joblib
import numpy as np
import pandas as pd
from mallorn import selected_ensemble as s


class RecoveredSelectionTests(unittest.TestCase):
    def test_correlation_pruning_and_empty_columns(self):
        x = pd.DataFrame({'a':np.arange(20.),'duplicate':-2*np.arange(20.),
                          'b':np.tile([0.,1.],10),'empty':np.nan,'constant':1.})
        self.assertEqual(s.select_columns(x,[10,9,8,100,100]),['a','b'])

    def test_forbidden_labels(self):
        with self.assertRaises(ValueError): s.select_columns(pd.DataFrame({'target':[0,1]}),[1])

    def test_auxiliary_target_precedence(self):
        meta = pd.DataFrame({'target':[1,0,0,0,0],'SpecType':['AGN','AGN','SN IIn','SLSN-I','SN Ia']})
        np.testing.assert_array_equal(s.labels_weights(meta,'fine5')[0],[4,0,2,3,1])

    def test_training_boundary_and_replay(self):
        rng=np.random.default_rng(7)
        x=pd.DataFrame(rng.normal(size=(100,8)),columns=[f'x{i}' for i in range(8)])
        meta=pd.DataFrame({'target':(x.x0>.8).astype(int),'SpecType':np.where(x.x1>0,'AGN','SN Ia')})
        held=pd.DataFrame(rng.normal(size=(15,8)),columns=x.columns,index=np.arange(100,115))
        seen=[]
        original=s.select_columns
        def select(frame,gain):
            seen.extend(frame.index)
            return original(frame,gain)
        with patch.object(s,'select_columns',side_effect=select),patch.dict(s.PARAMS,{'n_estimators':8,'min_child_samples':5}):
            model=s.fit(x,meta,'aux',42,1)
        self.assertEqual(set(seen),set(x.index))
        self.assertFalse(set(seen)&set(held.index))
        p=s.predict(model,held)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'model.joblib'
            joblib.dump(model,path)
            np.testing.assert_array_equal(p,s.predict(joblib.load(path),held))
