from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest

import joblib
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from sklearn.metrics import f1_score

from mallorn.cli import prepare,run
from mallorn.data import load_dataset,load_groups,validate_log,validate_photometry,validate_submission
from mallorn.features import (FeatureConfig,band_features,dust_scale,extract_features,interpolate_observed)
from mallorn.fixtures import make_fixture
from mallorn.modeling import (TrainConfig,best_threshold,folds,predict_bundle,score,fit_model,predict_model,fit_candidates,predict_candidates)
from mallorn.reporting import bootstrap_interval


class IntegrityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary=tempfile.TemporaryDirectory()
        cls.root=Path(cls.temporary.name)
        make_fixture(cls.root,train_count=48,test_count=8)
        cls.train,cls.test,cls.photo,cls.manifest=load_dataset(cls.root)
        cls.metadata=pd.concat([cls.train,cls.test],ignore_index=True)
        cls.features=extract_features(cls.metadata,cls.photo)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_official_layout_and_hashes(self):
        self.assertEqual(self.manifest['train_objects'],48)
        self.assertEqual(self.manifest['positive_objects'],12)
        self.assertEqual(len(self.manifest['files']),7)
        self.assertTrue(all(len(f['sha256'])==64 for f in self.manifest['files']))

    def test_missing_competition_data_fails_clearly(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(FileNotFoundError,'official data'):
                load_dataset(Path(d))

    def test_duplicate_metadata_rejected(self):
        with self.assertRaises(ValueError):
            validate_log(pd.concat([self.train,self.train.iloc[[0]]]),True)

    def test_metadata_target_must_be_binary(self):
        bad=self.train.copy();bad.loc[0,'target']=2
        with self.assertRaises(ValueError):validate_log(bad,True)

    def test_invalid_redshift_rejected(self):
        bad=self.train.copy();bad.loc[0,'Z']=-1
        with self.assertRaises(ValueError):validate_log(bad,True)

    def test_target_not_required_for_test_log(self):
        validate_log(self.test,False)

    def test_duplicate_photometry_rejected(self):
        with self.assertRaisesRegex(ValueError,'Duplicate'):
            validate_photometry(pd.concat([self.photo,self.photo.iloc[[0]]]))

    def test_nonpositive_uncertainty_rejected(self):
        bad=self.photo.copy();bad.loc[0,'Flux_err']=0
        with self.assertRaises(ValueError):validate_photometry(bad)

    def test_nonfinite_flux_rejected(self):
        bad=self.photo.copy();bad.loc[0,'Flux']=np.inf
        with self.assertRaises(ValueError):validate_photometry(bad)

    def test_unknown_filter_rejected(self):
        bad=self.photo.copy();bad.loc[0,'Filter']='X'
        with self.assertRaises(ValueError):validate_photometry(bad)

    def test_negative_flux_allowed(self):
        bad=self.photo.iloc[:20].copy();bad['Flux']=-2.
        self.assertTrue((validate_photometry(bad).Flux<0).all())

    def test_features_exclude_label_metadata(self):
        forbidden={'target','SpecType','split','Z_err','English Translation','object_id'}
        self.assertFalse(forbidden & set(self.features.columns))
        altered=self.metadata.copy()
        altered['target']=1-altered['target'];altered['SpecType']='leak';altered['split']='changed'
        assert_frame_equal(self.features,extract_features(altered,self.photo))

    def test_feature_order_invariant(self):
        shuffled=self.photo.sample(frac=1,random_state=19)
        assert_frame_equal(self.features,extract_features(self.metadata,shuffled))

    def test_object_features_do_not_depend_on_other_objects(self):
        meta=self.metadata.iloc[[4]];photo=self.photo[self.photo.object_id.eq(meta.iloc[0].object_id)]
        assert_frame_equal(self.features.loc[meta.object_id],extract_features(meta,photo))

    def test_missing_band_has_stable_columns(self):
        self.assertEqual(self.features.loc['synthetic_train_0000','u__n'],0)
        self.assertTrue(np.isnan(self.features.loc['synthetic_train_0000','u__max']))
        self.assertFalse(np.isinf(self.features.to_numpy()).any())

    def test_dust_scale_preserves_snr(self):
        self.assertAlmostEqual(dust_scale(0,'g'),1)
        factor=dust_scale(.1,'g')
        self.assertGreater(factor,1)
        self.assertAlmostEqual((5*factor)/(2*factor),2.5)

    def test_interpolation_no_extrapolation_or_season_bridge(self):
        t=np.array([0.,10.,200.]);f=np.array([10.,20.,30.]);e=np.ones(3)
        self.assertEqual(interpolate_observed(t,f,e,5,60,2),15)
        self.assertTrue(np.isnan(interpolate_observed(t,f,e,-1,60,2)))
        self.assertTrue(np.isnan(interpolate_observed(t,f,e,50,60,2)))

    def test_interpolation_requires_both_endpoints_detected(self):
        self.assertTrue(np.isnan(interpolate_observed(np.array([0.,10.]),np.array([-1.,10.]),np.ones(2),5,60,2)))

    def test_rest_frame_duration(self):
        t=np.array([0.,10.,20.]);f=np.array([10.,20.,10.]);e=np.ones(3)
        result=band_features(t,f,e,10,1)
        self.assertEqual(result['span'],10)
        self.assertEqual(result['width_snr5'],10)

    def test_phase_features_are_invariant_to_calendar_offset(self):
        shifted=self.photo.copy();shifted['Time (MJD)']+=1000
        np.testing.assert_allclose(self.features,extract_features(self.metadata,shifted),rtol=1e-7,atol=1e-7,equal_nan=True)

    def test_threshold_matches_bruteforce(self):
        y=np.array([0,1,0,1,1,0]);p=np.array([.1,.4,.4,.6,.8,.9])
        threshold,value=best_threshold(y,p)
        self.assertAlmostEqual(value,max(f1_score(y,p>=t) for t in np.unique(p)))
        self.assertAlmostEqual(value,f1_score(y,p>=threshold))

    def test_threshold_ties_keep_equal_probabilities_together(self):
        threshold,value=best_threshold([0,1,0,1],[.3,.3,.3,.3])
        self.assertAlmostEqual(threshold,.3)
        self.assertAlmostEqual(value,2/3)

    def test_threshold_rejects_bad_probabilities(self):
        for p in ([.1,np.nan],[-.1,.8],[.1,1.1]):
            with self.assertRaises(ValueError):best_threshold([0,1],p)

    def test_group_folds_disjoint(self):
        y=np.tile([0,1],24);groups=np.repeat(np.arange(24),2)
        for a,b in folds(y,groups,3,42):
            self.assertFalse(set(groups[a]) & set(groups[b]))

    def test_insufficient_minority_fails(self):
        with self.assertRaises(ValueError):folds(np.array([0,0,0,1]),None,3,42)

    def test_group_mapping_reorders_by_object(self):
        ids=self.train.object_id
        mapping=pd.DataFrame({'object_id':ids,'group_id':[f'g{i//2}' for i in range(len(ids))]})
        path=self.root/'groups.csv';mapping.iloc[::-1].to_csv(path,index=False)
        actual=load_groups(path,ids)
        self.assertEqual(actual[0],'g0');self.assertEqual(actual[-1],'g23')

    def test_submission_exact_ids_and_binary_values(self):
        good=pd.DataFrame({'object_id':self.test.object_id,'prediction':0})
        validate_submission(good,self.test.object_id)
        bad=good.copy();bad['prediction']=.5
        with self.assertRaises(ValueError):validate_submission(bad,self.test.object_id)
        with self.assertRaises(ValueError):validate_submission(good.iloc[:-1],self.test.object_id)

    def test_confusion_matrix_and_f1(self):
        actual=score([0,0,1,1],[.1,.9,.2,.8],[0,1,0,1])
        self.assertEqual(actual['confusion_matrix_tn_fp_fn_tp'],[1,1,1,1])
        self.assertEqual(actual['f1'],.5)

    def test_bootstrap_is_repeatable_and_bounded(self):
        a=bootstrap_interval([0,1,1,0],[0,1,0,0],repeats=30)
        b=bootstrap_interval([0,1,1,0],[0,1,0,0],repeats=30)
        self.assertEqual(a,b)
        self.assertTrue(0<=a['ci95_low']<=a['ci95_high']<=1)

    def test_lightgbm_fits_a_nonconstant_model_and_replays(self):
        rng=np.random.default_rng(94)
        x=pd.DataFrame({'g__mean':rng.normal(size=160),'color_g_r__p0':rng.normal(size=160)})
        y=(x['g__mean']>0).astype(int).to_numpy()
        model=fit_model('lgb_full',x,y,TrainConfig(iterations=20,threads=1),42)
        p=predict_model(model,x)
        self.assertGreater(np.ptp(p),.1)
        path=self.root/'temporary_test_model.joblib';joblib.dump(model,path)
        np.testing.assert_allclose(p,predict_model(joblib.load(path),x),rtol=0,atol=0)

    @unittest.skipUnless(os.environ.get('MALLORN_TEST_CATBOOST')=='1','CatBoost needs process-statistics access; enable on a compatible host')
    def test_optional_hybrid_backend_fit_and_blend(self):
        rng=np.random.default_rng(94)
        x=pd.DataFrame({'g__mean':rng.normal(size=160),'color_g_r__p0':rng.normal(size=160)})
        y=(x['g__mean']>0).astype(int).to_numpy()
        models=fit_candidates(x,y,TrainConfig(iterations=12,threads=1,backend='hybrid'),42)
        p=predict_candidates(models,x)
        self.assertGreater(np.ptp(p['cat_full']),.01)
        np.testing.assert_allclose(p['blend'],.5*p['cat_full']+.5*p['lgb_full'])


class EndToEndTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary=tempfile.TemporaryDirectory()
        cls.root=Path(cls.temporary.name)/'data';cls.output=Path(cls.temporary.name)/'run'
        make_fixture(cls.root,train_count=48,test_count=8)
        cls.metrics=run(cls.root,cls.output,TrainConfig(3,2,8,42,1),FeatureConfig())

    @classmethod
    def tearDownClass(cls):cls.temporary.cleanup()

    def test_run_is_labeled_synthetic(self):
        self.assertEqual(self.metrics['data_kind'],'synthetic_fixture')
        self.assertFalse((self.output/'submission.csv').exists())
        self.assertIn('NOT COMPETITION PERFORMANCE',(self.output/'RESULTS.md').read_text())

    def test_outer_and_inner_ids_do_not_leak(self):
        audit=json.loads((self.output/'fold_audit.json').read_text())
        tested=[]
        for fold in audit:
            train,test=set(fold['fit_ids']),set(fold['test_ids'])
            self.assertFalse(train&test);tested+=fold['test_ids']
            for inner in fold['inner_splits']:
                fit,val=set(inner['fit_ids']),set(inner['validation_ids'])
                self.assertFalse(fit&val)
                self.assertEqual(fit|val,train)
                self.assertFalse((fit|val)&test)
        self.assertEqual(len(tested),len(set(tested)))
        self.assertEqual(len(tested),48)

    def test_oof_metrics_independently_recompute(self):
        oof=pd.read_csv(self.output/'oof.csv')
        self.assertFalse(oof.isna().any().any())
        self.assertAlmostEqual(f1_score(oof.target,oof.selected_prediction),self.metrics['primary']['f1'])
        self.assertFalse(self.metrics['deployment']['tuning_f1_is_performance_estimate'])

    def test_saved_model_replays_and_submission_follows_sample_order(self):
        _,test,x,_=prepare(self.root,FeatureConfig())
        bundle=joblib.load(self.output/'model.joblib')
        p,pred=predict_bundle(bundle,x.loc[test.object_id])
        saved=pd.read_csv(self.output/'test_probabilities.csv')
        np.testing.assert_allclose(p,saved.probability,rtol=1e-10,atol=1e-10)
        submission=pd.read_csv(self.output/'synthetic_submission.csv')
        sample=pd.read_csv(self.root/'sample_submission.csv')
        self.assertEqual(submission.object_id.tolist(),sample.object_id.tolist())
        self.assertEqual(submission.prediction.tolist(),pd.Series(pred,index=test.object_id).loc[sample.object_id].tolist())

    def test_inference_schema_guard(self):
        _,test,x,_=prepare(self.root,FeatureConfig())
        with self.assertRaises(ValueError):
            predict_bundle(joblib.load(self.output/'model.joblib'),x.loc[test.object_id].iloc[:,:-1])

    def test_existing_run_is_not_overwritten(self):
        with self.assertRaises(FileExistsError):run(self.root,self.output,TrainConfig(),FeatureConfig())


if __name__=='__main__':unittest.main()
