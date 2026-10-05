"""Nested object-level CV: model AND F1 threshold chosen inside each outer fold."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.impute import SimpleImputer
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline

from .features import basic_columns


class PipelineCatBoostClassifier(ClassifierMixin, CatBoostClassifier, BaseEstimator):
    """Supply sklearn estimator tags for the pinned CatBoost 1.2.8 pipeline.

    CatBoost's native fitting is unchanged. sklearn 1.8 requires the public
    estimator tags API; the mixins follow sklearn's documented MRO ordering.
    """


@dataclass(frozen=True)
class TrainConfig:
    outer_folds: int = 5
    inner_folds: int = 3
    iterations: int = 500
    seed: int = 42
    threads: int = 2
    backend: str = "lightgbm"

    def validate(self):
        if min(self.outer_folds, self.inner_folds) < 2 or self.iterations < 1 or self.threads < 1:
            raise ValueError("Require folds>=2, iterations>=1, threads>=1")
        if self.backend not in ("lightgbm","hybrid"):
            raise ValueError("backend must be lightgbm or hybrid")


def base_models(cfg):
    return ("lgb_basic","lgb_full") + (("cat_full",) if cfg.backend=="hybrid" else ())


def candidate_names(cfg):
    return ("prior",*base_models(cfg)) + (("blend",) if cfg.backend=="hybrid" else ())


def check_probability(y, p):
    y, p = np.asarray(y), np.asarray(p, dtype=float)
    if y.ndim != 1 or p.shape != y.shape or not len(y):
        raise ValueError("Mismatched or empty probability vectors")
    if not np.isin(y, [0,1]).all() or not np.isfinite(p).all() or ((p<0)|(p>1)).any():
        raise ValueError("Expected binary labels and finite probabilities in [0,1]")
    return y, p


def best_threshold(y, p) -> tuple[float, float]:
    y, p = check_probability(y, p)
    if len(np.unique(y)) != 2:
        raise ValueError("Threshold fitting requires both classes")
    precision, recall, thresholds = precision_recall_curve(y, p)
    f1 = 2*precision[:-1]*recall[:-1]/np.maximum(precision[:-1]+recall[:-1], 1e-15)
    # Deterministic tie break: largest threshold, i.e. fewest positive calls.
    best = np.flatnonzero(np.isclose(f1, np.max(f1), atol=1e-12, rtol=0))[-1]
    threshold = float(thresholds[best])
    return threshold, float(f1_score(y, p>=threshold, zero_division=0))


def score(y, p, predictions) -> dict:
    y, p = check_probability(y, p)
    predictions = np.asarray(predictions)
    if predictions.shape != y.shape or not np.isin(predictions,[0,1]).all():
        raise ValueError("Expected aligned binary decisions")
    return {"n": int(len(y)), "positives": int(y.sum()), "predicted_positives": int(predictions.sum()),
            "f1": float(f1_score(y,predictions,zero_division=0)),
            "precision": float(precision_score(y,predictions,zero_division=0)),
            "recall": float(recall_score(y,predictions,zero_division=0)),
            "average_precision": float(average_precision_score(y,p)),
            "roc_auc": float(roc_auc_score(y,p)) if len(np.unique(y))==2 else None,
            "confusion_matrix_tn_fp_fn_tp": confusion_matrix(y,predictions,labels=[0,1]).ravel().tolist()}


def folds(y, groups, count, seed):
    y = np.asarray(y)
    if min(np.bincount(y.astype(int), minlength=2)) < count:
        raise ValueError("Too few examples of each class for requested folds")
    if groups is None:
        iterator = StratifiedKFold(count,shuffle=True,random_state=seed).split(np.zeros(len(y)),y)
    else:
        groups = np.asarray(groups)
        if len(groups)!=len(y) or len(np.unique(groups))<count:
            raise ValueError("Insufficient or misaligned source groups")
        iterator = StratifiedGroupKFold(count,shuffle=True,random_state=seed).split(np.zeros(len(y)),y,groups)
    result = list(iterator)
    seen = np.zeros(len(y), dtype=int)
    for train, val in result:
        if set(train)&set(val) or len(np.unique(y[train]))!=2 or len(np.unique(y[val]))!=2:
            raise ValueError("Invalid fold: overlap or a missing class")
        if groups is not None and set(groups[train])&set(groups[val]):
            raise ValueError("Source group leakage")
        seen[val]+=1
    if not np.all(seen==1):
        raise ValueError("Each object must be held out once")
    return result


def fit_model(name, x, y, cfg: TrainConfig, seed):
    columns = basic_columns(x.columns) if name.endswith("_basic") else list(x.columns)
    if name.startswith("cat_"):
        estimator = PipelineCatBoostClassifier(iterations=cfg.iterations,depth=5,learning_rate=.04,
            l2_leaf_reg=6,loss_function="Logloss",random_seed=seed,thread_count=cfg.threads,
            allow_writing_files=False,verbose=False)
    elif name.startswith("lgb_"):
        estimator = LGBMClassifier(n_estimators=cfg.iterations,num_leaves=15,max_depth=-1,
            learning_rate=.035,min_child_samples=20,reg_lambda=5.,colsample_bytree=.85,
            random_state=seed,n_jobs=cfg.threads,verbosity=-1,deterministic=True,force_col_wise=True)
    else:
        raise ValueError(f"Unknown model {name}")
    # Every imputer (including its missing indicators) is fitted only on fit rows.
    imputer=SimpleImputer(strategy="median",add_indicator=True,keep_empty_features=True).set_output(transform="pandas")
    pipeline = make_pipeline(imputer,estimator)
    pipeline.fit(x.loc[:,columns], y)
    return {"columns":columns,"pipeline":pipeline}


def predict_model(model, x):
    return model["pipeline"].predict_proba(x.loc[:,model["columns"]])[:,1]


def fit_candidates(x, y, cfg, seed):
    return {"prior":float(np.mean(y)), **{name:fit_model(name,x,y,cfg,seed) for name in base_models(cfg)}}


def predict_candidates(models, x):
    result = {"prior":np.full(len(x),models["prior"])}
    result.update({name:predict_model(model,x) for name,model in models.items() if name!="prior"})
    if "cat_full" in result:
        result["blend"] = .5*result["cat_full"]+.5*result["lgb_full"]
    return result


def choose_candidate(y, probabilities):
    tuning = {name:dict(zip(("threshold","tuning_f1"),best_threshold(y,p))) for name,p in probabilities.items()}
    # Prefer the predeclared simpler candidate on an exact F1 tie.
    selected = max(probabilities,key=lambda name:tuning[name]["tuning_f1"])
    return selected,tuning


def nested_evaluate(x: pd.DataFrame, y, cfg: TrainConfig, groups=None):
    cfg.validate()
    y = np.asarray(y)
    if not np.isin(y,[0,1]).all():
        raise ValueError("Labels must be binary before conversion")
    y = y.astype(int)
    if len(y)!=len(x) or x.index.duplicated().any() or len(set(x.columns))!=len(x.columns):
        raise ValueError("Feature rows, IDs or columns invalid")
    if not np.isin(y,[0,1]).all() or np.isinf(x.to_numpy()).any():
        raise ValueError("Invalid labels or features")
    outer = folds(y,groups,cfg.outer_folds,cfg.seed)
    names=candidate_names(cfg)
    oof = {name:np.full(len(y),np.nan) for name in names}
    decisions = {name:np.zeros(len(y),dtype=int) for name in names}
    selected_p,selected_pred = np.full(len(y),np.nan),np.zeros(len(y),dtype=int)
    assignments, audits = np.full(len(y),-1), []
    for fold,(fit_idx,test_idx) in enumerate(outer):
        print(f"Outer fold {fold+1}/{cfg.outer_folds}: {len(fit_idx)} fit, {len(test_idx)} test",flush=True)
        fit_x,fit_y = x.iloc[fit_idx],y[fit_idx]
        fit_groups = np.asarray(groups)[fit_idx] if groups is not None else None
        inner = folds(fit_y,fit_groups,cfg.inner_folds,cfg.seed+fold+1)
        inner_p = {name:np.full(len(fit_y),np.nan) for name in names}
        inner_audit = []
        for inner_fold,(a,b) in enumerate(inner):
            models = fit_candidates(fit_x.iloc[a],fit_y[a],cfg,cfg.seed+fold*100+inner_fold)
            for name,p in predict_candidates(models,fit_x.iloc[b]).items():
                inner_p[name][b]=p
            inner_audit.append({"fit_ids":fit_x.index[a].tolist(),"validation_ids":fit_x.index[b].tolist()})
        selected,tuning = choose_candidate(fit_y,inner_p)
        models = fit_candidates(fit_x,fit_y,cfg,cfg.seed+fold)
        predictions = predict_candidates(models,x.iloc[test_idx])
        for name,p in predictions.items():
            oof[name][test_idx]=p
            decisions[name][test_idx]=(p>=tuning[name]["threshold"]).astype(int)
        selected_p[test_idx] = predictions[selected]
        selected_pred[test_idx] = decisions[selected][test_idx]
        assignments[test_idx]=fold
        audits.append({"fold":fold,"fit_ids":fit_x.index.tolist(),"test_ids":x.index[test_idx].tolist(),
                       "inner_splits":inner_audit,"selected":selected,"inner_tuning":tuning,
                       "test_metrics":score(y[test_idx],selected_p[test_idx],selected_pred[test_idx])})
    metrics = {"validation":"Nested cross-validation; model and threshold selected inside outer training folds",
               "config":asdict(cfg),"grouping":"source_group" if groups is not None else "object_only",
               "primary":score(y,selected_p,selected_pred),
               "candidate_outer_metrics":{name:score(y,oof[name],decisions[name]) for name in names},
               "baselines":{"all_negative":score(y,np.zeros(len(y)),np.zeros(len(y),int)),
                            "all_positive":score(y,np.ones(len(y)),np.ones(len(y),int))}}
    # Deployment selection sees all OOF labels; this tuning score is NOT the unbiased result.
    final_name,final_tuning = choose_candidate(y,oof)
    metrics["deployment"] = {"candidate":final_name,**final_tuning[final_name],
                             "tuning_f1_is_performance_estimate":False}
    table = pd.DataFrame({"object_id":x.index,"target":y,"outer_fold":assignments,
                          "selected_probability":selected_p,"selected_prediction":selected_pred})
    for name in names:
        table[f"{name}_probability"] = oof[name]
        table[f"{name}_prediction"] = decisions[name]
    return metrics,table,audits


def refit(x,y,metrics,cfg):
    name = metrics["deployment"]["candidate"]
    names = ("cat_full","lgb_full") if name=="blend" else (() if name=="prior" else (name,))
    return {"candidate":name,"threshold":metrics["deployment"]["threshold"],
            "features":list(x.columns),"prior":float(np.mean(y)),
            "models":{n:fit_model(n,x,y,cfg,cfg.seed) for n in names}}


def predict_bundle(bundle,x):
    if list(x.columns)!=bundle["features"]:
        raise ValueError("Inference feature schema/order differs from training")
    name=bundle["candidate"]
    if name=="prior":
        p=np.full(len(x),bundle["prior"])
    elif name=="blend":
        p=.5*predict_model(bundle["models"]["cat_full"],x)+.5*predict_model(bundle["models"]["lgb_full"],x)
    else:
        p=predict_model(bundle["models"][name],x)
    check_probability(np.zeros(len(x)),p)
    return p,(p>=bundle["threshold"]).astype(int)
