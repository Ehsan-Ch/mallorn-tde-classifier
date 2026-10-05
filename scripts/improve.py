"""Local development screening. The reserved audit outcomes are never scored here."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split

from mallorn.data import sha256
from mallorn.modeling import best_threshold, score
from mallorn.reporting import write_json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/improvement"


def split_ids(train):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "split.json"
    if path.exists():
        split = json.loads(path.read_text())
    else:
        dev, audit = train_test_split(train.object_id.to_numpy(), test_size=.25,
                                     stratify=train.target, random_state=20261003)
        split = {"development": dev.tolist(), "audit": audit.tolist(), "seed": 20261003,
                 "note": "Reserved for this cycle; all labels were previously used in baseline evaluation"}
        write_json(path, split)
    assert not set(split["development"]) & set(split["audit"])
    assert set(split["development"]) | set(split["audit"]) == set(train.object_id)
    return split


def estimator(spec, seed):
    params = dict(spec)
    kind = params.pop("kind")
    params.pop("view", None)
    params.pop("auxiliary", None)
    params.pop("select_features", None)
    params.pop("hierarchical", None)
    if kind == "cat":
        return CatBoostClassifier(**params, random_seed=seed, thread_count=int(os.environ.get('MALLORN_THREADS',6)),
                                  allow_writing_files=False, verbose=False)
    return LGBMClassifier(**params, random_state=seed, n_jobs=int(os.environ.get('MALLORN_THREADS',6)), verbosity=-1,
                          deterministic=True, force_col_wise=True)


def candidate_specs(batch):
    cat = dict(kind="cat", iterations=1000, learning_rate=.035, l2_leaf_reg=6, loss_function="Logloss")
    lgb = dict(kind="lgb", n_estimators=700, learning_rate=.025, reg_lambda=5,
               colsample_bytree=.85, min_child_samples=20)
    specs = {
        "cat_d5": dict(cat, depth=5),
        "cat_d4_w3": dict(cat, depth=4, class_weights=[1, 3]),
        "cat_d6": dict(cat, depth=6),
        "lgb_l9_w3": dict(lgb, num_leaves=9, scale_pos_weight=3),
        "lgb_l15": dict(lgb, num_leaves=15),
    }
    if batch == "compact":
        specs = {k:v for k,v in specs.items() if k in ("cat_d5", "cat_d4_w3", "lgb_l9_w3")}
    elif batch == "advanced":
        specs = {
            "cat_d4_long":dict(cat,depth=4,iterations=1800,l2_leaf_reg=8),
            "cat_d5_w3":dict(cat,depth=5,iterations=1200,class_weights=[1,3]),
            "cat_aux3_d5":dict(cat,depth=5,iterations=1200,loss_function='MultiClass',
                               class_weights=[1,1,3],auxiliary=True),
            "lgb_l9_long":dict(lgb,num_leaves=9,n_estimators=1200),
            "lgb_l15_w3":dict(lgb,num_leaves=15,scale_pos_weight=3),
        }
    elif batch == "selected":
        specs = {
            "cat_d4_w3_top125": dict(cat, depth=4, class_weights=[1,3], select_features=125),
            "cat_d5_w3_top250": dict(cat, depth=5, class_weights=[1,3], select_features=250),
            "cat_aux3_d5_top125": dict(cat, depth=5, iterations=1200,
                                      loss_function='MultiClass', class_weights=[1,1,3],
                                      auxiliary=True, select_features=125),
        }
    elif batch == "physical":
        specs = {
            'cat_d4_w3': dict(cat, depth=4, class_weights=[1,3]),
            'cat_hierarchy_d4': dict(cat, depth=4, class_weights=[1,3], hierarchical=True),
            'cat_aux6_d4': dict(cat, depth=4, iterations=1200, loss_function='MultiClass',
                              class_weights=[1,1,1,1,1,3], auxiliary='six'),
        }
    return specs


def training_columns(x, y, spec, seed):
    """Rank and decorrelate columns using the fitting fold only."""
    count = spec.get('select_features')
    if count is None:
        return list(x.columns)
    selector = estimator(dict(kind='cat', iterations=500, depth=4,
                             learning_rate=.04, loss_function='Logloss',
                             class_weights=[1,3], l2_leaf_reg=8), seed)
    selector.fit(x, y)
    order = np.argsort(-selector.feature_importances_, kind='stable')
    candidates = list(x.columns[order[:min(len(order), count * 3)]])
    # Pairwise training-only correlation, including missing values as missing.
    corr = x[candidates].corr().abs()
    selected = []
    for column in candidates:
        if not selected or not (corr.loc[column, selected] > .95).any():
            selected.append(column)
        if len(selected) == count:
            break
    return selected


def fit_candidate(x, meta, spec, seed):
    y = meta.target.to_numpy()
    columns = training_columns(x, y, spec, seed)
    x = x[columns]
    model = estimator(spec, seed)
    if spec.get('hierarchical'):
        transient = ~meta.SpecType.eq('AGN').to_numpy()
        gate = estimator(dict(spec, class_weights=[1,1]), seed)
        gate.fit(x, transient.astype(int))
        model.fit(x.loc[transient], y[transient])
        return {'model': model, 'gate': gate, 'columns': columns, 'positive': 1}
    if spec.get('auxiliary') == 'six':
        # AGN, Ia family, IIn, other type-II, other transient, TDE.
        types = meta.SpecType.astype(str)
        labels = np.select([types.eq('AGN'), types.str.startswith('SN Ia'),
                            types.eq('SN IIn'), types.str.startswith('SN II'), y == 1],
                           [0, 1, 2, 3, 5], default=4)
        positive = 5
    elif spec.get('auxiliary'):
        labels = np.where(meta.SpecType.eq('AGN'), 0, np.where(y == 1, 2, 1))
        positive = 2
    else:
        labels, positive = y, 1
    model.fit(x, labels)
    return {'model': model, 'columns': columns, 'positive': positive}


def predict_candidate(fitted, x):
    x = x[fitted['columns']]
    model = fitted['model']
    p = model.predict_proba(x)[:, list(model.classes_).index(fitted['positive'])]
    if 'gate' in fitted:
        gate = fitted['gate']
        p *= gate.predict_proba(x)[:, list(gate.classes_).index(1)]
    return p


def screen(features_path, label, batch):
    train = pd.read_csv(ROOT / "data/mallorn/train_log.csv")
    split = split_ids(train)
    x = pd.read_csv(features_path, index_col="object_id").loc[split["development"]]
    meta = train.set_index("object_id").loc[x.index]
    y = meta.target.to_numpy()
    folder = OUT / label
    folder.mkdir(exist_ok=True)
    config = {"features_sha256": sha256(features_path), "development_objects": len(x),
              "feature_count": x.shape[1], "fold_seed": 20261004, "folds": 5,
              "scope": "Development OOF candidate/threshold tuning, not an unbiased final score",
              "candidates": candidate_specs(batch)}
    config_path = folder / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("Existing experiment configuration differs")
    write_json(config_path, config)
    cv = list(StratifiedKFold(5, shuffle=True, random_state=20261004).split(x, y))
    table = pd.DataFrame({"object_id": x.index, "target": y})
    results = {}
    for name, spec in config["candidates"].items():
        saved = folder / f"{name}.npy"
        start = time.monotonic()
        if saved.exists():
            p = np.load(saved)
        else:
            checkpoint = folder / f'{name}_partial.npy'
            p = np.load(checkpoint) if checkpoint.exists() else np.full(len(y), np.nan)
            for fold, (a,b) in enumerate(cv):
                if np.isfinite(p[b]).all():
                    continue
                fitted = fit_candidate(x.iloc[a], meta.iloc[a], spec, 42+fold)
                if spec.get('select_features'):
                    write_json(folder / f'{name}_fold{fold}_columns.json', fitted['columns'])
                p[b] = predict_candidate(fitted, x.iloc[b])
                temporary = folder / f'{name}_checkpoint.npy'
                np.save(temporary, p)
                temporary.replace(checkpoint)
                print(label, name, "fold", fold+1, "seconds", round(time.monotonic()-start,1), flush=True)
            np.save(saved, p)
            checkpoint.unlink(missing_ok=True)
        threshold, _ = best_threshold(y,p)
        results[name] = dict(score(y,p,p>=threshold), threshold=threshold,
                             elapsed_seconds=round(time.monotonic()-start,2))
        table[name] = p
        write_json(folder / "scores.json", results)
        print(label, name, json.dumps(results[name]), flush=True)
    base_names = list(config["candidates"])
    for i, left in enumerate(base_names):
        for right in base_names[i+1:]:
            name = f"blend_{left}_{right}"
            p = .5*table[left].to_numpy()+.5*table[right].to_numpy()
            threshold, _ = best_threshold(y,p)
            results[name] = dict(score(y,p,p>=threshold), threshold=threshold)
    table.to_csv(folder / "oof.csv", index=False, float_format="%.15g")
    write_json(folder / "scores.json", results)
    print("BEST",label,json.dumps(sorted(results.items(),key=lambda kv:kv[1]['f1'],reverse=True)[:5]),flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features",type=Path,required=True)
    parser.add_argument("--label",required=True)
    parser.add_argument("--batch",choices=["full","compact","advanced","selected","physical"],default="full")
    args=parser.parse_args()
    screen(args.features,args.label,args.batch)
