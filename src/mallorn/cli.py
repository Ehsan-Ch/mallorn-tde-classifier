"""Run locally on authorized MALLORN files; never uploads to Kaggle."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime,timezone
from importlib.metadata import version
from pathlib import Path
import tempfile

import joblib
import pandas as pd

from .data import load_dataset,load_groups,sha256,validate_submission
from .features import FeatureConfig,extract_features
from .fixtures import make_fixture
from .modeling import TrainConfig,nested_evaluate,predict_bundle,refit
from .reporting import bootstrap_interval,report,write_json


def prepare(root,feature_cfg):
    train,test,photo,manifest=load_dataset(root)
    # Concatenation here only enables per-object extraction; no across-object fit.
    metadata=pd.concat([train,test],ignore_index=True)
    x=extract_features(metadata,photo,feature_cfg)
    return train,test,x,manifest


def run(root: Path,output: Path,cfg: TrainConfig,feature_cfg: FeatureConfig,groups_path=None):
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Output directory is not empty; use a new run directory to preserve evidence")
    train,test,x,manifest=prepare(root,feature_cfg)
    groups=load_groups(groups_path,train.object_id)
    train_x=x.loc[train.object_id]
    output.mkdir(parents=True,exist_ok=True)
    print(f"Prepared {len(train)} training and {len(test)} test objects; {x.shape[1]} features",flush=True)
    metrics,oof,audit=nested_evaluate(train_x,train.target.to_numpy(),cfg,groups)
    synthetic=(root/"SYNTHETIC_FIXTURE.json").is_file()
    metrics.update(data_kind="synthetic_fixture" if synthetic else "user_supplied_mallorn_format",
                   generated_at_utc=datetime.now(timezone.utc).isoformat(),feature_count=x.shape[1],
                   feature_config=feature_cfg.metadata(),
                   environment={p:version(p) for p in ("numpy","pandas","scipy","scikit-learn","catboost","lightgbm","extinction","joblib")})
    metrics["conditional_f1_interval"]=bootstrap_interval(oof.target,oof.selected_prediction,groups,cfg.seed)
    bundle=refit(train_x,train.target.to_numpy(),metrics,cfg)
    bundle.update(feature_config=asdict(feature_cfg),data_kind=metrics["data_kind"])
    probabilities,predictions=predict_bundle(bundle,x.loc[test.object_id])
    sample=pd.read_csv(root/"sample_submission.csv")
    indexed=pd.Series(predictions,index=test.object_id)
    submission=sample[["object_id"]].copy()
    submission["prediction"]=submission.object_id.map(indexed).astype(int)
    validate_submission(submission,test.object_id)
    name="synthetic_submission.csv" if synthetic else "submission.csv"
    submission.to_csv(output/name,index=False)
    pd.DataFrame({"object_id":test.object_id,"probability":probabilities}).to_csv(output/"test_probabilities.csv",index=False,float_format="%.12g")
    oof.to_csv(output/"oof.csv",index=False,float_format="%.12g")
    x.to_csv(output/"features.csv",float_format="%.12g")
    joblib.dump(bundle,output/"model.joblib")
    manifest.update(feature_names=list(x.columns),feature_config=feature_cfg.metadata(),
                    feature_matrix_sha256=sha256(output/"features.csv"),
                    groups_sha256=sha256(groups_path) if groups_path is not None else None)
    write_json(output/"manifest.json",manifest)
    write_json(output/"fold_audit.json",audit)
    write_json(output/"metrics.json",metrics)
    report(output,metrics,oof)
    print(f"Completed {metrics['data_kind']}; nested local F1={metrics['primary']['f1']:.4f}. Not a Kaggle score.",flush=True)
    return metrics


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    subs=parser.add_subparsers(dest="command",required=True)
    train=subs.add_parser("run",help="Validate files, engineer features, run nested CV, refit and generate a local submission")
    train.add_argument("--data",type=Path,required=True)
    train.add_argument("--output",type=Path,required=True)
    train.add_argument("--groups",type=Path,help="Optional real source-template mapping: object_id,group_id")
    train.add_argument("--outer-folds",type=int,default=5)
    train.add_argument("--inner-folds",type=int,default=3)
    train.add_argument("--iterations",type=int,default=500)
    train.add_argument("--seed",type=int,default=42)
    train.add_argument("--threads",type=int,default=2)
    train.add_argument("--backend",choices=["lightgbm","hybrid"],default="lightgbm",help="hybrid adds CatBoost and a fixed blend; needs a compatible host")
    train.add_argument("--no-deredden",action="store_true")
    smoke=subs.add_parser("smoke",help="Run an explicitly synthetic software test, not a benchmark")
    smoke.add_argument("--output",type=Path,required=True)
    prediction=subs.add_parser("predict",help="Replay test predictions from a locally trusted model")
    prediction.add_argument("--data",type=Path,required=True)
    prediction.add_argument("--model",type=Path,required=True)
    prediction.add_argument("--output",type=Path,required=True)
    check=subs.add_parser("check-submission")
    check.add_argument("--submission",type=Path,required=True)
    check.add_argument("--test-log",type=Path,required=True)
    args=parser.parse_args()
    if args.command=="run":
        cfg=TrainConfig(args.outer_folds,args.inner_folds,args.iterations,args.seed,args.threads,args.backend)
        cfg.validate()
        run(args.data,args.output,cfg,FeatureConfig(deredden=not args.no_deredden),args.groups)
    elif args.command=="smoke":
        with tempfile.TemporaryDirectory(prefix="mallorn_synthetic_") as temporary:
            root=Path(temporary)
            make_fixture(root)
            run(root,args.output,TrainConfig(3,2,12,42,1),FeatureConfig())
    elif args.command=="predict":
        if args.output.exists():
            raise FileExistsError("Prediction file exists; choose a new output")
        # joblib uses pickle. Never load a model from an untrusted third party.
        bundle=joblib.load(args.model)
        _,test,x,_=prepare(args.data,FeatureConfig(**bundle["feature_config"]))
        _,pred=predict_bundle(bundle,x.loc[test.object_id])
        sample=pd.read_csv(args.data/"sample_submission.csv")
        sample["prediction"]=sample.object_id.map(pd.Series(pred,index=test.object_id)).astype(int)
        validate_submission(sample,test.object_id)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        sample.to_csv(args.output,index=False)
        print(f"Wrote {len(sample)} predictions; model data kind: {bundle['data_kind']}")
    else:
        validate_submission(pd.read_csv(args.submission),pd.read_csv(args.test_log).object_id)
        print("Submission schema, object set and binary predictions are valid")


if __name__=="__main__":
    main()
