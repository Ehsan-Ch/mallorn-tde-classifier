"""Local evidence export; synthetic runs are never competition scores."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve


def write_json(path: Path, value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n",encoding="utf-8")


def report(output: Path, metrics: dict, oof):
    figures=output/"figures"
    figures.mkdir(parents=True,exist_ok=True)
    synthetic=metrics["data_kind"]=="synthetic_fixture"
    title="SYNTHETIC SOFTWARE TEST — NOT COMPETITION PERFORMANCE" if synthetic else "Local nested validation — not a Kaggle leaderboard score"
    models=list(metrics["candidate_outer_metrics"])
    values=[metrics["candidate_outer_metrics"][n]["f1"] for n in models]
    with plt.rc_context({"font.size":10,"axes.spines.top":False,"axes.spines.right":False}):
        fig,axes=plt.subplots(1,2,figsize=(12,4.5))
        axes[0].barh(models,values,color="#177e89")
        axes[0].set(xlim=(0,1),xlabel="F1 on outer held-out objects",title="Fixed candidates; inner-selected thresholds")
        p,r,_=precision_recall_curve(oof.target,oof.selected_probability)
        axes[1].plot(r,p,color="#6d4596",label="Nested selected workflow")
        axes[1].axhline(oof.target.mean(),linestyle="--",color="gray",label="Positive prevalence")
        axes[1].set(xlim=(0,1),ylim=(0,1.02),xlabel="Recall",ylabel="Precision",title="Cross-validated probability ranking")
        axes[1].legend(fontsize=8)
        fig.suptitle(title,fontsize=11)
        fig.tight_layout()
        fig.savefig(figures/"validation.svg",metadata={"Date":None})
        plt.close(fig)
    lines=["# Run report","",f"**{title}**","",
           f"Primary nested F1: **{metrics['primary']['f1']:.4f}**; precision {metrics['primary']['precision']:.4f}; recall {metrics['primary']['recall']:.4f}.","",
           "| Candidate | Outer F1 | Precision | Recall | Average precision |","| --- | ---: | ---: | ---: | ---: |"]
    for name,m in metrics["candidate_outer_metrics"].items():
        lines.append(f"| {name} | {m['f1']:.4f} | {m['precision']:.4f} | {m['recall']:.4f} | {m['average_precision']:.4f} |")
    lines += ["","![Validation results](figures/validation.svg)","",
              "Model and threshold selection are nested within each outer fold. Final deployment tuning uses all OOF labels; its F1 is not an unbiased performance estimate.","",
              f"Grouping mode: `{metrics['grouping']}`. Without real source-template groups, related simulations may cross folds; object-only CV does not resolve that risk.","",
              "The train/test redshift measurement difference and repeated exploration of local results can reduce generalization. This report provides no claim of beating the private leaderboard.","",
              "Exact scores: [metrics.json](metrics.json). Predictions: [oof.csv](oof.csv). IDs and fitting boundaries: [fold_audit.json](fold_audit.json)."]
    (output/"RESULTS.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


def bootstrap_interval(y, decisions, groups=None, seed=42, repeats=1000):
    """Conditional bootstrap of fixed OOF decisions; does not refit models."""
    rng=np.random.default_rng(seed)
    y,decisions=np.asarray(y),np.asarray(decisions)
    units=np.arange(len(y)) if groups is None else np.asarray(groups)
    unique=np.unique(units)
    blocks={unit:np.flatnonzero(units==unit) for unit in unique}
    values=[]
    for _ in range(repeats):
        idx=np.concatenate([blocks[u] for u in rng.choice(unique,len(unique),replace=True)])
        tp=np.sum((y[idx]==1)&(decisions[idx]==1))
        denom=np.sum(y[idx])+np.sum(decisions[idx])
        values.append(2*tp/denom if denom else 0.)
    return {"ci95_low":float(np.quantile(values,.025)),"ci95_high":float(np.quantile(values,.975)),
            "repeats":repeats,"unit":"source_group" if groups is not None else "object",
            "scope":"Conditional on fixed OOF decisions; ignores model refitting and protocol selection uncertainty"}
