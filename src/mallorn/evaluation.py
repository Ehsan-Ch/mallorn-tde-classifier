"""Backend-independent metrics, identical to the frozen cycle-2 definitions.

Kept separate so LightGBM research never imports the optional native CatBoost
backend. The archived modeling module stays unchanged for hash verification.
"""
import numpy as np
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score, roc_auc_score)

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
