"""Independently audit a completed local run; load only your own model artifact."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from mallorn.data import sha256, validate_submission
from mallorn.modeling import predict_bundle
from mallorn.reporting import write_json


def verify(data: Path, run: Path):
    metrics = json.loads((run / "metrics.json").read_text())
    manifest = json.loads((run / "manifest.json").read_text())
    audit = json.loads((run / "fold_audit.json").read_text())
    train = pd.read_csv(data / "train_log.csv").set_index("object_id")
    test = pd.read_csv(data / "test_log.csv")
    sample = pd.read_csv(data / "sample_submission.csv")
    oof = pd.read_csv(run / "oof.csv")
    assert not oof.object_id.duplicated().any()
    assert set(oof.object_id) == set(train.index)
    assert not oof.isna().any().any()
    np.testing.assert_array_equal(oof.target, train.loc[oof.object_id, "target"])
    np.testing.assert_allclose(f1_score(oof.target, oof.selected_prediction), metrics["primary"]["f1"], atol=1e-14)
    held_out = []
    for fold in audit:
        fit, held = set(fold["fit_ids"]), set(fold["test_ids"])
        assert not fit & held and fit | held == set(train.index)
        held_out.extend(fold["test_ids"])
        inner_held_out = []
        for inner in fold["inner_splits"]:
            a, b = set(inner["fit_ids"]), set(inner["validation_ids"])
            assert not a & b and a | b == fit and not (a | b) & held
            inner_held_out.extend(inner["validation_ids"])
        assert len(inner_held_out) == len(set(inner_held_out)) == len(fit)
        rows = oof[oof.outer_fold.eq(fold["fold"])]
        assert set(rows.object_id) == held
        chosen = fold["selected"]
        np.testing.assert_array_equal(rows.selected_prediction, rows[f"{chosen}_prediction"])
        np.testing.assert_allclose(rows.selected_probability, rows[f"{chosen}_probability"])
    assert len(held_out) == len(set(held_out)) == len(train)
    for item in manifest["files"]:
        assert sha256(data / item["path"]) == item["sha256"]
    assert sha256(run / "features.csv") == manifest["feature_matrix_sha256"]
    features = pd.read_csv(run / "features.csv", index_col="object_id")
    assert not np.isinf(features.to_numpy()).any()
    bundle = joblib.load(run / "model.joblib")
    probabilities, predictions = predict_bundle(bundle, features.loc[test.object_id])
    saved = pd.read_csv(run / "test_probabilities.csv")
    assert saved.object_id.tolist() == test.object_id.tolist()
    np.testing.assert_allclose(probabilities, saved.probability, rtol=1e-8, atol=1e-10)
    submission = pd.read_csv(run / "submission.csv")
    validate_submission(submission, test.object_id)
    assert submission.object_id.tolist() == sample.object_id.tolist()
    expected = pd.Series(predictions, index=test.object_id).loc[sample.object_id]
    np.testing.assert_array_equal(submission.prediction, expected)
    result = {"status": "passed", "train_objects": len(train), "test_objects": len(test),
              "outer_folds": len(audit), "input_hashes_verified": len(manifest["files"]),
              "feature_matrix_hash_verified": True, "fold_partitions_verified": True,
              "oof_labels_and_f1_verified": True, "saved_model_replay_verified": True,
              "submission_ids_order_and_decisions_verified": True,
              "test_predicted_positives": int(submission.prediction.sum()),
              "submission_sha256": sha256(run / "submission.csv")}
    write_json(run / "verification.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    verify(args.data, args.run)
