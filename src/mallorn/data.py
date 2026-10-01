"""Strict data contracts for the official MALLORN CSV layout."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

BANDS = tuple("ugrizy")
PHOTOMETRY = ("object_id", "Time (MJD)", "Flux", "Flux_err", "Filter")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_log(frame: pd.DataFrame, train: bool) -> pd.DataFrame:
    required = {"object_id", "Z", "EBV"} | ({"target"} if train else set())
    if not required.issubset(frame):
        raise ValueError(f"Missing metadata columns: {sorted(required - set(frame))}")
    frame = frame.copy()
    if frame.empty or frame.object_id.isna().any() or frame.object_id.duplicated().any():
        raise ValueError("Metadata IDs must be nonempty, unique and non-null")
    frame["object_id"] = frame.object_id.astype(str)
    if frame.object_id.str.strip().eq("").any() or frame.object_id.duplicated().any():
        raise ValueError("Empty or duplicate normalized object ID")
    for col in ("Z", "EBV"):
        frame[col] = pd.to_numeric(frame[col], errors="raise")
        invalid = (frame[col] <= -1) if col == "Z" else (frame[col] < 0)
        if not np.isfinite(frame[col]).all() or invalid.any():
            raise ValueError(f"{col} must be finite; require Z > -1 and EBV >= 0")
    if train:
        if frame.target.isna().any() or not frame.target.isin([0, 1]).all():
            raise ValueError("target must contain binary 0/1 labels")
        if frame.target.nunique() != 2:
            raise ValueError("Training data must contain both classes")
        frame["target"] = frame.target.astype(int)
    return frame


def validate_photometry(frame: pd.DataFrame) -> pd.DataFrame:
    if not set(PHOTOMETRY).issubset(frame):
        raise ValueError(f"Missing photometry columns: {sorted(set(PHOTOMETRY) - set(frame))}")
    frame = frame.loc[:, list(PHOTOMETRY)].copy()
    if frame.empty or frame.object_id.isna().any() or frame.Filter.isna().any():
        raise ValueError("Photometry is empty or contains null IDs/bands")
    frame["object_id"] = frame.object_id.astype(str)
    if frame.object_id.str.strip().eq("").any():
        raise ValueError("Empty photometry object ID")
    frame["Filter"] = frame.Filter.astype(str).str.strip().str.lower()
    if not set(frame.Filter).issubset(BANDS):
        raise ValueError("Unexpected filter; expected u,g,r,i,z,y")
    for col in ("Time (MJD)", "Flux", "Flux_err"):
        frame[col] = pd.to_numeric(frame[col], errors="raise")
        values = frame[col].dropna() if col == "Flux" else frame[col]
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite photometry: {col}")
    if (frame.Flux_err <= 0).any():
        raise ValueError("Flux uncertainties must be strictly positive")
    if frame.duplicated(["object_id", "Filter", "Time (MJD)"]).any():
        raise ValueError("Duplicate object/filter/time observations; resolve explicitly")
    # Official files contain absent flux measurements. Never impute a measured
    # flux from other objects; omit these observations and audit counts in loader.
    frame = frame.dropna(subset=["Flux"])
    if frame.empty:
        raise ValueError("No measured flux observations remain")
    return frame.sort_values(["object_id", "Filter", "Time (MJD)"], kind="stable").reset_index(drop=True)


def load_dataset(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    root = root.resolve()
    for name in ("train_log.csv", "test_log.csv", "sample_submission.csv"):
        if not (root / name).is_file():
            raise FileNotFoundError(f"Missing {root / name}. Download and extract the official data after accepting Kaggle rules.")
    train = validate_log(pd.read_csv(root / "train_log.csv"), train=True)
    test = validate_log(pd.read_csv(root / "test_log.csv"), train=False)
    if set(train.object_id) & set(test.object_id):
        raise ValueError("Train/test object IDs overlap")
    sample = pd.read_csv(root / "sample_submission.csv")
    validate_submission(sample, test.object_id)
    frames, files = [], [root / n for n in ("train_log.csv", "test_log.csv", "sample_submission.csv")]
    cleaning = {}
    for mode, log in (("train", train), ("test", test)):
        paths = sorted(root.rglob(f"{mode}_full_lightcurves.csv"))
        if not paths:
            raise FileNotFoundError(f"No {mode}_full_lightcurves.csv under {root}")
        raw = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
        photo = validate_photometry(raw)
        cleaning[mode] = {"input_rows": len(raw), "missing_flux_rows_removed": len(raw)-len(photo),
                          "retained_rows": len(photo), "negative_redshifts_floored_in_features": int((log.Z < 0).sum())}
        if set(photo.object_id) != set(log.object_id):
            raise ValueError(f"{mode} photometry/metadata object sets differ")
        if "split" in log:
            for path in paths:
                ids = set(pd.read_csv(path, usecols=["object_id"]).object_id.astype(str))
                expected = set(log.loc[log.split.astype(str).eq(path.parent.name), "object_id"])
                if ids != expected:
                    raise ValueError(f"Metadata split disagrees with folder: {path.parent.name}")
        frames.append(photo)
        files.extend(paths)
    manifest = {"files": [{"path": str(p.relative_to(root)), "bytes": p.stat().st_size,
                            "sha256": sha256(p)} for p in files],
                "train_objects": len(train), "test_objects": len(test),
                "positive_objects": int(train.target.sum()),
                "photometry_rows": sum(map(len, frames)), "cleaning": cleaning}
    return train, test, pd.concat(frames, ignore_index=True), manifest


def validate_submission(frame: pd.DataFrame, expected_ids) -> None:
    if list(frame.columns) != ["object_id", "prediction"]:
        raise ValueError("Submission columns must be object_id,prediction in that order")
    if frame.object_id.isna().any() or frame.object_id.duplicated().any():
        raise ValueError("Submission IDs must be unique and non-null")
    if set(frame.object_id.astype(str)) != set(map(str, expected_ids)):
        raise ValueError("Submission IDs do not exactly match the test objects")
    if frame.prediction.isna().any() or not frame.prediction.isin([0, 1]).all():
        raise ValueError("Submission requires binary predictions, not probabilities")


def load_groups(path: Path | None, ids: pd.Series) -> np.ndarray | None:
    if path is None:
        return None
    frame = pd.read_csv(path, dtype={"object_id": str, "group_id": str})
    if not {"object_id", "group_id"}.issubset(frame):
        raise ValueError("Group mapping requires object_id,group_id")
    if frame.object_id.duplicated().any() or frame[["object_id", "group_id"]].isna().any().any():
        raise ValueError("Invalid group mapping")
    if set(frame.object_id) != set(ids):
        raise ValueError("Group mapping must cover exactly the training IDs")
    return frame.set_index("object_id").loc[ids, "group_id"].to_numpy()
