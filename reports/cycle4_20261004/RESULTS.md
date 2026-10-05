# Cycle 4 — local development results

**Selected by the frozen rule: Retained cycle-2 reference.** Highest pooled development F1: **0.694006** (Reference + five-class LightGBM).

These scores use all 3,043 labeled objects (148 TDEs), two stratified five-fold repetitions, and the mean of each object's held-out probabilities. Thresholds are tuned using 500 stratified bootstraps. This is repeated development selection on historically reused labels, not independent test evidence.

| Recipe | F1 | Precision | Recall | AP | Repeat 1 F1 | Repeat 2 F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Retained cycle-2 reference | 0.684058 | 0.598985 | 0.797297 | 0.677698 | 0.699708 | 0.672464 |
| Reference + five-class LightGBM | 0.694006 | 0.650888 | 0.743243 | 0.701315 | 0.681388 | 0.668790 |
| Reference + weighted confusing negatives | 0.686747 | 0.619565 | 0.770270 | 0.697214 | 0.672783 | 0.662577 |
| New LightGBM ensemble | 0.681818 | 0.656250 | 0.709459 | 0.705987 | 0.666667 | 0.675410 |
| Reference + new LightGBM ensemble | 0.692771 | 0.625000 | 0.777027 | 0.705391 | 0.683230 | 0.678788 |

## Selection and uncertainty

An alternative must gain at least 0.005 pooled F1, avoid any AP decrease, and avoid lowering either repetition's fixed-threshold F1 by more than 0.005. The rule was written before this cycle's new fits.

- Reference + five-class LightGBM: repetition stability failed. Descriptive paired-bootstrap F1 difference interval: [-0.0203, 0.0375].

- Reference + weighted confusing negatives: F1 gain below 0.005; repetition stability failed. Descriptive paired-bootstrap F1 difference interval: [-0.0262, 0.0294].

- New LightGBM ensemble: F1 gain below 0.005; repetition stability failed. Descriptive paired-bootstrap F1 difference interval: [-0.0420, 0.0347].

- Reference + new LightGBM ensemble: repetition stability failed. Descriptive paired-bootstrap F1 difference interval: [-0.0145, 0.0308].

The intervals keep fitted models and thresholds fixed. They do not account for historical selection, threshold tuning uncertainty, or unknown shared simulation sources. They are not proof of a population-level improvement.

## Verification and continuation

All 20 saved fold models were reloaded and replayed; maximum probability difference: **0**. Every checkpoint verifies its manifest and exact fitting/validation IDs. Each object is held out exactly once in each repetition. SpecType is used only as auxiliary training supervision or training sample weights, never as a feature.

See [restart instructions](../../docs/RESUME_RESEARCH.md). The worker resumes completed folds without retraining them and rejects duplicate workers. Eight checkpoint tests cover failed atomic writes, stale configuration, invalid fold records, concurrency and lock release after a killed worker.

**No GitHub or Kaggle upload was made.** Verified private F1 remains **0.6648**, versus the top private benchmark **0.6824**. New local development scores cannot establish that the top-ranked entry has been surpassed.
