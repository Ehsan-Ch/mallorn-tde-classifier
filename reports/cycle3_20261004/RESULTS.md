# Cycle 3 — local development results

**Selected by the frozen rule: Retained cycle-2 reference.** Highest pooled development F1: **0.703030** (Reference + physics three-class LightGBM).

These scores use all 3,043 labeled objects (148 TDEs), two stratified five-fold repetitions, and the mean of each object's held-out probabilities. Thresholds are tuned using 500 stratified bootstraps. This is repeated development selection on historically reused labels, not independent test evidence.

| Recipe | F1 | Precision | Recall | AP | Repeat 1 F1 | Repeat 2 F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Retained cycle-2 reference | 0.684058 | 0.598985 | 0.797297 | 0.677698 | 0.699708 | 0.672464 |
| Reference + physics three-class LightGBM | 0.703030 | 0.637363 | 0.783784 | 0.673992 | 0.678679 | 0.672783 |
| Reference + hierarchical LightGBM | 0.695652 | 0.643678 | 0.756757 | 0.679843 | 0.677215 | 0.668790 |
| Reference + complete binary LightGBM | 0.682353 | 0.604167 | 0.783784 | 0.692715 | 0.670695 | 0.672673 |
| New LightGBM ensemble | 0.690476 | 0.617021 | 0.783784 | 0.686134 | 0.666667 | 0.674699 |
| Reference + new LightGBM ensemble | 0.687117 | 0.629213 | 0.756757 | 0.689378 | 0.691589 | 0.677019 |

## Selection and uncertainty

An alternative must gain at least 0.005 pooled F1, avoid any AP decrease, and avoid lowering either repetition's fixed-threshold F1 by more than 0.005. The rule was written before this cycle's new fits.

- Reference + physics three-class LightGBM: AP decreased; repetition stability failed. Descriptive paired-bootstrap F1 difference interval: [-0.0055, 0.0430].

- Reference + hierarchical LightGBM: repetition stability failed. Descriptive paired-bootstrap F1 difference interval: [-0.0195, 0.0406].

- Reference + complete binary LightGBM: F1 gain below 0.005; repetition stability failed. Descriptive paired-bootstrap F1 difference interval: [-0.0265, 0.0215].

- New LightGBM ensemble: repetition stability failed. Descriptive paired-bootstrap F1 difference interval: [-0.0205, 0.0343].

- Reference + new LightGBM ensemble: F1 gain below 0.005; repetition stability failed. Descriptive paired-bootstrap F1 difference interval: [-0.0232, 0.0282].

The intervals keep fitted models and thresholds fixed. They do not account for historical selection, threshold tuning uncertainty, or unknown shared simulation sources. They are not proof of a population-level improvement.

## Verification and continuation

All 30 saved fold models were reloaded and replayed; maximum probability difference: **0**. Every checkpoint verifies its manifest and exact fitting/validation IDs. Each object is held out exactly once in each repetition. SpecType is used only as auxiliary training supervision or training sample weights, never as a feature.

See [restart instructions](../../docs/RESUME_RESEARCH.md). The worker resumes completed folds without retraining them and rejects duplicate workers. Eight checkpoint tests cover failed atomic writes, stale configuration, invalid fold records, concurrency and lock release after a killed worker.

**No GitHub or Kaggle upload was made.** Verified private F1 remains **0.6648**, versus the top private benchmark **0.6824**. New local development scores cannot establish that the top-ranked entry has been surpassed.
