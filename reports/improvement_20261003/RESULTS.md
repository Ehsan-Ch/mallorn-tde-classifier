# Improvement cycle — 3 October 2026

> **4 October update:** the unchanged full-data refit was submitted with user authorization and scored **0.6648 private F1 / 0.6825 public F1**. [Verified Kaggle result](../kaggle_20261004/RESULTS.md). The report below preserves the pre-submission local audit.

**The ~0.68 F1 target was not confirmed by the reserved audit.**
The frozen model scored **0.5833 F1**, despite a development selection score
of **0.6912**. No new GitHub publication or Kaggle submission was made.

| Result | F1 | Precision | Recall | Evaluation scope |
| --- | ---: | ---: | ---: | --- |
| Published baseline | 0.5254 | 0.4515 | 0.6284 | Earlier nested CV over 3,043 objects |
| Frozen recipe, development | 0.6912 | 0.7075 | 0.6757 | 2,282 objects; model and threshold selection |
| Frozen recipe, reserved audit | **0.5833** | **0.6000** | **0.5676** | 761 objects; threshold fixed before scoring |

These are different evaluation populations/protocols. Their differences do
not establish an unbiased improvement over the published baseline.

## Comparison with the leaders

| Competitor/result | F1 | Measurement |
| --- | ---: | --- |
| John Titor, first | 0.6824 | Final private leaderboard |
| Suxing's Astronomer, second | 0.6762 | Final private leaderboard |
| Sigrid Nissen, third | 0.6758 | Final private leaderboard |
| This project's reserved audit | 0.5833 | Local audit; different data |

[Official final standings](https://www.kaggle.com/competitions/mallorn-astronomical-classification-challenge/leaderboard),
verified 3 October 2026. The audit is 9.91 percentage points below
the winner's reported score, but it is not a leaderboard comparison on the same
objects. The larger development score does not establish a win.

## What changed

- Added 471 Gaussian-process, color, luminosity and approximate thermal features.
- Added 241 detection, distribution and descriptive flare-fit features.
- Compared physical-only and combined feature views, including 24 derived physical measurements.
- Trained binary, three-class, six-class and hierarchical classifiers.
- Performed supervised feature selection entirely inside training folds.
- Evaluated 29 fitted model configurations and 443 fixed model/ensemble recipes.
  Every recipe's threshold was selected on development OOF predictions.

The selected recipe equally averages a 471-feature, three-class CatBoost model
(depth 5, 1,200 iterations) and a binary CatBoost model (depth 4, 1,000 iterations)
whose 125 columns are selected from 1,126 candidate features. Threshold:
**0.322857105579911**. Both audit components were refitted on development only.

## Audit evidence

The recipe, feature bytes, code, metadata and split were hashed before the audit.
The recipe was frozen at `2026-10-03T17:11:44.132207+00:00`; the audit completed at
`2026-10-03T17:12:47.688547+00:00`. No threshold adjustment followed its outcome.

| Audit count | Value |
| --- | ---: |
| Objects | 761 |
| True TDEs | 37 |
| True positives | 21 |
| False positives | 14 |
| False negatives | 16 |
| True negatives | 710 |

Average precision: 0.5719; ROC AUC: 0.9760.
The conditional 95% bootstrap interval for audit F1 is
**[0.4348, 0.7077]** (5,000 resamples).
It holds the classifier fixed and excludes training, model-selection and
source-group uncertainty. Only 37 audit objects are TDEs.

The audit was reserved for this improvement cycle, but the old baseline had
already used every training label. It is not a historically pristine test set.
Real source/template groups are unavailable. Training/test redshift quality
also differs. An organizer-scored result is needed for a comparable competition
claim. Further development must disclose that this audit has now been opened.

## Verification and handoff

All **47 local unit/integration tests passed**, including native CatBoost.
Four train/test feature schemas matched for all 7,135 unlabeled objects.
A selected model's first development fold reproduced the stored probabilities
with maximum absolute error **0.0**, including exactly the same selected columns.
This is local verification; hosted GitHub Actions was not rerun.

The frozen audit models and experiment ledger are preserved privately. The unchanged recipe was refitted on all 3,043 labeled objects and produced
7,135 test predictions (406 positive). Saved-model inference replay matched
exactly, and the submission matched the official schema and object order.
This is an unsubmitted local candidate, not a new validation result.
`python scripts/deploy_improvement.py --predict-only` verifies its saved models.
See [methods](../../docs/IMPROVEMENT_METHODS.md) and
[protocol](../../docs/IMPROVEMENT_PROTOCOL.md) for reproducibility and boundaries.
