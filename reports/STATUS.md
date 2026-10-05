# Verification and experiment status

Publication update, 5 October 2026: source and aggregate reports are being
released to GitHub and a Kaggle code notebook. Earlier statements that publishing
was paused describe the research cycles at their completion. See
[publication checkpoint](../docs/PUBLICATION_CHECKPOINT.md) and
[workspace recovery limitations](RECOVERY_20261005.md).

## Official-data evaluation completed

The user supplied the official 43-file competition archive. The first fixed nested run evaluated all 3,043 labelled objects (148 TDEs), then refitted the selected model and generated 7,135 test predictions.

**Nested object-level F1: 0.5254; precision 0.4515; recall 0.6284.** [Full report and aggregate evidence](official_001/RESULTS.md).

Input inspection required explicit handling of missing flux and a negative photometric redshift before training. Every object was retained. This was the first completed model experiment; loader failures occurred before fitting. No test labels, pseudo-labels, mirrored data or competitor feature tables were used.

## Unpublished improvement cycle

The subsequent cycle evaluated 29 model configurations and 443 fixed model/ensemble recipes using 2,282 development objects. The selected recipe reached development F1 **0.6912**, then scored **0.5833** on the 761-object reserved audit with its threshold fixed. **The ~0.68 target was not confirmed.** The audit was reserved for this cycle, but the baseline had already used all training labels. [Full report](improvement_20261003/RESULTS.md).

The new cycle adds GP/thermal and flare features, auxiliary class supervision, and feature selection confined to training folds. One user-authorized late submission on 4 October 2026 scored private F1 **0.6648** and public F1 **0.6825**. No new GitHub publication occurred. [Verified Kaggle result](kaggle_20261004/RESULTS.md).

## Second-cycle candidate

Two repeated five-fold validations on all 3,043 labeled objects selected a four-component ensemble: local F1 **0.6841** versus **0.6707** for the reference component types under the same protocol; average precision **0.6777** versus **0.6602**. The recipe and threshold are frozen. Final fitting is currently blocked because CatBoost cannot read `/proc/self/statm` in the refreshed restricted runtime. **No second submission exists; private F1 remains 0.6648.** [Results and continuation](cycle2_20261004/RESULTS.md).

## Restartable research cycles

Cycles 3 and 4 completed **50 fitted validation folds**, with exact saved-model
prediction replay. The largest pooled development F1 was **0.7030** (reference
plus three-class physics LightGBM). Five-class LightGBM reached **0.7131 average
precision**. Neither cycle produced a challenger that passed every prespecified
F1/AP/repetition-stability requirement, so the selected cycle-2 recipe remains
unchanged. These are historically reused development labels, not a new test.
See [cycle 3](cycle3_20261004/RESULTS.md) and [cycle 4](cycle4_20261004/RESULTS.md).

The worker now checkpoints models and fold predictions atomically, verifies
hashes and validation IDs on resume, rejects duplicate workers, and records
failures. An hourly scheduled continuation check is enabled, subject to usage,
tool and workspace availability. It cannot detect an exact token-reset event or
guarantee workspace persistence. [Continuation instructions](../docs/RESUME_RESEARCH.md).
The user's current instruction pauses all GitHub and Kaggle uploads.

## Current software verification

The baseline had 40 passing tests and one optional skip. **The last complete local suite passed all 47 unit/integration tests, including CatBoost**, in the earlier compatible runtime. Added tests cover physical parameter recovery and date-translation invariance. The archived [baseline software verification](software_validation.json) remains unchanged. In the refreshed restricted runtime, eight new checkpoint tests cover the restartable workflow; the full suite was not rerun because native CatBoost is unavailable. See [current continuation verification](cycle3_20261004/software_verification.json).

The real-data run additionally passed independent input and feature hashes, outer/inner fold boundaries, exact OOF labels and F1, saved-model replay, and submission IDs/order/binary decisions. See [verification.json](official_001/verification.json). Synthetic fixtures remain software tests, not performance evidence.

LightGBM, native CatBoost and the optional hybrid test are verified locally. CatBoost required an execution environment exposing its own `/proc/self/statm`; the local sklearn compatibility adapter supplies estimator tags. The [first hosted run](https://github.com/Ehsan-Ch/mallorn-tde-classifier/actions/runs/36783703232) did not start because of an account-level restriction. Hosted CI was not rerun; local verification is recorded separately.

## Remaining research

- Genuine source-template groups are unavailable; object-level folds do not establish source independence.
- Spectroscopic training versus photometric test redshift quality, threshold transfer and seed sensitivity remain open.
- Neural pretraining, SALT2 and echo-state models are not implemented. GP and approximate physical fitting have now been evaluated, without confirming the target on the reserved audit.
- This cycle's audit is now open and must not be reused as independent confirmation of further tuning.
- Kaggle private F1 is 0.6648, below the winner's 0.6824. Beating top performers has not been demonstrated; this was a late submission, not an official placement.
