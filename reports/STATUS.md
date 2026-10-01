# Verification and experiment status

## Official-data evaluation completed

The user supplied the official 43-file competition archive. The first fixed nested run evaluated all 3,043 labelled objects (148 TDEs), then refitted the selected model and generated 7,135 test predictions.

**Nested object-level F1: 0.5254; precision 0.4515; recall 0.6284.** [Full report and aggregate evidence](official_001/RESULTS.md).

Input inspection required explicit handling of missing flux and a negative photometric redshift before training. Every object was retained. There was one completed model experiment; loader failures occurred before any model fitting. No test labels, pseudo-labels, mirrored data or competitor feature tables were used.

## Verification

**40 unit/integration tests passed; one optional CatBoost test skipped.** Regression checks cover missing-flux auditing and the negative-redshift feature policy. [Software verification](software_validation.json) records source hashes.

The real-data run additionally passed independent input and feature hashes, outer/inner fold boundaries, exact OOF labels and F1, saved-model replay, and submission IDs/order/binary decisions. See [verification.json](official_001/verification.json). Synthetic fixtures remain software tests, not performance evidence.

The LightGBM backend is verified. CatBoost fitting remains blocked by the sandbox's unavailable `/proc/self/statm`; hybrid results are not claimed. The optional hosted test is configured, but the [first hosted run](https://github.com/Ehsan-Ch/mallorn-tde-classifier/actions/runs/36783703232) did not start because of an account-level restriction. Local verification is recorded separately.

## Remaining research

- Genuine source-template groups are unavailable; object-level folds do not establish source independence.
- Spectroscopic training versus photometric test redshift quality, threshold transfer and seed sensitivity remain open.
- Optional hybrid, neural pretraining, Gaussian processes and physical fitting require separate experiments.
- No Kaggle submission has been made. Organizer evaluation is needed for a comparable leaderboard score; beating top performers has not been demonstrated.
