# Verification and experiment status

## Competition evaluation

**Pending authorized competition data. No MALLORN model score has been measured.**

The official data page requires sign-in and acceptance of the competition rules. The current browser is signed out of Kaggle. No bypass, mirrored dataset, competitor feature table, test labels or pseudo-labels have been used.

## Implemented

- Strict loader for the official train/test log and split-folder layout.
- Label-free features, CatBoost/LightGBM candidates and a fixed probability blend.
- Nested candidate and threshold selection, optional genuine source-group folds.
- Per-object OOF evidence, hashes, conditional uncertainty, local inference and submission validation.
- Offline tests with explicitly invented photometry, including saved-model replay.

The LightGBM default is verified locally. CatBoost imports, but fitting is blocked by the sandbox's unavailable `/proc/self/statm`. Its implementation remains optional and unverified here. The dedicated hybrid test is explicitly skipped locally and enabled in the hosted CI configuration.

**Local verification: 36 tests passed; one optional CatBoost integration test skipped.** The standalone CLI smoke run also completed (72 invented training objects, 12 invented test objects, 414 features). The test report in `software_validation.json` records the executed checks and source hashes. Fixture scores are intentionally not presented here as model performance. Full synthetic output, if generated locally, is labelled and kept under ignored `artifacts/`.

## Remaining work

1. Obtain the official competition files through authorized access.
2. Inspect real schema/cadence, class balance, redshift distributions and any available parent-source mapping.
3. Run the fixed nested protocol; audit feature missingness, fold membership and predictions.
4. Evaluate model/feature changes as separate experiments and document negative results.
5. If the user later requests a Kaggle submission, first verify whether late submissions are accepted. Only Kaggle evaluation can provide a comparable leaderboard score.

GitHub Actions is configured for offline tests, including the optional CatBoost check. The [first hosted run](https://github.com/Ehsan-Ch/mallorn-tde-classifier/actions/runs/36783703232) did not start because of an account-level restriction. It provides no code-test result. Local verification is recorded separately; hosted/hybrid verification remains pending.
