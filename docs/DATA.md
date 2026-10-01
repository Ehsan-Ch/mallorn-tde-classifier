# Data setup and execution

## Obtain the authorized files

Open the [official data page](https://www.kaggle.com/competitions/mallorn-astronomical-classification-challenge/data), sign in, review and accept the competition rules, and download the files. Rule acceptance is a user action. Do not share passwords or API credentials in a chat or commit them to GitHub.

Extract the archive so `train_log.csv`, `test_log.csv` and `sample_submission.csv` are directly inside `data/mallorn/`. The light-curve files may be in the official `split_01` through `split_20` folders; the loader discovers them recursively. Each folder contains `train_full_lightcurves.csv` and `test_full_lightcurves.csv`. A `split` field, if present in metadata, must match the parent folder name exactly.

The loader deliberately fails on missing files, unknown filters, duplicate observations, infinite flux, nonfinite times/errors, nonpositive errors, overlapping train/test IDs and inconsistent object coverage. Official observations with missing flux are omitted and counted by train/test partition in the manifest; no flux is imputed. Every metadata object must still have observations. Negative measured flux is retained.

The supplied archive contains one slightly negative test photometric redshift (`-0.01751`). Raw files and metadata remain unchanged. Feature extraction uses `max(Z, 0)` consistently for redshift features, rest-frame time and colour queries. The same fixed rule applies to train and test, requires no fitting, and was introduced before model evaluation. Z must be finite and greater than -1; EBV must be finite and nonnegative.

## Input schema

| File | Required columns | Treatment |
| --- | --- | --- |
| `train_log.csv` | `object_id`, `Z`, `EBV`, `target` | Binary target; finite Z > -1 and EBV >= 0; negative Z floored only in features |
| `test_log.csv` | `object_id`, `Z`, `EBV` | No target used |
| Light curves | `object_id`, `Time (MJD)`, `Flux`, `Flux_err`, `Filter` | Times in days, flux/errors in microjansky; filters u,g,r,i,z,y; missing flux observations omitted and audited |
| `sample_submission.csv` | `object_id`, `prediction` | All test IDs exactly once; original order preserved |
| Optional group file | `object_id`, `group_id` | Exactly one genuine source-template identifier per training object |

`SpecType`, the translation of the object name, file `split`, object ID, and `Z_err` are never classifier inputs. In particular, file splits are storage partitions, not assumed source-template families.

## Train and evaluate

```bash
mallorn run --data data/mallorn --output artifacts/run_001 --threads 2
```

If the organizers provide the true template/source mapping, enable group-aware splitting at both levels:

```bash
mallorn run --data data/mallorn --groups data/source_groups.csv --output artifacts/grouped_001
```

The group mapping must include exactly the training objects. Do not invent groups from the random object names or claim that unique object IDs solve template leakage.

For a separately labelled diagnostic run, parameters include `--iterations`, `--outer-folds`, `--inner-folds`, `--seed` and `--no-deredden`. These change the experiment and must be recorded; comparing many variants makes the existing local evaluation a development set. Keep a genuinely untouched period/dataset or organizer evaluation for final claims.

The locally verified backend is `--backend lightgbm` (default). `--backend hybrid` includes CatBoost and an equal-weight blend. CatBoost requires a host where its native runtime can read process statistics; it could not fit in this authoring sandbox. Before using hybrid results, run its optional integration check on a compatible host:

```bash
# Linux/macOS:
MALLORN_TEST_CATBOOST=1 python -m unittest discover -s tests -v
```

For PowerShell, set `$env:MALLORN_TEST_CATBOOST='1'` before the test command. This flag enables an additional genuine CatBoost fit/replay test; it does not replace CatBoost with a mock.

## Validate and replay

```bash
mallorn check-submission --submission artifacts/run_001/submission.csv --test-log data/mallorn/test_log.csv
mallorn predict --data data/mallorn --model artifacts/run_001/model.joblib --output artifacts/replayed_submission.csv
python scripts/verify_run.py --data data/mallorn --run artifacts/run_001
```

`verify_run.py` checks input hashes, fold boundaries, held-out labels/F1, saved-model replay from the hashed feature table, and exact submission order/decisions. It writes `verification.json`. `predict` reuses the saved feature configuration and does not fit a model. The current loader expects the original dataset layout, including training logs, for its object-coverage contract. Only test rows are sent to the saved estimator. `joblib` uses pickle: load only artifacts you produced and trust.

There is no Kaggle submission call, credential handling, automatic rule acceptance, or notebook publication in the package. Local outputs stay under the git-ignored `artifacts/` directory.

## Provenance and storage

The manifest records SHA-256 hashes and sizes of all inputs, feature names/configuration, the feature CSV hash, and an optional group-map hash. Metrics record package versions and the training configuration. Input data, feature tables, models and prediction files are ignored by git. Publish aggregate findings only after confirming the data-use terms. Exact replay needs the same files, core package versions and configuration; floating-point differences can occur across hardware.
