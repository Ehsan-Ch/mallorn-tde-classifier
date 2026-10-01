# MALLORN TDE Classifier

**Machine learning for finding rare tidal disruption events in irregular, six-band astronomical light curves.**

A reproducible Python pipeline inspired by the published second- and third-place MALLORN solutions: domain-informed light-curve features, gradient-boosted trees, and F1 threshold selection. The default compares basic and full LightGBM feature sets under nested validation. An optional hybrid backend adds CatBoost and a fixed blend.

> **Measured on the official dataset: nested local F1 0.5254.**
> Trained and evaluated on 3,043 labelled objects, with 148 TDEs. Generated and verified predictions for 7,135 test objects. No Kaggle submission or leaderboard score is claimed.

| F1 | Precision | Recall | Validation |
| ---: | ---: | ---: | --- |
| **0.5254** | 0.4515 | 0.6284 | 5 outer × 3 inner folds; model and threshold chosen inside training folds |

[**Read the experiment results →**](reports/official_001/RESULTS.md)

[Methodology](docs/METHODOLOGY.md) · [Data setup](docs/DATA.md) · [Leading-solution references](docs/SOURCES.md) · [Verification status](reports/STATUS.md)

## What this project demonstrates

| Research skill | Implementation |
| --- | --- |
| Irregular time series | Per-band statistics, detection duration, seasonal gaps and weighted Lomb–Scargle features |
| Domain-informed features | Approximate dust correction, rest-frame timing, common-phase colours and descriptive post-peak decline |
| Rare-event modelling | LightGBM, simple baselines and F1 threshold selection; optional CatBoost blend |
| Reliable evaluation | Inner folds choose model and threshold; outer folds estimate the complete selection workflow |
| Reproducibility | File hashes, fixed seeds, pinned core packages, per-object predictions and fold audits |
| Inference | Saved model, exact feature-schema checks and a validated binary submission file |

## Approach and boundaries

The [second-place write-up](https://www.kaggle.com/competitions/mallorn-astronomical-classification-challenge/writeups/mallorn-2nd-place-solution) informed the emphasis on shape, phase bins, colours and boosting ensembles. The [third-place write-up](https://www.kaggle.com/competitions/mallorn-astronomical-classification-challenge/writeups/3rd-place-write-up-catboost-with-threshold-tuning) supports physics-informed features, CatBoost and threshold tuning.

This is an independent implementation of a tractable subset of those ideas. It does **not** reproduce the winners' full systems: pretrained neural networks, two-dimensional Gaussian processes, blackbody/SALT2 fits and large ensembles are not implemented. [Sources and differences](docs/SOURCES.md) distinguish implemented work from possible extensions.

```mermaid
flowchart TD
    A[Authorized competition files] --> B[Validate and extract per-object features]
    B --> C[Inner folds select model and threshold]
    C --> D[Outer folds evaluate frozen decisions]
    D --> E[Export predictions and audit report]
    B --> F[Choose final recipe using OOF predictions]
    E --> F
    F --> G[Refit and generate local submission]
```

## Run locally

Use Python **3.11+** in a virtual environment. The implementation was tested with Python 3.12. A CPU is sufficient; no paid service is required by the code.

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell instead:
# .venv\Scripts\Activate.ps1
python -m pip install -e .
python -m unittest discover -s tests -v
```

After accepting Kaggle's rules and extracting the official files into `data/mallorn/`:

```bash
mallorn run --data data/mallorn --output artifacts/run_001
```

The default run performs five outer folds, three inner folds and 500 boosting iterations per model. Use a new output directory for each experiment. Full options and the expected file layout are in [Data setup](docs/DATA.md).

`--backend hybrid` adds CatBoost and its equal-weight blend with LightGBM. CatBoost fitting could not be verified in the authoring sandbox because a required process-statistics file is unavailable. The default LightGBM workflow is the locally tested path; optional-backend verification remains pending on a compatible host.

To check the software without competition data:

```bash
mallorn smoke --output artifacts/synthetic_smoke
```

The smoke run generates explicitly invented curves in a temporary directory. Its report is prominently marked **not competition performance**, and its output is named `synthetic_submission.csv`.

## Outputs from a data run

| File inside your run directory | Purpose |
| --- | --- |
| `RESULTS.md`, `figures/validation.svg` | Nested-validation results and plots |
| `metrics.json`, `oof.csv` | Exact scores, held-out probabilities and decisions |
| `fold_audit.json` | Every fitting, tuning and test object ID |
| `manifest.json`, `features.csv` | Input hashes, environment details in metrics, and feature matrix |
| `model.joblib` | Locally fitted deployment model |
| `submission.csv` | Binary decisions in sample-submission order; never uploaded automatically |

The primary score is the **outer nested F1**. The threshold selected from all out-of-fold predictions for the final model is a deployment tuning step; its best tuning F1 is not an unbiased performance estimate. Even valid local F1 cannot establish that this model beats a private Kaggle leaderboard score.

## Remaining limitations

- MALLORN curves are simulated from real source observations. Related simulations may cross object-level folds unless genuine source-template groups are supplied with `--groups`.
- Training redshifts are spectroscopic; test redshifts are photometric. `Z_err` is excluded, but the remaining measurement shift still needs evaluation.
- Dust correction uses representative wavelengths, and colours use bounded linear interpolation. Both are approximations; sparse or unreliable colours remain missing.
- This classifies full light curves. It is not an early-alert classifier or a deployed astronomical discovery system.

## Attribution and license

Created with **OpenAI Codex assistance for Ehsan Cheraghi**. The package is independently written, with methodology references credited in [SOURCES.md](docs/SOURCES.md). It is a portfolio research project, not a claim of Kaggle participation or placement.

Code: [MIT](LICENSE). Competition data remain subject to [Kaggle's competition rules](https://www.kaggle.com/competitions/mallorn-astronomical-classification-challenge/rules); they, trained models and per-object outputs are excluded from this repository. Only aggregate experiment evidence and hashes are published; raw observations, per-object features and predictions remain outside the public repository.
