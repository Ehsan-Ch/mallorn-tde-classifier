# Local improvement cycle

The public baseline remains unchanged while this experiment is evaluated.
The target is approximately 0.68 F1. Precision and recall are reported
separately; a local F1 is not a Kaggle private-test score.

## Features

Starting from the original 414 features, the physical extractor adds 471
measurements. A two-dimensional Gaussian process smooths flux over rest-frame
time and log wavelength. Its summaries include peak widths, time scales,
phase-dependent colors and their evolution. Approximate monochromatic
blackbody fits summarize temperature, radius and luminosity at several phases.
Per-band luminosity proxies and season-aware resampled shapes complement these
features. The GP uses at most 160 observations per object, retaining both
strong detections and time coverage.

The morphology extractor adds 241 measurements: detection spans, flux/SNR
distribution shape, band ratios, and Gaussian-rise flare fits with exponential
or power-law decay. A focused physical view adds 24 derived measurements,
including temperature/radius/luminosity changes in physical units. These are
descriptive approximations, not precise astrophysical parameter estimates.

Every extractor works on one object's photometry, redshift and extinction.
It does not use targets, spectral class labels, other objects' labels, or
competitor feature tables. Negative measured flux is retained.

## Models and supervised feature selection

CatBoost and LightGBM are compared using the same five development folds.
Binary classifiers are supplemented by three-class supervision (AGN, other
transients, TDE), a six-class variant, and a two-stage classifier that separates
AGN before discriminating TDE from other transients. Spectral class is used
only as a training-fold target, never as a prediction-time input.

Selected-feature candidates fit a preliminary CatBoost model on the training
fold, rank its feature importances, and retain 125 or 250 columns while pruning
training-fold correlations above 0.95. Validation objects never participate in
ranking or correlation estimation. Equal-weight probability ensembles are
compared; every tried recipe and optimized threshold is recorded.

## Evaluation boundaries

See [IMPROVEMENT_PROTOCOL.md](IMPROVEMENT_PROTOCOL.md). Development has 2,282
objects, including 111 TDEs; the reserved audit has 761 objects. Development
scores include candidate and threshold selection and therefore are optimistic.
The audit is reserved for this cycle, but the earlier published baseline had
already used all training labels. Real source/template groups are unavailable.

`audit_improvement.py freeze` writes a recipe with feature, split, metadata and
implementation hashes. The separate `audit` command fits on development only,
fixes predictions at the frozen threshold, and then scores reserved outcomes.
It cannot replace a completed audit with another tuned recipe. Interrupted
audits can resume only with the identical recipe.

## Reproduce locally

Use the pinned project dependencies and place the official competition data
under `data/mallorn`. The published baseline feature cache must first exist.

```bash
export MPLCONFIGDIR=artifacts/mpl
export OPENBLAS_NUM_THREADS=1
export MALLORN_THREADS=3
python scripts/build_physics_features.py --mode train
python scripts/build_morphology_features.py --mode train
python scripts/build_physical_view.py --mode train
python scripts/improve.py --features artifacts/improvement/complete_train.csv \
  --label complete_selected --batch selected
```

Other recorded batches and feature paths appear in each experiment's
`config.json`. Build test features using the same scripts with `--mode test`.
Training requires CatBoost to read its own process memory statistics; some
restricted runtimes hide that file. Do not bypass runtime access controls.

## Method references

- [Second-place write-up](https://www.kaggle.com/competitions/mallorn-astronomical-classification-challenge/writeups/mallorn-2nd-place-solution)
- [Third-place write-up](https://www.kaggle.com/competitions/mallorn-astronomical-classification-challenge/writeups/3rd-place-write-up-catboost-with-threshold-tuning)
- [Avocado paper](https://arxiv.org/abs/1907.04690)

These motivated the feature families and model comparisons. This code is an
independent implementation and does not reproduce their full pipelines.
