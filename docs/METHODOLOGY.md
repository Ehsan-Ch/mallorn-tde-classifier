# Experiment protocol

Written before any competition-data training run. The current task is binary full-light-curve classification: TDE is 1, all other types are 0. F1 is the primary metric. No real competition score is available at this stage.

## Feature extraction

Each object is processed independently. No labels, population imputation, feature selection or train/test distribution fitting occurs during extraction. Missing bands retain a fixed schema. Each learned imputer is fitted inside its own model's fitting fold.

1. Validate the official photometry and metadata contracts.
2. Optionally de-redden flux and its error using Fitzpatrick (1999), Rv=3.1, evaluated at fixed representative band wavelengths. Scaling both preserves SNR. This is a monochromatic approximation, not a passband-integrated SED correction.
3. Reference time is the highest-SNR observation in g/r/i, falling back to all bands if none exist. Time relative to this reference is divided by 1+Z. It is a practical phase proxy, not an estimate of the true physical peak with fitted uncertainty.
4. Compute per-band count, time coverage, gaps, flux quantiles and shape, inverse-variance mean, constant-flux chi-square, SNR and detection-width statistics. Negative flux is retained.
5. Bin normalized flux and SNR by fixed rest-frame phase intervals, retaining counts to expose sparsity. The bins are fixed before fitting.
6. For positive detections after phase +5 days, compute a weighted descriptive log-flux/log-time slope and its unweighted fit R-squared. It is not a physical TDE likelihood fit.
7. Compute a weighted floating-mean Lomb–Scargle summary where coverage permits. No claim of physical periodicity is inferred from the largest sampled periodogram value.
8. Compute pairwise AB colour differences from positive flux at shared phases -10, 0, +15, +30 and +60 days. Linear interpolation is bounded by observed samples, requires both endpoint SNRs >=2, and never bridges an observer-frame gap over 60 days. Unsupported colours remain missing.

Redshift error is excluded because the official train and test sets have different availability. Using supplied Z still leaves a spectroscopic/photometric measurement shift. It must be studied once data are available. Synthetic redshift noise is not silently introduced.

## Fixed candidate set

| Candidate | Features | Model |
| --- | --- | --- |
| `prior` | None | Training positive-class fraction |
| `lgb_basic` | Statistics, SNR and timing; no phase bins, colours, decline or period features | LightGBM |
| `lgb_full` | All features | LightGBM |
| `cat_full` (hybrid only) | All features | CatBoost |
| `blend` (hybrid only) | All features | Equal mean of cat_full and lgb_full probabilities |

All-negative and all-positive baselines are also reported. CatBoost uses depth 5, learning rate .04 and L2=6. LightGBM uses 15 leaves, learning rate .035, L2=5, minimum child samples=20 and column fraction .85. Both default to 500 iterations. Full settings are in `modeling.py` and the exported configuration. No early stopping sees an outer test fold. No SMOTE, pseudo-labels or test data are used for training.

Threshold tuning addresses the F1/imbalance tradeoff; synthetic oversampling and class weighting are not presumed beneficial. The blend weights are fixed, not optimized on test results. The basic/full LightGBM comparison is an ablation of the added features. `lightgbm` is the default backend. The optional `hybrid` backend adds CatBoost, whose local execution was blocked by unavailable `/proc/self/statm` in the authoring sandbox. No CatBoost score or successful fit is claimed from that environment.

## Nested evaluation

Default: five outer folds and three inner folds, stratified by target. With genuine source-group IDs, both levels use stratified group splits and validate group disjointness. If no such mapping exists, reports explicitly say `object_only`; they do not claim source-template independence.

For each outer fold:

1. Keep its test objects out of every fitting and selection operation.
2. Generate inner out-of-fold probabilities for every candidate using only the outer training portion. Fit all missing-value imputers on the corresponding inner fit rows.
3. For each candidate, choose the threshold maximizing inner F1. Equal F1 thresholds select the higher threshold. Candidate ties prefer the earlier, simpler candidate in the declared order.
4. Refit on the outer training portion, freeze the selected candidate and threshold, and score the outer test portion once.
5. Export every split membership, tuning choice, probability and binary decision.

The primary reported F1 pools the outer test decisions from the complete nested selection procedure. Per-candidate outer scores evaluate their inner-selected thresholds. PR-AUC (average precision), precision, recall, ROC AUC, prevalence and confusion counts supplement F1.

The conditional bootstrap resamples fixed OOF decisions by object, or by source group when supplied. It does not rerun training or account for previous human/agent choices, feature exploration, or model selection across repeated experiments. It is descriptive, not proof of leaderboard superiority.

## Deployment recipe

Use candidate OOF probabilities across the training set to choose a final candidate and threshold; explicitly label that optimum as a tuning score. Refit that recipe on all labelled training objects and predict the test set. OOF-to-full-refit probability distribution changes are a known threshold-transfer risk.

Submission is exactly `object_id,prediction`, with binary integers and the official sample order. The package never submits it to Kaggle. Test labels are unavailable, so private leaderboard performance cannot be computed locally.

## Interpretation

Local object-level validation may overestimate generalization when several simulated objects share an original source. Sparse cadence, redshift shifts, tiny positive counts and repeated development also matter. This first protocol prioritizes auditability and a tractable boosted-tree baseline inspired by top solutions. Adding pretrained neural networks, Gaussian processes, physical fitting or larger ensembles would be a new, separately evaluated experiment.
