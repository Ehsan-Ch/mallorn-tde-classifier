# Sources and attribution

Reviewed 30 September 2026. These sources inform the design; no competitor code, pretrained weights, raw data, feature matrices or prediction files were copied.

| Source | What informed this implementation | Difference |
| --- | --- | --- |
| [Official competition](https://www.kaggle.com/competitions/mallorn-astronomical-classification-challenge) and [data page](https://www.kaggle.com/competitions/mallorn-astronomical-classification-challenge/data) | Binary TDE target, F1 metric, six-band light curves and submission schema | Dataset access pending; no organizer score obtained |
| [Suxing: second-place solution](https://www.kaggle.com/competitions/mallorn-astronomical-classification-challenge/writeups/mallorn-2nd-place-solution) | Shape, phase-binned and colour features; rest-frame timing and boosted-tree ensembles | Small fixed CatBoost/LightGBM candidate set; no pretrained CNN, feature-view ensemble search or post-processing zones |
| [Sigrid Nissen: third-place solution](https://www.kaggle.com/competitions/mallorn-astronomical-classification-challenge/writeups/3rd-place-write-up-catboost-with-threshold-tuning) and [author repository](https://github.com/sigridhn/MALLORN-solution-sigrid) | Physics-informed light-curve characterization, CatBoost, detection duration and threshold tuning | Bounded linear colour interpolation; no 2D GP, blackbody model, SALT2 or echo-state network |
| [Magill et al.: MALLORN paper](https://arxiv.org/abs/2512.04946) | Simulated light curves based on real source observations and the train/test redshift distinction | No claim that object-level folds remove common-source dependence |
| [Fitzpatrick99 implementation](https://extinction.readthedocs.io/en/latest/api/extinction.fitzpatrick99.html) | Dust extinction correction in the `extinction` library | Representative wavelength approximation, not integrated photometry |
| [SciPy Lomb–Scargle](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.lombscargle.html) | Weighted periodogram for uneven sampling | Descriptive features only; no significance or physical-period claim |
| [CatBoost API](https://catboost.ai/docs/en/concepts/python-reference_catboostclassifier) | CPU classifier configuration | Independently fixed settings |

The finalized leaderboard and later write-up titles can describe different rankings. A write-up titled “2nd Place Private LB (0.6804)” also identifies itself as a fifth-place solution with a late-submission result. It was not used as evidence of the official second-place finish. This project does not claim to reproduce an unverified first-place method.

Nested model/threshold selection, source hashes, strict file contracts, synthetic-only integrity fixtures and the evidence export are implementation choices for this repository. Their usefulness must be separated from empirical performance: no feature set is called superior without running the corresponding experiment on authorized data.
