# Explain the project accurately

This repository was created with OpenAI Codex assistance. Run it, inspect the tests and understand the design before presenting it as evidence of your skills. Distinguish assistance from your own later experiments and contributions.

## One-minute explanation

“This project builds a reproducible classifier for rare astronomical tidal disruption events from irregular light curves. It uses features inspired by published leading MALLORN solutions and compares boosted-tree models. I focused on separating feature engineering, model selection, threshold tuning and evaluation, because optimizing F1 on the same predictions used to report performance can give an optimistic result. The code exports every fold and prediction so the experiment can be audited. On the official data, the fixed nested object-level evaluation achieved F1 0.5254, precision 0.4515 and recall 0.6284. Source-template independence is unresolved, and no Kaggle leaderboard score has been measured.”

The measured result and limitations are recorded in [the experiment report](../reports/official_001/RESULTS.md).

## Decisions to understand

- **Why F1?** TDEs are rare; an all-negative model can have high accuracy but finds none. Precision and recall must be reported alongside F1.
- **Why nested folds?** A threshold is a learned decision, just like a model parameter. The outer test labels cannot choose it.
- **Why object groups may not be enough?** Multiple simulated curves can originate from the same real transient. Genuine parent-source groups are needed to test independence at that level.
- **Why not interpolate every colour?** A seasonal gap or noisy negative flux can make a magnitude colour unsupported. Missing values are preferable to confident invented measurements.
- **Why no winner claim?** Local validation and the Kaggle private test set are different evaluations. Software fixtures measure neither.
- **Why boosting first?** Leading write-ups support strong engineered features and boosted trees. The first independent implementation is deliberately small enough to inspect; the neural/GP components of those solutions are not reproduced.

Start reading in this order: `data.py`, `features.py`, `modeling.py`, `cli.py`, then `tests/test_integrity.py`.
