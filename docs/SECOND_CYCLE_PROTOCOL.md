# Second improvement cycle — 4 October 2026

Reference: the unchanged first submitted recipe scored private F1 0.6648 and
public F1 0.6825. Its reserved audit has been opened. This cycle uses all 3,043
labeled objects, and makes no claim to have a new untouched local test set.

Before looking at new results, fix two stratified five-fold repetitions with
split seeds 20261014 and 20261015. Each object receives one held-out prediction
per repetition. Features and any supervised column selection remain confined
to each fitting fold. No fitting uses competition test labels.

Compare four already motivated components: the submitted physics-only
three-class CatBoost model, the submitted selected-feature binary CatBoost
model, selected-feature three-class CatBoost, and the physical-feature
hierarchical AGN/transient classifier. Do not add new hyperparameter trials
within this cycle. Compare exactly three equal-weight recipes:

1. Reference: the two submitted component types.
2. Selected-class ensemble: selected binary plus selected three-class model.
3. Diverse ensemble: all four component types.

Average each object's two held-out predictions. Choose each recipe's threshold
as the median of 500 stratified-bootstrap F1-optimal development thresholds,
with bootstrap seed 20261016. Record both repeat-specific and combined scores,
precision, recall and average precision. These remain tuning/selection scores.

Prefer an alternative recipe only if its combined F1 exceeds the reference by
at least 0.01 and its average precision is no more than 0.005 lower. Otherwise
retain the reference component types. Refit the chosen recipe with two fixed
seeds (42 and 142) on all training objects and average the resulting probabilities.
Freeze the recipe and threshold before any new organizer-scored submission.

This experiment tests whether use of all labeled objects, repeated validation,
seed averaging and a less brittle threshold improve the prior result. The
first Kaggle result is a reference, not a source for per-object label inference
or a threshold sweep. The private leaderboard is now observed and cannot be
claimed as a never-used research test after further submissions.
