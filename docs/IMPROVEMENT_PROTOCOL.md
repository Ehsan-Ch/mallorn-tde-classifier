# Local improvement protocol — 3 October 2026

No GitHub publication during model development. The target is approximately
0.68 F1, reflecting the leaders' reported private-test F1, not classification
precision alone. Local results cannot establish leaderboard equivalence.

The existing official_001 baseline is frozen. It has already evaluated all
training labels, so no subset of these data can be called historically unseen.
For this improvement cycle, reserve 25% of objects using stratified splitting
with seed 20261003. Do not inspect its outcomes while developing features,
models or thresholds. The remaining 75% is the development set. Use five-fold
stratified development OOF predictions (seed 20261004) for candidate and
threshold selection; label these scores explicitly as tuning estimates.

Investigate label-free light-curve shape, luminosity proxies, season-aware
interpolation, colors, analytic flare models and Gaussian-process summaries,
guided by the published second- and third-place solutions. Compare CatBoost,
LightGBM, class weighting and fixed probability ensembles. Never infer source
families from names or storage splits. No competition test labels, pseudo-labels
or competitor feature files enter fitting. If auxiliary class labels are used,
they are training-fold supervision only, never predictor features.

Freeze the chosen feature recipe, model configuration, ensemble and threshold
before evaluating the reserved audit subset once. Report precision, recall,
F1, counts and uncertainty even if the target is missed. Do not repeatedly tune
against the audit outcome. Any later cycle must disclose that reuse and cannot
call it independent confirmation. Record all candidate results and failures.

No result is guaranteed. All original-source independence concerns remain;
an organizer test-set evaluation is necessary for competition-level claims.
