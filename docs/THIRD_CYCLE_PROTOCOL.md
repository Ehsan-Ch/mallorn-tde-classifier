# Third cycle: model diversity, fixed before training

The goal is to improve the second cycle's local F1 0.684058 and average
precision 0.677698. All labels have historical exposure; these are development
experiments, not untouched validation. The verified Kaggle private score remains
0.6648; the top private benchmark is 0.6824. Neither a higher development score
nor repeated tuning establishes that the benchmark has been surpassed.

Use the same two stratified five-fold partitions (20261014, 20261015) and
object order as cycle 2. Freeze three LightGBM components:

1. Three-class AGN/other-transient/TDE supervision on 471 physics features.
2. An AGN/transient gate multiplied by a conditional TDE classifier on the
   736 physical-view features; fit the second stage only on training transients.
3. Binary classification on the 1,126 complete features.

Each estimator uses 900 trees, learning rate 0.025, nine leaves, at least
30 objects per leaf, L2 penalty 10, column sampling 0.8, row sampling 0.85
every iteration, and class weights 1:3 for TDE versus other classes. The gate
uses equal class weights. Seed is 42 + 100*repeat + fold. No early stopping on
outer validation labels, feature selection, extra trials or hyperparameter sweep.

Evaluate exactly six recipes: the frozen cycle-2 diverse reference; a 50:50
blend of that reference with each of the three new components; the equal-weight
three-LightGBM ensemble; and a 50:50 blend of that ensemble with the reference.
Average repeated held-out probabilities per object. Threshold selection uses
the same 500 stratified bootstraps as cycle 2 (seed 20261016).

Select an alternative only if combined F1 improves by at least 0.005, average
precision does not decrease, and neither repetition's F1 at the fixed threshold
is more than 0.005 below the reference's corresponding repetition. If none
qualifies, retain the reference. Break equal-F1 ties by recipe declaration order.
Report every recipe, not just the winner. Record bootstrap paired F1 differences
at frozen thresholds as descriptive uncertainty; these intervals do not correct
for historical tuning or simulated-source dependence.

Save each complete fitted fold, held-out probabilities, row IDs and input/source
hashes atomically. An incomplete fold is rerun; committed folds are verified and
reused. An OS lock prevents concurrent writers. Never silently reuse a checkpoint
with a changed specification, code, data, package version or validation assignment.

Only local research is authorized in this cycle. Do not upload to GitHub or
Kaggle. A completed queue must stop; failed work records an error rather than
retrying indefinitely. CatBoost's blocked full-data refit is a separate pending
task and must not prevent compatible LightGBM work.
