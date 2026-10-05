# Fourth cycle: confusing negative classes

Motivation, observed after cycle 3: the retained cycle-2 reference produces 79
false positives, including 33 AGN, 18 of the 41 SN IIn objects, and 10 of the
25 superluminous supernovae. Fine-grained class supervision and explicit
attention to those confusing negatives are concrete next hypotheses. This is
adaptive development on historically reused labels, not a new independent test.

Before fitting, fix two LightGBM candidates on the 1,126 complete features:

- `fine5`: five classes: AGN, ordinary supernovae, SN IIn, SLSN, and TDE. Use
  weights 1, 1, 1, 1, 3. Preserve the original binary target for all scoring.
- `hard_negative_binary`: binary target; sample weights 3 for TDE, 2 for SN IIn
  or SLSN, and 1 otherwise. These labels affect fitting-fold sample weights only.

Use the exact cycle-3 LightGBM hyperparameters and the same two five-fold
partitions. No extra hyperparameter trials or early stopping on validation.
Seeds remain 42 + 100*repeat + fold. All class construction, sample weights and
training operate on fitting rows only; SpecType is never a predictor.

Compare five fixed recipes: the cycle-2 diverse reference; its 50:50 blend with
each new candidate; the equal-weight new-candidate ensemble; and its 50:50 blend
with the reference. Apply cycle 3's unchanged threshold method and selection
rule: F1 improvement >=0.005, no AP decrease, and no individual repetition's F1
decrease greater than 0.005. Report all failures and uncertainty. Do not promote
cycle 3's 0.7030-F1 blend merely because it has a larger pooled tuning score.

Reuse the atomic fold checkpoint and model-replay checks. This cycle's source,
protocol, packages, inputs and reference predictions must be frozen separately.
No GitHub or Kaggle uploads. Stop the finite queue after comparison.
