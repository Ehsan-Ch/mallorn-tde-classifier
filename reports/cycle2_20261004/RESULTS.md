# Second-cycle validation — 4 October 2026

**The four-component ensemble passed the prespecified local selection rule. Final fitting is blocked by the current runtime; this candidate has not been submitted to Kaggle.** The best verified Kaggle result remains private F1 **0.6648**, public F1 **0.6825**.

## Comparable local results

All three recipes use the same 3,043 labeled objects and two stratified five-fold repetitions. Each object has two held-out probabilities, averaged before scoring. Thresholds are the median of 500 stratified-bootstrap F1-optimal thresholds. These are development selection scores, not independent test results.

| Fixed recipe | F1 | Precision | Recall | Average precision | Threshold |
| --- | ---: | ---: | ---: | ---: | ---: |
| Reference: submitted component types | 0.670659 | 0.602151 | 0.756757 | 0.660196 | 0.261459 |
| Selected binary + selected three-class | 0.675497 | 0.662338 | 0.689189 | 0.641331 | 0.313256 |
| **Diverse: all four components** | **0.684058** | **0.598985** | **0.797297** | **0.677698** | **0.238282** |

The diverse recipe improves local F1 by **0.013399** (1.34 percentage points) and average precision by **0.017502**. It meets the rule fixed in [the protocol](../../docs/SECOND_CYCLE_PROTOCOL.md): F1 improvement of at least 0.01, with average precision no more than 0.005 lower. The selected-class alternative fails that rule.

At its frozen threshold, the diverse ensemble's separate repetitions scored F1 **0.699708** and **0.672464**. Its combined confusion matrix is TN **2816**, FP **79**, FN **30**, TP **118**. The reference detects 112 positives with 74 false positives; the diverse ensemble detects six additional TDEs and produces five additional false positives.

## Frozen deployment recipe

Equal weights across physics three-class CatBoost, selected-feature binary CatBoost, selected-feature three-class CatBoost, and the hierarchical AGN/transient classifier. Fit each component with seeds **42 and 142** on all training objects, then average all eight probabilities. Classify as TDE at **0.23828184862427915** or above.

- Freeze timestamp: **2026-10-04T16:11:20.700492+00:00**.
- Frozen comparison SHA-256: `74e4bbc78934a5b506ec1dae502142b40b8b140d6df455b6c1a0afbb149301c6`.
- All 40 fold audits passed: no fitting/validation object overlap, exactly one held-out prediction per object per repetition, and no target, SpecType, Z_err or object_id predictor columns. Auxiliary SpecType labels and supervised feature selection are used only within fitting folds.
- No competition test labels were used. Historical labels have already been reused, and genuine source-template groups remain unavailable.

## Current blocker and continuation

Final fitting failed before saving any deployment model because native CatBoost could not open `/proc/self/statm`. The refreshed runtime's restricted filesystem does not expose that process-statistics file. Earlier completed training used a compatible runtime. No model change, threshold change or substitute submission was made to work around this failure.

In a runtime that permits CatBoost to read its process statistics, resume from the repository root:

```bash
MPLCONFIGDIR=artifacts/mpl OPENBLAS_NUM_THREADS=1 MALLORN_THREADS=6 \
  .venv/bin/python scripts/cycle2.py deploy
```

The command verifies the frozen implementation and feature hashes, fits missing models, and writes the candidate to `artifacts/cycle2/deployment/submission.csv`. Verify saved-model replay, the official object order, binary predictions and the file hash before submitting once to Kaggle. Record private and public scores separately. GitHub publication remains on hold until the requested competitive standard is demonstrated.

The new local F1 cannot be compared directly with the winner's private F1 **0.6824**. The existing verified private result **0.6648** remains **0.0176** below that benchmark. No official placement is claimed for a late submission.
