# Cycle-5 recovery investigation — 7 October 2026

**Conclusion:** the original cycle-5 trained models were not found in the accessible workspace or surviving archives. Rebuilding the reconstructed method is feasible with the surviving inputs. Full cycle-5 training has not been started.

## Recovery search

- Searched workspace files, including ignored model/checkpoint directories and notebook extraction directories. Cycle-5 matches are reconstructed source copies, not trained models.
- Inspected the official, improvement and cycle-2 ZIP inventories. None contains cycle-5 artifacts.
- `mallorn-research-checkpoint-20261004.zip` is incomplete/unreadable as a ZIP archive (38,127,104 bytes). Preserve it; do not use it as the sole backup.
- Git object inspection found no unreachable cycle-5 commits. The sole unreachable blob is a plot SVG.
- The connected GitHub repository has no releases containing a recovery archive. The published Kaggle version runs a synthetic check; it did not train cycle-5 models.
- These findings cover the accessible sources only. No search of unrelated Library files or other conversations was performed.

Exact restoration remains possible if the original `mallorn-recovery-20261005.zip` survives outside these sources. Its recorded size is 438,808,626 bytes and SHA256 is `37552784035ba990aa6f30e1cca665fe9ab6ec413978920a9812d8f826beb49d`. Verify that hash before using a supplied copy.

## Verified rebuild prerequisites

| Item | Result |
| --- | --- |
| Cycle-3 and cycle-4 input/source manifests | All 17 and 20 recorded file hashes match, respectively |
| Complete training features | 3,043 rows, 1,126 columns |
| Physical-view training features | 3,043 rows, 736 columns |
| Positive labels | 148; metadata and feature row IDs align |
| Frozen cycle-2 reference | Eight probability arrays available, finite, correctly shaped and in [0, 1] |
| Surviving fold models | 30 cycle-3 and 20 cycle-4 checkpoints |
| Runtime replay | One cycle-3 saved fold model reproduced its recorded validation probabilities exactly; maximum error 0.0 |
| Reconstructed method tests | Four tests passed, including training-only selection and exact saved-model replay |

The restored default Python environment lacked LightGBM. An isolated preflight environment at `artifacts/recovery_preflight_env` was created with LightGBM 4.6.0; the existing NumPy 2.3.5, pandas 2.2.3, scikit-learn 1.8.0 and joblib 1.5.3 versions match the historical manifests. This rebuild uses LightGBM and cached reference predictions, so it does not need to refit CatBoost or regenerate photometry features.

## Rebuild procedure

1. Recheck the manifests and package versions after any workspace reset. Confirm there is no active research process.
2. Run the reconstructed code with four threads and a bounded time budget:

```bash
PYTHONPATH=src:scripts OPENBLAS_NUM_THREADS=1 MALLORN_THREADS=4 \
  artifacts/recovery_preflight_env/bin/python scripts/cycle5_recovered.py --max-seconds 1500
```

3. Repeat the command to resume. It preserves completed matching folds and checks their manifest and row assignments. The time budget is checked between folds, so a running fold may finish after it expires.
4. Expect two components × two repeats × five folds = **20 fold checkpoints**. Each performs two selector fits and three final fits: **100 LightGBM fits**, with 60 final models retained across the checkpoints.
5. After training, the existing comparison code checks saved-model replay, computes the fixed blends, and applies the unchanged acceptance gates. Compare new results with the historical record; do not assume equality or improvement.
6. Save and verify a complete backup of the new checkpoint directory before ending the work. Local atomic files support process restarts but cannot guarantee survival of a workspace rollback. Do not publish object-level artifacts or make a new competition submission as part of this investigation.

Rebuilt artifacts belong under `artifacts/cycle5_recovered`. They are newly trained models from reconstructed source, not recovered originals. No public-visibility change, full training run or competition submission was performed during this investigation.
