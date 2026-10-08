# Cycle-5 rebuild results

The reconstructed method completed all **20 fold checkpoints**, retaining **60 final LightGBM models**. All 20 saved checkpoints passed prediction replay verification (maximum absolute error 0.0).

| Recipe | Historical F1 | Rebuilt F1 | Rebuilt AP | Repeat 1 F1 | Repeat 2 F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| reference | 0.684058 | 0.684058 | 0.677698 | 0.699708 | 0.672464 |
| blend_selected_complete_fine5 | 0.689024 | 0.689024 | 0.704545 | 0.668693 | 0.660494 |
| blend_selected_physical_aux | 0.695906 | 0.695906 | 0.685389 | 0.688427 | 0.674487 |
| lgb_ensemble | 0.672897 | 0.672897 | 0.707030 | 0.645963 | 0.672956 |
| blend_lgb_ensemble | 0.688235 | 0.688235 | 0.699186 | 0.686391 | 0.680352 |

5 of 5 recipes match all four recorded historical metrics to six-decimal rounding. These comparisons use historical session records; the original cycle-5 model files remain unavailable, so byte-for-byte equivalence cannot be established.

The unchanged acceptance gates selected **reference**. These are repeated development-selection results, not independent validation or new Kaggle scores. The existing Kaggle submission remains private F1 0.6648 / public F1 0.6825.

The runner resumed the four checkpoints that survived the interruption, verified their experiment and row assignments, and fitted the remaining folds. The backup includes checkpoint hashes, training feature caches, the frozen reference predictions and the source needed to resume or audit. Preserve the verified ZIP separately: local files alone do not guarantee survival across workspace resets.

## Backup receipt

The complete private ZIP contains all 20 fold checkpoints and passed every archived-file SHA256 check. Size: 137,727,849 bytes. SHA256: `c37286dfbd79eb21942c1d58d42c4a88f6b16ebcc497a3f3ecf84a3914c874dd`. Persistent upload failed because the upload service could not read its authentication. The downloadable workspace archive must be retained outside this transient workspace. Source and aggregate verification receipts are saved on GitHub; the private ZIP and model files are not public.
