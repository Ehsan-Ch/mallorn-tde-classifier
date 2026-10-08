# Publication checkpoint — 5 October 2026

User authorization: update the MALLORN project on GitHub and upload its code as
a Kaggle notebook. This supersedes the earlier pause on code publication.
It does not request another competition prediction submission.

## Verified starting state

- GitHub: `Ehsan-Ch/mallorn-tde-classifier`, branch `main`, commit
  `d05fdd34816264360a9050bb24f7d79d2b929d19`.
- Kaggle account: `ehsan1993`, display name Ehsan.Ch. Your Work / Code was empty.
- Existing prediction submission: `mallorn-improvement-submission.csv`, private
  F1 0.6648, public F1 0.6825, completed after the competition deadline.
- The workspace was restored to local commit `5c71f33`, containing research
  through cycles 3 and 4. The later cycle-5 fitted models and the 438,808,626-byte
  `mallorn-recovery-20261005.zip` are not present in this restored workspace.
  Do not claim those artifacts have been recovered. Their earlier reported
  outcomes remain historical session evidence, not freshly replayed results.

## Publication steps

1. Prepare a self-contained Kaggle notebook and current README/status notes.
2. Validate notebook structure, embedded source and a synthetic smoke run.
3. Publish GitHub files atomically as one commit based on the current main tip.
   Verify remote tree hashes before recording GitHub as complete.
4. Inspect Kaggle Your Work before creating anything. If the notebook exists,
   update that exact notebook; do not create duplicates after an interruption.
5. Upload `notebooks/mallorn_research.ipynb`, save a version, verify its page
   and version status, and record its exact URL.
6. Publish final links and version receipts to this repository.

## Resume after the user says continue

Read this file and `publication/state.json` in the remote GitHub repository
first. Remote state is authoritative if the local workspace rolled back.
Verify the existing Kaggle notebook URL/version before any new upload. An
in-flight save must be inspected, never assumed failed. Continue only incomplete
steps. Keep historical model and leaderboard scores distinct. Never insert a
reported score into a notebook's execution outputs as if that run computed it.

Token-reset detection is unavailable. The user gives the continuation signal.
Stored source and publication receipts survive through GitHub; running processes
and unuploaded model artifacts cannot be guaranteed across workspace resets.

## Completed publication — verified 7 October 2026

- GitHub source commit: `6c86d32927bfb76edbd972ef10d9324dc3bee3d7`; all 66 changed file hashes verified.
- Kaggle bootstrap fix: `1558f2d17d6e63a2c5eb56af6de874ecede25d4a`; all four changed file hashes verified. The environment is created without ensurepip and populated by the host pip.
- Kaggle version 2 / script version `355572876`: successfully executed in 66.4 seconds, private visibility.
- Exact saved version: https://www.kaggle.com/code/ehsan1993/mallorn-reproducible-tde-research?scriptVersionId=355572876
- All five default code cells completed; feature extraction, synthetic LightGBM fit, exact saved-model replay and prediction schema checks passed. Official data training stayed disabled.
- Version 1 failed during ensurepip bootstrap; version 2 supersedes it. No extra notebook or competition submission was created.
- Publication is complete. Resume by checking remote state; do not repeat uploads. Original cycle-5 trained artifacts remain unavailable.


## Subsequent cycle-5 rebuild — 8 October 2026

The original binaries remain unavailable, but rebuilding is complete. See the
existing public [rebuild results](../reports/CYCLE5_REBUILD_RESULTS.md).
This does not change Kaggle version 2: that saved notebook is the earlier
source snapshot and has not received the rebuilt private model files.
