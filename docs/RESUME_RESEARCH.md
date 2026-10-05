# Restartable local research

Current scope: research runners operate locally and never publish automatically.
On 5 October 2026 the user separately authorized a GitHub source update and a
Kaggle code notebook. See [publication checkpoint](PUBLICATION_CHECKPOINT.md).
This publication does not request another prediction submission or paid compute.
No Excel output. The competitive target is private F1 above 0.6824, but only
a future organizer-scored evaluation can confirm that. Best verified private
F1 remains 0.6648. Local development results must retain their own labels.

## Resume command

From the repository root, with dependencies installed:

```bash
MPLCONFIGDIR=artifacts/mpl OPENBLAS_NUM_THREADS=1 MALLORN_THREADS=4 \
  .venv/bin/python scripts/resume_research.py run --max-seconds 1500
```

Inspect progress without starting work:

```bash
.venv/bin/python scripts/resume_research.py status
```

In a fresh environment, create the virtual environment and install the pinned
project dependencies first. Do not change the frozen package versions to make a
checkpoint load. On Windows, set the environment variables with PowerShell and
use `.venv\Scripts\python.exe` instead. The runner's filesystem lock has Linux
and Windows implementations; only the Linux path has been exercised here.

## What continues automatically

The worker completes the fixed cycle-3 and cycle-4 queues without needing another chat
message for each fold. It saves each fold atomically and then moves to the next.
The wall-time budget is checked at fold boundaries, so a running fold may
finish slightly after that budget. On restart, completed checkpoints are
verified and skipped; only an interrupted, uncommitted fold is retrained.

An OS lock prevents two workers from writing the same experiment. The lock is
released if a process crashes. A changed configuration, package version, source
hash, input hash or validation assignment causes a visible error. Never delete
the manifest or edit it to force incompatible checkpoints to load.

The compatible LightGBM work runs before the blocked cycle-2 CatBoost full-data
refit. If CatBoost's process-statistics access becomes available, the worker
attempts that refit locally. Otherwise it records the blocker and exits. It
does not modify sandbox permissions or emulate the missing process file.
The new `scripts/refit_frozen.py` preserves the original fitting recipe while
saving each complete model and its probabilities atomically. It verifies saved
predictions on replay and writes the final output manifest last. This full-data
path is implemented but cannot be exercised until native CatBoost is available.

The bounded queue stops when complete. Repeated scheduled calls do not launch
more hyperparameter trials or change frozen thresholds. New research needs a
new written protocol and output directory, preserving all earlier outcomes.

## Limits of unattended continuation

There is no token-reset event exposed to this code. A scheduled chat check can
attempt continuation once usage allows, at most hourly with the available
scheduler. It does not bypass usage limits, keep this compute environment
alive, or guarantee access to this workspace in another run. Web scheduled
tasks do not preserve local folders between runs; see the [official scheduled
tasks documentation](https://learn.chatgpt.com/docs/automations).

If the workspace or its authorized data is unavailable, the scheduled task must
report the blocker and stop, rather than recreate results or read unrelated
files. Do not repeatedly notify the user about an unchanged blocker. A saved
experiment archive preserves expensive outputs; it does not by itself provide
a running machine or an automatic restore service.

## Continuation records

- `artifacts/research_state.json`: latest worker outcome and concrete blocker.
- `artifacts/cycle3/manifest.json`: frozen code, input hashes and package versions.
- `artifacts/cycle3/progress.json`: active fold or completion.
- `artifacts/cycle3/*/repeat*_fold*.joblib`: atomic fold model, probabilities and audit.
- `artifacts/cycle3/comparison.json`: all six prespecified recipes and selection.
- `artifacts/cycle4/`: the separate confusing-negative experiment, with the same
  atomic checkpoint structure and five fixed recipe comparisons.
- `artifacts/cycle2/frozen_comparison.json`: unchanged preceding candidate.

Do not load arbitrary untrusted joblib files. These checkpoints are produced
by this project from the user-authorized competition data.
