"""Resume the finite, local-only MALLORN experiment queue; never publishes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

from mallorn.checkpointing import AlreadyRunning, atomic_json, exclusive_lock
import cycle3
import cycle4
from report_research import render

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT/'artifacts/research_state.json'


def catboost_access():
    if sys.platform.startswith('linux'):
        try:
            with open('/proc/self/statm') as handle:
                handle.read(100)
        except OSError:
            return False
    return True


def status():
    return {'cycle3_completed_folds': sum(1 for _ in (ROOT/'artifacts/cycle3').glob('*/repeat*_fold*.joblib')),
            'cycle3_total_folds': 30,
            'cycle3_compared': (ROOT/'artifacts/cycle3/comparison.json').exists(),
            'cycle4_completed_folds': sum(1 for _ in (ROOT/'artifacts/cycle4').glob('*/repeat*_fold*.joblib')),
            'cycle4_total_folds': 20,
            'cycle4_compared': (ROOT/'artifacts/cycle4/comparison.json').exists(),
            'cycle2_refit_complete': (ROOT/'artifacts/cycle2/deployment/manifest.json').exists(),
            'catboost_process_statistics_access': catboost_access(),
            'publishing_allowed': False,
            'previous_run': json.loads(STATE.read_text()) if STATE.exists() else None}


def run(seconds, max_folds=None):
    deadline = time.monotonic()+seconds
    with exclusive_lock(ROOT/'artifacts/research.lock'):
        atomic_json(STATE, {'state': 'running', 'started_at': cycle3.now(), 'publishing_allowed': False})
        try:
            manifest = cycle3.initialize()
            # Verify committed folds even when rerun after successful completion.
            complete = cycle3.run_folds(manifest, deadline, max_folds)
            if complete and not (cycle3.OUT/'comparison.json').exists():
                cycle3.compare(manifest)
            if complete:
                render(3)
            cycle4_complete = False
            if complete and time.monotonic() < deadline:
                fourth_manifest = cycle4.initialize(manifest)
                fourth = cycle4.engine()
                cycle4_complete = fourth.run_folds(fourth_manifest, deadline, max_folds)
                if cycle4_complete and not (cycle4.OUT/'comparison.json').exists():
                    fourth.compare(fourth_manifest)
                if cycle4_complete:
                    render(4)
            queue_complete = complete and cycle4_complete
            blocked = []
            cat_done = (ROOT/'artifacts/cycle2/deployment/manifest.json').exists()
            if not cat_done:
                if not catboost_access():
                    blocked.append('cycle2_refit: CatBoost cannot read /proc/self/statm')
                elif queue_complete and max_folds is None and time.monotonic() < deadline:
                    # Atomic full-data checkpoints preserve completed siblings if
                    # a timeout interrupts training or serialization of one model.
                    with (ROOT/'artifacts/cycle2-resume-deployment.log').open('a') as log:
                        subprocess.run([sys.executable, 'scripts/refit_frozen.py'], cwd=ROOT,
                                       stdout=log, stderr=subprocess.STDOUT,
                                       timeout=max(1, deadline-time.monotonic()), check=True)
                    cat_done = True
            state = 'complete' if queue_complete and cat_done else 'blocked' if queue_complete and blocked else 'checkpointed'
            result = {'state': state, 'updated_at': cycle3.now(), 'cycle3_complete': complete,
                      'cycle4_complete': cycle4_complete,
                      'cycle2_refit_complete': cat_done, 'blocked': blocked, 'publishing_allowed': False}
            atomic_json(STATE, result)
            return result
        except BaseException as exc:
            atomic_json(STATE, {'state': 'interrupted' if isinstance(exc, (KeyboardInterrupt, subprocess.TimeoutExpired)) else 'failed',
                              'updated_at': cycle3.now(), 'error': str(exc), 'publishing_allowed': False})
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['run', 'status'])
    parser.add_argument('--max-seconds', type=int, default=1500)
    parser.add_argument('--max-new-folds', type=int)
    args = parser.parse_args()
    if args.max_seconds < 1 or (args.max_new_folds is not None and args.max_new_folds < 1):
        parser.error('Limits must be positive')
    try:
        print(json.dumps(status() if args.action == 'status' else run(args.max_seconds, args.max_new_folds), indent=2))
    except AlreadyRunning:
        print(json.dumps({'state': 'already_running', 'action': 'No duplicate worker started'}))
    except Exception:
        traceback.print_exc()
        sys.exit(1)
