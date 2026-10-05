"""Rerun the recovered cycle-5 recipe in a NEW namespace, preserving evidence."""
import argparse
import importlib.util
import json
from pathlib import Path
import time

import cycle3
from mallorn.selected_ensemble import fit, predict
from mallorn.checkpointing import atomic_json, exclusive_lock, freeze_json
from mallorn.data import sha256

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'artifacts/cycle5_recovered'
COMPONENTS = {'selected_complete_fine5': ('complete','fine5'),
              'selected_physical_aux': ('physical_view','aux')}


def inputs(name):
    view = COMPONENTS[name][0] if name in COMPONENTS else 'complete'
    return cycle3.inputs('lgb_complete_binary' if view == 'complete' else 'lgb_physical_hierarchy')


def run(seconds):
    with exclusive_lock(ROOT/'artifacts/research.lock'):
        manifest = json.loads(json.dumps(cycle3.initialize()))
        manifest.update(experiment='cycle5_recovered',components=COMPONENTS,
                        feature_limit=125,correlation_limit=.95,
                        selector_offsets=[10000,20000],final_offsets=[0,1000,2000])
        for path in ['scripts/cycle5_recovered.py','src/mallorn/selected_ensemble.py',
                     'docs/FIFTH_CYCLE_RECOVERY.md']:
            manifest['files'][path] = sha256(ROOT/path)
        manifest = json.loads(json.dumps(manifest))
        freeze_json(OUT/'manifest.json',manifest)
        if not (OUT/'start_receipt.json').exists():
            atomic_json(OUT/'start_receipt.json',{'started_at':cycle3.now(),'recovered_source':True})
        spec = importlib.util.spec_from_file_location('recovered_engine',ROOT/'scripts/cycle3.py')
        worker = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(worker)
        worker.OUT, worker.COMPONENTS = OUT, COMPONENTS
        worker.inputs, worker.fit, worker.predict = inputs, fit, predict
        complete = worker.run_folds(manifest,time.monotonic()+seconds)
        if complete and not (OUT/'comparison.json').exists():
            worker.compare(manifest)
        print(json.dumps({'complete':complete,'output':str(OUT)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--max-seconds',type=int,default=1500)
    args = parser.parse_args()
    if args.max_seconds < 1: parser.error('Time budget must be positive')
    run(args.max_seconds)
