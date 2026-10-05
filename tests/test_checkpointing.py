"""Failure-mode tests: interruption, conflicting work and invalid fold reuse."""
import ast
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd

from mallorn.checkpointing import (AlreadyRunning, atomic_json, atomic_model,
                                   exclusive_lock, freeze_json)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from cycle3 import verify_fold
from refit_frozen import atomic_csv


class CheckpointTests(unittest.TestCase):
    def test_metrics_match_frozen_implementation_without_native_import(self):
        src = Path(__file__).resolve().parents[1]/'src/mallorn'
        names = {'check_probability', 'best_threshold', 'score'}
        def functions(path):
            return {node.name: ast.dump(node, include_attributes=False)
                    for node in ast.parse(path.read_text()).body
                    if isinstance(node, ast.FunctionDef) and node.name in names}
        self.assertEqual(functions(src/'modeling.py'), functions(src/'evaluation.py'))

    def test_failed_replace_preserves_previous_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'state.json'
            atomic_json(path, {'completed': 4})
            with patch('mallorn.checkpointing.os.replace', side_effect=OSError('interrupted')):
                with self.assertRaises(OSError):
                    atomic_json(path, {'completed': 5})
            self.assertEqual(json.loads(path.read_text()), {'completed': 4})
            self.assertEqual(list(Path(directory).glob('*.pending')), [])

    def test_freeze_rejects_changed_experiment(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'manifest.json'
            freeze_json(path, {'seed': 42})
            freeze_json(path, {'seed': 42})
            with self.assertRaises(ValueError):
                freeze_json(path, {'seed': 43})
            self.assertEqual(json.loads(path.read_text()), {'seed': 42})

    def test_lock_blocks_second_process_and_releases(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'lock'
            program = (
                'from mallorn.checkpointing import exclusive_lock, AlreadyRunning\n'
                'import sys\n'
                'try:\n'
                ' with exclusive_lock(sys.argv[1]): pass\n'
                'except AlreadyRunning: sys.exit(7)\n')
            with exclusive_lock(path):
                p = subprocess.run([sys.executable, '-c', program, str(path)], capture_output=True)
                self.assertEqual(p.returncode, 7, p.stderr)
            p = subprocess.run([sys.executable, '-c', program, str(path)], capture_output=True)
            self.assertEqual(p.returncode, 0, p.stderr)

    def test_atomic_model_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'fold.joblib'
            atomic_model(path, {'probabilities': np.array([.1, .9]), 'fold': 2})
            result = joblib.load(path)
            np.testing.assert_array_equal(result['probabilities'], [.1, .9])
            self.assertEqual(result['fold'], 2)

    def test_interrupted_csv_write_preserves_previous_output(self):
        class InterruptedFrame:
            def to_csv(self, handle, index):
                handle.write('incomplete')
                raise OSError('interrupted')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'submission.csv'
            frame = pd.DataFrame({'object_id': ['a'], 'prediction': [1]})
            atomic_csv(path, frame)
            before = path.read_bytes()
            with self.assertRaises(OSError):
                atomic_csv(path, InterruptedFrame())
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(list(Path(directory).glob('*.pending')), [])

    def test_killed_worker_releases_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'lock'
            ready = Path(directory)/'ready'
            program = (
                'from mallorn.checkpointing import exclusive_lock\n'
                'from pathlib import Path\n'
                'import sys\n'
                'with exclusive_lock(sys.argv[1]):\n'
                ' Path(sys.argv[2]).write_text("ready")\n'
                ' sys.stdin.read()\n')
            proc = subprocess.Popen([sys.executable, '-c', program, str(path), str(ready)],
                                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            try:
                import time
                deadline = time.monotonic()+10
                while not ready.exists() and proc.poll() is None and time.monotonic() < deadline:
                    time.sleep(.01)
                self.assertTrue(ready.exists())
                with self.assertRaises(AlreadyRunning):
                    with exclusive_lock(path):
                        pass
                proc.kill()
                proc.communicate(timeout=10)
                with exclusive_lock(path):
                    pass
            finally:
                if proc.poll() is None:
                    proc.kill()
                proc.communicate(timeout=10)

    def test_invalid_fold_is_never_reused(self):
        ids = pd.Index(['a', 'b', 'c'])
        a, b = np.array([0, 1]), np.array([2])
        record = {'manifest_sha256': 'frozen', 'fitting_ids': ['a', 'b'],
                  'validation_ids': ['c'], 'probabilities': np.array([.2])}
        np.testing.assert_array_equal(verify_fold(record, 'frozen', ids, a, b), [.2])
        for bad in [dict(record, manifest_sha256='changed'),
                    dict(record, validation_ids=['b']),
                    dict(record, probabilities=[np.nan]),
                    dict(record, probabilities=[1.1]),
                    dict(record, probabilities=[.1, .2])]:
            with self.assertRaises(ValueError):
                verify_fold(bad, 'frozen', ids, a, b)


if __name__ == '__main__':
    unittest.main()
