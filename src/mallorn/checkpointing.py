"""Atomic checkpoints and an OS-owned lock for restartable local experiments."""
from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import tempfile

import joblib


class AlreadyRunning(RuntimeError):
    pass


@contextmanager
def exclusive_lock(path):
    """A crashed process releases the lock; a stale filename never blocks work."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+b') as handle:
        handle.seek(0)
        if os.name == 'nt':
            import msvcrt
            if path.stat().st_size == 0:
                handle.write(b'0')
                handle.flush()
            handle.seek(0)
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise AlreadyRunning(str(path)) from exc
        else:
            import fcntl
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise AlreadyRunning(str(path)) from exc
        try:
            yield
        finally:
            if os.name == 'nt':
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _atomic(path, writer):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + '.', suffix='.pending', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as handle:
            writer(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        if os.name != 'nt':
            fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def atomic_json(path, value):
    payload = json.dumps(value, indent=2, allow_nan=False).encode() + b'\n'
    _atomic(path, lambda handle: handle.write(payload))


def atomic_model(path, value):
    _atomic(path, lambda handle: joblib.dump(value, handle, compress=3))


def freeze_json(path, value):
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f'Frozen configuration differs: {path}. Create a new experiment.')
    else:
        atomic_json(path, value)
