"""Durable campaign state on storage that does not implement POSIX locks."""
from contextlib import closing, contextmanager
import errno
import json
import os
from pathlib import Path
import shutil
import socket
import sqlite3
import tempfile
import time


def _sync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        try:
            os.fsync(fd)
        except OSError as exc:
            if exc.errno not in (errno.EINVAL, errno.ENOTSUP):
                raise
    finally:
        os.close(fd)


@contextmanager
def atomic_directory_lock(path):
    """Fail closed on contention or abandoned locks; never guess a stale TTL."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.mkdir()  # Atomic across clients; EEXIST must propagate.
    try:
        owner = {"hostname": socket.gethostname(), "pid": os.getpid(), "created_at": time.time()}
        with (path / "owner.json").open("x") as stream:
            json.dump(owner, stream)
            stream.flush()
            os.fsync(stream.fileno())
        yield
    finally:
        (path / "owner.json").unlink(missing_ok=True)
        path.rmdir()


def durable_atomic_copy(source, destination):
    """Publish complete bytes by replacing a same-directory temporary file."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix="." + destination.name + ".", dir=destination.parent)
    try:
        with os.fdopen(fd, "wb") as target, Path(source).open("rb") as incoming:
            shutil.copyfileobj(incoming, target)
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, destination)
        _sync_directory(destination.parent)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _check_database(path):
    # The caller must close every writer before publishing; no sidecar may be
    # dropped while moving the database back to the shared filesystem.
    for suffix in ("-journal", "-wal", "-shm"):
        if Path(str(path) + suffix).exists():
            raise RuntimeError("SQLite sidecar remains before publication: " + suffix)
    with closing(sqlite3.connect(f"file:{path}?mode=ro", uri=True)) as db:
        if db.execute("PRAGMA quick_check").fetchall() != [("ok",)]:
            raise RuntimeError("Campaign quota database failed integrity check")


@contextmanager
def local_sqlite_transaction(state_path):
    """Stage SQLite locally and publish on success, under a caller-owned lock.

    The shared filesystem is never opened by SQLite. A failed transaction keeps
    the prior shared database byte-for-byte intact. SQLite connections inside
    this context must be closed before leaving it.
    """
    state_path = Path(state_path)
    with tempfile.TemporaryDirectory(prefix="isg-campaign-state-", dir="/tmp") as temporary:
        local = Path(temporary) / "quotas.sqlite3"
        if state_path.exists():
            shutil.copyfile(state_path, local)
            _check_database(local)
        yield local
        _check_database(local)
        durable_atomic_copy(local, state_path)
