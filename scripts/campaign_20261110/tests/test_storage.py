from pathlib import Path
import sqlite3

import pytest

from scripts.campaign_20261110 import storage


def write_value(path, value):
    db = sqlite3.connect(path)
    try:
        with db:
            db.execute("CREATE TABLE IF NOT EXISTS values_ (value INTEGER)")
            db.execute("INSERT INTO values_ VALUES (?)", (value,))
    finally:
        db.close()


def test_lock_contention_fails_closed_and_exception_releases(tmp_path):
    lock = tmp_path / "allocation.lockdir"
    with pytest.raises(ValueError):
        with storage.atomic_directory_lock(lock):
            assert (lock / "owner.json").is_file()
            with pytest.raises(FileExistsError):
                with storage.atomic_directory_lock(lock):
                    pytest.fail("Another claimant entered a held lock")
            raise ValueError("interrupted")
    assert not lock.exists()
    lock.mkdir()  # An abandoned lock must not be automatically stolen.
    with pytest.raises(FileExistsError):
        with storage.atomic_directory_lock(lock):
            pytest.fail("Stale lock was stolen")


def test_sqlite_never_opens_shared_path_and_failed_write_preserves_bytes(tmp_path, monkeypatch):
    state = tmp_path / "state.sqlite3"
    original = sqlite3.connect
    def connect(path, *args, **kwargs):
        assert str(tmp_path) not in str(path), "SQLite attempted to lock shared storage"
        return original(path, *args, **kwargs)
    monkeypatch.setattr(storage.sqlite3, "connect", connect)
    with storage.local_sqlite_transaction(state) as local:
        assert local.parent.parent == Path("/tmp")
        write_value(local, 1)
        assert not state.exists()
    before = state.read_bytes()
    with pytest.raises(ValueError):
        with storage.local_sqlite_transaction(state) as local:
            write_value(local, 2)
            raise ValueError("failed audit")
    assert state.read_bytes() == before
    with storage.local_sqlite_transaction(state) as local:
        write_value(local, 3)
    with original(state) as db:
        assert db.execute("SELECT value FROM values_").fetchall() == [(1,), (3,)]


def test_publication_failure_preserves_prior_state_and_cleans_temporary(tmp_path, monkeypatch):
    state = tmp_path / "state.sqlite3"
    with storage.local_sqlite_transaction(state) as local:
        write_value(local, 1)
    before = state.read_bytes()
    def fail_replace(*args):
        raise OSError("lost storage connection")
    monkeypatch.setattr(storage.os, "replace", fail_replace)
    with pytest.raises(OSError, match="lost storage"):
        with storage.local_sqlite_transaction(state) as local:
            write_value(local, 2)
    assert state.read_bytes() == before
    assert list(tmp_path.iterdir()) == [state]


def test_uncheckpointed_sidecars_and_corruption_are_not_published(tmp_path):
    state = tmp_path / "state.sqlite3"
    with pytest.raises(RuntimeError, match="sidecar"):
        with storage.local_sqlite_transaction(state) as local:
            write_value(local, 1)
            Path(str(local) + "-wal").write_bytes(b"pending")
    assert not state.exists()
    state.write_bytes(b"corrupt")
    with pytest.raises(sqlite3.DatabaseError):
        with storage.local_sqlite_transaction(state):
            pytest.fail("Corrupt state accepted")
