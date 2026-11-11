"""Hash immutable dependency bytes once within a single preparation audit.

This is an in-process read cache, not a persisted verification or release cache.
Every reuse checks file identity; any observed change aborts the audit instead of
silently mixing two generations of a dependency.
"""
from __future__ import annotations

import hashlib
from pathlib import Path


class ImmutableFileAudit:
    def __init__(self):
        self._records = {}
        self.files_hashed = 0
        self.reused_hashes = 0
        self.bytes_hashed = 0

    @staticmethod
    def _identity(path):
        s = path.stat()
        return (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)

    def sha256(self, value):
        path = Path(value)
        key = str(path.absolute())
        before = self._identity(path)
        if key in self._records:
            identity, digest = self._records[key]
            if before != identity:
                raise RuntimeError(f"Dependency changed during immutable audit: {path}")
            self.reused_hashes += 1
            return digest
        result = hashlib.sha256()
        count = 0
        with path.open('rb') as handle:
            while block := handle.read(8 * 1024 * 1024):
                result.update(block)
                count += len(block)
        if self._identity(path) != before or count != before[2]:
            raise RuntimeError(f"Dependency changed while hashing: {path}")
        digest = result.hexdigest()
        self._records[key] = (before, digest)
        self.files_hashed += 1
        self.bytes_hashed += count
        return digest

    def summary(self):
        return {'files_hashed': self.files_hashed, 'reused_hashes': self.reused_hashes,
                'bytes_hashed': self.bytes_hashed,
                'scope': 'one process; every cached reuse checks inode/size/mtime/ctime identity'}
