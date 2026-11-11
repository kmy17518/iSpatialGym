import hashlib
import os
from pathlib import Path
import tempfile
import unittest

from scripts.campaign_20261110.immutable_file_audit import ImmutableFileAudit


class ImmutableFileAuditTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / 'library.bin'
        self.path.write_bytes(b'physical dependency bytes')
        self.cache = ImmutableFileAudit()

    def test_repeated_heights_reuse_exact_source_hash(self):
        expected = hashlib.sha256(self.path.read_bytes()).hexdigest()
        for _ in range(20):
            self.assertEqual(self.cache.sha256(self.path), expected)
        self.assertEqual(self.cache.files_hashed, 1)
        self.assertEqual(self.cache.reused_hashes, 19)
        self.assertEqual(self.cache.bytes_hashed, self.path.stat().st_size)

    def test_changed_dependency_is_not_silently_rehashed(self):
        self.cache.sha256(self.path)
        self.path.write_bytes(b'a changed dependency')
        with self.assertRaisesRegex(RuntimeError, 'changed during'):
            self.cache.sha256(self.path)

    def test_atomic_replacement_with_same_size_and_mtime_is_detected(self):
        before = self.path.stat()
        self.cache.sha256(self.path)
        replacement = self.path.with_suffix('.replacement')
        replacement.write_bytes(b'x' * before.st_size)
        os.utime(replacement, ns=(before.st_atime_ns, before.st_mtime_ns))
        replacement.replace(self.path)
        with self.assertRaisesRegex(RuntimeError, 'changed during'):
            self.cache.sha256(self.path)

    def test_missing_file_never_yields_a_cached_hash(self):
        self.cache.sha256(self.path)
        self.path.unlink()
        with self.assertRaises(FileNotFoundError):
            self.cache.sha256(self.path)

    def test_cache_is_not_shared_between_audits(self):
        self.cache.sha256(self.path)
        self.path.write_bytes(b'new audit input')
        fresh = ImmutableFileAudit()
        self.assertEqual(fresh.sha256(self.path), hashlib.sha256(b'new audit input').hexdigest())
        self.assertEqual(fresh.files_hashed, 1)


if __name__ == '__main__':
    unittest.main()
