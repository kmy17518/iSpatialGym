import json
from pathlib import Path
import tempfile
import unittest

from scripts.campaign_20261110.controller_families import reserve_indices, _allocation_signature
from scripts.campaign_20261110.storage import atomic_directory_lock


class IndexReservationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / 'manifests').mkdir()

    def test_large_reservation_advances_past_entire_range(self):
        first = reserve_indices(self.root, 'large', 301)
        second = reserve_indices(self.root, 'next', 3)
        self.assertGreaterEqual(second, first + 301)
        self.assertEqual(first, reserve_indices(self.root, 'large', 301))

    def test_outstanding_reservation_blocks_stale_next_pointer(self):
        path = self.root / 'manifests/family_index_reservations.json'
        path.write_text(json.dumps({'next': 50000100, 'reservations': {'unsubmitted': {'start': 50000000, 'n': 301}}}))
        self.assertGreaterEqual(reserve_indices(self.root, 'new', 6), 50000301)

    def test_existing_ledger_and_lock_contention(self):
        (self.root / 'manifests/test_allocations.tsv').write_text('key\tstart\tn\nold\t50000000\t201\n')
        self.assertGreaterEqual(reserve_indices(self.root, 'new', 1), 50000201)
        with atomic_directory_lock(self.root / 'manifests/family_index_reservations.lockdir'):
            with self.assertRaises(FileExistsError): reserve_indices(self.root, 'concurrent', 1)

    def test_signature_matches_full_resolution_with_symlinks(self):
        manifests = self.root / 'manifests'
        original = manifests / 'a_allocations.tsv'
        original.write_text('key\tstart\tn\nold\t50000000\t201\n')
        (manifests / 'z_allocations.tsv').symlink_to(original.name)
        (manifests / '.hidden_allocations.tsv').write_text('')
        (manifests / 'ignored.json').write_text('{}')
        alias = self.root / 'alias'; alias.mkdir()
        (alias / 'manifests').symlink_to(manifests, target_is_directory=True)
        for root in (self.root, alias):
            expected = []
            for path in sorted((root / 'manifests').glob('*allocations.tsv')):
                target = path.resolve(strict=True); stat = target.stat()
                expected.append((str(path), str(target), stat.st_dev, stat.st_ino,
                                 stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns))
            self.assertEqual(_allocation_signature(root), tuple(expected))

    def test_signature_detects_append_new_file_and_symlink_retarget(self):
        manifests = self.root / 'manifests'
        first = manifests / 'first_allocations.tsv'; first.write_text('one')
        before = _allocation_signature(self.root)
        first.write_text('one\ntwo')
        after = _allocation_signature(self.root)
        self.assertNotEqual(before, after)
        second = manifests / 'second_allocations.tsv'; second.write_text('two')
        self.assertNotEqual(after, _allocation_signature(self.root))
        link = manifests / 'link_allocations.tsv'; link.symlink_to(first.name)
        before = _allocation_signature(self.root)
        link.unlink(); link.symlink_to(second.name)
        self.assertNotEqual(before, _allocation_signature(self.root))

    def test_broken_ledger_link_fails_closed(self):
        (self.root / 'manifests/broken_allocations.tsv').symlink_to('missing')
        with self.assertRaises(FileNotFoundError):
            _allocation_signature(self.root)


if __name__ == '__main__': unittest.main()
