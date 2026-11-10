import json
from pathlib import Path
import tempfile
import unittest

from scripts.campaign_20261110.controller_families import reserve_indices
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


if __name__ == '__main__': unittest.main()
