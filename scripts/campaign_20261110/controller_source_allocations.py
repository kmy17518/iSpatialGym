"""Lazy source discovery with exact saved-ledger revalidation.

A cache is an index only. Every returned saved row is reread from its owned
allocation ledger; immutable source/demo/replay checks remain with the caller.
"""
import csv
from pathlib import Path

from scripts.campaign_20261110.contracts import require


class SourceAllocations:
    def __init__(self, root, state, scan):
        self.root = Path(root).resolve()
        self.saved = state.setdefault('source_allocations', {})
        self.scan = scan
        self._rows = None

    def all(self):
        if self._rows is None:
            self._rows = self.scan(self.root)
        return self._rows

    def find(self, job, *, ledger=None):
        job = str(Path(job).resolve())
        require(Path(job).is_relative_to(self.root/'out'), 'source output outside campaign')
        prior = self.saved.get(job)
        selected = ledger or (prior or {}).get('ledger')
        if selected:
            path = Path(selected).resolve()
            require(path.is_relative_to(self.root/'manifests'), 'source ledger outside campaign')
            with path.open() as stream:
                found = [{**row, 'ledger': str(selected)} for row in csv.DictReader(stream, delimiter='\t')
                         if Path(row['job_dir']).resolve() == Path(job)]
        else:
            found = [row for row in self.all() if Path(row['job_dir']).resolve() == Path(job)]
        require(len(found) == 1, 'source allocation missing or ambiguous')
        row = found[0]
        require(prior is None or row == prior, 'saved source allocation changed')
        self.saved[job] = row
        return row
