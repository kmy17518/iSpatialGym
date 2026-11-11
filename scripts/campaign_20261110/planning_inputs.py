"""Audit immutable source bytes before entering the shared quota lock.

This is a per-attempt handoff, not a durable hash cache. The ordinary anchor
validator still reads every byte. The reservation transaction checks that none
of those files changed while it waited for the lock.
"""
from dataclasses import dataclass
from pathlib import Path
import json
import csv

from scripts.campaign_20261110.contracts import digest, require


def _signature(path):
    path = Path(path)
    target = path.resolve(strict=True)
    stat = target.stat()
    require(target.is_file(), 'planning input is not a file')
    return (str(target), stat.st_dev, stat.st_ino, stat.st_size,
            stat.st_mtime_ns, stat.st_ctime_ns)


def _allocation(ledger, job):
    with Path(ledger).open() as stream:
        rows = [row for row in csv.DictReader(stream, delimiter="\t")
                if row.get("job_dir") and Path(row["job_dir"]).resolve() == Path(job).resolve()]
    require(len(rows) == 1, 'planning source allocation is absent or ambiguous')
    return rows[0]


def _identity(job_dir, root, ledger, started_at):
    return (str(Path(job_dir).resolve()), str(Path(root).resolve()),
            str(Path(ledger).resolve()), float(started_at))


@dataclass(frozen=True)
class PreparedAnchor:
    identity: tuple
    signatures: dict
    anchor: dict
    anchor_sha256: str


def prepare_anchor(job_dir, *, campaign_root, ledger, campaign_started_at,
                   extra_paths=()):
    from scripts.campaign_20261110.goal_groups import successful_anchor
    job = Path(job_dir).resolve()
    selectors = [job / 'results.jsonl', job / 'job.env', *map(Path, extra_paths)]
    allocation = _allocation(ledger, job)
    initial = {str(p): _signature(p) for p in selectors}
    successes = []
    for line in (job / 'results.jsonl').read_text().splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if row.get('success') is True:
            successes.append(row)
    require(successes, 'planning anchor has no successful demo')
    iid = successes[0]['instance']
    matches = list((job / 'inst').rglob(iid + '-tro_state.json'))
    require(len(matches) == 1, 'planning anchor TRO is absent or ambiguous')
    tro = matches[0]
    paths = selectors + [tro, tro.with_name(iid + '.oracle.json'),
        tro.with_name(iid + '.summary.json'), tro.parent.parent / 'template_meta.json',
        job / 'demo' / iid / 'meta.json', job / 'demo' / iid / 'demo.hdf5']
    before = {str(p): _signature(p) for p in paths}
    require(all(before[p] == sig for p, sig in initial.items()),
            'planning source changed during input selection')
    anchor = successful_anchor(job, campaign_root=campaign_root, ledger=ledger,
                               campaign_started_at=campaign_started_at)
    require(anchor['instance_id'] == iid, 'planning anchor selection changed')
    require(anchor['allocation'] == allocation == _allocation(ledger, job),
            'planning source allocation changed during audit')
    require(all(_signature(p) == sig for p, sig in before.items()),
            'planning source changed during byte audit')
    return PreparedAnchor(_identity(job, campaign_root, ledger, campaign_started_at),
                          before, anchor, digest(anchor))


def checked_anchor(prepared, job_dir, *, campaign_root, ledger, campaign_started_at):
    require(isinstance(prepared, PreparedAnchor), 'unverified planning source')
    require(prepared.identity == _identity(job_dir, campaign_root, ledger, campaign_started_at),
            'planning source belongs to another reservation')
    require(digest(prepared.anchor) == prepared.anchor_sha256, 'prepared anchor was modified')
    require(_allocation(ledger, job_dir) == prepared.anchor['allocation'],
            'planning source allocation changed while waiting')
    require(all(_signature(p) == sig for p, sig in prepared.signatures.items()),
            'planning source changed while waiting for reservation')
    return prepared.anchor
