"""Reserve only a disjoint index interval before the shared quota transaction.

This does not reserve dataset quota, authorize dispatch, or publish instances.
If quota changes while waiting, unused indices stay reserved permanently.
"""
from dataclasses import dataclass
from pathlib import Path
import os
import socket
from scripts.campaign_20261110.contracts import require
from scripts.campaign_20261110.controller_families import read, reserve_indices


@dataclass(frozen=True)
class PreparedIndices:
    root: str
    token: str
    start: int
    count: int


def prepare(root, token, count):
    root = Path(root).resolve()
    owner = root / 'controllers/held_planning.lockdir/owner.json'
    if owner.exists():
        lock = read(owner, {})
        require(not (lock.get('hostname') == socket.gethostname() and lock.get('pid') == os.getpid()),
                'index preparation must not occupy this process quota transaction')
    previous = read(root / 'manifests/family_index_reservations.json', {'reservations': {}})['reservations'].get(token)
    reserved_count = previous['n'] if previous is not None else count
    require(isinstance(count, int) and count > 0 and isinstance(reserved_count, int) and reserved_count > 0,
            'invalid prepared interval size')
    start = reserve_indices(root, token, reserved_count)
    return PreparedIndices(str(root), token, start, reserved_count)


def consume(prepared, root, token, count):
    require(isinstance(prepared,PreparedIndices) and prepared.root==str(Path(root).resolve())
            and prepared.token==token and isinstance(count,int) and 0<count<=prepared.count,
            'prepared index reservation identity/count differs')
    path=Path(root)/'manifests/family_index_reservations.json'
    entry=read(path)['reservations'].get(token)
    require(entry is not None and entry['start']==prepared.start and entry['n']==prepared.count,
            'prepared disjoint interval changed')
    return prepared.start,{'token':token,'start':prepared.start,'reserved_count':prepared.count,
                           'used_count':count,'unused_count':prepared.count-count,
                           'quota_reserved_by_index_preparation':False}
