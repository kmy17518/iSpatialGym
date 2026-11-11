import json
from contextlib import contextmanager
import pytest
from scripts.campaign_20261110 import controller_families as f
from scripts.campaign_20261110.contracts import ContractError


def ledger(root, name, start, count):
    p = root/'manifests'/name
    p.parent.mkdir(exist_ok=True)
    p.write_text('start\tn\n'+str(start)+'\t'+str(count)+'\n')
    return p


def test_ledger_bytes_are_read_outside_lock_and_retry_is_idempotent(tmp_path, monkeypatch):
    ledger(tmp_path, 'existing_allocations.tsv', 50000000, 120)
    real = f.allocations; calls = []
    def read(root):
        assert not (root/'manifests/family_index_reservations.lockdir').exists()
        calls.append(True); return real(root)
    monkeypatch.setattr(f, 'allocations', read)
    assert f.reserve_indices(tmp_path, 'first', 3) == 50000200
    assert f.reserve_indices(tmp_path, 'first', 3) == 50000200
    assert calls == [True]
    with pytest.raises(ContractError, match='reservation changed'):
        f.reserve_indices(tmp_path, 'first', 4)


@pytest.mark.parametrize('race', ['new_ledger', 'changed_ledger', 'new_reservation'])
def test_concurrent_obligation_is_rechecked_before_commit(tmp_path, monkeypatch, race):
    path = ledger(tmp_path, 'existing_allocations.tsv', 1, 2)
    original = f.atomic_directory_lock; attempts=[]
    @contextmanager
    def enter(lock):
        if not attempts:
            if race == 'new_ledger':
                ledger(tmp_path, 'new_allocations.tsv', 50000000, 200)
            elif race == 'changed_ledger':
                path.write_text('start\tn\n50000000\t200\n')
            else:
                (tmp_path/'manifests/family_index_reservations.json').write_text(json.dumps({
                    'next':50000000, 'reservations':{'other':{'start':50000000,'n':200}}}))
        attempts.append(True)
        with original(lock):yield
    monkeypatch.setattr(f, 'atomic_directory_lock', enter)
    assert f.reserve_indices(tmp_path, 'new', 3) == 50000200
    state=f.read(tmp_path/'manifests/family_index_reservations.json')
    assert state['reservations']['new']['n']==3
    assert len(attempts)==(1 if race=='new_reservation' else 2)


def test_busy_owner_is_not_removed_and_no_reservation_written(tmp_path):
    ledger(tmp_path, 'existing_allocations.tsv', 1, 2)
    lock=tmp_path/'manifests/family_index_reservations.lockdir'
    with f.atomic_directory_lock(lock):
        before=(lock/'owner.json').read_bytes()
        with pytest.raises(FileExistsError):f.reserve_indices(tmp_path,'new',3)
        assert (lock/'owner.json').read_bytes()==before
    assert not (tmp_path/'manifests/family_index_reservations.json').exists()
