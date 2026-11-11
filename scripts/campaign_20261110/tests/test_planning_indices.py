from dataclasses import replace
import pytest
from scripts.campaign_20261110 import planning_indices as p
from scripts.campaign_20261110.contracts import write_json_atomic
from scripts.campaign_20261110.storage import atomic_directory_lock


def test_index_scan_outside_quota_lock_and_shrinking_quota_keeps_tail(tmp_path,monkeypatch):
    calls=[]
    def reserve(root,token,count):
        assert not(root/'controllers/held_planning.lockdir').exists()
        calls.append((token,count))
        write_json_atomic(root/'manifests/family_index_reservations.json',{'reservations':{token:{'start':100,'n':count}}})
        return 100
    monkeypatch.setattr(p,'reserve_indices',reserve)
    prepared=p.prepare(tmp_path,'token',6)
    with atomic_directory_lock(tmp_path/'controllers/held_planning.lockdir'):
        start,proof=p.consume(prepared,tmp_path,'token',3)
    assert start==100 and proof['unused_count']==3 and calls==[('token',6)]
    assert proof['quota_reserved_by_index_preparation'] is False


def test_existing_interval_does_not_expand_on_later_preview(tmp_path,monkeypatch):
    write_json_atomic(tmp_path/'manifests/family_index_reservations.json',{'reservations':{'token':{'start':100,'n':3}}})
    def reserve(root,token,count):
        assert count==3
        return 100
    monkeypatch.setattr(p,'reserve_indices',reserve)
    prepared=p.prepare(tmp_path,'token',18)
    assert prepared.count==3
    with pytest.raises(ValueError,match='identity/count'):
        p.consume(prepared,tmp_path,'token',6)


def test_wrong_owner_or_changed_interval_is_rejected(tmp_path,monkeypatch):
    write_json_atomic(tmp_path/'manifests/family_index_reservations.json',{'reservations':{'token':{'start':100,'n':6}}})
    prepared=p.PreparedIndices(str(tmp_path.resolve()),'token',100,6)
    for bad in (replace(prepared,token='other'),replace(prepared,start=101),replace(prepared,count=9)):
        with pytest.raises(ValueError):p.consume(bad,tmp_path,'token',3)
    with atomic_directory_lock(tmp_path/'controllers/held_planning.lockdir'):
        with pytest.raises(ValueError,match='quota transaction'):
            p.prepare(tmp_path,'token',6)


def test_another_planner_does_not_serialize_index_preparation(tmp_path,monkeypatch):
    write_json_atomic(tmp_path/'controllers/held_planning.lockdir/owner.json',{'hostname':'another-host','pid':1})
    monkeypatch.setattr(p,'reserve_indices',lambda *args:100)
    assert p.prepare(tmp_path,'token',3).start==100

@pytest.mark.parametrize('final_groups,expected', [(1, 1), (4, 2), (None, None)])
def test_refill_prepares_indices_before_lock_and_rechecks_quota(tmp_path, monkeypatch, final_groups, expected):
    from dataclasses import asdict
    from types import SimpleNamespace
    from scripts.campaign_20261110 import controller_held as held
    from scripts.campaign_20261110 import contracts as c
    from scripts.campaign_20261110.tests.test_goal_groups import GoalGroupControllerTests
    _, _, _, original, _ = GoalGroupControllerTests().fixture(tmp_path)
    template = replace(original, split='eval_model')
    config, gate = tmp_path/'config.json', tmp_path/'gate.json'
    c.write_json_atomic(config, {'template':asdict(template), 'validation_key':{}})
    c.write_json_atomic(gate, {'success':True})
    spec = {'class_id':template.class_id, 'seed_config':str(config),
            'seed_config_sha256':c.file_digest(config), 'source_gate':str(gate),
            'source_gate_sha256':c.file_digest(gate), 'max_candidates':18}
    lock = tmp_path/'controllers/held_planning.lockdir'
    calls = []
    def indices(spec, state, decision, root):
        assert not lock.exists() and decision['groups'] == 2
        calls.append('indices')
        return p.PreparedIndices(str(tmp_path.resolve()), 'token', 100, 6)
    def next_work(*args, **kwargs):
        count = final_groups if lock.exists() else 2
        calls.append('locked_quota' if lock.exists() else 'preview')
        return {'groups':count} if count is not None else None
    def author(spec, state, decision, cfg, root, template, save, **kwargs):
        assert lock.exists() and decision['groups'] == expected
        assert kwargs['prepared_indices'].count == 6
        calls.append('author')
        raise RuntimeError('reservation committed')
    adapter = SimpleNamespace(supports=lambda task:True, prepare_indices=indices,
        read_counts=lambda *args:{}, pending=lambda root:[], reserve_pending_counts=lambda *args:{},
        next_work=next_work, prepare_queue=author)
    def invoke():
        return held.advance(spec, {}, {'root':str(tmp_path), 'stop_at':1e12},
                            tmp_path, [], {}, 1, lambda:None, adapter=adapter)
    if expected is None:
        assert invoke() == 0
        assert calls == ['preview','indices','locked_quota']
    else:
        with pytest.raises(RuntimeError, match='reservation committed'):
            invoke()
        assert calls == ['preview','indices','locked_quota','author']
    assert not lock.exists()
