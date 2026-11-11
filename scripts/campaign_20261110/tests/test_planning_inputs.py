"""Source hashing must not occupy the quota lock or survive changed inputs."""
import json
from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest

from scripts.campaign_20261110 import planning_inputs as p, goal_groups, contracts as c
from scripts.campaign_20261110 import controller_held as held, controller_admission as admission


def fixture(tmp_path, monkeypatch):
    job = tmp_path / 'out/job'
    inst = job / 'inst/template/inst'
    demo = job / 'demo/i'
    inst.mkdir(parents=True); demo.mkdir(parents=True)
    paths = [job/'job.env', tmp_path/'ledger', inst/'i-tro_state.json',
             inst/'i.oracle.json', inst/'i.summary.json', inst.parent/'template_meta.json',
             demo/'meta.json', demo/'demo.hdf5', tmp_path/'config']
    for path in paths:
        path.write_text('original')
    (tmp_path/'ledger').write_text('job_dir\tjobid\n'+str(job)+'\t1\n')
    (job/'results.jsonl').write_text(json.dumps({'success': True, 'instance': 'i'})+'\n')
    calls = []
    def audit(*a, **kw):
        assert not (tmp_path/'controllers/held_planning.lockdir').exists()
        calls.append('read full demo')
        return {'instance_id': 'i', 'allocation': {'job_dir':str(job), 'jobid':'1'}, 'actual_bytes': (demo/'demo.hdf5').read_text()}
    monkeypatch.setattr(goal_groups, 'successful_anchor', audit)
    args = dict(campaign_root=tmp_path, ledger=tmp_path/'ledger', campaign_started_at=1)
    return job, paths, args, calls


def test_byte_audit_outside_lock_only_identity_checks_inside(tmp_path, monkeypatch):
    job, paths, args, calls = fixture(tmp_path, monkeypatch)
    prepared = p.prepare_anchor(job, **args, extra_paths=[tmp_path/'config'])
    (tmp_path/'controllers').mkdir()
    with held.planning_lock(tmp_path/'controllers/held_planning.lockdir'):
        assert p.checked_anchor(prepared, job, **args)['actual_bytes'] == 'original'
    assert calls == ['read full demo']


@pytest.mark.parametrize('which', ['demo', 'config', 'metadata', 'results'])
def test_changed_source_rejected_before_reservation(tmp_path, monkeypatch, which):
    job, paths, args, _ = fixture(tmp_path, monkeypatch)
    prepared = p.prepare_anchor(job, **args, extra_paths=[tmp_path/'config'])
    path = {'demo': job/'demo/i/demo.hdf5', 'config': tmp_path/'config',
            'metadata': job/'inst/template/template_meta.json', 'results': job/'results.jsonl'}[which]
    path.write_text('changed source')
    with pytest.raises(c.ContractError, match='changed while waiting'):
        p.checked_anchor(prepared, job, **args)


def test_same_length_write_with_restored_mtime_still_rejected(tmp_path, monkeypatch):
    import os
    job, _, args, _ = fixture(tmp_path, monkeypatch)
    prepared = p.prepare_anchor(job, **args)
    path = job/'demo/i/demo.hdf5'; before = path.stat()
    path.write_text('modified')
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    with pytest.raises(c.ContractError, match='changed while waiting'):
        p.checked_anchor(prepared, job, **args)


def test_mid_audit_change_and_wrong_reservation_are_rejected(tmp_path, monkeypatch):
    job, _, args, _ = fixture(tmp_path, monkeypatch)
    prepared = p.prepare_anchor(job, **args)
    with pytest.raises(c.ContractError, match='another reservation'):
        p.checked_anchor(prepared, job, **{**args, 'campaign_started_at': 2})
    def mutate(*a, **kw):
        (job/'demo/i/demo.hdf5').write_text('modified')
        return {'instance_id': 'i', 'allocation': {'job_dir':str(job), 'jobid':'1'}}
    monkeypatch.setattr(goal_groups, 'successful_anchor', mutate)
    with pytest.raises(c.ContractError, match='during byte audit'):
        p.prepare_anchor(job, **args)


@pytest.mark.parametrize("mode", ["available", "taken_during_audit", "already_complete"])
def test_refill_prepares_before_lock_but_counts_and_save_remain_locked(tmp_path, monkeypatch, mode):
    from scripts.campaign_20261110.tests.test_goal_groups import GoalGroupControllerTests
    _, _, _, original, _ = GoalGroupControllerTests().fixture(tmp_path)
    template = replace(original, split='eval_model')
    config = tmp_path/'config.json'; gate = tmp_path/'gate.json'
    c.write_json_atomic(config, {'template':asdict(template), 'validation_key':{}})
    c.write_json_atomic(gate, {'success':True})
    spec = {'class_id':template.class_id, 'seed_config':str(config), 'seed_config_sha256':c.file_digest(config),
            'source_gate':str(gate), 'source_gate_sha256':c.file_digest(gate), 'max_candidates':3}
    lock = tmp_path/'controllers/held_planning.lockdir'; lock.parent.mkdir(exist_ok=True)
    order = []
    def prepare_source(*a):
        assert not lock.exists(); order.append('audit'); return 'checked'
    def counts(*a):
        order.append('quota' if lock.exists() else 'preview'); return {}
    def author(*a, **kw):
        assert lock.exists() and kw == {'prepared_source':'checked'}
        order.append('reserve_and_save')
        raise RuntimeError('stop after reservation boundary')
    decisions = []
    def next_work(*a, **kw):
        decisions.append(True)
        return None if mode == 'already_complete' or (mode == 'taken_during_audit' and len(decisions)==2) else {'groups':1}
    adapter = SimpleNamespace(supports=lambda task:True, prepare_source=prepare_source,
        read_counts=counts, pending=lambda root:[], reserve_pending_counts=lambda *a:{},
        next_work=next_work, prepare_queue=author)
    invoke = lambda: held.advance(spec, {}, {"root":str(tmp_path), "stop_at":1e12}, tmp_path, [], {}, 1, lambda:None, adapter=adapter)
    if mode == 'available':
        with pytest.raises(RuntimeError, match='reservation boundary'):
            invoke()
        assert order == ['preview', 'audit', 'quota', 'reserve_and_save']
    else:
        assert invoke() == 0
        assert order == (['preview'] if mode == 'already_complete' else ['preview', 'audit', 'quota'])
    assert not lock.exists()


def test_failed_dry_run_stays_reserved_and_cannot_be_marked_passed(tmp_path, monkeypatch):
    record = {'status':'ready', 'command':['submit'], 'directory':str(tmp_path), 'dry_run_passed':False}
    saves = []
    monkeypatch.setattr(admission, 'run_logged', lambda *a: (_ for _ in ()).throw(RuntimeError('parser failed')))
    with pytest.raises(RuntimeError, match='parser failed'):
        admission.ensure_dry_run(record, lambda:saves.append(True))
    assert not record['dry_run_passed'] and not saves and record['status']=='ready'
    monkeypatch.setattr(admission, 'run_logged', lambda *a: None)
    admission.ensure_dry_run(record, lambda:saves.append(True))
    assert record['dry_run_passed'] and saves == [True]


def test_unrelated_ledger_append_allowed_but_source_row_change_rejected(tmp_path, monkeypatch):
    job, _, args, _ = fixture(tmp_path, monkeypatch)
    prepared = p.prepare_anchor(job, **args)
    ledger = tmp_path/'ledger'
    with ledger.open('a') as stream:
        stream.write(str(tmp_path/'out/unrelated')+'\t2\n')
    assert p.checked_anchor(prepared, job, **args)['instance_id'] == 'i'
    ledger.write_text(ledger.read_text().replace(str(job)+'\t1', str(job)+'\t3'))
    with pytest.raises(c.ContractError, match='allocation changed'):
        p.checked_anchor(prepared, job, **args)
