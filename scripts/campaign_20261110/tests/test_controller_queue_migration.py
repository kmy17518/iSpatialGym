import json
from copy import deepcopy
from pathlib import Path
import pytest
from scripts.campaign_20261110 import controller_queue_migration as m
from scripts.campaign_20261110.contracts import file_digest, ContractError


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))
    return path


def source(tmp_path, monkeypatch):
    from ispatialgym.sampling import campaign_goal_queue as q
    root = tmp_path
    group = root/'groups'/'old_0000'
    key = {'code_sha256': 'native123', 'template_sha256': 'template123'}
    cfg = write(group/'campaign_config.json', {'campaign_id': 'campaign_20261110', 'validation_key': key})
    empty = write(group/'empty_eval_index.json', {'instances': []})
    plan = write(group/'goal_plan.json', {'group_id': 'old_0000', 'schema': 'isg-pair-goal-plan-v1',
        'goal_pose': {'position': [1., 2., 3.], 'orientation': [0., 0., 0., 1.]},
        'max_members': 3, 'strict_collision_gate': True})
    queue = write(root/'queues/old/queue.json', {'campaign_id': 'campaign_20261110',
        'config_path': str(cfg), 'config_sha256': file_digest(cfg),
        'entries': [{'path': str(plan), 'sha256': file_digest(plan)}], 'group_ids': ['old_0000'], 'seed': 123})
    cmd = ['python', 'submit_fast.py', 'submit', 'class', '3', '--base-index', '50000100',
           '--code', str(root/'code/native123'), '--extra-env', 'ISPATIALGYM_CAMPAIGN_CONFIG='+str(cfg),
           '--extra-env', 'ISPATIALGYM_CAMPAIGN_GOAL_QUEUE='+str(queue), '--extra-env', 'EVAL_INDEX='+str(empty)]
    row = {'directory': str(queue.parent), 'queue': str(queue), 'validation_key': key, 'command': cmd,
           'index': 50000100, 'count': 3, 'groups': {}, 'status': 'external_submission', 'dry_run_passed': True,
           'prepared': {'queue': str(queue), 'config': str(cfg), 'group_ids': ['old_0000']},
           'allocation': {'jobid': '123', 'job_dir': str(root/'out/old')}}
    intent=write(root/'dispatch/ready/old.json', {'backends':['minyeong_svl','vast_4090']})
    row['dispatch_intent']=str(intent)
    monkeypatch.setattr(q, 'read_queue', lambda path, config: json.loads(Path(path).read_text()))
    monkeypatch.setattr(m, 'reserve_indices', lambda root, token, count: 50000200)
    return row, cfg, plan, queue


def test_fresh_identities_preserve_every_physical_field_and_source_byte(tmp_path, monkeypatch):
    row, cfg, plan, queue = source(tmp_path, monkeypatch)
    before = {p: p.read_bytes() for p in (cfg, plan, queue)}
    fresh = m.prepare_replacement(tmp_path, row)
    assert fresh['dispatch_backends']==['minyeong_svl','vast_4090']
    assert fresh['migration_routing']['source_intent']['sha256']==file_digest(row['dispatch_intent'])
    assert fresh['index'] == 50000200 and fresh['count'] == 3 and 'allocation' not in fresh
    assert fresh['validation_key'] == row['validation_key']
    assert Path(fresh['prepared']['config']).read_bytes() == cfg.read_bytes()
    cloned = json.loads((Path(fresh['prepared']['config']).parent/'goal_plan.json').read_text())
    original = json.loads(plan.read_text())
    assert cloned.pop('group_id') == 'old_0000_pending_m01'
    original.pop('group_id'); assert cloned == original
    assert all(p.read_bytes() == data for p, data in before.items())
    env = m.environment(fresh['command'])
    assert env['ISPATIALGYM_CAMPAIGN_GOAL_QUEUE'] == fresh['queue']
    assert fresh['command'][fresh['command'].index('--code')+1] == row['command'][row['command'].index('--code')+1]


def test_changed_plan_and_native_journal_prevent_any_preparation(tmp_path, monkeypatch):
    row, cfg, plan, queue = source(tmp_path, monkeypatch)
    plan.write_text(plan.read_text()+' ')
    with pytest.raises(ContractError, match='source plan changed'):
        m.prepare_replacement(tmp_path, row)
    raw = json.loads(queue.read_text()); raw['entries'][0]['sha256'] = file_digest(plan); write(queue, raw)
    (queue.parent/'queue.events.jsonl').write_text('{"published": true}\n')
    with pytest.raises(ContractError, match='native execution events'):
        m.prepare_replacement(tmp_path, row)


def test_running_job_never_calls_scheduler_or_cancellation(tmp_path, monkeypatch):
    row, *_ = source(tmp_path, monkeypatch)
    monkeypatch.setattr(m.boundary, 'scheduler_proof', lambda *a: pytest.fail('must not query/cancel running job'))
    assert not m.migrate_pending_queue(tmp_path, row, lambda: None,
        {'123': ['123', 'RUNNING', 'owner']}, owner='owner')
    assert row['status'] == 'external_submission'


def test_cancelled_boundary_is_durable_before_new_enqueue_and_dryrun(tmp_path, monkeypatch):
    from scripts.campaign_20261110 import controller_families
    row, *_ = source(tmp_path, monkeypatch)
    events = []
    proof = {'never_started': True, 'after': {'accounting': {'state': 'CANCELLED', 'start': 'Unknown'}}}
    monkeypatch.setattr(m.boundary, 'scheduler_proof', lambda jid: {})
    monkeypatch.setattr(m.boundary, 'eligible', lambda *a, **kw: True)
    def cancel(root, record, **kwargs):
        assert record['migration_replacement']['index'] == 50000200
        events.append('cancel'); record.update(status='complete', migration=proof); kwargs['save'](); return proof
    monkeypatch.setattr(m.boundary, 'migrate_record', cancel)
    monkeypatch.setattr(controller_families, 'run_logged', lambda *a: events.append('dryrun'))
    def enqueue(root, record, save, **kwargs):
        assert 'allocation' not in record and 'migration' not in record
        assert Path(record['pending_migration_source']['proof']['path']).exists()
        assert kwargs['backends']==['minyeong_svl','vast_4090']
        assert record['migration_routing']['proof']==record['pending_migration_source']['proof']
        events.append('enqueue'); record['status'] = 'external_submission'; save()
    monkeypatch.setattr(m, 'enqueue_record', enqueue)
    saved = []
    assert m.migrate_pending_queue(tmp_path, row, lambda: saved.append(deepcopy(row)),
        {'123': ['123', 'PENDING', 'owner']}, owner='owner')
    assert events == ['cancel', 'dryrun', 'enqueue']
    assert any(s.get('migration_replacement') for s in saved)
    assert row['index'] == 50000200 and row['prepared']['group_ids'] == ['old_0000_pending_m01']
    assert row['pending_migration_source']['original_job_id'] == '123'


def test_restart_after_publication_boundary_only_retries_idempotent_enqueue(tmp_path, monkeypatch):
    row, *_ = source(tmp_path, monkeypatch)
    row = m.prepare_replacement(tmp_path, row)
    proof = write(tmp_path/'proof.json', {'never_started': True})
    row['pending_migration_source']['proof'] = {'path': str(proof), 'sha256': file_digest(proof)}
    calls = []
    monkeypatch.setattr(m, 'enqueue_record', lambda *a, **kw: calls.append(a[1]['index']))
    monkeypatch.setattr(m.boundary, 'migrate_record', lambda *a, **kw: pytest.fail('no recancel'))
    assert m.migrate_pending_queue(tmp_path, row, lambda: None, {}, owner='owner')
    assert calls == [50000200]


def test_old_prepared_replacement_recovers_original_routes_before_enqueue(tmp_path, monkeypatch):
    row, *_ = source(tmp_path, monkeypatch)
    row = m.prepare_replacement(tmp_path, row)
    row.pop('migration_routing');row['dispatch_backends']=['vast_4090','vast_5090']
    proof = write(tmp_path/'proof.json', {'never_started': True})
    row['pending_migration_source']['proof']={'path':str(proof),'sha256':file_digest(proof)}
    calls=[]
    monkeypatch.setattr(m,'enqueue_record',lambda *args,**kwargs:calls.append(kwargs['backends']))
    monkeypatch.setattr(m.boundary,'migrate_record',lambda *a,**k:pytest.fail('do not recancel'))
    assert m.migrate_pending_queue(tmp_path,row,lambda:None,{},owner='owner')
    assert calls==[['minyeong_svl','vast_4090']]
