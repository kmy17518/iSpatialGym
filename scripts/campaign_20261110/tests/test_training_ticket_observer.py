import csv
import json
from pathlib import Path
import sqlite3
import subprocess
import time

import pytest

from scripts.campaign_20261110 import training_ticket_observer as observer
from scripts.campaign_20261110 import training_diversity as tickets
from scripts.campaign_20261110.contracts import ContractError, digest, file_digest, write_json_atomic
from scripts.campaign_20261110.tests.test_training_diversity import setup, reserve, C


def fixture(root, monkeypatch):
    plan, _, _ = setup(root, monkeypatch, capacity=3, future=2)
    cfg = root/'groups/group/campaign_config.json'
    config = {'purpose': 'training_production', 'split': 'train', 'validation_key': {'code_sha256': 'a'*64}}
    write_json_atomic(cfg, config); config_sha = file_digest(cfg)
    ticket = reserve(root, plan, count=3, config=config_sha)
    queue_path = root/'queues/token/queue.json'
    write_json_atomic(queue_path, {'config_path': str(cfg), 'config_sha256': config_sha})
    command = ['python', 'submit.py', 'submit', C, '3', '--code', str(root/'code/native')]
    allocation = {'key': C, 'start': '50000', 'n': '3', 'code': str(root/'code/native'),
                  'tag': 'goal_groups', 'jobid': '12345', 'job_dir': str(root/'out/job')}
    ledger = root/'manifests/allocations.tsv'
    with ledger.open('w') as stream:
        writer = csv.DictWriter(stream, delimiter='\t', fieldnames=list(allocation)); writer.writeheader(); writer.writerow(allocation)
    allocation['ledger'] = str(ledger)
    identity = {k: int(allocation[k]) if k in ('start', 'n') else allocation[k] for k in ('key','start','n','code','tag')}
    receipt = root/'dispatch/receipts/token.json'; intent = root/'dispatch/ready/token.json'
    write_json_atomic(intent, {'token': 'token', 'identity': identity, 'command': command, 'command_sha256': digest(command)})
    write_json_atomic(receipt, {'status': 'submitted', 'token': 'token', 'identity': identity,
                               'command_sha256': digest(command), 'allocation': allocation})
    record = {'queue': str(queue_path), 'status': 'complete', 'command': command, 'index': 50000, 'count': 3,
              'allocation': allocation, 'directory': str(queue_path.parent), 'dispatch_intent': str(intent),
              'dispatch_receipt': str(receipt), 'groups': {'group': {'accepted': {}}}}
    producer = root/'controllers/producer/state.json'
    write_json_atomic(producer, {'updated_at': time.time(), 'seeds': {'seed': {'queues': [record]}}})
    controller = root/'controllers/producer/ticket.json'
    write_json_atomic(controller, {'schema': 'isg-training-ticket-controller-v1', 'campaign_id': tickets.CAMPAIGN,
        'ticket': ticket, 'owner': '/root/dataset_contract', 'config_sha256': config_sha,
        'start': 50000, 'count': 3, 'phase': 'prepared', 'queue': observer.reference(queue_path),
        'command': command, 'command_sha256': digest(command), 'jobs': [],
        'pending_dispatch': ['token'], 'pending_acceptance': ['group']})
    tickets.bind_ticket_controller(root, ticket, controller)
    registration = observer.register(root, ticket, producer)
    job = root/'out/job'; job.mkdir(parents=True)
    release = root/'release.json'; write_json_atomic(release, {'config_sha256': config_sha, 'campaign_started_at': 1794321600})
    environment = {'ISPATIALGYM_TRAINING_DIVERSITY_TICKET': ticket['path'],
        'ISPATIALGYM_TRAINING_DIVERSITY_TICKET_SHA256': ticket['sha256'], 'START': '50000', 'N': '3',
        'ISPATIALGYM_CAMPAIGN_CONFIG': str(cfg), 'ISPATIALGYM_TRAINING_RELEASE': str(release),
        'ISPATIALGYM_TRAINING_RELEASE_SHA256': file_digest(release)}
    (job/'job.env').write_text(''.join(k+'='+v+'\n' for k,v in environment.items()))
    validation = root/'validation/controller.json'
    write_json_atomic(validation, {'updated_at': time.time(), 'gates': {'12345': {
        'source_job_id': '12345', 'source_job_dir': str(job), 'status': 'demo_missing'}}})
    database = root/'dataset/allocations.sqlite'; database.parent.mkdir()
    with sqlite3.connect(database) as db: db.execute('CREATE TABLE members(record TEXT)')
    # The existing queue parser owns native-plan and closure-event validation;
    # these tests exercise producer, dispatch, job, quota and acceptance joins.
    from scripts.campaign_20261110 import controller_held
    monkeypatch.setattr(controller_held, 'queue_closures', lambda *args: {'group': {'written': 0}})
    def scheduler(command, **kwargs):
        assert command[0] in ('squeue', 'sacct')
        ids = command[command.index('-j')+1].split(',')
        body = '' if command[0] == 'squeue' else ''.join(i+'|COMPLETED|2026-11-11T18:00:00|0:0\n' for i in ids)
        return subprocess.CompletedProcess(command, 0, body, '')
    monkeypatch.setattr(tickets.subprocess, 'run', scheduler)
    return {'ticket': ticket, 'registration': registration, 'controller': controller, 'producer': producer,
            'validation': validation, 'job': job, 'receipt': receipt, 'plan': plan, 'config': cfg}


def mutate(path, fn):
    value = json.loads(path.read_text()); fn(value); write_json_atomic(path, value)


def test_actual_terminal_producer_returns_only_unused_quota(tmp_path, monkeypatch):
    f = fixture(tmp_path, monkeypatch)
    result = observer.reconcile(tmp_path, f['registration'])
    receipt = tickets._read_reference(tmp_path, result['receipt'])
    assert result['status'] == 'retired' and receipt['released_indices'] == [50000, 50001, 50002]
    assert receipt['indices_remain_reserved'] and not receipt['production_authorization']
    assert observer.reconcile(tmp_path, f['registration'])['status'] == 'already_retired'
    assert observer.register(tmp_path, f['ticket'], f['producer']) == f['registration']
    with pytest.raises(ContractError, match='overlaps'):
        reserve(tmp_path, f['plan'], 'next', 50000, 3, config=file_digest(f['config']))


@pytest.mark.parametrize('mutation', [
    lambda f: mutate(f['producer'], lambda x: x.update(updated_at=1)),
    lambda f: mutate(f['producer'], lambda x: x['seeds']['seed']['queues'][0].update(status='external_submission')),
    lambda f: mutate(f['producer'], lambda x: x['seeds']['seed']['queues'][0].update(index=50001)),
    lambda f: mutate(f['producer'], lambda x: x['seeds']['seed']['queues'][0]['groups'].update(extra={'accepted': {}})),
    lambda f: mutate(f['receipt'], lambda x: x.update(status='submitting')),
    lambda f: mutate(f['validation'], lambda x: x['gates']['12345'].update(status='replay_queued', replay_job_id='67890')),
    lambda f: mutate(f['validation'], lambda x: x.update(updated_at=1)),
    lambda f: mutate(f['validation'], lambda x: x['gates']['12345'].update(source_job_dir='/foreign/job')),
    lambda f: mutate(f['controller'], lambda x: x.update(jobs=[{'backend':'cluster','job_id':'67890','stage':'rendering'}])),
])
def test_incomplete_stale_or_different_scope_stays_reserved(tmp_path, monkeypatch, mutation):
    f = fixture(tmp_path, monkeypatch); mutation(f)
    with pytest.raises((ContractError, KeyError)): observer.reconcile(tmp_path, f['registration'])
    assert not tickets._read_state(tmp_path).get('retirements')
    assert json.loads(f['controller'].read_text())['phase'] == 'prepared'


def test_pending_retry_keeps_the_original_ticket_open(tmp_path, monkeypatch):
    f = fixture(tmp_path, monkeypatch)
    write_json_atomic(tmp_path/'validation/replay_retry_requests.json', {'requests': {'12345': {'request_id': 'retry2'}}})
    with pytest.raises(ContractError, match='unapplied replay retry'): observer.reconcile(tmp_path, f['registration'])


def test_raw_running_stage_blocks_finalization(tmp_path, monkeypatch):
    f = fixture(tmp_path, monkeypatch); real = tickets.subprocess.run
    def active(command, **kwargs):
        if command[0] == 'squeue': return subprocess.CompletedProcess(command, 0, '12345|RUNNING|gpu0\n', '')
        return real(command, **kwargs)
    monkeypatch.setattr(tickets.subprocess, 'run', active)
    with pytest.raises(ContractError, match='active or scheduler unavailable'): observer.reconcile(tmp_path, f['registration'])
    assert json.loads(f['controller'].read_text())['phase'] == 'prepared'


def test_all_retry_and_migration_attempts_are_observed(tmp_path, monkeypatch):
    f = fixture(tmp_path, monkeypatch)
    mutate(f['validation'], lambda x: x['gates']['12345'].update(status='replay_missing', replay_job_id='333',
        attempt_history=[{'replay_job_id':'222','vast_migration':{'original_replay_job_id':'111'}}]))
    registration = tickets._read_reference(tmp_path, f['registration'])
    candidate, _, _ = observer.derive(tmp_path, registration, tmp_path/'candidate')
    assert {j['job_id'] for j in candidate['jobs'] if j['stage']=='replay'} == {'111','222','333'}


def test_producer_race_after_backend_query_cannot_finalize(tmp_path, monkeypatch):
    f = fixture(tmp_path, monkeypatch); real = tickets.observe_ticket_jobs
    def race(*args, **kwargs):
        result = real(*args, **kwargs)
        mutate(f['producer'], lambda x: x['seeds']['seed']['queues'][0].update(status='external_submission'))
        return result
    monkeypatch.setattr(tickets, 'observe_ticket_jobs', race)
    with pytest.raises(ContractError, match='original source changed'): observer.reconcile(tmp_path, f['registration'])
    assert json.loads(f['controller'].read_text())['phase'] == 'prepared'


def test_cycle_reports_owned_blocker_without_dispatch(tmp_path, monkeypatch):
    f = fixture(tmp_path, monkeypatch)
    mutate(f['producer'], lambda x: x['seeds']['seed']['queues'][0].update(status='external_submission'))
    report = observer.cycle(tmp_path)
    assert not report['jobs_modified'] and not report['instances_accepted']
    assert report['rows'][0]['status'] == 'awaiting_evidence_or_owned_repair'
    assert 'unfinished' in report['rows'][0]['reason']
    assert not tickets._read_state(tmp_path).get('retirements')


def published_member(root, f, monkeypatch):
    from scripts.campaign_20261110 import controller_held, acceptance
    monkeypatch.setattr(controller_held, 'queue_closures', lambda *args: {'group': {'written': 1}})
    monkeypatch.setattr(acceptance, 'report_passed', lambda *args: None)
    iid = 'member__50000'; tro = f['job']/'inst'/(iid+'-tro_state.json')
    config = json.loads(f['config'].read_text())
    write_json_atomic(tro, {'ispatialgym': {'instance_id': iid, 'campaign': {
        'validation_key':config['validation_key'], 'class_id':C, 'task':'object_scaling', 'goal_group_id':'group'}}})
    write_json_atomic(f['job']/'demo'/iid/'meta.json', {'success':True,'instance_id':iid})
    mutate(f['validation'], lambda x: x['gates']['12345'].update(status='validated', replay_job_id='67890'))
    return iid, tro


def test_successful_unclaimed_member_is_not_released(tmp_path, monkeypatch):
    f = fixture(tmp_path, monkeypatch); published_member(tmp_path, f, monkeypatch)
    with pytest.raises(ContractError, match='successful uncommitted member'):
        observer.reconcile(tmp_path, f['registration'])
    assert not tickets._read_state(tmp_path).get('retirements')


def test_committed_member_requires_review_and_stays_reserved(tmp_path, monkeypatch):
    from scripts.campaign_20261110 import acceptance
    f = fixture(tmp_path, monkeypatch); iid, tro = published_member(tmp_path, f, monkeypatch)
    member = {'instance_id':iid, 'index':50000,'group_id':'group','split':'train','task':'object_scaling',
        'class_id':C,'training_diversity_ticket':f['ticket'],'training_config_sha256':file_digest(f['config'])}
    with sqlite3.connect(tmp_path/'dataset/allocations.sqlite') as db:
        db.execute('INSERT INTO members VALUES (?)',(json.dumps(member),))
    mutate(f['producer'], lambda x: x['seeds']['seed']['queues'][0]['groups']['group'].update(accepted={iid:'train'}))
    with pytest.raises(ContractError,match='lacks original strict review'): observer.reconcile(tmp_path,f['registration'])
    review={'source_job':str(f['job']), 'gate':{'path':str(tmp_path/'gate.json')}, 'members':[{'instance_id':iid}]}
    rp=tmp_path/'groups/group/release_review.json';write_json_atomic(rp,review)
    mutate(f['producer'],lambda x:x['seeds']['seed']['queues'][0]['groups']['group'].update(review_sha256=digest(review)))
    monkeypatch.setattr(acceptance,'audit_job',lambda *args,**kwargs:review)
    result=observer.reconcile(tmp_path,f['registration'])
    receipt=tickets._read_reference(tmp_path,result['receipt'])
    assert receipt['released_indices']==[50001,50002] and receipt['committed_records']==[member]
