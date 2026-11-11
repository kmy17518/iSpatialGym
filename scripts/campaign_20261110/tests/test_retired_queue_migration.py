from scripts.campaign_20261110 import controller_held as held
from scripts.campaign_20261110 import controller_queue_migration as queues
from scripts.campaign_20261110 import controller_migration as producers


def test_saved_queue_uses_restartable_migration_not_cancel_and_reauthor(tmp_path,monkeypatch):
    calls=[];record={'status':'external_submission','queue':'exact'}
    monkeypatch.setattr(producers,'controller_check',lambda *a,**k: (_ for _ in ()).throw(AssertionError('cannot author in a retired scope')))
    def migrate(root,row,save,scheduler,*,minimum_age):
        assert row is record and minimum_age==0
        calls.append('exact queue continuation')
    monkeypatch.setattr(queues,'migrate_pending_queue',migrate)
    cfg={'migrate_pending_seconds':0,'_migration_checks_remaining':1}
    held.check_pending_record(record,{},cfg,tmp_path,{},lambda:None,reconcile_only=True)
    assert calls==['exact queue continuation'] and cfg['_migration_checks_remaining']==0


def test_prepared_cancellation_resumes_before_complete_skip(tmp_path,monkeypatch):
    record={'status':'complete','migration_replacement':{'queue':'prepared'},'migration':{'never_started':True}}
    calls=[]
    monkeypatch.setattr(queues,'migrate_pending_queue',lambda *a,**kw:calls.append(a[1]))
    cfg={'migrate_pending_seconds':0,'_migration_checks_remaining':2}
    held.check_pending_record(record,{},cfg,tmp_path,{},lambda:None,reconcile_only=True)
    assert calls==[record]
    # An old stranded cancellation without reserved continuation needs explicit
    # current-quota repair, never an automatic second allocation.
    record.pop('migration_replacement')
    held.check_pending_record(record,{},cfg,tmp_path,{},lambda:None,reconcile_only=True)
    assert len(calls)==1


def test_native_started_or_failed_preflight_does_not_trigger_new_author(tmp_path,monkeypatch):
    record={'status':'external_submission','allocation':{'jobid':'10'}};saves=[]
    def reject(*args,**kwargs):raise ValueError('source has execution artifacts')
    monkeypatch.setattr(queues,'migrate_pending_queue',reject)
    held.check_pending_record(record,{}, {'migrate_pending_seconds':0,'_migration_checks_remaining':1},
        tmp_path,{},lambda:saves.append(True),reconcile_only=True)
    assert record['status']=='external_submission' and record['allocation']['jobid']=='10'
    assert record['migration_error']['reason']=='source has execution artifacts' and saves==[True]


def test_ordinary_producer_keeps_existing_authoring_migration(tmp_path,monkeypatch):
    calls=[]
    monkeypatch.setattr(producers,'controller_check',lambda *a,**kw:calls.append(kw))
    held.check_pending_record({}, {}, {},tmp_path,{},lambda:None,reconcile_only=False)
    assert calls==[{'family':'held'}]
