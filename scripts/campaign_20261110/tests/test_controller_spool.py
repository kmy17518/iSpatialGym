import pytest

from scripts.campaign_20261110 import controller_spool as spool, infra_dispatch
from scripts.campaign_20261110.contracts import ContractError


def test_spool_requires_preflight_and_keeps_original_command_and_pending_ownership(tmp_path,monkeypatch):
    calls=[];saved=[]
    def enqueue(root,token,command,**kwargs):
        calls.append((root,token,list(command),kwargs))
        return root/'dispatch/ready'/(token+'.json')
    monkeypatch.setattr(infra_dispatch,'enqueue',enqueue)
    command=['python','submit_fast.py','submit','class','6','--base-index','50000000']
    record={'directory':str(tmp_path/'queues/fresh'),'command':command,'status':'ready'}
    with pytest.raises(ContractError,match='dry-run'):
        spool.enqueue_record(tmp_path,record,lambda:saved.append(True))
    record['dry_run_passed']=True
    spool.enqueue_record(tmp_path,record,lambda:saved.append(True))
    assert record['status']=='external_submission' and 'allocation' not in record
    assert calls[0][2]==command and calls[0][1]=='fresh' and saved==[True]
    assert record['dispatch_receipt']==str(tmp_path/'dispatch/receipts/fresh.json')


def test_known_legacy_nfs_runtime_routes_only_to_local_storage(tmp_path,monkeypatch):
    calls=[]
    monkeypatch.setattr(infra_dispatch,'enqueue',lambda root,token,command,**kwargs:calls.append(kwargs) or root/'intent.json')
    code=tmp_path/'code/frozen';source=code/'ispatialgym/sampling/campaign_runtime.py';source.parent.mkdir(parents=True)
    source.write_text('fcntl.flock(stream, fcntl.LOCK_EX)\n')
    record={'directory':str(tmp_path/'queue'),'command':['python','submit_fast.py','submit','class','3','--code',str(code)],'dry_run_passed':True}
    spool.enqueue_record(tmp_path,record,lambda:None)
    assert calls[0]['backends']==('vast_5090','vast_4090')
    assert record['routing_constraint']['reason']=='known_unsupported_nfs_failure_journal_lock'
    record['dispatch_backends']=['shiyuc_svl']
    with pytest.raises(ContractError,match='local-storage'):
        spool.enqueue_record(tmp_path,record,lambda:None)
    source.write_text('append_using_atomic_directory_lock()\n')
    spool.enqueue_record(tmp_path,record,lambda:None)
    assert calls[-1]['backends']==('shiyuc_svl',)
    spool.enqueue_record(tmp_path,record,lambda:None,backends=('vast_4090',))
    assert calls[-1]['backends']==('vast_4090',)
