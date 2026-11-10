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
