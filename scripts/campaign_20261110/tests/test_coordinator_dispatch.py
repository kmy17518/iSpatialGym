import json
from types import SimpleNamespace
import pytest
from scripts.campaign_20261110.coordinator_dispatch import pending_status, submit


def ticket(tmp_path):
    return {'id':'test','command':['sbatch','--parsable','job.sh'],'inputs':{},
            'claim_path':str(tmp_path/'launch.claim'),'claim_kind':'directory',
            'submission_path':str(tmp_path/'submission.json')}


def test_no_duplicate_submission(tmp_path):
    row=ticket(tmp_path);calls=[]
    def run(*a,**kw):
        calls.append(a);return SimpleNamespace(stdout='123\n',stderr='',returncode=0)
    cap={'observed_at':1,'accounts_checked':['minyeong','shiyuc']}
    assert submit(row,cap,run)['job_id']=='123'
    assert submit(row,cap,run) is None and len(calls)==1


def test_ambiguous_submission_never_retries(tmp_path):
    row=ticket(tmp_path)
    def interrupted(*a,**kw):raise OSError('connection interrupted')
    with pytest.raises(OSError):submit(row,{'observed_at':1,'accounts_checked':[]},interrupted)
    assert pending_status(row)=='submission_requires_reconciliation'
    assert submit(row,{},interrupted) is None


def test_existing_owner_claim_is_preserved(tmp_path):
    row=ticket(tmp_path);(tmp_path/'launch.claim').mkdir()
    assert pending_status(row)=='claim_requires_reconciliation'
    assert submit(row,{}) is None


def test_changed_input_fails_before_claim(tmp_path):
    row=ticket(tmp_path);p=tmp_path/'source';p.write_text('changed');row['inputs']={str(p):'0'*64}
    with pytest.raises(ValueError):submit(row,{})
    assert not (tmp_path/'launch.claim').exists()
