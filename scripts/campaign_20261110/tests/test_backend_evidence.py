import csv
import json

from scripts.campaign_20261110.backend_evidence import vast_job_ids
from scripts.campaign_20261110 import infra_gpu_accounting


def test_vast_backend_uses_owned_records_after_the_thousandth_job(tmp_path):
    for folder,jid in [('vast/queue','900001000'),('vast/state/jobs','900010234')]:
        path=tmp_path/folder/(jid+'.json');path.parent.mkdir(parents=True,exist_ok=True);path.write_text('{}')
    assert vast_job_ids(tmp_path)=={'900001000','900010234'}
    assert '900000001' not in vast_job_ids(tmp_path)  # A numeric shape is not allocation evidence.


def test_accounting_never_queries_slurm_for_an_authoritative_vast_allocation(tmp_path,monkeypatch):
    job=tmp_path/'out/job';job.mkdir(parents=True)
    state=tmp_path/'vast/state/jobs/900001000.json';state.parent.mkdir(parents=True)
    state.write_text(json.dumps({'id':'900001000','status':'queued','tries':0}))
    ledger=tmp_path/'manifests/dispatch_test_allocations.tsv';ledger.parent.mkdir()
    with ledger.open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=['jobid','job_dir','key','tag','start','n','code'],delimiter='\t')
        writer.writeheader();writer.writerow({'jobid':'900001000','job_dir':str(job),'key':'test','tag':'fresh','start':1,'n':3,'code':'native'})
    monkeypatch.setattr(infra_gpu_accounting.subprocess,'run',lambda *a,**kw:(_ for _ in ()).throw(AssertionError('must not query Slurm')))
    value=infra_gpu_accounting.snapshot(tmp_path)
    assert value['records'][0]['backend']=='vast' and value['records'][0]['state']=='queued'
