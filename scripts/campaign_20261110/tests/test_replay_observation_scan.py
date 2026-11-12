import json
import pytest
from scripts.campaign_20261110.controller_validation import terminal_observation_unchanged, order_ledger_rows, observation_waits_for_demo, observation_waits_for_sampling


def fixture(tmp_path):
    job=tmp_path/'job';job.mkdir();(job/'sampling.done').touch()
    out=tmp_path/'replay';out.mkdir();gate={'success':True,'validated_at':'2026-11-11T01:00:00Z'}
    (out/'gate.json').write_text(json.dumps(gate))
    row={'jobid':'1','job_dir':str(job),'key':'example','code':'native'}
    entry={'source_job_id':'1','source_job_dir':str(job),'key':'example','output':str(out),
           'status':'validated','gate':gate,'gate_quality':{'passed':True}}
    return row,entry,out


def test_terminal_reuse_preserves_historical_proof(tmp_path):
    row,entry,out=fixture(tmp_path);before=json.dumps(entry,sort_keys=True)
    assert terminal_observation_unchanged(entry,row,None)
    assert json.dumps(entry,sort_keys=True)==before


@pytest.mark.parametrize('change',['retry','gate','quality','source','sampling','migration','active'])
def test_actions_or_unverified_terminal_changes_traverse_source_gate(tmp_path,change):
    row,entry,out=fixture(tmp_path);request=None
    if change=='retry':request={'request_id':'fresh'}
    elif change=='gate':(out/'gate.json').write_text(json.dumps({'success':False}))
    elif change=='quality':entry.pop('gate_quality')
    elif change=='source':row['job_dir']=str(tmp_path/'other')
    elif change=='sampling':(tmp_path/'job/sampling.done').unlink()
    elif change=='migration':entry['vast_migration']={'status':'prepared'}
    else:entry['status']='replay_queued'
    assert not terminal_observation_unchanged(entry,row,request)


def test_explicit_retry_and_new_work_precede_history_and_duplicates():
    rows=[{'jobid':str(i),'job_dir':str(i),'key':'x','code':'n'} for i in range(4)]
    gates={'0':{'status':'validated'},'1':{'status':'replay_failed'},'2':{'status':'replay_queued'}}
    ordered=order_ledger_rows(rows+[dict(rows[0])],gates,{'1':{'request_id':'retry'}})
    assert [r['jobid']for r in ordered]==['1','2','3','0']


def test_changed_ledger_identity_is_not_silently_deduplicated():
    rows=[{'jobid':'1','job_dir':p,'key':'x','code':'n'}for p in ('one','two')]
    assert len(order_ledger_rows(rows,{},{}))==2


@pytest.mark.parametrize('status',['waiting_for_demo','demo_failed','demo_missing'])
def test_jobs_without_success_cannot_authorize_or_dispatch_replay(status):
    entry={'status':status}
    assert observation_waits_for_demo(entry,[],None)
    assert not observation_waits_for_demo(entry,[{'success':True}],None)
    assert not observation_waits_for_demo(entry,[],{'request_id':'retry'})


@pytest.mark.parametrize('field',['replay_job_id','vast_migration'])
def test_existing_replay_or_migration_still_reconciles_without_demo_results(field):
    entry={'status':'waiting_for_demo',field:'existing'}
    assert not observation_waits_for_demo(entry,[],None)


def test_open_source_cannot_dispatch_but_closure_immediately_rechecks_authorization(tmp_path):
    entry={'status':'waiting_for_demo','demo_successes':6,'production_gate_block':{'allowed':False}}
    assert observation_waits_for_sampling(entry,tmp_path,None)
    (tmp_path/'sampling.done').touch()
    assert not observation_waits_for_sampling(entry,tmp_path,None)
    assert entry['production_gate_block']=={'allowed':False}


@pytest.mark.parametrize('field',['replay_job_id','vast_migration'])
def test_open_source_owned_replay_and_migration_still_reconcile(tmp_path,field):
    assert not observation_waits_for_sampling({'status':'waiting_for_demo',field:'owned'},tmp_path,None)


def test_explicit_retry_still_checks_current_source_even_before_close(tmp_path):
    assert not observation_waits_for_sampling({'status':'waiting_for_demo'},tmp_path,{'request_id':'r1'})


def test_interrupted_retry_and_new_sources_precede_prior_denials():
    rows=[{'jobid':str(i),'job_dir':str(i),'key':'x','code':'n'}for i in range(5)]
    entries={'0':{'status':'waiting_for_demo','production_gate_block':{'allowed':False}},
             '1':{'status':'waiting_for_demo','retry_output':'exact_retry'},
             '2':{'status':'replay_queued','replay_job_id':'42','production_gate_block':{'allowed':False}},
             '4':{'status':'validated'}}
    assert [r['jobid']for r in order_ledger_rows(rows,entries,{})]==['1','2','3','0','4']
