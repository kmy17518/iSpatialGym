import json,sqlite3
from scripts.campaign_20261110 import current_eligibility as e
from scripts.campaign_20261110.contracts import digest,file_digest,write_json_atomic


def members():
    base={'task':'object_scaling','class_id':'object_scaling-apple','sampling_split':'eval_standard','group_id':'g'}
    return [{**base,'instance_id':'eval__50000','split':'eval_standard'},
            {**base,'instance_id':'train__50001','split':'train'},
            {**base,'instance_id':'train__50002','split':'train'}]


def ledger(root,rows,statuses):
    proof=root/'review.json';write_json_atomic(proof,{'executed_check':'fixture'})
    decisions=[{'instance_id':r['instance_id'],'accepted_record_sha256':digest(r),'status':status,
      'action':'compatible'if status=='verified'else'investigate','change_ids':['change'],
      'review_evidence':[{'path':str(proof),'sha256':file_digest(proof)}]}for r,status in zip(rows,statuses)]
    value={'schema':'isg-change-impact-ledger-v1','campaign_id':'campaign_20261110','decisions':decisions}
    write_json_atomic(root/'validation/change_impact/ledger.json',value);return value


def group(rows):return {'g':{'group_id':'g','accepted':{r['instance_id']:r['split']for r in rows},'members':rows,'deferred':[]}}


def test_lost_sibling_removes_eval_without_mutating_history(tmp_path):
    rows=members()[:2];original=digest(rows);ledger(tmp_path,rows,['verified','excluded'])
    status,consequences=e.assess(tmp_path,rows,group(rows))
    assert status[rows[0]['instance_id']]['status']=='under_review'
    assert not any(s['quota_eligible']for s in status.values())
    assert consequences['g']['effective_accepted']=={} and digest(rows)==original


def test_one_surviving_sibling_keeps_eval_and_failed_eval_reassesses_singleton(tmp_path):
    rows=members();ledger(tmp_path,rows,['verified','verified','excluded'])
    status,_=e.assess(tmp_path,rows,group(rows));assert [s['quota_eligible']for s in status.values()]==[True,True,False]
    ledger(tmp_path,rows,['excluded','verified','excluded'])
    status,c=e.assess(tmp_path,rows,group(rows));assert c['g']['singleton_reassessment']
    effective=e.effective_group(tmp_path,rows[1],group(rows)['g'])
    assert effective['accepted']=={rows[1]['instance_id']:'train'}
    from scripts.campaign_20261110.training_eligibility import classify_record
    assert classify_record(rows[1],group(rows)['g'],root=tmp_path)['category']=='eval_singleton'


def test_multiple_train_survivors_without_eval_need_explicit_reassignment(tmp_path):
    rows=members();ledger(tmp_path,rows,['excluded','verified','verified'])
    status,_=e.assess(tmp_path,rows,group(rows))
    assert all(not s['quota_eligible']for s in status.values())


def test_unknown_scope_is_visible_not_a_blanket_exclusion_and_camera_exempt(tmp_path):
    rows=members()[:1];rows[0]['task']='camera_relocalization-standard'
    status,_=e.assess(tmp_path,rows,{})
    assert status[rows[0]['instance_id']]['status']=='scope_not_reconciled' and status[rows[0]['instance_id']]['quota_eligible']
    ledger(tmp_path,rows,['verified']);status,_=e.assess(tmp_path,rows,{})
    assert status[rows[0]['instance_id']]['quota_eligible']


def test_changed_review_or_accepted_identity_fails_closed_for_exact_scope(tmp_path):
    rows=members()[:1];ledger(tmp_path,rows,['verified']);(tmp_path/'review.json').write_text('{}')
    status,_=e.assess(tmp_path,rows,{})
    assert status[rows[0]['instance_id']]['status']=='under_review'
    ledger(tmp_path,rows,['verified']);rows[0]['bucket']=2
    status,_=e.assess(tmp_path,rows,{})
    assert not status[rows[0]['instance_id']]['quota_eligible']


def test_connection_local_quota_view_preserves_all_historical_rows(tmp_path):
    rows=members()[:2];ledger(tmp_path,rows,['verified','excluded']);groups=group(rows)
    db=sqlite3.connect(':memory:');db.executescript('CREATE TABLE members(id,group_id,split,task,class_id,bucket,record);CREATE TABLE groups(id,record);')
    for r in rows:db.execute('INSERT INTO members VALUES (?,?,?,?,?,?,?)',(r['instance_id'],'g',r['split'],r['task'],r['class_id'],0,json.dumps(r)))
    db.execute('INSERT INTO groups VALUES (?,?)',('g',json.dumps(groups['g'])))
    assert len(e.install_quota_view(tmp_path,db))==2
    assert db.execute('SELECT COUNT(*) FROM current_members').fetchone()[0]==0
    assert db.execute('SELECT COUNT(*) FROM members').fetchone()[0]==2
