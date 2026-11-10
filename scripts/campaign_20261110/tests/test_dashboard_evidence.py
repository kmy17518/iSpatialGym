from dataclasses import asdict

import pytest

from scripts.campaign_20261110 import dashboard_evidence as de, contracts as c
from scripts.campaign_20261110.tests.test_group_lane import candidate, claim


def catalog(path):
    rows=[]
    for quota in c.task_quotas():
        for index in range(quota.standard_classes):
            name='object_scaling-apple' if quota.task=='object_scaling' and index==0 else f'{quota.task}_{index+1:02}'
            rows.append({'class_id':name,'task':quota.task})
    c.write_json_atomic(path,rows)
    return rows


def test_coherent_authority_counts_not_demos_or_orphan_exports(tmp_path):
    f,review=candidate(tmp_path,1)
    result=claim(tmp_path,f,review)
    # Source contracts are the exact accepted group identity, not an inventory scan.
    config_path=tmp_path/'groups'/review['group_id']/'campaign_config.json'
    c.write_json_atomic(config_path,{'template':asdict(f.template),'split':f.template.split,'validation_key':result['members'][0]['validation_key']})
    (f.job/'job.env').write_text('ISPATIALGYM_CAMPAIGN_CONFIG='+str(config_path)+'\n')
    c.write_json_atomic(tmp_path/'dataset/groups/orphan.json',{'accepted':{'not-committed':'eval_standard'}})
    c.write_json_atomic(tmp_path/'validation/controller.json',{'gates':{'123':{
        'key':f.template.class_id,'source_job_dir':str(f.job),'status':'replay_failed','demo_successes':900,'reason':'physical restoration failed'}}})
    standards=tmp_path/'standards.json';inventory=catalog(standards)
    inventory[next(i for i,row in enumerate(inventory) if row['class_id']=='object_scaling-apple')]['class_id']=f.template.class_id
    c.write_json_atomic(standards,inventory)
    database=tmp_path/'dataset/quotas.sqlite3';before=database.read_bytes()
    report=de.build(tmp_path,standards,database=database,observed_at=1000)
    assert database.read_bytes()==before
    assert report['counts']=={'eval_standard':1,'eval_model':0,'eval_category':0,'train':1}
    assert report['accepted_total']==len(result['members'])==2
    assert len(report['classes'])==322 and len(report['tasks'])==23
    assert {split:sum(task['targets'][split] for task in report['tasks']) for split in de.SPLITS}==report['targets']
    assert {key:report['templates'][key] for key in ('accepted_physical','accepted_conceptual','accepted_classes')}=={'accepted_physical':1,'accepted_conceptual':1,'accepted_classes':1}
    assert report['templates']['identity_audit_complete'] and report['templates']['accepted_contract_fingerprints']==1
    row=next(row for row in report['classes'] if row['class_id']==f.template.class_id and row['split']=='eval_standard')
    assert row['reason']=='physical restoration failed'
    assert row['accepted']==1 and row['remaining']==9
    assert any(source.get('sha256')==c.file_digest(database) for source in report['sources'])
    assert not report['errors']


def test_family_reasons_without_unversioned_config_and_split_separation(tmp_path):
    standard=tmp_path/'standards.json';catalog(standard);classes=de.standard_catalog(standard)
    state=tmp_path/'controllers/families/state.json'
    c.write_json_atomic(state,{'updated_at_unix':900,'families':{'object_scaling-apple':{
        'status':'blocked','error':{'message':'goal outside support'}}}})
    c.write_json_atomic(state.with_name('config.json'),{'families':[{'class_id':'object_scaling-apple','split':'eval_standard'}]})
    reasons,sources,errors=de.controller_reasons(tmp_path,classes)
    assert 'goal outside support' in reasons['object_scaling','object_scaling-apple'][0]['reason']
    assert sources[0]['sha256']==c.file_digest(state) and not errors
    reasons['object_scaling','object_scaling-apple'].append({'source_kind':'controller_observation','split':'eval_model','observed_at':950,'reason':'model unavailable'})
    reasons['object_scaling','object_scaling-apple'].append({'source_kind':'validation_observation','status':'validated','observed_at':999,'reason':'validated'})
    report=de.summarize([],classes,root=tmp_path,observed_at=1000,reasons=reasons)
    standard_row=next(row for row in report['classes'] if row['class_id']=='object_scaling-apple' and row['split']=='eval_standard')
    model_row=next(row for row in report['classes'] if row['class_id']=='object_scaling-apple' and row['split']=='eval_model')
    category_row=next(row for row in report['classes'] if row['task']=='object_scaling' and row['split']=='eval_category')
    assert 'goal outside support' in standard_row['reason']
    assert model_row['reason']=='model unavailable'
    assert category_row['reason']=='No controller evidence in this snapshot.'
    assert len(report['unassigned_reason_evidence'])==1


def test_history_measures_only_accepted_counts_and_rejects_regression(tmp_path):
    standards=tmp_path/'standards.json';catalog(standards);classes=de.standard_catalog(standards)
    report=de.summarize([],classes,root=tmp_path,observed_at=1000)
    output=tmp_path/'reports/dashboard_evidence.json'
    first=de.publish(report,output)
    assert first['throughput'][0]['total_accepted_per_hour'] is None
    second=de.summarize([],classes,root=tmp_path,observed_at=4600)
    second['counts']['train']=100;second['accepted_total']=100
    next(row for row in second['tasks'] if row['task']=='object_scaling')['counts']['train']=100
    de.publish(second,output)
    hourly=next(row for row in second['throughput'] if row['window_seconds']==3600)
    assert hourly['total_accepted_per_hour']==100
    assert hourly['accepted_per_hour']['eval_standard']==0
    assert next(row for row in hourly['tasks'] if row['task']=='object_scaling')['accepted_per_hour']['train']==100
    assert next(row for row in hourly['tasks'] if row['task']=='disembedding')['accepted_per_hour']['train']==0
    third=de.summarize([],classes,root=tmp_path,observed_at=4660)
    with pytest.raises(c.ContractError,match='regressed'):
        de.publish(third,output)
    assert de.read(output)['accepted_total']==100


def test_missing_inventory_classes_and_cross_allocator_duplicates_fail_closed(tmp_path):
    standards=tmp_path/'standards.json';rows=catalog(standards)
    c.write_json_atomic(standards,rows[:-1])
    with pytest.raises(c.ContractError,match='complete frozen'):
        de.standard_catalog(standards)
    row={'instance_id':'same'}
    row.update(class_id='apple')
    with pytest.raises(c.ContractError,match='multiple authoritative'):
        de.summarize([row,row],{},root=tmp_path,observed_at=0)


def test_legacy_camera_class_comes_only_from_its_exact_accepted_contract(tmp_path):
    f,review=candidate(tmp_path,1);claim(tmp_path,f,review)
    rows,_,_=de.snapshot_allocations(tmp_path,tmp_path/'dataset/quotas.sqlite3')
    rows[0].pop('class_id');rows[0]['contract']=asdict(f.template)
    standards=tmp_path/'standards.json';catalog(standards)
    report=de.summarize(rows,de.standard_catalog(standards),root=tmp_path,observed_at=1)
    assert report['accepted_total']==2
    rows[0]['contract']['class_id']='changed'
    with pytest.raises(c.ContractError,match='legacy camera class'):
        de.summarize(rows,de.standard_catalog(standards),root=tmp_path,observed_at=2)
