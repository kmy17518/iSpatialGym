import copy
import json
import pytest
from scripts.campaign_20261110 import training_diversity as d
from scripts.campaign_20261110.contracts import ContractError, file_digest, write_json_atomic

C='object_scaling-apple'
H='a'*64

def setup(tmp_path, monkeypatch, capacity=4, future=2, accepted=None):
    cell={'id':'cell1','class_id':C,'scene':'house','room':'kitchen','support_id':'table1',
          'model_keys':['apple/ready1'],'target_scale_id':'s1','support_height_id':'h1','capacity':capacity}
    dims=d.cell_dimensions(cell)
    plan={'schema':d.SCHEMA,'campaign_id':d.CAMPAIGN,'task':'object_scaling','task_budget':capacity+future,
          'class_budgets':{C:capacity+future},'inventory_complete':True,'dimension_inventory':copy.deepcopy(dims),
          'cells':[cell],'reserved_future_capacity':[]}
    if future:
        extra={'class':[C],'location':['house/bedroom/table2'],'model':['apple/ready2'],'scale':['s2/h2']}
        for key,vals in extra.items():plan['dimension_inventory'][key]=sorted(set(dims[key]+vals))
        plan['reserved_future_capacity']=[{'id':'future','class_id':C,'capacity':future,'dimensions':extra}]
    p=tmp_path/'plan.json';write_json_atomic(p,plan)
    write_json_atomic(tmp_path/'manifests/family_index_reservations.json',{'reservations':{'owner':{'start':50000,'n':100}}})
    records=accepted if accepted is not None else []
    monkeypatch.setattr(d,'_records',lambda root,task:copy.deepcopy(records))
    return p,plan,records

def reserve(root,p,request='r1',start=50000,count=2,config=H):
    return d.reserve_ticket(root,request,config,start,count,'cell1',p,file_digest(p))

def test_exact_idempotent_ticket_is_read_only_and_not_authorization(tmp_path,monkeypatch):
    p,_,_=setup(tmp_path,monkeypatch)
    ref=reserve(tmp_path,p)
    before={str(x):x.read_bytes() for x in tmp_path.rglob('*') if x.is_file()}
    proof=d.check_ticket(tmp_path,H,50000,2,ref)
    assert not proof['production_authorization']
    assert before=={str(x):x.read_bytes() for x in tmp_path.rglob('*') if x.is_file()}
    assert reserve(tmp_path,p)==ref
    for config,start,count in [('b'*64,50000,2),(H,50001,2),(H,50000,1)]:
        with pytest.raises(ContractError):d.check_ticket(tmp_path,config,start,count,ref)
    with pytest.raises(ContractError,match='idempotent'):reserve(tmp_path,p,count=3)

def test_global_overlap_and_common_index_namespace(tmp_path,monkeypatch):
    p,_,_=setup(tmp_path,monkeypatch)
    reserve(tmp_path,p)
    with pytest.raises(ContractError,match='overlaps'):reserve(tmp_path,p,'r2',50001)
    with pytest.raises(ContractError,match='common index'):reserve(tmp_path,p,'r2',60000)
    reserve(tmp_path,p,'r2',50002)
    with pytest.raises(ContractError,match='cell capacity'):reserve(tmp_path,p,'r3',50004,1)

@pytest.mark.parametrize('mutation',[
    lambda p:p.update(inventory_complete=False),
    lambda p:p['dimension_inventory']['model'].append('apple/unreserved'),
    lambda p:p['reserved_future_capacity'][0].update(dimensions={'class':[C]}),
    lambda p:p['class_budgets'].update({C:10001}),
    lambda p:p.update(task_budget=10001),
    lambda p:p['cells'][0].update(id=''),
    lambda p:p['cells'][0].update(class_id='mental_rotation-book'),
])
def test_incomplete_or_overbudget_plans_fail_closed(tmp_path,monkeypatch,mutation):
    p,plan,_=setup(tmp_path,monkeypatch);mutation(plan);write_json_atomic(p,plan)
    with pytest.raises(ContractError):d.load_plan(tmp_path,p,file_digest(p))

def test_existing_acceptance_cannot_consume_future_reserve(tmp_path,monkeypatch):
    p,_,records=setup(tmp_path,monkeypatch)
    records.extend({'instance_id':f'prior__{i}','class_id':C} for i in range(10,13))
    with pytest.raises(ContractError,match='future task capacity'):reserve(tmp_path,p)
    reserve(tmp_path,p,count=1)

def test_accepted_members_count_once_and_claim_after_commit(tmp_path,monkeypatch):
    p,_,records=setup(tmp_path,monkeypatch,capacity=2,future=2)
    ref=reserve(tmp_path,p)
    with pytest.raises(ContractError,match='committed eligible'):d.claim_capacity(tmp_path,ref,['member__50000'])
    records.append({'instance_id':'member__50000','class_id':C,'training_diversity_ticket':ref,'training_config_sha256':H})
    assert d.check_ticket(tmp_path,H,50000,2,ref)['count']==2
    claim=d.claim_capacity(tmp_path,ref,['member__50000'])
    assert claim['accepted_count']==1 and claim['unused_positions_still_reserved']==1
    assert d.claim_capacity(tmp_path,ref,['member__50000'])==claim
    records[0]['training_config_sha256']='b'*64
    with pytest.raises(ContractError,match='configuration'):d.claim_capacity(tmp_path,ref,['member__50000'])
    records[0]['training_config_sha256']=H;records[0]['extra']='changed'
    with pytest.raises(ContractError,match='claim bytes'):d.claim_capacity(tmp_path,ref,['member__50000'])

def test_claim_requires_exact_ticket_and_range(tmp_path,monkeypatch):
    p,_,records=setup(tmp_path,monkeypatch)
    ref=reserve(tmp_path,p)
    records.append({'instance_id':'outside__50009','class_id':C,'training_diversity_ticket':ref,'training_config_sha256':H})
    with pytest.raises(ContractError,match='range'):d.claim_capacity(tmp_path,ref,['outside__50009'])
    records[0]['instance_id']='member__50000';records[0].pop('training_diversity_ticket')
    with pytest.raises(ContractError,match='ticket proof'):d.claim_capacity(tmp_path,ref,['member__50000'])

def test_tamper_and_plan_revision_rejected(tmp_path,monkeypatch):
    p,plan,_=setup(tmp_path,monkeypatch)
    ref=reserve(tmp_path,p)
    plan['note']='new revision';write_json_atomic(p,plan)
    with pytest.raises(ContractError,match='plan changed'):d.check_ticket(tmp_path,H,50000,2,ref)
    with pytest.raises(ContractError,match='plan revision'):reserve(tmp_path,p,'r2',50002)
    from pathlib import Path
    Path(ref['path']).write_text('{}')
    with pytest.raises(ContractError,match='ticket bytes'):d.check_ticket(tmp_path,H,50000,2,ref)

def test_orphan_ticket_recovery_is_exact_and_does_not_issue_twice(tmp_path,monkeypatch):
    p,_,_=setup(tmp_path,monkeypatch)
    ref=reserve(tmp_path,p)
    (tmp_path/'controllers/training_diversity/state.json').unlink()
    assert reserve(tmp_path,p)==ref
    (tmp_path/'controllers/training_diversity/state.json').unlink()
    with pytest.raises(ContractError,match='unregistered ticket'):reserve(tmp_path,p,count=3)

def test_external_plan_rejected(tmp_path,monkeypatch):
    p,_,_=setup(tmp_path,monkeypatch)
    root=tmp_path/'other';root.mkdir()
    with pytest.raises(ContractError,match='outside campaign'):d.load_plan(root,p,file_digest(p))
