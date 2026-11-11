from copy import deepcopy
import json
import pytest
from scripts.campaign_20261110.contracts import ContractError,file_digest
from scripts.campaign_20261110.training_restoration import same_loaded_state
from scripts.campaign_20261110.training_batch_probe import inputs


def state():
    return {'objects':{'o':{'category':'book','model':'abc','fixed_base':False,'scale':[1,1,1],
        'position':[1,2,3],'orientation':[0,0,0,1],'joint_positions':[.2],'joint_velocities':[0],
        'linear_velocity':[0,0,0],'angular_velocity':[0,0,0]}},'lights':{'lamp':1000},
        'skybox':{'intensity':2000,'color':[1,1,1]}}


@pytest.mark.parametrize('field,value', [('joint_positions',[.3]),('joint_velocities',[.1]),
    ('position',[1.1,2,3]),('scale',[.9,.9,.9]),('linear_velocity',[.1,0,0])])
def test_loaded_state_rejects_observed_contamination(field,value):
    a=state();b=deepcopy(a);b['objects']['o'][field]=value
    with pytest.raises(ContractError,match='contaminated'):same_loaded_state(a,b)
    same_loaded_state(a,deepcopy(a))


def test_probe_rejects_same_instance_and_configuration_mixing(tmp_path):
    refs=[]
    for i in range(2):
        artifact=tmp_path/f'demo{i}';artifact.write_bytes(b'demo')
        p=tmp_path/f'review{i}.json'
        p.write_text(json.dumps({'validation_passed':True,'accepted':False,'export_record':{
            'instance_id':str(i),'render_compatibility_sha256':'same',
            'artifacts':{'demo':{'path':str(artifact),'sha256':file_digest(artifact)}}}}))
        refs.append({'path':str(p),'sha256':file_digest(p)})
    value={'schema':'isg-training-batch-probe-request-v1','campaign_id':'campaign_20261110','canaries':refs}
    request=tmp_path/'request.json';request.write_text(json.dumps(value))
    assert len(inputs(tmp_path,request)[1])==2
    value['canaries']=[refs[0],refs[0]];request.write_text(json.dumps(value))
    with pytest.raises(ContractError,match='distinct canaries'):inputs(tmp_path,request)
    value['canaries']=refs;request.write_text(json.dumps(value))
    p=tmp_path/'review1.json';row=json.loads(p.read_text());row['export_record']['render_compatibility_sha256']='other'
    p.write_text(json.dumps(row));refs[1]['sha256']=file_digest(p);request.write_text(json.dumps(value))
    with pytest.raises(ContractError,match='configuration isolation'):inputs(tmp_path,request)
