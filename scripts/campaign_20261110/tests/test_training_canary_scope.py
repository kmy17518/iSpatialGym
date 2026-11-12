import copy
import pytest
from scripts.campaign_20261110.contracts import ContractError, digest
from scripts.campaign_20261110.training_canary_scope import validate_scope


def test_future_production_config_unchanged_under_exact_zero_credit_authorization():
    config={'purpose':'training_production','split':'train','template':{'split':'train'}};before=digest(config)
    auth={'purpose':'training_pipeline_canary','validation_scope':'exact_production_configuration',
          'production_released':False,'benchmark_credit':0,'count':1,'max_instances':1}
    assert validate_scope(config,auth)['benchmark_credit']==0 and digest(config)==before
    for key,value in [('validation_scope','bulk'),('production_released',True),('benchmark_credit',1),('count',25),('max_instances',25)]:
        wrong={**auth,key:value}
        with pytest.raises(ContractError):validate_scope(config,wrong)
    with pytest.raises(ContractError):validate_scope(config,{'purpose':'training_pipeline_canary'})


def test_existing_canary_scope_and_foreign_split():
    config={'purpose':'training_pipeline_canary','split':'train','template':{'split':'train'}}
    assert not validate_scope(config,{'purpose':'training_pipeline_canary'})['production_released']
    config['template']['split']='eval_standard'
    with pytest.raises(ContractError):validate_scope(config,{'purpose':'training_pipeline_canary'})


def test_closed_bulk_gate_allows_only_exact_production_config_canary(tmp_path,monkeypatch):
    from datetime import datetime,timedelta,timezone
    from scripts.campaign_20261110 import production_gate as gate
    from scripts.campaign_20261110.contracts import file_digest,write_json_atomic
    config=tmp_path/'config.json';value={'purpose':'training_production','split':'train',
        'template':{'split':'train','task':'object_scaling','class_id':'object_scaling-apple'}}
    write_json_atomic(config,value);before=config.read_bytes()
    auth=tmp_path/'authorization.json';proof={'purpose':'training_pipeline_canary',
        'validation_scope':'exact_production_configuration','production_released':False,'benchmark_credit':0,
        'start':50000,'count':1,'max_instances':1,'config_sha256':file_digest(config)}
    write_json_atomic(auth,proof)
    entry={**proof,'expires_at':(datetime.now(timezone.utc)+timedelta(hours=1)).isoformat(),
        'authorization_path':str(auth),'authorization_sha256':file_digest(auth)}
    write_json_atomic(tmp_path/'controllers/training_production_gate.json',{'schema':'isg-training-production-gate-v1',
        'campaign_id':'campaign_20261110','mode':'blocked','standalone_training_allowed':False,'allowed_canaries':[entry]})
    monkeypatch.setattr(gate,'canonical_training_ready',lambda root:False)
    monkeypatch.setattr(gate,'heldout_parallel_ready',lambda *a,**kw:{'actual_test_fixture':True})
    env={'ISPATIALGYM_CAMPAIGN_CONFIG':str(config)}
    result=gate.check_environment(tmp_path,env,start=50000,count=1)
    assert result['allowed'] and result['category']=='validation_canary'
    assert not gate.check_environment(tmp_path,env,start=50001,count=1)['allowed']
    assert config.read_bytes()==before and not(tmp_path/'dataset').exists()


def test_oracle_production_scope_requires_exact_job_runtime_and_actual_index(tmp_path,monkeypatch):
    import copy,json
    from scripts.campaign_20261110.training_canary_scope import restoration_execution
    from scripts.campaign_20261110.contracts import file_digest,write_json_atomic,ContractError
    native=tmp_path/'code/native';native.mkdir(parents=True);(native/'.snapshot_commit').write_text('native')
    key={'code_sha256':file_digest(native/'.snapshot_commit')}
    meta={'runtime':{'purpose':'training_production'},'validation_key':key,'scene_setup':{'restoration_options':{'pristine_mass_before_reset_v1':True}},'perturbations':{}}
    config=tmp_path/'config.json';write_json_atomic(config,{'purpose':'training_production','split':'train',
        'template':{'split':'train','scene_setup':meta['scene_setup']},'validation_key':key,'perturbations':{}})
    auth=tmp_path/'auth.json';write_json_atomic(auth,{'purpose':'training_pipeline_canary','validation_scope':'exact_production_configuration',
        'production_released':False,'benchmark_credit':0,'count':1,'max_instances':1})
    job=tmp_path/'out/exact';tro=job/'inst/template/instances/exact__50000-tro_state.json'
    raw={'ispatialgym':{'campaign':meta,'instance_id':'exact__50000'}};write_json_atomic(tro,raw)
    (job/'job.env').write_text(f'ISPATIALGYM_CAMPAIGN_CONFIG={config}\nSTART=50000\nN=1\n')
    decision={'allowed':True,'category':'validation_canary','config_sha256':file_digest(config),
        'authorization':{'authorization_path':str(auth),'authorization_sha256':file_digest(auth)}}
    calls=[]
    def gate(root,source,**kwargs):calls.append((root,source,kwargs));return decision
    monkeypatch.setattr('scripts.campaign_20261110.production_gate.check_job',gate)
    proof=restoration_execution(meta,tro,native,environment={'ISGF_ROOT':str(tmp_path)})
    assert proof['instance_id']=='exact__50000' and proof['category']=='validation_canary'
    assert calls==[(tmp_path,job,{'kind':'source_validation'})]
    raw['ispatialgym']['instance_id']='exact__50001';write_json_atomic(tro,raw)
    with pytest.raises(ContractError,match='outside authorized job range'):
        restoration_execution(meta,tro,native,environment={'ISGF_ROOT':str(tmp_path)})
    raw['ispatialgym']['instance_id']='exact__50000';write_json_atomic(tro,raw)
    decision['allowed']=False
    with pytest.raises(ContractError,match='current exact bounded'):
        restoration_execution(meta,tro,native,environment={'ISGF_ROOT':str(tmp_path)})
    decision['allowed']=True;altered=copy.deepcopy(meta);altered['scene_setup']['restoration_options']['pristine_mass_before_reset_v1']=False
    with pytest.raises(ContractError,match='physical input differs'):
        restoration_execution(altered,tro,native,environment={'ISGF_ROOT':str(tmp_path)})
    (native/'.snapshot_commit').write_text('another native')
    with pytest.raises(ContractError,match='native runtime differs'):
        restoration_execution(meta,tro,native,environment={'ISGF_ROOT':str(tmp_path)})
    assert restoration_execution({},'/missing','/missing',environment={})is None


def test_production_observational_runtime_must_bind_exact_input_and_unchanged_code(tmp_path,monkeypatch):
    import json
    from scripts.campaign_20261110.training_canary_scope import restoration_execution
    from scripts.campaign_20261110.contracts import file_digest,write_json_atomic
    native=tmp_path/'code/observed';native.mkdir(parents=True);(native/'.snapshot_commit').write_text('observed')
    meta={'runtime':{'purpose':'training_production'},'validation_key':{'code_sha256':'original'},'scene_setup':{},'perturbations':{}}
    config=tmp_path/'config.json';write_json_atomic(config,{'purpose':'training_production','validation_key':meta['validation_key'],
        'template':{'scene_setup':{}},'perturbations':{}})
    tro=tmp_path/'out/job/inst/template/instances/exact__50000-tro_state.json'
    write_json_atomic(tro,{'ispatialgym':{'campaign':meta,'instance_id':'exact__50000'}})
    (tmp_path/'out/job/job.env').write_text(f'ISPATIALGYM_CAMPAIGN_CONFIG={config}\nSTART=50000\nN=1\n')
    decision={'allowed':True,'category':'released_training','config_sha256':file_digest(config)}
    monkeypatch.setattr('scripts.campaign_20261110.production_gate.check_job',lambda *a,**k:decision)
    req={'probe_runtime_sha256':file_digest(native/'.snapshot_commit')}
    records=[{'instance_id':'exact__50000','artifacts':{'tro':{'sha256':file_digest(tro)}}}]
    path=tmp_path/'request.json';write_json_atomic(path,req)
    env={'ISGF_ROOT':str(tmp_path),'ISPATIALGYM_TRAINING_OBSERVATION_REQUEST':str(path),
         'ISPATIALGYM_TRAINING_OBSERVATION_REQUEST_SHA256':file_digest(path)}
    monkeypatch.setattr('scripts.campaign_20261110.training_batch_probe.inputs',lambda *a:(req,records))
    proofs=[]
    def verify(*args):proofs.append(args);return {'verified':'exact narrow native runtime'}
    monkeypatch.setattr('scripts.campaign_20261110.training_batch_probe.verify_runtime_compatibility',verify)
    result=restoration_execution(meta,tro,native,environment=env)
    assert len(proofs)==1 and result['observational_runtime_compatibility']=={'verified':'exact narrow native runtime'}
    records[0]['artifacts']['tro']['sha256']='changed'
    with pytest.raises(ContractError,match='exact input differs'):
        restoration_execution(meta,tro,native,environment=env)
    records[0]['artifacts']['tro']['sha256']=file_digest(tro);req['physical_candidate']={'path':'overridden'}
    with pytest.raises(ContractError,match='exact input differs'):
        restoration_execution(meta,tro,native,environment=env)
