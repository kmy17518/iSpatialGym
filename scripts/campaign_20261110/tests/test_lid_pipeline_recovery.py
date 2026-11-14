from dataclasses import asdict, replace
import json
import pytest
from scripts.campaign_20261110 import lid_pipeline_recovery as L, production_gate as G
from scripts.campaign_20261110.contracts import ContractError, ObjectRef, Template, file_digest


def fixture(tmp_path, monkeypatch):
    objects = tuple(ObjectRef(instance, role, category, model, 'a'*64, (.6499999761581421,)*3, ready=True, readiness_source='fixture')
                    for instance,role,category,model in [('lid.n.02_1','target','lid','pbmsji'),
                                                       ('stockpot.n.01_1','partner','stockpot','oshwps')])
    source = Template('canonical','configuration_matching-attachment',
                      'configuration_matching-attachment_04','eval_standard','office_cubicles_right',
                      'private_office_2',objects,'b'*64)
    p=tmp_path/'canonical.json';p.write_text(json.dumps({'configuration':{'template':asdict(source)}}))
    scope={'schema':'isg-exact-canonical-lid-pipeline-recovery-v1','production_released':False,
           'benchmark_credit':0,'start':50313000,'count':1,
           'canonical_configuration':{'path':str(p),'sha256':file_digest(p)}}
    template=replace(source,template_id='pipeline',split='train',scene_setup={
        'bounded_lid_pipeline_recovery':scope,'source_alias':'configuration_matching-attachment-lid-stockpot-desk',
        'recovery_options':{'lid_vertical_final_retreat':True}})
    q={'purpose':'training_production','split':'train','template':asdict(template),
       'perturbations':{'scales':{o.instance:{'mode':'fixed','applied':list(o.default_scale),
                                           'default':list(o.default_scale)}for o in objects}}}
    path=tmp_path/'config.json'
    def save():path.write_text(json.dumps(q))
    save();env={'ISGF_ROOT':str(tmp_path),'ISPATIALGYM_CAMPAIGN_CONFIG':str(path)};calls=[]
    def gate(root, environment, **kwargs):
        calls.append(kwargs)
        return {'allowed':True,'category':'validation_canary','config_sha256':file_digest(path),
                'authorization':{'purpose':'training_pipeline_canary'}}
    monkeypatch.setattr(G,'check_environment',gate)
    return template,q,env,calls,save


def test_exact_canonical_float32_pair_requires_current_one_source_authorization(tmp_path,monkeypatch):
    t,q,env,calls,_=fixture(tmp_path,monkeypatch)
    assert L.authorized(t,environment=env)
    assert calls==[{'kind':'source_validation','start':50313000,'count':1}]


@pytest.mark.parametrize('fault',['no_scope','source_hash','source_models','variant','scale','count','credit','release','denied','unbounded'])
def test_unapproved_or_changed_inputs_cannot_use_recovery(tmp_path,monkeypatch,fault):
    t,q,env,calls,save=fixture(tmp_path,monkeypatch)
    scope=t.scene_setup['bounded_lid_pipeline_recovery']
    if fault=='no_scope':
        t=replace(t,scene_setup={});assert L.authorized(t,environment=env) is False;return
    if fault=='source_hash':scope['canonical_configuration']['sha256']='f'*64
    if fault=='source_models':
        t=replace(t,objects=(replace(t.objects[0],model='wrong'),t.objects[1]))
    if fault=='variant':t.scene_setup['source_alias']='other'
    if fault=='scale':q['perturbations']['scales']['lid.n.02_1']['applied']=[.64]*3
    if fault=='count':scope['count']=2
    if fault=='credit':scope['benchmark_credit']=1
    if fault=='release':scope['production_released']=True
    if fault in ('denied','unbounded'):
        monkeypatch.setattr(G,'check_environment',lambda *a,**k:{'allowed':fault=='unbounded','category':'released_training'})
    q['template']=asdict(t);save()
    with pytest.raises(ContractError):L.authorized(t,environment=env)
