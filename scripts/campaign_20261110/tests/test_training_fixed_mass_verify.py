import copy
import json
import numpy as np
import pytest
from scripts.campaign_20261110.contracts import digest, file_digest, ContractError
from scripts.campaign_20261110.training_fixed_mass_probe import DELTA, principal_inertia
from scripts.campaign_20261110.training_fixed_mass_verify import audit_persistence


def fixture(tmp_path):
    def put(name, data):
        p=tmp_path/name;p.write_text(json.dumps(data));return {'path':str(p),'sha256':file_digest(p)}
    def state(mass):
        matrix=np.diag([.001,.002,.003])*mass
        values,quat=principal_inertia(matrix)
        return dict(mass_kg=mass,inertia=matrix.reshape(1,9,order='F').tolist(),usd_mass=mass,
                    usd_diagonal_inertia=values,usd_principal_axes=quat)
    expected={k:state(2.)for k in ('trayA','trayB')}
    campaign={'source':'exact'}
    tro=put('tro.json',{'ispatialgym':{'campaign':campaign,'constraints':[
        {'type':'mass','object':k,'mass_kg':2.}for k in ('tray.n.01_1','tray.n.01_2')]}})
    compat={'path':'proof','sha256':'exact'}
    baseline=dict(physical_delta=DELTA,source_tro=tro,source_campaign_sha256=digest(campaign),
        required_companion=str(tmp_path/'fixed_mass_audit.json'),intended=expected,
        ordinary_scene_save_alone_is_sufficient=False)
    lifecycle=dict(native_calls=['stop','play'],observed_physics_step_calls=[],physics_steps_requested=0,
                   time_step_settings_before={'dt':.01},time_step_settings_after={'dt':.01})
    def event(stage,**kwargs):return dict(stage=stage,passed=True,native=copy.deepcopy(expected),**kwargs)
    events=[event('world_initialized_after_one_override',stock={k:state(.4)for k in expected},
                  initial_actor_reconstruction=lifecycle)]
    entries=[]
    for i,iid in enumerate(('A','B','A')):
        if i:events.extend([event('before_existing_stop_play'),event('after_existing_stop_play')])
        events.extend([event('before_episode_restore',instance_id=iid),event('after_episode_restore',instance_id=iid)])
        entries.append({'report':put('report'+str(i)+'.json',{'instance_id':iid})})
    audit=dict(schema='isg-fixed-tray-mass-inertia-audit-v1',physical_delta=DELTA,
               runtime_compatibility=compat,physics_steps_added=0,production_released=False,
               expected=expected,events=events)
    probe=put('probe.json',dict(entries=entries,runtime_compatibility=compat))
    def run(value=audit,late=None,*,separate=False,integrated=False):
        refs={'audit':put('fixed_mass_audit.json',value),'baseline':put('fixed_mass_baseline_config.json',baseline)}
        request={'fixed_mass_baseline':True,'runtime_compatibility':compat}
        if integrated: request['integrated_candidate_v1']=True
        extra={'separate_reconstruction_lifecycle':separate}
        if late is not None:
            request['late_incoming_physics_rebuild_v1']=True
            extra['additional_reconstruction_reference']=put('late_rebuild_audit.json',late)
        return audit_persistence(tmp_path,refs,probe,request,[{'artifacts':{'tro':tro}}],3,**extra)
    return audit,run


def test_exact_initial_and_reconstructed_baseline(tmp_path):
    audit,run=fixture(tmp_path)
    result=run()
    assert result['episodes']==3 and result['reused_actor_reconstructions']==2
    assert result['benchmark_credit']==0 and not result['production_released']


@pytest.mark.parametrize('mutation',[
    lambda a:a['events'][-1]['native']['trayA']['inertia'][0].__setitem__(0,.01),
    lambda a:a['events'][-1]['native']['trayA'].__setitem__('usd_mass',.4),
    lambda a:a['events'][-1].__setitem__('instance_id','wrong'),
    lambda a:a['events'].pop(),
    lambda a:a['events'][0]['initial_actor_reconstruction'].__setitem__('observed_physics_step_calls',['step']),
    lambda a:a.__setitem__('production_released',True),
    lambda a:a['events'][-1].__setitem__('passed',False),
])
def test_persistence_proof_rejects_missing_or_inconsistent_evidence(tmp_path,mutation):
    audit,run=fixture(tmp_path);mutation(audit)
    with pytest.raises(ContractError):run(audit)


def test_separately_declared_late_reconstructions_require_exact_native_evidence(tmp_path):
    from scripts.campaign_20261110.training_late_rebuild_probe import DELTA as LATE_DELTA
    audit,run=fixture(tmp_path);template=audit['events'][0]
    for _ in range(3):
        audit['events'] += [dict(stage=s,passed=True,native=copy.deepcopy(audit['expected']))
                            for s in ['before_existing_stop_play','after_existing_stop_play']]
    with pytest.raises(ContractError):run(audit)
    late={'physical_delta':LATE_DELTA,'production_released':False,'benchmark_credit':0,
          'explicit_sdk_reconstructions':3,'events':[]}
    for iid in ['A','B','A']:
        late['events'].append(dict(instance_id=iid,status='completed',static_body_properties_bit_identical=True,
            before_static_bodies={'all':'native'},after_static_bodies={'all':'native'},
            whole_boundary_physics_step_calls=[],reference_state_copied=False,
            sdk=dict(explicit_sdk_reconstructions=1,release_calls=1,force_load_calls=1,manual_attached_stage=0),
            lifecycle=copy.deepcopy(template['initial_actor_reconstruction'])))
    result=run(audit,late);assert result['additional_declared_reconstructions']==3
    late['events'][1]['whole_boundary_physics_step_calls']=['step']
    with pytest.raises(ContractError):run(audit,late)


def test_common_candidate_separate_lifecycle_still_requires_complete_native_proof(tmp_path):
    from scripts.campaign_20261110.training_late_rebuild_probe import DELTA as LATE_DELTA
    audit,run=fixture(tmp_path)
    late={'physical_delta':LATE_DELTA,'production_released':False,'benchmark_credit':0,
          'explicit_sdk_reconstructions':3,'events':[]}
    for iid in ['A','B','A']:
        late['events'].append(dict(instance_id=iid,status='completed',static_body_properties_bit_identical=True,
            before_static_bodies={'all':'native'},after_static_bodies={'all':'native'},
            whole_boundary_physics_step_calls=[],reference_state_copied=False,
            sdk=dict(explicit_sdk_reconstructions=1,release_calls=1,force_load_calls=1,manual_attached_stage=0),
            lifecycle=copy.deepcopy(audit['events'][0]['initial_actor_reconstruction'])))
    with pytest.raises(ContractError):run(audit,late,separate=True)
    with pytest.raises(ContractError):run(audit,None,separate=True,integrated=True)
    result=run(audit,late,separate=True,integrated=True)
    assert result['reused_actor_reconstructions']==2
    assert result['additional_declared_reconstructions']==3
    for mutate in [lambda v:v['events'].pop(),
                   lambda v:v['events'][1]['lifecycle'].__setitem__('native_calls',['play']),
                   lambda v:v['events'][1]['after_static_bodies'].__setitem__('all','changed')]:
        bad=copy.deepcopy(late);mutate(bad)
        with pytest.raises(ContractError):run(audit,bad,separate=True,integrated=True)
