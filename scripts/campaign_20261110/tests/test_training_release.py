import copy
import json
from pathlib import Path
import pytest
from scripts.campaign_20261110 import training_release as r
from scripts.campaign_20261110.contracts import ContractError, file_digest, write_json_atomic


def production_inputs(root):
    config=root/'config.json';write_json_atomic(config,{'purpose':'training_production'})
    release={'path':str(root/'release.json'),'sha256':'a'*64}
    env={'ISPATIALGYM_TRAINING_RELEASE':release['path'],'ISPATIALGYM_TRAINING_RELEASE_SHA256':release['sha256'],
         'ISPATIALGYM_TRAINING_DIVERSITY_TICKET':str(root/'ticket.json'),
         'ISPATIALGYM_TRAINING_DIVERSITY_TICKET_SHA256':'b'*64}
    policy={'mode':'scoped_releases','standalone_training_allowed':True,
            'configuration_releases':[{'config_sha256':file_digest(config),'release':release}]}
    return config,release,env,policy


def test_no_flag_or_release_registry_bypasses_exact100(tmp_path,monkeypatch):
    config,release,env,policy=production_inputs(tmp_path)
    monkeypatch.setattr('scripts.campaign_20261110.production_gate.canonical_training_ready',lambda root:False)
    monkeypatch.setattr(r,'verify_configuration',lambda *a:pytest.fail('must stop before scoped release'))
    with pytest.raises(ContractError,match='all100'):
        r.check_production(tmp_path,config,env,policy,start=50000,count=3)


def test_missing_exact_registry_and_changed_ticket_are_rejected(tmp_path,monkeypatch):
    config,release,env,policy=production_inputs(tmp_path)
    monkeypatch.setattr('scripts.campaign_20261110.production_gate.canonical_training_ready',lambda root:True)
    proof={'release':release,'task':'task','class_id':'class'}
    monkeypatch.setattr(r,'verify_configuration',lambda *a:copy.deepcopy(proof))
    monkeypatch.setattr(r,'verify_diversity_cell',lambda *a:{})
    def ticket(root,config_sha,start,count,reference):
        assert config_sha==file_digest(config) and (start,count)==(50000,3)
        if reference['sha256']!='b'*64:raise ContractError('ticket changed')
        return {'task':'task','class_id':'class'}
    monkeypatch.setattr('scripts.campaign_20261110.training_diversity.check_ticket',ticket)
    assert r.check_production(tmp_path,config,env,policy,start=50000,count=3)['production_authorization']
    for entries in ([],[policy['configuration_releases'][0]]*2):
        with pytest.raises(ContractError,match='registered once'):
            r.check_production(tmp_path,config,env,{**policy,'configuration_releases':entries},start=50000,count=3)
    with pytest.raises(ContractError,match='ticket changed'):
        r.check_production(tmp_path,config,{**env,'ISPATIALGYM_TRAINING_DIVERSITY_TICKET_SHA256':'c'*64},policy,start=50000,count=3)


def test_original_canary_range_is_still_zero_credit_after_policy_removal(tmp_path):
    path=tmp_path/'training_pipeline_canaries/old/requests.json'
    write_json_atomic(path,{'campaign_id':'campaign_20261110','records':[{'start':50000,'count':2}]})
    before=path.read_bytes()
    with pytest.raises(ContractError,match='zero-credit'):
        r.assert_production_range(tmp_path,49999,3,{})
    r.assert_production_range(tmp_path,50002,3,{})
    assert path.read_bytes()==before


def test_empty_or_partial_restoration_suite_cannot_release(tmp_path):
    for refs in ({},{'ABA':{}},{'fresh_A':{},'successful_AA':{},'ABA':{}}):
        with pytest.raises(ContractError,match='all four'):
            r.verify_controls(tmp_path,refs,1794321600)


def test_exact_config_and_applicable_family_not_inferred_from_shared_label(tmp_path):
    config=tmp_path/'config.json'
    write_json_atomic(config,{'purpose':'training_pipeline_canary','split':'train','template':{'split':'train'}})
    release=tmp_path/'release.json';write_json_atomic(release,{'schema':'isg-training-configuration-release-v2',
        'campaign_id':'campaign_20261110','config_sha256':file_digest(config)})
    with pytest.raises(ContractError,match='exact production input'):
        r.verify_configuration(tmp_path,config,{'path':str(release),'sha256':file_digest(release)})
    release.write_text('{}')
    with pytest.raises(ContractError,match='missing or changed'):
        r.verify_configuration(tmp_path,config,{'path':str(release),'sha256':'0'*64})


def test_historical_standalone_stays_excluded_and_forged_proof_unverified(tmp_path):
    from scripts.campaign_20261110.training_eligibility import classify_record
    row={'split':'train','sampling_split':'train','instance_id':'old__50000'}
    assert classify_record(row,root=tmp_path)['status']=='excluded'
    row['training_production_receipt']={'path':str(tmp_path/'absent'),'sha256':'0'*64}
    assert classify_record(row,root=tmp_path)['status']=='unverified'


def test_group_commit_retains_production_identity_before_capacity_journal(tmp_path,monkeypatch):
    from scripts.campaign_20261110.tests.test_group_lane import candidate,claim
    from scripts.campaign_20261110 import training_diversity as d
    import sqlite3
    f,review=candidate(tmp_path,20,split='train')
    proof={'training_config_path':'exact','training_config_sha256':'c'*64,
           'training_release':{'path':'release','sha256':'r'*64},
           'training_diversity_ticket':{'path':'ticket','sha256':'t'*64},
           'training_production_receipt':{'path':'receipt','sha256':'p'*64}}
    monkeypatch.setattr(r,'acceptance_proof',lambda root,review:proof)
    ticket=tmp_path/'ticket.json';write_json_atomic(ticket,{'start':50000,'count':3})
    proof['training_diversity_ticket']['path']=str(ticket)
    monkeypatch.setattr(d,'check_ticket',lambda *args:{})
    observed=[]
    def capacity(root,ticket,ids):
        with sqlite3.connect(root/'dataset/quotas.sqlite3')as db:
            rows=[json.loads(x[0])for x in db.execute('SELECT record FROM members')]
        assert set(ids)=={x['instance_id']for x in rows}
        assert all(all(x[k]==v for k,v in proof.items())for x in rows)
        observed.append(ids)
    monkeypatch.setattr(d,'claim_capacity',capacity)
    result=claim(tmp_path,f,review)
    assert result['accepted'] and set(result['accepted'].values())=={'train'}
    assert observed==[list(result['accepted'])]


def test_stage_specific_code_cannot_borrow_pool_runtime_for_restoration(tmp_path):
    native=tmp_path/'code/production';native.mkdir(parents=True)
    witness=tmp_path/'code/pool';witness.mkdir()
    (witness/'.snapshot_commit').write_text('pool-runtime')
    evidence={stage:{'path':stage,'sha256':'a'*64} for stage in r.STAGE_COMPONENTS}
    scope={'stages':{}};observed={}
    for stage,paths in r.STAGE_COMPONENTS.items():
        files=[]
        for relative in paths:
            for folder in(native,witness):
                file=folder/relative;file.parent.mkdir(parents=True,exist_ok=True);file.write_text('same implementation')
            file=witness/relative
            files.append({'relative_path':relative,'validated_implementation':{'path':str(file),'sha256':file_digest(file)}})
        scope['stages'][stage]={'evidence':evidence[stage],'component_files':files}
        observed[stage]={'evidence':evidence[stage],'runtime_hashes':{file_digest(witness/'.snapshot_commit')}}
    scope['stages']['restoration']['effective_options']={'pristine_mass_before_reset_v1':True}
    observed['restoration']['effective_options']=[{'pristine_mass_before_reset_v1':True}]
    r.verify_stage_scope(tmp_path,native,scope,observed,{'pristine_mass_before_reset_v1':True})
    observed['restoration']['runtime_hashes']={'b'*64}
    with pytest.raises(ContractError,match='did not execute this stage'):
        r.verify_stage_scope(tmp_path,native,scope,observed,{'pristine_mass_before_reset_v1':True})
    observed['restoration']['runtime_hashes']={file_digest(witness/'.snapshot_commit')}
    with pytest.raises(ContractError,match='options differ'):
        r.verify_stage_scope(tmp_path,native,scope,observed,{})


def test_limited_shared_pool_cannot_clear_no_pool_exact_configuration(tmp_path):
    own={'source_proof':{'config':{'sha256':'a'*64}}};exact={'path':'exact','sha256':'b'*64}
    execution=tmp_path/'unit_pool.json';write_json_atomic(execution,{'canaries':[{'path':'unit','sha256':'c'*64}]})
    scope=tmp_path/'applicable.json';write_json_atomic(scope,{'schema':'isg-training-applicable-pool-v1',
        'config_sha256':'a'*64,'execution':{'path':str(execution),'sha256':file_digest(execution)}})
    with pytest.raises(ContractError,match='did not execute the applicable pool'):
        r.verify_applicable_pool(tmp_path,{'path':str(scope),'sha256':file_digest(scope)},own,exact,1794321600)


def test_diversity_cell_checks_actual_model_scale_and_support(tmp_path):
    obj={'instance':'target','role':'target','category':'apple','model':'ready','asset_sha256':'a'*64,'default_scale':[1,1,1]}
    support={'instance':'table','role':'support','category':'table','model':'ready','asset_sha256':'b'*64,'default_scale':[1,1,1]}
    template={'class_id':'object_scaling-apple','scene':'house','room':'kitchen','objects':[obj,support],
              'scene_setup':{'measured_height_m':.8}}
    config=tmp_path/'config.json';value={'template':template,'perturbations':{'scales':{'target':{'applied':[1.1]*3},'table':{'applied':[1]*3}}}}
    write_json_atomic(config,value)
    physical={'objects':[support,obj],'applied_scales':{'table':[1]*3,'target':[1.1]*3},'scene_setup_sha256':r.digest(template['scene_setup'])}
    plan=tmp_path/'plan.json';cell={'id':'cell','class_id':template['class_id'],'scene':'house','room':'kitchen',
        'support_id':'table','model_keys':['apple/ready','table/ready'],'physical_identity':physical,
        'configuration_sha256s':[file_digest(config)]};write_json_atomic(plan,{'cells':[cell]})
    ticket=tmp_path/'ticket.json';write_json_atomic(ticket,{'config_sha256':file_digest(config),'cell_id':'cell',
        'plan':{'path':str(plan),'sha256':file_digest(plan)}})
    assert r.verify_diversity_cell(tmp_path,config,{'path':str(ticket),'sha256':file_digest(ticket)})['cell_id']=='cell'
    cell['physical_identity']['applied_scales']['target']=[1]*3;write_json_atomic(plan,{'cells':[cell]})
    write_json_atomic(ticket,{'config_sha256':file_digest(config),'cell_id':'cell','plan':{'path':str(plan),'sha256':file_digest(plan)}})
    with pytest.raises(ContractError,match='model/scale/measured support'):
        r.verify_diversity_cell(tmp_path,config,{'path':str(ticket),'sha256':file_digest(ticket)})


def reset_observation(session='cold'):
    from scripts.campaign_20261110.tests.test_training_restoration import state
    loaded=state();context={'physics_steps_added':0,'dynamic_roots':{'o':{'mass':.3876259,
        'sleep_threshold':.005,'solver_position_iteration_count':16,'solver_velocity_iteration_count':1}}}
    return {'loaded_state':loaded,'loaded_state_sha256':r.digest(loaded),
        'native_dynamics_context':context,'native_dynamics_context_sha256':r.digest(context),
        'scheduler_job_id':'123','process_session':session,'simulator_module':'omnigibson.simulator'}


def test_equal_final_pose_cannot_hide_changed_reset_mass_or_orientation():
    a=reset_observation();r.same_reset_boundary(a,copy.deepcopy(a))
    b=copy.deepcopy(a);b['native_dynamics_context']['dynamic_roots']['o']['mass']=2.
    b['native_dynamics_context_sha256']=r.digest(b['native_dynamics_context'])
    with pytest.raises(ContractError,match='mass differs'):r.same_reset_boundary(a,b)
    b=copy.deepcopy(a);b['loaded_state']['objects']['o']['orientation']=[0,0,.0002,1]
    with pytest.raises(ContractError,match='orientation contaminated'):r.same_reset_boundary(a,b)
    b=copy.deepcopy(a);b.pop('native_dynamics_context')
    with pytest.raises(ContractError,match='context missing'):r.same_reset_boundary(a,b)


def test_matched_cold_baseline_checks_input_identity_and_unchanged_first_reset(tmp_path,monkeypatch):
    def save(path,value):
        write_json_atomic(path,value);return {'path':str(path),'sha256':file_digest(path)}
    demo=tmp_path/'demo.pkl';demo.write_bytes(b'unchanged actions')
    exported={'instance_id':'exact-A','artifacts':{'demo':{'path':str(demo),'sha256':file_digest(demo)}}}
    review={'path':str(tmp_path/'review.json'),'sha256':'a'*64}
    own={'instance_id':'exact-A','task':'disembedding','export_record':exported}
    monkeypatch.setattr(r,'validated_canary',lambda *a:copy.deepcopy(own))
    monkeypatch.setattr('scripts.campaign_20261110.training_batch_probe.inputs',
        lambda root,path:(json.loads(Path(path).read_text()),[exported]))
    monkeypatch.setattr('scripts.campaign_20261110.training_batch_probe.verify_runtime_compatibility',lambda *a:{'checked':True})
    probes={};reports={}
    native=tmp_path/'native'
    for relative in ('ispatialgym/oracle/run_isg.py','ispatialgym/oracle/campaign_restore.py'):
        path=native/relative;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('exact native implementation')
    identity=native/'.snapshot_commit';identity.write_text('candidate')
    for name in ('original','candidate'):
        observation=reset_observation(name)
        restoration={'scale_verified':True,**{key:{'observation':copy.deepcopy(observation)}for key in (
            'native_reset_boundary_observation','native_reset_step_completed_observation','native_final_pre_action_observation')},
            'next_native_control_step_observation':{'before':copy.deepcopy(observation),'after':copy.deepcopy(observation),
                'physics_steps_added':0,'native_control_calls_observed':1}}
        report={'mode':'replay','status':'success','success':True,'qa':{'enabled':True},'instance_id':'exact-A',
            'replay':str(demo),'loader':{'campaign_restoration':restoration}}
        request={'canaries':[review],'probe_runtime_sha256':file_digest(identity)}
        if name=='candidate':
            request.update(physical_candidate={'path':'exact-physical-proof'},diagnostic_control='FRESH_A')
            report['loader']['pre_reset_pristine_mass']={'revision':'pristine_mass_before_reset_v1',
                'captured_at':'build_env_before_first_restore','actors_checked':2,'actors_restored':[],
                'physics_steps_added':0,'later_serialized_mass_application_unchanged':True}
        report_ref=save(tmp_path/(name+'_report.json'),report);reports[name]=report
        probes[name]={'kind':'physics_action_open_loop','request':save(tmp_path/(name+'_request.json'),request),
            'probe_runtime':{'path':str(identity),'sha256':file_digest(identity)},'runtime_compatibility':{'checked':True},
            'entries':[{'canary_review':review,'returncode':0,'report':report_ref}]}
    def refresh():
        proof={'schema':'isg-training-cold-reset-compatibility-v1'}
        for name,probe in probes.items():
            probe['entries'][0]['report']=save(tmp_path/(name+'_report.json'),reports[name])
            proof[name]=save(tmp_path/(name+'_probe.json'),probe)
        return save(tmp_path/'cold.json',proof)
    assert r.verify_cold_reset_compatibility(tmp_path,refresh(),0,native,'disembedding')['status']=='matched_original_cold_reset_verified'
    changed=reports['candidate']['loader']['campaign_restoration']['native_reset_step_completed_observation']['observation']
    changed['loaded_state']['objects']['o']['orientation']=[0,0,.0002,1]
    changed['loaded_state_sha256']=r.digest(changed['loaded_state'])
    with pytest.raises(ContractError,match='orientation contaminated'):
        r.verify_cold_reset_compatibility(tmp_path,refresh(),0,native,'disembedding')
    changed['loaded_state']=reset_observation()['loaded_state'];changed['loaded_state_sha256']=r.digest(changed['loaded_state'])
    reports['candidate']['loader']['pre_reset_pristine_mass']['actors_restored']=['tray']
    with pytest.raises(ContractError,match='fresh candidate changed'):
        r.verify_cold_reset_compatibility(tmp_path,refresh(),0,native,'disembedding')


def test_diversity_cell_names_complete_multi_support_set(tmp_path):
    objects=[{'instance':name,'role':'support','category':name,'model':'ready','asset_sha256':'a'*64,
              'default_scale':[1,1,1]}for name in ('floor','table','tray_destination','tray_source')]
    config=tmp_path/'config.json';value={'template':{'objects':objects,'class_id':'disembedding_07','scene':'office',
        'room':'meeting','scene_setup':{'exact_measured_height':.8}},'perturbations':{'scales':{o['instance']:{'applied':[1,1,1]}for o in objects}}}
    write_json_atomic(config,value);names=sorted(o['instance']for o in objects)
    cell={'id':'cell','class_id':'disembedding_07','scene':'office','room':'meeting','support_id':'support_set:'+r.digest(names),
        'support_instances':names,'model_keys':sorted(o['category']+'/ready'for o in objects),
        'configuration_sha256s':[file_digest(config)],'physical_identity':{'objects':objects,
            'applied_scales':{n:[1,1,1]for n in names},'scene_setup_sha256':r.digest(value['template']['scene_setup'])}}
    def reference():
        plan=tmp_path/'plan.json';write_json_atomic(plan,{'cells':[cell]})
        ticket=tmp_path/'ticket.json';write_json_atomic(ticket,{'config_sha256':file_digest(config),'cell_id':'cell',
            'plan':{'path':str(plan),'sha256':file_digest(plan)}})
        return {'path':str(ticket),'sha256':file_digest(ticket)}
    assert r.verify_diversity_cell(tmp_path,config,reference())['cell_id']=='cell'
    cell['support_instances']=names[:-1]
    with pytest.raises(ContractError,match='named support set'):r.verify_diversity_cell(tmp_path,config,reference())
