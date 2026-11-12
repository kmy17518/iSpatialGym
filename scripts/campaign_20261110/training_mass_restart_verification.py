"""Targeted fresh-B/ABA diagnosis; cannot certify the full cleanup matrix."""
from pathlib import Path
from .contracts import require, file_digest, digest
from .training_restoration import checked, validated_canary, same_loaded_state
from .training_batch_probe import inputs, verify_runtime_compatibility
from .training_release import verify_probe_execution, _observation
from .acceptance import report_passed
from .training_sequential_augmentation import one_world, compare_physics_reference, verify_original_cold, observed_augmentation_change


def verify(root, reference, started_at):
    value=checked(root,reference)
    require(value.get('schema')=='isg-sequential-mass-restart-verification-input-v1' and
            set(value['controls'])=={'fresh_B','ABA'} and set(value['original_cold'])=={'fresh_A','fresh_B'},
            'targeted mass-ordering scope differs')
    sources={}; references=None; runtime=None; cases={}; proofs={}; processes=set(); common=None; compatibility=None
    for name,control,order in [('fresh_B','FRESH_B',[1]),('ABA',None,[0,1,0])]:
        item=value['controls'][name];probe=checked(root,item['probe'])
        execution=verify_probe_execution(root,item['probe'],item['execution'])
        checked(root,probe['request']);request,records=inputs(root,probe['request']['path'])
        require(request.get('sequential_mass_restart') is True and request.get('sequential_anchor_rebase') is True and
                request.get('sequential_augmentation') is True and request.get('diagnostic_control')==control and
                request.get('production_released') is False and request.get('benchmark_credit')==0 and
                probe['kind']=='physics_action_open_loop' and len(probe['entries'])==len(order),
                'targeted request/control/episode identity differs')
        references=references or request['canaries'];runtime=runtime or probe['probe_runtime']
        require(request['canaries']==references and probe['probe_runtime']==runtime and
                file_digest(runtime['path'])==runtime['sha256']==request['probe_runtime_sha256'],
                'targeted sources or executed runtime differs')
        identity=digest({k:v for k,v in request.items() if k not in ('diagnostic_control','diagnostic_only')})
        if common is None:
            compatibility=verify_runtime_compatibility(root,request,records,Path(runtime['path']).parent);common=identity
        require(common==identity and compatibility==probe['runtime_compatibility'],'targeted runtime proof differs')
        case=[]; observations=[]
        for ordinal,(entry,member) in enumerate(zip(probe['entries'],order)):
            require(entry['canary_review']==references[member] and entry['returncode']==0,'targeted member or execution differs')
            if member not in sources:sources[member]=validated_canary(root,references[member],started_at)
            source=sources[member];exported=source['export_record'];report=checked(root,entry['report']);report_passed(report,'replay')
            require(report['instance_id']==source['instance_id'] and Path(report['replay']).resolve()==Path(exported['artifacts']['demo']['path']).resolve() and
                    bool(report['loader'].get('reused_world'))==(ordinal>0),'targeted native source or world reuse differs')
            restoration=report['loader']['campaign_restoration']
            boundaries=[_observation(report,k) for k in ('native_reset_boundary_observation','native_reset_step_completed_observation','native_final_pre_action_observation')]
            step=restoration['next_native_control_step_observation'];observation=boundaries[-1]
            require(restoration['scale_verified'] is True and observation.get('scheduler_job_id') and
                    observation.get('simulator_module','').startswith('omnigibson.') and
                    observation['validation_key']==exported['validation_key'] and
                    observation['perturbations_sha256']==digest(exported['perturbations']) and
                    step['native_control_calls_observed']==1 and step['physics_steps_added']==0,
                    'targeted native state binding differs')
            for endpoint in ('before','after'):
                require(step[endpoint]['loaded_state_sha256']==digest(step[endpoint]['loaded_state']),'targeted first action state changed')
            same_loaded_state(observation['loaded_state'],step['before']['loaded_state'])
            case.append({'member':member,'boundaries':boundaries,'step':step,'report':entry['report']});observations.append(observation)
        process=one_world(probe,observations);require(process not in processes,'targeted cold reference shares process');processes.add(process)
        cases[name]=case;proofs[name]={'probe':item['probe'],'execution':execution,'reports':[e['report'] for e in case],'world_load_count':1,'process_session':process}
    original={}
    for name,member,current in [('fresh_A',0,cases['ABA'][0]),('fresh_B',1,cases['fresh_B'][0])]:
        original[name]=verify_original_cold(root,value['original_cold'][name],references[member],current,started_at)
    compare_physics_reference(cases['fresh_B'][0],cases['ABA'][1])
    compare_physics_reference(cases['ABA'][0],cases['ABA'][2])
    changes=observed_augmentation_change(root,[sources[i]['export_record']for i in (0,1)],
        cases['ABA'][0]['boundaries'][-1]['loaded_state'],cases['fresh_B'][0]['boundaries'][-1]['loaded_state'])
    return {'schema':'isg-sequential-mass-restart-verification-v1','input':reference,'status':'verified',
        'controls':proofs,'runtime':runtime,'runtime_compatibility':compatibility,'sources':references,
        'original_cold_compatibility':original,'observed_augmentation_changes':changes,
        'full_matrix_verified':False,'cleanup_controls_verified':False,'production_released':False,'benchmark_credit':0,
        'next_action':'Targeted physics only; complete prospective same-runtime success/failure/interruption controls before full shared verification.'}
