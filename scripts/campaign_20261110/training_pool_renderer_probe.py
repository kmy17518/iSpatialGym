"""Bounded A/B/A native rendering with an explicitly initialized fixed actor union."""
import argparse
import copy
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys
import time
import traceback

from .contracts import require, file_digest, digest, write_json_atomic
from .training_pool_render_scene import scene_union, member_scene, observe_visibility

PARENT_COMMIT = 'a7f490047d5b4d7abdb6426ff047a6a39c293514'
ALLOWED = {'scripts/campaign_20261110/'+name for name in (
    'training_pool_render_scene.py', 'training_pool_renderer_probe.py',
    'tests/test_training_pool_render_scene.py', 'tests/test_training_pool_renderer_probe.py')}
DELTA = {'fixed_initialized_actor_union': True, 'source_recordings_unchanged': True,
         'native_slot_start_load_writer_cameras_retained': True,
         'selection_scope': 'exact_checked_two_source_records_A_B_A',
         'member_configuration': 'restore_each_exact_hash_bound_source_campaign_config_environment',
         'frame_reference': 'exact_own_independent_native_cold_readback_bound_to_same_recorded_frame',
         'inactive_actor': 'parked_and_hidden_before_native_restoration_observation',
         'added_physics_steps': 0, 'added_stop_play_pairs': 0, 'explicit_sdk_reconstructions': 0,
         'proprio': 'native_loaded_state', 'collection_time_behavior_proprio_recorded': False,
         'scope': 'same recorded room/support, two exact initialized model/scale actors only'}


def ref(path): return {'path': str(path), 'sha256': file_digest(path)}


def checked(root, reference):
    path = Path(reference['path']).resolve()
    require(path.is_relative_to(root) and ref(path) == reference, 'changed pool renderer input')
    return json.loads(path.read_text())


def selected_pool_records(records, wanted, expected):
    paths = [str(Path(path).resolve()) for path in wanted]
    require(paths == expected and len(paths) == 3 and paths[0] == paths[2] != paths[1],
            'pool renderer requires the exact authorized A/B/A sequence')
    by_path = {str(Path(r['artifacts']['demo']['path']).resolve().parent): r for r in records}
    require(len(records) == len(by_path) == 2 and set(paths) == set(by_path), 'unbound pool render member')
    # Each untouched record still passes native checked_metadata. Compatibility
    # across these two sources is the explicit fixed actor union, not a fake hash.
    return by_path


def project_active_loaded(loaded, contract, active):
    result = copy.deepcopy(loaded)
    inactive = set(contract['pool_names']) - {active}
    require(inactive <= set(result['objects']), 'initialized inactive actor missing from native observation')
    for name in inactive: del result['objects'][name]
    return result


def source_state_expected(actual, state):
    """Use existing restoration tolerances for fields actually present in the recording."""
    result = copy.deepcopy(actual); objects = state['registry']['object_registry']
    require(set(objects) <= set(actual['objects']), 'recording references an uninitialized object')
    result['objects'] = {name: copy.deepcopy(actual['objects'][name]) for name in objects}
    for name, value in objects.items():
        target = result['objects'][name]; root = value['root_link']
        for source, dest in [('pos','position'), ('ori','orientation'), ('lin_vel','linear_velocity'),
                             ('ang_vel','angular_velocity')]:
            if source in root:
                require(target[dest] is not None, 'recorded motion cannot map to missing native motion')
                target[dest] = root[source].tolist() if hasattr(root[source], 'tolist') else root[source]
        for source, dest in [('joint_pos','joint_positions'), ('joint_vel','joint_velocities')]:
            if source in value: target[dest] = value[source].tolist() if hasattr(value[source], 'tolist') else value[source]
    observed = copy.deepcopy(actual); observed['objects'] = {name: actual['objects'][name] for name in objects}
    return result, observed


def record_comparison(expected, observed, checker, publish):
    """Retain the first mismatching native state as well as successful checks."""
    evidence = {'cold_native_expected': expected, 'native_observed': observed, 'strict_match': False}
    try:
        checker(expected, observed)
        evidence['strict_match'] = True
    except BaseException as error:
        evidence['error'] = repr(error)
        raise
    finally:
        publish(evidence)
    return evidence


@contextmanager
def prohibit_native_mutation(sim, events):
    saved = []
    try:
        for target, name in ((sim,'step_physics'), (sim._sim_context,'step'), (sim,'stop'), (sim,'play')):
            had = name in vars(target); previous = vars(target).get(name)
            def reject(*args, _name=name, **kwargs):
                events.append(_name); require(False, 'retained renderer attempted physics/lifecycle mutation: '+_name)
            saved.append((target,name,had,previous)); setattr(target,name,reject)
        yield
    finally:
        for target,name,had,previous in reversed(saved):
            if had: setattr(target,name,previous)
            else: delattr(target,name)


def inputs(root, request_path):
    q = json.loads(Path(request_path).read_text())
    require(q['schema'] == 'isg-pool-native-render-reuse-request-v1' and q['sequence'] == [0,1,0]
            and q['physical_delta'] == DELTA and q['benchmark_credit'] == 0
            and q['production_released'] is False, 'unbounded renderer scope')
    runtime = Path(__file__).resolve().parents[2]; parent = Path(q['parent_runtime']['path']).parent
    require(ref(runtime/'.snapshot_commit') == q['candidate_runtime']
            and ref(parent/'.snapshot_commit') == q['parent_runtime']
            and (parent/'.snapshot_commit').read_text().strip() == PARENT_COMMIT,
            'renderer native/runtime provenance differs')
    paths = {str(p.relative_to(tree)) for tree in (parent,runtime) for folder in ('ispatialgym','scripts')
             for p in (tree/folder).rglob('*.py') if p.is_file()}
    changed = {name for name in paths if not (parent/name).is_file() or not (runtime/name).is_file()
               or file_digest(parent/name) != file_digest(runtime/name)}
    require(changed == set(q['changed_payloads']) and changed <= ALLOWED
            and all(file_digest(runtime/name) == sha for name,sha in q['changed_payloads'].items()),
            'unapproved native renderer payload change')
    cold = checked(root,q['cold_verification']); require(cold['status']=='native_cold_render_verified', 'cold renderer proof failed')
    helper = Path(q['cold_driver']['path']); require(ref(helper)==q['cold_driver'], 'cold source verifier changed')
    driver = runpy.run_path(str(helper)); records=[];documents=[]; requests=[]; observations=[]; cold_frames=[]
    for control, reference in zip(('FRESH_A','FRESH_B'),q['cold_requests']):
        checked(root,reference); request,record,document,old_runtime = driver['inputs'](root,reference['path'])
        require(request['control']==control and old_runtime==parent, 'cold renderer source differs')
        coldobs=checked(root,cold['controls'][control]['observation'])
        require(coldobs['request']==reference and coldobs['status']=='native_cold_render_passed', 'cold renderer observation differs')
        frame_ref = cold['controls'][control]['frame_readback']; frame_path = Path(frame_ref['path']).resolve()
        require(frame_path.is_relative_to(root) and ref(frame_path) == frame_ref == coldobs['frame_readback']
                and cold['controls'][control]['original_cold_state_matched'] is True,
                'native cold frame readback missing or changed')
        frames = [json.loads(line) for line in frame_path.read_text().splitlines()]
        require(len(frames) == cold['controls'][control]['native_frame_count'] == coldobs['episodes'][0]['frames']
                and [f['frame'] for f in frames] == list(range(len(frames))), 'native cold frame sequence differs')
        cold_frames.append(frames)
        records.append(record);documents.append(document);requests.append(request);observations.append(coldobs)
    environment_keys = set(requests[0]['source_environment']) | set(requests[1]['source_environment'])
    differences = {key for key in environment_keys
                   if requests[0]['source_environment'].get(key) != requests[1]['source_environment'].get(key)}
    require(differences <= {'ISPATIALGYM_CAMPAIGN_CONFIG'}, 'renderer backend environment differs')
    for index, request in enumerate(requests):
        path = Path(request['source_environment']['ISPATIALGYM_CAMPAIGN_CONFIG']).resolve()
        require(path.is_relative_to(root) and ref(path) == q['member_campaign_configs'][index],
                'renderer source member campaign configuration changed')
    source_request=checked(root,requests[0]['source_recording_request']); physical=checked(root,source_request['pool_contract'])
    names=[d['metadata']['ispatialgym']['model_pool']['active_names'][0] for d in documents]
    contract=scene_union(documents,names,'robot_r1',physical)
    require(digest(contract)==q['scene_union_sha256'], 'checked renderer scene union changed')
    for doc in documents: require(not doc['state']['registry']['system_registry'], 'new particle/system boundary is unmeasured')
    return q,records,documents,requests,observations,contract,names,runtime,cold_frames


def run(root,request_path,output,preflight=False):
    root=Path(root).resolve();request_path=Path(request_path).resolve();output=Path(output).resolve()
    require(request_path.is_relative_to(root) and output.is_relative_to(root), 'renderer outside campaign')
    q,records,documents,requests,cold,contract,names,runtime,cold_frames=inputs(root,request_path)
    if preflight: return {'passed':True,'request':ref(request_path),'native_execution':False,'scene_union_sha256':digest(contract)}
    require(not output.exists(),'preserve renderer output');output.mkdir(parents=True)
    os.environ.update(requests[0]['source_environment'])
    for name in ('ISPATIALGYM_HEAD_CAMERA_ONLY','ISPATIALGYM_PHYSICS_ONLY'): os.environ.pop(name,None)
    os.environ.update(ISGF_ROOT=str(root),ISPATIALGYM_CAMPAIGN_RESTORATION_PROBE='1',ISPATIALGYM_FAST_COLLECTOR_PREBUILD='1')
    from ispatialgym.multienv import replay as renderer, campaign_rendering as campaign
    from ispatialgym.sampling.campaign_runtime import restoration_observation
    from .training_pool_probe import world_construction_counter
    from .training_restoration import same_loaded_state
    import omnigibson as og
    require(Path(renderer.__file__).resolve().is_relative_to(runtime), 'loaded renderer outside candidate')
    directories=[str(Path(r['artifacts']['demo']['path']).parent) for r in records]
    wanted=[directories[index] for index in q['sequence']]
    result={'schema':'isg-pool-native-render-reuse-observation-v1','request':ref(request_path),'physical_delta':DELTA,
            'candidate_runtime':q['candidate_runtime'],'parent_runtime':q['parent_runtime'],'started_at':time.time(),
            'entries':[],'mutation_calls':[],'frame_counts':[], 'benchmark_credit':0,'production_released':False}
    selected=campaign.selected_records;environment=renderer.env_config;start=renderer.Slot.start;load=renderer.Slot.load
    frame_file=open(output/'frame_readback.jsonl','x');guard=None;identities=None;scene_identity=None
    def select(manifest,demo_dirs):
        rows=[json.loads(line) for line in Path(manifest).read_text().splitlines()]
        require(rows==records,'derived renderer manifest changed')
        return selected_pool_records(rows,demo_dirs,wanted)
    def env_config(first,demos,first_dir,n_slots,scene='recorded'):
        require(n_slots==1 and scene=='recorded' and first.dir==directories[0],'pool world construction differs')
        saved=first.scene_file;private,_=member_scene(contract,documents[0],names[0]);first.scene_file=private
        try: return environment(first,demos,first_dir,n_slots,scene=scene)
        finally: first.scene_file=saved
    def started(slot,demo):
        nonlocal guard,identities,scene_identity
        index=len(result['entries']);require(index<len(wanted) and demo.dir==wanted[index],'native renderer order differs')
        member=q['sequence'][index];active=names[member];saved=demo.scene_file
        os.environ.update(requests[member]['source_environment'])
        for key in ('ISPATIALGYM_HEAD_CAMERA_ONLY','ISPATIALGYM_PHYSICS_ONLY'):os.environ.pop(key,None)
        require(digest(saved)==digest(documents[member]),'native loaded recorded scene differs')
        private,parking=member_scene(contract,saved,active)
        if guard is None:
            identities={o.name:id(o) for o in slot.scene.objects};scene_identity=id(slot.scene)
            require(set(identities)==set(contract['initialized_actors']),'preloaded native actor inventory differs')
            guard=prohibit_native_mutation(og.sim,result['mutation_calls']);guard.__enter__()
        require(id(slot.scene)==scene_identity and {o.name:id(o) for o in slot.scene.objects}==identities,'native world actors replaced')
        restore=slot.scene.restore;had='restore' in vars(slot.scene);raw=vars(slot.scene).get('restore')
        def restored(document,*args,**kwargs):
            require(digest(document)==digest(private),'native restore input differs from checked union')
            answer=restore(document,*args,**kwargs)
            for name in contract['pool_names']:slot.scene.object_registry('name',name).visible=(name==active)
            observe_visibility(slot.scene,contract,active,parking)
            return answer
        slot.scene.restore=restored;demo.scene_file=private
        try: answer=start(slot,demo)
        finally:
            demo.scene_file=saved
            if had:slot.scene.restore=raw
            else:del slot.scene.restore
        require({o.name:id(o) for o in slot.scene.objects}==identities,'native restore changed actor identities')
        slot.n_objects_recorded=len(saved['objects_info']['init_info'])
        visibility=observe_visibility(slot.scene,contract,active,parking)
        native=slot.campaign_restoration['native_observation'];projected=project_active_loaded(native['loaded_state'],contract,active)
        original=cold[member]['rendered_observations'][0]['campaign_restoration']['native_observation']['loaded_state']
        result['entries'].append({'member':member,'active':active,'source_demo':records[member]['artifacts']['demo'],
            'source_scene':records[member]['artifacts']['recorded_scene'],'private_scene_sha256':digest(private),
            'visibility':visibility,'parking':parking,'native_restoration':slot.campaign_restoration,
            'active_loaded_state':projected,'cold_state_strict_match':False,'object_identities':dict(identities),'scene_identity':scene_identity})
        def publish_comparison(evidence):
            result['entries'][-1]['cold_state_comparison']=evidence
            result['entries'][-1]['cold_state_strict_match']=evidence['strict_match']
            write_json_atomic(output/'observation.json',result)
        record_comparison(original,projected,same_loaded_state,publish_comparison)
        result['frame_counts'].append(0);write_json_atomic(output/'observation.json',result);return answer
    def loaded(slot,flat,delta):
        require(delta is False,'native full state loads required');answer=load(slot,flat,delta)
        entry=result['entries'][-1];member=entry['member'];active=entry['active']
        observe_visibility(slot.scene,contract,active,entry['parking'])
        state,consumed=slot.scene.deserialize(flat);require(consumed==len(flat),'native state reader did not consume full source')
        require(not(set(state['registry']['object_registry']) & (set(names)-{active})),'source frame contains inactive actor')
        native=restoration_observation(slot.demo.campaign_meta['campaign'],slot.scene,og.sim)['loaded_state']
        require({o.name:id(o) for o in slot.scene.objects}==identities,'native frame replaced initialized actors')
        source_hash=hashlib.sha256(flat.detach().cpu().numpy().tobytes()).hexdigest()
        cold_frame=cold_frames[member][slot.i]
        require(cold_frame['frame']==slot.i and cold_frame['source_flat_sha256']==source_hash,
                'cold native readback does not belong to this exact source frame')
        expected=cold_frame['native_observation']['loaded_state']
        observed=project_active_loaded(native,contract,active)
        def publish_frame(evidence):
            frame_file.write(json.dumps({'episode':len(result['entries'])-1,'frame':slot.i,
                'source_flat_sha256':source_hash,**evidence})+'\n');frame_file.flush()
        record_comparison(expected,observed,same_loaded_state,publish_frame)
        result['frame_counts'][-1]+=1;return answer
    campaign.selected_records=select;renderer.env_config=env_config;renderer.Slot.start=started;renderer.Slot.load=loaded
    try:
        manifest=output/'diagnostic_sources.jsonl';manifest.write_text(''.join(json.dumps(r)+'\n' for r in records))
        listing=output/'demos.txt';listing.write_text(''.join(d+'\n' for d in wanted));native_results=output/'results.jsonl'
        with world_construction_counter(og) as world:
            result['world_construction']=world
            code=renderer.main(['--demos','@'+str(listing),'--campaign-manifest',str(manifest),'--root',str(output/'rendered'),
                '--task',records[0]['variant'],'--slots','1','--scene','recorded','--proprio','loaded',
                '--results',str(native_results),'--repo-prefix','pool_reuse_aba'])
        rows=[json.loads(line) for line in native_results.read_text().splitlines()];episodes=[r for r in rows if r.get('tag')=='REPLAY_DEMO']
        require(code==0 and world=={'attempted':1,'completed':1} and len(episodes)==len(result['entries'])==3
                and all(e['kept'] is True for e in episodes) and result['mutation_calls']==[], 'native reuse renderer incomplete')
        inputs(root,request_path)
        result.update(status='native_reuse_render_passed',native_results=ref(native_results),episodes=episodes,
                      videos=[ref(p) for p in sorted((output/'rendered').rglob('*.mp4'))],finished_at=time.time())
    except BaseException as error:
        result.update(status='failed',error=repr(error),traceback=traceback.format_exc(),finished_at=time.time());raise
    finally:
        if guard is not None:guard.__exit__(None,None,None)
        campaign.selected_records=selected;renderer.env_config=environment;renderer.Slot.start=start;renderer.Slot.load=load
        frame_file.close();result['frame_readback']=ref(output/'frame_readback.jsonl');write_json_atomic(output/'observation.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);parser.add_argument('--request',required=True)
    parser.add_argument('--output',required=True);parser.add_argument('--preflight',action='store_true');args=parser.parse_args()
    try: print(json.dumps(run(args.root,args.request,args.output,args.preflight)),flush=True)
    finally:
        native=sys.modules.get('omnigibson')
        if native is not None:native.shutdown()
