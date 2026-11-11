"""Bounded physical pool mechanics probe; never generates benchmark instances.

Two different selected model identities are preloaded once, switched A/B/A,
observed natively and exported active-only. A separate process reloads the
exports. This does not substitute for oracle demos, true action replay, or the
remaining family/configuration validation gates.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import time
from scripts.campaign_20261110.contracts import file_digest, require, write_json_atomic


def observation(world, output, name):
    import torch as th
    from PIL import Image
    world.render(5)
    pixels = world.visible_pixels(exclude_structure=False)
    image = output/(name+'.png')
    Image.fromarray(world.rgb().numpy().astype('uint8')).save(image)
    return {'pixels':pixels,'image':{'path':str(image),'sha256':file_digest(image)},
            'native':world.observe_model_pool(world._pool_objects, world.spawned),
            'observed_at_unix':time.time()}


def run(manifest_path, output, mode, source=None):
    from ispatialgym.sampling.model_pool import ModelPool, read_manifest
    from ispatialgym.sampling import robot_utils  # imports warp before OG
    from ispatialgym.sampling.sim_utils import SimWorld
    import omnigibson as og
    import omnigibson.utils.transform_utils as T
    import torch as th
    manifest = json.loads(Path(manifest_path).read_text())
    manifest, reference = read_manifest(manifest_path, scene=manifest['scene'], rooms=manifest['rooms'],
                                        fixed_bindings=manifest['fixed_bindings'])
    require(len(manifest['entries']) == 2 and len({r['model'] for r in manifest['entries']}) == 2,
            'this bounded probe needs two distinct models')
    require(not output.exists(), 'preserve previous probe output')
    output.mkdir(parents=True)
    result = {'schema':'isg-native-model-pool-probe-v1','campaign_id':'campaign_20261110',
              'mode':mode,'manifest':reference,'benchmark_credit':0,'production_released':False,
              'native_runtime':{'path':str(Path(__file__).resolve().parents[2]/'.snapshot_commit'),
                  'sha256':file_digest(Path(__file__).resolve().parents[2]/'.snapshot_commit')},
              'job_id':os.environ.get('SLURM_JOB_ID'),'cpu_affinity':sorted(os.sched_getaffinity(0)),
              'process_id':os.getpid(),'started_at_unix':time.time(),'stages':[],
              'scope':'Physical mechanics only. Oracle demo, action replay, TRO integration and other family paths remain unverified.'}
    write_json_atomic(output/'result.json',result)
    try:
        if mode == 'switch':
            world = SimWorld(manifest['scene'],manifest['rooms'],cameras=True,image_size=256)
            world.model_pool = ModelPool(world,manifest,reference,output/'execution.json')
            cp,cq = world.camera_pose(); position=cp+T.quat2mat(cq)@th.tensor([0.,0.,-.8])
            initial_ids={name:id(obj) for name,obj in world._pool_objects.items()}
            for seq,index in enumerate((0,1,0)):
                row=manifest['entries'][index]
                obj=world.spawn('target',row['category'],row['model'],dataset=row['dataset'],scale=row['scale'])
                obj.set_position_orientation(position,th.tensor([0.,0.,0.,1.]));obj.keep_still()
                for _ in range(4):og.sim.step_physics()
                observed=observation(world,output,str(seq))
                require(observed['pixels'].get(obj.name,0)>0,'active pool model was not rendered')
                inactive=set(world._pool_objects)-{obj.name}
                for name in inactive:
                    state=observed['native']['objects'][name]
                    require(not state['visible'] and not state['planner_included'] and not any(state['collision_enabled']),
                            'inactive object leaks into rendering/planning/contacts')
                    require(observed['pixels'].get(name,0)==0,'inactive object leaked into rendered observation')
                require({n:id(o) for n,o in world._pool_objects.items()}==initial_ids,'pool identity was replaced')
                folder=output/str(seq)
                partial,full=world.save_template(str(folder),{'target.n.01_1':obj.name,'agent.n.01_1':world.robot.name},{})
                exported=json.loads(Path(partial).read_text())
                require(not inactive & set(exported['objects_info']['init_info']),'inactive pool object exported')
                result['stages'].append({'sequence':seq,'active':obj.name,'model':row['model'],'observation':observed,
                    'template':{'path':full,'sha256':file_digest(full)},'partial_template':{'path':partial,'sha256':file_digest(partial)},
                    'robot_pose':[[float(v) for v in x] for x in world.robot.get_position_orientation()],
                    'robot_joints':world.robot.get_joint_positions().tolist(),
                    'object_pose':[[float(v) for v in x] for x in obj.get_position_orientation()]})
                world.model_pool.release_template()
                write_json_atomic(output/'result.json',result)
            result['mechanics_passed']=True;result['scene_loads']=1;result['pool_physics_rebuilds']=world._pool_physics_rebuilds
        else:
            source=Path(source);prior=json.loads(source.read_text())
            require(prior['manifest']==reference and prior.get('mechanics_passed') is True,'source pool mechanics incomplete')
            result['source']={'path':str(source),'sha256':file_digest(source)}
            for entry in prior['stages']:
                ref=entry['template'];require(file_digest(ref['path'])==ref['sha256'],'export bytes changed')
                world=SimWorld(manifest['scene'],manifest['rooms'],cameras=True,image_size=256,
                               scene_variant={'path':ref['path']})
                world.robot.set_position_orientation(*[th.tensor(x) for x in entry['robot_pose']])
                world.robot.set_joint_positions(th.tensor(entry['robot_joints']));world.robot.keep_still()
                obj=world.obj(entry['active'])
                require(obj.model==entry['model'],'independent restoration selected wrong model')
                require(not (set(prior['stages'][i]['active'] for i in range(3))-{obj.name}) & {o.name for o in world.scene.objects},
                        'inactive model is present in independently loaded export')
                world._pool_objects={obj.name:obj};world.spawned={obj.name:obj}
                observed=observation(world,output,str(entry['sequence']))
                require(observed['pixels'].get(obj.name,0)>0,'independent restored active model was not rendered')
                result['stages'].append({'sequence':entry['sequence'],'active':obj.name,'model':obj.model,'observation':observed})
                write_json_atomic(output/'result.json',result);og.clear()
            result['independent_load_render_passed']=True
        result['finished_at_unix']=time.time();write_json_atomic(output/'result.json',result)
    except Exception as exc:
        result['failure']={'type':type(exc).__name__,'message':str(exc),'observed_at_unix':time.time()}
        write_json_atomic(output/'result.json',result);raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--mode',choices=('switch','restore'),required=True);p.add_argument('--source',type=Path);a=p.parse_args()
    run(a.manifest,a.output,a.mode,a.source)
