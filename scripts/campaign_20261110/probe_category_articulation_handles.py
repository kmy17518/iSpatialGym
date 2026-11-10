"""Measure and render native handle candidates in an audited private scene."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import traceback


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene-probe',type=Path,required=True)
    p.add_argument('--model-key',required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    proof=json.loads(a.scene_probe.read_text())
    assert proof['status']=='private_context_geometry_and_reload_passed'
    scene=Path(proof['native_scene_path'])
    assert sha(scene)==proof['native_scene_sha256']
    rows=[x for x in proof['objects'] if x['model_key']==a.model_key]
    assert len(rows)==1
    row=rows[0]
    assert row.get('joint_sweeps') and all(not x['blockers'] for x in row['joint_sweeps'])
    a.output.mkdir(parents=True,exist_ok=False)
    report={'status':'started','model_key':a.model_key,'scene':proof['scene'],'room':proof['room'],
        'scene_probe_sha256':sha(a.scene_probe),'native_scene_sha256':sha(scene),
        'asset_sha256':row['asset_sha256'],'default_scale':row['default_scale'],
        'helper_sha256':sha(__file__),'production_released':False}
    __import__('ispatialgym.sampling.robot_utils')
    import omnigibson as og
    import torch as th
    from ispatialgym.sampling.sim_utils import SimWorld
    from ispatialgym.sampling.scene_inventory import Inventory, prior_reviews
    try:
        world=SimWorld(proof['scene'],[proof['room']],scene_variant={'path':str(scene)},cameras=False)
        obj=world.obj(row['name'])
        assert obj.category+'/'+obj.model==a.model_key
        assert max(abs(float(x)-y) for x,y in zip(obj.scale,row['default_scale']))<1e-6
        # Reuse the native inventory's geometry and handle inference on this
        # explicitly selected private world, without loading another scene.
        inv=Inventory.__new__(Inventory)
        inv.scene_model=proof['scene'];inv.world=world;inv.scene=world.scene
        inv.trav=world.scene.trav_map;inv.floor_map=inv.trav.floor_map[0]
        inv.eroded=inv.trav._erode_trav_map(th.clone(inv.floor_map),robot=world.robot)
        inv.out=str(a.output);inv.markers=None;inv.render=True
        inv._prim_to_obj={link.prim_path:o for o in world.scene.objects for link in o.links.values()}
        for name in ['artic','artic/renders','artic/strips']:(a.output/name).mkdir(exist_ok=True)
        state=og.sim.dump_state()
        record=inv._artic_record(obj,prior_reviews())
        assert {x['joint'] for x in record['joints']} >= {'j_leftdoor','j_rightdoor'}
        record['campaign_provenance']=report.copy()
        target=a.output/'artic'/f'{obj.name}.json'
        target.write_text(json.dumps(record,indent=2)+'\n')
        og.sim.load_state(state);world.keep_still(world.scene.objects)
        world.robot.set_position_orientation(th.tensor([60.,60.,5.]),th.tensor([0.,0.,0.,1.]))
        world.robot.disable_gravity();world.robot.keep_still()
        strip=inv._render_strip(obj,record)
        report.update(status='native_handle_geometry_requires_visual_and_task_validation',
            annotation=str(target),annotation_sha256=sha(target),strip=str(a.output/strip),
            strip_sha256=sha(a.output/strip),joints=[{'joint':j['joint'],'status':j['status'],
              'handles':j.get('handles',[]),'fallback':j.get('fallback')} for j in record['joints']])
        assert sha(scene)==proof['native_scene_sha256']
    except BaseException as exc:
        report.update(status='failed',error=str(exc),traceback=traceback.format_exc())
        traceback.print_exc();raise
    finally:
        (a.output/'handle_probe.json').write_text(json.dumps(report,indent=2)+'\n')
        sys.stdout.flush();sys.stderr.flush();og.shutdown()


if __name__=='__main__':main()
