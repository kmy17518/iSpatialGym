#!/usr/bin/env python3
"""Measure heldout tray floors with the exact native disembedding algorithm."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import traceback
from types import SimpleNamespace


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ('campaign-root','defaults','output'):p.add_argument('--'+arg,type=Path,required=True)
    p.add_argument('--models',nargs=2,default=['gsxbym','mhhoga'])
    p.add_argument('--scene',default='Rs_int');p.add_argument('--room',default='living_room_0')
    p.add_argument('--support',default='breakfast_table_skczfi_0');p.add_argument('--scene-path',type=Path)
    a=p.parse_args();out=a.output.resolve()
    if out.exists() or not out.is_relative_to(a.campaign_root.resolve()):raise ValueError('new private output required')
    defaults=json.loads(a.defaults.read_text())
    if defaults['status']!='native_defaults_and_geometry_recorded':raise ValueError('native defaults not complete')
    selected={r['model']:r for r in defaults['objects'] if r['category']=='tray' and r['model'] in set(a.models)}
    if len(selected)!=2:raise ValueError('missing reserved source/destination tray')
    out.mkdir(parents=True);report={'schema':'campaign-heldout-tray-floor-v1','status':'started','production_released':False,'defaults_sha256':sha(a.defaults),'objects':{}}
    __import__('ispatialgym.sampling.robot_utils')
    import omnigibson as og
    from ispatialgym.sampling.sim_utils import SimWorld
    from ispatialgym.sampling.v2.tasks.disembedding import TrayFloor
    from ispatialgym.robot import conventions as C
    try:
        world=SimWorld(a.scene,[a.room],scene_variant={'path':str(a.scene_path)} if a.scene_path else None,cameras=False)
        if a.scene_path and sha(Path(world.stable_path))!=sha(a.scene_path):raise ValueError('runtime private support scene differs')
        support=world.obj(a.support);top=world.support_top(support)
        report.update(scene=a.scene,room=a.room,support_name=support.name,support_scale=support.scale.tolist(),native_scene_sha256=sha(Path(world.stable_path)),floor_z_world=float(world.floor_z),tabletop_height_world=float(top.z),tabletop_height_key=f'{top.z-world.floor_z:.4f}')
        fake=SimpleNamespace(world=world,robot=world.robot)
        for model,row in selected.items():
            if not row['ready'] or sha(Path(row['asset_path']))!=row['asset_sha256']:raise ValueError('tray changed or unready')
            tray=world.spawn('heldout_tray_'+model,'tray',model,scale=row['default_scale'])
            hull=world.local_hull(tray);bottom=world.bottom_offset(hull,C.IDENTITY_QUAT)
            floor=TrayFloor.measure(fake,tray,top,.01)
            height=float(top.z)+float(bottom)+floor.z_local-float(world.floor_z)
            report['objects'][model]={'asset_sha256':row['asset_sha256'],'default_scale':row['default_scale'],'bottom_offset_identity':float(bottom),'inner_floor_z_local':floor.z_local,'inner_floor_area_m2':floor.area(),'inner_floor_cell_count':floor.n_cells,'rim_z_local':floor.rim_local,'map_height_above_floor':height,'map_height_key':f'{height:.4f}','placed_key':'tray/'+model+'@focus_centred_edge_aligned','method':'native disembedding.TrayFloor.measure + native BandSpec z_support formula'}
        report['status']='native_tray_floor_measurement_passed'
    except BaseException as exc:
        report.update(status='failed',error=str(exc),traceback=traceback.format_exc());traceback.print_exc();raise
    finally:
        (out/'tray_floor_probe.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n');sys.stdout.flush();sys.stderr.flush();og.shutdown()


if __name__=='__main__':main()
