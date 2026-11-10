#!/usr/bin/env python3
"""Measure proper-rotation symmetry by visual geometry AND material groups.

Writes scalar evidence only; no provider mesh or decoded asset is exported.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import sys
import traceback


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign-root',type=Path,required=True)
    p.add_argument('--defaults',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();out=a.output.resolve()
    if out.exists() or not out.is_relative_to(a.campaign_root.resolve()):raise ValueError('new private output required')
    row=next(r for r in json.loads(a.defaults.read_text())['objects'] if r['class_id']=='mental_rotation-classroom_mock_up')
    if hashlib.sha256(Path(row['asset_path']).read_bytes()).hexdigest()!=row['asset_sha256']:raise ValueError('asset changed')
    out.mkdir(parents=True);report={'asset_sha256':row['asset_sha256'],'model':row['model'],'status':'started','release_eligible':False}
    import numpy as np
    from scipy.spatial import cKDTree
    import torch as th
    import trimesh
    import omnigibson as og
    from omnigibson.macros import gm
    from omnigibson.objects import DatasetObject
    import omnigibson.utils.transform_utils as T
    try:
        with gm.unlocked():gm.ENABLE_TRANSITION_RULES=False
        env=og.Environment(configs={'env':{'action_frequency':30,'physics_frequency':120,'rendering_frequency':30},'scene':{'type':'Scene'},'robots':[]})
        from pxr import Usd, UsdShade  # Kit exposes pxr after Environment launch.
        og.sim.stop();obj=DatasetObject(name='symmetry_target',category=row['category'],model=row['model'],dataset_name='behavior-1k-assets')
        env.scene.add_object(obj);og.sim.play();obj.set_position_orientation(th.tensor([0.,0.,2.]),th.tensor([0.,0.,0.,1.]))
        rp,rq=obj.root_link.get_position_orientation();groups={};textures=[]
        for link in obj.links.values():
            for vm in link.visual_meshes.values():
                if vm.geom_type!='Mesh':raise ValueError('visual primitive type requires additional handling')
                pos,quat=vm.get_position_orientation();pos,quat=T.relative_pose_transform(pos,quat,rp,rq)
                verts=(vm.points.cpu().numpy()*vm.get_world_scale().cpu().numpy())@T.quat2mat(quat).cpu().numpy().T+pos.cpu().numpy()
                mesh=trimesh.Trimesh(vertices=verts,faces=vm.faces.cpu().numpy(),process=False)
                material,_=UsdShade.MaterialBindingAPI(vm.prim).ComputeBoundMaterial()
                signature=[]
                if material:
                    for prim in Usd.PrimRange(material.GetPrim()):
                        for attr in prim.GetAttributes():
                            if attr.GetName().startswith('inputs:'):
                                value=str(attr.Get());signature.append((prim.GetTypeName(),attr.GetName(),value))
                                if 'asset' in str(attr.GetTypeName()).lower() and value not in ['None','']:
                                    textures.append(value)
                key=json.dumps(sorted(signature),sort_keys=True)
                groups.setdefault(key,[]).append(mesh)
        all_mesh=trimesh.util.concatenate([m for meshes in groups.values() for m in meshes]);centre=all_mesh.bounds.mean(axis=0)
        clouds={}
        for key,meshes in groups.items():
            mesh=trimesh.util.concatenate(meshes);n=max(20000,int(150000*mesh.area/all_mesh.area))
            points,_=trimesh.sample.sample_surface(mesh,n,seed=20261110)
            points-=centre;clouds[key]=(points,cKDTree(points),float(mesh.area))
        scores=[]
        for perm in itertools.permutations(range(3)):
            for signs in itertools.product((-1,1),repeat=3):
                R=np.eye(3)[list(perm)]*np.array(signs)[:,None]
                if round(np.linalg.det(R))!=1 or np.allclose(R,np.eye(3)):continue
                errors=[]
                for key,(points,tree,area) in clouds.items():
                    d,_=tree.query(points@R.T,k=1);errors.append({'material_sha256':hashlib.sha256(key.encode()).hexdigest(),
                        'p95_m':float(np.quantile(d,.95)),'p99_m':float(np.quantile(d,.99)),'area_m2':area})
                scores.append({'rotation':R.astype(int).tolist(),'material_errors':errors,
                    'max_material_p95_m':max(e['p95_m'] for e in errors),'max_material_p99_m':max(e['p99_m'] for e in errors)})
        scores.sort(key=lambda r:r['max_material_p95_m'])
        report.update(status='measured',material_groups=len(groups),texture_inputs=sorted(set(textures)),
            method='150k area-weighted visual-surface points; proper cube rotations about visual bbox centre; each material group compared separately',
            thresholds={'p95_m':.0015,'p99_m':.003},rotations=scores,
            symmetry_candidates=[s for s in scores if s['max_material_p95_m']<.0015 and s['max_material_p99_m']<.003],
            decision='requires visual material/texture review together with geometric scores; absence of candidates is not proof of asymmetry')
    except BaseException as exc:
        report.update(status='failed',error=str(exc),traceback=traceback.format_exc());traceback.print_exc();raise
    finally:
        (out/'symmetry.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n');sys.stdout.flush();sys.stderr.flush();og.shutdown()


if __name__=='__main__':main()
