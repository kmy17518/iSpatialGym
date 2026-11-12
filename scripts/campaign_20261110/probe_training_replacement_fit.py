"""Bounded same-place native replacement geometry; no task or release credit."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(request_path, output):
    q=json.loads(Path(request_path).read_text());output=Path(output)
    if output.exists():raise ValueError('preserve previous native evidence')
    for source in q['sources']:
        if sha(source['path'])!=source['sha256']:raise ValueError('native input bytes changed: '+source['path'])
    output.mkdir(parents=True)
    from ispatialgym.sampling import robot_utils  # native bootstrap order
    import numpy as np
    import torch as th
    import omnigibson as og
    from omnigibson.utils.transform_utils import quat2mat
    from ispatialgym.sampling.sim_utils import SimWorld, _collision_piece_aabbs
    from ispatialgym.sampling.knowledge import is_ready
    report={'schema':'isg-training-native-replacement-fit-v1','request':{'path':str(request_path),'sha256':sha(request_path)},
            'started_at':time.time(),'status':'started','rows':[],'benchmark_credit':0,'production_authorized':False,
            'scope':'Conservative maximum uniform replacement scale, actual native bounds and same-place context collision only. '
                    'Joint motion, goal/layout feasibility, support usable geometry, IK, visibility, dynamic stability, demo and replay remain required.'}
    def save():
        (output/'result.json').write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
    save()
    try:
        world=SimWorld(q['scene'],q['rooms'],scene_variant={'path':q['scene_source']['path']},cameras=False)
        assert sha(world.stable_path)==q['scene_source']['sha256']
        reference=world.obj(q['reference']['name']);assert reference.category+'/'+reference.model==q['reference']['model_key']
        assert np.allclose(reference.scale.tolist(),q['reference']['scale'],atol=1e-6,rtol=0)
        rpos,rquat=reference.get_position_orientation();rpos=rpos.clone();rquat=rquat.clone();rlo,rhi=reference.aabb;rlo=rlo.clone();rhi=rhi.clone();extent=(rhi-rlo).cpu().numpy();rotation=np.abs(quat2mat(rquat).cpu().numpy())
        assert np.isfinite(extent).all() and (extent>0).all()
        report['reference_observation']={'name':reference.name,'scale':reference.scale.tolist(),'pose':[rpos.tolist(),rquat.tolist()],
                                         'aabb':[rlo.tolist(),rhi.tolist()],'fixed_base':q['reference']['fixed_base']}
        # The original reference is absent from the candidate context; every
        # other scene object remains at its exact loaded pose.
        world.park(reference,500)
        context={o.name:_collision_piece_aabbs(o) for o in world.scene.objects
                 if o.name not in (reference.name,world.robot.name) and o.category!='floors'}
        for ordinal,candidate in enumerate(q['candidates']):
            assert is_ready(candidate['model']) and sha(candidate['asset']['path'])==candidate['asset']['sha256']
            native=candidate['native_default_geometry'];default_extent=np.asarray(native['native_extent'],dtype=float)
            assert default_extent.shape==(3,) and np.isfinite(default_extent).all() and (default_extent>0).all()
            # Rotate the default native bounding box. The 2% inset is a proposal
            # margin; actual loaded candidate bounds and collision checks decide.
            factor=min(1.,float(np.min(extent/(rotation@default_extent))))*.98
            scale=[float(s)*factor for s in native['default_scale']]
            obj=world.spawn('replacement_fit_'+str(ordinal),candidate['category'],candidate['model'],
                            dataset=candidate['dataset'],scale=scale,fixed_base=q['reference']['fixed_base'])
            obj.set_position_orientation(rpos,rquat);world.keep_still([obj]);lo,hi=obj.aabb
            # Keep the replacement's footprint centre and bottom at the original
            # object's place, retaining its world orientation.
            delta=th.stack([(rlo[0]+rhi[0]-lo[0]-hi[0])*.5,(rlo[1]+rhi[1]-lo[1]-hi[1])*.5,rlo[2]-lo[2]])
            obj.set_position_orientation(rpos+delta,rquat);world.keep_still([obj]);lo,hi=obj.aabb
            size_ok=bool(th.all(hi-lo<=rhi-rlo+1e-5));pieces=_collision_piece_aabbs(obj);overlaps=[]
            for name,others in context.items():
                depth=max((min(min(float(a[1][i]),float(b[1][i]))-max(float(a[0][i]),float(b[0][i])) for i in range(3))
                           for a in pieces for b in others),default=-1.)
                if depth>.002:overlaps.append({'object':name,'piece_aabb_overlap_m':depth})
            row={'model_key':candidate['category']+'/'+candidate['model'],'asset':candidate['asset'],
                 'maximum_uniform_scale_proposal':scale,'relative_to_native_default':factor,'actual_scale':obj.scale.tolist(),
                 'actual_aabb':[lo.tolist(),hi.tolist()],'actual_pose':[x.tolist()for x in obj.get_position_orientation()],
                 'no_larger_than_reference':size_ok,'bottom_delta_m':float(lo[2]-rlo[2]),'collision_piece_count':len(pieces),
                 'context_collision_piece_aabb_overlaps':overlaps,'consumer_input_ids':candidate['consumer_input_ids'],
                 'status':'geometry_candidate_passed' if size_ok and not overlaps and pieces else 'geometry_candidate_rejected',
                 'task_ready':False,'configured_scale_selected':False}
            report['rows'].append(row);save();world.park(obj,600+ordinal)
        report.update(status='native_geometry_measurements_complete',finished_at=time.time());save()
    except BaseException as exc:
        report.update(status='failed',failure={'type':type(exc).__name__,'message':str(exc)},finished_at=time.time());save();raise
    finally:
        og.shutdown()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--request',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.request,a.output)
