"""Read-only cold/reused replay trajectory comparisons; never release production."""
from pathlib import Path
import json
import numpy as np
from .contracts import file_digest, require


def _metric(delta, tolerance):
    require(delta.ndim >= 1 and len(delta) > 0 and np.isfinite(delta).all(),
            'trajectory comparison has empty or nonfinite observations')
    per_frame = delta.reshape(len(delta), -1).max(axis=1)
    failed = np.flatnonzero(per_frame > tolerance)
    return {'maximum': float(per_frame.max()), 'tolerance': tolerance,
            'frames_exceeding': len(failed),
            'first_frame_exceeding': int(failed[0]) if len(failed) else None,
            'passed': len(failed) == 0}


def compare_arrays(reference, actual, *, prefix=False):
    """Use existing reset bounds across every recorded frame, without alignment.

    A controlled abort may compare its explicitly requested prefix. Ordinary
    episodes must contain the same full actions and frame counts. No DTW,
    dropped failing frames, or inferred velocities are used.
    """
    require(reference.keys() == actual.keys(), 'trajectory field inventory differs')
    required = {'action', 'joint_positions', 'joint_velocities', 'base_pose',
                'eef_pose_left', 'eef_pose_right'}
    require(required <= reference.keys() and any(k.startswith('object_pose/') for k in reference),
            'required recorded dynamics are missing')
    frames = len(actual['action']); cold_frames = len(reference['action'])
    require(frames > 0 and (frames <= cold_frames if prefix else frames == cold_frames),
            'trajectory frame count differs')
    metrics = {}
    for key in sorted(reference):
        a = np.asarray(reference[key], dtype=np.float64)
        b = np.asarray(actual[key], dtype=np.float64)
        require(len(a) == cold_frames and len(b) == frames and a.shape[1:] == b.shape[1:]
                and np.isfinite(a).all() and np.isfinite(b).all(),
                'trajectory shape or finite values differ: '+key)
        a = a[:frames]
        if key == 'action':
            require(np.array_equal(a,b), 'cold/reused requested actions differ')
            metrics[key] = _metric(np.abs(a-b),0.)
        elif key in ('joint_positions','joint_velocities'):
            # Same numerical bounds as training_restoration.same_loaded_state.
            metrics[key] = _metric(np.abs(a-b),1e-5 if key=='joint_positions' else 1e-4)
        else:
            require(key.startswith('object_pose/') or key in
                    ('base_pose','eef_pose_left','eef_pose_right','head_camera_pose'),
                    'unsupported trajectory observation: '+key)
            require(a.shape == (frames,7), 'pose trajectory layout differs')
            metrics[key+'/position_m'] = _metric(np.linalg.norm(a[:,:3]-b[:,:3],axis=1),.002)
            na=np.linalg.norm(a[:,3:],axis=1);nb=np.linalg.norm(b[:,3:],axis=1)
            require((na>0).all() and (nb>0).all(), 'zero recorded orientation quaternion')
            dot=np.abs(np.sum(a[:,3:]*b[:,3:],axis=1)/(na*nb))
            metrics[key+'/orientation_rad'] = _metric(2*np.arccos(np.clip(dot,0.,1.)),1e-4)
    return {'status':'matched' if all(m['passed'] for m in metrics.values()) else 'mismatch',
            'frames_compared':frames, 'reference_frames':cold_frames,
            'controlled_abort_prefix':prefix,'requested_actions_exact':True,'metrics':metrics,
            'scope':'Recorded robot joints/velocities and robot/task-object poses at every frame. '
                    'Unrecorded object velocities and untracked scene actors are not inferred.',
            'production_released':False,'benchmark_credit':0}


def _trajectory(root, report_ref):
    import h5py
    root=Path(root).resolve();p=Path(report_ref['path']).resolve()
    require(p.is_relative_to(root) and file_digest(p)==report_ref['sha256'],'trajectory report changed')
    report=json.loads(p.read_text());path=Path(report['hdf5']).resolve()
    require(path == p.parent/'demo.hdf5' and path.is_relative_to(root),
            'trajectory does not belong to its native report')
    reference={'path':str(path),'sha256':file_digest(path)}
    with h5py.File(path,'r') as stream:
        data=stream['data'];group=data['demo_0'];observations=group['isg']
        identity={k:data.attrs[k] for k in ('joint_names','pose_layout','robot_name','controller_order','controller_dims')}
        arrays={'action':group['action'][:]}
        for key in ('joint_positions','joint_velocities','base_pose','eef_pose_left','eef_pose_right','head_camera_pose'):
            if key in observations:arrays[key]=observations[key][:]
        for key in observations['object_pose']:
            arrays['object_pose/'+key]=observations['object_pose'][key][:]
    require(file_digest(path)==reference['sha256'],'trajectory changed during observation')
    return report,identity,arrays,reference


def audit(root, reference_report, reused_report, *, controlled_abort_prefix=False):
    a,ai,aa,ar=_trajectory(root,reference_report)
    b,bi,ba,br=_trajectory(root,reused_report)
    require(ai==bi,'recorded robot/controller identity differs')
    require(a['instance_id']==b['instance_id'] and a['replay']==b['replay'],
            'replay input differs between cold and reused controls')
    source=Path(a['replay']).resolve()
    require(source.is_relative_to(Path(root).resolve()),'source action file escapes campaign')
    if controlled_abort_prefix:
        abort=b['loader']['campaign_restoration']['controlled_abort']
        require(abort.get('injected') is True and abort.get('native_control_calls_completed')==1
                and abort.get('exception_type') in ('ControlledEpisodeFailure','ControlledEpisodeInterrupt'),
                'prefix comparison requires the recorded one-step controlled abort')
    comparison=compare_arrays(aa,ba,prefix=controlled_abort_prefix)
    return {'schema':'isg-cold-reused-recorded-trajectory-audit-v1',
            'reference_report':reference_report,'reused_report':reused_report,
            'reference_trajectory':ar,'reused_trajectory':br,
            'source_actions':{'path':str(source),'sha256':file_digest(source)},**comparison}
