"""Bounded diagnostic of a closed single lid; writes no accepted motion library.

The exact convex-hull audit proposes underside contacts. This checks whether
the real free-standing carton and robot can follow a small opening arc from
the closed state. A positive result still needs a complete native program,
body-contact QA, initial-to-goal uniqueness, demo and independent replay.
"""
from __future__ import annotations
import argparse
import copy
import json
import time
from pathlib import Path

from scripts.campaign_20261110.contracts import file_digest, require, write_json_atomic

ALIAS = 'configuration_matching-articulation_open_small_scale-carton-breakfast_table'
ASSET_SHA256 = '2705cd76c20b11802698088fdcf75d59420eff135a08db2cf8a92c0f48add649'
KIND = 'diagnostic_single_closed_lid_lift'


def contact_variants(audit, arm):
    require(audit['schema']=='isg-carton-lip-contact-audit-v1'
            and audit['asset_sha256']==ASSET_SHA256
            and audit['scale']==[1,1,1] and audit['joint_initial_rad']==0,
            'wrong geometry audit or closed initial state')
    rows=[]
    for pitch in [0,10]:
        candidates=[r for r in audit['results']if r['arm']==arm and r['yaw_deg']==0
                    and r['pitch_deg']==pitch and r['side_y_m']==-.07
                    and r['geometry_contact_candidate']]
        require(len(candidates)==1,'missing exact bounded contact proposal')
        r=candidates[0];contact=r['first_lid_contact']
        require(contact['role']=='finger' and r['conservative_body_plane_clearance_m']>=.001-1e-10,
                'contact lacks conservative body clearance')
        origin=list(r['origin_closed_fingers']);origin[2]+=contact['lift_m']
        R=r['rotation_eef_to_object'];approach=[R[i][2]for i in range(3)];closing=[R[i][1]for i in range(3)]
        h={'kind':'diagnostic_lip_fist','contact':[origin[i]+.012*approach[i]for i in range(3)],
           'approach_dir':approach,'pinch_axis':closing,'width':None,
           'note':'Exact closed-lid underside contact; geometry proposal only.'}
        rows.append({'name':f'closed_lip_yaw0_pitch{pitch}_{arm}','handle':h,
                     'preshape_q':0.,'contact_q':0.,'width_m':None,'geometry_proposal':r})
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit',required=True);p.add_argument('--customs',required=True)
    p.add_argument('--out',required=True);p.add_argument('--arm',choices=['right','left'],required=True)
    a=p.parse_args();out=Path(a.out).resolve();require(not out.exists(),'preserve existing diagnostic output')
    audit=json.loads(Path(a.audit).read_text());variants=contact_variants(audit,a.arm)
    # All imports of simulation infrastructure occur after bounded inputs pass.
    from ispatialgym import REPO_ROOT
    from ispatialgym.grasping import validate_tabletop as VT
    from ispatialgym.grasping import validate_artic as VA
    native=Path(REPO_ROOT)
    require(file_digest(native/'ispatialgym/assets/robot/r1pro_hand_collision.json')==audit['hand_collision_sha256'],
            'native hand collision geometry differs from the offline audit')
    asset=native/'Behavior-1K/datasets/behavior-1k-assets/objects/carton/hhlmbi/usd/hhlmbi.encrypted.usd'
    require(file_digest(asset)==ASSET_SHA256,'native asset differs from audited geometry')
    custom=json.loads(Path(a.customs).read_text())[ALIAS]
    require(custom['roles']['target']['representative']=='hhlmbi'
            and not custom.get('generation',{}).get('prep_push'),
            'single-lid probe must not invoke a two-flap preparation')

    class ProbeWorld(VA.MotionWorld):
        def push_motion(self,motion):
            require(motion['motion']==KIND and motion['diagnostic_only']is True
                    and motion['alias']==ALIAS and motion['joint']=='j_link0'
                    and motion['q_start']==0. and motion['q_goal']==.3,
                    'unexpected diagnostic motion')
            return True

    kw={'arm':a.arm}
    if a.arm=='left':kw['base_offsets']=VA.mirrored_offsets(VT.ValidateConfig().base_offsets)
    world=VT.TabletopWorld(VT.ValidateConfig(base_yaws=1,**kw))
    # Disable the extra contact retry. The validator's initial IK fallback is
    # still explicitly labelled start_at=contact and never proves an approach.
    mw=ProbeWorld(VA.ArticWorld(world,VA.ArticConfig(max_segments=1)),
                  VA.MotionConfig(max_trials=2,contact_retry=False))
    out.mkdir(parents=True);t0=time.time();ctx=None
    report={'schema':'isg-carton-lip-native-probe-v1','campaign_id':'campaign_20261110','owner':'/root',
            'diagnostic_only':True,'benchmark_credit':0,'library_exported':False,'arm':a.arm,
            'audit':str(Path(a.audit).resolve()),'audit_sha256':file_digest(a.audit),
            'customs_sha256':file_digest(a.customs),'runtime':str(native),
            'max_trials':2,'trials':[],'started_at':t0,
            'limitations':'Contact/motion diagnosis only. No native program, approach/body-contact certificate, demo, replay or acceptance pass.'}
    write_json_atomic(out/'probe.json',report)
    try:
        ctx=mw.setup(ALIAS,str(out/'isolated_geometry'),a.customs)
        require(len(ctx['obj'].joints)==1,'single-lid joint identity changed')
        require(ctx['motion']['category']=='carton' and ctx['motion']['model']=='hhlmbi'
                and ctx['motion']['scale']==[1.,1.,1.] and ctx['motion']['lower']==0.,
                'native model, scale or closed joint differs from the audited asset')
        report.update(rest=ctx['rest'],profile=ctx['profile'])
        require(ctx['rest']['stable'],'native rest is not stable')
        motion=copy.deepcopy(ctx['motion'])
        motion.update(motion=KIND,diagnostic_only=True,q_start=0.,q_goal=.3,q_arc_end=.35,variants=variants)
        report['motion']=motion
        state,rest_sim=ctx['state']
        for variant in variants:
            trial=mw.trial(ctx['obj'],motion,variant,state,rest_sim,ctx['geom'])
            trial.update(diagnostic_only=True,benchmark_credit=0)
            report['trials'].append(trial);report['updated_at']=time.time()
            write_json_atomic(out/'probe.json',report)
            print(json.dumps({'variant':variant['name'],**{k:trial.get(k)for k in ['stop','achieved_rad','validated_travel_rad','steps','wall_s']} }),flush=True)
    except Exception as exc:
        report['error']=type(exc).__name__+': '+str(exc)
        raise
    finally:
        report.update(completed_at=time.time(),wall_s=time.time()-t0)
        write_json_atomic(out/'probe.json',report)
        if ctx is not None:world.unload(ctx['obj'])
        import omnigibson as og
        og.shutdown()


if __name__=='__main__':main()
