#!/usr/bin/env python3
"""Diagnostic-only native renders of lighting bounds from one exact TRO frame.

Uses the task's pose loader directly, avoiding the known expected-fixture replay
diagnostic. No TRO, template, acceptance record, or source scene is rewritten.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import traceback


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign-root',type=Path,required=True)
    p.add_argument('--tro',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();root=a.campaign_root.resolve();out=a.output.resolve();tro_path=a.tro.resolve()
    if out.exists() or not out.is_relative_to(root) or not tro_path.is_relative_to(root):raise ValueError('private source and new diagnostic output required')
    out.mkdir(parents=True)
    report={'schema':'campaign-lighting-render-review-v1','diagnostic_only':True,'production_released':False,
        'source_tro':str(tro_path),'source_tro_sha256':sha(tro_path),'status':'started','renders':[]}
    import numpy as np
    from PIL import Image,ImageDraw
    import omnigibson as og
    try:
        from ispatialgym.oracle.run_isg import InstanceLoader
        from ispatialgym.sampling import campaign_augmentation as A
        loader=InstanceLoader(str(tro_path),image_size=512)
        env=loader.build_env()
        loader.task.load_tro_instance(env,tro=loader.tro,settle_steps=0)
        cam=loader.robot.sensors[loader.cam_key]
        # Sensor world-pose caches still refer to the build pose immediately
        # after the task applies robot joints; synchronize through render only.
        for _ in range(16):og.sim.render()
        camera_pose=[x.cpu().tolist() for x in cam.get_position_orientation()]
        scene_file=Path(loader.cfg['scene']['scene_file'])
        campaign=loader.tro.meta.raw.get('campaign') or {}
        report.update(source_scene_file=str(scene_file),source_scene_sha256=sha(scene_file),
            camera_key=loader.cam_key,camera_pose=camera_pose,scene_model=loader.tmeta['scene'],
            source_campaign_lighting=campaign.get('perturbations',{}).get('lighting'),
            source_campaign_removed_objects=campaign.get('perturbations',{}).get('removed_objects',[]),
            restoration='build_env + task.load_tro_instance(settle_steps=0); only native render calls afterward',
            expected_fixture_verification='intentionally not called for diagnostic renders; this does not pass the replay gate')
        request=lambda i,c:{'schema':'isg-lighting-v1','intensity_factor':i,'skybox_color_factors':c}
        cases=[('default',request(1.,[1.,1.,1.])),('intensity_min',request(.7,[1.,1.,1.])),
               ('intensity_max',request(1.3,[1.,1.,1.])),('sky_min',request(1.,[.92,.92,.92])),
               ('sky_max',request(1.,[1.08,1.08,1.08])),('combined_min',request(.7,[.92,.92,.92])),
               ('combined_max',request(1.3,[1.08,1.08,1.08])),('cool_limit',request(1.,[.92,1.,1.08])),
               ('warm_limit',request(1.,[1.08,1.,.92]))]
        source_request=report['source_campaign_lighting']
        if source_request:cases.insert(1,('source_draw',source_request))
        baseline=None;frames=[]
        for name,settings in cases:
            actual=A.apply_lighting(env.scene,og.sim,settings)
            if baseline is None:baseline=actual;report['loaded_default_lighting']=actual
            else:
                if {k:v['default'] for k,v in actual['fixtures'].items()}!={k:v['default'] for k,v in baseline['fixtures'].items()}:
                    raise ValueError('fixture defaults compounded or changed')
                if actual['skybox']['default']!=baseline['skybox']['default']:raise ValueError('skybox defaults changed')
            for _ in range(16):og.sim.render()
            current=[x.cpu().tolist() for x in cam.get_position_orientation()]
            if not all(np.allclose(a,b,rtol=0,atol=1e-6) for a,b in zip(camera_pose,current)):
                report['camera_frame_failure']={'case':name,'reference':camera_pose,'observed':current}
                raise ValueError('camera frame moved')
            obs,_=cam.get_obs();rgb=obs['rgb'].cpu().numpy()[...,:3]
            if not np.isfinite(rgb).all() or float(rgb.std())<2:raise ValueError('invalid or empty head-camera render')
            rgb=rgb.astype(np.uint8);path=out/(name+'.png');Image.fromarray(rgb).save(path)
            luma=rgb.astype(float)@np.array([.2126,.7152,.0722])
            report['renders'].append({'name':name,'settings':settings,'actual_lighting':actual,'path':str(path),'sha256':sha(path),
                'luma_mean':float(luma.mean()),'luma_p05':float(np.quantile(luma,.05)),'luma_p95':float(np.quantile(luma,.95)),
                'rgb_near_black_fraction':float((rgb.max(axis=2)<5).mean()),'rgb_any_channel_clipped_fraction':float((rgb.max(axis=2)>250).mean())})
            frames.append((name,settings,rgb))
        restored=A.apply_lighting(env.scene,og.sim,request(1.,[1.,1.,1.]))
        if restored!=baseline:raise ValueError('default lighting restore differs after bounds sweep')
        sheet=Image.new('RGB',(3*512,((len(frames)+2)//3)*550),'white');draw=ImageDraw.Draw(sheet)
        for i,(name,settings,rgb) in enumerate(frames):
            x=(i%3)*512;y=(i//3)*550
            draw.text((x+5,y+3),name+'; intensity='+str(round(settings['intensity_factor'],4))+' RGB='+str([round(v,4) for v in settings['skybox_color_factors']]),fill='black')
            sheet.paste(Image.fromarray(rgb),(x,y+32))
        sheet_path=out/'lighting_contact_sheet.png';sheet.save(sheet_path)
        report.update(status='captured_for_visual_review',contact_sheet={'path':str(sheet_path),'sha256':sha(sheet_path)},
            source_tro_unchanged=sha(tro_path)==report['source_tro_sha256'],source_scene_unchanged=sha(scene_file)==report['source_scene_sha256'])
        if not report['source_tro_unchanged'] or not report['source_scene_unchanged']:raise ValueError('diagnostic source changed')
    except BaseException as exc:
        report.update(status='failed',error=str(exc),traceback=traceback.format_exc());traceback.print_exc();raise
    finally:
        (out/'lighting_review.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n');sys.stdout.flush();sys.stderr.flush();og.shutdown()


if __name__=='__main__':main()
