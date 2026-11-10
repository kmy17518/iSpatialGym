#!/usr/bin/env python3
"""Render all exact converted assets in a landscape grid without asset edits."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import traceback


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign-root',type=Path,required=True)
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();root=a.campaign_root.resolve();out=a.output.resolve();dataset=a.dataset.resolve()
    if not out.is_relative_to(root) or out==root or out.exists() or not dataset.is_relative_to(root):
        raise ValueError('capture output must be new and all inputs private')
    source=dataset/'campaign_readiness.json';data=json.loads(source.read_text());rows=data['objects']
    for row in rows:
        if not row['ready'] or row['smoke']['status']!='passed' or sha(Path(row['usd']))!=row['asset_sha256']:
            raise ValueError('physical evidence or actual asset hash mismatch')
    out.mkdir(parents=True)
    import numpy as np
    import torch as th
    from PIL import Image
    import omnigibson as og
    from omnigibson.objects import DatasetObject
    from omnigibson.utils.transform_utils import mat2quat
    report={'source_readiness_sha256':sha(source),'dataset':str(dataset),'models':[r['model'] for r in rows],
            'asset_sha256':{r['model']:r['asset_sha256'] for r in rows},'images':[],'status':'started'}
    try:
        env=og.Environment(configs={'env':{'action_frequency':30,'physics_frequency':120,'rendering_frequency':30},
            'render':{'viewer_width':1600,'viewer_height':1200},'scene':{'type':'Scene'},'robots':[]})
        og.sim.stop();objects=[]
        for row in rows:
            obj=DatasetObject(name='review_'+row['model'],category=row['category'],model=row['model'],
                              dataset_name=str(dataset),scale=[1.,1.,1.])
            env.scene.add_object(obj);objects.append(obj)
        og.sim.play()
        cols=min(7,len(objects));nrows=(len(objects)+cols-1)//cols
        for i,obj in enumerate(objects):
            obj.set_position_orientation(th.tensor([(i%cols)*.3,(i//cols)*.3,.01]),th.tensor([0.,0.,0.,1.]))
        centre=np.array([(cols-1)*.15,(nrows-1)*.15,.02]);distance=max(cols*.3,nrows*.3)*1.3
        for name,offset in [('all_top',[0.,0.,distance]),('all_oblique',[.1,-.3,distance*1.1])]:
            eye=centre+offset;back=(eye-centre)/np.linalg.norm(eye-centre);right=np.cross([0.,1.,0.],back);right/=np.linalg.norm(right);up=np.cross(back,right)
            og.sim.viewer_camera.set_position_orientation(th.tensor(eye,dtype=th.float32),mat2quat(th.tensor(np.column_stack([right,up,back]),dtype=th.float32)))
            for _ in range(8): og.sim.render()
            obs,_=og.sim.viewer_camera.get_obs();rgb=obs['rgb'].cpu().numpy()[...,:3]
            if not np.isfinite(rgb).all() or float(rgb.std())<2.: raise ValueError('empty/invalid render')
            image=out/(name+'.png');Image.fromarray(rgb.astype(np.uint8)).save(image);report['images'].append({'path':str(image),'sha256':sha(image)})
        report.update(status='captured_for_visual_review',columns=cols,rows=nrows)
    except BaseException as exc:
        report.update(status='failed',error=str(exc),traceback=traceback.format_exc());traceback.print_exc();raise
    finally:
        (out/'capture.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n');sys.stdout.flush();sys.stderr.flush();og.shutdown()


if __name__=='__main__': main()
