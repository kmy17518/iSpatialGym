#!/usr/bin/env python3
"""Admit exact generated assets only after native physical and visual evidence.

Run beside the private datasets, so asset hashes are re-read from actual bytes.
This records model readiness only; no task/template becomes production-ready.
"""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign-root',type=Path,required=True)
    p.add_argument('--datasets',type=Path,nargs='+',required=True)
    p.add_argument('--reviews-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    root=a.campaign_root.resolve();output=a.output.resolve()
    if not output.is_relative_to(root) or output==root or output.exists():
        raise ValueError('registry must be a new private campaign file')
    objects={}
    for dataset in a.datasets:
        dataset=dataset.resolve()
        if not dataset.is_relative_to(root):
            raise ValueError('dataset outside private campaign root')
        readiness_path=dataset/'campaign_readiness.json'
        report=json.loads(readiness_path.read_text())
        review_path=a.reviews_root/dataset.name/'visual_review.json'
        review=json.loads(review_path.read_text())
        if review.get('decision')!='passed':
            raise ValueError(f'visual review not passed: {dataset}')
        images=report['visual_smoke_images']
        if len(images)<2 or set(review['images_sha256'])!={Path(i['path']).name for i in images}:
            raise ValueError('visual review does not cover all native smoke views')
        for image in images:
            path=Path(image['path']).resolve()
            if not path.is_relative_to(dataset) or sha(path)!=image['sha256'] or sha(path)!=review['images_sha256'][path.name]:
                raise ValueError('visual evidence hash mismatch')
        additional=review.get('additional_capture')
        if additional:
            capture_path=Path(additional['path']).resolve()
            if not capture_path.is_relative_to(root) or sha(capture_path)!=additional['sha256']:
                raise ValueError('additional visual capture hash mismatch')
            capture=json.loads(capture_path.read_text())
            if capture['status']!='captured_for_visual_review' or capture['source_readiness_sha256']!=sha(readiness_path) or set(capture['models'])!={r['model'] for r in report['objects']}:
                raise ValueError('additional visual capture lacks exact model/source coverage')
            for image in capture['images']:
                image_path=Path(image['path']).resolve()
                if not image_path.is_relative_to(root) or sha(image_path)!=image['sha256'] or additional['images_sha256'].get(image_path.name)!=image['sha256']:
                    raise ValueError('additional reviewed image differs')
        if (dataset/'campaign_conversion_failure.json').exists():
            raise ValueError('dataset has unresolved conversion failure')
        for item in report['objects']:
            key=item['category']+'/'+item['model']
            if key in objects or not item['ready'] or item['conversion']!='passed' or item['smoke']['status']!='passed' or item['smoke']['steps']<90 or item['collision_prims']<1:
                raise ValueError(f'invalid or duplicate physical evidence for {key}')
            usd=Path(item['usd']).resolve()
            if not usd.is_relative_to(dataset) or sha(usd)!=item['asset_sha256']:
                raise ValueError(f'actual asset hash differs from physically tested bytes: {key}')
            evidence={'type':'campaign_generated_project_asset','asset_sha256':item['asset_sha256'],
                'physical_report_path':str(readiness_path),'physical_report_sha256':sha(readiness_path),
                'visual_review_path':str(review_path),'visual_review_sha256':sha(review_path),
                'build_index_sha256':report['build_index_sha256'],'source_urdf_sha256':item['source_urdf_sha256'],
                'build_manifest_sha256':item['build_manifest_sha256'],
                'scope':'Model readiness only; task geometry, grasps, fresh demo and open-loop replay remain required.'}
            objects[key]={'category':item['category'],'model':item['model'],'dataset':str(dataset),'usd':str(usd),
                'ready':True,'asset_sha256':item['asset_sha256'],'default_scale':[1.,1.,1.],
                'scale_source':'Exact explicit DatasetObject scale in the physically tested converter load configuration',
                'readiness_source':json.dumps(evidence,sort_keys=True,separators=(',',':')),
                'family':item['family'],'split_reservation':item['split_reservation'],'task_validation':'pending'}
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps({'schema':'campaign-generated-project-readiness-v1','objects':objects,'production_templates_released':0},indent=2,sort_keys=True)+'\n')
    print(json.dumps({'ready_models':len(objects),'output':str(output),'registry_sha256':sha(output)}))


if __name__=='__main__':
    main()
