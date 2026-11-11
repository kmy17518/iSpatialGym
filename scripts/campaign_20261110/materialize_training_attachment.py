"""Coupled attachment scale plans for the shared native training materializer.

The attachment caller additionally requires its own measured mating geometry
and exact attachment-map audit. Tabletop maps alone never validate this path.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

from scripts.campaign_20261110.contracts import Perturbations, ScaleState, require, file_digest
from scripts.campaign_20261110.training_attachment_dependencies import verify_pair_request


def scale_plan(template, request, probe, profile, job):
    require(template.task=='configuration_matching-attachment','attachment plan requires the attachment task')
    require(template.class_id in request['classes'] and template.scene==probe['scene'],
            'attachment measured dependency belongs to another class or scene')
    pair=request['coupled_pair'];verify_pair_request(job,pair,asdict(template))
    names=template.scene_setup['inst_to_name']
    supports=[o for o in template.objects if names.get(o.instance)==probe['object']]
    require(len(supports)==1 and supports[0].role=='support' and supports[0].semantic_kind=='object',
            'measured support does not uniquely match native attachment support')
    support=supports[0]
    require((support.category,support.model,support.asset_sha256)==
            (profile['category'],profile['model'],profile['asset_sha256']), 'support identity differs')
    require(list(support.default_scale)==profile['default_scale'],'original support default differs')
    require(request['target_scale']==job['scale'] and request['support_scale']==probe['absolute_scale'],
            'requested applied scales differ from measured dependencies')
    require(request['actual_height_key']==probe['actual_height_key'],'attachment support height differs')
    states={o.instance:ScaleState(o.default_scale,o.default_scale,'fixed') for o in template.objects}
    for role,scale in (('target',job['scale']),('partner',pair['partner']['scale'])):
        objects=[o for o in template.objects if o.role==role]
        require(len(objects)==1,'attachment pair must contain one male and one female')
        obj=objects[0];states[obj.instance]=ScaleState(obj.default_scale,tuple(scale),'uniform')
    states[support.instance]=ScaleState(support.default_scale,tuple(probe['absolute_scale']),'height')
    result=Perturbations(states);result.validate('train',states)
    return result


def materialize(root,release,code,output,classes,jobs_manifest):
    from scripts.campaign_20261110.materialize_training_table import materialize as shared_materialize
    from scripts.campaign_20261110.training_attachment_release import verify_release
    release=Path(release)
    report=json.loads((release/'training_height_reader_audit.json').read_text())
    require(report['schema']=='isg-training-attachment-release-v1' and report['native_reader_errors']==0,
            'dedicated attachment dependency release required')
    require(file_digest(report['audit'])==report['audit_sha256'],'attachment dependency audit changed')
    proof=json.loads(Path(report['audit']).read_text())
    require(proof['schema']=='isg-training-attachment-native-audit-v1' and proof['verified']==proof['requests']==1
            and not proof['failures'],'complete exact attachment audit required')
    for path,sha in proof['files'].items():require(file_digest(path)==sha,'attachment source evidence changed')
    verify_release(release,proof,report['routes'])
    return shared_materialize(root,release,code,output,classes,plan_builder=scale_plan,jobs_manifest=jobs_manifest)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','release','code','output','jobs-manifest'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--class-id',action='append',required=True);a=p.parse_args()
    materialize(a.root,a.release,a.code,a.output,set(a.class_id),a.jobs_manifest)
