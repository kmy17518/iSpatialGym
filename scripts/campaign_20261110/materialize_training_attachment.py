"""Coupled attachment scale plans for the shared native training materializer.

The attachment caller additionally requires its own measured mating geometry
and exact attachment-map audit. Tabletop maps alone never validate this path.
"""
from dataclasses import asdict

from scripts.campaign_20261110.contracts import Perturbations, ScaleState, require
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
