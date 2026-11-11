from dataclasses import asdict

import pytest

from scripts.campaign_20261110.contracts import ObjectRef, Template
from scripts.campaign_20261110.materialize_training_attachment import scale_plan
from scripts.campaign_20261110.training_attachment_dependencies import coupled_profile


def fixture():
    def obj(instance,role,category,model,scale):
        return ObjectRef(instance,role,category,model,'a'*64,scale,ready=True,readiness_source='verified inventory')
    target=obj('lid.n.02_1','target','lid','male',(.65,)*3)
    partner=obj('stockpot.n.01_1','partner','stockpot','female',(.65,)*3)
    support=obj('desk.n.01_1','support','desk','support',(1.,1.,1.))
    template=Template('source','configuration_matching-attachment','configuration_matching-attachment_04',
        'eval_standard','office_cubicles_right','private_office_2',(target,partner,support),'b'*64,
        {'inst_to_name':{'desk.n.01_1':'desk_support_1'}})
    coupled=coupled_profile(asdict(target),asdict(partner))
    job={**asdict(target),'scale':coupled['absolute_scales']['target'][0],
        'campaign_profile':coupled['id'],'campaign_scale_slot':0,'campaign_classes':[template.class_id]}
    pair={'class_id':template.class_id,'profile_id':coupled['id'],'scale_slot':0,
          'partner':{**asdict(partner),'scale':coupled['absolute_scales']['partner'][0]}}
    probe={'scene':template.scene,'object':'desk_support_1','absolute_scale':[1.,1.,.9],
           'actual_height_key':'0.7248'}
    profile={**asdict(support),'default_scale':list(support.default_scale)}
    request={'classes':[template.class_id],'coupled_pair':pair,'target_scale':job['scale'],
             'support_scale':probe['absolute_scale'],'actual_height_key':'0.7248'}
    return template,request,probe,profile,job


def test_applied_pair_scales_are_coupled_and_defaults_remain_original():
    template,request,probe,profile,job=fixture()
    result=scale_plan(template,request,probe,profile,job)
    male=result.scales['lid.n.02_1'];female=result.scales['stockpot.n.01_1']
    assert male.default==female.default==(.65,)*3
    assert male.applied==female.applied and male.applied!=male.default
    assert result.scales['desk.n.01_1'].applied==(1.,1.,.9)
    with pytest.raises(ValueError,match='no perturbations'):result.validate('eval_standard',result.scales)


def test_attachment_height_plan_rejects_unmeasured_height_or_xy_scaling():
    template,request,probe,profile,job=fixture();request['actual_height_key']='0.8126'
    with pytest.raises(ValueError,match='height differs'):scale_plan(template,request,probe,profile,job)
    request['actual_height_key']='0.7248';probe['absolute_scale']=[.9,1.,.9];request['support_scale']=probe['absolute_scale']
    with pytest.raises(ValueError,match='preserve x/y'):scale_plan(template,request,probe,profile,job)


def test_attachment_plan_rejects_partner_identity_substitution():
    args=fixture();args[1]['coupled_pair']['partner']['model']='unverified'
    with pytest.raises(ValueError,match='ready source asset'):scale_plan(*args)
