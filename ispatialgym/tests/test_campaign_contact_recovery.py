from types import SimpleNamespace as NS

import pytest

from ispatialgym.sampling import campaign_runtime as CR
from ispatialgym.oracle import execute as EX
from ispatialgym.oracle.motions import MotionsConfig
from scripts.campaign_20261110.contracts import ContractError


def bind(monkeypatch, class_id, task, options):
    template = NS(class_id=class_id, task=task, scene_setup={'recovery_options': options})
    monkeypatch.setenv('ISPATIALGYM_CAMPAIGN_CONFIG', '/unused')
    monkeypatch.setattr(CR, 'configuration', lambda: (None, template, None, None))
    return template


def test_legacy_closure_and_unconfigured_campaign_stay_identical(monkeypatch):
    cfg = MotionsConfig()
    action = NS(kind='library')
    monkeypatch.delenv('ISPATIALGYM_CAMPAIGN_CONFIG', raising=False)
    assert EX.grasp_closure(action, .014, cfg) == (cfg.gripper_close_rate, None, False)
    assert not CR.contact_recovery('touching_push_mass_center')
    bind(monkeypatch, 'any', 'any', {})
    assert EX.grasp_closure(action, .014, cfg) == (cfg.gripper_close_rate, None, False)


def test_marker_contact_closure_is_bounded_and_does_not_change_handles(monkeypatch):
    bind(monkeypatch, 'inter_object_relation_satisfaction-nextto_01',
         'inter_object_relation_satisfaction-nextto_small_scale', {'marker_gentle_pinch': True})
    cfg = MotionsConfig()
    assert EX.grasp_closure(NS(kind='library'), .014, cfg) == (.0005, 0.0, True)
    for width in [None, .021, float('nan'), 0.0, -1.0]:
        assert EX.grasp_closure(NS(kind='library'), width, cfg) == EX.closure_for_width(width, cfg)
    assert EX.grasp_closure(NS(kind='handle'), .014, cfg) == (cfg.gripper_close_rate, None, False)


def test_mass_center_option_uses_existing_physical_constructor(monkeypatch):
    from ispatialgym.sampling.v2.tasks import iors_small as IS
    monkeypatch.delenv(IS.PUSH_LINE_COM_ENV, raising=False)
    template = bind(monkeypatch, 'inter_object_relation_satisfaction-touching_04',
                    'inter_object_relation_satisfaction-touching_small_scale', {'touching_push_mass_center': True})
    assert IS.push_line_com_on()
    template.class_id = 'inter_object_relation_satisfaction-touching_01'
    with pytest.raises(ContractError, match='another audited class'):
        IS.push_line_com_on()
    template.scene_setup['recovery_options']['touching_push_mass_center'] = 'true'
    with pytest.raises(ContractError, match='boolean'):
        IS.push_line_com_on()


def test_spoon_pad_policy_only_caps_audited_campaign_speed(monkeypatch):
    monkeypatch.delenv('ISPATIALGYM_CAMPAIGN_CONFIG', raising=False)
    assert EX.push_speed_of(NS(extra={})) is None
    assert EX.push_speed_of(NS(extra={'push_speed_m_s': .025})) == .025
    template = bind(monkeypatch, 'alignment-axial_05', 'alignment-axial', {'axial_spoon_slow_pad': True})
    assert EX.push_speed_of(NS(extra={})) == .01
    assert EX.push_speed_of(NS(extra={'push_speed_m_s': .025})) == .01
    assert EX.push_speed_of(NS(extra={'push_speed_m_s': .005})) == .005
    template.class_id = 'alignment-axial_04'
    with pytest.raises(ContractError, match='another audited class'):
        EX.push_speed_of(NS(extra={}))


def test_clutter_approach_is_only_enabled_for_audited_milk_class(monkeypatch):
    monkeypatch.delenv('ISPATIALGYM_CAMPAIGN_CONFIG', raising=False)
    assert not CR.contact_recovery('milk_clutter_approach')
    template = bind(monkeypatch, 'disembedding_04', 'disembedding', {'milk_clutter_approach': True})
    assert CR.contact_recovery('milk_clutter_approach')
    template.class_id = 'disembedding_06'
    with pytest.raises(ContractError, match='another audited class'):
        CR.contact_recovery('milk_clutter_approach')


def test_tray_joint_margin_is_bound_to_the_audited_cellulose_tape_class(monkeypatch):
    monkeypatch.delenv('ISPATIALGYM_CAMPAIGN_CONFIG', raising=False)
    assert not CR.contact_recovery('tray_grasp_joint_margin')
    template = bind(monkeypatch, 'disembedding_07', 'disembedding', {'tray_grasp_joint_margin': True})
    assert CR.contact_recovery('tray_grasp_joint_margin')
    template.class_id = 'disembedding_08'
    with pytest.raises(ContractError, match='another audited class'):
        CR.contact_recovery('tray_grasp_joint_margin')


def test_held_jar_brace_is_exact_category_model_and_default_scale_only(monkeypatch):
    monkeypatch.delenv('ISPATIALGYM_CAMPAIGN_CONFIG', raising=False)
    assert not CR.contact_recovery('held_jar_close_brace')
    template = bind(monkeypatch, 'configuration_matching-articulation_close_small_scale-hinged_jar',
                    'configuration_matching-articulation_close_small_scale', {'held_jar_close_brace': True})
    target = NS(role='target', category='hinged_jar', model='jnjtrl', default_scale=(1., 1., 1.))
    template.split, template.objects = 'eval_category', [target]
    assert CR.contact_recovery('held_jar_close_brace')
    for field, invalid in [('split', 'eval_standard'), ('class_id', 'configuration_matching-articulation_close_small_scale_01')]:
        old = getattr(template, field)
        setattr(template, field, invalid)
        with pytest.raises(ContractError): CR.contact_recovery('held_jar_close_brace')
        setattr(template, field, old)
    for field, invalid in [('model', 'other'), ('default_scale', (.9, .9, .9))]:
        old = getattr(target, field)
        setattr(target, field, invalid)
        with pytest.raises(ContractError): CR.contact_recovery('held_jar_close_brace')
        setattr(target, field, old)
    template.scene_setup['recovery_options']['held_jar_close_brace'] = 'true'
    with pytest.raises(ContractError, match='boolean'): CR.contact_recovery('held_jar_close_brace')


def test_touching_strokes_are_hash_bound_and_cannot_be_overridden(monkeypatch):
    from ispatialgym.sampling.v2.tasks import iors_small as IS
    monkeypatch.delenv('ISPATIALGYM_CAMPAIGN_CONFIG', raising=False)
    monkeypatch.delenv(IS.TOUCH_PUSH_STROKE_ENV, raising=False)
    assert IS.touch_push_stroke_m() is None
    template=bind(monkeypatch,'inter_object_relation_satisfaction-touching_04',
        'inter_object_relation_satisfaction-touching_small_scale',{'touching_push_stroke_m':.03})
    assert IS.touch_push_stroke_m()==.03
    monkeypatch.setenv(IS.TOUCH_PUSH_STROKE_ENV,'0.030')
    assert IS.touch_push_stroke_m()==.03
    for value in ['off','0','.04']:
        monkeypatch.setenv(IS.TOUCH_PUSH_STROKE_ENV,value)
        with pytest.raises(ValueError,match='differs from campaign'):IS.touch_push_stroke_m()
    monkeypatch.delenv(IS.TOUCH_PUSH_STROKE_ENV)
    template.class_id='inter_object_relation_satisfaction-touching_01'
    with pytest.raises(ContractError,match='audited spray-bottle'):IS.touch_push_stroke_m()


@pytest.mark.parametrize('value',[True,'0.03',0,-.03,.11,float('nan'),float('inf')])
def test_invalid_stroke_policy_never_enters_native_program(monkeypatch,value):
    bind(monkeypatch,'inter_object_relation_satisfaction-touching_04',
        'inter_object_relation_satisfaction-touching_small_scale',{'touching_push_stroke_m':value})
    with pytest.raises(ContractError,match='finite'):CR.touching_push_stroke_m()


def test_yaw_gain_requires_exact_class_and_stroke_configuration(monkeypatch):
    monkeypatch.delenv('ISPATIALGYM_CAMPAIGN_CONFIG', raising=False)
    assert CR.touching_push_yaw_gain() is None
    template=bind(monkeypatch,'inter_object_relation_satisfaction-touching_04',
        'inter_object_relation_satisfaction-touching_small_scale',
        {'touching_push_stroke_m':.03,'touching_push_yaw_gain_m_per_rad':.06})
    assert CR.touching_push_yaw_gain()==.06
    template.class_id='inter_object_relation_satisfaction-touching_01'
    with pytest.raises(ContractError,match='audited spray-bottle'):CR.touching_push_yaw_gain()
    template.class_id='inter_object_relation_satisfaction-touching_04'
    template.scene_setup['recovery_options'].pop('touching_push_stroke_m')
    with pytest.raises(ContractError,match='configured stroke'):CR.touching_push_yaw_gain()


@pytest.mark.parametrize('value',[True,'0.06',0,-.06,.16,float('nan'),float('inf')])
def test_invalid_yaw_gain_is_rejected(monkeypatch,value):
    bind(monkeypatch,'inter_object_relation_satisfaction-touching_04',
        'inter_object_relation_satisfaction-touching_small_scale',
        {'touching_push_stroke_m':.03,'touching_push_yaw_gain_m_per_rad':value})
    with pytest.raises(ContractError,match='finite'):CR.touching_push_yaw_gain()


@pytest.mark.parametrize('option', ['shelf_board_draw','shelf_other_arm'])
def test_shelf_options_are_bound_to_exact_released_class(monkeypatch, option):
    from types import SimpleNamespace
    from scripts.campaign_20261110.contracts import ContractError
    monkeypatch.setattr(CR, 'active', lambda: True)
    template=SimpleNamespace(class_id='inter_object_relation_satisfaction-touching_03',
        task='inter_object_relation_satisfaction-touching_small_scale',scene_setup={'recovery_options':{option:True}})
    monkeypatch.setattr(CR,'configuration',lambda:({},template,None,None))
    assert CR.contact_recovery(option)
    template.class_id='inter_object_relation_satisfaction-touching_04'
    with pytest.raises(ContractError):CR.contact_recovery(option)


def test_touching_inhand_reaim_requires_the_exact_configuration(monkeypatch):
    monkeypatch.delenv('ISPATIALGYM_CAMPAIGN_CONFIG', raising=False)
    monkeypatch.setattr(EX, 'variant_generation', lambda ctx: {})
    cfg=NS(place_inhand_reaim=False)
    assert not EX.place_inhand_reaim_wanted({},cfg)
    template=bind(monkeypatch,'inter_object_relation_satisfaction-touching_04',
        'inter_object_relation_satisfaction-touching_small_scale',{'touching_place_inhand_reaim':True})
    assert EX.place_inhand_reaim_wanted({},cfg)
    template.class_id='inter_object_relation_satisfaction-touching_02'
    with pytest.raises(ContractError,match='another audited class'):
        EX.place_inhand_reaim_wanted({},cfg)


def test_touching_online_carry_is_opt_in_and_exact_class_bound(monkeypatch):
    monkeypatch.delenv('ISPATIALGYM_CAMPAIGN_CONFIG', raising=False)
    assert not EX.carry_online_wanted(NS(extra={}))
    assert EX.carry_online_wanted(NS(extra={'carry_online': True}))
    template=bind(monkeypatch,'inter_object_relation_satisfaction-touching_04',
        'inter_object_relation_satisfaction-touching_small_scale',{'touching_carry_online':True})
    assert EX.carry_online_wanted(NS(extra={}))
    template.class_id='inter_object_relation_satisfaction-touching_03'
    with pytest.raises(ContractError,match='another audited class'):
        EX.carry_online_wanted(NS(extra={}))
    template.class_id='inter_object_relation_satisfaction-touching_04'
    template.scene_setup['recovery_options']['touching_carry_online']='true'
    with pytest.raises(ContractError,match='boolean'):
        EX.carry_online_wanted(NS(extra={}))


def test_touching_protected_carry_requires_exact_class_policy(monkeypatch):
    monkeypatch.delenv('ISPATIALGYM_CAMPAIGN_CONFIG', raising=False)
    monkeypatch.delenv('ISPATIALGYM_CAMPAIGN_PROTECTED_CARRY', raising=False)
    assert not EX.protected_carry_recovery_wanted()
    template=bind(monkeypatch,'inter_object_relation_satisfaction-touching_04',
        'inter_object_relation_satisfaction-touching_small_scale',{'touching_protected_carry':True})
    assert EX.protected_carry_recovery_wanted()
    template.class_id='inter_object_relation_satisfaction-touching_03'
    with pytest.raises(ContractError,match='another audited class'):
        EX.protected_carry_recovery_wanted()


def test_shelf_depth_proposal_is_hash_bound_to_only_touching03(monkeypatch):
    monkeypatch.delenv('ISPATIALGYM_CAMPAIGN_CONFIG', raising=False)
    assert not CR.contact_recovery('shelf_target_depth_draw')
    template=bind(monkeypatch,'inter_object_relation_satisfaction-touching_03',
        'inter_object_relation_satisfaction-touching_small_scale',{'shelf_target_depth_draw':True})
    assert CR.contact_recovery('shelf_target_depth_draw')
    template.class_id='inter_object_relation_satisfaction-touching_04'
    with pytest.raises(ContractError,match='another audited class'):
        CR.contact_recovery('shelf_target_depth_draw')
