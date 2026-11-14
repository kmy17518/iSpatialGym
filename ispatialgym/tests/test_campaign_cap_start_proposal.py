import math
from types import SimpleNamespace as NS
import pytest
from ispatialgym.sampling import campaign_attachment_goals as goals
from ispatialgym.sampling import campaign_runtime as runtime
from scripts.campaign_20261110.contracts import ContractError


@pytest.mark.parametrize('base,start,allowed', [
    ([9.19180584, 3.42423296, 3.13904102], [8.54128265, 3.42333913], True),
    ([9.19195080, 3.48090553, 3.13904102], [8.53593159, 3.47114110], True),
    ([9.19186783, 3.44840717, 3.13904102], [8.54086971, 3.46654105], True),
    ([9.19185829, 3.44478726, 3.13904102], [8.49455738, 3.45485735], False),
    ([9.19187546, 3.45142245, 3.13904102], [8.51555061, 3.42869568], False),
    ([9.19193172, 3.47369313, 3.13904102], [8.43276787, 3.43028164], False),
])
def test_measured_successful_and_failed_start_geometries(base, start, allowed):
    assert goals.cap_start_within_reach(start, base) is allowed
    # The screen is in the robot's forward frame, independent of room axes.
    angle = .7; c, s = math.cos(angle), math.sin(angle)
    def rotated(p): return [c*p[0]-s*p[1]+5, s*p[0]+c*p[1]-2]
    assert goals.cap_start_within_reach(rotated(start), [*rotated(base), base[2]+angle]) is allowed


def test_invalid_or_behind_robot_geometry_is_not_an_admitted_start():
    assert not goals.cap_start_within_reach([-.3, 0], [0, 0, 0])
    with pytest.raises(ValueError): goals.cap_start_within_reach([float('nan'), 0], [0, 0, 0])


def test_proposal_is_opt_in_and_bound_to_exact_canonical_pair(monkeypatch):
    target = NS(role='target', category='cap', model='xxqyrt', default_scale=(1.,)*3)
    partner = NS(role='partner', category='lemonade_bottle', model='yprkek', default_scale=(1.,)*3)
    template = NS(class_id='configuration_matching-attachment_02', task='configuration_matching-attachment',
                  split='eval_standard', objects=[target, partner], scene_setup={'recovery_options': {}})
    monkeypatch.setattr(runtime, 'active', lambda: True)
    monkeypatch.setattr(runtime, 'configuration', lambda: (None, template, None, None))
    assert not runtime.contact_recovery('cap_shallow_start_proposal')
    template.scene_setup['recovery_options']['cap_shallow_start_proposal'] = True
    assert runtime.contact_recovery('cap_shallow_start_proposal')
    for obj, key, value in [(target, 'model', 'other'), (partner, 'model', 'other'),
                            (partner, 'default_scale', (.8,)*3), (template, 'split', 'eval_model'),
                            (template, 'class_id', 'configuration_matching-attachment_04')]:
        previous = getattr(obj, key); setattr(obj, key, value)
        with pytest.raises(ContractError): runtime.contact_recovery('cap_shallow_start_proposal')
        setattr(obj, key, previous)


def test_shallow_start_screen_needs_the_anchored_source_stance():
    # Online fresh-goal draw (no plan) and online plans without the source stance: the screen is not applied.
    assert not goals.cap_shallow_start_applies(None)
    assert not goals.cap_shallow_start_applies(NS(anchor_manipulation_stance=None))
    assert not goals.cap_shallow_start_applies(NS())
    # The measured fixed-goal plans (source stance base 9.19 m, focus 0.67 m forward) keep it.
    anchored = NS(anchor_manipulation_stance={'base_pose2d': [9.1918, 3.4242, 3.1390], 'focus': [8.5218, 3.4259, .8182],
                                              'translate': .5, 'tilt': -.4542})
    assert goals.cap_shallow_start_applies(anchored)
