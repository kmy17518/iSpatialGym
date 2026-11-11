import unittest

from ispatialgym.sampling import campaign_pair_goals as g


class PairPlanTests(unittest.TestCase):
    def fixture(self, relation="ontop"):
        return {"schema": g.SCHEMA, "task": f"inter_object_relation_satisfaction-{relation}_small_scale",
                "campaign_id": "fresh", "group_id": "g", "split": "train", "template_id": "t", "scene": "s", "room": "r",
                "contract_template_sha256": "a" * 64, "target_name": "target", "partner_name": "partner",
                "release_pose": {"position": [0, 0, .81], "orientation": [0, 0, 0, 1]},
                "goal_pose": {"position": [0, 0, .8], "orientation": [0, 0, 0, 1]},
                "partner_pose": {"position": [0, 0, .7], "orientation": [0, 0, 0, 1]},
                "created_at": 100, "budget_seconds": 300, "max_members": 3}

    def parse(self, relation="ontop", now=1000):
        return g.validate_pair_plan(self.fixture(relation), template_id="t", scene="s", rooms=["r"],
                                     relation=relation, activated_at=999, now=now)

    def test_contact_plan_requires_distinct_pre_push_place_and_deadline_rejects(self):
        with self.assertRaises(g.GoalPlanError):
            self.parse("touching")
        with self.assertRaises(g.GoalGroupTimeout):
            self.parse(now=1299)

    def test_touching_keeps_contact_goal_separate_from_the_pre_push_place(self):
        raw = self.fixture("touching")
        raw["place_pose"] = {"position": [0, -.08, .8], "orientation": [0, 0, 0, 1]}
        plan = g.validate_pair_plan(raw, template_id="t", scene="s", rooms=["r"], relation="touching", activated_at=999, now=1000)
        self.assertEqual(plan.place[0], (0., -.08, .8))
        self.assertEqual(plan.summary()["requested_goal"]["target"]["position"], [0., 0., .8])
        self.assertEqual(plan.summary()["requested_place"]["position"], [0., -.08, .8])

    def test_recover_object_place_handles_rotated_grasp_frame(self):
        import math
        half = math.sqrt(.5)
        gripper = {"position": [1, 2.1, 3], "orientation": [0, 0, half, half]}
        local = {"position": [.1, 0, 0], "orientation": [0, 0, 0, 1]}
        position, rotation = g.object_pose_from_grasp(gripper, local)
        self.assertAlmostEqual(position[0], 1.)
        self.assertAlmostEqual(position[1], 2.)
        self.assertEqual(rotation, (0., 0., half, half))

    def test_native_settling_tolerance_and_inside_orientation_policy(self):
        p = self.parse()
        p.verify_settled(([.002, 0, .8], [0, 0, 0, 1]), p.partner)
        with self.assertRaises(g.GoalPlanError):
            p.verify_settled(([.04, 0, .8], [0, 0, 0, 1]), p.partner)
        with self.assertRaises(g.GoalPlanError):
            p.verify_settled(p.goal, ([.006, 0, .7], p.partner[1]))
        p = self.parse("inside")
        p.verify_settled((p.goal[0], [1, 0, 0, 0]), p.partner)
        self.assertFalse(p.summary()["target_orientation_required"])

    def test_native_grasp_height_is_explicit_and_only_for_small_inside(self):
        raw = self.fixture('inside')
        kw = dict(template_id='t', scene='s', rooms=['r'], relation='inside', activated_at=999, now=1000)
        self.assertEqual(g.validate_pair_plan(raw, **kw).release_policy, 'fixed')
        raw['release_policy'] = 'native_grasp_height'
        self.assertEqual(g.validate_pair_plan(raw, **kw).release_policy, 'native_grasp_height')
        raw['task'] = 'inter_object_relation_satisfaction-ontop_small_scale'
        with self.assertRaises(g.GoalPlanError):
            g.validate_pair_plan(raw, **{**kw, 'relation': 'ontop'})

    def test_nextto_gap_varies_in_middle_eighty_percent_before_start_draw(self):
        goal = ([0, .2, .7], [0, 0, 0, 1])
        partner = ([0, 0, .7], [0, 0, 0, 1])
        for fraction, expected in ((0, .01), (.5, .05), (1, .09)):
            changed, released, gap = g.vary_nextto_gap(goal, goal, partner, gap_max=.1, observed_gap=.08, base_yaw=0, fraction=fraction)
            self.assertAlmostEqual(gap, expected)
            self.assertAlmostEqual(changed[0][1], .2 + expected - .08)
            self.assertEqual(changed, released)


if __name__ == "__main__":
    unittest.main()


def test_large_pair_schema_rejects_missing_cell_and_preserves_inside_orientation():
    import pytest
    raw = PairPlanTests().fixture("inside")
    raw["task"] = "inter_object_relation_satisfaction-inside_large_scale"
    kw = dict(template_id="t", scene="s", rooms=["r"], relation="inside", scale="large", activated_at=999, now=1000)
    with pytest.raises(g.GoalPlanError, match="fixed cell"):
        g.validate_pair_plan(raw, **kw)
    raw["goal_cell"] = [0, 0]
    p = g.validate_pair_plan(raw, **kw)
    assert p.orientation_required and p.goal_cell == (0., 0.)
    with pytest.raises(g.GoalPlanError, match="target drifted"):
        p.verify_settled((p.goal[0], [1, 0, 0, 0]), p.partner)


def test_large_under_requires_forward_native_lane_and_rejects_altered_geometry():
    import copy
    import pytest
    raw = PairPlanTests().fixture('under')
    raw.update(task='inter_object_relation_satisfaction-under_large_scale',
               placement_mode='drop_push', goal_cell=[0., 0.],
               place_pose={'position': [-.3, 0., 1.], 'orientation': [0., 0., 0., 1.]},
               floor_push={'edge': '-x', 'direction': [1., 0.]})
    kw = dict(template_id='t', scene='s', rooms=['r'], relation='under', scale='large', activated_at=999, now=1000)
    plan = g.validate_pair_plan(raw, **kw)
    assert plan.summary()['floor_push'] == raw['floor_push']
    for field, value in [('floor_push', None), ('placement_mode', 'surface'),
                         ('floor_push', {'edge': '-x', 'direction': [-1., 0.]}),
                         ('floor_push', {'edge': '-x', 'direction': [1., .1]}),
                         ('place_pose', {'position': [-.3, .01, 1.], 'orientation': [0., 0., 0., 1.]})]:
        broken = copy.deepcopy(raw)
        broken[field] = value
        with pytest.raises(g.GoalPlanError):
            g.validate_pair_plan(broken, **kw)


def test_large_base_push_requires_geometry_and_middle_eighty_percent():
    import pytest
    raw = PairPlanTests().fixture('nextto')
    raw['task'] = 'inter_object_relation_satisfaction-nextto_large_scale'
    kwargs = dict(template_id='t', scene='s', rooms=['r'], relation='nextto', scale='large', activated_at=999, now=1000)
    with pytest.raises(g.GoalPlanError, match='audited base push'):
        g.validate_pair_plan(raw, **kwargs)
    raw['base_push'] = dict(side='-x', face='-x', direction=[1, 0, 0], side_inward=[1, 0], goal_gap_m=.1, nextto_bound_m=.2)
    p = g.validate_pair_plan(raw, **kwargs)
    assert p.goal_cell is None and p.summary()['base_push'] == raw['base_push']
    raw['base_push']['goal_gap_m'] = .195
    with pytest.raises(g.GoalPlanError, match='middle eighty percent'):
        g.validate_pair_plan(raw, **kwargs)
    raw['base_push']['goal_gap_m'] = .1
    raw['base_push']['direction'] = [1, 0, .1]
    with pytest.raises(g.GoalPlanError, match='direction'):
        g.validate_pair_plan(raw, **kwargs)
    raw['base_push']['direction'] = [1, 0, 0]
    raw['task'] = 'inter_object_relation_satisfaction-touching_large_scale'
    assert g.validate_pair_plan(raw, **{**kwargs, 'relation': 'touching'}).place is None


def test_base_push_gap_moves_along_audited_partner_side():
    import pytest
    goal = ([.6, -.2, .003], [0, 0, 0, 1])
    for fraction, desired in ((0, .02), (.5, .1), (1, .18)):
        changed, release, gap = g.vary_base_push_gap(goal, goal, inward=[1, 0], gap_max=.2, observed_gap=.12, fraction=fraction)
        assert gap == pytest.approx(desired)
        assert changed[0] == pytest.approx([.6+.12-desired, -.2, .003])
        assert changed == release


def test_large_surface_touching_requires_native_mode_and_goal_cell():
    import pytest
    raw = PairPlanTests().fixture('touching')
    raw.update(task='inter_object_relation_satisfaction-touching_large_scale', placement_mode='surface')
    kwargs = dict(template_id='t', scene='s', rooms=['r'], relation='touching', scale='large', activated_at=999, now=1000)
    with pytest.raises(g.GoalPlanError, match='fixed cell'):
        g.validate_pair_plan(raw, **kwargs)
    raw['goal_cell'] = [0, 0]
    p = g.validate_pair_plan(raw, **kwargs)
    assert p.placement_mode == 'surface' and p.base_push is None and p.place is None
    assert p.summary()['placement_mode'] == 'surface'
    raw['placement_mode'] = 'lean'
    with pytest.raises(g.GoalPlanError, match='unsupported pair placement mode'):
        g.validate_pair_plan(raw, **kwargs)


def test_drop_push_pins_pre_push_place_and_contact_endpoint_separately():
    import pytest
    raw = PairPlanTests().fixture('touching')
    raw.update(task='inter_object_relation_satisfaction-touching_large_scale', placement_mode='drop_push', goal_cell=[0, 0])
    kwargs = dict(template_id='t', scene='s', rooms=['r'], relation='touching', scale='large', activated_at=999, now=1000)
    with pytest.raises(g.GoalPlanError, match='pre-push place'):
        g.validate_pair_plan(raw, **kwargs)
    raw['place_pose'] = {'position': [0, -.15, .85], 'orientation': [0, 0, 0, 1]}
    p = g.validate_pair_plan(raw, **kwargs)
    assert p.place[0] == (0, -.15, .85) and p.release[0] == (0, 0, .81)
    raw['goal_cell'] = [.02, 0]
    with pytest.raises(g.GoalPlanError, match='contact endpoint'):
        g.validate_pair_plan(raw, **kwargs)


def test_fixed_sink_release_retains_original_level_reference():
    import math
    from dataclasses import replace
    raw=PairPlanTests().fixture('inside')
    raw.update(task='inter_object_relation_satisfaction-inside_large_scale',
               class_id='inter_object_relation_satisfaction-inside_large_scale_03',goal_cell=[0.,0.],
               inside_release_attitude={'tilt_deg':30.,'level_quat':[0.,0.,0.,1.],'source_summary_sha256':'b'*64})
    raw['release_pose']['orientation']=[math.sin(math.pi/12),0.,0.,math.cos(math.pi/12)]
    p=g.validate_pair_plan(raw,template_id='t',scene='s',rooms=['r'],relation='inside',scale='large',activated_at=999,now=1000)
    assert p.inside_release_attitude['tilt_deg']==30
    assert p.inside_release_attitude['level_quat']==[0,0,0,1]
    assert g.release_attitude_matches(p,p.release[1],.01)
    assert not g.release_attitude_matches(p,[0.,0.,0.,1.],.01)
    p.verify_settled(p.goal,p.partner)
    # Preserve the ordinary final-goal orientation check as well as uprightness.
    import pytest
    with pytest.raises(g.GoalPlanError):p.verify_settled((p.goal[0],p.release[1]),p.partner)
    for key,value in [('class_id','inter_object_relation_satisfaction-inside_large_scale_01'),('inside_release_attitude',{'tilt_deg':0.,'level_quat':[0.,0.,0.,1.],'source_summary_sha256':'b'*64})]:
        bad={**raw,key:value}
        with pytest.raises(g.GoalPlanError):g.validate_pair_plan(bad,template_id='t',scene='s',rooms=['r'],relation='inside',scale='large',activated_at=999,now=1000)
