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

    def test_unsupported_contact_plan_and_deadline_reject(self):
        with self.assertRaises(g.GoalPlanError):
            self.parse("touching")
        with self.assertRaises(g.GoalGroupTimeout):
            self.parse(now=1299)

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
