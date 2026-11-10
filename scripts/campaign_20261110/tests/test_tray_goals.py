import copy
import unittest

from ispatialgym.sampling import campaign_tray_goals as g


class TrayGoalsTests(unittest.TestCase):
    def fixture(self):
        return {"schema": g.SCHEMA, "task": "disembedding", "campaign_id": "fresh", "group_id": "tray1",
                "split": "eval_standard", "template_id": "t1", "scene": "s", "room": "r",
                "contract_template_sha256": "a" * 64, "target_name": "target.n.01_1", "source_name": "tray.n.01_1",
                "destination_name": "tray.n.01_2", "goal_pose": {"position": [1, 2, .8], "orientation": [0, 0, 0, 1]},
                "source_pose": {"position": [0, 2, .7], "orientation": [0, 0, 0, 1]},
                "destination_pose": {"position": [1, 2, .7], "orientation": [0, 0, 0, 1]},
                "crowdedness_range": [.3, .45], "created_at": 100, "budget_seconds": 900, "max_members": 3}

    def parse(self, raw, now=1000):
        return g.validate_tray_plan(raw, template_id="t1", scene="s", rooms=["r"], activated_at=999, now=now)

    def test_eval_coverage_boundaries_and_soft_training(self):
        raw = self.fixture()
        plan = self.parse(raw)
        self.assertTrue(plan.accepts_coverage(.3))
        self.assertFalse(plan.accepts_coverage(.45))
        self.assertFalse(plan.accepts_coverage(.299))
        raw["split"] = "train"
        self.assertTrue(self.parse(raw).accepts_coverage(.2))
        raw["split"], raw["crowdedness_range"] = "eval_standard", [.45, 1]
        self.assertTrue(self.parse(raw).accepts_coverage(1))

    def test_goal_identity_includes_destination_and_rejects_bad_pose(self):
        raw = self.fixture()
        plan = self.parse(raw)
        self.assertEqual(set(plan.summary()["requested_goal"]), {"target", "destination"})
        changed = copy.deepcopy(raw)
        changed["destination_pose"]["position"][0] += .1
        self.assertNotEqual(plan.sha256, self.parse(changed).sha256)
        raw["goal_pose"]["orientation"] = [0, 0, 0, 2]
        with self.assertRaises(g.GoalPlanError):
            self.parse(raw)

    def test_queue_wait_excluded_but_active_deadline_strict(self):
        self.parse(self.fixture(), now=1898.99)
        with self.assertRaises(g.GoalGroupTimeout):
            self.parse(self.fixture(), now=1899)

    def test_geometry_errors_are_sign_invariant_and_coverage_is_area(self):
        self.assertEqual(g.pose_error(([0, 0, 0], [0, 0, 0, 1]), ([0, 0, 0], [0, 0, 0, -1])), (0, 0))
        self.assertAlmostEqual(g.polygon_area([[0, 0], [2, 0], [2, 3], [0, 3]]), 6)
        self.assertAlmostEqual(g.pose_error(([0, 0, 0], [0, 0, 0, 1]), ([.002, 0, 0], [0, 0, 0, 1]))[0], .002)


if __name__ == "__main__":
    unittest.main()
