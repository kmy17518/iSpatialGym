import copy
import json
import os
import random
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from ispatialgym.sampling import campaign_held_goals as g


def plan(**kwargs):
    raw = {"schema": g.SCHEMA, "campaign_id": "campaign", "group_id": "group1", "split": "eval_standard",
           "task": "object_scaling", "template_id": "template", "scene": "scene", "room": "room",
           "goal_pose": {"position": [0, 1, 2], "orientation": [0, 0, 0, 1]}, "base_pose2d": [0, 0, 1],
           "magnitude_range": [0.05, 0.1], "created_at": 100, "deadline_at": 1000, "max_members": 3}
    raw.update(kwargs)
    return raw


def validate(raw):
    return g.validate_goal_plan(raw, task="object_scaling", template_id="template", scene="scene", rooms=["room"], now=500)


class GoalPlanTests(unittest.TestCase):
    def test_fixed_goal_identity_and_bucket_policy(self):
        p = validate(plan())
        self.assertEqual(p.position, (0, 1, 2))
        self.assertTrue(p.summary()["bucket_strict"])
        self.assertFalse(validate(plan(split="train")).summary()["bucket_strict"])
        self.assertNotEqual(validate(plan(group_id="group2")).sha256, p.sha256)

    def test_plan_rejects_expiry_wrong_identity_bad_quaternion_or_bucket(self):
        cases = [plan(task="mental_rotation"), plan(template_id="other"), plan(room="other"), plan(deadline_at=499),
                 plan(goal_pose={"position": [0, 0, 0], "orientation": [0, 0, 0, 2]}),
                 plan(magnitude_range=[0.01, 0.1]), plan(max_members=4), plan(split=None)]
        for raw in cases:
            with self.assertRaises(g.GoalPlanError):
                validate(raw)

    def test_loader_is_opt_in_and_rejects_replaced_plan(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(g.load_goal_plan(task="object_scaling", template_id="template", scene="scene", rooms=["room"]))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "plan.json"
            raw = plan(created_at=time.time() - 1, deadline_at=time.time() + 100)
            path.write_text(json.dumps(raw))
            with patch.dict(os.environ, {g.ENV: str(path)}):
                original = g.load_goal_plan(task="object_scaling", template_id="template", scene="scene", rooms=["room"])
                raw["group_id"] = "replaced"
                path.write_text(json.dumps(raw))
                with self.assertRaises(g.GoalPlanError):
                    g.load_goal_plan(task="object_scaling", template_id="template", scene="scene", rooms=["room"],
                                     expected_sha256=original.sha256)

    def test_conditional_starts_preserve_goal_and_range(self):
        rng = random.Random(8)
        starts = [g.draw_start_depth(rng, 0.5, [0.25, 0.8], [0.05, 0.1]) for _ in range(1000)]
        self.assertTrue(all(0.05 - 1e-12 <= abs(x - 0.5) <= 0.1 + 1e-12 for x in starts))
        self.assertTrue(any(x < 0.5 for x in starts) and any(x > 0.5 for x in starts))
        self.assertGreater(len(set(starts)), 990)

    def test_direction_band_and_impossible_bucket(self):
        rng = random.Random(2)
        for direction, sign in [("toward", 1), ("away", -1)]:
            for _ in range(50):
                start = g.draw_start_depth(rng, 0.5, [0.25, 0.8], [0.05, 0.1], direction=direction)
                self.assertGreater((start - 0.5) * sign, 0)
        with self.assertRaises(g.GoalPlanError):
            g.draw_start_depth(rng, 0.5, [0.46, 0.54], [0.05, 0.1])
        # A soft train scheduler can fall back from an impossible preferred bucket.
        with self.assertRaises(g.GoalPlanError):
            g.draw_start_depth(rng, 0.5, [0.4, 0.6], [0.15, 0.3])
        self.assertTrue(0.4 <= g.draw_start_depth(rng, 0.5, [0.4, 0.6], [0.05, 0.3]) <= 0.6)


if __name__ == "__main__":
    unittest.main()
