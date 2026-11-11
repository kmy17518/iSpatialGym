import math
import unittest

from ispatialgym.sampling import campaign_surface_goals as s


def raw(**kwargs):
    p = {"schema": s.SCHEMA, "campaign_id": "new", "group_id": "g1", "split": "eval_standard",
         "task": "alignment-world_vertical", "template_id": "wall_template", "scene": "scene", "room": "room",
         "contract_template_sha256": "a" * 64, "goal_pose": {"position": [0, 1, 2], "orientation": [0, 0, 0, 1]},
         "magnitude_range_deg": [15, 20], "created_at": 100, "budget_seconds": 300, "max_members": 3}
    p.update(kwargs)
    return p


def validate(p, now=210):
    return s.validate_surface_plan(p, template_id="wall_template", scene="scene", rooms=["room"], activated_at=200, now=now)


class SurfacePlanTests(unittest.TestCase):
    def test_plan_preserves_goal_but_does_not_pin_robot_base(self):
        p = validate(raw())
        self.assertEqual(p.deadline_at, 500)
        self.assertEqual(p.summary()["requested_goal"]["position"], [0, 1, 2])
        self.assertNotIn("base_pose2d", p.summary())
        self.assertTrue(p.summary()["bucket_strict"])

    def test_identity_and_time_checks(self):
        for p in (raw(template_id="different"), raw(room="different"), raw(task="unrelated"),
                  raw(contract_template_sha256=""), raw(max_members=6), raw(magnitude_range_deg=[20, 15])):
            with self.assertRaises(s.GoalPlanError):
                validate(p)
        with self.assertRaises(s.GoalGroupTimeout):
            validate(raw(), now=501)
        self.assertEqual(validate(raw(task="alignment-axial")).task, "alignment-axial")

    def test_optional_anchor_guides_proposals_without_pinning_the_base(self):
        p = validate(raw(anchor_base_pose2d=[1., 2., .3]))
        self.assertEqual(p.anchor_base_pose2d, (1., 2., .3))
        self.assertNotIn('base_pose2d', p.summary())
        with self.assertRaises(s.GoalPlanError):
            validate(raw(anchor_base_pose2d=[1., float('nan'), .3]))

    def test_training_buckets_do_not_relax_physical_bounds_or_block_sampling(self):
        allowed = tuple(math.radians(v) for v in [15, 25])
        train = validate(raw(split="train", magnitude_range_deg=[30, 40]))
        self.assertEqual(s.deviation_range(train, allowed), allowed)
        strict = validate(raw(magnitude_range_deg=[30, 40]))
        with self.assertRaises(s.GoalPlanError):
            s.deviation_range(strict, allowed)
        preferred = validate(raw(split="train", magnitude_range_deg=[18, 22]))
        self.assertEqual(s.deviation_range(preferred, allowed), tuple(math.radians(v) for v in [18, 22]))
        self.assertEqual(s.deviation_range(preferred, allowed, fallback=True), allowed)


if __name__ == "__main__":
    unittest.main()
