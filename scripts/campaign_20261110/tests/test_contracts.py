"""Campaign safety contracts without simulator, cluster, network, or credentials."""

import copy
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from scripts.campaign_20261110 import contracts as c


def sha(value):
    return c.digest(value)


def obj(instance="target", role="target", category="apple", model="model_a", **kwargs):
    return c.ObjectRef(instance, role, category, model, sha([category, model]), ready=True, readiness_source="test-ready-manifest", **kwargs)


def template(split="eval_standard", objects=None, **kwargs):
    args = dict(template_id="t1", task="object_scaling", class_id="apple", split=split,
                scene="house_single_floor", room="kitchen_0", objects=tuple(objects or [obj()]),
                definition_sha256=sha("definition"))
    args.update(kwargs)
    return c.Template(**args)


def key(t=None):
    t = t or template()
    p = c.Perturbations({o.instance: c.ScaleState(o.default_scale, o.default_scale) for o in t.objects})
    return c.ValidationKey(t.fingerprint, sha("code"), sha("physics"), sha("maps"), p.scale_fingerprint)


class QuotaTests(unittest.TestCase):
    def test_totals_and_pending_decisions(self):
        q = c.task_quotas()
        self.assertEqual(len(q), 23)
        self.assertEqual(sum(x.standard_classes for x in q), 100)
        self.assertEqual(sum(x.model_classes for x in q), 99)
        self.assertEqual(sum(x.model_classes * x.model_instances_per_class for x in q), 891)
        self.assertEqual(sum(x.train_instances for x in q), 100000)
        self.assertEqual(sum(x.category_instances is None for x in q), 0)
        self.assertEqual(sum(x.category_instances for x in q), 109)
        pending = c.task_quotas(articulation_category_count=None, block_category_instances=None)
        self.assertEqual(sum(x.category_instances is None for x in pending), 5)
        balanced = c.task_quotas(articulation_category_count="balanced_four", block_category_instances=1)
        self.assertEqual(sum(x.category_instances for x in balanced), 115)

    def test_aliases_include_legacy_small_relation_names(self):
        for alias, expected in [
            ("object_scaling_01", "object_scaling"),
            ("inter_object_relation_satisfaction-inside_04", "inter_object_relation_satisfaction-inside_small_scale"),
            ("inter_object_relation_satisfaction-inside_large_scale_03_sinkvekaaf", "inter_object_relation_satisfaction-inside_large_scale"),
            ("configuration_matching-articulation_close_small_scale_01", "configuration_matching-articulation_close_small_scale"),
        ]:
            self.assertEqual(c.task_for_alias(alias), expected)
        with self.assertRaises(c.ContractError):
            c.task_for_alias("unknown")


class SplitTests(unittest.TestCase):
    def test_model_holdout_leaks_across_role_and_task(self):
        standard = template(objects=[obj(), obj("table", "support", "desk", "desk_a")])
        heldout = template("eval_model", [obj(category="desk", model="desk_a")],
                           template_id="t2", task="mental_rotation")
        errors = c.audit_splits([standard, heldout])
        self.assertTrue(any("eval_model models leak" in x for x in errors))

    def test_copying_and_renaming_asset_does_not_evade_holdout(self):
        standard = template()
        renamed = replace(obj(category="novel", model="renamed"), asset_sha256=standard.objects[0].asset_sha256)
        errors = c.audit_splits([standard, template("eval_model", [renamed], template_id="t2")])
        self.assertTrue(any("hashes leak" in x for x in errors))

    def test_category_holdout_cannot_reuse_any_seen_role_or_room(self):
        standard = template(objects=[obj(), obj("table", "support", "desk", "desk_a")])
        category = template("eval_category", [obj(category="desk", model="desk_new")], template_id="t2")
        errors = c.audit_splits([standard, category])
        self.assertTrue(any("categories leak" in x for x in errors))
        self.assertTrue(any("rooms leak" in x for x in errors))

    def test_sharing_within_holdout_and_model_categories_is_allowed(self):
        standard = template()
        model = template("eval_model", [obj(model="model_b")], template_id="t2")
        category = template("eval_category", [obj(category="novel", model="model_c")], template_id="t3", room="heldout_0")
        shared = replace(category, template_id="t4", task="mental_rotation")
        self.assertEqual(c.audit_splits([standard, model, category, shared]), [])

    def test_invalid_template_cannot_enter_manifest(self):
        with self.assertRaises(c.ContractError):
            template(scene="outside_scene")
        with self.assertRaises(c.ContractError):
            template(objects=[obj(), obj()])
        with self.assertRaises(c.ContractError):
            template(objects=[c.ObjectRef("target", "target", "apple", "a", sha("a"))])

    def test_structural_exemption_preserves_room_holdout(self):
        floor = obj("floor", "support", "floor", "floor_a", semantic_kind="floor")
        standard = template(objects=[obj(), floor])
        category = template("eval_category", [obj(category="novel", model="new"), floor], template_id="t2", room="heldout_0")
        self.assertEqual(c.audit_splits([standard, category]), [])
        errors = c.audit_splits([standard, replace(category, room=standard.room)])
        self.assertTrue(any("rooms leak" in x for x in errors))

    def test_second_task_room_is_part_of_room_holdout(self):
        standard = template(scene_setup={"task_rooms": ["kitchen_0", "doorway_0"]})
        category = template("eval_category", [obj(category="novel", model="new")], template_id="t2", room="doorway_0")
        self.assertTrue(any("rooms leak" in x for x in c.audit_splits([standard, category])))

    def test_tray_category_exception_does_not_waive_model_holdout(self):
        tray = obj("tray", "support", "tray", "tray_a", semantic_kind="tray")
        standard = template(objects=[obj(), tray])
        category = template("eval_category", [obj(category="novel", model="new"), replace(tray, model="tray_b", asset_sha256=sha("tray_b"))],
                            template_id="t2", room="heldout_0")
        self.assertEqual(c.audit_splits([standard, category]), [])
        model = template("eval_model", [obj(model="new"), tray], template_id="t3")
        self.assertTrue(any("eval_model models leak" in x for x in c.audit_splits([standard, model])))


class PerturbationTests(unittest.TestCase):
    def test_candidates_use_actual_default_and_height_only(self):
        default = (0.3, 0.5, 0.7)
        a = c.scale_candidates(default, seed=173)
        self.assertEqual(a, c.scale_candidates(default, seed=173))
        self.assertEqual(len(a), 2)
        for state in a:
            self.assertTrue(all(0.8 <= x / d <= 1.2 for x, d in zip(state.applied, default)))
        for state in c.scale_candidates(default, seed=173, height_only=True):
            self.assertEqual(state.applied[:2], default[:2])

    def test_eval_rejects_every_perturbation_kind(self):
        cases = [c.Perturbations({"target": c.ScaleState((1, 1, 1), (0.9, 0.9, 0.9))}),
                 c.Perturbations(removed_objects=("chair_1",)),
                 c.Perturbations(lighting={"skybox_color": [0.4, 0.5, 0.6]})]
        for split in ("eval_standard", "eval_model", "eval_category"):
            for perturb in cases:
                with self.assertRaises(c.ContractError):
                    perturb.validate(split, ["target"])

    def test_task_relevant_objects_cannot_be_parked(self):
        with self.assertRaises(c.ContractError):
            c.Perturbations(removed_objects=("target",)).validate("train", ["target"])

    def test_scale_constraints(self):
        for state in [((1, 1, 1), (1, 0.9, 1), "uniform"),
                      ((1, 1, 1), (0.9, 1, 1.1), "height"),
                      ((1, 1, 1), (1.3, 1.3, 1.3), "uniform"),
                      ((1, 1, 1), (float("nan"), 1, 1), "uniform")]:
            with self.assertRaises(c.ContractError):
                c.ScaleState(*state)

    def test_cache_keys_separate_scale_height_asset_and_joint(self):
        state = c.ScaleState((1, 1, 1), (1, 1, 1))
        kwargs = dict(support_height_m=0.75, robot_sha256=sha("robot"), builder_sha256=sha("builder"), physics_sha256=sha("physics"))
        base = c.scale_cache_key(obj(), state, **kwargs)
        variants = [c.scale_cache_key(obj(), c.ScaleState((1, 1, 1), (0.9, 0.9, 0.9)), **kwargs),
                    c.scale_cache_key(obj(), state, **dict(kwargs, support_height_m=0.76)),
                    c.scale_cache_key(obj(model="new"), state, **kwargs),
                    c.scale_cache_key(obj(joint="door_joint"), state, **kwargs)]
        self.assertEqual(len(set([base] + variants)), 5)


class ValidationTests(unittest.TestCase):
    def evidence(self, kind="demo", **kwargs):
        args = dict(key=key(), kind=kind, success=True, qa_passed=True, created_at=101,
                    report_path="not-used", report_sha256=sha("report"), demo_sha256=sha("demo"),
                    instance_id="fresh_1", restoration_verified=True)
        args.update(kwargs)
        return c.ValidationEvidence(**args)

    def test_single_success_and_exact_replay_are_sufficient(self):
        demo = self.evidence()
        replay = self.evidence("open_loop_replay", created_at=102)
        self.assertEqual(c.require_validation(key(), [demo, replay], campaign_started_at=100,
                                             verify_artifacts=False), (demo, replay))

    def test_no_stale_mismatched_failed_or_unrestored_evidence(self):
        demo = self.evidence()
        replay = self.evidence("open_loop_replay", created_at=102)
        cases = [[], [demo], [demo, replace(replay, success=False)],
                 [demo, replace(replay, qa_passed=False)],
                 [demo, replace(replay, restoration_verified=False)],
                 [demo, replace(replay, demo_sha256=sha("other demo"))],
                 [demo, replace(replay, key=replace(key(), scale_sha256=sha("new scale")))],
                 [replace(demo, created_at=99), replay]]
        for evidence in cases:
            with self.assertRaises(c.ContractError):
                c.require_validation(key(), evidence, campaign_started_at=100, verify_artifacts=False)

    def test_artifact_hash_is_verified_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "report.json"
            path.write_text('{"success":true}')
            demo = self.evidence(report_path=str(path), report_sha256=c.file_digest(path))
            replay = replace(demo, kind="open_loop_replay", created_at=102)
            c.require_validation(key(), [demo, replay], campaign_started_at=100)
            path.write_text('{"success":false}')
            with self.assertRaises(c.ContractError):
                c.require_validation(key(), [demo, replay], campaign_started_at=100)


class GroupTests(unittest.TestCase):
    def member(self, iid, **kwargs):
        args = dict(instance_id=iid, campaign_id="new", template_sha256=sha("template"), goal_sha256=sha("goal"),
                    start_grasp_sha256=sha(iid), perturbations=c.Perturbations(), created_at=101, demo_success=True)
        args.update(kwargs)
        return c.GroupMember(**args)

    def allocate(self, members, **kwargs):
        return c.allocate_goal_group(members, campaign_id="new", campaign_started_at=100,
                                     standard_eval_requested=True, **kwargs)

    def test_singleton_is_training_but_two_distinct_successes_unlock_eval(self):
        self.assertEqual(self.allocate([self.member("b")]), {"b": "train"})
        members = [self.member("b"), self.member("a"), self.member("c")]
        self.assertEqual(self.allocate(members), {"b": "train", "a": "eval_standard", "c": "train"})

    def test_cannot_promote_duplicates_old_instances_or_changed_goals(self):
        a = self.member("a")
        b = self.member("b")
        for invalid in [replace(b, goal_sha256=sha("other goal")), replace(b, start_grasp_sha256=a.start_grasp_sha256),
                        replace(b, campaign_id="old"), replace(b, created_at=99), replace(b, demo_success=False)]:
            with self.assertRaises(c.ContractError):
                self.allocate([a, invalid])

    def test_perturbed_group_cannot_produce_eval_but_can_produce_train(self):
        p = c.Perturbations(lighting={"fixture": 1.2})
        members = [self.member("a", perturbations=p), self.member("b", perturbations=p)]
        with self.assertRaises(c.ContractError):
            self.allocate(members)
        self.assertEqual(c.allocate_goal_group(members, campaign_id="new", campaign_started_at=100,
                                              standard_eval_requested=False), {"a": "train", "b": "train"})

    def test_limit_and_timeout_are_bounded(self):
        with self.assertRaises(c.ContractError):
            self.allocate([self.member(str(i)) for i in range(4)])
        self.assertEqual(c.group_budget_seconds([10]), 300)
        self.assertEqual(c.group_budget_seconds([300]), 900)
        self.assertEqual(c.group_budget_seconds([100000]), 1800)


class TroTests(unittest.TestCase):
    def tro(self):
        return {"target": {"root_link": {"pos": [0, 0, 1]}}, "robot_poses": {"robot": []},
                "ispatialgym": {"object_bindings": {"target": {"scene_object": "apple"}}, "pose_targets": {}}}

    def test_roundtrip_preserves_tro_and_default_scale_and_scene_setup(self):
        original = self.tro()
        before = copy.deepcopy(original)
        t = template(scene_setup={"replacements": {"desk_original": "desk_new"}, "removed_chairs": ["chair_1"]})
        scales = c.Perturbations({"target": c.ScaleState((1, 1, 1), (1, 1, 1))})
        tro = c.attach_tro_metadata(original, campaign_id="new", split="eval_standard", template=t,
                                    created_at=101, seed=2, perturbations=scales, validation_key=key(t), goal_group_id="g1")
        self.assertEqual(original, before)
        self.assertEqual(tro["target"], original["target"])
        restored = c.restoration_plan(json.loads(json.dumps(tro)))
        self.assertEqual(restored["fingerprint"], scales.fingerprint)
        self.assertEqual(restored["scene_setup"], t.scene_setup)
        self.assertLess(restored["order"].index("scales"), restored["order"].index("object_states"))

    def test_live_object_state_serialization_remains_the_tro_writers_job(self):
        class TensorLike:
            def tolist(self):
                return [0, 0, 1]
        original = self.tro()
        original["target"]["root_link"]["pos"] = TensorLike()
        t = template()
        p = c.Perturbations({"target": c.ScaleState((1, 1, 1), (1, 1, 1))})
        result = c.attach_tro_metadata(original, campaign_id="new", split="eval_standard", template=t,
                                       created_at=101, seed=2, perturbations=p, validation_key=key(t))
        self.assertEqual(result["target"]["root_link"]["pos"].tolist(), [0, 0, 1])

    def test_missing_scale_or_wrong_template_evidence_rejected(self):
        t = template()
        for p, k in [(c.Perturbations(), key(t)),
                     (c.Perturbations({"target": c.ScaleState((1, 1, 1), (1, 1, 1))}),
                      replace(key(t), template_sha256=sha("other")))]:
            with self.assertRaises(c.ContractError):
                c.attach_tro_metadata(self.tro(), campaign_id="new", split="eval_standard", template=t,
                                      created_at=101, seed=2, perturbations=p, validation_key=k)

    def test_parser_contract_rejects_forged_or_incomplete_campaign_block(self):
        t = template()
        p = c.Perturbations({"target": c.ScaleState((1, 1, 1), (1, 1, 1))})
        tro = c.attach_tro_metadata(self.tro(), campaign_id="new", split="eval_standard", template=t,
                                    created_at=101, seed=2, perturbations=p, validation_key=key(t))
        valid = tro["ispatialgym"]["campaign"]
        for path, value in [(("seed",), True), (("created_at",), float("nan")), (("split",), "evaluation"),
                            (("template_sha256",), sha("wrong")), (("validation_key", "scale_sha256"), sha("wrong")),
                            (("perturbations", "scales"), {}), (("restoration_order",), [])]:
            bad = copy.deepcopy(valid)
            target = bad
            for part in path[:-1]:
                target = target[part]
            target[path[-1]] = value
            with self.assertRaises(c.ContractError):
                c.validate_campaign_block(bad, object_bindings=tro["ispatialgym"]["object_bindings"])

    def test_parser_only_exempts_explicit_structural_bindings(self):
        t = template()
        p = c.Perturbations({"target": c.ScaleState((1, 1, 1), (1, 1, 1))})
        tro = c.attach_tro_metadata(self.tro(), campaign_id="new", split="eval_standard", template=t,
                                    created_at=101, seed=2, perturbations=p, validation_key=key(t))
        block = tro["ispatialgym"]["campaign"]
        c.validate_campaign_block(block, object_bindings={"target": {}, "floor.n.01_1": {}})
        c.validate_campaign_block(block, object_bindings={"target": {}, "structural": {"category": "walls"}})
        for misleading in ("floor_lamp.n.01_1", "wall_clock.n.01_1", "floor.n.01_1_extra"):
            with self.assertRaises(c.ContractError):
                c.validate_campaign_block(block, object_bindings={"target": {}, misleading: {}})


if __name__ == "__main__":
    unittest.main()
