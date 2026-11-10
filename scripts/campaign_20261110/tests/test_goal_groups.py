import json
import os
import tempfile
import time
import unittest
from dataclasses import replace
from pathlib import Path

from scripts.campaign_20261110 import contracts as c
from scripts.campaign_20261110 import goal_groups as g


class GoalGroupControllerTests(unittest.TestCase):
    def fixture(self, root):
        job = root / "out/canary/apple/g9000000-9000000"
        instance_dir = job / "inst/apple/runtime_template/instances"
        instance_dir.mkdir(parents=True)
        iid = "fresh_9000000"
        tro = {"ispatialgym": {"instance_id": iid, "object_bindings": {"apple.n.01_1": {"category": "apple", "model": "a"}},
                               "scene": {"model": "house_single_floor", "task_rooms": ["living_room_0"]}}}
        tro_path = instance_dir / (iid + "-tro_state.json")
        tro_path.write_text(json.dumps(tro))
        (instance_dir / (iid + ".oracle.json")).write_text(json.dumps({"instance_id": iid, "tro_sha256": c.file_digest(tro_path)}))
        (instance_dir / (iid + ".summary.json")).write_text(json.dumps({"instance_id": iid,
            "goal_realised": {"position": [1, 2, 3], "orientation": [0, 0, 0, 1]}, "base_pose2d": [1, 2, 0]}))
        (instance_dir.parent / "template_meta.json").write_text(json.dumps({"template_id": "runtime_template", "scene": "house_single_floor"}))
        demo = job / "demo" / iid
        demo.mkdir(parents=True)
        (demo / "meta.json").write_text(json.dumps({"instance_id": iid, "success": True, "status": "success", "mode": "oracle",
                                                    "qa": {"enabled": True, "hard_triggered": False, "metrics": {}}}))
        (demo / "demo.hdf5").write_bytes(b"fixture")
        (job / "job.env").write_text(f"ISGF_ROOT={root}\nSTART=9000000\n")
        (job / "results.jsonl").write_text(json.dumps({"instance": iid, "success": True}) + "\n")
        ledger = root / "manifests/canary.tsv"
        ledger.parent.mkdir()
        ledger.write_text(f"job_dir\tkey\tjobid\tstart\tn\n{job}\tobject_scaling-apple\t1\t9000000\t1\n")
        obj = c.ObjectRef("apple.n.01_1", "target", "apple", "a", c.digest("asset"), ready=True, readiness_source="ready fixture")
        template = c.Template("standard_apple", "object_scaling", "apple", "eval_standard", "house_single_floor", "living_room_0",
                              (obj,), c.digest("definition"), {"runtime_template_id": "runtime_template"})
        code = root / "code/snapshot"
        (code / "scripts/trainfast").mkdir(parents=True)
        (code / "scripts/trainfast/submit_fast.py").write_text("# fixture")
        (code / ".snapshot_commit").write_text("fixturecommit\n")
        return job, ledger, tro_path, template, code

    def test_fresh_anchor_prepares_new_group_without_copying_instance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            job, ledger, _, template, code = self.fixture(root)
            anchor = g.successful_anchor(job, campaign_root=root, ledger=ledger, campaign_started_at=time.time() - 10)
            out = root / "groups/g1"
            result = g.prepare_group(anchor=anchor, template=template, campaign_id="new", group_id="g1", output=out,
                                     code_sha256=c.file_digest(code / ".snapshot_commit"), physics_sha256=c.digest("physics"),
                                     data_sha256=c.digest("maps"), magnitude_range=[0.05, 0.10])
            self.assertEqual(result["status"], "ready_for_group_validation")
            self.assertEqual(len(list(out.glob("*-tro_state.json"))), 0)
            self.assertFalse(g.read_json(out / "source.json")["anchor_is_final_member"])
            cmd = g.submission_command(out, snapshot=code, campaign_root=root, v2_root=root / "maps")
            self.assertIn("10000000", cmd)
            self.assertIn("ISGF_COLLECT_EXTRA=--no-goal-override", cmd)
            self.assertIn("EVAL_INDEX=" + str(out / "empty_eval_index.json"), cmd)
            with self.assertRaises(c.ContractError):
                g.submission_command(out, snapshot=code, campaign_root=root, v2_root=root / "maps", extra_env=["ISGF_COLLECT_EXTRA="])
            (code / ".snapshot_commit").write_text("changedcode")
            with self.assertRaises(c.ContractError):
                g.submission_command(out, snapshot=code, campaign_root=root, v2_root=root / "maps")

    def test_historical_or_unallocated_anchor_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            job, ledger, tro, _, _ = self.fixture(root)
            started = time.time() - 10
            os.utime(tro, (started - 1, started - 1))
            with self.assertRaises(c.ContractError):
                g.successful_anchor(job, campaign_root=root, ledger=ledger, campaign_started_at=started)
            os.utime(tro, None)
            ledger.write_text("job_dir\tkey\tjobid\tstart\tn\n")
            with self.assertRaises(c.ContractError):
                g.successful_anchor(job, campaign_root=root, ledger=ledger, campaign_started_at=started)

    def test_anchor_qa_must_be_enabled_and_passing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            job, ledger, _, _, _ = self.fixture(root)
            meta_path = next((job / "demo").glob("*/meta.json"))
            meta = g.read_json(meta_path)
            meta["qa"]["enabled"] = False
            meta_path.write_text(json.dumps(meta))
            with self.assertRaises(c.ContractError):
                g.successful_anchor(job, campaign_root=root, ledger=ledger, campaign_started_at=time.time() - 10)

    def test_tray_author_uses_fresh_validated_goal_region_and_fixed_destination(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            job, ledger, tro_path, template, code = self.fixture(root)
            anchor = g.successful_anchor(job, campaign_root=root, ledger=ledger, campaign_started_at=time.time() - 10)
            source = c.ObjectRef("tray.n.01_1", "support", "tray", "source", c.digest("source"), semantic_kind="tray",
                                 ready=True, readiness_source="ready fixture")
            dest = replace(source, instance="tray.n.01_2", model="dest", asset_sha256=c.digest("dest"))
            template = replace(template, task="disembedding", objects=template.objects + (source, dest))
            anchor["template_meta"]["bindings"] = [{"name": "apple.n.01_1", "role": "target"},
                {"name": source.instance, "role": "source"}, {"name": dest.instance, "role": "destination"}]
            anchor["tro"]["ispatialgym"]["object_bindings"].update({x.instance: {"category": x.category, "model": x.model} for x in (source, dest)})
            anchor["planned_start_poses"] = {source.instance: [[0, 2, .7], [0, 0, 0, 1]], dest.instance: [[1, 2, .7], [0, 0, 0, 1]]}
            region = [{"position": [1.02, 2.01, .8], "orientation": [0, 0, 0, 1]}]
            anchor["summary"]["campaign_goal_space"] = {"target_poses": region}
            out = root / "groups/tray"
            g.prepare_group(anchor=anchor, template=template, campaign_id="new", group_id="tray", output=out,
                            code_sha256=c.file_digest(code / ".snapshot_commit"), physics_sha256=c.digest("physics"),
                            data_sha256=c.digest("maps"), magnitude_range=[.3, .45])
            plan = g.read_json(out / "goal_plan.json")
            self.assertEqual(plan["schema"], "isg-tray-goal-plan-v1")
            self.assertEqual(plan["goal_pose"], region[0])
            self.assertEqual(plan["destination_pose"]["position"], [1, 2, .7])
            self.assertEqual(plan["crowdedness_range"], [.3, .45])
            anchor["summary"].pop("campaign_goal_space")
            with self.assertRaises(c.ContractError):
                g.prepare_group(anchor=anchor, template=template, campaign_id="new", group_id="bad", output=root / "groups/bad",
                                code_sha256=c.file_digest(code / ".snapshot_commit"), physics_sha256=c.digest("physics"),
                                data_sha256=c.digest("maps"), magnitude_range=[.3, .45])

    def test_pair_author_preserves_partner_and_varies_nextto_gap_before_new_starts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            job, ledger, _, template, code = self.fixture(root)
            anchor = g.successful_anchor(job, campaign_root=root, ledger=ledger, campaign_started_at=time.time() - 10)
            partner = c.ObjectRef("partner.n.01_1", "partner", "platter", "p", c.digest("partner"),
                                  ready=True, readiness_source="ready fixture")
            template = replace(template, task="inter_object_relation_satisfaction-nextto_small_scale", objects=template.objects + (partner,))
            anchor["template_meta"]["bindings"] = [{"name": "apple.n.01_1", "role": "target"}, {"name": partner.instance, "role": "partner"}]
            bindings = anchor["tro"]["ispatialgym"]["object_bindings"]
            bindings["apple.n.01_1"]["scene_object_name"] = "target_apple_a_0"
            bindings[partner.instance] = {"category": partner.category, "model": partner.model}
            anchor["planned_start_poses"] = {partner.instance: [[0, 0, .7], [0, 0, 0, 1]]}
            anchor["summary"].update({"goal_realised": {"position": [0, .2, .81], "orientation": [0, 0, 0, 1]},
                "goal_check": {"settled_poses": {"target_apple_a_0": [[0, .2, .8], [0, 0, 0, 1]]}},
                "nextto_gap_m": .08, "pair": {"g_NT": .1}, "manipulation_stance": {"base_pose2d": [0, 0, 0]}})
            out = root / "groups/pair"
            g.prepare_group(anchor=anchor, template=template, campaign_id="new", group_id="pair", output=out,
                            code_sha256=c.file_digest(code / ".snapshot_commit"), physics_sha256=c.digest("physics"),
                            data_sha256=c.digest("maps"), magnitude_range=[0, 1])
            plan = g.read_json(out / "goal_plan.json")
            self.assertEqual(plan["schema"], "isg-pair-goal-plan-v1")
            self.assertTrue(.01 <= plan["nextto_gap_proposal_m"] <= .09)
            self.assertEqual(plan["partner_pose"]["position"], [0, 0, .7])
            self.assertAlmostEqual(plan["goal_pose"]["position"][1], .2 + plan["nextto_gap_proposal_m"] - .08)


if __name__ == "__main__":
    unittest.main()
