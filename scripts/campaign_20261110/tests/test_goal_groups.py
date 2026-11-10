import json
import os
import tempfile
import time
import unittest
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


if __name__ == "__main__":
    unittest.main()
