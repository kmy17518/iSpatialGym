import json
import math
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

    def test_all_collectors_can_start_after_full_sampling_window(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            job, ledger, _, template, code = self.fixture(root)
            anchor = g.successful_anchor(job, campaign_root=root, ledger=ledger, campaign_started_at=time.time()-10)
            out = root / 'groups/window'
            g.prepare_group(anchor=anchor, template=template, campaign_id='new', group_id='window', output=out,
                code_sha256=c.file_digest(code/'.snapshot_commit'), physics_sha256=c.digest('physics'),
                data_sha256=c.digest('maps'), magnitude_range=[.05, .10])
            # Exercise only the launch budget policy. Native block geometry and
            # semantics have their own tests; the observed failure was no oracle
            # start because the deadline preceded the ready-instance marker.
            plan = g.read_json(out/'goal_plan.json')
            for task in ('configuration_matching-block_design', 'configuration_matching-attachment',
                         'inter_object_relation_satisfaction-inside', 'object_scaling'):
                with self.subTest(task=task):
                    plan['task'] = task
                    c.write_json_atomic(out/'goal_plan.json', plan)
                    cmd = g.submission_command(out, snapshot=code, campaign_root=root, v2_root=root/'maps')
                    sampling = int(cmd[cmd.index('--sample-limit-s')+1])
                    allocation = int(cmd[cmd.index('--time-s')+1])
                    collector_deadline = allocation - 3600 - 420
                    self.assertGreater(collector_deadline, sampling)
                    self.assertGreater(collector_deadline, 1977)  # block18423951
                    self.assertGreater(collector_deadline, 806)  # attachment1409

    def test_required_native_visibility_survives_group_dispatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            job, ledger, _, template, code = self.fixture(root)
            anchor = g.successful_anchor(job, campaign_root=root, ledger=ledger, campaign_started_at=time.time()-10)
            out = root/'groups/visibility'
            g.prepare_group(anchor=anchor, template=template, campaign_id='new', group_id='visibility', output=out,
                code_sha256=c.file_digest(code/'.snapshot_commit'), physics_sha256=c.digest('physics'),
                data_sha256=c.digest('maps'), magnitude_range=[.05, .10])
            args = dict(snapshot=code, campaign_root=root, v2_root=root/'maps')
            self.assertNotIn('ISGF_GOAL_OCCLUSION=on', g.submission_command(out, **args))
            config = g.read_json(out/'campaign_config.json')
            config['template'].update(class_id='inter_object_relation_satisfaction-inside_02',
                                      task='inter_object_relation_satisfaction-inside_small_scale')
            config['template']['scene_setup']['recovery_options'] = {'native_goal_occlusion_screen': True}
            c.write_json_atomic(out/'campaign_config.json', config)
            command = g.submission_command(out, **args)
            for name in ('ISGF_GOAL_OCCLUSION', 'ISPATIALGYM_GOAL_OCCLUSION'):
                self.assertIn(name+'=on', command)
                with self.assertRaises(c.ContractError):
                    g.submission_command(out, **args, extra_env=[name+'=off'])
                self.assertIn(name+'=on', g.submission_command(out, **args, extra_env=[name+'=on']))
            self.assertIn('ISPATIALGYM_FAST_SAMPLER_CAMERAS=0', command)
            config['template'].update(class_id='inter_object_relation_satisfaction-under_large_scale_02',
                                      task='inter_object_relation_satisfaction-under_large_scale')
            c.write_json_atomic(out/'campaign_config.json', config)
            self.assertIn('ISPATIALGYM_FAST_SAMPLER_CAMERAS=0', g.submission_command(out, **args))
            with self.assertRaises(c.ContractError):
                g.submission_command(out, **args, extra_env=['ISPATIALGYM_FAST_SAMPLER_CAMERAS=1'])

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
            anchor["summary"]["base_pose2d"] = [0., 1., math.pi / 2]
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
            anchor["planned_goal_poses"] = {"apple.n.01_1": [[1.02, 2.01, .8], [0, 0, 0, 1]]}
            repeated = root / "groups/reused"
            g.prepare_group(anchor=anchor, template=template, campaign_id="new", group_id="reused", output=repeated,
                            code_sha256=c.file_digest(code / ".snapshot_commit"), physics_sha256=c.digest("physics"),
                            data_sha256=c.digest("maps"), magnitude_range=[.3, .45], reuse_anchor_goal=True)
            exact = g.read_json(repeated / "goal_plan.json")
            self.assertEqual(exact["goal_pose"], region[0])
            self.assertEqual(exact["destination_pose"], plan["destination_pose"])
            self.assertEqual(exact["source_pose"], plan["source_pose"])
            self.assertEqual(exact["goal_selection"], "fresh_anchor")
            self.assertFalse(exact["source_instance_is_member"])

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
            reused = root / "groups/pair_reused"
            g.prepare_group(anchor=anchor, template=template, campaign_id="new", group_id="pair_reused", output=reused,
                            code_sha256=c.file_digest(code / ".snapshot_commit"), physics_sha256=c.digest("physics"),
                            data_sha256=c.digest("maps"), magnitude_range=[0, 1], reuse_anchor_goal=True)
            exact = g.read_json(reused / "goal_plan.json")
            self.assertEqual(exact["goal_pose"]["position"], [0, .2, .8])
            self.assertEqual(exact["release_pose"]["position"], [0, .2, .81])
            self.assertEqual(exact["partner_pose"]["position"], [0, 0, .7])
            self.assertNotIn("nextto_gap_proposal_m", exact)
            self.assertEqual(exact["goal_selection"], "fresh_anchor")



if __name__ == "__main__":
    unittest.main()


def test_runtime_bddl_binding_normalizes_legacy_spelling_and_measure_but_not_goal(tmp_path):
    source = tmp_path / "source"
    runtime = tmp_path / "runtime"
    source.mkdir()
    runtime.mkdir()
    old = """(define (problem object_scaling-3) (:domain ispatialgym) (:alias object_scaling-apple)
    (:objects apple.n.01_1 - apple.n.01 agent.n.01_1 - agent.n.01)
    (:init (grasped agent.n.01_1 apple.n.01_1))
    (:goal (and (grasped ?agent.n.01_1 ?apple.n.01_1) (pose_reached ?apple.n.01_1))))"""
    new = old.replace(":alias", ":variant")[:-1] + "(:measure (pose_reached ?apple.n.01_1)))"
    for directory, text in ((source, old), (runtime, new)):
        (directory / "problem.bddl").write_text(text)
        c.write_json_atomic(directory / "template_meta.json", {"activity_definition": "problem.bddl", "problem": "object_scaling-3",
                                                              "bddl_sha256": c.file_digest(directory / "problem.bddl")})
    obj = c.ObjectRef("apple.n.01_1", "target", "apple", "a", c.digest("asset"), ready=True, readiness_source="test")
    template = c.Template("t", "object_scaling", "apple", "eval_standard", "house_single_floor", "living_room_0", (obj,), c.digest("definition"),
                          {"source_template": str(source), "source_bddl_sha256": c.file_digest(source / "problem.bddl")})
    changed = g.bind_runtime_definition(template, runtime)
    assert changed.fingerprint != template.fingerprint
    assert changed.scene_setup["runtime_bddl_sha256"] == c.file_digest(runtime / "problem.bddl")
    assert changed.scene_setup["source_bddl_sha256"] == c.file_digest(source / "problem.bddl")
    import shutil
    copied = tmp_path / "fresh_runtime"
    shutil.copytree(runtime, copied)
    assert g.bind_runtime_definition(changed, copied).fingerprint == changed.fingerprint
    # Same union of goal/measure atoms is insufficient: removing the gating atom is forbidden.
    (runtime / "problem.bddl").write_text(new.replace("(pose_reached ?apple.n.01_1)", "", 1))
    meta = g.read_json(runtime / "template_meta.json")
    meta["bddl_sha256"] = c.file_digest(runtime / "problem.bddl")
    c.write_json_atomic(runtime / "template_meta.json", meta)
    import pytest
    with pytest.raises(c.ContractError, match="semantics differ"):
        g.bind_runtime_definition(template, runtime)
    with pytest.raises(c.ContractError, match="previously pinned"):
        g.bind_runtime_definition(changed, copied)


def test_joint_author_uses_actual_root_target_and_all_joint_names(tmp_path):
    job, ledger, tro_path, template, code = GoalGroupControllerTests().fixture(tmp_path)
    template = replace(template, task="configuration_matching-articulation_close_large_scale",
                       objects=(replace(template.objects[0], joint="hinge"),))
    tro = g.read_json(tro_path)
    tro["ispatialgym"]["campaign"] = {"runtime": {"planned_goal_joints": {"apple.n.01_1": [.01, .2]},
        "joint_names": {"apple.n.01_1": ["hinge", "passive"]}, "bound_object_poses": {"apple.n.01_1": [[0, 0, 0], [0, 0, 0, 1]]}}}
    from dataclasses import asdict
    perturb = c.Perturbations({"apple.n.01_1": c.ScaleState((1, 1, 1), (1, 1, 1), "fixed")})
    tro["ispatialgym"]["campaign"].update(campaign_id="new", task=template.task, class_id=template.class_id,
        split=template.split, perturbations=asdict(perturb), validation_key={"scale_sha256": perturb.scale_fingerprint})
    tro["ispatialgym"]["campaign"]["runtime"]["observed_scales"] = {"apple.n.01_1": [1, 1, 1]}
    tro["ispatialgym"]["pose_targets"] = {"apple.n.01_1": {"position": [1, 2, 3], "orientation": [0, 0, 0, 1], "frame": "world"}}
    c.write_json_atomic(tro_path, tro)
    side_path = tro_path.with_name("fresh_9000000.oracle.json")
    side = g.read_json(side_path)
    side["tro_sha256"] = c.file_digest(tro_path)
    c.write_json_atomic(side_path, side)
    summary_path = tro_path.with_name("fresh_9000000.summary.json")
    summary = g.read_json(summary_path)
    summary.pop("goal_realised")
    summary.update(joint="hinge", direction="close")
    c.write_json_atomic(summary_path, summary)
    meta_path = tro_path.parent.parent / "template_meta.json"
    meta = g.read_json(meta_path)
    meta["template_facts"] = {"limits": [0, 1]}
    c.write_json_atomic(meta_path, meta)
    anchor = g.successful_anchor(job, campaign_root=tmp_path, ledger=ledger, campaign_started_at=time.time()-10)
    out = tmp_path / "groups/joint"
    g.prepare_group(anchor=anchor, template=template, campaign_id="new", group_id="joint", output=out,
                    code_sha256=c.file_digest(code / ".snapshot_commit"), physics_sha256=c.digest("physics"),
                    data_sha256=c.digest("maps"), magnitude_range=[0, .5])
    plan = g.read_json(out / "goal_plan.json")
    assert plan["schema"] == "isg-joint-goal-plan-v1"
    assert plan["goal_pose"]["position"] == [1, 2, 3]
    assert plan["goal_joints"] == {"hinge": 0, "passive": .2}
    assert plan["fraction_range"] == [0, .5]


def test_large_pair_author_binds_scene_partner_and_goal_cell(tmp_path):
    job, ledger, _, template, code = GoalGroupControllerTests().fixture(tmp_path)
    anchor = g.successful_anchor(job, campaign_root=tmp_path, ledger=ledger, campaign_started_at=time.time()-10)
    partner = c.ObjectRef("table.n.02_1", "partner", "table", "p", c.digest("partner"), ready=True, readiness_source="test")
    template = replace(template, task="inter_object_relation_satisfaction-ontop_large_scale", objects=template.objects+(partner,))
    anchor["template_meta"]["bindings"] = [{"name": "apple.n.01_1", "role": "target"}, {"name": partner.instance, "role": "partner"}]
    bindings = anchor["tro"]["ispatialgym"]["object_bindings"]
    bindings["apple.n.01_1"]["scene_object_name"] = "apple_scene"
    bindings[partner.instance] = {"category": "table", "model": "p"}
    anchor["bound_object_poses"] = {partner.instance: [[0, 0, .7], [0, 0, 0, 1]]}
    anchor["summary"].update(goal_cell=[1, 2], goal_check={"settled_poses": {"apple_scene": [[1, 2, 2.99], [0, 0, 0, 1]]}})
    out = tmp_path / "groups/large"
    g.prepare_group(anchor=anchor, template=template, campaign_id="new", group_id="large", output=out,
                    code_sha256=c.file_digest(code / ".snapshot_commit"), physics_sha256=c.digest("physics"),
                    data_sha256=c.digest("maps"), magnitude_range=[0, 1])
    plan = g.read_json(out / "goal_plan.json")
    assert plan["goal_cell"] == [1, 2]
    assert plan["partner_pose"]["position"] == [0, 0, .7]
    assert plan["goal_pose"]["position"] == [1, 2, 2.99]


def test_large_under_author_requires_hashed_native_drop_contact_and_fixed_endpoint(tmp_path):
    import pytest
    job, ledger, _, template, code = GoalGroupControllerTests().fixture(tmp_path)
    anchor = g.successful_anchor(job, campaign_root=tmp_path, ledger=ledger, campaign_started_at=time.time()-10)
    partner = c.ObjectRef('desk.n.01_1', 'partner', 'desk', 'p', c.digest('partner'), ready=True, readiness_source='test')
    template = replace(template, task='inter_object_relation_satisfaction-under_large_scale', objects=template.objects+(partner,))
    target = template.objects[0].instance
    anchor['template_meta']['bindings'] = [{'name': target, 'role': 'target'}, {'name': partner.instance, 'role': 'partner'}]
    anchor['tro']['ispatialgym']['object_bindings'][target]['scene_object_name'] = 'target_scene'
    anchor['tro']['ispatialgym']['object_bindings'][partner.instance] = {'category': 'desk', 'model': 'p'}
    anchor['bound_object_poses'] = {partner.instance: [[0, 0, .7], [0, 0, 0, 1]]}
    anchor['planned_goal_poses'] = {target: [[1., 2., .003], [0., 0., 0., 1.]]}
    anchor['summary']['goal_check'] = {'settled_poses': {'target_scene': [[1.001, 2., .002], [0., 0., 0., 1.]]}}
    sidepath = Path(anchor['sources']['sidecar']['path'])
    side = g.read_json(sidepath)
    side.update(stances={'placement': {'edge': '-x'}},
        grasps=[{'object': target, 'place': {'position': [.7, 2., .15], 'orientation': [0., 0., 0., 1.]},
                 'eef_in_object': {'position': [0., 0., 0.], 'orientation': [0., 0., 0., 1.]}}],
        contacts={'push': {'kind': 'push_under', 'object': target, 'direction': [1., 0., 0.], 'drop_cell': [.7, 2.], 'goal_cell': [1., 2.]}})
    c.write_json_atomic(sidepath, side)
    anchor['sources']['sidecar']['sha256'] = c.file_digest(sidepath)
    kwargs = dict(anchor=anchor, template=template, campaign_id='new', code_sha256=c.file_digest(code/'.snapshot_commit'),
                  physics_sha256=c.digest('physics'), data_sha256=c.digest('maps'), magnitude_range=[0, 1])
    g.prepare_group(**kwargs, group_id='under', output=tmp_path/'groups/under')
    plan = g.read_json(tmp_path/'groups/under/goal_plan.json')
    assert plan['floor_push'] == {'edge': '-x', 'direction': [1., 0.]}
    assert plan['place_pose']['position'] == [.7, 2., .15]
    assert plan['release_pose']['position'] == [1., 2., .003]
    assert plan['goal_pose']['position'] == [1.001, 2., .002]
    side['contacts']['push']['drop_cell'][1] += .02
    c.write_json_atomic(sidepath, side)
    with pytest.raises(c.ContractError, match='sidecar changed'):
        g.prepare_group(**kwargs, group_id='changed', output=tmp_path/'groups/changed')
    anchor['sources']['sidecar']['sha256'] = c.file_digest(sidepath)
    with pytest.raises(c.ContractError, match='disagree on drop cell'):
        g.prepare_group(**kwargs, group_id='bad_drop', output=tmp_path/'groups/bad_drop')


def test_alignment_buckets_use_applied_native_range_without_widening(tmp_path):
    import pytest
    _, _, _, template, _ = GoalGroupControllerTests().fixture(tmp_path)
    template = replace(template, task="alignment-world_vertical")
    bound = g.with_alignment_buckets(template, [15, 25], source={"sha256": c.digest("meta"), "field": "applied_range"})
    assert bound.scene_setup["bucket_policy"]["edges"] == pytest.approx([15, 18.333333, 21.666667, 25])
    assert bound.fingerprint != template.fingerprint
    with pytest.raises(c.ContractError, match="native alignment range"):
        g.with_alignment_buckets(template, [25, 25], source={"sha256": c.digest("meta"), "field": "applied_range"})


def test_only_documented_legacy_articulation_measure_migration_is_allowed(tmp_path, monkeypatch):
    import pytest
    source, runtime = tmp_path / "source", tmp_path / "runtime"
    source.mkdir()
    runtime.mkdir()
    atom = "(layout_reached ?apple.n.01_1)"
    old = """(define (problem configuration_matching-1) (:domain ispatialgym)
    (:alias configuration_matching-articulation_open_small_scale-test)
    (:objects apple.n.01_1 - apple.n.01 agent.n.01_1 - agent.n.01)
    (:init (grasped agent.n.01_1 apple.n.01_1))
    (:goal (and (not (grasped ?agent.n.01_1 ?apple.n.01_1)) (pose_settled ?apple.n.01_1) (layout_reached ?apple.n.01_1))))"""
    new = old.replace(":alias", ":variant").replace(" " + atom, "")[:-1] + "(:measure " + atom + "))"
    def write(directory, text):
        (directory / "problem.bddl").write_text(text)
        c.write_json_atomic(directory / "template_meta.json", {"activity_definition": "problem.bddl", "problem": "configuration_matching-1",
                                                               "bddl_sha256": c.file_digest(directory / "problem.bddl")})
    write(source, old)
    write(runtime, new)
    obj = c.ObjectRef("apple.n.01_1", "target", "apple", "a", c.digest("asset"), ready=True, readiness_source="test")
    template = c.Template("t", "configuration_matching-articulation_open_small_scale", "x", "eval_standard", "house_single_floor", "living_room_0", (obj,), c.digest("definition"),
                          {"source_template": str(source), "source_bddl_sha256": c.file_digest(source / "problem.bddl")})
    bound = g.bind_runtime_definition(template, runtime)
    proof = bound.scene_setup["runtime_bddl_equivalence"]
    assert proof["semantic_equivalence"] is False
    assert proof["migration"]["id"] == "2026-10-14_articulation_measure_migration"
    assert proof["migration"]["removed_gating_atoms"] == [["layout_reached", "?apple.n.01_1"]]
    write(source, old[:-1] + "(:measure " + atom + "))")
    template = replace(template, scene_setup={**template.scene_setup, "source_bddl_sha256": c.file_digest(source / "problem.bddl")})
    with pytest.raises(c.ContractError, match="semantics differ"):
        g.bind_runtime_definition(template, runtime)
    exact = (template.class_id, c.file_digest(source / "problem.bddl"), c.file_digest(runtime / "problem.bddl"))
    monkeypatch.setattr(g, "EXPLICIT_ARTICULATION_MEASURE_REPAIRS", {exact: "fixture-demonstrated-current-definition"})
    pinned = g.bind_runtime_definition(template, runtime)
    assert pinned.scene_setup["runtime_bddl_equivalence"]["migration"]["exact_explicit_measure_repair"]["source_sha256"] == exact[1]
    write(runtime, new.replace("(pose_settled ?apple.n.01_1)", ""))
    with pytest.raises(c.ContractError, match="semantics differ"):
        g.bind_runtime_definition(template, runtime)


def test_exact_articulation_repair_preserves_parsed_legacy_measurement(tmp_path, monkeypatch):
    import pytest
    source, runtime = tmp_path / "source", tmp_path / "runtime"
    source.mkdir(); runtime.mkdir()
    layout = "(layout_reached ?apple.n.01_1)"
    pose_atom = "(pose_reached ?apple.n.01_1)"
    old = """(define (problem configuration_matching-1) (:domain ispatialgym)
    (:variant configuration_matching-articulation_close_small_scale-test)
    (:objects apple.n.01_1 - apple.n.01 agent.n.01_1 - agent.n.01)
    (:init (grasped agent.n.01_1 apple.n.01_1))
    (:goal (and (not (grasped ?agent.n.01_1 ?apple.n.01_1)) (pose_settled ?apple.n.01_1) (layout_reached ?apple.n.01_1)))
    (:measure (pose_reached ?apple.n.01_1)))"""
    new = old.replace(" " + layout, "").replace("(:measure " + pose_atom + ")", "(:measure " + pose_atom + " " + layout + ")")
    def write(directory, text):
        (directory / "problem.bddl").write_text(text)
        c.write_json_atomic(directory / "template_meta.json", {"activity_definition": "problem.bddl", "problem": "configuration_matching-1",
            "bddl_sha256": c.file_digest(directory / "problem.bddl")})
    write(source, old); write(runtime, new)
    obj = c.ObjectRef("apple.n.01_1", "target", "apple", "a", c.digest("asset"), ready=True, readiness_source="test")
    template = c.Template("t", "configuration_matching-articulation_close_small_scale", "x", "eval_standard",
        "house_single_floor", "living_room_0", (obj,), c.digest("definition"),
        {"source_template": str(source), "source_bddl_sha256": c.file_digest(source / "problem.bddl")})
    with pytest.raises(c.ContractError, match="semantics differ"):
        g.bind_runtime_definition(template, runtime)
    def allow_current():
        exact = (template.class_id, c.file_digest(source / "problem.bddl"), c.file_digest(runtime / "problem.bddl"))
        monkeypatch.setattr(g, "EXPLICIT_ARTICULATION_MEASURE_REPAIRS", {exact: "fresh-demo-and-physics-replay"})
    allow_current()
    migration = g.bind_runtime_definition(template, runtime).scene_setup["runtime_bddl_equivalence"]["migration"]
    assert migration["measurement_atoms_unchanged"] is True
    assert migration["removed_gating_atoms"] == [["layout_reached", "?apple.n.01_1"]]
    # Even an accidentally added allowlist entry cannot remove another measure
    # or weaken an unrelated goal clause.
    for invalid in (new.replace(pose_atom, ""), new.replace("(pose_settled ?apple.n.01_1)", "")):
        write(runtime, invalid); allow_current()
        with pytest.raises(c.ContractError, match="semantics differ"):
            g.bind_runtime_definition(template, runtime)


def test_axial_author_uses_fresh_goal_and_native_range_policy(tmp_path):
    job,ledger,_,template,code=GoalGroupControllerTests().fixture(tmp_path)
    template=replace(template,task="alignment-axial")
    anchor=g.successful_anchor(job,campaign_root=tmp_path,ledger=ledger,campaign_started_at=time.time()-10)
    anchor["template_meta"]["template_facts"]={"deviation_range_applied_deg":[20,70]}
    out=tmp_path/"groups/axial"
    g.prepare_group(anchor=anchor,template=template,campaign_id="new",group_id="axial",output=out,
        code_sha256=c.file_digest(code/".snapshot_commit"),physics_sha256=c.digest("physics"),data_sha256=c.digest("maps"),magnitude_range=[20,36.6666667])
    plan=g.read_json(out/"goal_plan.json")
    config=g.read_json(out/"campaign_config.json")
    assert plan["schema"] == "isg-surface-goal-plan-v1" and plan["task"] == "alignment-axial"
    assert config["template"]["scene_setup"]["bucket_policy"]["native_range_deg"] == [20,70]


def test_large_opening_author_selects_fixed_joint_goal_in_requested_bucket_before_starts(tmp_path):
    job, ledger, _, template, code = GoalGroupControllerTests().fixture(tmp_path)
    template = replace(template, task='configuration_matching-articulation_open_large_scale',
                       objects=(replace(template.objects[0], joint='hinge'),))
    anchor = g.successful_anchor(job, campaign_root=tmp_path, ledger=ledger, campaign_started_at=time.time()-10)
    target = template.objects[0].instance
    anchor['summary'].update(joint='hinge', direction='open')
    anchor['joint_names'] = {target: ['hinge', 'other']}
    anchor['planned_goal_joints'] = {target: [.4, .3]}
    anchor['template_meta']['template_facts'] = {'limits': [0., 2.]}
    goals = []
    for group in ('opening_a', 'opening_b'):
        out = tmp_path / 'groups' / group
        g.prepare_group(anchor=anchor, template=template, campaign_id='new', group_id=group, output=out,
            code_sha256=c.file_digest(code / '.snapshot_commit'), physics_sha256=c.digest('physics'),
            data_sha256=c.digest('maps'), magnitude_range=[.5, 1.])
        plan = g.read_json(out / 'goal_plan.json')
        goals.append(plan['goal_joints']['hinge'])
        assert 1 < goals[-1] <= 2 and plan['goal_joints']['other'] == .3
        assert plan['goal_joint_proposal']['requires_native_passive_stability']
    assert goals[0] != goals[1]


def test_large_opening_anchor_reuse_preserves_verified_goal_and_strict_bucket(tmp_path):
    import pytest
    job, ledger, _, template, code = GoalGroupControllerTests().fixture(tmp_path)
    template = replace(template, task='configuration_matching-articulation_open_large_scale',
                       objects=(replace(template.objects[0], joint='hinge'),))
    anchor = g.successful_anchor(job, campaign_root=tmp_path, ledger=ledger, campaign_started_at=time.time()-10)
    target = template.objects[0].instance
    anchor['summary'].update(joint='hinge', direction='open')
    anchor['joint_names'] = {target: ['hinge', 'other']}
    anchor['planned_goal_joints'] = {target: [.28274476528167725, .3]}
    anchor['template_meta']['template_facts'] = {'limits': [0., 1.5707963267948966]}
    kwargs = dict(anchor=anchor, template=template, campaign_id='new',
        code_sha256=c.file_digest(code / '.snapshot_commit'), physics_sha256=c.digest('physics'),
        data_sha256=c.digest('maps'), reuse_anchor_goal=True)
    out = tmp_path / 'groups/reuse_opening'
    g.prepare_group(**kwargs, group_id='reuse_opening', output=out, magnitude_range=[0., .5])
    plan = g.read_json(out / 'goal_plan.json')
    assert plan['goal_joints'] == {'hinge': .28274476528167725, 'other': .3}
    assert plan['goal_joint_proposal']['method'] == 'verified_anchor_joint_goal_before_starts'
    assert plan['goal_joint_proposal']['requires_native_passive_stability'] is True
    with pytest.raises(c.ContractError, match='outside the requested strict bucket'):
        g.prepare_group(**kwargs, group_id='wrong_bucket', output=tmp_path / 'groups/wrong_bucket',
                        magnitude_range=[.5, 1.])


def test_nextto_marker_group_preserves_native_raised_release_and_source_view(tmp_path):
    """A new grasp may need a new release z; its settled goal stays fixed."""
    job, ledger, _, template, code = GoalGroupControllerTests().fixture(tmp_path)
    anchor = g.successful_anchor(job, campaign_root=tmp_path, ledger=ledger, campaign_started_at=time.time()-10)
    partner = c.ObjectRef('partner.n.01_1', 'partner', 'colored_pencil', 'p', c.digest('partner'), ready=True, readiness_source='fixture')
    template = replace(template, task='inter_object_relation_satisfaction-nextto_small_scale',
        class_id='inter_object_relation_satisfaction-nextto_01', objects=template.objects+(partner,))
    anchor['template_meta']['bindings'] = [{'name':'apple.n.01_1','role':'target'}, {'name':partner.instance,'role':'partner'}]
    anchor['tro']['ispatialgym']['object_bindings']['apple.n.01_1']['scene_object_name'] = 'target_marker'
    anchor['tro']['ispatialgym']['object_bindings'][partner.instance] = {'category':partner.category,'model':partner.model}
    anchor['planned_start_poses'] = {partner.instance:[[5.9486,3.4677,.815],[0,0,0,1]]}
    stance = {'base_pose2d':[6.2146,2.5701,1.5682], 'focus':[6.2169,3.4801,.8197], 'translate':.5,'tilt':-.4542}
    settled = [[5.9684,3.4834,.8187],[0,0,0,1]]
    release = {'position':[5.9684,3.4834,.8388],'orientation':[0,0,0,1]}
    anchor['summary'].update(goal_realised=release, goal_check={'settled_poses':{'target_marker':settled}},
        raised_release={'raise_m':.02,'partner_clearance_m':.005,'max_raise_m':.03}, manipulation_stance=stance)
    summary_path=Path(anchor['sources']['summary']['path']);c.write_json_atomic(summary_path,anchor['summary'])
    anchor['sources']['summary']['sha256']=c.file_digest(summary_path)
    out=tmp_path/'groups/marker'
    kwargs=dict(anchor=anchor,template=template,campaign_id='new',group_id='marker',code_sha256=c.file_digest(code/'.snapshot_commit'),
        physics_sha256=c.digest('physics'),data_sha256=c.digest('maps'),magnitude_range=[0,1],reuse_anchor_goal=True)
    g.prepare_group(output=out,**kwargs)
    plan=g.read_json(out/'goal_plan.json')
    assert plan['release_policy']=='native_raised_height'
    assert plan['goal_pose']=={'position':settled[0],'orientation':settled[1]}
    assert plan['release_pose']==release
    assert plan['anchor_manipulation_stance']==dict(stance,source_summary_sha256=c.file_digest(summary_path))
    assert not plan['source_instance_is_member']
    summary_path.write_text('changed')
    with __import__('pytest').raises(c.ContractError,match='summary changed'):
        g.prepare_group(output=tmp_path/'groups/changed',**kwargs)
