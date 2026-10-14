"""Harness tests without the simulator: configuration, instructions, the pose convention, bounds, probes, the observation
turn, image horizon, tool surface, the three provider wire formats and the retry policy, the RoboProbe turn loop of the L3
agent (scripted model), the adapter's episode accounting, the code-agent bridge, recipes and the summary."""

from __future__ import annotations

import json
import math
import threading
import time

import numpy as np
import pytest

from ispatialgym.harness.config import HarnessConfig, MotionLimits
from ispatialgym.harness.core import pose
from ispatialgym.harness.core.approver import SafetyAbort, TargetApprover, check_action, parse_eef_targets
from ispatialgym.harness.core.instructions import clean, instruction_for, parse_alias
from ispatialgym.harness.core.observation import ObservationEncoder, state_lines
from ispatialgym.harness.core.perturb import Perturbations
from ispatialgym.harness.core.session import MotionSession, RobotAPI, ThreadBridge, serve_bridge
from ispatialgym.harness.core.trace import EpisodeTrace
from ispatialgym.harness.core.types import (ArmState, CapabilityFailure, InfrastructureFailure, MotionOutcome, MotionPlan, MotionRejected,
                                            MotionRequest, Observation, RobotState, arrival_lines)
from ispatialgym.harness.llm import base as llm_base
from ispatialgym.harness.llm import make_client
from ispatialgym.harness.llm.anthropic_messages import AnthropicMessagesClient
from ispatialgym.harness.llm.base import (IMAGE_PLACEHOLDER, AssistantEntry, AssistantTurn, ImagePart, TextPart, ToolCall, ToolEntry, UserEntry,
                                          apply_image_horizon, parse_arguments, post_json)
from ispatialgym.harness.llm.openai_chat import OpenAIChatClient
from ispatialgym.harness.llm.openai_responses import OpenAIResponsesClient
from ispatialgym.harness.llm.scripted import ScriptedClient
from ispatialgym.harness.planners import EpisodeContext, make_planner
from ispatialgym.harness.planners.l3_tools.docs import build_r1pro_docs
from ispatialgym.harness.planners.l3_tools.prompts import goal_text, system_message
from ispatialgym.harness.planners.l3_tools.recipes import recipe_key, task_recipe
from ispatialgym.harness.planners.l3_tools.tools import ToolArgError, merge_parallel_calls, to_decision, tool_specs
from ispatialgym.harness.policy import HarnessPolicy


def _arm(y: float) -> ArmState:
    return ArmState(grasp={"x": 0.55, "y": y, "z": 1.0, "pitch_deg": 0.0, "roll_deg": 0.0, "yaw_deg": 0.0}, gripper_opening=1.0,
                    joints=tuple(0.1 * i for i in range(7)))


def _state(step=0) -> RobotState:
    return RobotState(step=step, arms={"left": _arm(0.2), "right": _arm(-0.2)}, base_odometry=(0.1, -0.2, 5.0), torso_translate=0.5, torso_tilt_deg=0.0)


def _obs(step=0, remaining=17000) -> Observation:
    img = np.zeros((32, 32, 3), np.uint8)
    img[:, :16] = 255
    return Observation(step=step, images={"head": img, "left_wrist": img, "right_wrist": img}, state=_state(step), steps_remaining=remaining)


def _agent(steps, **cfg_kw):
    cfg = HarnessConfig(model="scripted:x", **cfg_kw)
    agent = make_planner("l3_tools", client=ScriptedClient(steps=steps), cfg=cfg, encoder=ObservationEncoder(cfg))
    agent.start(EpisodeContext("Move the bar soap on top of the soap dish.", np.zeros((16, 16, 3), np.uint8),
                               alias="inter_object_relation_satisfaction-ontop_small_scale-bar_soap_utgixp-soap_dish-shelf"))
    return agent


def _eef(note="move", **targets):
    return {"name": "move_eef", "arguments": {"targets": targets, "note": note}}


# -- configuration / instructions ---------------------------------------------------------------------------------
def test_config_round_trip(tmp_path):
    cfg = HarnessConfig(model="anthropic/x", max_calls=7, limits=MotionLimits(max_duration_s=6.0))
    p = tmp_path / "c.json"
    cfg.save(str(p))
    back = HarnessConfig.from_file(str(p))
    assert back.max_calls == 7 and back.limits.max_duration_s == 6.0 and back.model == "anthropic/x"
    assert HarnessConfig().max_calls == 170 and HarnessConfig().image_size is None and HarnessConfig().jpeg_quality == 95
    with pytest.raises(ValueError):
        HarnessConfig.from_dict({"bogus": 1})


@pytest.mark.parametrize("alias,expect", [
    ("inter_object_relation_satisfaction-under_small_scale-beer_glass-coaster-bar", "Move the beer glass so that the coaster is under it"),
    ("inter_object_relation_satisfaction-under_large_scale-sandal-bed-bed_iorsl", "Move the sandal under the bed"),
    ("inter_object_relation_satisfaction-ontop_small_scale-bar_soap_utgixp-soap_dish-shelf", "Move the bar soap on top of the soap dish"),
    ("inter_object_relation_satisfaction-touching_small_scale-box_of_cereal_s080-box_of_cereal-bar", "Move the box of cereal so that it touches the box of cereal"),
    ("inter_object_relation_satisfaction-inside_large_scale-plate-drop_in_sink_lkklqs_0-bar", "Move the plate inside the drop in sink"),
    ("camera_relocalization-obstructed-trash_can_1", "A trash can may be in the way"),
    ("mental_rotation-teaspoon", "You are holding the teaspoon"),
    ("object_scaling-bottle_of_pop", "the same size as in the goal image"),
    ("alignment-axial-tablespoon-breakfast_table", "The tablespoon lies on the breakfast table"),
    ("alignment-world_vertical-wall_clock-walls", "The wall clock is mounted on a wall"),
    ("configuration_matching-articulation_close_large_scale-door", "Close the door"),
    ("configuration_matching-attachment-lid-teapot-bar", "Attach the lid to the teapot"),
    ("configuration_matching-block_design-block_4-breakfast_table_board", "breakfast table (with a board)"),
    ("disembedding-tablefork-tablefork_1_table_knife_1_tablespoon_1-breakfast_table", "tablefork, table knife, tablespoon"),
])
def test_instructions(alias, expect):
    assert expect in instruction_for(alias, "alias")


def test_family_level_names_no_objects():
    s = instruction_for("inter_object_relation_satisfaction-ontop_small_scale-carving_knife-cutting_board-commercial_kitchen_table", "family")
    assert "knife" not in s and "cutting" not in s and "goal image" in s
    assert parse_alias("camera_relocalization-standard")["variant"] == "standard"
    assert clean("floor_pitched") == "floor" and clean("hamper_1") == "hamper"


# -- pose convention ----------------------------------------------------------------------------------------------
def test_pose_reference_and_axes():
    assert pose.matrix_to_angles(pose.REFERENCE) == (0.0, 0.0, 0.0)
    assert np.allclose(pose.angles_to_matrix(90, 0, 0)[:, 2], [0, 0, -1])  # pitch +90: straight down
    assert np.allclose(pose.angles_to_matrix(90, 0, 0)[:, 1], [0, 1, 0])  # jaws still close along y
    assert np.allclose(pose.angles_to_matrix(0, 0, 90)[:, 2], [0, 1, 0])  # yaw +90: pointing left
    assert np.allclose(pose.angles_to_matrix(0, 90, 0)[:, 1], [0, 0, 1])  # roll +-90: the closing axis vertical (singular)
    assert pose.labels()[:7] == ("left_x", "left_y", "left_z", "left_pitch_deg", "left_roll_deg", "left_yaw_deg", "left_gripper")


def test_pose_round_trips():
    rng = np.random.default_rng(0)
    for _ in range(2000):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        a = pose.matrix_to_angles(pose.quat_to_matrix(q))
        assert -90.0 <= a[1] <= 90.0
        assert pose.orientation_error_deg(q, pose.matrix_to_quat(pose.angles_to_matrix(*a))) < 1e-4
    a = pose.matrix_to_angles(pose.angles_to_matrix(30, 90, 0))
    assert pose.orientation_error_deg(pose.matrix_to_quat(pose.angles_to_matrix(30, 90, 0)), pose.matrix_to_quat(pose.angles_to_matrix(*a))) < 1e-6
    v = {"x": 0.6, "y": 0.2, "z": 0.9, "pitch_deg": 10.0, "roll_deg": 5.0, "yaw_deg": -20.0}
    back = pose.pose_to_values(*pose.values_to_pose(v))
    assert all(abs(back[k] - v[k]) < 1e-9 for k in v)
    eef0, _ = pose.values_to_pose(v)
    eef1, q1 = pose.values_to_pose(dict(v, pitch_deg=80.0))
    assert abs(pose.pose_to_values(eef1, q1)["x"] - 0.6) < 1e-12 and np.linalg.norm(eef1 - eef0) > 1e-3  # the wrist swings, the grasp point stays
    assert abs(pose.PAD_LENGTH_M - 0.05) < 1e-6 and round(pose.FINGERTIPS_BEYOND_GRASP_M * 1000) == 31


def test_eef_target_bounds_and_approver():
    by_arm, err = parse_eef_targets({"left_z": 0.9, "right_gripper": 0, "left_yaw_deg": -90})
    assert err is None and by_arm == {"left": {"z": 0.9, "yaw_deg": -90.0}, "right": {"gripper": 0.0}}
    assert "unknown dimension 'left_q'" in parse_eef_targets({"left_q": 1})[1]
    assert "must be a finite number" in parse_eef_targets({"left_x": float("nan")})[1]
    assert "must be a finite number" in parse_eef_targets({"left_x": True})[1]
    assert "outside [-90, 90]" in parse_eef_targets({"left_roll_deg": 95})[1]
    assert "non-empty object" in parse_eef_targets({})[1]
    assert parse_eef_targets({"left_x": 1.2})[1] is None and "outside" in parse_eef_targets({"left_x": 1.2}, negate_xyz=True)[1]
    ap = TargetApprover(MotionLimits())
    for bad in ((2.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 270.0)):
        with pytest.raises(MotionRejected) as e:
            ap.check_base(*bad)
        assert e.value.repairable
    with pytest.raises(MotionRejected):
        ap.check_torso(3.0, None)
    with pytest.raises(SafetyAbort):
        check_action(np.array([0.0, np.nan]))


# -- probes / observation -------------------------------------------------------------------------------------------
def test_perturbations():
    pz = Perturbations(flip_vision_lr=True, mask_cameras=["head"], negate_xyz=True)
    img = _obs().images["left_wrist"]
    assert pz.view("left_wrist", img)[0, 0, 0] == 0 and pz.view("left_wrist", img)[0, -1, 0] == 255
    assert pz.view("head", img).max() == 0 and pz.view("head", img).shape == img.shape and pz.is_masked("head")
    seen = pz.observed(_state())
    assert seen.arms["left"].grasp["x"] == -0.55 and seen.arms["left"].grasp["pitch_deg"] == 0.0 and seen.base_odometry == (-0.1, 0.2, 5.0)
    r = pz.request(MotionRequest("move_eef", {"targets": {"left_x": -0.6, "left_yaw_deg": 30.0}}))
    assert r.params["targets"] == {"left_x": 0.6, "left_yaw_deg": 30.0} and "jitter_m" not in r.params
    r = pz.request(MotionRequest("move_base", {"forward_m": 0.5, "left_m": 0.1, "turn_deg": 10}))
    assert r.params["forward_m"] == -0.5 and r.params["left_m"] == -0.1 and r.params["turn_deg"] == 10
    fixed = Perturbations(jitter_dx=0.01, jitter_sphere_r=0.02, seed=3)
    j1 = fixed.request(MotionRequest("move_eef", {"targets": {"left_z": 1.0}})).params["jitter_m"]
    j2 = fixed.request(MotionRequest("move_eef", {"targets": {"left_z": 1.0}})).params["jitter_m"]
    assert j1 == j2 and abs(np.linalg.norm(np.array(j1) - [0.01, 0, 0]) - 0.02) < 1e-9  # once per episode
    per_move = Perturbations(jitter_sphere_mean_r=0.05, jitter_sphere_per_move=True, seed=1)
    a = per_move.request(MotionRequest("move_eef", {"targets": {"left_z": 1.0}})).params["jitter_m"]
    b = per_move.request(MotionRequest("move_eef", {"targets": {"left_z": 1.0}})).params["jitter_m"]
    assert a != b


def test_observation_turn():
    lines = state_lines(_state(), "Do it.")
    assert lines[0] == "Instruction: Do it." and "left_x=0.5500" in lines and "left_pitch_deg=0.0" in lines and "right_gripper=1.00" in lines
    assert "base_x=0.1000" in lines and "torso_translate=0.50" in lines and "left_arm_joint7=0.6000" in lines
    assert lines.index("left_x=0.5500") < lines.index("right_x=0.5500") < lines.index("base_x=0.1000") < lines.index("left_arm_joint1=0.0000")
    cfg = HarnessConfig(cameras=["head", "left_wrist"], perturb={"mask_cameras": ["head"]})
    enc = ObservationEncoder(cfg, Perturbations.from_dict(cfg.perturb))
    content = enc.observation_content(_obs(remaining=123), "Do it.", 4, ["Arrival check for the previous move_eef target: left 4 mm and 0.6 deg away."])
    text = content[0].text
    assert "Env steps remaining before the episode ends: 123" in text and text.endswith("left 4 mm and 0.6 deg away.")
    assert [p.label for p in content[1:]] == ["camera 'head' (step 4), MASKED", "camera 'left_wrist' (step 4)"]
    import base64
    import io

    from PIL import Image
    big = Observation(0, {"head": np.zeros((600, 900, 3), np.uint8)}, _state())
    part = ObservationEncoder(HarnessConfig(cameras=["head"])).camera_parts(big, 0)[0]
    assert Image.open(io.BytesIO(base64.b64decode(part.data_b64))).size == (900, 600)  # native resolution


def test_motion_plan_and_arrival_text():
    assert MotionPlan("move_eef", 75, 1 / 30, gripper_steps=20, reports="grasp point").accepted_text() == \
        "Accepted: playing 75 waypoints (2.5s). The last 20 move the gripper, once the arm has arrived. The next observation reports how close the grasp point landed."
    assert MotionPlan("move_base", 60, 1 / 30, approximate=True, reports="base").accepted_text().startswith("Accepted: playing about 60 waypoints (2.0s).")
    eef = {"arrival": {"kind": "move_eef", "arms": {"right": {"pos_mm": 5.2, "rot_deg": 0.24}, "left": {"pos_mm": 12.0, "rot_deg": 3.41}}}}
    assert arrival_lines(eef) == ["Arrival check for the previous move_eef target: left 12 mm and 3.4 deg away; right 5 mm and 0.2 deg away."]
    assert arrival_lines({"arrival": {"kind": "move_base", "pos_mm": 31.0, "yaw_deg": 1.2}}) == ["Arrival check for the previous move_base target: 31 mm and 1.2 deg away."]
    assert arrival_lines({"arrival": {"kind": "move_torso", "translate": None, "tilt_deg": 0.4}}) == ["Arrival check for the previous move_torso target: tilt 0.4 deg away."]
    assert arrival_lines(None) == [] and arrival_lines({}) == []


def test_paced_resampling_respects_the_per_step_budget():
    import torch as th

    from ispatialgym.harness.core.motion import joint_travel, resample_path
    path = th.zeros(11, 28)
    path[:, 9] = th.tensor([0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.6, 0.9, 1.2, 1.5])  # uneven: slow, then fast
    inv = np.zeros(28)
    inv[9] = 1.0 / 0.1  # budget 0.1 rad per step
    steps = int(math.ceil(joint_travel(path, inv)[-1] - 1e-9))
    out = resample_path(path, steps, inv)
    assert steps == 15 and out.shape == (15, 28) and abs(float(out[-1, 9]) - 1.5) < 1e-6
    assert np.abs(np.diff(np.r_[0.0, out[:, 9].numpy()])).max() <= 0.1 + 1e-6  # index-uniform resampling would step 0.2 rad there


def test_planner_world_floor_slab_and_root_frame():
    from ispatialgym.harness.core import world as W
    floor = W.floor_slab(0.005, (2.0, 0.0))  # RoboDojo's slab: 5 cm thick, its top 2.5 cm under the mounting surface
    assert math.isclose(floor.top, -0.02) and floor.dims == (W.FLOOR_SPAN_M, W.FLOOR_SPAN_M, 0.05) and floor.center[:2] == (2.0, 0.0)
    inside = W.overlapping(floor, np.array([[3.0, 1.0, -0.03, 0.0], [3.0, 1.0, -0.01, 0.0], [3.0, 1.0, 0.0, 0.025]]))
    assert inside.tolist() == [True, False, True]
    box = W.Box("b", (1.0, 3.0, 0.5), math.radians(90), (0.4, 0.2, 0.05), "b")
    root_q = pose.matrix_to_quat(np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]))  # root yawed +90 deg at (1, 2, 0)
    cub = W.scene_model([box], (1.0, 2.0, 0.0), root_q)["cuboid"]["b"]
    assert np.allclose(cub["pose"][:3], (1.0, 0.0, 0.5)) and np.allclose(np.abs(cub["pose"][3]), 1.0) and cub["dims"] == [0.4, 0.2, 0.05]


def test_planner_world_floor_clears_the_robot_at_the_start():
    from ispatialgym.harness.core import world as W
    floor = W.floor_slab(0.0, (0.0, 0.0))
    assert W.lowered(floor, np.array([[0.2, 0.1, 0.08, 0.075]]), "the robot") is floor  # a wheel on the floor clears the slab under it
    low = W.lowered(floor, np.array([[0.2, 0.1, 0.08, 0.075], [0.0, 0.0, 0.165, 0.18]]), "the robot")  # a base sphere reaching below the floor
    assert math.isclose(low.top, 0.165 - 0.18 - W.CLEARANCE_M, abs_tol=1e-5) and low.dims == floor.dims
    assert low.notes == ("lowered 10 mm under the robot at the start",) and not W.overlapping(low, np.array([[0.0, 0.0, 0.165, 0.18]]), W.CLEARANCE_M).any()


# -- conversation ----------------------------------------------------------------------------------------------------
def test_image_horizon_keeps_newest_observations_and_pinned():
    tr = [UserEntry([TextPart("Goal: x"), ImagePart("goal", "AA", pinned=True)])]
    for i in range(4):
        tr.append(UserEntry([TextPart(f"obs {i}"), ImagePart(f"cam {i}", f"B{i}")]))
        tr.append(AssistantEntry(AssistantTurn(tool_calls=[ToolCall(f"c{i}", "move_eef", {})])))
        tr.append(ToolEntry(f"c{i}", "move_eef", [TextPart("Accepted: ...")]))
    out = apply_image_horizon(tr, 2)
    imgs = [p.label for e in out for p in getattr(e, "content", []) if isinstance(p, ImagePart)]
    assert imgs == ["goal", "cam 2", "cam 3"]
    assert out[1].content[1] == TextPart(IMAGE_PLACEHOLDER) and IMAGE_PLACEHOLDER == "[earlier camera image omitted to save context]"
    assert len([p for e in apply_image_horizon(tr, None) for p in getattr(e, "content", []) if isinstance(p, ImagePart)]) == 5
    assert parse_arguments('{"a": 1}') == {"a": 1} and parse_arguments("{bad") is None and parse_arguments("") == {}


def test_tool_surface():
    cfg = HarnessConfig()
    specs = {t.name: t for t in tool_specs(cfg)}
    assert list(specs) == ["move_eef", "move_base", "move_torso", "give_up"]
    assert specs["move_eef"].parameters["required"] == ["targets", "note"] and specs["give_up"].parameters["required"] == ["reason", "hindsight"]
    desc = specs["move_eef"].description
    assert "over 10s is refused" in desc and "left_x: [-0.3, 1.3]" in desc and "+90 is pointing straight down" in desc
    assert [t.name for t in tool_specs(HarnessConfig(allow_done=True, disable_give_up=True))] == ["move_eef", "move_base", "move_torso", "done"]
    assert "left_x: [-1.3, 0.3]" in tool_specs(HarnessConfig(perturb={"negate_xyz": True}))[0].description
    d = to_decision(ToolCall("1", "move_eef", {"targets": {"left_z": 0.9, "left_gripper": 0}, "note": "grasp"}), cfg)
    assert d.request.kind == "move_eef" and d.request.params == {"targets": {"left_z": 0.9, "left_gripper": 0.0}} and d.request.note == "grasp"
    with pytest.raises(ToolArgError, match="does not take 'arm'"):
        to_decision(ToolCall("1", "move_eef", {"arm": "left", "position": [0.5, 0, 1]}), cfg)
    with pytest.raises(ToolArgError, match="unknown dimension"):
        to_decision(ToolCall("1", "move_eef", {"targets": {"left_w": 1}}), cfg)
    with pytest.raises(ToolArgError):
        to_decision(ToolCall("1", "move_torso", {"note": "x"}), cfg)
    assert to_decision(ToolCall("1", "move_torso", {"tilt_deg": 5, "note": "look down"}), cfg).request.params == {"tilt_deg": 5.0}
    assert to_decision(ToolCall("1", "move_base", {"turn_deg": 10, "note": "x"}), cfg).request.params == {"forward_m": 0.0, "left_m": 0.0, "turn_deg": 10.0}


def test_merge_parallel_calls():
    a = ToolCall("a", "move_eef", {"targets": {"left_z": 0.9}, "note": "left down"})
    b = ToolCall("b", "move_eef", {"targets": {"right_z": 0.9}, "note": "right down"})
    m = merge_parallel_calls([a, b])
    assert len(m) == 1 and m[0].id == "a" and m[0].arguments == {"targets": {"left_z": 0.9, "right_z": 0.9}, "note": "left down | right down"}
    assert len(merge_parallel_calls([a, ToolCall("c", "move_eef", {"targets": {"left_z": 1.0}})])) == 2  # overlapping dimension
    assert len(merge_parallel_calls([a, ToolCall("d", "move_base", {"turn_deg": 5})])) == 2


def _transcript():
    return [UserEntry([TextPart("Goal: x"), ImagePart("goal image", "GG", pinned=True)]),
            UserEntry([TextPart("Instruction: x"), ImagePart("camera 'head' (step 0)", "HH")]),
            AssistantEntry(AssistantTurn(text="thinking", tool_calls=[ToolCall("call_1", "move_eef", {"targets": {}}, '{"targets": {}}')])),
            ToolEntry("call_1", "move_eef", [TextPart("Accepted: playing 30 waypoints (1.0s).")]),
            UserEntry([TextPart("Instruction: x"), ImagePart("camera 'head' (step 1)", "II")])]


def test_openai_responses_wire():
    c = OpenAIResponsesClient("gpt-6-astra")
    body = c.build_request("sys", _transcript(), tool_specs(HarnessConfig()), None)
    assert body["instructions"] == "sys" and body["reasoning"] == {"effort": "medium"} and body["include"] == ["reasoning.encrypted_content"]
    assert "tool_choice" not in body and body["parallel_tool_calls"] is False and body["store"] is False
    kinds = [i.get("type") or i.get("role") for i in body["input"]]
    assert kinds == ["user", "user", "assistant", "function_call", "function_call_output", "user"]
    assert body["input"][4]["output"] == "Accepted: playing 30 waypoints (1.0s)." and "detail" not in body["input"][1]["content"][-1]
    assert "reasoning" not in OpenAIResponsesClient("m", effort=None).build_request("s", [], [], None)
    turn = c.parse_response({"output": [{"type": "reasoning", "encrypted_content": "x"}, {"type": "function_call", "call_id": "c9", "name": "give_up",
                                                                                         "arguments": '{"reason": "r"}'}],
                             "usage": {"input_tokens": 10, "output_tokens": 5, "output_tokens_details": {"reasoning_tokens": 3}}})
    assert turn.tool_calls[0].arguments == {"reason": "r"} and turn.usage["reasoning_tokens"] == 3 and len(turn.raw) == 2
    assert c.build_input([AssistantEntry(turn)])[0]["type"] == "reasoning"
    with pytest.raises(CapabilityFailure):
        c.parse_response({"output": [], "status": "incomplete", "incomplete_details": {"reason": "content_filter"}})


def test_anthropic_wire():
    c = AnthropicMessagesClient("claude-x")
    tr = _transcript()
    msgs = c.build_messages(tr)
    assert [m["role"] for m in msgs] == ["user", "assistant", "user"]
    assert msgs[2]["content"][0]["type"] == "tool_result" and msgs[2]["content"][1]["type"] == "text"  # results first, then the observation
    body = c.build_request("sys", tr, tool_specs(HarnessConfig()), None)
    assert body["tool_choice"] == {"type": "auto", "disable_parallel_tool_use": True}
    turn = c.parse_response({"content": [{"type": "thinking", "thinking": "t", "signature": "s"}, {"type": "tool_use", "id": "t1", "name": "move_base",
                                                                                                    "input": {"turn_deg": 10}}], "usage": {"input_tokens": 1}})
    assert turn.tool_calls[0].arguments == {"turn_deg": 10} and turn.raw[0]["type"] == "thinking"


def test_openai_chat_wire():
    c = OpenAIChatClient("deepseek-x", base_url="http://x/v1")
    msgs = c.build_messages("sys", _transcript())
    assert [m["role"] for m in msgs] == ["system", "user", "user", "assistant", "tool", "user"]
    body = c.build_request("sys", _transcript(), [], None)
    assert body["parallel_tool_calls"] is False and "tool_choice" not in body
    turn = c.parse_response({"choices": [{"message": {"content": None, "tool_calls": [{"id": "a", "function": {"name": "give_up", "arguments": "{}"}}]},
                                          "finish_reason": "tool_calls"}], "usage": {"prompt_tokens": 3}})
    assert turn.tool_calls[0].name == "give_up" and turn.usage["input_tokens"] == 3
    with pytest.raises(CapabilityFailure):
        c.parse_response({"choices": [{"message": {}, "finish_reason": "content_filter"}]})
    with pytest.raises(ValueError):
        make_client("nonsense")


class _Resp:
    def __init__(self, status, body=None, headers=None):
        self.status_code, self._body, self.headers = status, body or {}, headers or {}
        self.text = json.dumps(self._body)

    def json(self):
        return self._body


def test_post_json_retry_policy(monkeypatch):
    sleeps = []
    monkeypatch.setattr(llm_base.time, "sleep", lambda s: sleeps.append(s))
    seq = [_Resp(429, {"error": {"message": "slow down"}}), _Resp(503), _Resp(200, {"ok": 1})]
    monkeypatch.setattr(llm_base, "_post_once", lambda *a, **k: seq.pop(0))
    assert post_json("http://x", {"Authorization": "Bearer sk-secret"}, {}) == {"ok": 1}
    assert len(sleeps) == 2 and 1.0 <= sleeps[0] < 2.0 and 2.0 <= sleeps[1] < 4.0  # the jittered 1, 2, 4 ... ladder
    monkeypatch.setattr(llm_base, "_post_once", lambda *a, **k: _Resp(429, headers={"retry-after": "7"}))
    sleeps.clear()
    with pytest.raises(InfrastructureFailure, match="after 3 attempts"):
        post_json("http://x", {}, {}, retries=2)
    assert sleeps == [7.0, 7.0]
    monkeypatch.setattr(llm_base, "_post_once", lambda *a, **k: _Resp(400, {"error": {"code": "content_policy_violation", "message": "no"}}))
    with pytest.raises(CapabilityFailure):
        post_json("http://x", {}, {})
    monkeypatch.setattr(llm_base, "_post_once", lambda *a, **k: _Resp(401, {"error": {"message": "bad key sk-secret"}}))
    with pytest.raises(InfrastructureFailure) as e:
        post_json("http://x", {"Authorization": "Bearer sk-secret"}, {})
    assert "sk-secret" not in str(e.value) and not e.value.retryable

    def boom(*a, **k):
        raise TimeoutError("hard deadline")
    monkeypatch.setattr(llm_base, "_post_once", boom)
    with pytest.raises(InfrastructureFailure, match="hard deadline"):
        post_json("http://x", {}, {}, retries=1)


# -- the turn loop ---------------------------------------------------------------------------------------------------
def test_l3_agent_roboprobe_turns():
    agent = _agent([_eef("reach", right_x=0.6, right_z=0.9),
                    _eef("too far down", right_z=0.4),
                    _eef("pick", right_z=0.85, right_gripper=0),
                    {"name": "give_up", "arguments": {"reason": "cannot see it", "hindsight": "should have looked first"}}])
    goal = agent.transcript[0].content
    assert goal[0].text.startswith("Goal: Move the bar soap on top of the soap dish.\n\nTASK RECIPE:\n# Inter-object Relation Satisfaction")
    assert isinstance(goal[1], ImagePart) and goal[1].pinned and agent.recipe == "inter_object_relation_satisfaction"
    assert "170 LLM calls" in agent.system and "Embodiment notes:" in agent.system
    d = agent.decide(_obs(0), None)
    assert d.kind == "motion" and d.request.params == {"targets": {"right_x": 0.6, "right_z": 0.9}} and d.request.note == "reach"
    agent.accepted(MotionPlan("move_eef", 45, 1 / 30, reports="grasp point"))
    assert agent.transcript[-1].content[0].text == "Accepted: playing 45 waypoints (1.5s). The next observation reports how close the grasp point landed."
    outcome = MotionOutcome(True, "x", details={"arrival": {"kind": "move_eef", "arms": {"right": {"pos_mm": 4.0, "rot_deg": 0.6}}}})
    d = agent.decide(_obs(46), outcome)
    obs_text = agent.transcript[-2].content[0].text
    assert "Arrival check for the previous move_eef target: right 4 mm and 0.6 deg away." in obs_text and agent.turns == 2
    n = len(agent.transcript)
    d = agent.rejected("right arm pose unreachable (planner status: IK Fail). The robot did not move; choose another pose. Neither arm moved.", False)
    assert d.kind == "motion" and d.request.params["targets"]["right_gripper"] == 0.0
    new = agent.transcript[n:]
    assert isinstance(new[0], ToolEntry) and new[0].is_error and not any(isinstance(e, UserEntry) for e in new)  # no new observation
    assert agent.repairs == 0 and agent.unreachable == 1
    agent.accepted(MotionPlan("move_eef", 80, 1 / 30, gripper_steps=45, reports="grasp point"))
    d = agent.decide(_obs(130), MotionOutcome(True, "x"))
    assert d.kind == "stop" and d.reason == "give_up" and d.details == {"detail": "cannot see it", "hindsight": "should have looked first"}
    assert agent.transcript[-1].content[0].text == "Acknowledged give_up."
    labels = [p.label for p in agent.transcript[1].content if isinstance(p, ImagePart)]
    assert labels == ["camera 'head' (step 0)", "camera 'left_wrist' (step 0)", "camera 'right_wrist' (step 0)"]
    s = agent.summary()
    assert s["calls"] == 4 and s["turns"] == 3 and s["stop_reason"] == "give_up" and s["hindsight"] == "should have looked first"


def test_l3_agent_repairs():
    agent = _agent([{"text": "Let me think."}, {}, {"name": "move_eef", "raw_arguments": "{bad"}])
    d = agent.decide(_obs(), None)
    assert d.kind == "stop" and d.reason == "capability_failure" and agent.calls == 3
    texts = [p.text for e in agent.transcript for p in getattr(e, "content", []) if isinstance(p, TextPart)]
    assert "Reply with exactly one tool call from the provided tools." in texts
    assert "The model returned an empty response. Reply with exactly one tool call from the provided tools." in texts
    assert any(t.startswith("invalid JSON arguments: ") for t in texts)
    agent = _agent([{"name": "declare_success", "arguments": {}}, _eef("bad", left_q=1), _eef("ok", left_z=1.0)])
    d = agent.decide(_obs(), None)
    assert d.kind == "motion" and agent.repairs == 2
    agent.rejected("target for 'left_z' is outside [-0.2, 2]", True)  # a repairable motion error is the third repair
    assert agent.stop_reason == "capability_failure"
    agent = _agent([_eef("x", left_z=1.0)] * 4)  # repairs reset with every new observation
    agent.decide(_obs(), None)
    agent.rejected("bad", True)
    agent.rejected("bad", True)
    agent.accepted(MotionPlan("move_eef", 3, 1 / 30))
    assert agent.decide(_obs(), MotionOutcome(True, "x")).kind == "motion"


def test_l3_agent_parallel_calls_budget_and_switches():
    agent = _agent([{"calls": [_eef("l", left_z=0.9), _eef("r", right_z=0.9)]}])
    d = agent.decide(_obs(), None)
    assert d.request.params["targets"] == {"left_z": 0.9, "right_z": 0.9}
    merged = [e for e in agent.transcript if isinstance(e, ToolEntry)]
    assert [e.call_id for e in merged] == ["call_0_1"] and merged[0].content[0].text == "Merged into the first move_eef call of this turn."
    agent = _agent([{"calls": [_eef("l", left_z=0.9), {"name": "move_base", "arguments": {"turn_deg": 5, "note": "x"}}]}])
    assert agent.decide(_obs(), None).reason == "capability_failure"
    agent = _agent([{"name": "move_base", "arguments": {"turn_deg": 10, "note": "look"}}] * 5, max_calls=2)
    assert agent.decide(_obs(), None).kind == "motion"
    agent.accepted(MotionPlan("move_base", 40, 1 / 30, approximate=True, reports="base"))
    assert agent.decide(_obs(), MotionOutcome(True, "ok")).kind == "motion"
    agent.accepted(MotionPlan("move_base", 40, 1 / 30, approximate=True, reports="base"))
    d = agent.decide(_obs(), MotionOutcome(True, "ok"))
    assert d.kind == "stop" and d.reason == "call_budget"
    agent = _agent([{"name": "done", "arguments": {"summary": "placed", "hindsight": ""}}], allow_done=True)
    assert agent.decide(_obs(), None).reason == "done"
    agent = _agent([{"name": "give_up", "arguments": {"reason": "x", "hindsight": "y"}}] * 3, disable_give_up=True)
    d = agent.decide(_obs(), None)
    assert d.reason == "capability_failure" and "give_up" not in agent.tool_names


def test_l3_agent_refusal_is_a_capability_failure():
    class Refuser(ScriptedClient):
        def generate(self, *a, **k):
            raise CapabilityFailure("model refusal: content_filter")
    cfg = HarnessConfig(model="scripted:x")
    agent = make_planner("l3_tools", client=Refuser(steps=[]), cfg=cfg, encoder=ObservationEncoder(cfg))
    agent.start(EpisodeContext("x", None))
    assert agent.decide(_obs(), None).reason == "capability_failure"


# -- adapter accounting ----------------------------------------------------------------------------------------------
def test_planner_stops_are_failures_and_outages_unscored():
    pol = HarnessPolicy(HarnessConfig(model="scripted:x"))
    pol._stop("give_up", detail="cannot see it", reason="a planner detail named reason")
    assert pol.stopped["reason"] == "give_up"
    ep = {"success": True}
    pol.finalize(ep)
    assert ep["success"] is False and pol.meta["harness"]["forced_failure"] and pol.meta["harness"]["epilogue_success"] is True
    pol = HarnessPolicy(HarnessConfig(model="scripted:x"))
    pol.stopped = {"reason": "success", "step": 10}
    ep = {"success": True}
    pol.finalize(ep)
    assert ep["success"] is True and not pol.meta["harness"]["forced_failure"]
    pol = HarnessPolicy(HarnessConfig(model="scripted:x"))
    with pytest.raises(InfrastructureFailure):
        pol._infrastructure("OPENAI_API_KEY is not set")
    assert pol.meta["harness"]["failure_kind"] == "infrastructure" and pol.stopped["reason"] == "infrastructure"
    assert not isinstance(InfrastructureFailure("x"), Exception)


def test_trace_takes_a_kind_field(tmp_path):
    tr = EpisodeTrace(str(tmp_path))
    tr.event("motion_start", kind="move_eef", params={})
    tr.close()
    rec = json.loads((tmp_path / "transcript.jsonl").read_text().splitlines()[0])
    assert rec["type"] == "motion_start" and rec["kind"] == "move_eef"


# -- prompts / recipes ---------------------------------------------------------------------------------------------
def test_system_message_docs_and_recipes():
    docs = build_r1pro_docs({"left shoulder": (0.1, 0.2, 1.3)})
    assert "pads are 50 mm long" in docs and "reach 31 mm past it" in docs and "97 mm" in docs
    assert docs.endswith("- left shoulder: (0.100, 0.200, 1.300) m")
    msg = system_message(HarnessConfig(max_calls=99), docs)
    assert "99 LLM calls" in msg and msg.index("Embodiment notes:") > msg.index("env steps remaining")
    keys = {
        "camera_relocalization-obstructed-trash_can_1": "camera_relocalization",
        "configuration_matching-articulation_open_small_scale-cabinet-table": "configuration_matching-articulation",
        "configuration_matching-block_design-block_4-breakfast_table": "configuration_matching-block_design",
        "configuration_matching-attachment-lid-teapot-bar": "configuration_matching-attachment",
        "alignment-world_vertical-wall_clock-walls": "alignment-world_vertical",
        "alignment-axial-tablespoon-breakfast_table": "alignment-axial",
        "mental_rotation-teaspoon": "mental_rotation", "object_scaling-mug": "object_scaling",
        "disembedding-a-b_1-c": "disembedding", "inter_object_relation_satisfaction-nextto_small_scale-a-b-c": "inter_object_relation_satisfaction",
    }
    for alias, key in keys.items():
        assert recipe_key(alias) == key
        k, text = task_recipe(alias)
        assert k == key and "## Success check" in text and "settling period" in text
    assert task_recipe(None) is None and goal_text("Do it.") == "Goal: Do it."
    assert goal_text("Do it.", "# R\n") == "Goal: Do it.\n\nTASK RECIPE:\n# R"


# -- code-agent bridge ----------------------------------------------------------------------------------------------
class _FakeLayer:
    def prepare(self, req):
        if req.params.get("targets") == {"left_z": 9.0}:
            raise MotionRejected("target for 'left_z' is outside [-0.2, 2]")

        def gen():
            for _ in range(3):
                yield np.zeros(23)
            return MotionOutcome(True, f"{req.kind} done")
        return MotionPlan(req.kind, 3, 1 / 30), gen()


def test_bridge_blocking_api():
    session, bridge = MotionSession(_FakeLayer()), ThreadBridge()
    api = RobotAPI(bridge)
    results = []

    def program():
        results.append(api.move_eef({"left_z": 1.0}))
        results.append(api.move_eef({"left_z": 9.0}))
        results.append(api.move_eef({"left_gripper": 0.0}))

    t = threading.Thread(target=program, daemon=True)
    t.start()
    actions = 0
    try:
        for _ in range(1000):
            a, idle = serve_bridge(session, bridge, wait_s=0.05)
            actions += a is not None
            if not t.is_alive() and idle:
                break
    finally:
        bridge.close()
        t.join(timeout=5)
    assert not t.is_alive()
    assert [r.ok for r in results] == [True, False, True] and actions == 6
    assert results[0].steps == 3 and "outside" in results[1].message


def test_bridge_timeout_and_close_release_the_worker():
    with pytest.raises(TimeoutError):
        ThreadBridge().submit(MotionRequest("hold", {"steps": 1}), timeout=0.2)
    bridge, errors = ThreadBridge(), []

    def program():
        try:
            RobotAPI(bridge).wait(5)
        except RuntimeError as e:
            errors.append(str(e))

    t = threading.Thread(target=program, daemon=True)
    t.start()
    time.sleep(0.2)
    bridge.close()
    t.join(timeout=5)
    assert not t.is_alive() and errors == ["the episode has ended"]


# -- summary -----------------------------------------------------------------------------------------------------------
def test_summarize(tmp_path, capsys):
    from ispatialgym.harness.summarize import episode_rows, main

    metas = [{"success": True, "harness": {"stop": {"reason": "success"}, "planner": {"calls": 3, "usage": {"input_tokens": 9}}}},
             {"success": False, "harness": {"stop": {"reason": "give_up"}, "forced_failure": True, "planner": {"calls": 5}}},
             {"status": "error", "error": "InfrastructureFailure('HTTP 503')", "harness": {"stop": {"reason": "infrastructure"}, "failure_kind": "infrastructure"}},
             None]
    for i, m in enumerate(metas):
        ep = tmp_path / "s3" / f"a_{i}"
        ep.mkdir(parents=True)
        (ep / "harness_config.json").write_text(json.dumps({"alias": f"alignment-axial-x{i}-y"}))
        if m is not None:
            (ep / "run" / "iid").mkdir(parents=True)
            (ep / "run" / "iid" / "meta.json").write_text(json.dumps(m))
    rows = episode_rows(str(tmp_path))
    assert [r["status"] for r in rows] == ["scored", "scored", "unscored", "not run"] and rows[0]["calls"] == 3
    main([str(tmp_path)])
    out = capsys.readouterr().out
    assert "2 scored, 1 unscored, 1 not run" in out and "1/2  50.0%" in out and "HTTP 503" in out
    assert "flagged: 2 meta.json" in out  # written before the scoring change: symbolic only
