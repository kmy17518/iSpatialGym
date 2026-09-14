# Articulated-object opening tests: state after the second full pass (2026-09-07 18:30)

Everything lives in `runlogs/grasp_library_20260905/artic/` (synced copy of `/tmp/glib/artic/`, minus the large `inv/*.obj`)
and `runlogs/grasp_library_20260905/scripts/artic_*.py`. Working copies used by the scripts are under `/tmp/glib/` (volatile).

## Question answered: handle annotations
None in the assets. Meta-links in the 19 articulation categories: only `lights`, `togglebutton`, `fillable`; link tags
`openable / glass / locked / openablebothsides`; `openable_joint_ids` is joint-level (170/310 models). Handles are baked into the
single door/drawer/lid mesh and its convex collision decomposition. Exceptions: 4 `door` models with `handle1..4` lever links,
`bottom_cabinet/songes` with named handle links. OmniGibson's own open planner ignores handles (bbox face point).

## Question answered: joint dynamics
All assets: joint friction 0.2, damping 2.86. Vertical-hinge doors and drawers hold any position (0 drift in 3 s). Horizontal-hinge
lids drift to the gravitational equilibrium: hinged_jar gqtsam -61 % of range in 3 s (falls closed), cooler -50 %, toolbox +14 %
(falls open), hinged_jar vzwhbg +7 %, carton msfzpz -2 %. To hold an angle: `obj.set_joint_positions` (kinematic), raise
`obj.joints[name].friction` (2-5), or a joint drive target.

## Pipeline (scripts/, in order)
1. `artic_inventory.py` -> `inv/<cat>__<model>.json` (+ `.obj` collision pieces per movable link): 100 models, 159 openable
   joints, joint axes / hinge points / opening sign measured empirically in sim, link and base bboxes (object frame, closed).
2. `artic_handles.py` -> `handles/*.json`: protruding convex pieces -> handle candidates (bar/knob), door handle links, edge
   fallbacks (lid free edge, drawer top edge, leaf edge, contact 3.5 cm inside), status handle/edge/ungraspable.
3. `artic_render.py` -> `strips/*.png` (overview, per-joint zoom with markers A-D/F/X, opened 60 %).
4. Review round 1: `handle_review_contract.md`, 10 agents -> `review/*.json` (chosen marker, pinch, motion, graspable).
5. Review round 2 (no-handle joints): `artic_render_corners.py` -> `strips2/`, `render2/*_corners.json`;
   `handle_review2_contract.md`, 4 agents -> `review2/*.json` (own proposals in outer-face coordinates: 19 feasible / 38 not);
   `artic_apply_review2.py` -> merged into `manual_handles.json` (18 candidates + 43 skips incl. 5 visual-only handles).
6. `artic_scene_assign.py` -> `scene_assign.json`: 80 scene-object tests mapped to host instances in 6 challenge scenes
   (Rs_int 22, house_single_floor 25, house_double_floor_lower 12, house_double_floor_upper 11, restaurant_diner 8,
   office_cubicles_right 2; hotel_suite_large unused), same-model hosts preferred, else aspect-closest same-category host + up to 2
   alternatives; swaps scaled to the host AABB (guard 0.2-5x per axis), AABB centres aligned.
7. `artic_validate_table.py` (non-scene: table 0.75 m for objects < 0.6 m tall, else floor, 1.25 m ahead, opening side towards
   the robot) and `artic_validate_scene.py <scene>` (LIMIT env = tests per process). Per joint: candidate = manual > review >
   heuristic (+ alternative); orientations front / front_down45 / topdown (bars, knobs) or edge_down30 (edges), both wrist rolls;
   `Skills.stance_for` with `eef_candidates=[pre, grasp]`, distances 0.45-0.8, trunk translates 0-2, lateral -0.2/0/-0.35, approach
   angles 0/+-0.5 rad (scene); arms hanging during the search; scene harness teleports to the stance (no navigation); `set_torso`;
   `Skills.grasp(grasp_pose=..)` (AG must attach); pull in 6 chunks along the joint axis / hinge arc with torso IK, base back-off on
   failure, base-follow from the start for doors/fridges/ovens/dishwashers/dryers and hinges > 0.45 m from the handle; records
   reachability, AG, dq vs target (0.15 m / 45 deg), `Open` state, frames `frames/<key>__<joint>_{0,1,2}.png`.
   Run one OmniGibson process per non-scene model and per <= 4 scene tests (`run_artic_chain4.sh`), one process at a time
   (two OG processes + a scene load caused a Vulkan device loss once; GPU is 24 GB, idle 1 GB - two scene processes in parallel
   are probably fine, three are not).
8. `artic_aggregate.py` -> `artic_results.csv/.md` (per joint), viable-model list, exclusion list. `artic_ik_probe.py` /
   `artic_joint_hold.py` are the probes quoted above.

## Results (pass 2, all 100 models / 157 tested joints)
* opened 13 joints on 11 models: bottom_cabinet_no_top pluwfl (door 26 deg + drawer 14 cm) and qudfwe (drawer 14 cm), carton
  msfzpz (lid 46 deg), door bexenl (45 deg), fridge jtqazu (45 deg), hinged_jar vzwhbg (46 deg), oven kfftgk / leqtlc / rwuazb
  (26-34 deg), toolbox redmtl (34 deg), top_cabinet dmwxyl (both doors 45-56 deg). The 45-deg-down orientation produced 6 of them.
* grasped but < half target: 4 (spojpj door 16 deg, crtrpa door 17 deg, dishwasher ynrhuk 22 deg, dishwasher cgugie 0).
* unreachable 64: handles below 0.65 m or above 1.3 m, or no collision-free stance (counters, islands, neighbours). IK probe: right
  arm horizontal pinch reachable only at 0.7-1.3 m within 0.75 m of the base (1.3 m needs trunk tilt +0.5..1); trunk translate
  >= 1 (downward / ground keyframes) never helps a horizontal approach. Top cabinets in the scenes sit at 1.5-2.1 m.
* infeasible per two review rounds 33, visual-only handles (asset issue) 4, model errors 26 (9 swap aspect skips, joint-name
  mismatch after swap 3, singular scale 2, physics exception 1, rest see `results/*.json` `error`), planning/collision/other 9.
* Reachable-height reference from the probe and results: viable handles were at 0.55-1.49 m.

## If another pass is wanted
* Low handles (0.34-0.6 m, most bottom drawers): only a top-down / steep pinch from a crouch can work; `orientation_options`
  already adds `topdown` for bars with >= 2.5 cm protrusion - extend it to edges and lower the depth threshold, and test with
  translate 1.5-2.0 explicitly.
* Rerun only the failed joints: delete their `results/<key>.json` and run `run_artic_chain4.sh` (skips existing results).
* Speed: each scene test takes ~5-9 min because of the stance search; trim `orientation_options` or the stance grid, or run two
  scenes in parallel (see GPU note).
* Reviewer proposals that failed in sim (round-2 candidates) are listed in `artic_results.md` with their error; the
  `frames/*_fail*.png` show the gripper at the failure.
