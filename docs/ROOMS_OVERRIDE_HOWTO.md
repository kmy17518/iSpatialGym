# Rooms override — producing a `rooms_override.json` entry for a new template

For someone who knows the evaluator (`run_isg`, `run_isg_vec`, the instance lists under `$V/instances`) but not campaign rooms
(2026-10-27/28). The procedure is the campaign's; the scripts are `scripts/vec_eval/rooms/` (its `README.md` lists their
arguments). Abbreviations: `$V=/vision/u/minyeong/projects/isg_home/vec_eval85` (the campaign record;
`$V/campaigns/rooms/{DECISIONS.md,summary/,seg/,gate/}`), `$R=/vision/u/minyeong/projects/ispatialgym-eval-vec-rooms` (the
campaign's worktree, branch `vec-eval-rooms`, where the mechanism is implemented), `<root>` = a vec_eval root like `$V`,
`<campaign>` = the folder of your run (`$V/campaigns/rooms_<date>`).

## 0. Prerequisites

**Code.** The mechanism lives in `$R` (diff vs the production worktree: `$V/campaigns/rooms/code_delta_vs_E_20261028.patch`):

| file | what |
|---|---|
| `ispatialgym/oracle/vec_rooms.py` | `RoomVisibility` (the segmentation probe), `FrameDumper` + `parse_dump_spec` (lossless frame dumps), `resolve_rooms_override` (the `--rooms-override` lookup by alias), `ALWAYS_LOADED_CATEGORIES` |
| `ispatialgym/oracle/run_isg.py` | `rooms_for_template(meta)` (§3, §4) and `build_env_config` setting `scene.load_room_instances` from it |
| `ispatialgym/oracle/vec_loader.py` | `VecInstanceLoader(load_rooms: list or None, seg_instance: bool)`; `report["load_rooms"]` |
| `ispatialgym/oracle/run_isg_vec.py` | `--load-rooms template, task or <comma list>`, `--rooms-override JSON`, `--seg-visibility`, `--seg-every K`, `--dump-frames SPEC` (`--dump-frames-spec` in vec_next), `--render-override KEY=VALUE`; `meta["load_rooms"]`, `meta["rooms_seen"]`, `meta["rooms_visibility"]`, `meta["frames_dumped"]` |
| `scripts/vec_eval/fleet/worker/eval_worker.py` | unit kinds `vec_cmd` / `vec_runs` (`run_cmd_unit`: arbitrary `run_isg_vec` invocations, several on one host) — only needed for the fleet route of §2.2 |

Check your checkout before starting: `grep -n rooms_for_template ispatialgym/oracle/run_isg.py` and
`grep -n "rooms-override\|seg-visibility" ispatialgym/oracle/run_isg_vec.py` must hit. In `ispatialgym-eval-vec-next` (campaign
vec_next, 2026-10-29) the whole mechanism is merged (three-way, onto the vec_speed driver): `run_isg.rooms_for_template`, the
loader's `load_rooms` list / `seg_instance`, the driver's `--load-rooms template | task | <comma list>`, `--rooms-override`,
`--seg-visibility`, `--seg-every`, and the fleet worker's `vec_cmd` / `vec_runs` units. One flag is renamed there: the rooms
dumper's step spec is **`--dump-frames-spec 0-60:5,25`** (`--dump-frames STEPS` is the vec_speed tiled-experiment dumper, a
comma list of steps / `last`). The commands below use the vec_next spelling; in `$R` itself write `--dump-frames` instead.

**Data.** The alias's instance lists, as `scripts/vec_eval/build_instance_lists.py --out <root>` writes them:
`<root>/instances/all/<alias>.json` (every demo: the eval instance first, then the val instances;
`{"task", "task_id", "instances": [{"name", "tro", "demo", "length", "action_steps", "settle_steps", "alias", "variant", "task_id",
"task_family", "task", "subtask", "instance", ...}]}`), `<root>/instances/eval/<alias>.json` (the eval instance alone; the scripts read
the template meta `<template dir>/<template id>-meta.json` through its `tro` path), `<root>/instances/task_ids.json` (`{alias:
task_id}`; a new alias need not be in it — pass `--tasks <alias>` to `analyze_visibility.py`). Every `demo.hdf5` the list names must
exist on the host that replays (the campaign's first Vast units failed because the eval bundle lacked the 536 val demos;
`val_demos.tar.zst` + `post_refresh.sh` fixed it).

**Environment.** Conda `ispatialgym-vectorized-envs`, Behavior-1K `6a6194600` + the production overlay (`$R`'s checkout:
`/vision/u/minyeong/projects/Behavior-1K-vec-rooms-6a6194600`). Cluster: `$V/campaigns/rooms/jobs/run_vec.sbatch` runs one
`run_isg_vec` per array line of `$ISG_TASKFILE` (`"<out dir> <run_isg_vec args...>"`) from `$R`; lambda:
`scripts/vec_eval/env_lambda.sh` with `ISG_EVAL_ROOT` pointing at the rooms checkout. GPU: 24 GB is enough at N=1 (the campaign ran
on RTX 3090 / A5000 / Quadro RTX 8000).

## 1. What a room override is, and why

A template's meta (`<template id>-meta.json`) carries `task_rooms` (the room instances the task's objects stand in) and
`load_room_instances` = `task_rooms` + every room whose segmentation-map mask, dilated by 0.6 m, touches theirs
(`ispatialgym.sampling.sim_utils.adjacent_rooms(scene, rooms, reach_m=0.6)`: "walls, doors and furniture seen through doorways
and windows exist at evaluation time"). The garden of `house_single_floor` touches every room, so the two door tasks (task rooms
`garden_0, sauna_0`) load all 19 rooms; its dining-room tasks load 6; `Rs_int`'s `mental_rotation-mug` loads 5.

OmniGibson (fork `6a6194600`, `InteractiveTraversableScene._should_load_object`) loads walls, ceilings, roofs, fences, backgrounds and
doors whatever the list (`vec_rooms.ALWAYS_LOADED_CATEGORIES`); floors, windows, lights and furniture follow it. The demos were
rendered with the full `load_room_instances`; the policy's training frames show whatever of the neighbours the robot's cameras saw.

A **room override** is, for one template (matched by its `variant`), an explicit `load_room_instances` list: the task rooms plus
the neighbours the demos' cameras actually see. Rooms whose contents never appear in any frame of any demo are left unloaded:
smaller worlds and faster builds (the 19-room door worlds — N=8 builds of 25–50 min, the most expensive of the benchmark — run
with 4–5 rooms within the renderer's noise), with the renders provably unchanged (§2.2). The structure of a dropped room (walls,
doors) stays; its floor, windows, lights and furniture go. The guarantee is relative to the *demos'* views: a closed-loop policy
that looks where no demo looked would see a dropped room's shell without its contents.

Not the same as the BEHAVIOR challenge's `*-partial_rooms.json` (`load_room_types` = the task rooms' types, no neighbours at
all), nor `run_isg_vec --load-rooms task` (task rooms only): that changes the frames by 40.8/255 (vec_speed, rejected).

## 2. The two-pass method

### 2.1 Pass 1 — segmentation / visibility: which rooms' objects ever appear in the robot-camera frames

Replay every demo of the alias at N=1 with instance segmentation on the three robot cameras (the campaign: 589 demos of 53
tasks, 383k steps):

```bash
python -u -m ispatialgym.oracle.run_isg_vec --out <campaign>/seg/<alias> \
    --mode replay --instances <root>/instances/all/<alias>.json --num-envs 1 \
    --seg-visibility --seg-every 1 --render-override /rtx/raytracing/showLights=0 \
    --record-view '' --no-grid-video --no-external-camera
```

- `--seg-visibility` → `VecInstanceLoader(seg_instance=True)`: `seg_instance` is appended to every robot's `obs_modalities`
  (per-pixel object ids; `info["obs_info"][robot][cam]["seg_instance"]` maps id → object name | `background` | `unlabelled` |
  `groundPlane`), and the `SlotRecorder` attaches a `vec_rooms.RoomVisibility` probe over the head (`zed_link`) and the two wrist
  (`realsense_link`) cameras. Every `--seg-every`-th step it reduces each camera's segmentation to visible objects and pixel
  counts (`torch.unique`), resolves each object through `scene.object_registry` to its `in_rooms` and `category`, skips the robot
  and the non-object labels, flags `ALWAYS_LOADED_CATEGORIES` (reported, not counted for a room) and accumulates per room and
  per camera `frames`, `max_px`, `sum_px`, `first_step`. There is no `ISG_SEG` environment variable: the flag is the only switch.
- `--render-override /rtx/raytracing/showLights=0` is the sampler's crash guard (`SimWorld.LIGHT_SHAPES_SETTING` in
  `ispatialgym/sampling/sim_utils.py`): the instance-segmentation node segfaults on light emitter shapes in view; the setting
  stops drawing the shapes and leaves the illumination unchanged. This pass's RGB is never compared with anything, so the
  non-production setting is harmless here; the gate (§2.2) runs with the production render settings.
- `--record-view '' --no-grid-video --no-external-camera`: no videos, no external camera (fewer render products). The driver's
  `--wallclock` default is 0 (no wall-clock cut); nothing else to set. The segmentation buffer trails the rendered frame by 1–3
  frames (`SimWorld.seg_instance`), irrelevant over a whole episode.
- An alias whose instances span two worlds (the driver prints `RUN_ISG_VEC_WORLDS` and exits 3) is run once per
  `--world-group K` into the **same** `<campaign>/seg/<alias>/` (the stockpot precedent: units `__wg0` / `__wg1` with one
  `out_key`; episode folders have unique names; `results.jsonl` is overwritten but not read by the analysis).

Outputs per episode, `<campaign>/seg/<alias>/<episode>/`: `rooms_visibility.json` = `{"steps_sampled", "every", "pixels":
{"head": 518400 (720²), "left": 230400, "right": 230400 (480²)}, "rooms": {room: {"frames", "max_px", "sum_px", "max_frac", "cams"}},
"unmapped" (ids the mapping did not name: should be 0), "mapping_keys_sample", "rooms_by_camera", "objects_by_camera"}`; `meta.json`
carries `rooms_visibility` (the path), `rooms_seen` (`{room: {frames, max_frac}}`) and `load_rooms`; stdout has one
`RUN_ISG_VEC_ROOMS alias=<alias> {...}` line per invocation.

Then the decision:

```bash
python scripts/vec_eval/rooms/analyze_visibility.py --root <root> --campaign <campaign> --tasks <alias> --keep garden_0
```

A neighbour (a room of `load_room_instances` not in `task_rooms`) is *seen* when its `max_frac` ≥ `--min-frac` (default 0.0002 =
104 px of 720²) in any sampled step of any demo — "guaranteed not to see": a 104-px sliver in one demo keeps the room (carton's
`living_room_1`). Candidate rooms = `task_rooms` + seen neighbours + `--keep`. **Pass `--keep garden_0`**: the campaign ran
without it, the sunflower candidate dropped the garden and failed the gate (§5); the garden is the exterior (lawn, ground,
trees are room-filtered too) and is never dropped. Outputs: `<campaign>/summary/visibility.md` (one row per task: scene, task
rooms, template room count, seen neighbours with max % and demos-seen / demos, dropped, demos) and
`<campaign>/summary/rooms_override_candidate.json` (`overrides: {alias: {rooms, task_rooms, template_rooms, dropped, scene, demos,
seen, seen_not_in_template}}`). An alias with an empty `dropped` needs no override; stop here.

### 2.2 Pass 2 — the same-host gate: the candidate rooms must reproduce the template's replays

On **one** GPU host, in sequence (GPU-to-GPU rendering differences must not enter the comparison), replay the first three
instances of `all/<alias>.json` (eval + 2 val) three times — under the template rooms, under the candidate rooms, and under the
template rooms again (the host's run-to-run noise floor) — with lossless frame dumps and videos, production render settings:

```bash
COMMON="--mode replay --num-envs 1 --dump-frames-spec 0-60:5,25 --record-view tiled --tile-px 240 --external-size 240 --grid-video \
        --instances <root>/instances/all/<alias>.json --limit-batches 3"
python -u -m ispatialgym.oracle.run_isg_vec --out <campaign>/gate/<alias>/template        $COMMON
python -u -m ispatialgym.oracle.run_isg_vec --out <campaign>/gate/<alias>/override        $COMMON --load-rooms <candidate rooms, comma list>
python -u -m ispatialgym.oracle.run_isg_vec --out <campaign>/gate/<alias>/template_repeat $COMMON
```

`--dump-frames-spec 0-60:5,25` (`vec_rooms.parse_dump_spec`) writes `<episode>/frames/step_NNNNN_{head,left,right,external}.png` at steps
0, 5, …, 60 and every 25th step after (lossless PNG at the cameras' native sizes); each episode also has `trajectory.npz`,
`meta.json` (`load_rooms`, `frames_dumped`), `tiled_rgb.mp4`, and each batch `_batches/batch_NNNN_grid.mp4` — the before / after
videos per alias. Fleet route: `scripts/vec_eval/rooms/make_gate_units.py --root <root> --campaign <campaign> --tasks <alias>` writes
exactly these three runs as one `vec_runs` unit (`out/gate/<alias>/{template,override,template_repeat}/`) into
`<campaign>/units.json`; it needs `$R`'s worker (§0). A two-world alias gets runs `<mode>_wg0` / `<mode>_wg1`; merge them by
symlinks into `template/`, `override/`, `template_repeat/` before comparing (the campaign did, DECISIONS 2026-10-28 08:35).

Compare and decide:

```bash
python scripts/vec_eval/rooms/compare_gate.py --root <root> --campaign <campaign> --task <alias> --a template --b override
python scripts/vec_eval/rooms/compare_gate.py --root <root> --campaign <campaign> --task <alias> --a template --b template_repeat --no-dataset
python scripts/vec_eval/rooms/decide_gate.py  --root <root> --campaign <campaign> --source "campaign rooms_<date> (segmentation pass + same-host gate)"
```

`compare_gate.py` writes `<campaign>/gate/<alias>/gate_template_vs_override.json` and `gate_template_vs_template_repeat.json`: per
episode the verdicts of both runs (`success`, `symbolic_success`, `geometric_success`, `geometric_success_step`, `n_steps`,
`reason`, `status`) and `verdicts_equal`, the trajectory deviation (`scripts/vec_eval/compare_slots.py::diff`: max position /
rotation per tracked key, joints), and per dumped step and camera the mean |Δ| at native size and at the policy's 224² bilinear
input (`mean_abs_224`), the fraction of pixels differing by > 2 / > 8, the max. With `--dataset` (default) the val episodes' head
frames are also measured against the recorded demo frames (`--dataset-root`, default the isg-1020-b1k LeRobot dataset):
informational — both runs sit ≈ 2–4/255 from the demo, and where the override passes it is never farther than the template.

`decide_gate.py` applies the rule, per alias over its three episodes:

- **physics**: identical verdicts and success steps in every episode; trajectory deviation of the override vs the template
  ≤ 0.02 m / 0.05 rad (EVAL_RECORD §5's replay-verification thresholds — any object-set change perturbs PhysX at the mm level,
  e.g. carton-desk 1.3 mm, plate 2.8 mm);
- **frames**: robot cameras (head, left, right) on the first-60-step dumps only (the robot has barely moved: neither a physics
  divergence nor the renderer's load-dependent frame latency enters), at 224²; per step the *excess* = override mean |Δ|₂₂₄ −
  repeat mean |Δ|₂₂₄ (both vs the template run); pass when, for every robot camera, mean excess ≤ 0.25/255 **and** p90 excess
  ≤ 0.6/255 — the override is within the renderer's own run-to-run noise (the repeat's own level is 0.3–1.0/255). The p90, not
  the max: single-frame spikes are the frame-latency effect on a moving arm (gym_shoe, crab, mug). Later frames (every 25th
  step) and the external camera are reported, not decisive.

Outputs: `<campaign>/summary/gate.md` (the table + the before / after / repeat video index), `<campaign>/summary/gate_rows.json`
(every gated alias, pass or fail, with the statistics), `<campaign>/summary/rooms_override.json` — **the passing aliases only**,
in the format of §3, each entry's `variant` read from the template meta. A failing alias keeps the template rooms: do not add it.

## 3. The override file: format and matching

```json
{
 "generated": "2026-10-28T08:36",
 "source": "campaign rooms 2026-10-28 (segmentation pass + same-host frame / physics gate)",
 "rule": {"pos_tol_m": 0.02, "rot_tol_rad": 0.05, "excess_mean_255": 0.25, "excess_p90_255": 0.6,
          "frames": "steps 0..60 every 5, robot cameras, 224^2 bilinear (the policy input)"},
 "overrides": {
  "<alias>": {
   "rooms": ["<room instances to load>"],
   "variant": "<the template meta's variant>",
   "template_rooms": ["<the template's load_room_instances>"],
   "dropped": ["<template_rooms minus rooms>"],
   "scene": "<scene model>",
   "evidence": {"segmentation_demos": 11, "gate_episodes": 3, "head_excess_mean_255": 0.0128, "traj_pos_max_m": 0.0, "traj_rot_max_rad": 0.00098}
  }
 }
}
```

The production file is `$V/campaigns/rooms/summary/rooms_override.json` (14 entries, 60 room loads removed). Add a new alias by
merging `decide_gate.py`'s `overrides` entry into it (one entry per template variant; keep `generated` / `source` honest — note the
addition in the record entry). Two lookups read the file:

- `run_isg.rooms_for_template(meta)` (`ISG_ROOMS_OVERRIDE`, both evaluators): takes `d["overrides"]` (or the whole dict when there is
  no `overrides` key), iterates the entries and takes the first whose **key equals `meta["variant"]` or whose `"variant"` field
  equals it**; an entry may be a bare list or `{"rooms": [...]}`; an empty list is ignored; any error is logged
  (`ISG_ROOMS_OVERRIDE ... not applied`) and the template rooms are used. A file that does not exist is silently ignored.
- `vec_rooms.resolve_rooms_override(path, alias)` (`run_isg_vec --rooms-override`): the entry whose **key equals the alias of the
  invocation's first instance spec** (`specs[0]["alias"]`); no variant matching.

Hence every entry is keyed by the **alias** and carries the **variant**: they differ for one task
(`inter_object_relation_satisfaction-nextto-gym_shoe-gym_shoe-bench` vs variant `...-nextto_small_scale-gym_shoe-...`), and the
two lookups must agree. `decide_gate.py` writes both.

## 4. How the evaluators consume it

- **`ISG_ROOMS_OVERRIDE=<file>`** (environment; off when unset — the default everywhere): `run_isg.build_env_config` sets
  `scene.load_room_instances = rooms_for_template(meta)`. The single-env evaluator (`run_isg`, every mode) and the vectorized
  one (`VecInstanceLoader.build_env` calls the same `build_env_config`) therefore load the same rooms, and the single-env
  reference of a clone-count verification matches its clones. `run_isg` logs `rooms override for <variant>: [...] (template:
  [...])` once per template; note that the single-env `meta.json["load_room_instances"]` records the **template's** list
  (`loader.tmeta.get("load_room_instances")`), so the applied override is visible only in that log line and in the vectorized meta.
- **Vectorized driver**, precedence (`run_isg_vec.py`, "room loading"): `--rooms-override <file>` (by alias, §3) > `--load-rooms
  <comma list>` > `--load-rooms task` (the template's `task_rooms`; an experiment) > `--load-rooms template` (default; the
  template's list, or `ISG_ROOMS_OVERRIDE`'s through `build_env_config`). The loader then adds `seg_instance` to the robots if
  asked, applies `--render-override`s, and reports.
- **Record**: every episode's `meta.json["load_rooms"]` = `{"mode": "template" | "override" | "env-override" | "task", "rooms": [...],
  "template_rooms": [...], "dropped": [...], "added": [...], "source": "override <file>" | "--load-rooms list" | "--load-rooms task" | <the
  ISG_ROOMS_OVERRIDE path>}` (the template mode carries only `mode` and `rooms`); the same dict is `loader.report["load_rooms"]` and
  the stdout line `RUN_ISG_VEC_ROOMS alias=<alias> {...}`. `compare_gate.py` reads it as `rooms_a` / `rooms_b`.

## 5. Known failure modes (from the campaign: 32 candidates, 14 pass, 18 fail)

- **Rooms seen through glass or an open doorway are missed by the segmentation pass and caught by the gate.** Every
  `house_single_floor` dining-room task (11) dropped `living_room_1` on the pass's evidence and failed the gate: head excess
  +0.8 to +6.8/255 from step 0 — the room is visible through its open doorway, and the garden view through the glass wall
  changes with it; the instance-segmentation node does not label through glass (those pixels are `unlabelled`, lab note
  2026-09-15). Same for `stockpot` (`private_office_3` through the cubicle glass, +0.76). The pass proposes; the gate decides.
- **Exterior / garden rooms change illumination and physics.** `sunflower-vase` dropped `garden_0`: +0.4/255 through the windows
  and the free-standing sunflower's physics realisation moved 1.3 cm / 0.46 rad. Never drop `garden_0` (`--keep garden_0`).
- **Dishwasher tasks** (`articulation_{open,close}_large_scale-dishwasher`): the pass kept only `garden_0` of the four neighbours;
  the gate failed at +4.7 / +5.9/255 head excess from step 0 (the open-plan dining / living side is in view).
- Marginal and physics failures: `beer_glass` (right wrist +0.70), `soup_ladle` (+0.44), `bottle_of_catsup` (right wrist +0.27,
  marginal), `spatula` (rotation 0.10 rad), `blender`, `cap-lemonade`, `brussels_sprouts`, `salt-pepper`, `tablespoon`,
  `kielbasa`, `sanitary_napkins`, `teacup`, `plate` (the dining-room pattern). Rule of thumb: a candidate that drops a room of
  an open-plan house seldom passes; office / hotel / double-floor rooms behind doors do.
- **Later frames are not evidence either way.** With bit-identical physics (0.0 mm on every tracked pose), wrist frames of a
  moving arm still shift by a frame's motion between a full and a lighter scene (gym_shoe: 12/255 at steps 350 / 425 / 700): the
  observation lags the physics by a load-dependent 1–2 frames, a property of the whole pipeline (clone count, GPU model), not
  of the room set. That is why the gate judges the first 60 steps only.
- The guarantee is w.r.t. the demos' views (§1): a policy that turns towards a dropped room sees its shell.

## 6. Adoption rule

An override entry changes the alias's world. After adding one:

1. **Re-verify the alias's clone count** with the override active (EVAL_RECORD §5 procedure): the eval demo replayed at N=1
   (reference) and with `--same-instance-all-slots` at N=8, then 4, then 2, each under `ISG_ROOMS_OVERRIDE=<file>` (or
   `--rooms-override <file>`), judged by `scripts/vec_eval/compare_slots.py` + `verify_rules.py` (identical settled verdicts,
   geometric success step within 5 steps, trajectories ≤ 2 cm / 0.05 rad, joints ≤ 0.05 rad). In the fleet this is one
   `replay_verify` unit per task; `verify/<task>.json` must be regenerated, not copied from a campaign that ran the template rooms.
2. **Turn it on where the evaluators run**: export `ISG_ROOMS_OVERRIDE=<file>` in the worker job scripts and the lambda / cluster
   `env.sh` (and pass `--rooms-override <file>` where `run_isg_vec` is invoked directly), so the single-env reference and the clones
   load the same rooms. Check the first episodes' `meta.json["load_rooms"].mode` (`env-override` / `override`).
3. **Freeze it in the record**: list the file (path, sha256, the entries it holds) as a frozen input of the campaign's record entry
   (EVAL_RECORD §2 style; the entry template is RUNBOOK §0). Finished campaigns keep their rooms; results under different room
   sets are not compared without saying so.

## 7. Worked example: `mental_rotation-mug` (scene `Rs_int`)

Template meta (`.../ispatialgym-eval-instances/scenes/Rs_int/json/Rs_int_task_mental_rotation_mug_instances/Rs_int_task_mental_rotation_mug_0_0_template-meta.json`):
`variant = mental_rotation-mug`, `task_rooms = [living_room_0]`, `load_room_instances = [living_room_0, bathroom_0, bedroom_0,
entryway_0, kitchen_0]` (5 rooms; alias = variant here).

Pass 1 over the 11 demos (`$V/campaigns/rooms/seg/mental_rotation-mug/`; `visibility.md` row): seen neighbours `bedroom_0` (8.69 %
of a frame, 4/11 demos), `entryway_0` (9.91 %, 5/11), `kitchen_0` (12.94 %, 5/11); `bathroom_0` reached at most 20 px of the head's
518 400 (`max_frac` 3.9e-05 < 0.0002) → **dropped: `bathroom_0`**; candidate `rooms = [living_room_0, bedroom_0, entryway_0,
kitchen_0]`.

Pass 2 (`$V/campaigns/rooms/gate/mental_rotation-mug/{template,override,template_repeat}/`, the override run with `--load-rooms
living_room_0,bedroom_0,entryway_0,kitchen_0`; `gate.md` row): verdicts identical, trajectory 0.00 cm / 0.001 rad; first-60-step
excess over the repeat, mean / p90 / max in /255 at 224²: head +0.01 / +0.04 / +0.91, left +0.02 / +0.04 / +0.04, right −0.47 / +0.03 /
+0.22; later head frames +0.02 → **PASS** (the head's single +0.91 spike is the frame-latency effect, which the p90 ignores). The
override episode's `meta.json["load_rooms"]` reads `{"mode": "override", "rooms": [living_room_0, bedroom_0, entryway_0, kitchen_0],
"template_rooms": [the 5], "dropped": ["bathroom_0"], "added": [], "source": "--load-rooms list"}`.

The entry `decide_gate.py` wrote (`$V/campaigns/rooms/summary/rooms_override.json`):

```json
"mental_rotation-mug": {
 "rooms": ["living_room_0", "bedroom_0", "entryway_0", "kitchen_0"],
 "variant": "mental_rotation-mug",
 "template_rooms": ["living_room_0", "bathroom_0", "bedroom_0", "entryway_0", "kitchen_0"],
 "dropped": ["bathroom_0"],
 "scene": "Rs_int",
 "evidence": {"segmentation_demos": 11, "gate_episodes": 3, "head_excess_mean_255": 0.012845558233750172,
              "traj_pos_max_m": 0.0, "traj_rot_max_rad": 0.0009765625}
}
```

With `ISG_ROOMS_OVERRIDE` pointing at that file, `rooms_for_template` matches `variant == "mental_rotation-mug"` and any world of
this template loads the 4 rooms; `run_isg_vec --rooms-override <file>` matches the key for instance lists whose alias is
`mental_rotation-mug`. The task's clone count (N=8 in the campaigns so far) is not yet re-verified under the override (§6).

## 8. The command sequence, in order

```bash
# 0. a checkout with the mechanism (§0), the alias's instance lists under <root>/instances/{all,eval}/<alias>.json
ROOT=/vision/u/minyeong/projects/isg_home/vec_eval85; C=$ROOT/campaigns/rooms_<date>; S=scripts/vec_eval/rooms
# 1. segmentation pass: every demo, N=1 (once per --world-group K if the driver exits with RUN_ISG_VEC_WORLDS)
python -u -m ispatialgym.oracle.run_isg_vec --out $C/seg/<alias> --mode replay --instances $ROOT/instances/all/<alias>.json \
    --num-envs 1 --seg-visibility --seg-every 1 --render-override /rtx/raytracing/showLights=0 --record-view '' --no-grid-video --no-external-camera
python $S/analyze_visibility.py --root $ROOT --campaign $C --tasks <alias> --keep garden_0     # -> summary/rooms_override_candidate.json
# 2. gate: three replays on ONE host (or: make_gate_units.py + the vec_runs worker)
COMMON="--mode replay --num-envs 1 --dump-frames-spec 0-60:5,25 --record-view tiled --tile-px 240 --external-size 240 --grid-video --instances $ROOT/instances/all/<alias>.json --limit-batches 3"
python -u -m ispatialgym.oracle.run_isg_vec --out $C/gate/<alias>/template        $COMMON
python -u -m ispatialgym.oracle.run_isg_vec --out $C/gate/<alias>/override        $COMMON --load-rooms <rooms of the candidate, comma list>
python -u -m ispatialgym.oracle.run_isg_vec --out $C/gate/<alias>/template_repeat $COMMON
python $S/compare_gate.py --root $ROOT --campaign $C --task <alias> --a template --b override
python $S/compare_gate.py --root $ROOT --campaign $C --task <alias> --a template --b template_repeat --no-dataset
python $S/decide_gate.py  --root $ROOT --campaign $C --source "campaign rooms_<date> (segmentation pass + same-host gate)"   # -> summary/{gate.md,rooms_override.json}
# 3. if PASS: merge the entry into the production rooms_override.json, re-verify the clone count under ISG_ROOMS_OVERRIDE (§6),
#    export ISG_ROOMS_OVERRIDE in the job scripts, list the file as a frozen input in the record entry
```

## 9. Sources

- Campaign report: `$V/docs/ROOMS_REPORT.md` (§1 how rooms are loaded, §2 method, §3 results, §4 mechanism, §5 cost); decisions
  log: `$V/campaigns/rooms/DECISIONS.md` (2026-10-27 23:00 → 2026-10-28 08:35: the flags, the gate rule's derivation at 05:50 /
  05:52); results: `$V/campaigns/rooms/summary/{visibility.md, gate.md, gate_rows.json, rooms_override_candidate.json,
  rooms_override.json, before_after_{pass,fail}_step25.png}`; record entry: `$V/docs/EVAL_RECORD.md` §11, 2026-10-27 23:15.
- Code: `$R` (`ispatialgym/oracle/{vec_rooms,vec_loader,run_isg,run_isg_vec}.py`, `scripts/vec_eval/fleet/worker/eval_worker.py`),
  `$V/campaigns/rooms/code_delta_vs_E_20261028.patch`; the campaign's scripts `$V/campaigns/rooms/*.py` (this checkout's
  `scripts/vec_eval/rooms/` are their path-parameterised copies); the cluster job `$V/campaigns/rooms/jobs/run_vec.sbatch`.
- Outputs: `$V/campaigns/rooms/{seg,gate}/` (14 GB: frames, videos, trajectories, `rooms_visibility.json`), R2
  `s3://isg-demos/isg-vec-eval-rooms/`.
