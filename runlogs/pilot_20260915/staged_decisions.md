# Staged decision runs (owner request 2026-09-16 16:40, Priority 3) — prepared, NOT run

Each block: the open decision, the alias, the code path that changes per option, and the exact loop command. The
subtasks sit on `hold` in `ready_queue.json`; the minute the owner answers: apply the one-line switch (or the agent's
code change), `python scripts/isg/pod_scheduler.py release --subtask <S>` and `python scripts/isg/pod_scheduler.py add
--alias <A> --priority 15 --by owner --note "<decision>"` — the scheduler launches it on the next free pool pod (≤ 5 min).

## 1. Block design — the flipped in-hand orientation has no pre-place IK at any trunk (tabletop 14:45)

Alias: `configuration_matching-block_design-block_4-breakfast_table` (last loop alias; `…-conference_table` / `…-countertop` are the fallbacks).
Code path: `ispatialgym/sampling/v2/tasks/block_design.py:352-355` — the up-face draw
`p_goal_up = 1.0 if self.gen("block_no_flip", False) else float(self.gen("goal_face_up_prob", 0.5))`.

| option | change | loop command |
|---|---|---|
| (a) goal face up at init (no flip; yaw-align stays) | `isg_customs.json` → the alias' `generation.block_no_flip: true` (prepared `5e30334`, off) | `python scripts/isg/first_demo.py run --alias configuration_matching-block_design-block_4-breakfast_table --max-attempts 8 --timeout 7200 --note "owner: block_no_flip"` |
| (b) flip on the table (place tilted on an edge, re-grasp) | tabletop agent: new place-kind `flip_on_table` in `block_design.py` (pre-place of the *upright* orientation, second grasp record) + `programs.program_block_design` two-grasp sequence (oracle) — ~½ day, both agents | same command after the commits (fix-log entries first) |
| (c) 3 no-flip blocks | `generation.block_no_flip: true` + `generation.n_blocks: 3` (pattern of 3 distinct colours) | same command with `--note "owner: 3 no-flip blocks"` |

## 2. Attachment — the A1 brace gate rejects 902 / 1694 partner cells (bands do not overlap)

Alias: `configuration_matching-attachment-lid-teapot-breakfast_table` (last loop alias; `cork-vinegar_bottle`, `cap-lemonade_bottle`, `dip_candle-candle_holder` fallbacks).
Code path: `ispatialgym/sampling/v2/tasks/attachment.py:319-353` — the joint two-arm placement (`jp["brace_ok"]` from the left contact band; `fails["brace_off_band"]`), `program_check.PROGRAM_BLOCKS["configuration_matching-attachment"]` (brace required today).

| option | change | loop command |
|---|---|---|
| brace optional (as axial `6c3d967` / blocks) | `attachment.py`: record `contacts.brace` only when the joint placement exists, else a one-arm attach (`summary.brace: none`); `program_check`: brace optional; oracle `program_attachment` without the brace leg — tabletop + oracle, ~2 h | `python scripts/isg/first_demo.py run --alias configuration_matching-attachment-lid-teapot-breakfast_table --max-attempts 8 --timeout 7200 --note "owner: brace optional"` |
| partner only in the band overlap | `attachment.py`: partner cells restricted to right-band ∩ left-contact-band (anchor rule) before the screening — tabletop only, ~1 h; yield drops to the overlap | same command with `--note "owner: partner in the overlap"` |

## 3. Small articulation — hinged_jar best focus 24° off the optical axis vs the 20° Phase-2 cone

Alias: `configuration_matching-articulation_open_small_scale-hinged_jar-breakfast_table` (+ the close alias after the first open demo).
Code path: `ispatialgym/sampling/v2/tabletop.py:1389-1456` `phase2_manipulation_stance(..., angle_classes_deg=ANGLE_CLASSES_DEG)` (classes 8° / 20°: pairs beyond the loose class are dropped at line 1443); `articulation_small.py` calls it with the defaults.

| option | change | loop command |
|---|---|---|
| 25° loose class for small articulation | `articulation_small.py`: pass `angle_classes_deg=(8.0, float(self.gen("phase2_loose_class_deg", 25.0)))`; spec §4.5 note (small articulation only) — tabletop, 20 min | `python scripts/isg/first_demo.py run --alias configuration_matching-articulation_open_small_scale-hinged_jar-breakfast_table --max-attempts 8 --timeout 7200 --note "owner: 25 deg cone"` |
| keep 20° (yield) | none; the toolbox / laptop aliases stay the only candidates (toolbox `no_workspace`, laptop `target_placement` today) | `… --alias configuration_matching-articulation_open_small_scale-laptop-breakfast_table …` |

## Also held (not a decision, a gate)

`inter_object_relation_satisfaction-under_small_scale-teacup-saucer-breakfast_table` — owner 16:40: re-run as soon as the core's release / settle change lands; released by the coordinator when the core's fix-log entry appears.

## 4. Touching — the left-arm brace never has a feasible candidate (core 17:10, run 0916-141225)

With the A1 left contact band in place (`629b737`: 76 cells at trunk (0.25, −0.80), 66 at (0.0, −0.92), gate admitting every drawn position) 8 attempts wrote nothing: 5× `contact_unreachable` — in 3 the right‑arm `push_blade_0` chain passes but 0 of ~10 brace candidates (pads `left_v`/`left_h`, blades pitch 0, both wrist turns) pass their pre‑contact; 2× `approach_untrackable` (carried 11 cm box vs itself on the descent); 1× `goal_infeasible`. A push‑only touching would have passed the contact stage in 3/8. Owner 08:10: "the same rule may apply later to touching's brace if its joint placement never exists".

Alias: `inter_object_relation_satisfaction-touching-box_of_sanitary_napkins-box_of_tissues-breakfast_table` (16:52 run on HEAD in flight; `teacup-teapot` aliases are the fallbacks).
Code path: `ispatialgym/sampling/v2/tasks/iors_small.py` touching joint gate (`brace_off_band` / `contact_unreachable` at the brace candidates), `program_check.PROGRAM_BLOCKS["inter_object_relation_satisfaction-touching"]` (brace required today), oracle `programs.program_touching` (push closure = the stop, as axial `6c3d967`).

| option | change | loop command |
|---|---|---|
| push‑only (as axial) | record `contacts.brace` only when a joint placement exists, else push‑only with the oracle's push closure; `program_check` brace optional; oracle program without the brace leg — core + oracle, ~1 h | `python scripts/isg/first_demo.py run --alias inter_object_relation_satisfaction-touching-box_of_sanitary_napkins-box_of_tissues-breakfast_table --max-attempts 8 --timeout 7200 --note "owner: touching push-only"` |
| keep the brace (yield) | none; try the `teacup-teapot` aliases (lower target, different brace classes) | `… --alias inter_object_relation_satisfaction-touching-teacup-teapot-breakfast_table …` |
