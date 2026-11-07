# oracle-fix: the eval83 / isg-1020 aliases that never sampled, sampled slowly or failed their demos (2026-11-03)

Branch `oracle-fix` (worktree `~/Desktop/projects/ispatialgym-oracle-fix`, from `grasp-bands` 88ec8084 + its uncommitted files copied in on
2026-11-03; nothing committed). Behavior-1K symlink -> `~/Desktop/projects/Behavior-1K-grasp-bands-6a6194600` (shared, read-only use).
Data copies (never touch `~/grasp_bands`): `~/oracle_fix/{glib_v2,maps,cache,out}` = rsync of `~/grasp_bands` (minus appdata); the v2
runlog `runlogs/grasp_library_v2/` copied into this worktree; the committed legacy bands `runlogs/reach_bands_20260915/bands` are the
worktree's own checkout. Env: `source scripts/oracle_fix/env_lambda.sh [slot]` (private caches, `ISPATIALGYM_REACH_BANDS=~/oracle_fix/cache/
reach_bands`, app-data `~/oracle_fix/appdata/p<slot>` - copy a template from `~/grasp_bands/appdata/p0` (7.5 GB) before the first Kit start).

Records read (copies in the session scratchpad `rec/`): `$H=/vision/u/minyeong/projects/isg_home`: `eval85/runs/{s2_3tasks,s3_55tasks,
iors_large}/{SELECTED,MANIFEST,COUNTED_PINS}.tsv`, `eval85/COMPLAINTS.md`, `eval85/DECISIONS_LOG.md`, `reconcile83/out/counted83.tsv`,
`train85/manifests/{PROGRESS.md,progress.tsv,priority.tsv,counted_costs.tsv}`, `train85/PLAN.md`, the failed runs' `sample.log` /
`demo/*/meta.json`, `train85/out/<key>/g*/sample.log`.

## 1. What "the aliases we used" are

- 100 picks in `ispatialgym/assets/icl/roles/representatives.csv`; 85 were run for eval (30 s2_3tasks + 55 s3_55tasks; the 15 large IORS
  picks in iors_large); **83 counted** (29 + 48 + 6, `counted83.tsv` pick -> alias). isg-1020 = 53 of the 83 (the other 30 were held in
  train85, `priority.tsv`, or never got a template).
- The s2_3tasks pick names are aliases (`camera_relocalization-obstructed-<obstacle>_1`, `mental_rotation-<cat>`, `object_scaling-<cat>`);
  the `#NN` rows of those families in representatives.csv are the same tasks, not missing ones.
- Not counted and NOT a target here: `object_scaling-beefsteak_tomato` (open complaint = the owner's "blacklisted" object-scaling alias:
  representative must become ogpans; its three eval runs passed). `blacklist.csv` only lists two wall_clock models.

## 2. The 16 picks never sampled for eval (no counted instance), with the measured cause

Evidence = the eval85 run logs (rejection reasons normalised, counts per run) and demo meta verdicts.

| # | pick -> alias(es) tried | what happened | side |
|---|---|---|---|
| 1 | alignment-axial#02 -> `alignment-axial-paperback_book-desk` (hotel desk_ohjotf_0, c3), `-coffee_table` | sampler: `program_check` rejects `approach_untrackable` (push replay: pre-contact witness on another IK branch, closed-loop arc misses, escape leg stalls), `ik_joint_limit` retreat; demos: `plan_failed` push IK unreachable (best 39.5-41 mm), `tolerance` goal never held | **oracle push (A)** + sampler push replay |
| 2 | articulation_close_large#03 -> `..-bottom_cabinet`, `..-bottom_cabinet_push` (office bottom_cabinet_rvunhj_0 j_link_2, bar 8 mm at 0.73 m) | 0 instances: hybrid IK rejected by joint-limit margin (right_arm_joint2 / 7) and self-collision clearance (arm_link5/6 vs torso), `hybrid_follow_infeasible`, `handle_unreachable`, `no_manipulation_stance` | sampler (articulation_large hybrid follow, low handle) |
| 3 | articulation_open_large#03 -> `..-bottom_cabinet` | same as 2 (51 + 41 + 36 + 30 rejects in one run) | sampler |
| 4 | disembedding#06 -> `disembedding-carton_of_milk-*` x4 supports | `target_placement` (`place_unreachable` pre-place, `path_collision`); breakfast table: `RuntimeError: no stance on free floor around breakfast_table_rhjoby_0` (crash, not a reject); countertop: "destination tray fits beside the source tray at no edge" | sampler (tray layout / stance); **the RuntimeError is now a `no_manipulation_stance` reject** (`v2/tabletop.py compute_bands`) so the run goes on to the next support / edge |
| 5 | ontop_small#02 -> `..-steak-frying_pan-{commercial_kitchen_table,burner,stove}` | `gripper_not_in_frame` at the place (17/21), `goal_tipped`, `goal_unstable` (the steak stands on a 6.3 cm base) | sampler (frame rule) + asset |
| 6 | touching#03 -> `..-hardback-hardback-bookcase_inside` | 159 x `not_in_frame` (target not in the init image, shelf interior); the c2 bookcase run wrote nothing in 303 s (setup) | sampler (frame rule / shelf layout) |
| 7 | touching#04 -> `..-spray_bottle[_wegawe]-bottle_of_detergent-*` (5 supports x 2 models) | sampler: `trembling` (9-11 per run), `ik_joint_limit` / `grasp_unreachable` pre-grasp, `contact_unreachable` push; demos (3 instances written): `placed_orientation` 0.231 / 0.522 rad > 0.20, `protected_contact` (finger touched the target during the push pre-contact) | **oracle push / place (F)** + sampler |
| 8 | inside_large#01 -> hardback / bookcase_jaysra_0 | never run: representative status `incomplete:no_grasps:tabletop` (no library grasp for hardback bqvvvx) | data (grasp library; glib v2 now covers it?) |
| 9 | inside_large#02 -> `..-jar_of_spaghetti_sauce-fridge-{bar,breakfast_table[_shelf070/097],floor}` | 1000 x `grasp_unreachable: empty feasible set F` (B candidate), `time_budget`, inside stance C few survivors | sampler (bands / reach at the fridge shelf) |
| 10 | nextto_large#01 (swivel_chair / desk), #03 (coffee_table / sofa) | never run: no customs alias exists (only the 4 serving_cart aliases of #02) | grounding |
| 11 | ontop_large#02 -> `..-mouse-desk-floor` | 832 x `grasp_unreachable: empty feasible set F` | sampler (bands) |
| 12 | ontop_large#03 (pillow / bed) | never run: no customs alias (pillow soft body) | grounding / asset |
| 13 | touching_large#01 (ottoman / sofa), #02 (umbrella / hall_tree) | never run: no customs alias (only storage_box-bookcase) | grounding |
| 14 | under_large#02 -> `..-briefcase-desk[_mdhelw_0]-floor{,_flat,_pitched,_place}` (26 runs) | 1000 x `grasp_unreachable: empty feasible set F`, stance C survivors only at one edge, `time_budget` | sampler (bands / briefcase grasp) |

Priority among these for oracle work: 1 (A), 7 (F). Items 2-6, 9, 11, 14 are sampler feasibility (the grasp-bands reach maps and the
hybrid-follow / frame rules), 8, 10, 12, 13 need grounding or assets first.

## 3. isg-1020 aliases with low yield / slow sampling (train85 PROGRESS.md 2026-10-18, priority.tsv holds)

| alias | ok / allocated | s per ok | dominant cause (train85 sample.log, demo meta) | side |
|---|---|---|---|---|
| alignment-axial_04 (magazine-desk) | 13 / 300 (cap) | 14,509 | 123 `target_placement` (no yaw/position with a pushable long-axis end), 84 `repeats_eval`, **44 + 33 + 23 `approach_untrackable` push replay**, 22 `scene_disturbed` (table lamp); demos 4/4 `no_motion` (postcondition of push failed) | **oracle push (A)** |
| alignment-axial_05 (tablespoon) | 62 / 300 (cap) | 2,681 | 119 demo failures (push), 198 sample fails | **oracle push (A)** |
| nextto_01 (marker / colored_pencil) | 0 / 3 | - | demos 3/3 `not_reached` (nextto): the release clearance moves the place up to 30 mm, the goal tolerance is 30 mm | **oracle place (G) - fixed below** |
| attachment_03 (cork / wine_bottle) | 0 / 6 (+150 attempts) | - | `rest_unstable` cork (8), `path_collision`, `lift_unreachable`; eval needed 8 attempts | sampler / physics |
| block_design_01 | 0 / 3 | - | `board_placement` (11), `layout` (5) | sampler |
| inside_02 (hard_candy / jar) | 1 / 3 | 3,663 | 36 `gripper_not_in_frame` (pre-place), `approach_untrackable` pre-grasp (10) - train-fast E11: limited by the pre-place frame check | sampler (frame rule) |
| inside_04 (garlic_clove / saucepot) | 0 / 1 (+50) | - | 35 `gripper_not_in_frame` | sampler (frame rule) |
| ontop_small_04 (carving_knife / cutting_board) | 1 / 3 | 2,188 | 34 `target_placement`, 14 `goal_tipped` | sampler |
| disembedding_02 / _01 | 100 | 8,827 / 5,903 | 196 / 50 sample fails (split-cone collapse, train-fast E8, fixed on train-fast) | sampler (fixed) |
| disembedding_04/05/07-10 | 0 | - | old disembedding version, no template copy (train-fast open item) | infra |
| large IORS under_large_01/03, ontop_large_01, touching_large_03 | 0-1 | 2.5-3 h per instance | per-instance poses: stance C / bands (train-fast E5) | sampler |
| articulation_close_large_01 (door) | 100 / 110 | 2,376 | 162 sample fails | sampler |
| articulation_open_large_02 (dishwasher) | 100 / 191 | 706 | 90 demo failures | oracle (not yet triaged) |
| touching_01 (napkins / tissues) | 101 / 196 | 1,787 | 95 demo failures | oracle (not yet triaged) |
| ontop_small_03 (bar_soap / soap_dish) | 101 / 146 | 678 | 45 demo failures | oracle (not yet triaged) |
| world_vertical_03 (paper_towel_dispenser) | 102 / 146 | 502 | 44 demo failures | oracle (not yet triaged) |

## 4. Oracle-side findings and fixes

### G. nextto release clearance (nextto_01; applies to every small / large next-to alias) - FIXED, needs a demo run

`execute.clear_release_of_partner` (owner 2026-10-09, for touching) moves every place target away from the partner by what the opening
fingers would sweep into it, up to `place_partner_clear_max_m` = 0.03. For touching the push that follows extends its line by the shift
(`Motions.place_shifts`) and closes the gap; a NEXT-TO place is the goal itself, so the shift stays in the final pose and competes with
the goal position tolerance (0.03) and the NextTo gap margin (`nextto_gap_extra_m` 0.03) -> marker demos 0/3 `not_reached`.
Fix (`ispatialgym/oracle/execute.py`): `ExecuteConfig.place_partner_clear_nextto_max_m = 0.008`; `_is_nextto_place(ctx)` reads the
sidecar's subtask (`subtask_of_variant`, prefix `inter_object_relation_satisfaction-nextto`; touching shares the relation word
"next to", so the relation cannot discriminate); the cap is recorded in `ctx["partner_release_clearance"]["cap"]`. Touching unchanged.
Validate: sample 3 train instances of `inter_object_relation_satisfaction-nextto_small_scale-marker-colored_pencil-breakfast_table`
(`multiply_isg --split train`, or the train85 template copy `$H/train85/templates/inter_object_relation_satisfaction-nextto_01/`) and run
`run_isg --mode oracle`: expect `success` with `partner_release_clearance.cap` in the demo meta notes; also re-run one nextto_02 / _03 / _04
instance (they passed before: the cap must not regress them - their shifts were presumably < 8 mm, check `shift_m` in their metas).

### A. alignment-axial push (axial#02 never sampled; axial_04 13/300, axial_05 62/300) - A.1 FIXED (sampler program check), A.2 open

Two coupled problems, both in the push chain `programs.program_alignment_axial` -> `execute.execute_push` (pre-contact transit,
straight leg to first contact, `_push_arc_closed_loop`, contact retreat) mirrored by `sampling/v2/program_check.py` (push replay):
1. Sampler-side rejection `approach_untrackable: push pre-contact witness N rad from the raw IK from the stance posture (another
   branch)` (`program_check.py:35-67`): the recorded pre-contact witness is on a different IK branch than the stance posture's raw IK; the
   oracle keeps the stance branch (`arm_to(accept_other_branch=True)`), so the witness is unusable and the instance is rejected. For
   axial_04 this is 44 of ~300 attempts, for axial#02 most of the program-check rejects. The 2026-09-26 rule assumed the oracle lands on
   the stance branch; since 2026-10-12 the oracle plans straight to the recorded witness (`execute.reach_pre_pose`), and the 2026-09-27 arc
   replay simulates the closed-loop arc the rule guarded against, so the rule rejected instances the replay had validated.
   **Implemented (`sampling/v2/program_check.py`, `_flipped_push_witness` / `_rewitness_flipped_push`)**: a flipped witness is no longer
   rejected outright; the arc is replayed from the raw IK of the pre-contact seeded at the stance posture (trajopt reaches a flipped
   witness only through the pushed object - owner 2026-09-26 - so the stance branch is the one to use); when it follows, the `pre`
   witness (`witnesses["pre"]`, `single_q["pre"]` - what the oracle plans to and what `push_pre_standoff` reads) becomes that
   configuration, polished under the pre-pose residual limit, `branch` note "re-witnessed on the stance posture's branch"; when it does
   not, the reject keeps both reasons. `ISG_FLIPPED_PUSH_CHECK=strict` restores the old behaviour. Compiled, not yet run.
   Validate: re-sample `alignment-axial-paperback_book-desk` (hotel_suite_large desk_ohjotf_0, `--seed-split eval --seed-index 0/1`) and
   `alignment-axial-magazine-desk` (3 train attempts); count `re-witnessed` lines vs `approach_untrackable` in sample.log, then demo
   the written instances: the push must reach first contact from the new witness (no `plan_failed`, no `no_motion`).
2. Demo-side `no_motion` (axial_04 4/4 demos: "postcondition of push failed") and `plan_failed` (axial#02: push IK best error 39.5-41 mm
   "even without obstacles"): the contact pose the oracle re-solves at demo time (after the transit / standoff fallback - `reach_pre_pose
   ... planning straight to the pre-pose with the witness, supports present`) is not the sampler's; the arc then starts off the object's
   edge (first contact never happens -> no motion) or is unreachable. Fix direction: when the witness is on the stance branch, execute the
   push from the witness joints (no re-IK), and bound the standoff fallback by the sampler's replay (the same checks); when the first
   contact does not come within the recorded approach length + 2 cm, re-aim once at the measured edge (the `rock-back round` already
   exists for the small push: reuse it) before failing with `no_motion`.
   Reproduce on lambda: the axial_04 train job `$H/train85/out/alignment-axial_04/g0000-0099/` has instances whose demos failed `no_motion`
   (`demo/*/meta.json`); copy one instance (`inst/...-tro_state.json` + sidecar) and run `run_isg --instance ... --mode oracle --video`.

### F. touching#04 spray bottle (never sampled) - F(a) ALREADY FIXED in the eval campaign, F(b) an owner decision

Demo logs (m3_k0.demo_failed, code e5658d93 / 5a5423c1): `PROTECT plan-check fired: leg=pre_contact object=target_spray_bottle
clearance=-0.0541 m ... -> kept: via-above failed (no lift (arm_along(right): self collision` - the planner's transit to the push
pre-contact went through the bottle because the pushed object was left out of the collision world; the via-above lift failed, the plan was
kept, the finger touched the bottle (`protected_contact`). **This is already fixed on this branch**: `execute.reach_push_pre` (owner, code
85cd375b, 2026-10-12 18:xx) keeps the pushed objects in the collision world for the pre-contact transit, the via-above and the re-plans,
and falls back to the world without them only when no plan exists with them present - its docstring cites this very m3_k0 case. The later
w5_k1 demo (code 85cd375b) passed the transit and failed only `placed_orientation` 0.522 rad > 0.20 (`metric: full`) after the push.
F(b), open: a tall trigger bottle yaws under the push; its touching goal has no `pair_goal_align`, so judging the touching target yaw-free
about the vertical (the push-under gate's rule, `executor.py` `pushed_under_objects`) would pass it - owner decision; alternatively clamp the
push contact height to <= 0.4 x the target height above the support. Sampler-side (`trembling` 9-11 per run, pre-grasp `ik_joint_limit`):
re-sample one touching#04 instance on this branch (vec physics + sleep assist) before deciding anything.

## 4b. The six picks with no customs alias (owner 2026-11-03: investigate these too, be creative)

Why they have none: `ground_isg` grounds a large-IORS target only with a validated library grasp or a scene-object push rule; these
representatives have neither (hardback: `incomplete:no_grasps:tabletop`; swivel chair / coffee table / ottoman: scene or floor objects with
no push rule for their shape; umbrella: no grasp for a leaning pose; pillow: cloth asset). Building blocks that already exist and what each
pick still needs:

| pick | task | existing pieces | missing work (sampler S / oracle O / data D) |
|---|---|---|---|
| inside_large#01 hardback bqvvvx -> bookcase_jaysra_0 (shelves 0.52 / 0.94 / 1.34 m) from bed / desk / shelf / floor | push the flat book to the support edge, pinch the overhang across its thickness, lift, reorient, set it inside a shelf bay | **overhang pick** `sampling/v2/overhang.py` (owner 2026-10-13, `generation.pick_overhang_push`, `IORSLarge._overhang_pick`: push ladder to an overhang that keeps the COM inside, then the library grasp at the overhang); `Reorient(mode="line")` carry waypoints; interior-shelf place (`execute.interior_split`, owner 2026-10-11); glib v2 `side` grasp mode | D: an **overhang grasp family** in glib v2 - run `candidates.py` for the hardback with the support plane removed over the overhanging strip (a new rest constraint `overhang:<depth>`: top-down pinch across the 2.6-4.6 cm thickness with the lower finger in the free space below the slab, approach from the edge side), validate in `validate_tabletop` on a slab whose edge the book overhangs by the drawn depth; S: `pick_overhang_push` mode `only` for this alias + the goal pose inside a bay (flat on the shelf: `init=inside` rows exist for bookcases; upright needs a 90 deg reorient about the pinch axis = `Reorient` waypoints the sampler already records for carries); O: **nothing** - `program_iors_large_inside` already runs the overhang pick (`contacts.push_pick`, kind `pick_overhang`, via `_pick_actions`: push -> retreat -> [travel to the `pick` stance] -> grasp -> lift) and the `Reorient` carry from `grasps[0].waypoints`, then the drop inside. Start: the hardback's overhang grasp family (D) - without it nothing can be sampled. |
| nextto_large#01 swivel_chair_iiihwn_0 next to desk_uqcmzf_0 (bedroom_2) | the chair is a scene object the sampler moves at init; goal: next to the desk | `program_iors_large_nextto`: the serving-cart **base push** (`BasePush`, one or two hands, `contacts.push.second_hand`), `BaseFollow` | S: a push rule for a swivel chair (5-star base rolls; push on the seat back at 0.4-0.5 m with two hands as the cart; the rolling base needs the chair's wheel joints free or the chair treated as a rigid slider with low friction - check `customs` physics of iiihwn); grounder: allow a scene object as the pushed target (the nextto_large sampler already handles a spawned cart on the floor, the chair only differs by being pre-existing: bind it as `target` with `init=floor`, remove nothing). O: no change if the push rule yields a `base_push` record. |
| nextto_large#03 coffee_table_osroux_0 next to sofa_ukfaaz_0 | as above with a 1.17 m^2 coffee table (heavy, wide) | same | S: two-hand base push on the table's long side (contact at 0.36 m top edge is below the hands' comfortable push height - push on the apron / top edge with the fingertips closed, the brace-map `top press` classes of the grasp-bands contact maps give the contact classes); mass check (OmniGibson coffee tables 20-40 kg: the 250 kg base pushes them, friction allowing). O: none beyond `BasePush`. |
| ontop_large#03 pillow mltefs on bed_kddswt_0 | place a pillow on the bed | `program_iors_large` ontop (drop / support place) | D: the pillow is a cloth asset (no rigid grasps; the library validator cannot pinch cloth). Options: (a) ground the pick with a rigid stand-in of the same category (a `pillow` model with `prim_type` rigid, if the inventory has one - check `scene_inventory`/`candidates.csv` `kind`), (b) treat the cloth pillow as a rigid box for sampling and grasping (OmniGibson cloth objects can be loaded as rigid via `prim_type=RIGID` in the customs physics) - owner decision. O: none. |
| touching_large#01 ottoman miftfy against sofa_lugrhk_0 | floor-standing block pushed until touching the sofa | `_iors_large_touching_program` (drop + `push_touching`), `BasePush` | S: the ottoman starts on the floor (no drop) -> the touching_large sampler needs the floor-start variant (storage box had a tabletop start); the push is a base push on the ottoman's side at 0.3-0.45 m (the cart rule's contact-height check must accept it). O: **done** - `_iors_large_touching_program` takes a sidecar with no grasp record + a `push_touching` contact as the push chain alone (init -> push stance -> push -> retreat); it needs the sampler to record a floor start and a push stance for the ottoman before it can run. |
| touching_large#02 umbrella ziqzqo leaned against hall_tree_daxzaw_0 | carry the long umbrella and lean it on the hall stand | grasp `umbrella__ziqzqo__t` exists (top pinch of the shaft); carry waypoints; place kinds support / drop / attach | S+O: a **lean place** (new `Place.kind="lean"`): the tip set on the floor 0.15-0.25 m from the stand's foot, the shaft tilted 10-15 deg so the upper shaft rests on the stand's rail / hook, released only after `touching(umbrella, hall_tree)` holds and the object is still (reuse the drop's settle probe); sampler: goal = `touching` + a tilt interval, feasibility = the two contact points (floor, rail) from the stand's mesh. Alternative needing no new kind: lay the umbrella on the floor touching the stand's foot (`drop` + `push_touching`, as the storage box) - the owner should say whether "leaned" is required. |

Grasp library v2 check (cluster `$GBH/out/roles_all_20261102/glib/models`, 2026-11-03): `hardback__bqvvvx__s1` (and acbrnv, agbqsy),
every `magazine__*__s1` and `paperback_book__jhaltj__s1` have **0 candidate families, 0 trials, 0 library grasps** - flat wide slabs: the
generator's support-plane rule leaves no top or side pinch on a book lying on a table (GRASP_BANDS.md: 126 new models without a
representative, mostly books / board games / pictures). So inside_large#01 cannot be grounded from the existing library under any switch;
the overhang family is the required data work, and it is the general answer for every flat target (the touching#03 hardbacks too).
`umbrella__ziqzqo__s1` has 10 families tested and **1 valid v2 grasp** (`library.json`), so touching_large#02's blocker is only the lean
place, not the grasp.

Overhang grasp family - implementation spec (`ispatialgym/grasping/candidates.py generate`, the constraint branches at the
`attachment_free` / `flower_stem_top` tests around lines 363, 465-505; the manifest job field `constraint`, `grasping/manifest.py`):
- job `constraint: "overhang"` (+ `overhang_depth_m`, default 0.06; one job per horizontal rest-frame axis and sign = up to 4 families,
  or one family with the edge axis as the family parameter). In the rest frame (`geom0.rotated(R)`, tabletop z = z_min) the edge line is
  `u = u_max - depth` along the chosen axis `u`; the strip `u > u_max - depth` is the overhang.
- the support-plane rules (the top-mode retreat along the approach `shift`, the side-mode lift, the closing-sweep / finger-blade checks
  against the slab top) apply only to hand hull points with `u <= u_max - depth`; beyond the edge the half-space below the slab top is
  free. Keep: object clearance (`_clear_grouped`), pad sweep, lever cap, policy geometry. Add a filter: both pads' contact points and the
  lower finger's whole closing sweep lie in the strip (the finger must pass under the slab in free space) and the pinch closes across
  the book's thickness (`width` within 3 mm..aperture - 2.6-4.6 cm hardbacks qualify).
- `validate_tabletop`: for an overhang job the slab's near edge is set at `u_max - depth` of the object (the object placed so the strip
  overhangs the slab edge; slab half size and centre adjusted), the measured rest is taken WITH that overhang (the COM margin rule of
  `v2/overhang.py` keeps it from tipping: assert COM >= 1 cm inside the edge), then the usual closure / hold / unload trial; `rests.json`
  records `overhang: {axis, sign, depth_m}` so `schema.place_object` and `ReachMap.find` key on it (a family only valid at an overhang).
- sampler: `generation.pick_overhang_push` mode `only` for inside_large#01's alias (the push ladder of `overhang.py` already produces the
  overhang start; it takes "the library grasp at the overhang" - point it at the `overhang` family's key via `grasp_keys`).
- oracle: already there (`programs._pick_actions` + `program_iors_large_inside`); an upright shelf placement only needs the sampler to
  record the 90 deg in-hand turn as `grasps[0].waypoints` (the `Reorient(mode="line")` carry).

Suggested order: touching_large#01 (program branch done; sampler floor start next), inside_large#01 (overhang grasp family: the general pattern for every flat target),
nextto_large#01/#03 (push rules), touching_large#02 (lean place), ontop_large#03 (asset decision).

## 5. What the next session (validation / easy jobs) should do, in order

0. Code changed in this worktree (all uncommitted, on top of grasp-bands' own uncommitted files): `ispatialgym/oracle/execute.py` (G),
   `ispatialgym/sampling/v2/program_check.py` (A.1), `ispatialgym/oracle/programs.py` + `ispatialgym/oracle/sidecar.py` (touching large floor start: the program branch and the validator's
   required blocks without grasps / paths / placement when a `push_touching` contact and no grasp are recorded),
   `ispatialgym/sampling/v2/tabletop.py` (band-stance reject), `ispatialgym/tests/test_iors_large_touching.py` (new test
   `test_touching_large_floor_start_is_the_push_chain_alone`), `tests/test_v2_core.py` (new `test_rewitness_flipped_push_...`),
   `tests/test_oracle_loads.py` (new `test_is_nextto_place_...`). CPU tests run 2026-11-03 18:45 in the `ispatialgym` env: test_sidecar_programs +
   the flipped-push / partner-clear / touching-large tests 242 passed; test_v2_core push/witness subset 8 passed; test_v2_tabletop 62 passed;
   test_iors_large_touching + test_sidecar_programs 256 passed (18:47). Nothing has run in the simulator yet.
1. Smoke the worktree: `source scripts/oracle_fix/env_lambda.sh 0` (copy `~/grasp_bands/appdata/p0` to `~/oracle_fix/appdata/p0` first),
   `python -m pytest ispatialgym/tests/test_v2_oracle.py -q -k "place or release"` (CPU), then G's demo validation above.
2. Run G on nextto_01 (3 demos) and one instance each of nextto_02/03/04; record pass/fail + the `partner_release_clearance` note here.
3. A.1 validation (above) on axial#02 c3 and three axial_04 train attempts; then reproduce A.2 on one axial_04 `no_motion` instance
   on lambda, attach the push log lines (`first contact after`, `closed-loop arc ended`, `postcondition`) here, and implement A.2
   (execute from the witness, re-aim once at the measured edge).
4. touching#04: one sample + demo of `..-spray_bottle-bottle_of_detergent-commercial_kitchen_table` on this branch (F(a) is in the
   code already); report the `placed_orientation` error to the owner with the yaw-free question (F(b)).
5. Triage the four "oracle (not yet triaged)" rows of §3 from their train85 demo metas (`$H/train85/out/<key>/g*/demo/*/meta.json`,
   group by `verdict.category`).
6. §4b in its suggested order: touching_large#01's floor-start program branch first, then the hardback overhang grasp family.
7. Everything else in §2 / §3 is sampler- or data-side; log it for the owner.

## 6. Opus session 2026-11-03 19:00- (record: cluster `isg_home/oracle_fix/DECISIONS.md`, runs under `isg_home/oracle_fix/runs/`)

Cluster copy `/vision/u/minyeong/projects/ispatialgym-oracle-fix` (detached 43ad206a + rsync of ispatialgym/ scripts/ docs/ configs/
runlogs/grasp_library_v2/; Behavior-1K -> Behavior-1K-grasp-bands-6a6194600). The 21 files the 2026-11-01 small-scale rename renamed were
left behind by the rsync and broke sidecar validation ('cannot read the iddl id'); removed (list `isg_home/oracle_fix/stale_removed_20261103.txt`).
**The grasp-bands cluster copy has the same 21 stale files.** Validation jobs: `scripts/oracle_fix/job_val.sh <runs/name>` (KIND demo /
train / eval; sample_isg takes --variant, run_isg has no --video / --dev on this branch), env `scripts/oracle_fix/env_cluster.sh`.

New / changed since §5:
- **G corrected**: the nextto_01 train metas show the opening fingers sweeping 43-47 mm into the pencil at every release, so the 8 mm cap
  alone cannot fix it. The fix is data: `generation.raised_release` (nextto#03's key) on the four `...-nextto_small_scale-marker-colored_pencil-*`
  aliases (A/B jobs nx01_rr / nx01_norr; `ISG_CUSTOMS=isg_home/oracle_fix/customs_no_rr.json` is the control). The 8 mm cap stays.
- **A.2 implemented** (`execute.py` `_push_recontact_rounds`, `ExecuteConfig.push_recontact_rounds` 2, env `ISPATIALGYM_PUSH_RECONTACT_ROUNDS`):
  measured on the old axial#02 c3_k0 instance with the current code: every action ok, the closed-loop arc ended `door_lags` with the book
  turned 0.32 of 0.69 rad (a free book pushed at one end slides as well as turns; the pad's circle about its centre loses it), re-push
  skipped (beyond the 0.25 x arc cap) -> not_reached 0.371 rad. Re-contact rounds: while the residual is beyond the goal's orientation
  tolerance, the recorded pre-contact / contact / arc are moved rigidly with the object, the arc shortened to the residual about its current
  centre, the pad backed off, planned to the moved pre-contact (target protected) and the closed loop run again; free-object arcs only.
  Test `test_push_recontact_rounds_move_the_contact_with_the_object`. A/B: replay_ax04 (baseline, code before the change) vs replay_ax04_rc,
  replay_ax02_rc.
- **Place re-aim** (opt-in `ISPATIALGYM_PLACE_INHAND_REAIM=1`, `reaim_place_for_inhand_pose`) + the in-hand rotation diagnostic in
  `place_inhand_raise` (`inhand_rot_rad`, `inhand_xy_m`): for ontop_small_03's bar soap (36 placed_orientation, median 0.35 rad). A/B:
  replay_os03 / replay_os03_reaim.
- **New aliases for the never-grounded picks** (hand-made like the eval campaign's large IORS aliases; problems 151-157):
  inside_large#01 hardback -> bookcase (desk 151 / bed 152; the legacy `hardback__bqvvvx__bo` book_overhang grasp + `pick_overhang_push`
  only = the owner's push-to-edge idea, already in the sampler and the oracle), ontop_large#03 pillow -> hotel bed (153, `pillow__mltefs__lp`),
  touching_large#02 umbrella -> hallstand (154, the storage box's drop + floor push; the v2 grasp served from a COPY of its v2 entry with a
  legacy key, `ISPATIALGYM_GRASP_LIBRARY_V2=isg_home/oracle_fix/glib_v2`), nextto_large#01 swivel chair -> desk (155) and #03 coffee table
  -> sofa (156) as spawned targets of the scene models (scene instances removed; the serving cart's two-hand base push),
  touching_large#01 ottoman -> sofa (157) with a **new sampler mode** `iors_large_nextto.TouchingBasePushMode` (`generation.touching_push:
  "base"`; the base push with a touching goal, nudged to contact) and the oracle's base push stopping on Touching (`relation: touching`).
- **Triage of four isg-1020 aliases** (train85 metas): touching_01 (69 of 95 failures) and world_vertical_03 (43 of 44) failed
  `protected_contact` at the push pre-contact on snapshots that predate 85cd375b (this branch has it) -> replays replay_tch01 / replay_wv03;
  articulation_open_large_02 (dishwasher): 54 robot_wobble, **all marginal** (torso 1.05-1.53 deg vs the 1.00 deg gate) - an owner
  decision on the QA threshold, plus 16 plan_failed / 14 gripper_out_of_fov in base_follow (replay_dw02).

### 6.1 Results so far (20:10; cluster runs `isg_home/oracle_fix/runs/<name>`, summary `python3 isg_home/oracle_fix/rsum.py runs/*/`)

| run | what | result |
|---|---|---|
| ax02_a1 / ax02_strict | axial#02 (never sampled in eval85) on hotel desk_ohjotf_0 T3, eval seed 0, N=2 | 2 / 2 written in 5.6 min each, **4 / 4 demos success** (both arms; no flipped witness drawn) |
| replay_wv03 | world_vertical_03: 4 train85 protected_contact failures + 1 success | **5 / 5 success** (85cd375b on this branch) |
| replay_tch01 | touching_01: train85 protected_contact failures | 0116, 0117 success so far |
| replay_ax04 | axial_04 0307 / 0315 no_motion, 0308 tolerance, 0327 success - code before the re-contact | 0 / 4 (0327 plan_failed) |
| ax04_a1 / strict | axial_04 new train-seed samples, N=4 | 4 / 4 written each; re-witness fired 2x in the stale-tree run; demo 0000 no_motion in both (the pad circles outside the 1 cm magazine's end: frames `isg_home/oracle_fix/frames/`); with re-contact a partial turn (residual 0.185) was finished to 0.004 rad |
| replay_os03 / _reaim | soap: 4 placed_orientation + 1 success | 2 / 5 and 3 / 5; in-hand rotation 0.008-0.015 rad (not the cause); the dish is shoved 25.7 mm during the descent |
| replay_dw02 | dishwasher: 4 robot_wobble, 1 plan_failed, 1 success | 1 / 6; every wobble = ONE step of 1.003-1.093 deg (limit 1.0; the success peaked at 0.914) |
| regress_nextto | counted nextto#01-#04 eval instances | nextto#04 success; **nextto#01 (pair1013_c1_k5) symbolic_only**: marker orientation 0.382 rad, position 0.3 mm (rolled about its axis after the release; the release clearance did not fire) - a regression of this branch's physics / release on that instance, not of oracle-fix |

Changes made after §6 (all CPU-tested):
- `_push_recontact_rounds`: the base follows the object's planar shift (1-15 cm, collision-checked `realign_base_clear`, hand lifted 5 cm) -
  axial#02's first round had the moved pre-contact 25 mm out of reach; and a round that found the object untouched moves the contact
  `push_recontact_inward_m` 2.5 cm toward the pivot (the magazine miss); rounds default 3.
- `place_touch_stop` (opt-in, `generation.place_touch_stop` / `ISPATIALGYM_PLACE_TOUCH_STOP=1`): a support place onto a partner ends its
  descent at the first Touching (or a 1.5 mm partner move) and releases there - for the soap; A/B replay_os03_touch, replay_os03b(_touch).
- `QAThresholds.wobble_min_steps` (default 1 = today's rule; `ISPATIALGYM_QA_WOBBLE_MIN_STEPS`): robot_wobble fires once the torso stayed
  over 1 deg that many consecutive steps; replay_dw02_min3 (=3) measures what the owner's rule costs - **owner decision**.
- `NOTE_KEYS` export `place_touch_stop`, `place_inhand_reaim`, `push_recontact`.

Open (sampler-side, not changed): pillow from the floor - 1000 x `grasp_unreachable: empty feasible set F_init` (the floor-level reach of
the large IORS bands, as under_large#02 / inside_large#02); coffee table - no push face at 0.24-0.345 m (needs a face-probe diagnostic of
its collision pieces); swivel chair - `stance_reach`; hardback from the desk - no free edge stance at desk_bhkhxo_0 (the bed variant runs).

### 6.2 Results 20:30

| item | measured | conclusion |
|---|---|---|
| touching_01 | replay_tch01 5 / 5 (4 were train85 protected_contact) | fixed on this branch (85cd375b); re-collect |
| world_vertical_03 | replay_wv03 5 / 5 | fixed on this branch; re-collect |
| axial#02 | ax02 2 arms x 2 / 2 demos success | samples and demos on this branch (eval85: never) |
| axial_04 | new samples 2 / 4 per arm; replay_ax04 baseline 0 / 4; replay_ax04_rc: 0308 no_motion -> not_reached | re-contact (with the inward step) reaches the magazine; still short of the goal - pending |
| nextto_01 | nx01_rr 3 / 4, nx01_norr 2 / 4 (+1 goal_occlusion); release clearance overlap 0 in 8 / 8 | the train85 cause does not occur here; the limit is the **marker roll** at the place release (pinch off-centre: fingers 7.5 / 6.3 mm at the attach) - open |
| soap (ontop_small_03) | place touch stop: no effect (0 / 6, 2 / 5 = baseline); re-aim: 3 / 5 vs 2 / 5 (noise level) | the soap turns while settling in the ridged well -> `partner_aligned_to_target` A/B (soap_align / soap_base) pending |
| dishwasher (open_large_02) | wobble_min_steps 3: 1 / 6 = baseline - the bend grows (1.003 -> 1.098 deg in 3 steps) | a real base-follow load, not a QA spike; the knob stays default 1. Needs a load / follow-speed change - open |

Keep / drop: keep the re-contact rounds (default on), the nextto cap and raised release (guards), the touching base push, the aliases;
the place touch stop and the in-hand re-aim stay opt-in (no measured benefit); the wobble knob stays default 1.

## 7. Owner decisions 2026-11-03 21:05 (five items) - what was done and measured

Record: cluster `isg_home/oracle_fix/DECISIONS.md` (21:05-), runs `isg_home/oracle_fix/runs/<name>`.

Status at 2026-11-04 00:00 (details in 7.1-7.7):

| item | outcome |
|---|---|
| 1 dishwasher | already arm-only / sequential (0 base follow in 75); with the 0.35 rad arm-only cap + 0.30 rad segments: 4 / 4 success vs 3 / 4 without, 1 / 6 in the isg-1020 replay |
| 2 marker | orientation 0.95 rad (per alias only; the pencil too); the counted instance now succeeds |
| 3a annotations | B6 `base_push` rows for coffee table / ottoman / swivel chair (26); not role `push` (that switches on the axial / world_vertical requirement) |
| 3b stand-offs / trunks | tuned from a reach-margin table; kept for the coffee table and chair, reverted for the ottoman (defaults 2.5-5x faster, same result) |
| 3c swivel | iiihwn is one rigid link |
| coffee table 156 | re-bound to house_single_floor `sofa_hiphpn_0`, push to the goal: **4 / 4 instances, 4 / 4 demos** |
| ottoman 157 | re-bound to `sofa_hiphpn_0`, goal at the sectional's material, touching via the task predicate, program-check fix: **4 / 4 demos** (2 per knob set) |
| swivel chair 155 | open: 0 instances in 109 attempts (contact out of the head camera's frame, then push IK) |
| 4 hardback desk 151 | chair removed and a shifted start stance added; now blocked by `goal_region_not_visible` (the goal bookcase behind the robot) - owner decision |
| 5 pillow 158 | 6 attempts in 2.25 h, all `target_placement` at the far bed; no instance |

### 7.1 Dishwasher (articulation_open_large#02): arm only, else sequential

- **Where the mode is chosen**: `sampling/v2/tasks/articulation_large.py` (door class cabinet -> arm; `hybrid_follow`: `arm_alone_follows`
  first, then `base_follow_policy` = sequential: arm segments with `release_move_regrasp` between them). Recorded per instance in
  `<iid>.oracle.json` `articulation.follow_mode` (`arm` | `hybrid`) and `articulation.base_follow_path` (`mode hybrid`, `policy
  sequential`, `segments`). The program emits a `BaseFollow` action named `base_follow` in both modes.
- **Audit**: the 75 isg-1020 instances (0073-0147) are 31 arm-only and 44 sequential, none base follow. The alias already followed the
  rule. The six replayed in `replay_dw02` (1 / 6): 0074 / 0075 / 0078 / 0079 / 0083 arm-only, 0101 sequential.
- **Change** (customs, open-dishwasher alias): `hybrid_follow.arm_only_max_travel_rad` 0.35 (a longer arm-only pull goes sequential;
  `articulation_large.arm_alone_follows`) and `seq_max_pull_rad` 0.30 (each arm segment ends there; `articulation_hybrid.run_sequential`).
  Reason: the torso bend grows with the pull (0075 crossed 1 deg after 0.43 rad). `QAThresholds.wobble_min_steps` stays 1.
- **A/B** (train seeds 0-3, N = 4, demos): `dw_cap` (the change) vs `dw_nocap` (`ISG_CUSTOMS=customs_no_dw_cap.json`).

| run | instances | follow | demos |
|---|---|---|---|
| replay_dw02 (isg-1020) | 6 | 5 arm, 1 sequential | 1 success, 4 robot_wobble, 1 plan_failed |
| dw_nocap | 4 | 3 sequential, 1 arm (0.471 rad) | 3 success, 1 robot_wobble (the arm-only one, 1.01 deg) |
| dw_cap | 4 (0 rejections) | 4 sequential (arm segments 0.236-0.275 rad) | **4 success**, 0 robot_wobble, 0 plan_failed |

Keep the cap: the one failure without it is the arm-only pull it removes. Four demos per arm is a small sample.

### 7.2 nextto_01 marker roll: orientation tolerance

- `evaluation.tolerances.pose.orientation` = 0.95 rad on the four `...-nextto_small_scale-marker-colored_pencil-*` aliases: the smallest
  value covering the measured rolls 0.382 / 0.386 / 0.949 rad.
- **Per alias only.** `eval/tolerance_profile.py` `OVERRIDE_KEYS` has the sections pose / layout / joints, and a section applies to every
  pose target of the alias, so the colored pencil gets 0.95 too (its errors 0.000-0.015 rad). A marker-only value needs a per-instance
  section (code change, not made).
- Re-check: `tol_nx01_counted` (the counted instance, `regress_nextto` 0000, symbolic_only at 0.382) -> **success** (0.382);
  `tol_nx01_norr_0000` (0.386) -> success; `tol_nx01_rr_0002` (was 0.949) -> success, but this replay did not roll (0.050).

### 7.3 Large-scale base pushes (coffee table 156, swivel chair 155, ottoman 157)

**a. Push-spot annotations (icl B6 slot).** Same path as the handle annotations: a per-model JSON under `runlogs/contacts_<date>/`,
collected into `ispatialgym/assets/icl/require/contacts.csv` by `scripts/icl/build_icl.py` (`Builder.build_contacts`; columns `face`,
`height_m`, `height_range_m`, `inset_m`, `spacing_m`, `spots` added). Only contacts.csv was rewritten (26 rows); no other icl CSV.
- Geometry: the sampler dumps the target's collision pieces in its root frame (`ISPATIALGYM_PUSH_FACE_DUMP=<dir>`,
  `iors_large_nextto.dump_push_geometry`; copies in `~/oracle_fix/push_geometry`).
- Spots: `scripts/oracle_fix/push_spot_annotate.py` probes every face and height with the sampler's own `face_probe`. It keeps the best
  height plus up to two more at least 5 cm apart, each with its acceptable height range. It writes `runlogs/contacts_20261103/` with a
  review table `push_spots__<model>.md`.
- Sampler: `iors_large_nextto.annotated_faces` pushes ONLY at the annotated spots of an annotated model. Without a file (the serving
  cart) the automatic probe runs as before. The stance searches use the first two spots per face, so the file's order is the preference.

| model | geometry | annotated spots (root frame) |
|---|---|---|
| coffee_table osroux | two end frames under a 1 cm glass top; no long-side rails; legs 1.33 m apart | every face at the glass edge, h 0.35 (0.350-0.3525), two hands 0.40 apart (probe +-3 mm vertical). The end rails 4.5 cm under the overhang are not used |
| ottoman miftfy | flush box on all faces 0.15-0.31 m | 0.27 / 0.22 / 0.17 m per face, two hands 0.34 (x faces) / 0.30 (y faces) |
| swivel_chair iiihwn | one rigid link | one hand only (no flat pair of 0.22 m or more): +x seat front 0.45 (inset 0.039) / 0.38; -x back 0.60 / 0.65 / 0.70; +-y armrest 0.66, seat side 0.44-0.49 |

**b. Stand-offs and trunks.** The reach margin is the stretched arm (0.7929 m) minus the shoulder-to-contact distance at the stance's
trunk, accepted in [0.03, 0.45]. Tabulated offline (`conventions.torso_fk`) and confirmed by the chair run's logged margins
(`iors_large_nextto.note_reach_margin`, logged with every `no_push`):
- ottoman, 0.19-0.29 m: only trunks (0.95, -0.3), (0.95, -0.6), (0.75, -0.6), (0.95, 0) reach. Six of the eight default trunks never do.
- coffee glass edge, 0.35 m: (0.75, -0.3), (0.95, 0), (0.95, -0.3), (0.75, -0.6), (0.95, -0.6).
- chair seat / back: most trunks reach; (0.25, 0) never does (max 0.007 at stand-off 0.55) and (0.5, 0) rarely. The chair's large
  `stance_reach` count is the cheap filter; what blocks it is downstream (`push_ik`: pre-contact fingertip out of frame or exact IK;
  "every push stance collides" on +x; `goal_not_floor`).

Tuned per alias in `isg_home/oracle_fix/customs_push_tuned.json` (A/B against the annotation-only runs):

| alias | push_stand_offs_m | push_trunks |
|---|---|---|
| coffee table 156 | 0.45, 0.40, 0.50, 0.55 | (0.75,-0.3) (0.95,0) (0.95,-0.3) (0.75,-0.6) (0.95,-0.6) |
| swivel chair 155 | 0.45, 0.50, 0.55, 0.40, 0.60 | (0.75,-0.3) (0.95,0) (0.75,0) (0.5,-0.6) (0.95,-0.3) (0.5,-0.3) |
| ottoman 157 | 0.40, 0.45, 0.35, 0.50 | (0.95,-0.3) (0.95,-0.6) (0.75,-0.6) (0.95,0) - **reverted to the defaults** after the A/B in 7.7 |

**Partners (found with the annotations in place).** With the coffee table annotated, `no_push_face` went from 48 to 0 per draw, but
92-95 of about 96 face x goal draws then failed `goal_not_floor` / `init_not_floor`. The opt-in `ISPATIALGYM_NEXTTO_FLOOR_DEBUG=<n>`
(`iors_large_nextto._floor_debug`, run `coffee_floor_dbg`) logs the failing floor points:
- `sofa_ukfaaz_0` is a 3.55 x 2.77 m L sectional in the room corner. Its west and south sides are outside the room, the east side runs
  into the armchair and desk, and the north side leaves 1.2 m before the bookcases: goals fit there broadside, but every start 0.6-1.8 m
  further back lands in the bookcases. The scene's own coffee table stood in the L's inner pocket, inside the sofa's AABB, where the
  sampler never places a goal.
- The ottoman's `sofa_lugrhk_0` has no side with 2.3 m of free floor over 0.8 m.
- A free-floor scan of every scene's sofas (depth beyond each AABB side on `floor_trav_0`; the script is in the session scratchpad,
  copied to `isg_home/oracle_fix/partner_scan.py`) finds 3.2 m over 1.6 m (a broadside coffee-table push) only at restaurant_hotel
  `sofa_wjytsn_0`, office_bike `sofa_kupqto_0` and hotel_gym_spa `sofa_hxlckz_0`. The best house sofa is house_single_floor
  `sofa_hiphpn_0` (+x side, 1.55 m wide at 2.3 m or more).
- Re-bound: coffee table 156 to house_single_floor `sofa_hiphpn_0` (house_double_floor_lower `sofa_frlxgz_0` was tried and dropped: its
  free floor lies outside living_room_0); ottoman 157 to `sofa_hiphpn_0` (same room as before). restaurant_hotel is not usable: the sampler loads scenes from
  `2026-challenge-task-instances` (hotel_suite_large, house_double_floor_lower / _upper, house_single_floor, office_cubicles_right,
  restaurant_diner, Rs_int). The tuned stand-offs / trunks are now in the main customs for 155 / 156 / 157. Results in 7.6.

**c. Swivel check.** swivel_chair iiihwn has one link (`base_link`), no seat-base joint; it slides as one rigid body. coffee_table osroux
has `base_link` plus the glass link.

### 7.4 Hardback from the desk (alias 151, desk_bhkhxo_0)

- `generation.remove_objects` = `eames_chair_hndxiw_0`, the only chair within 1 m (the scene inventory's `CHAIR_CATEGORIES` lacks
  eames_chair, so its `chairs_within_1m` missed it). `hb_desk`: still `no_free_edge`.
- New opt-in `ISPATIALGYM_STANCE_BLAME` (tabletop Phase 1a names the colliding body, `IKChecker._collision_blame`). `hb_desk_blame`:
  the +u / +v / -v stances are outside the house (lawn / roof / walls); the -u stance collides with `bookcase_jaysra_0`, the GOAL bookcase,
  0.56 m from the base.
- New opt-in `generation.init_edge_laterals_m` (`tabletop.shifted_init_stance`): a colliding Phase-1a midpoint stance is shifted along
  the edge (within the edge, 0.30 m from its ends); Phase 1c still draws the lateral. Alias 151: [0.25, -0.25, 0.4, -0.4, 0.55, -0.55].
  `hb_desk2`: the -u stance passes shifted +0.25 m and Phase 1c draws laterals 0.28-0.73, then setup fails with
  `goal_region_not_visible`: from the desk's only open edge the goal bookcase stands behind and to the right of the robot, out of the
  start view. **Owner decision**: the start-view visibility rule is spec and was not relaxed (relax it for alias 151, or pick another
  desk / scene).
- The bed variants 152 / 159 / 160 ended with no stance C at the bookcase (setup).

### 7.5 Pillow from the other bed (alias 158, `pillow_bed`)

Setup OK (hotel_suite_large, bed_kddswt_1 ~T2; no setup error; the instance `summary.json` is written at the job's end). 4 attempts in
1.6 h (1007-1210 s each), all `target_placement` / `grasp_unreachable` at C: the pre-place / place exact IK with the pillow attached fails
at pose 1 of 2 (both arms, trunks (0.5 | 0.75, -0.3 | -0.6), stand-off 0.55), and at stand-off 0.35 the carried sweep collides with
bed_kddswt_0 (0.32 m from the base). No instance, no demo.

Final (22:27): 6 attempts in 2.25 h (1007-1226 s each), all `target_placement`. The job was killed at its wall limit
(`SAMPLE_EXIT 124`, 8103 s), so neither the SAMPLING_REPORT nor the instance `summary.json` was written. Per place stance C over the last
two attempts: stand-off 0.45 - the carry leg fails (11 of 17); 0.35 - the carried sweep collides with bed_kddswt_0 (3); 0.55 - the
pre-place / place exact IK with the pillow attached fails at the place pose (3). Open: the pillow's place on the far bed is out of
reach from every stance the sampler tries.

### 7.6 Results after the re-binding (22:45)

- **Coffee table 156 on house_single_floor `sofa_hiphpn_0`** (`coffee_hsf`): 2 / 2 instances, the first on attempt 0 (100 s). Demo 0000:
  the two-hand base push ran (1.428 m of 1.672 m, drift 1 cm, turn 1.4 deg, NextTo true) and stopped 0.05 m after NextTo first held, so
  the table ended 0.244 m short of its goal: `symbolic_only` (pose_reached position 0.244 > 0.05). NextTo's bound grows with the object,
  so on a large object it first holds about 0.3 m before the goal. Fix: `generation.nextto_push_on_m` 2.0 on 156 and 155 (the drive ends
  at the line's recorded end, where the sampler checked NextTo). Resample `coffee_hsf2` (N = 4).
- **Coffee table on house_double_floor_lower `sofa_frlxgz_0`** (`coffee_hdfl`): every goal `goal_outside_rooms` / `goal_not_floor` - its
  free floor lies outside living_room_0.
- **Ottoman 157 on `sofa_hiphpn_0`**: draws now reach the goal check, but every touching check failed after 25 x 1 mm nudges (19 / 19
  `goal_infeasible`): the sofa side at ottoman height lies more than 2.7 cm inside its AABB. `nudge_m` 0.002 x `max_nudges` 100 on 157;
  A/B `ottoman_hip2` (tuned knobs) / `ottoman_hip2_def` (default knobs). Before the nudge fix the default knobs reached the goal check
  faster (40 s against 299 s per attempt).
- **Swivel chair 155** (tuned): `stance_reach` no longer dominates; the head camera framing of the contact does
  (`stance_contact_out_of_frame` 650-1220 per draw), then `push_ik`.

### 7.7 Results 23:45

| alias | run | instances | demos |
|---|---|---|---|
| coffee table 156 (house_single_floor `sofa_hiphpn_0`, `nextto_push_on_m` 2.0) | `coffee_hsf2` | 4 / 4, 0 rejections | **4 / 4 success**: every push ran its full line (1.663-1.684 m, "distance done"); final error 0.006-0.015 m / 0.004-0.034 rad |
| coffee table 156, push-on 0.05 (before) | `coffee_hsf` | 2 / 2 | 0 / 2, both `symbolic_only` (0.244 / 0.261 m short) |
| ottoman 157 (`sofa_hiphpn_0`, surface goal, default knobs) | `ottoman_surf2_def` | 2 / 2 on attempt 0 (50 / 20 s), two hands, trunk (0.95, -0.3) | **2 / 2 success**, pushes ran their full line (1.385 / 1.686 m), error 0.007 / 0.002 m, 0.010 / 0.007 rad |
| ottoman 157 (surface goal, tuned knobs) | `ottoman_surf2` | 2 / 2 on attempt 0 (137 / 106 s), trunk (0.95, -0.6) | **2 / 2 success**, error 0.002 / 0.003 m, 0.004 rad |

Owner decision 3b for the ottoman, measured: both knob sets succeed, the defaults sample 2.5-5x faster (and 40 against 299 s per
attempt in the earlier pair), so the alias keeps the defaults. The coffee table keeps the tuned list (4 / 4 with 0 rejections; no
default-knob run on the new partner). The executor's end-of-push read of `Touching` returned None (unreadable) on the ottoman; the
drive ran to its line's end and the task's evaluation read touching, so it did not matter here.
| swivel chair 155 | `nxl_chair_ann` / `nxl_chair_tuned` | 0 (42 / 67 attempts) | - |

The ottoman needed three more fixes after the re-binding:
1. **Sectional partner.** `sofa_hiphpn_0` is a chaise sectional: its AABB +y face is the chaise tip, the seat front 0.5 m inside. A
   diagnostic in the nudge's reject (direction, AABB gap before / after) showed 20 cm of nudges ending 18.6 cm inside the AABB. New opt-in
   `generation.touching_partner_surface` (`iors_large_nextto.partner_sections` / `advance_to_surface`): the partner's collision pieces are
   cut at 0.1 / 0.35 / 0.6 / 0.85 x the target height, the touching goal advances along the push from the AABB face to
   `touching_goal_gap_m` short of that material (at most `touching_advance_max_m` 1.0), and goal / start / sweep footprints are tested
   against those sections instead of the AABB. Set on 157.
2. **Kinematic-only partner.** A fixed-base object without joints is loaded kinematic-only (`usd_utils.compute_kinematic_only`) and is in
   no PhysX contact report, so OmniGibson's `Touching` never reads True against the sofa. The task's own touching
   (`sampling.predicates.evaluate_predicate`) falls back to a 3 mm gap to the partner's merged convex hull. The sampler's nudge now reads
   the task predicate through that wrapper. The executor's touching stop keeps the raw state on purpose: the merged hull covers a
   sectional's chaise pocket, so the wrapper would stop the drive short of the goal, while the raw False lets it run to the line's end.
   The hull leniency is an evaluation property worth the owner's attention: inside a sectional's hull any pose reads touching.
3. **Program check.** `program_check.check_blocks` required a `grasps` block for touching_large; it now mirrors the sidecar's floor-start
   relaxation (a base push needs no grasp). 10 of 17 default-knob attempts had failed there.

**Swivel chair (open).** With the annotations and either knob set, no instance in 109 attempts. At the stances that reach the seat, the
contact leaves the head camera's frame (45 % of the tuned run's stance rejections), then push IK.

**B6 role (fixed 23:55).** The first contacts.csv rows used role `push`. The grounder enforces the iddl's `(contacts target)` (axial,
world_vertical: at least one push row per target model) as soon as contacts.csv has any row, so the three large models' rows would have
blacklisted every axial / world_vertical target on the next grounding (5 `test_ground_isg` tests and 1 `test_support_candidates` test
failed here, passing on grasp-bands). The push spots now use role **`base_push`**; `ground_isg.ICL.contacts_available` counts only `push`
rows; the sampler reads `base_push` (and `push`). The tests pass again; `test_eval_instances::test_step_budget` and two `test_v2_core`
tests fail on grasp-bands too.

## 8. Owner items 2026-11-04 (second round)

Record: `isg_home/oracle_fix/DECISIONS.md` (2026-11-04 entries). Opt-in diagnostics: `ISPATIALGYM_STANCE_BLAME=<n>` now covers Phase 1a,
both stance-C searches (`prompt0.blame_stances`: robot links by class arm / torso / base / head + the body hit), the inside stance C's
out-of-frame candidates, the B -> C carry path (`navigation.path_blame`: world contact vs self-collision) and an empty F_goal
(`iors_large.blame_release`: the place poses with collisions off).

| item | change | result |
|---|---|---|
| 1 swivel chair 155 | annotator `--off-centre` pairs, spacings to 0.50 m, armrest-level flatness: 20 `base_push` rows (seat-front, armrest 0.50 m apart, backrest, side pairs); knob `push_face_spots_max` (annotated models try every spot); trunks / stand-offs from an offline reach + head-frame table (`isg_home/oracle_fix/chair_table.{py,txt}`) | 0 instances in 72 attempts; out-of-frame 45 % -> 5 % of stance rejections; now reach 39 %, stance floor 26 %, goal / init floor 19 %; push IK mostly "pre-contact exact IK failed" (34 of 41 sampled). Open |
| 2 desk 151 | opt-in `generation.free_start` (`iors_large.free_start_stance`): stance A on free floor seeing the target and the goal region; B from the edge as before | stance A found 0.99 m from the target after 16 draws; setup passes; every attempt F_goal = 0: the place poses are 0.67-0.85 m out of reach with collisions off from the C stance (0.79 m from the deep goal cell). Open |
| 3 bed variants 152 / 159 / 160 | stance-C blame -> the eames chair beside the bookcase's open side; removed on 152 / 159 / 160 | stance C 0 -> 45 survivors; then the desk's F_goal = 0. 160: the 0.545 m shelf has 0 goal cells after the 0.14 m erosion |
| 4 pillow 158 | path blame; opt-in `generation.carry_lift_by_extent_m` | the B -> C collision is the robot's base / wheels (world contact), not the pillow; the lift margin changes nothing (lifts up to 0.47 m). The 0.35 skip is a body check, 0.55 is out of reach (146 mm). Open |
| 5 pillow 153 / umbrella 154 | v2 pipeline into the copy (`scripts/oracle_fix/job_glib_copy.sh`, GBH = our record), `ISG_GLIB_CONE_DEG`; job_val `PERSIST_BANDS` / env `OF_BANDS` (`$OFH/bands_floor`); a cached band of another grasp set is a miss | pillow: 0 v2 candidates (45 deg cone: 1, invalid in physics) - no floor-feasible grasp; floor band 0 cells; F_init 0 in 120 attempts. umbrella: F_init 25-31, blocked at the push-under C ("no push chain at the blade-window yaws") |

Notes: `reach_maps` has no floor mode and the floor start never reads maps; the floor bands are the in-scene legacy bands (now kept in
`$OFH/bands_floor`). The mesh offender search (`world_collision_offenders`) named an armchair, then ceiling downlights, for the same
colliding pose - do not trust it for base collisions.

### 8.1 Third round (2026-11-04 09:30-, owner: one successful demo per eval pick; creative fixes allowed)

Record: cluster `isg_home/oracle_fix/DECISIONS.md` entries "THIRD ROUND" a-ah; runs `isg_home/oracle_fix/runs/<name>`; job lists `r3_jobs.txt`;
`rr.py` (rejection-reason histogram per run). Read-only diagnosis agents read the logs of each blocked pick (their findings are in the entries).

Picks with a successful demo: 83 counted + axial#02, disembedding#06, ontop_small#02, nextto_large#03 (coffee table), touching_large#01 (ottoman) from
rounds 1-2, and this round:

| pick | how | demo |
|---|---|---|
| touching#04 spray bottle | per-alias `evaluation.tolerances.pose.orientation` 0.45 (the trigger bottle ends leaning 20 deg on the partner; the marker precedent) | tch04_tol_redemo 2 / 2, w3_tch04_ckt_s1 0001 |
| object_scaling-beefsteak_tomato (open complaint) | ogpans (already the customs representative) | os_beef_ogpans_s0 / _s1 2 / 2 |
| nextto_large#01 swivel chair | partner re-bound to office_cubicles_right desk_pfsjbc_0 (lobby, 4.2 m free side); `push_end_ignore_target` (the push end / settle stance checks against the chair at its INIT pose); pitches 35-65 | chair8_lobby_s1 0000 |

Code added this round (all uncommitted; CPU tests: only the pre-existing failures, which fail identically on grasp-bands: 2 in test_v2_core, 4 in
test_v2_large_scale):
- `ik.IKChecker.reach_collisions_off` - the collisions-off reach of the blames, a DLS tracked along 24 line waypoints (the F_goal blame's 674-846 mm
  "short" was the residual of the START: one far waypoint returns no configuration); `ISPATIALGYM_STANCE_BLAME_VALIDATE` blames a feasible pair (it
  reports reached). `iors_large_nextto.blame_push_pre` (chair pre-contact), `InsideMode.blame_roll`, `navigation._world_bodies_freeing` (path blame:
  each nearby body removed alone).
- `iors_large_touching.LeanTouchingMode` (`generation.touching_place: "lean"`, umbrella 154): the lean pose from horizontal ray profiles of the partner
  per lane (`lean_offset`, `lean_quat`, `axis_and_top`), the release with the partner and the floor out (`IORSLarge.place_ignores`), program = pick,
  carry, drop-release at the lean pose (`grasps[0].place_kind` "lean"; sidecar drops `contacts.push`). Unit tests in test_iors_large_touching.py.
- `iors_large_inside`: rolled (standing) releases `generation.release_roll_deg` / `release_level` (hardback 151 / 152 / 159 / 160; ROLL_KEY bookkeeping,
  `roll_fits` footprint rule, optional `inside_wall_hug_m`), `pre_place_of` / `pre_place_tag` hooks with `inside_pre_outward` (also in the F_goal screen,
  `ik.feasible_sets(pre_of=)`).
- `articulation_hybrid.check_program_follow`: opt-in `follow_first_pose_ignore_object` (poses[0] checked as step 5a).
- `tabletop.shelf_lean_pose`: `generation.shelf_spine_behind_edge_m`, `shelf_bay_lateral_m`; `iors_small.shelf_pre_place` (`shelf_pre_place_up_m`).
- `IORSLarge`: opt-in `grasp_frame_rule: "fingertip"` (B grasp judged by QA's fingertip criterion), `floor_b_trunks_exclude`,
  `push_end_ignore_target` in `check_stances_with_objects`. job_val.sh persists tabletop compute_bands bands too.

Still open at 12:00 (runs going; per-pick blocker in the DECISIONS entries):
- close / open_large#03 cabinet: instances now sample (33; eval85 never), demos fail robot_wobble 1.006-1.073 deg (gate 1.00) / failed_grasp at the
  sequential re-grasps / push-close no_motion -> arm-only 16-22 deg pulls (art4_*).
- under_large#02 briefcase (desk_mdhelw_0 pitched): chains solve at d 0.90-0.95; B grasp frame (fingertip rule), push stance settle check (brief7 / brief8).
- inside_large#02 jar: the attached jar's spheres hit the target shelf plate -> drop_height_m 0.03, left arm (jar5_*).
- ontop_large#02 mouse: first instance (mouse2_s5), demo robot_wobble 1.03 deg at the floor grasp -> no ground trunks at B (mouse3_*).
- touching#03 hardbacks, inside_large#01 hardback, touching_large#02 umbrella, ontop_large#03 pillow (158; 153 given up): see the entries.

Update 14:45 (DECISIONS entries 1104j-1104q): ontop_large#02 mouse - mouse2_s4 0000 SUCCESS (92 / 100). Further changes: `core.stance_check_ignores` hook
(the settle stance AND transition re-checks skip the target after the pick - the base push end / the push-under crouch stood where the target's init is);
`generation.path_margin_m` (empty-handed legs); `articulation_large`: `close_goal_margin_rad`, `other_joints_init` (+ Open restricted to the moved joint);
`IORSLarge.floor_b_trunks_exclude`, `drop_pre_m`; `iors_small`: the shelf's vertical pre-place in every caller, the finger pre-shape on the approach
(`approach_finger_support_touch`), `shelf_working_arm_default`; execute.py logs the handle grasp's AG link. Cabinet #03: 60 instances, 0 successes - the
door never moves (joint_rad <= 0.002) whatever was tried (counterbalance + friction 0.05, the neighbour door held open, arm-only); a simulator probe is
investigating. Umbrella: the lean coupling + goal pass; the leaning umbrella topples in the goal settle (> 40 deg) on the hall tree's edges -> shallower leans.

### 8.2 Fourth round (2026-11-04 18:30-, owner items: cabinet geometry, hardback overhang / spine, touching#03 layout, umbrella on both partners)

Record: DECISIONS entries 18:30-; runs `isg_home/oracle_fix/runs/<name>`. Status at the start: 96 / 100 picks with a successful demo (the cabinet pair
counted through the run_isg floor filter).

**Cabinet (articulation_open / close_large#03, bottom_cabinet_rvunhj_0 door j_link_2) - geometry instead of the collision filter.** Simulator measurement
(`probe_cab2/out_cabgeo_18312380.json`): all six rvunhj cabinets at root z 0.79694 sit IN the floor (floors_ovpbsn_0 top 7.143 mm): base_link lowest point
-0.87 mm (8.02 mm deep), lower doors link_0 / link_2 +1.51 mm (5.64 mm deep), upper doors clear - the lower doors could not swing. Lift 8.1 mm (root z
0.80504: body 0.1 mm, doors 2.5 mm above the floor). New hook `scene_variants.stable_scene_override` / env `ISPATIALGYM_STABLE_SCENE_DIR` (a directory of
`<scene>_stable.json` copies; read by `sim_utils.stable_scene_path`, `sample_isg.stable_scene_json`, `reach_bands._stable_scene`; unset = unchanged); the
templates (per `--out`) are merged from it and run_isg loads the template, so the demo gets the raised cabinets. Campaign copy
`isg_home/oracle_fix/scenes/office_cubicles_right_stable.json` (only the six root z changed; md5 original e435b6a5... untouched, copy 7c2f39d3...).
`generation.floor_collision_filter` false on the three bottom_cabinet aliases (the run_isg loader code stays). Result (cabgeo_*): open 4 / 4 and close 5 / 5
demos successful at eval seeds 10 / 20 and 11 / 13 / 20, no floor filter anywhere; close_push (open 0.30-0.55) still samples no instance.

**touching#03 (hardbacks, bookcase_inside).** The push-class failures are now named per lateral (`iors_small.push_stance`): at opryhp_0 every lateral gives
10 / 15 candidates failing the exact IK at the push pre-pose and the rest hand-vs-book margins; a 2 cm pre-pose or a 4 cm gap changes nothing.
Compartments (`probe_shelf/out_shelf_18312748.json`): opryhp_0 board 0.372 one bay 0.72 m x 0.36 m, headroom 0.33; kmuacd_0 board 0.835 three cubes 0.35 m,
depth 0.27-0.30; otwukr_1 boards 0.389 / 0.720 two bays 0.30 m, depth 0.22-0.24, headroom 0.28-0.29. New opt-in `generation.shelf_tip_release` (shelf
layout): the carried book is released at the drafted AABB-contact lean (where the goal check's settle starts - that settle already tips the book onto the
partner) from the second base pose, no push (`grasps[0].place_kind` "shelf_tip"; sidecar / program check drop `contacts.push`; program = travel, grasp,
lift, carried travel, place, retreat); at most `shelf_tip_max_nudges` (3) 1 mm nudges in the goal check. The lying-flat variant (the `__bo` overhang rows)
would fit opryhp_0's 0.72 m bay; not built while the tip release runs.

**Umbrella (touching_large#02).** STANCE_BLAME: every empty F_goal fails at the pre-place already with collisions off (median 310-430 mm short on both
partners and both poses): lying INTO the face puts the mid-shaft grasp (all 12 rows within 6 cm of the model origin) 0.25-0.55 m ahead of C's base at
floor height. Upright leans on hall_tree_upkvgr_0 get further: one attempt passed sampling + goal check and died in the program check's block list
("missing contacts.push"), one in the goal check's place re-check at the settled (slightly rolled) pose. Changes: `program_check.check_blocks` knows the
lean (no push); `LeanTouchingMode.after_goal_check` falls back to the release pose; opt-in `lean_free_spin` (the spin about the leaning axis free of the
carried init yaw); opt-in `lean_azimuths_deg` (90 at a 90 deg lean = lying ALONG the face, the grasp in front of the robot as in B's floor pick).

**Hardback (inside_large#01).** Spine verified on the decrypted visual USD (-y long edge); `generation.grasp_rows` [grasp_0..2] (the spine pinches)
on the four aliases. No hardback instance has reached a demo yet in this campaign (no results.jsonl), so the in-demo pinch check is pending.

Update 20:10 (DECISIONS entries 18:40-20:10):
- **Umbrella done**: umbD_upk_flat_s0 0000 (lying 86 deg into hall_tree_upkvgr_0's face) and umbD_upk_up_s0 0000 (upright 22 deg lean) SUCCESS. The
  main customs now carry the lying upkvgr configuration (+ `lean_free_spin`); hall_tree_daxzaw_0 is in the copies (no instance yet). Also new: opt-in
  `lean_azimuths_deg` (lying along a face; no lane on upkvgr). 97 / 100 picks with a successful demo.
- **Cabinet**: open 6 / 6, close 7 / 7 demos on the raised scene copy, no floor filter.
- **touching#03**: the tip release needed two more pieces - the place from the second base pose (the push stance search keeps the first lateral whose place
  passes) and `grasps[i].lift_away` (new: the oracle's `Lift(away=)` and every sampler lift end via `program_check.lift_end_pose`; the shelf tip lifts the
  book 3 cm toward the partner, `generation.shelf_lift_away_m` - straight up it stayed against the side board, the carried drive's first pose collided);
  `iors_small.require_in_frame` takes the opt-in `grasp_frame_rule: "fingertip"`. First instance t03tip3_kmuacd_s1 0000; its demo failed plan_failed at the
  grasp transit (cuRobo trajopt to the pre-grasp witness inside the 0.35 m cube, train-fast PLAN_LADDER); reruns with the ladder off / 8 seeds.
- **Hardback**: hb11 33 attempts / 0 instances (F_init 9 x F_goal 0-3 of the rolled releases never at the carried yaw) -> opt-in `inside_free_yaw`
  (rolled releases screened over the whole yaw grid, the drawn row released at its screened yaw): hb12_*.
- **Pillow 158**: the carried B -> C leg's collision at (-0.38, 0.92) is the robot base against meshes named after armchair_ybcnmu_0 (whose root is 5.7 m
  away); raised carry poses / swapped beds do not help (the swap: no start stance at bed_kddswt_0); a simulator probe is locating the geometry.

Update 00:10 (2026-11-05; DECISIONS entries 20:10-00:10): **99 / 100 picks with a successful demo.**
- **Pillow 158 done**: pillow16_keepoff_back_s4 0000 SUCCESS. The probe (`probe_pillow/out_pillow_18313824.json`) showed the carried leg's
  collision was a wheel sphere 8.7 mm from bed_kddswt_1 at the 1 cm path margin (the blame had named an armchair: `world_collision_offenders` never checked
  the pose at the margin first - fixed, and `path_blame` now prints object:mesh). New opt-ins: `generation.path_keep_off` (base centres kept `margin_m` off
  an object's footprint on the local AND the eroded global map; `NAV.plan_path(global_blocked=)`) and `carry_departure_min_back_m`. Main customs carry it.
- **touching#03 done**: t03tip5_kmuacd_s5 0000 SUCCESS (bookcase_kmuacd_0 centre cube, tip release). The last blockers were the oracle's transit to the
  shelf pre-grasp (8 / 8 demos plan_failed: the arm swept into the neighbouring bookcase_opryhp_1 / the support - named by a new diagnostic in
  `motions._straight_plan`) -> opt-in `generation.shelf_transit_check` (the straight joint ramp stance -> pre-grasp witness, or via "ready", must be
  penetration-free), plus the fingertip frame rule and `grasps[i].lift_away`; an opt-in pre-shape before the transit (`grasps[i].preshape_transit` /
  env `ISPATIALGYM_PRESHAPE_TRANSIT`) exists but was not the fix. Main customs carry customs_t03_tip4.json's alias.
- **Umbrella**: 3 / 3 demos on hall_tree_upkvgr_0 (lying / upright); hall_tree_daxzaw_0: no instance in 6 runs (along / lying / upright).
- **Hardback (open)**: `inside_free_yaw` gives F 3-9 at C; with `f_draws_max` 12 one attempt reached the program check and died on 4 duplicate
  manipulation -> pick transitions (the overhang pick's grasp stance kept by failed draws) -> fixed (`_drop_leg_records(pick)` before recording); hb14_*.

Update 01:20 (2026-11-05; DECISIONS entries 01:08-01:21), hardback still open:
- hb14_desk_s3 replayed hb13_desk_s3's attempt 0 (seeded, the same sampler lines) and passed sampling (3482 s), settle and the goal check; the program
  check rejected action 3 of 12, the retreat after the overhang push (`ik_joint_limit: left_arm_joint7 within 3 mrad`). The check had judged the recorded
  leg (10 cm along -a, toward the torso) as a fresh IK; the oracle runs the contact escape there instead (radially off the book over its far rim, then up,
  shortened when unreachable, from the push's end configuration). `program_check` now replays that escape for the retreat after the overhang push
  (`replay_contact_escape`, factored out of the arc-push replay; the overhang push records its tracked end) - only in this situation. Opt-in diagnostic
  `ISG_PC_DIAG_ALL=1`: the program check logs every rejected action instead of stopping at the first.
- Runs on the fixed check: hb16_* (desk s1 / s3 / s6 / s7-s12, shelf052 s0 / s1, bed s0, side_any s0) and hb17_desk_s3_diag (seed 3 with the diagnostic).

Update 02:40 (2026-11-05; DECISIONS entry 02:40), hardback still open:
- The retreat fix holds (hb16_desk_s3 / hb17_desk_s3_diag: the escape replayed, 0.10 m, probe ok). The next rejection, at 3410 s: the grasp from the
  pick stance (B backed off 0.10 m) had its pre-grasp fingertip below the head camera's frame (v 1058.7 vs the 1004 px limit, 0.425 m from the camera).
  With `ISG_PC_DIAG_ALL` the program check named it as the only rejected action of 12. The sampler had judged the overhang grasp by the eef box only;
  `_overhang_grasp` now applies the program check's fingertip gate (`fingertips_in_frame`, 5 % + 2 cm) to the pre-grasp and the approach's end from the
  grasp stance, so an out-of-frame stance falls through to the next back-off (0.20 m). Customs copy `customs_hb_bo3.json` adds a 0.30 m back-off.
- Full suite: the worktree now fails exactly the clean-HEAD set (13 failed / 3 errors, 2439 passed); two campaign regressions in tests were fixed
  (`self.grasps` guard in the stale-band check; the handle-row test updated to the owner's 3 mm pinch floor of 2026-11-02).
- Runs: hb18_* (the gate; desk s0-s4 / s6-s9 / s11-s14, desk s1 / s3 / s6 with the 0.30 m back-off, shelf052 s0 / s1, bed s0 / s1, side_any s0);
  hb16_* still running; hb14 / hb15 / hb17 cancelled.

Update 03:53 (2026-11-05; DECISIONS entry 03:53), hardback still open:
- hb18_desk_s13 attempt 1 reached the program check (its grasp from the 0.20 m back-off, the new gate's fall-through) and was rejected at the push: the
  pre-contact re-witnessed on the stance posture's IK branch (the arc replays from it) polished to 2.1 mm from the recorded pose, over the 2 mm limit.
  The hardback customs copies now allow 5 mm on pre-poses (`witness_residual_pre_m` 0.005; the attachment aliases allow 15 mm), read by the program
  check only, so the seeded rerun hb19_desk_s13 replays the same instance. hb19_* (17 runs, all with `ISG_PC_DIAG_ALL=1`) add desk seeds 15-26, bed s2,
  shelf052 s2, side_any s1. Seeds desk 6 / 8 are stuck at setup (the init -> B path collides every attempt) and were cancelled.

Update 04:05 (2026-11-05; DECISIONS entry 04:05), hardback still open:
- hb16_desk_s1 attempt 3 passed the program check through the grasp, lift and carried drive and was rejected at the place's carry leg: it stalled at
  dense pose 9 of 45 and the finish would land 1.12 rad away (the oracle allows 0.6). The sampler's own carry check at C finished such a stall with an
  unbounded IK. Opt-in `generation.carry_track_oracle` (iors_large `_carry_track_kw`) now gives that check the program check's acceptance (finish within
  0.6 rad, the oracle's tolerances); the hardback customs copies `customs_hb_*_wr5_ct.json` turn it on. hb20_* (20 runs) carry all three fixes; hb16 was
  cancelled.

Update 04:40 (2026-11-05; DECISIONS entry 04:40), hardback still open:
- hb18_desk_s1 failed the place's carry the same way (stall at 10 of 45, finish 1.21 rad away): both place checks reached so far stall there. The sampler
  chose the first carry option with free B -> C legs and checked the carry at C only afterwards. With `carry_track_oracle` the carry at C is now checked
  per carry option (`_carry_track_at_c`), so an option whose arm branch cannot reach the rolled pre-place gives way to the next option or lift height.
  hb21_* (36 runs, desk seeds 1 / 3 / 13 / 27-54, bed s3 / s4, shelf052 s3 / s4, side_any s2 / s3) run it; hb18 / hb20 were cancelled.
- 05:22: hb19_desk_s13 (the replay with the 5 mm allowance) passed the push; the diagnostic found 1 of 12 actions rejected, the place's carry (stall at 13 of
  40, finish 0.67 rad). All three place checks so far fail only there - the case hb21's per-option carry check is for. Four more seeds stuck at setup
  (s27 / s34 / s40 / s47) were replaced by s55-s58.

Update 05:46 (2026-11-05; DECISIONS entry 05:46), hardback still open:
- hb19_desk_s13_bo3 stalled on the same carry (4 / 4 place checks: the position 11-13 mm behind on the straight line with the 90 deg roll, orientation
  tracked). Opt-in `generation.carry_via_search`: a straight carry at C that does not track is retried through one carry waypoint (rotate first near
  the body, a raised / shifted midpoint, or translate first), each with the fingertip in C's image along the whole leg and the chain tracked with the
  oracle's acceptance; the one that passes is recorded as `grasps[k].carry`, which the oracle's place carries through and the program check checks (as
  `core.verify_carry_legs` does where the place has no base move). hb22_* (24 runs, `customs_hb_d12_wr5_ctv.json`) run it; hb21 / hb19 keep running.
- 06:12: hb19_desk_s24 (old carry tracking) passed the sampler's carry check, and the program check found its carry path colliding with the bookcase -
  five of five place checks fail only on the carry at C. hb21 / hb22 track it with the program check's settings. Overhang picks succeed in 2.4 % of the
  tries (56 / 2369 in hb21 + hb22), which is why an attempt takes about an hour.
- 06:19: about half of the overhang-pick failures are recomputations of the same inputs in a later attempt. Opt-in `generation.overhang_fail_memo`
  remembers a failed pick for the run (`_overhang_memo_key`: B's pose and trunk, the arms' stances, the row, the yaw, the focus cell). hb23_* (30 runs,
  `customs_hb_d12_wr5_ctvm.json`) carry every fix so far; hb19 was cancelled.
- 07:25: first hardback instance through the program check - hb21_desk_s55 (`customs_hb_d12_wr5_ct.json`, eval seed 55), attempt 2 (1365 s of
  sampling): B (0.00, -0.92), left arm, grasp_1, a 0.160 m push to the edge (0.085 m overhang), the carry at C tracked with the program check's
  settings (0.31 rad finish swing). The program check passed all 12 actions; the demo started 07:23.
- 07:40: hb21_desk_s55's demo failed at the push (plan_failed: the arm_along stalled 11.5 mm behind at waypoint 21/28). The program check had moved
  the straight push's pre-contact witness to the stance posture's IK branch (3.13 rad from the recorded one) without tracking the push from there -
  the arc replay returns "not an arc" for a straight push. Fixed: `program_check.track_line_push` (the oracle's `along_reaches` logic) now tracks a
  re-witnessed straight push (`ISG_PUSH_REWITNESS_LINE=off` restores the old move), and the opt-in `generation.overhang_push_branch_check` refuses an
  overhang push chain in the sampler unless the straight push tracks from where the oracle leaves. hb24_* (31 runs, `customs_hb_d12_wr5_ctvmp.json`)
  replace the 30 unstarted hb23 runs and rerun seed 55.
- 07:46: full suite = the HEAD failure set (13 failed / 3 errors / 2445 passed). Diagnostic, not counted: a copy of s55's instance with the push
  pre-contact witness restored to the recorded branch (`chain_arm_q[0]`) runs demo-only (`runs/hbD1_s55_rewit`) to see the grasp / carry / place
  in the demo while hb24 samples.
- 07:56: hb22_desk_s13 wrote the second instance (carry through the new carry waypoint) and its demo failed at the push the same way. The
  diagnostic pushed s55's hardback from the recorded witness without trouble (trajopt reached it; push, retreat, transition OK). Opt-in
  `generation.push_flipped_keep_recorded`: a flipped straight push that fails from the stance branch and tracks from the recorded witness keeps the
  recorded witness (program check and sampler). Customs `..._ctpk` / `..._ctvpk` / `..._ctvmpk.json`; hb24 (pending) repointed to ctvmpk; hb25 reruns
  seeds 55 and 13 deterministically with the fix; diagnostic hbD2 replays s13's instance from its recorded witness.
- 08:12: hb22_desk_s73 / s72 wrote instances (s73's demo failed at the push, 3 of 3; deterministic reruns hb25_desk_s73 / s72 queued). Diagnostics
  from the recorded push witness get past the push into the grasp: s55's grasp took three attempts (a wrist-camera touch on the desk, a failed plan,
  then grasp + lift OK); s13's transit into the pre-grasp passed through the book. After the 5 cm escape the base's back-off to the grasp stance
  leaves the hand above the book's far end. Opt-in `generation.overhang_retreat_up_m` -> `contacts.push_pick.retreat_up_m` -> `Retreat.escape_up_m`:
  the escape off the desk is that high (oracle and program check). Diagnostics hbD3 / hbD4 test 0.15 m on the two instances; hbD1b reruns s55's to
  see the carry and the place.
- 08:33: with the escape at 0.15 m the s55 diagnostic grasped at the first attempt with the desk present; both s55 diagnostics then failed at the
  departure from the desk - the idle right arm is sent to the placement stance's "ready" posture while the torso is still bent over the desk (the
  program check sweeps the departure with the arm already there, not the move). Fix by the existing `carry_departure_min_back_m` (0.3): drive back
  first. The s13 diagnostic (idle left arm) passed the departure, the carried drive and the arrival, and is placing. hb26 reruns seeds 55 / 13 / 73 /
  72 / 60 with keep-recorded + escape 0.15 + drive-back 0.3 (`..._ctpkub` / `..._ctvpkub.json`); hb24's pending runs use `..._ctvmpkub.json`.
- 08:45: the s13 diagnostic failed at the place: the carry at C stalled at 10/55. The real push left the hardback 2.7 cm aside and 5.5 deg turned
  from the predicted pose, so the grasp and lift ended up to 0.16 rad from the configuration the carry was checked from (base and trunk matched).
  Opt-in `generation.carry_via_online` (-> `grasps[k].carry_online`): the oracle dry-runs the recorded carry from where the arm is and, when it does
  not track, tries the straight leg and the 12 carry waypoints from the live hand pose (fingertips in frame, collision-checked dry run), taking the
  first that passes. Customs `..._ctpkubo` / `..._ctvpkubo` / `..._ctvmpkubo.json`. hb27 (seeds 55 / 13 / 73 / 72 / 60) replaces hb25 / hb26; the
  diagnostic hbD5 replays s13's instance with the online carry.
- 09:15: hbD5 ran the online carry search: the dry run reproduced the stall exactly, but none of the 13 candidates passed (their reasons were not
  logged). The search now dry-runs every candidate with and without the collision check, judges the frame with the program check's 2 cm slack,
  picks by tier (in frame + collision-free, in frame + tracked, tracked) and logs every candidate's verdict; hbD6 replays the instance.
- 09:20: the online carry first tries returning the arm to the program check's lift witness (the configuration the carry was verified from) when
  the recorded carry tracks from there - a small planned joint move with the book in hand - then carries as checked; the live-hand candidates are
  the fallback. Diagnostic hbD7 replays s13's instance with it; hb27 (seeds 55 / 13 / 73 / 72 / 60) samples with the full configuration.
- 09:38: hbD6's candidate verdicts: from the drifted hand the rotating carries do not track and the ones that track collide (the straight leg,
  executed anyway, jammed the arm). The online search now takes only in-frame, collision-free candidates; the re-pose to the lift witness (hbD7)
  is the remedy for the drift.
- 09:48: the re-pose did not help: the recorded carry does not track even from the program check's lift witness - the program check accepted it
  with a stall finished by IK, which the oracle can only finish by a planned move (it fails next to the bookcase). The push drift is systematic
  (~2.5 cm away from the pushing arm, turned 0.05-0.1 rad, both arms). Opt-in `generation.overhang_push_drift` (0.025 m / 0.07 rad) puts that
  drift into the predicted post-push pose, so the whole plan follows where the book ends up. Customs `..._ctvmpkubod.json` (+ ctp / ctvp variants).
  hb28 (31 runs: the seeds of the cancelled hb21 / hb22) samples with it; hb24 (31) runs on `..._ctvmpkubo`, hb27 (5) on the `..._kubo` reruns.
- 09:58: opt-in `generation.carry_finish_max_remaining_m` (0.05): the carry at C may end in a stall finished by IK only within 5 cm of the
  pre-place (sampler and program check) - the oracle finishes a stall with a planned move, which fails next to the bookcase over 0.38 m. Customs
  `customs_hb_d12_wr5_ctvmpkubodr.json` carries every hardback fix. All 64 runs were cancelled and hb29 (63 distinct seeds) submitted on it from
  the sc login node (svl11 is a compute node: its ssh sessions die with the job they are adopted into).
- 10:55: the first instance with every fix - hb29_desk_s44 (attempt 0, 2992 s; left arm, push witness on its recorded branch, drift in the plan,
  straight carry within the finish bound, escape 0.15, drive-back to 0.85 m). Its demo started 10:51.
- 11:03: hb29_desk_s44's demo failed in the grasp (the arm touched the torso closing a 54 mm / 0.32 rad residual): its book turned 0.389 rad to
  lie flush with the pushing pad, not the 0.07 assumed. The drift model now predicts the flush turn (`overhang_push_drift.snap_max_rad` 0.70: a
  book within 40 deg of flush turns flush; near 45 deg the small turn). `customs_hb_d12_wr5_ctvmpkubodrs.json`; hb30 (62 seeds) replaces hb29.
- 13:38: **inside_large#01 solved - 100 / 100 picks with a successful demo.** hb30_desk_s41 (instance at attempt 23; left arm, the push witness
  flipped and the recorded one kept, the book 45 deg off flush so the small predicted turn) ran 12 / 12 actions: push, escape, backed-off grasp
  (reached via the untucked posture after the direct plan self-collided), lift, departure, carry, place inside the bookcase, partial retreat.
  Success at step 2142 settled, 2.8 mm / 0.022 rad, 0 collisions, QA clean (no self-touch, gripper never out of view).
- 13:44: the main `isg_customs.json` takes `customs_hb_d12_wr5_ctvmpkubodrs.json`'s 4 hardback aliases (md5 508093ae -> d412befa; nothing else
  changed; the later pillow / umbrella / t03 folds kept). The 49 remaining hb30 desk jobs were cancelled; the 12 bed / side_any / shelf052 runs
  continue as a check of the folded customs on the other aliases.

## 9. Speed round (2026-11-05 16:35-; owner: "from the failure logs try to speed up the data generation time for the slow aliases")

Record: DECISIONS entries 16:35-; analysis scripts `isg_home/oracle_fix/diag/{gen_time,gen_batches,rej_time,warmup}.py`. The slow aliases (final
configurations): the hardback inside_large#01 (desk: 102.8 sampling h for 1 instance; bed aliases 0 in ~50 h), touching#03 (7.5 h per instance at
kmuacd, 0 in 10.3 h at opryhp), mouse (8.0 h / 4 instances, 1 / 4 demos), jar (2.2 h / 1), briefcase (5.5 h / 13, 2 / 12 demos), umbrella (5.5 h / 8),
swivel chair (1 / 8 demos). Where the time goes:
- **cold caches**: every oracle-fix job starts with an empty job-local sampling cache; the first hardback attempt takes ~2800 s, the steady ones
  ~65 s (the first 10 attempts are 45 % of hb30's attempt time); mouse / jar / umbrella attempt 0 1400-2600 s. Production (trainfast) shares one.
- **attach**: every attach refits the object's collision spheres (~2.2 s; 58 of 147 s of a steady hardback attempt; 2.9 of 5.5 h for the briefcase).
- **late rejections**: touching#03 recomputes the same fixed-start grasp IK / transit verdicts per stance (97 % of its time); the briefcase fails
  the program check's fingertip gate at the pre-grasp after the whole draw (51 % of its rejected time).
Changes (opt-in): `ISPATIALGYM_SAMPLER_PROFILE` (cProfile dump), `generation.b_fingertip_frame_check` (iors_large: B's pre-grasp / grasp by the
program check's fingertip gate), `generation.fixed_start_memo` + `generation.shelf_transit_per_grasp` (iors_small, with `grasp_at_pose(accept=)`),
job_val.sh `SHARED_SAMPLING_CACHE` / `SPHERE_FIT=cache`. Customs copy `customs_spd1.json`. Profiles of four aliases running (spd0_*).
- 17:35: the first profile was empty (the Kit launch replaces the thread's profile hook): the hook now starts after the sampler's setup and
  dumps after every attempt (spd0b_* resubmitted). Failure memos can persist in the shared cache (opt-in `overhang_fail_memo_persist`,
  `fixed_start_memo_persist`; keyed by scope + the alias's generation block). A/B spd1 (38 jobs): fast arm (spd1 keys + `SPHERE_FIT=cache` +
  a shared cache per alias) vs the reference settings on the same seeds, for hardback desk (base = hb30), touching#03, briefcase, mouse, jar,
  umbrella.
- 17:50 (spd1 interim, same seeds): briefcase sampling 3.5x faster (5947 -> 1690 s attempt time for 12 instances; attach 513-1094 -> 12 s per
  run) - its demo yield stays the oracle's problem (per-seed navigate wobble / wall contact); mouse 1.6x and jar 1.4x to the same instance;
  hardback desk 1.54x attempts per hour (no instance either arm). touching#03's fixed-start memo is harmful (its successes come from stances
  that failed before - the IK branch is random): dropped. touching#03's time is 83 % exact IK in grasp_at_pose: new opt-in
  `fixed_start_prescreen` (one batched screen of every row / twin first). Waves spd2 / spd3 running (touching#03 base / per-grasp /
  per-grasp + prescreen; hardback with the persisted pick memo).
- 19:07 PAUSE. Snapshot: the hardback with the persisted pick memo + warm cache makes 5.0x the attempts of hb30 on the same seeds (227 vs 45 in
  ~4560 s each), still no instance; touching#03 base found 1 more (s11), per-grasp transit / the prescreen none yet (the batched screen costs
  more than the exact IK it saves); umbrella 4 / 4 runs with 2 instances and all demos ok, fast arm 1.1-1.5x; mouse / jar runs without an
  instance spend 3000-7300 s in batched screens, and the warm cache did not hit on a new mouse seed.

### Resume (speed round)

State on 2026-11-05 19:07: nothing committed; all code lambda = cluster (md5 in DECISIONS 16:35-19:07). Every speed change is opt-in:
`ISPATIALGYM_SAMPLER_PROFILE` (cProfile per attempt), `generation.b_fingertip_frame_check`, `fixed_start_memo` (harmful, do not use),
`fixed_start_memo_persist`, `shelf_transit_per_grasp`, `fixed_start_prescreen`, `overhang_fail_memo_persist`, job_val.sh `SPHERE_FIT=cache` /
`SHARED_SAMPLING_CACHE=<dir per alias>`. Customs copies `$OFH/customs_spd{1,2,3}.json` (jobs/mk_spd*.py); shared caches `$OFH/cache_spd1/<alias>`.
Batches (runs/): spd0b_* profiles; spd1b/f_* (base / fast, same seeds); spd2b/f_t03 + spd3f_t03 (base / per-grasp / per-grasp + prescreen);
spd2f_hb_desk (persisted memo; base = hb30_desk same seeds); spd4b/f_* (warm cache, new seeds). Submit scripts jobs/sub_spd*.sh (run on sc).
1. Tabulate: on scdt, `cd $OFH/diag; python ab.py "spd4?_*" "spd?[bf]_t03_s*" "spd2f_hb_desk_s*" ...`, `python paired.py "hb30_desk_s*"
   "spd2f_hb_desk_s*"` (base attempts in the fast run's attempt time), `python rej_time.py <glob>`, `python prof_top.py runs/<run>/sample.prof`
   (env /vision/u/minyeong/miniconda3/envs/ispatialgym-vectorized-envs/bin/python). Check the spd4 kit lines for `feasible_sets.hit / .miss`:
   whether the shared cache hits on new seeds (mouse / jar / umbrella cold attempts are 80-90 % batched screens).
2. Decide per alias what goes into the main customs / job defaults (owner's call for defaults): SPHERE_FIT=cache is the clear win (attach 2.2 s
   -> ~0 per call; briefcase 3.5x, mouse 1.6x, jar 1.4x); b_fingertip_frame_check for the briefcase; overhang_fail_memo_persist + a shared
   cache for the hardback if spd2f keeps its 5x; touching#03 stays on base unless spd2f / spd3f show instances.
3. Open (not speed changes, from the failure logs): the hardback's acceptance (792 picks -> 1 instance: the carried path pick -> placement
   collides near (22.8, 13.6), no departure s_T); the briefcase's demo yield (2 / 12: crouched trunk (0.95, -0.3) sags 1.24-1.38 deg - the
   oracle sets B's trunk before driving in s10; wall contact on the first navigate in s15); mouse / swivel chair demo failures; touching#03's
   success is random per stance (+u cell 17).

Committed 2026-11-05 20:45 (owner): lambda branch oracle-fix 8516d022 = the 100 / 100 state (this document through section 8.2); the speed
round (section 9 and its code) stays uncommitted on top. The job-pipeline speed redo is on hold.
- 20:44 videos: the 17 picks closed in this campaign replayed (run_isg --mode replay, tiled view: left wrist | head | right wrist over
  the external camera) - 17 / 17 replays succeeded at their demos' step counts; one concatenated, labelled video (677 s) at
  `~/Desktop/projects/ispatialgym/tmp/outputs/oracle_fix_17_closed_picks_tiled.mp4`; jobs `$OFH/jobs/job_video.sh`, runs `vid17_*`.

## 10. One trunk (owner decisions 2026-11-05) and the demo-feedback round (2026-11-06)

Record: DECISIONS entries 2026-11-05 23:30 - 2026-11-06. Nothing pushed.

### 10.1 The trunk

- `oracle-fix` is the one trunk for alias validation, train-demo collection and evaluation. grasp-bands' uncommitted work = 4 commits on branch
  grasp-bands (8cf6203e..6f7519b4), oracle-fix's single 8516d022 split into 7 logical commits on top (backup branch
  `oracle-fix-single-8516d022`, identical tree), the speed round committed, train-fast-vecnext ac7b4bc2 merged (08514913, no conflicts;
  run_isg.py / tabletop.py keep both sides, TILT_SNAP default off). No commits on train-fast / train-fast-vecnext from here on.
- Tests: run with the trunk's PYTHONPATH (`R:R/Behavior-1K/OmniGibson:R/Behavior-1K/bddl3`; the conda env otherwise imports the main
  repo's Behavior-1K): 6 failed = the new_base_fails set (2464 passed).
- Runs use FROZEN snapshots only: `scripts/trainfast/snapshot_trunk.sh` (lambda) -> `$OFH/code/trunk_<commit>` (git archive, read-only,
  Behavior-1K = links into the frozen `$OFH/code/b1k_6a6194600_p<patch md5>` + its own datasets dir).

### 10.2 The pipeline (validation = collection = the fast flow)

- Submit on sc: `python3 <snap>/scripts/trainfast/submit_fast.py submit <key> <n> --flow fast --code <snap> --kcollect 2 [--gres l40s]`
  with, for validation, `--root $OFH --job-prefix isgof_ --base-index 300000 --tag <tag>`; `--extra-env NAME=VALUE` (job.env's
  ISPATIALGYM_* / ISG_* now win over env_fast.sh / fast_env.sh); `--customs template` to read a counted template's own copy (default: the
  snapshot's isg_customs.json for every alias - pre-rename variants resolve through customs.entry_key).
- Aliases outside train85's counted table: `scripts/trainfast/registry.tsv` (template copy, eval index, limits, band stores, extra env),
  made by `register_alias.py <key> <run dir> --root $OFH ...` from a template run: `submit_fast.py template <alias> <key> --code <snap>
  --seed-index K [--scene/--support/--support-row]` (= job_template.sh, the KIND=eval template step; PERSIST_BANDS supported).
- `submit_fast.py recollect <job dir>` (job_recollect.sh: collector-only, pinned; = KIND=demo), `job_video.sh <job> <snap> <out> [n]`
  (open-loop tiled review videos), `fleet_summary.py --root $OFH --tag <tag>` (successes per GPU-hour).
- fast_env.sh: `ISGF_GOAL_OCCLUSION` (on for mental_rotation / disembedding keys only), `ISG_VEC_ASSET_CACHE_DIR=$OFH/asset_cache`.
- Goal occlusion at the replay stage: `multienv/replay.py --goal-occlusion auto|on|off` (job_replay_multi.sh: auto) - the oracle's screen at
  the demo's settled last frame, the task's tracked objects only; an occluded demo is dropped (REPLAY_OCCLUSION lines). Validated on
  isg-1020 demos the train85 sampler measured: |predicted - replay| median 0.002-0.003, verdict agreement 45 / 45.
- Smoke (one job per family, N 2): 16 / 18 demos; scripts/oracle_fix/job_val.sh is retired for validation.

### 10.3 Demo feedback (owner 2026-11-06) - fixes and validation (fast flow, tag fix1)

| item | cause | fix | validation |
|---|---|---|---|
| steak standing | only key steak__lcebsq__t (x_down rest; lying widths > the 8.9 cm opening); goal centred on the pan's bbox incl. the handle | scale 0.6 + key steak__lcebsq__t060 (lying, 5 / 5 valid), `ontop_centre: container` (the pan's measured floor) | 1 / 1: lying flat (up axis 1.00, was 0.00), 7 mm from the floor centre (was 9.1 cm, against the handle wall) |
| coffee table push | closer stances die out of frame; d 0.50 | tried d 0.55-0.65: the extended arms ride 1.9 cm high over the 2 cm glass edge (0 / 6 contact) -> REVERTED to the d 0.50 set | 4 / 4 before (coffee_hsf2) |
| spray bottle wobble | the push at 83 % of its height (raised blade fallback) | `push_height_sweep` (default on, touching small): lowest reachable contact on the centre line | spray 1 / 1: contact 5 cm (18 % of H, was 83 %), tilt <= 5.8 deg (was 20.3), final 0.021 rad; touching_01 4 / 4; touching_02 (box of cereal) keeps today's push - `push_height_sweep: false` on its 12 aliases (the sweep: 1 instance in 245 attempts vs 1 / 37.7 in trainfast; goal check 1 / 45 vs 30 / 175) |
| umbrella compartment | floor-only lean modes (outer AABB edges) | `touching_place: surface` (SurfaceTouchingMode on the inside mode) | 3 / 3 on the alcove board |
| jar path | carried legs always 'recorded' (409 deg rotation) | face_travel for IORS-large legs within polyline + 10 deg | 1 / 1, the carried leg face_travel |
| touching#03 arm swing | flipped IK branch (joint1 -3.23) | shelf back-off 0.20 m + near-branch reach limit; tip release prefers the manipulation stance / tries each lateral's carried drive | 3 / 3 (review runs with an empty eval index: with the repeat rule every completed attempt repeated the eval instance - the alias's start diversity is narrow) |
| hardback in the bay | roll about the grasp radial, first-found yaw | edge-normal roll, spine out, shelf place (settled lean 16.5-21.6 deg predicted) | no instance yet: 46 attempts (8 jobs with STANCE_BLAME 2, ~1000 s per attempt vs 63-68 s) all rejected at the desk pick; the goal side is not the limit (C cells without a rolled release 40 % vs 50 % at base; past both exact IKs 0.11 vs 0.135 per attempt) - the alias's acceptance is (hb30: 1 instance in 6186 attempts); 12 jobs without blame running |
| blender pre-close twice | departure path lacked arms_q (navigate re-posed) | departure path records its swept arms; departure label | 4 / 4 |
| dishwasher grasp at the edge | bar offset drawn per instance; canonical row unscaled | handle_row record:bar, centre offset (open) / centre +-0.069 (close) | open 8 / 8 + 7 / 7, close 4 / 4 |
| dishwasher spring-back | policy release kick + damping-5 coast (oracle demos hold) | counterbalance_damping 15 | A/B 7 / 7 at 15 vs 8 / 8 at 5 |
| kielbasa ontop | support_rays 1 / 9 probe | centre of the smaller footprint + >= 3 hits within 2 cm | isg-1020 rays verdicts 600 / 600 unchanged |
| soup ladle / bottle of pop | chrome + black / black cola materials | hgvzdm / pvabxf representatives | 4 / 4 each |
| salt / pepper too close | init gap 3-7 cm past g_NT, target moves < 0.09 tolerance | nextto_gap_extra_m 0.12 (17 nextto_small aliases) | 8 / 10 demos, moves 0.136-0.158 m; ~2x attempts per instance |

Also found and fixed on the way: submit_fast.py took a counted row's customs PATH as the mode (a7080903); the collector loop broke on rc 0 although
a queue restart (rc 75) reaches the shell as 0, so an instance landing with sampling.done was never collected (72bff7f6; `recollect` reran the two
lost ones: 2 / 2); touching#03's tip release took the first lateral whose place passed and its carried drive swept the book into the cube wall
(a5f1e229); job_template.sh PERSIST_BANDS. Review videos (open loop, tiled): `$OFH/out/fix1_video/<key>`; concatenated on lambda
`~/Desktop/projects/ispatialgym/tmp/outputs/oracle_fix_feedback_fixes_tiled*.mp4`. Open: the hardback (12 validation jobs without the blame
diagnostics). The spray bottle's 0.45 rad tolerance was removed in round 2 (c15adeb1, section 11).

## 11. Round 2 (owner 2026-11-06 afternoon): requests, the low-yield survey, fixes (fast flow, tags survey1 / fix2*)

Owner: "spray bottle tolerance - go back to default; cereal box - does it push with the gripper open? if not try open at a lower
height; touching 3 - vary the start (other compartments' walls, lean angles); then investigate the low-yield / slow aliases and fix
them". Full write-up for the owner: `~/Desktop/projects/ispatialgym/tmp/outputs/oracle_fix_round2_report.md`; every number is in
`$OFH/DECISIONS.md` (2026-11-06 entries after 15:00 UTC).

| change | commit | default | aliases |
|---|---|---|---|
| spray tolerance back to the default profile | c15adeb1 | - | 9 spray-bottle touching aliases |
| Cartesian leg with the supports present before a grasp transit ignores them (jar brushing the table) | 2c20bdda | on (`ExecuteConfig.pre_grasp_cartesian_fallback`) | all |
| job_video.sh replays in the full template (`VIDEO_SCENE=full`) | 0d3e197c | on | review videos |
| SPLIT_CONE_RESTORE keeps a start iors_small drew itself (BUG since 11-01: every attempt after the first drove to the first start's stances) | c261ef5c / 3c198874 | on | every IORS-small split-cone alias |
| `generation.init_edges` (block_design#01: +v only) | d4603537 | opt-in | block_design_01 |
| `generation.c_skip_dead_b` (skip a C whose every drawable pick already failed at B) + under_large#01 ground trunks / c_tries 72 | 01ad8d8c / f5bd955b | opt-in | under_large_01, ontop_large_01, inside_large_02 |
| `generation.push_gripper_open` (open-gripper low push classes first) | dfae6b70 / c6bdef13 | opt-in | OFF since 60083d53 (validation: 9 / 16 demos, 8.1 attempts per instance vs closed 16 / 16, 5.9; tilt 0.4 vs 16.1 deg) |
| `generation.shelf_start_draw` (books' compartment / board / lean per attempt) | 97613441 | opt-in | touching#03 |
| `generation.push_arc_stall_ends_arc` + `push_arc_replay_lag_rad` (arc push at the reach end) | 9656c537 | opt-in | axial_01 (8 / 12); reverted on axial_05 in 60083d53 (0 / 12, all not_reached) |
| `witness_residual_pre_m` 0.005 | 7c34c732 | customs | nextto_large_02 |
| disembedding templates copied to `$OFH/templates` + registry rows | 19e0d1ca | - | disembedding_04 / 07 / 08 / 09 / 10 |
| `generation.place_inhand_reaim`, `target_orientation_equivalences` | f960f9cb | opt-in | disembedding_07, disembedding_10 |
| IORS-large trunk-first A -> B leg records its arms (`paths[].arms_q`) | bfb2f946 | on | under_large_02 (+ any alias taking that fallback) |
| `generation.init_setup_draws` (template setup retries) | b632d4b4 | opt-in | inside_large_03 |
| `prompt0.beta_max_deg` 40 (unmeasured) | 2db87c19 | customs | articulation close / open_large_03 |
| getattr guards (test fakes) | 91ab9413 | - | - |

Validation runs: `$OFH/out/{survey1,fix2,fix2_t02open,fix2_t03init,fix2_t03base}/<key>/g*`; status script `$OFH/jobs/r2_status.sh`
(per tag / key stats via `jobs/survey.py` + push class / tilt / shelf draws / hardback checks).
Validated (09:12 PST snapshot): touching#03 4 / 4 (baseline 0 written in 718 attempts); inside_large_03 4 / 4 (was 0 attempts); the
split-cone fix's controls ontop_small_01 1.75 / touching_01 2.1 / touching_02 5.9 attempts per instance (were 8-37); jar 230 s per
attempt (875), under_large_01 793 s (1569-2128), ontop_large_01 642 s (1033). Open: ontop_small_01 4 / 12 (the kielbasa tips 21-27 deg
at the place touchdown; trainfast 78 / 80), ontop_small_04 goal_tipped, inside_04, under_large_02 / 03, disembedding_07 / 10,
articulation close / open_large_03, the hardback (0 in 4143 attempts).

## 12. The hardback fix (owner 2026-11-06 evening: "fix the alias"; inside_large#01, hardback -> bookcase_jaysra_0)

**Where it failed.** In the spine-out jobs (fix1, 12 × N 1), 0 of 4,746 attempts wrote an instance. The spine-out release was
reachable in every attempt, and its in-hand place IK failed only 4 times over all tries. What failed was the desk → bookcase carry:
0 of 369 carry legs passed. hb30, the base without spine-out, passed 3 of 839, all from one rare pick (trunk −0.92, hand 0.41 m ahead).
- Local reproductions ran on lambda with the job's environment. `ISPATIALGYM_PATH_BLAME=1` turns on the path blame alone; with
  `STANCE_BLAME` attempts ran 16× slower.
- Every collision was at the end of the path, while turning into stance C (0.65 m stand-off, 0.39 m from the bed's footprint).
  With the torso at the pick's tilt, the carried book or arm hit the bookcase. With the torso straightened, the base's wheels touched
  the bed 0.37 m away. Keeping the path off the bed (0.42 / 0.45 m) only moved the collision to the bookcase on the same turn.
  Carry poses nearer the body centre failed IK or left the head camera's view.

**The fix:**
- Code, 65b6e486: opt-in `generation.carry_arrive_straight_m`. The carried path ends that far behind C along C's axis, at C's
  heading, and drives straight in (`NAV.plan_path arrive_straight_m`); otherwise the plain path is used as before.
- With 0.35 m, all 8 straight-arrival tries in 2 carry legs still touched the bookcase at the 0.65 m stance or 9 cm before it
  (`attached_object_left_eef_link`). The common pick's carry configuration holds the book 0.51 m ahead, which is too long for 0.65 m.
- Customs, dc5ba069: `carry_arrive_straight_m` 0.35 and `inside_stand_offs_m` [0.55, 0.65, 0.75, 0.85] (the default stops at 0.65).
  At 0.85 m the spine-out release screens F_goal 15 for both rolls; at 0.55–0.65 m it was 10 / 0.
- Local results: as3 wrote an instance at its 4th attempt, and its demo succeeded (1,771 steps, no QA flags). as4 (0.75 / 0.85 only)
  wrote 1 in 13 attempts.

**Remaining limiter (local):** the arm's last leg at C, from the carry configuration to the pre-place, touches the bookcase
(0.84 m from the base) in most coupled tries. The program check rejected 2 of as4's 3 accepted draws: it tracks that leg from where
the arrival leaves the arm, where the sampler tracked it through a carry waypoint.

**Validation** (fast flow, 6 jobs × N 2, k 1, `$OFH/out/<tag>/inter_object_relation_satisfaction-inside_large_scale_01/g*`):

| Tag | Snapshot | Written / attempts | Demos | Note |
|---|---|---|---|---|
| `fix1` (before) | ebd0fb16 | 0 / 4,746 | – | spine-out, every carry collided |
| `fix3_hb` | dc5ba069 | 7 / 30 | 1 / 6 | 5 × `robot_touching_itself` in push[2], all from picks pushed at open pitch 40− |
| `fix4_hb` | f2fc6efb (pitches [0, 20]) | 0 / 31 | – | pitch-0 push at joint2's limit everywhere; reverted (89e5bbf8) |
| `fix3b_hb` | dc5ba069 = the trunk | 12 / 104 | 11 / 12 | 1 `plan_failed`; 2.88 GPU-h, 0.26 GPU-h per successful demo |

- **Open: the pitch-40 push.** The open-pitch-40 push (the "40−" wrist) is written by the sampler, but its demo fails on self-contact
  in execution (`left_arm_link3 touches left_arm_link5`). It was 4 of 4 in fix3_hb. The sampler's push-chain IK and branch checks
  don't catch it.
- Restricting the pitch list doesn't work. With [0, 20] the pitch-0 push itself failed (`ik_joint_limit`, joint2 within 3 mrad);
  the pitch list likely changes the B ranking (`rank_b`). The fix would be a sampler-side self-proximity check on that push witness.
- How often pitch 40 is taken varies by run (fix3_hb 8 of 22 picks, fix3b_hb 1 of 28).

## 13. Round 4 (2026-11-07): the pitch-40 elbow graze, the hardback carry at C

### The pitch-40 push: `robot_touching_itself` (left_arm_link3 / left_arm_link5)
- **Where it occurs.** Across `$OFH/out` (19 tags, 467 demos) there are 7 `robot_touching_itself` failures, all on left link3 / link5:
  - 5 in the hardback push (fix3_hb 300023 / 24 / 28 / 29 / 33);
  - 1 in the nextto_04 gym-shoe grasp;
  - 1 in the ontop_large_01 soap-dispenser lift.
  The right arm has 0, and so does under_large_scale.

  Each is one step: the hard QA stops the run at the first contact. Left joint 4 is at −2.075..−2.090, against a limit of −2.0944.
- **What touches.** At those steps the collision hulls `link3_col_1` / `link5_col_0` are −0.1..0.0 mm apart, while the cuRobo spheres are 17.1–18.1 mm apart.
  - Sweep: +4.3 mm at q4 −2.045, and −0.9..−1.2 mm at the limit.
  - Demos that passed close by stayed ≥ 2.0 mm apart.
  - The right arm stays ≥ 6.9 mm apart at the limit.

  The pair is in neither cuRobo's ignore list nor r1pro.yaml's disabled pairs, and the USD has no FilteredPairs. This is a convex-hull graze within the joint limits, not a fold.
- **The QA exclusion (trunk acdea64e), opt-in `ISPATIALGYM_QA_SELF_EXCLUDE_ELBOW=1`.**
  - Left and right link3 / link5 leave `robot_touching_itself`.
  - Their contacts are still recorded in `qa.metrics.robot_touching_itself.excluded`.
  - Every report names the switch and `excluded_pairs`.

  Recollect selfx1 of the 5 failed instances: 5 / 5 succeed. The contact lasted 29–30 steps and perturbed the motion:
  - the hand slowed from 0.103 to 0.028 m/s;
  - joint 4 accelerated at up to 19–21 rad/s², against 0.76 in a pitch-40 push without the contact.
- **The PhysX filter (trunk 52f91f19), opt-in `ISPATIALGYM_PHYSX_FILTER_ELBOW=1`.**
  - `r1pro_model.filter_link_pairs_stopped` sets both links' FilteredPairsAPI targets on the loaded robot, with the sim stopped, in `apply_eval_base_mass`'s stop.
  - It is recorded in `meta.loader.physx_filtered_robot_pairs`.
  - No robot asset is touched.

  Recollect selfx2 (both switches): 5 / 5 succeed, with 0 excluded contacts. Joint 4 peaks at 0.82–0.85 rad/s², and its velocity stays within −0.008..+0.034 rad/s.

  Hardback selfx_hb (QA switch, 6 × N 2): 12 written in 29 attempts, demos 11 / 11 at the time of writing. That attempt count isn't comparable with fix3b_hb's 104, because of the persisted overhang memo and caches.
- **Open item.** A replay or evaluation world of these demos must apply the PhysX switch too. Today only `run_isg.build_env` applies it.

### The hardback carry (fix3b_hb)
- **Carried legs.**
  - face_travel is used on 8 of 12.
  - The other 4 are 0.41–0.44 m legs into C where face_travel would turn the body 81° against the polyline's 32°, beyond the +10° rule. They stay recorded on purpose.
  - There is no base hit in 66 carried paths since the straight arrival.
- **The limiter is the carry leg at C** (81 failures):
  - mostly the lift configuration's line sweeping the book into bookcase_jaysra_0, 150–246 mm into the leg;
  - differential-IK stalls at 13–14 / 41;
  - 11 branch failures.

  Every CARRY_POSES pose at B failed: 5 were out of the head camera frame, and 1 had joint 4 / 6 within 2 mrad of its limit. So all 12 instances carried at the lift configuration.
- **The compact-carry probe (trunk 340a0a3e, diagnostic only), opt-in `ISPATIALGYM_CARRY_COMPACT_PROBE=1`.**
  - It tests 27 chest poses at the travel trunk, logged as `CARRY_COMPACT_PROBE`: head camera box, IK with the book, in-hand line, and the carry leg at C from each.
  - Running as hbprobe1.
  - A real compact carry needs new plumbing: a transition never moves the holding arm (`execute.move_idle_arms`).
