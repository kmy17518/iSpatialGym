# Table / work-surface pre-selection for the table-based iSpatialGym tasks (2026-09-11)

Goal: for every problem that binds a fixed `table` role (disembedding, block design, articulation of a non-scene
object, attachment, world alignment / table surface, IORS ontop / under / inside / nextto / touching), find the scene
tables and work surfaces of the seven candidate scenes (`house_single_floor`, `house_double_floor_lower`,
`house_double_floor_upper`, `Rs_int`, `hotel_suite_large`, `office_cubicles_right`, `restaurant_diner`) on which the
instance sampler can generate a valid instance, and record them as candidates in
`runlogs/grasp_library_20260905/select/task_object_selection.csv` (the `table` row of each task).

## Method

1. **Geometric first pass** (`scripts/inventory_work_surfaces.py`, no simulator) - every object of the stable scene
   jsons whose category is a table (`TABLE_CATEGORIES` of the sampler: breakfast_table, desk, coffee_table,
   conference_table, pedestal_table, countertop, bar, nightstand, console_table, side_table) or another work surface
   present in the scenes (commercial_kitchen_table, cubicle, booth, bottom_cabinet, bottom_cabinet_no_top,
   metal_bottom_cabinet, serving_cart, stand, shelf, bench, ottoman), with its room(s), scaled bbox, estimated top
   height (`root z + base_link_offset`, verified against the raycast top of `breakfast_table_rhjoby_0`) and a verdict:
   `keep`, `low` (top < 0.55 m), `high` (top > 1.20 m), `small` (< 0.30 m side or < 0.12 m^2), `no_room`.
   Output: `inventory.csv` / `inventory.json`. Beds / sofas are listed as `under_partner` for IORS-1.
   Candidates handed to the sampler: `keep`, the cubicles (their desk surface is what the raster finds, not the
   1.2 m partition top) and `low` supports with top >= 0.40 m (`--include-low`); outdoor rooms excluded (the
   segmentation renderer crashes with the garden in view, see `instance_sampling_spec.md` section 8).
2. **Sampler validation** (`scripts/validate_tables.py`, one OmniGibson process per scene, `scripts/run_scene.sh`
   restarts after crashes, `scripts/queue_scenes.sh` keeps two processes on the GPU) - for every (support, problem) the
   test-round task spec entry is rewritten so that `scenes.<scene>.fixed.table = [<support>]` (rooms = the support's
   `in_rooms`, floor = its first room) and the sampler runs exactly like `sample_instances.py`:
   `TaskSampler.setup()` (support raster from raycasts, canonical stance + viewing trunk, reachability / grasp
   tables) then `TaskSampler.run(n)` (sample, settle, BEHAVIOR validation, `:init` true, binary goal false, goal
   feasibility by teleport + settle). A support is **viable** for a problem when >= 1 instance was written
   (`--n-instances 1`, `--max-attempts 6`; disembedding 10 attempts and a second template seed, i.e. a re-drawn
   distractor set). The pinned test objects of `task_custom_lists.json` are used (cereal box + trays, Kohs blocks,
   toolbox, cap + lemonade bottle, keyboard, kielbasa + platter, lemon + bowl, spatula + ladle, baking-mix can +
   board game). IORS-1 (under) runs over (bed / sofa partner, table) pairs within 4 m (`--under`).
   Shortcuts: a support around which the most lenient stance search (IORS ontop, four stand-offs) finds no stance at
   all skips the other problems; articulation close (`-2`) only runs where open (`-1`) produced an instance.
   Driver-side robustness: >= 6 renders before every segmentation read and a re-registration of in-view spawned
   objects the label map does not list (the batch runs otherwise read 0 px for freshly placed trays), restart on GPU
   out-of-memory instead of recording a result.
3. **Aggregation** (`scripts/aggregate_results.py`) -> `results_summary.csv` (one row per run), `candidates_by_task.md`
   / `.json` (per problem, per scene: viable supports with evidence, rejected supports with the sampler's reason).
4. **Sheet update** (`scripts/update_selection_csv.py`) - rewrites the `table` header row of each task in
   `task_object_selection.csv` (counts + per-scene summary) and inserts one `viable` / `spec` sub-row per validated
   support; regenerates the `.xlsx` / `.html` twins with the same styling.

## Outcome (2026-09-12)

1114 sampler runs (7 scenes, 90 supports, 10 table problems + 34 under pairs; passes: main sweep with 6 attempts,
disembedding second pass, 15-attempt re-run of every `no_instances` support). 185 runs viable. Viable supports per
problem (details and evidence in `candidates_by_task.md`):

| problem | viable supports | where |
| --- | --- | --- |
| disembedding-0 | 2 | house_single_floor breakfast_table_rhjoby_0 (test table, 1/7), office desk_mdhelw_0 (1/7) |
| configuration_matching-0 (block design) | 21 | house_single_floor 5, upper 3, Rs_int 2, office 3, diner 8 |
| configuration_matching-1 / -2 (articulation, toolbox) | 2 | breakfast_table_rhjoby_0, office desk_pfsjbc_0 (lobby) |
| configuration_matching-3 (attachment) | 25 | 6 scenes (not house_double_floor_lower) |
| world_alignment-0 (keyboard) | 15 | house_single_floor 2, hotel desk_ohjotf_0, office 5, diner 7 (deep tops only) |
| IORS-0 ontop / -2 inside / -3 nextto / -4 touching | 30 / 30 / 27 / 31 | 6 scenes; none in house_double_floor_lower |
| IORS-1 under | 0 | no (bed/sofa, table) pair passed with the table as init support; 3 house_single_floor pairs pass only through the sampler's bed-top fallback |

Typical rejection reasons: no collision-free / traversable stance at 0.45 m from any edge (cubicles, nightstands
against beds, cabinets in corridors), top unreachable for the generic top-down grasp (seats at 0.4-0.5 m, bar tops
at 1.05-1.08 m), too little free top (kitchen base cabinets under countertops, office pedestals), placement region too
small for the object (keyboard needs >= ~0.55 m of usable depth; two trays need a long free edge), destination tray
never "barely visible" (disembedding), no floor point with clearance under the partner (IORS under).
`house_double_floor_lower` has no viable support for any task: the kitchen breakfast table has no traversable
stance around it even with its four chairs removed (nook), the two bar counters give no placement region / no
collision-free stance (bar_byvbuc_0's template was fine but 15 ontop attempts yielded no instance), the coffee table
top is out of reach (0.41 m) and the bathroom / living-room cabinets are unreachable from any stance.

## Sampler version

The sampler was being developed in parallel (uncommitted working-tree changes on top of `d451c49`: chair-removal
scene variants for table tasks, `distractors.pools` spec section, `init_arms` / no silent `ready` fallback, ...); the
spec schema changed once while a batch was running (`[spec] no 'distractors' pool` rejections). The batch therefore
runs against a **frozen copy of the working tree taken at 2026-09-11 22:32** (`/tmp/isg_table_sel/snapshot`, `rsync`
of `ispatialgym/` + symlinked `runlogs/` and `Behavior-1K/`; `working_tree.diff` and `HEAD.txt` next to it record the
exact code; `run_scene.sh` exports `ISG_TABLE_REPO` / `PYTHONPATH` to it). Like `sample_instances.py` of that version,
the driver removes the chairs within 1 m of the work table (`scene_variants.write_variant`, traversability map patched)
before loading the scene; `results/*.json` record `removed_chairs` and `sampler_code`. Results produced earlier with
the unfrozen tree and without chair removal are kept in `superseded/results_v0_no_chair_removal/` for reference only.

`DONE` (json) is written when the whole run, including the visual-check renders in `renders/` (one viable template
per problem and scene for disembedding, block design, articulation, world alignment and IORS ontop, BEHAVIOR replay
layout), has finished - other sessions can poll for it.

Evidence per run: `results/<scene>__<support>[__under_<partner>]__<problem>.json` (status, instances written,
rejection counts, template_meta digest - support raster, stance, trunk, reach cells -, per-instance summary flags,
path of the generated template / TRO under `/tmp/isg_table_sel/<scene>/<tag>/`). `results/<scene>__<support>__precheck.json`
holds the generic support raster / stance check. `debug/` holds head-camera renders of the disembedding stance search
on the diner back-bar cabinet (`scripts/debug_support_view.py`).
