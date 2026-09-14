# Resume notes (paused 2026-09-12 00:10; COMPLETED 2026-09-12 08:31 - see DONE.json)

## State

Done:
* Object selection for `world_alignment-0` / `world_alignment-1`, rounds 1-3: `selection.md` / `selection.json`,
  `spec_fragments.json` (picks + per-model frames in the sampler 1.4.1 dict form), `duplicates.json`; renders in
  `renders/`, reviews in `review/`; written into `runlogs/grasp_library_20260905/select/task_object_selection.csv`
  (+ `.xlsx` / `.html`) by `scripts/update_selection_csv.py`.
* Sampler 1.4.1 (uncommitted, in the working tree together with another session's 1.4.0 disembedding work):
  `ispatialgym/sampling/tasks/world_alignment.py` (`axis_param`, `_frame_axes`, `_hang_quat`, `_object_dims` along the
  frame axes, `fixed.wall` / `fixed.wall_regions`, `_region_spots`, wall with the most spots), `CHANGELOG.md` 1.4.1,
  `ispatialgym/sampling/__init__.py` version, `docs/instance_sampling_spec.md` section 5.8 items 5-6. The rotation
  maths was verified numerically and both problems were smoke-tested in the simulator on 2026-09-12 (see README, step 3 below is done).
* Wall survey (`scripts/survey_walls.py` via `scripts/run_survey_all.sh`, results `results/<scene>__<room>.json`,
  templates / TROs under `/tmp/isg_wall_survey`, logs `/tmp/isg_wall_survey/logs/`):
  * Rs_int: 5/5 rooms (1 validated region, entryway_0).
  * hotel_suite_large: 2/2 rooms (7 validated regions).
  * office_cubicles_right: 8/12 rooms done (`bathroom_0`, `conference_hall_0`, `copy_room_0`, `corridor_0`,
    `empty_room_0`, `lobby_0`, `meeting_room_0`, `private_office_0`); missing `private_office_1` (was in progress
    when paused; its `.inprogress` marker was removed so it is re-run), `private_office_2`, `private_office_3`,
    `shared_office_0`.
  * NOT surveyed yet: `restaurant_diner`, `house_double_floor_lower`, `house_double_floor_upper`,
    `house_single_floor`.
  * `wall_candidates.json` / `.md` and `maps/` currently hold the partial results above (regenerated at pause time),
    and the sheet's `world_alignment-1 / wall` block lists the 23 validated regions so far.

Nothing of mine is running. All survey processes and the wrapper were stopped; the GPU was idle at pause time.

Concurrent work by another session (not mine): it runs `runlogs/table_selection_20260911/scripts/validate_tables.py`
processes (two at a time, ~8 GB each) and rewrites the `table` blocks of the same sheet. At 21:43 I paused its
`queue_scenes.sh` launcher to free a GPU slot for the survey (killed the just-started hotel_suite_large validator;
its results are resumable with `--skip-done`), relaunched the queue at `ISG_TABLE_MAXP=1` at 21:52, and stopped my
relaunched queue at 22:42 when that session started managing its runs directly again. The two sessions' sheet
updates only touch their own blocks (both scripts re-read the CSV and replace whole blocks), so they compose, but do
not run both updaters at the same second.

## To do (in order)

1. Finish the wall survey (GPU: ~6-11 GB per process; the wrapper waits until >= 10 GB are free, never runs two
   survey processes, retries a scene up to 4 times after crashes and re-runs rooms left `.inprogress`):

       cd /home/minyeongk/Desktop/projects/ispatialgym/runlogs/world_alignment_selection_20260911
       nohup scripts/run_survey_all.sh "office_cubicles_right restaurant_diner house_double_floor_lower house_double_floor_upper house_single_floor" >> /tmp/isg_wall_survey/logs/all.log 2>&1 &
       # progress: grep -E "wall_survey.*\] .* -> " /tmp/isg_wall_survey/logs/<scene>.log ; tail /tmp/isg_wall_survey/logs/all.log

   Rooms take 1-10 min (big halls / corridors: rays + up to 60 strided reach tests + up to 8 instances); expect
   ~2-3 h for the remaining four scenes. If `/tmp/isg_wall_survey` was cleaned, just recreate it (`mkdir -p
   /tmp/isg_wall_survey/logs`); results in `results/` are what matters and `--skip-done` skips them.
   Rooms recorded as `"status": "crashed"` or `"error"` in `results/` (e.g. after a CUDA OOM while the table
   validators were running) can be re-run by deleting their json and relaunching the scene.

2. Aggregate and write the sheet again (CPU only):

       python3 scripts/aggregate_walls.py            # -> wall_candidates.json / .md, maps/<scene>_walls.png
       ~/miniconda3/bin/python scripts/update_selection_csv.py   # -> task_object_selection.csv / .xlsx / .html

3. (DONE 2026-09-12 07:25, both problems produced instances; renders in /tmp/isg_wa_test/renders) Test the sampler 1.4.1 changes in the simulator (needs ~7 GB GPU, ~5 min); a ready test spec is in
   `/tmp/wa_test_spec.json` (recreate it with the snippet below if /tmp was cleaned): `world_alignment-1` with
   `picture/jmvdxm` (front +z, up -x) and `painting/dacihj` on the validated Rs_int entryway region via
   `fixed.wall_regions`, and `world_alignment-0` with `hardback/agbqsy` (long x, front +x) + `keyboard/injrgc`:

       cd /home/minyeongk/Desktop/projects/ispatialgym
       OMNIGIBSON_HEADLESS=1 ~/miniconda3/envs/ispatialgym/bin/python -m ispatialgym.sampling.sample_instances \
           --spec /tmp/wa_test_spec.json --problem world_alignment-1 --scene Rs_int --n-instances 1 --out /tmp/isg_wa_test --overwrite
       OMNIGIBSON_HEADLESS=1 ~/miniconda3/envs/ispatialgym/bin/python -m ispatialgym.sampling.sample_instances \
           --spec /tmp/wa_test_spec.json --problem world_alignment-0 --scene house_single_floor --n-instances 1 --out /tmp/isg_wa_test --overwrite
       # then: python -m ispatialgym.sampling.render_instances --root /tmp/isg_wa_test  (check the picture hangs upright, the book lies cover-up)

   Spec snippet used to build `/tmp/wa_test_spec.json`: copy `world_alignment-1` / `world_alignment-0` from
   `ispatialgym/assets/task_specs/task_custom_lists.json`; for -1 set `roles.target.models = {"picture": ["jmvdxm"],
   "painting": ["dacihj"]}`, `scenes = {"Rs_int": wall_candidates.json["scenes"]["Rs_int"]["task_spec"]}` (rooms,
   fixed.floor, fixed.wall, fixed.wall_regions), `generation.front_axis = {"default": "+x", "picture/jmvdxm": "+z"}`,
   `generation.up_axis = {"default": "+z", "picture/jmvdxm": "-x"}`; for -0 set `roles.target.models =
   {"hardback": ["agbqsy"], "keyboard": ["injrgc"]}`, `generation.long_axis = {"default": "y", "hardback/agbqsy": "x"}`,
   `generation.front_axis = {"default": "+x"}`; `instances_per_template = 1`.

4. Optional follow-ups noted during the work: the test spec still pins keyboard/hgtbot + injrgc and painting/dacihj +
   ljabhq; the reviewers rank keyboard/jtdyix (16) above hgtbot (14) and painting/jayvqq (16) above ljabhq (12) -
   swap if wanted. Rs_int living_room_0 regions fail the instance stage with `scene_disturbed` (the robot base on
   the arc displaces `coffee_table_fqluyq_0` by ~1 m although the whole-body collision check passed) - a sampler
   issue worth a look, not a wall issue.

## Files touched outside this directory

* `runlogs/grasp_library_20260905/select/task_object_selection.{csv,xlsx,html}` (world-alignment blocks),
  `runlogs/grasp_library_20260905/select/legend_extra.json` (new; legend items picked up by every re-render),
  `runlogs/table_selection_20260911/scripts/update_selection_csv.py` (`render(rows, extra_legend=())` + reads
  `legend_extra.json`; behaviour otherwise unchanged).
* `ispatialgym/sampling/tasks/world_alignment.py`, `ispatialgym/sampling/CHANGELOG.md`,
  `ispatialgym/sampling/__init__.py` (1.4.1), `ispatialgym/docs/instance_sampling_spec.md` (5.8 items 5-6).
* Nothing committed. Backup of the sheet before my first update: `/tmp/task_object_selection.backup.csv`.
