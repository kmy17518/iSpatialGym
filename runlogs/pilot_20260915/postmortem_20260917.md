# Post-mortem — the 23:57–00:05 PDT subagent cut-off (written 2026-09-17 02:35 PDT, read-only step)

## Summary

Seven subagents of the coordinator chat stopped between 23:57 and 00:05 PDT, each in the middle of a tool call
(their transcripts end on an unanswered `tool_use`). The parent (this coordinator chat) had ended its last turn at
23:47 and received nothing until the owner's 01:05 message; between those two events no agent produced a transcript
line after 00:05. Every later attempt to resume one of the seven (23:08, 01:08–01:12, 01:36) was refused by the
platform with `Agent host session already exists`, including an explicit `interrupt: true` — consistent with a
restart of the agent host that recycled the child sessions while their session ids stayed registered. The one agent
started after the cut-off (hybrid-follow, 01:08) runs normally. **Cause hypothesis (e):** an agent-host / parent-session
interruption at ~23:57–00:05 (not a repo or pod event: pods, the scheduler loop and the pod-side jobs all kept running;
`git` history shows the agents' last commits at 23:56 and 00:01 and nothing of theirs after). What the parent transcript
shows: three background-shell notifications from agents' pod jobs at 23:43 / 23:45 / 23:47 (each answered), then silence
until 01:05 — no error, no system message. The coordinator's 01:03 side-chat status called the agents "active" on the
basis of their last *messages*; the transcript **mtimes** (all 23:57–00:05) already said otherwise. Rule from now on:
report agent activity from transcript timestamps, never from message content.

## (a) Per interrupted agent — last tool call, landed, in flight

| agent | id | transcript ends | last tool call (what it was doing) | landed today (fix log / git) | in flight / half-done |
|---|---|---|---|---|---|
| core (D2) | c851dacc | 00:04:48 | Shell: editing a TEMP customs JSON (`ISG_CUSTOMS` copy) to run one focused knob probe on a thin object at +10 mm ("verify the height-extra-to-grasp and touching-lift code paths") | `36067482` `7737dfe5` `0bd9e1f` `bcc1478` (inside), `6de8803` (touching brace optional), `dc6a1a5` (carry waypoint), `bd10c72` (Transition.holding), `667416d`/`02c700d` (grasp draw), `1ba5e74` (open-pad clearance), `385de1e` (open-jaw release), `5555526` (robust Under, D-007), `a6d7123` (touching push length), `453b306` (pinch centring), `260ff93` (A7 pair naming), `dde4f11` (witness quality), `7dc48ed` (handle grasp carries nothing), `835ee13` (contact_height knob + `ISG_CUSTOMS`) | the knob probe (never launched); the D-007 wrapper is live; `dde4f11`'s 0.05 rad margin **regressed wall alignment** (large-scale-B note 00:05: 8/8 `push_unreachable`, the wall pushes need joint 2 at its limit) — an exemption / relaxation is owed |
| tabletop (D3) | 68c175a7 | 00:04:44 | Shell: a Python probe over `scripts/runpod` — block-design place_unreachable at the low trunks ("trunks reachable within the 30° cone all sit low with steep tilts") | `5e30334` (release helper), `aea6bd0a` (25° cone), `58afae2`/`bd30c0cf`/`deb0093c` (attachment 2nd base pose, brace faces, carry ladder), `4c66d53a`/`3a7db49c`/`96042cfe`/`3ec1a1b0`/`99251f9` (block design gate, caps, knobs, anchor-first), `0ec0153d` (flip-on-table prep), `7bb88c17`/`d4546de2`/`6dede1f5` (hinged-jar brace search), `ac8992de` (D-008/D-009 switches), `a9ae9ab` (close hinge facing) | block design still `target_placement` ×8 (owner 01:00: reach, not the gap — region rule (a) / recorded place stance (b) not started); attachment SLERP start (carried lift 0.70–0.82 m unreachable from the attach stance) not started; toolbox `no_stance` (all three supports, 01:19–02:19) undiagnosed; TEMP customs `runlogs/smoke_inputs/customs_block*.json`, probe scripts `probe_*.py` (uncommitted, ignored) |
| large-scale-B | 87e668ec | 00:05:11 | Shell: demo-only of the painting from `runlogs/smoke_inputs/large-scale-B/inst_ab_margin0/…` (an instance sampled with `JOINT_LIMIT_MARGIN = 0` in-process) with the hinge record | `b07fc7b` (wall hang frame from the icl, never defaulted), `21dddce` (hinge friction + damping in the constraints record + loader), `aa11be5` (two-arm push witnesses), `b37782b` (push-under 900 s budget); fix-log NOTE 00:05 on the `dde4f11` regression | the painting demo-only with damping (probe-margin0 instance) — result unknown (its run dir is on a pod, see (b)); the hinge damping values were measured (`hinge_probe.py`) but the demo verdict is not recorded; both wall subtasks blocked by the margin rule |
| primary oracle | 7deabbed | 23:59:23 | Shell: `git commit` of `execute.py` ("the approach / descent finish may cover the whole 10 cm leg") — landed as `16c079b` 23:56; then working the touching place descent stall (waypoint 2/12, 5.5 mm) | wall chain (`4b2bb60` … `3eb8e51`), `166005ff`/`513c536` (FOV rule b + carry leg), `041ca85c`/`0d9c116b` (axial retreat), `72b55d6`/`80779eb`/`b2bdabf`/`f3546f9`/`eac6c96`/`16c079b` (witness refinement, finishes), `3a06791` (retreat wind-up), `0ee927f6` (block flip program) | **the straight-carry-leg regression** (30 waypoints, stall 25/30, finish refused at 3.74 rad — teacup 0916-224214, touching 0916-225035; owner 01:00 item 4 = top priority) NOT fixed; the D-007 branch report of a passing teacup run therefore pending; ghost_hand on the handle link (drawer / dishwasher `base_follow` — the dishwasher instance of 02:01 failed exactly there) not done; nextto place plan_failed not diagnosed |
| oracle-B | ab3f104d | 00:05:03 | Shell: `sleep 295` then reading an axial demo remote.log (`0916-235025`) — watching a table-loading measurement run | `0a81f739` (touching program), `6baaaa3b` (attachment program), `38258f00` (gate + cylinder equivalences), `bcc937df` (closure ramp + pre-shape), `de4e7231` (staged release), `59f6426f` (small-articulation program), `2e889f3e` (meta.binary_branch), `ab60baa7` (symmetric closure), `067257dd` (meta.support_disturbance) | table-loading measurement partly done (`runlogs/table_loading_20260916/report.md`, `measurements.json` exist — the fixed_base answer: the breakfast table is **dynamic**, not fixed; table displacement 0.28 mm without / 0.02 mm with the pre-shape on the measured run); the full 69-demo table not finished |
| A/B agent | be09c08d | 23:57:23 | Grep: reading how `extras` is flattened in the sidecar while waiting for its jobs | its launcher (`launch_ab.py`, a background loop) **outlived the agent**: all 18 jobs ran (run ids 23:43–00:56), outputs pulled under `runlogs/ab_contact_height_20260916/<alias>/<offset>/`, `manifest.json` complete: 3 `done` (nextto 0 mm / 10 mm, axial 5 mm), 4 `demo_failed`, 11 `no_instance`; `report.md` is the launch-time skeleton | the analysis (`analyze_ab.py`, the per-alias × offset table, the decision inputs) was never run |
| primary large-scale | f0a45699 | 23:07:41 | its 23:05 report was its last output; the 23:08 follow-up (icl mirror of D-004, candidates' verdicts, sandal, countertop) never produced a transcript line | `0427202` … `1665380` (drawer bar offset, base clearance), `9a3e678`, `1c829ae`, `6c97f8e`, `7162fe7`/`8c93d0b`/`2cfce47` (push-under lanes), `4df6d0a`/`514ad3f`/`c5040c8`/`b931b31`/`729a1ec`/`1dd3820`/`f0684cb` (ontop-large) | icl mirror of the D-004 blacklist not done; chained open for doors (owner 01:00 3(2)) not started; sandal F_goal-0 diagnosis and the countertop-floor coupling open; TEMP customs `runlogs/smoke_inputs/isg_d2d3_rsint/isg_customs.json` (the bamfsz_1 probe — the pick is now in the real customs `0589d9d`) |

`git status` is clean under `ispatialgym/` and `scripts/` (only `scripts/slack_sidechat/`, the owner's side-chat helper, is untracked). Nothing is half-edited in the tree. TEMP inputs still referenced by runs on pods: `runlogs/smoke_inputs/hybrid_follow/customs_hybrid.json` (the live hybrid probes), `runlogs/smoke_inputs/ab_contact_height/customs_{0,5,10}mm.json` (finished A/B jobs), `runlogs/smoke_inputs/large-scale-B/inst_ab_margin0/` (the margin-0 painting instance), `runlogs/smoke_inputs/isg_d2d3_rsint/isg_customs.json`.

## (b) Pod-side state (survey 02:25 PDT, 17 live pool pods, 0 band pods)

Processes alive on 6 pods, none orphaned: 5 are the live hybrid-follow agent's probes (`customs_hybrid.json`: dishwasher on `1c88jsy46ubecf`, top_cabinet on `h4enpm19l0t3ng` and `pjf6451ahyqxll`, door on `6pdrunfn78iofy`, close top_cabinet on `5gm563m9wgnrwc`; their `remote_run`s are alive — reparented to systemd because the agent's tool shells ended — and hold the locks), and 1 is the scheduler's block-design first-demo run on `htvfrubebp6tnj` (`first_demo.py` pid 2699972 / remote_run 2699996, started 00:39, `sample_isg` alive 1 h 43 min at the survey). Every pod-side process of the dead agents has ended by itself (their `timeout` wrappers or completion); no `run_isg` / `reach_bands` process anywhere; `/root/samp/running/` empty on every pod.

Run directories on the pods vs local pulls (a run counts as pulled when a local `remote.log` names its `(pod, run id)` — 395 such pairs locally; agents that pulled to `/tmp` or wrote no remote.log are not credited, so these are upper bounds of what is missing):

| pod | name | /root/smoke runs | not credited locally | /root/samp jobs | sizes (smoke, samp) |
|---|---|---|---|---|---|
| 1c88jsy46ubecf | pilot2-a40-6 | 16 | 7 | 14 | 43M, 18M |
| 29s7u374iifjbj | pilot2-a40-10 | 19 | 7 | 32 | 87M, 19M |
| 5gm563m9wgnrwc | pilot2-l40s-10 | 18 | 4 | 62 | 82M, 71M |
| 6pdrunfn78iofy | pilot2-a40-4 | 23 | 7 | 32 | 89M, 24M |
| bl91tuqop45ila | pilot2-l40s-2 | 21 | 5 | 60 | 74M, 24M |
| bpu6l9bffwaf1y | pilot2-l40s-11 | 35 | 10 | 17 | 128M, 54M |
| cy0f5t6i58i08v | pilot2-l40s-3 | 11 | 4 | 66 | 36M, 47M |
| gvua02xnwm4ti6 | pilot2-a40-9 | 17 | 8 | 32 | 94M, 16M |
| h4enpm19l0t3ng | smoke2-a40-0 | 36 | 9 | 0 | 154M |
| htvfrubebp6tnj | pilot2-a40-3 | 23 | 7 | 13 | 82M, 22M |
| nfgjwzgjajpr5u | pilot2-l40s-7 | 14 | 2 | 63 | 49M, 45M |
| oliefq3ymuexna | pilot2-a40-1 | 23 | 1 | 18 | 125M, 35M |
| pjf6451ahyqxll | smoke2-a40-1 | 35 | 5 | 0 | 135M |
| s7m4h2ay2slau4 | pilot2-l40s-4 | 10 | 2 | 62 | 41M, 26M |
| uilwpybjxnekg0 | smoke2-a40-2 | 26 | 8 | 1 | 165M, 96K |
| xnam1nh3y2706e | smoke-l40s-3 | 55 | 15 | 4 | 277M, 1020K |
| z2qdfyfyrgcjys | pilot2-a40-0 | 17 | 7 | 18 | 61M, 33M |

Total ≈ 1.7 GB of `/root/smoke` + 0.4 GB of `/root/samp` (rb_cache excluded) — everything is pulled in step 3 regardless of the credit check. The 10 pods terminated at 21:37 were empty of running work but their `/root/smoke` dirs (agent probes pulled to `/tmp`, old pilot runs) are gone with them; the held-out band files they held had been collected at 20:4x–21:4x (1448 files in `bands/`; `fleet_remaining` found 0 remaining at 21:50).

## (c) Scheduler actions since 00:00 and the current locks

Launches (all on unchanged or coordinator-only code): 00:03 under-large `apple-desk-floor` (→ `no_init_surface`), 00:09 `dishwasher` (→ `handle_unreachable` ×8) + `hinged_jar` open (→ `no_stance`), 00:39 block design (`dde4f11`/`99251f9` entered it; running since), 01:08 attachment on `41782c9` (→ `program_check_failed` ×6 + `target_placement` ×2) + hinged-jar close (cancelled 01:11 by the coordinator, D-012), 01:17 close-large drawer on `7c7bac8` (→ `program_check_failed` ×6 + `handle_unreachable` ×2). No role moves, no preemptions since 00:00 (the band fleet was empty; `rb/pods.txt` still listed 4 terminated pods until 01:06). Locks held at 02:25: `1c88jsy46ubecf`, `5gm563m9wgnrwc`, `6pdrunfn78iofy`, `h4enpm19l0t3ng`, `pjf6451ahyqxll` (hybrid-follow probes), `htvfrubebp6tnj` (block design first-demo run). 11 pool pods free.

Coordinator-launched runs since 01:05 and their verdicts: the five door aliases on the 20° first-demo opening (`7c7bac8`) — oven, fridge, bottom_cabinet_no_top `handle_unreachable` ×8, top_cabinet ×7 + `program_check_failed`, **dishwasher wrote an instance at 02:01** (`approach_untrackable` ×2, `goal_unstable` ×1 before) whose demo failed `ghost_hand` at `base_follow` (the known executor issue: the object root is compared to the hand while the door link moves); the five small-articulation candidates (`41782c9`) — laptop `missing_handle` (pinch width), carton_msfzpz `missing_handle` ×2, toolbox `no_stance` at setup on all three supports (edge +v / +u, the earlier `no_workspace` class).

## (d) A/B experiment state

`runlogs/ab_contact_height_20260916/manifest.json`: 18 jobs, all finished (launcher loop, run ids 234359 → 005640, pods 5gm5…, bpu6…, nfgj…, xnam…, 1c88…, bl91…, s7m4…, 6pdr…, 29s7…, olie…, pjf6…, h4en…). Status: `done` = nextto 0 mm (`001730-902ccf`), nextto 10 mm (`001911-ca32ca`), axial 5 mm (`004421-cb6e70`); `demo_failed` = under 5 mm, touching 5 mm, touching 10 mm, axial 0 mm; `no_instance` = ontop 0/5/10 mm, under 0/10 mm, inside 0/5/10 mm, nextto 5 mm, touching 0 mm, axial 10 mm. All pods released. The outputs are local (`<alias>/<offset>/`); `analyze_ab.py` exists but was never run; the report is the skeleton. The knob (`835ee13`) and `ISG_CUSTOMS` are in the tree; the TEMP customs are `runlogs/smoke_inputs/ab_contact_height/customs_*.json`.

## (e) Cause hypothesis

See the summary: an agent-host / parent-session interruption at ≈ 23:57–00:05. Evidence: (1) seven independent agents stopped within eight minutes, each mid tool-call; (2) the parent chat was idle from 23:47 to 01:05 and shows no event; (3) resumes and an interrupt of those ids fail with `Agent host session already exists` while a new agent works; (4) the machine, the pods, the scheduler and the agents' own background launchers (`launch_ab.py`, pod jobs) kept running — the cut was in the agent host, not the OS.
