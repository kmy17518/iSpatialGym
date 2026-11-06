# trainfast: the fast flow's environment (sourced by job_fast.sh after env_fast.sh). Existing opt-ins of the code base that production
# left off for bit-identity with the counted eval runs (the scale-up needs no bit-identity) plus the train-fast switches
# (ISPATIALGYM_FAST_* default on in the worktree; ISPATIALGYM_FAST=0 turns them all off for an A/B on the same code).
export ISPATIALGYM_SAMPLER_MESH_CACHE=${ISGF_SAMPLER_MESH_CACHE:-auto}     # snug cuRobo mesh store: a padded batched IK call 10 s -> ~1-2 s
export ISPATIALGYM_EXECUTOR_MESH_CACHE=${ISGF_EXECUTOR_MESH_CACHE:-room}   # the oracle's store sized to the room's meshes (trajopt -25 %)
export ISPATIALGYM_SAMPLER_STEP1_ONCE=1                                     # large articulation: Phase-2 step 1 once per attempt (2.5x)
export ISPATIALGYM_SEEDED_SPHERE_FIT=${ISGF_SPHERE_FIT:-cache}             # attach-sphere fits seeded + cached (not bit-identical)
export ISPATIALGYM_GOAL_OCCLUSION=off                                       # the goal-occlusion renders move to the replay stage
export ISPATIALGYM_HEAD_CAMERA_ONLY=1                                       # no wrist cameras in either process
export ISPATIALGYM_LAZY_FRAMES=1 ISPATIALGYM_ASYNC_SHARD=1                 # recorder: no per-step frame copies without a video sink
unset ISPATIALGYM_PHYSICS_ONLY                                              # set per process: collectors only (job_fast.sh)
COLLECT_EXTRA=${ISGF_COLLECT_EXTRA:-}
# round 2 (2026-10-31 19:50, after the first smoke2 attempts): coarser start-draw cells so the per-template cache warms up within a job
# (59 cells per edge at 2 cm -> 24 at 5 cm; the lateral is still drawn uniformly, then snapped), fewer screening seeds (12 -> 8: the exact
# IK confirms every drawn pair anyway), and the band's stored feasible sets reused instead of re-screening (code switch BAND_FEASIBLE_REUSE)
export ISPATIALGYM_FAST_SPA_CELL_M=${ISGF_SPA_CELL_M:-0.05}
export ISPATIALGYM_FAST_BATCH_IK_SEEDS=${ISGF_BATCH_IK_SEEDS:-8}
