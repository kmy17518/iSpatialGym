# trainfast: the fast flow's environment (sourced by job_fast.sh after env_fast.sh). Existing opt-ins of the code base that production
# left off for bit-identity with the counted eval runs (the scale-up needs no bit-identity) plus the train-fast switches
# (ISPATIALGYM_FAST_* default on in the worktree; ISPATIALGYM_FAST=0 turns them all off for an A/B on the same code).
export ISPATIALGYM_SAMPLER_MESH_CACHE=${ISGF_SAMPLER_MESH_CACHE:-auto}     # snug cuRobo mesh store: a padded batched IK call 10 s -> ~1-2 s
export ISPATIALGYM_EXECUTOR_MESH_CACHE=${ISGF_EXECUTOR_MESH_CACHE:-room}   # the oracle's store sized to the room's meshes (trajopt -25 %)
export ISPATIALGYM_SAMPLER_STEP1_ONCE=1                                     # large articulation: Phase-2 step 1 once per attempt (2.5x)
export ISPATIALGYM_SEEDED_SPHERE_FIT=${ISGF_SPHERE_FIT:-cache}             # attach-sphere fits seeded + cached (not bit-identical)
# goal occlusion (owner 2026-11-05): screened in the SAMPLER only for mental_rotation and disembedding (submit_fast.py sets ISGF_GOAL_OCCLUSION=on
# for their keys: under SAMPLER_CAMERAS they keep a camera, the check costs ~2 s per instance and rejects ~6 %), in the replay stage for every
# other family (multienv/replay.py --goal-occlusion); off here also keeps the collector's metric off (PHYSICS_ONLY_TRIM records it skipped)
export ISPATIALGYM_GOAL_OCCLUSION=${ISGF_GOAL_OCCLUSION:-off}
export ISPATIALGYM_HEAD_CAMERA_ONLY=1                                       # no wrist cameras in either process
export ISPATIALGYM_LAZY_FRAMES=1 ISPATIALGYM_ASYNC_SHARD=1                 # recorder: no per-step frame copies without a video sink
unset ISPATIALGYM_PHYSICS_ONLY                                              # set per process: collectors only (job_fast.sh)
COLLECT_EXTRA=${ISGF_COLLECT_EXTRA:-}
# round 2 (2026-10-31 19:50, after the first smoke2 attempts): coarser start-draw cells so the per-template cache warms up within a job
# (59 cells per edge at 2 cm -> 24 at 5 cm; the lateral is still drawn uniformly, then snapped), fewer screening seeds (12 -> 8: the exact
# IK confirms every drawn pair anyway), and the band's stored feasible sets reused instead of re-screening (code switch BAND_FEASIBLE_REUSE)
export ISPATIALGYM_FAST_SPA_CELL_M=${ISGF_SPA_CELL_M:-0.05}
export ISPATIALGYM_FAST_BATCH_IK_SEEDS=${ISGF_BATCH_IK_SEEDS:-8}
# 2026-11-05 one trunk (owner): vec_next's persistent asset-geometry cache (Behavior-1K overlay utils/isg_asset_cache.py, content-keyed, a
# process writes a shard only when it computed something new) shared by the sampler and the collectors of every job: the first world load
# -22 % warm, demos bit-identical filling or reading it (merge_vecnext 2026-11-02). ISGF_ASSET_CACHE= (empty) turns it off.
ISGF_ASSET_CACHE=${ISGF_ASSET_CACHE-/vision/u/minyeong/projects/isg_home/oracle_fix/asset_cache}
if [ -n "$ISGF_ASSET_CACHE" ]; then export ISG_VEC_ASSET_CACHE_DIR=$ISGF_ASSET_CACHE; mkdir -p $ISGF_ASSET_CACHE; else unset ISG_VEC_ASSET_CACHE_DIR; fi
