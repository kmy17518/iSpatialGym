# Batched replay (render) on Vast.ai — og-isgrend-20261115

Renders campaign_20261110 demos (physics-only `demo.hdf5`) into observation data, for training (LeRobot v3) and for eval instances
(the `ispatialgym-eval-data` layout). Adapted from `train85/work/vast_replay` (isg-1020-b1k); branch `campaign-20261115-batched-replay`
(= train_r7 77989209 + these commits).

Record dir `RV=/vision/u/minyeong/projects/isg_home/campaign_20261110/training_generation/batched_replay_r1`, R2 prefix
`JOB_S3=s3://isg-demos/jobs/og-isgrend-20261115`, instance labels `og-isgrend-<tag>`, ledger `$RV/ledger/events.jsonl`.
Credentials: `source $RV/tool/creds.sh` (cluster only; never prints secrets).

## What the replayer does (`python -m ispatialgym.multienv.replay`)

Fixed flags: `--slots 1 --batched-usd-edits --async-flush --proprio loaded --warmup-s 15`, `ISPATIALGYM_SHARD_FORMAT=legacy`,
`--scene full` (the cropped recorded world merged into the template's task rooms + adjacent rooms; room assertion), and:

| flag | meaning |
|---|---|
| `--image-size 224` | every camera (head, both wrists) renders and is written at 224 x 224, same apertures (same FOV) |
| `--settle-pad 0.2 --settle-pad-min 50` | episode = action steps A (meta.json `n_action_steps`) + max(ceil(0.2 A), 50) settle frames, REPLACING the 45 recorded settle steps; rows past the recorded end repeat the last recorded row |
| `--goal-images on` | the episode's last rendered head frame -> `goal_images/<dataset>/episode_<i>.png` (+ record) |
| `--instance-meta <jsonl>` | linkage fields carried into `episodes_isg.jsonl` (with demo / TRO / template paths + sha256) |
| train: `--scene-pert remove --scene-pert-plan auto --lighting-pert replay` | replay-time scene perturbation (one object outside the crop, per goal group, p=0.5; never for camera relocalization, which keeps its collection-time removal) and lighting (fixtures x U[0.5,1.5] + tint per goal group, skybox per process, p=0.5); the recorded collection lighting is ignored |
| eval: `--scene-pert off --lighting-pert off --goal-occlusion off` | no perturbation |

## Steps

1. Code snapshot: `git archive HEAD ispatialgym scripts` into `$RV/code/rend_<sha>` (other entries symlinked to the frozen r6 snapshot
   `R/code/train_fbdf9d93bbbc`), then `bash setup_bundle.sh $RV/code/rend_<sha> $RV/code/rend_<sha>/scripts/batched_replay/vast`
   (boot list = the eval fleet's, read only, + patched Behavior-1K; code tarball; worker.tar.gz).
2. Rows -> units: `make_units.py rows-eval1000 --manifest <eval1000 manifest> --out rows.jsonl --purpose eval|train ...` (training
   demos: any jsonl with `demo_dir`, `purpose`, `class_id`, linkage fields), then
   `make_units.py build --rows rows.jsonl --run <run> --code rend_<sha> --upload` (units = same purpose + same full template +
   same rooms; goal groups whole; ~6000 frames).
3. Fleet: `vastctl.py --run <run> offers|launch --offer A,B --procs P|status|reap|loop --gpus N|destroy ID`. Only og-isgrend- instances
   in this ledger are ever touched; `R/vast_global/STOP_ALL_RENTALS` or `$RV/STOP_LAUNCH` stops rentals. Workers self-destroy when no
   unit is left. Progress without the (429-limited) Vast API: `sync.py --run <run> status`.
4. Pull: `sync.py --run <run> pull` -> `$RV/runs/<run>/out/<unit>/`.
5. Merge:
   - training: `merge_train.py --runs <run>[,..] --out <dir> --name <name> --load-test` (LeRobot v3; `isg_meta/` goal images, TROs,
     templates, settle windows, task groups, per-episode linkage). Loads with LeRobotDataset and ACT's B1KDataset
     (`goal_source=episode_last`, `--settle-steps 0.2`).
   - eval: `export_eval.py --runs <run> --out <dir>/ispatialgym-eval-instances` (+ `ispatialgym-eval-demos`; loads with
     `ispatialgym.eval.instances.load_index`); optionally `merge_train.py --purpose eval` for the eval episodes in LeRobot v3.
6. Upload (owner's go-ahead only): `sbatch ... --export=ALL,SRC=<dir>,REPO=kmy17518/<name> upload_hf.sh` (private repo enforced).
7. Review: `review_video.py --out review.mp4 <unit out dirs>` (tiled head / wrists / goal image, perturbation and settle marked).

## Pitfalls

- 953 eval-1000 input files (80 online units) are shiyuc-owned mode 600: `readable.py` reads them from the shiyuc-made mirror
  `/vision/u/shiyuc/minyeong/isgrend_mirror/<abs path>` (refresh: as shiyuc, `cp -p` each unreadable file there and chmod 644).
- s5cmd's temporary download names overflow 255 characters on the replayed TRO names: `sync.py` streams `instances/`.
- LeRobot's load test needs node-local HF caches (`HF_HOME=/tmp/...`): NFS file locks fail.
