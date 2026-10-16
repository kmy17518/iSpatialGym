# iSpatialGym

iSpatialGym is a comprehensive simulation benchmark for testing embodied AI agents on spatial intelligence.

## 📢 News

- **[2026-10]** Pre-release of iSpatialGym with 13 evaluation instances and the evaluation pipeline.

## 🛠️ Installation

Tested on Linux x86_64 with an NVIDIA RTX GPU (driver `>= 535`, Isaac Sim 5.1), `conda`, `git`, and ~60 GB of disk space.

```bash
git clone --recurse-submodules https://github.com/kmy17518/iSpatialGym.git
cd iSpatialGym
./setup.sh --new-env ispatialgym --omnigibson --bddl --dataset \
    --accept-conda-tos --accept-nvidia-eula --accept-dataset-tos
```

Activate the environment and verify the installation:

```bash
conda activate ispatialgym
python -m ispatialgym.eval.instances list --level subtask
python -m pytest
```

## 📊 Evaluation

### Understanding the Dataset Format

The evaluation dataset (`ispatialgym-eval-instances`) is downloaded during setup into `Behavior-1K/datasets/ispatialgym-eval-instances/` (or `$ISPATIALGYM_INSTANCES_PATH`). It contains 13 instances across 7 task families (`alignment`, `camera_relocalization`, `configuration_matching`, `disembedding`, `inter_object_relation_satisfaction`, `mental_rotation`, `object_scaling`), organized hierarchically as `task_family` → `task` → `subtask` → `instance`.

```text
ispatialgym-eval-instances/
├── metadata/
│   ├── instances.jsonl   # Per-instance index (task hierarchy, scene, problem, max_steps, file paths)
│   ├── episodes.jsonl    # Per-instance demo metadata (length, demo HDF5, and rollout video paths)
│   └── tasks.yaml        # Hierarchy mapping: task_family -> task -> [subtask, ...]
└── scenes/<scene>/json/
    ├── <scene>_task_<subtask>_0_0_template.json       # Scene template with task objects
    ├── <scene>_task_<subtask>_0_0_template-meta.json  # Template metadata
    └── <scene>_task_<subtask>_instances/
        ├── <instance_id>-tro_state.json               # Initial object/robot states and target poses
        └── <instance_id>-goal_rgb.png                 # Goal RGB image (1080x1080, head camera view)
```

Inspect available instances from the command line (or pass `download --demos` to also download reference `.hdf5` trajectories and rollout videos into `ispatialgym-eval-demos/`):

```bash
python -m ispatialgym.eval.instances list --level subtask
python -m ispatialgym.eval.instances show alignment 0 1
python -m ispatialgym.eval.instances download --demos   # optional: reference HDF5 trajectories & videos
```

See [`docs/demos.md`](docs/demos.md) for visual previews of the goal images and reference demonstrations for all 13 instances.

### Preparing Your Model Output

Evaluation runs closed-loop at `30 Hz`. Implement your policy as a Python class (instantiated with `--policy-kwargs`) or a callable `policy(obs, info) -> action`, passed to the runner as `--policy module:Attribute`.

- **Goal Image**: Each task is specified solely by its goal image (`1080 × 1080` RGB from the head camera), with no language instruction. Retrieve its path via `loader.goal_image_path` in `bind()`.
- **Observations (`obs`)**:
  - `obs[robot_name][sensor]["rgb"]`: `1080 × 1080` RGB images from the head camera (`...:zed_link:Camera:0`) and wrist cameras (`...:{left,right}_realsense_link:Camera:0`).
  - `obs[robot_name]["proprio"]`: robot proprioceptive state.
- **Action (`action`)**: Return a 23-D action vector for the R1Pro robot (`ispatialgym/robot/r1pro.py`), or `None` to terminate the episode early:
  - `base` (3): base velocity `(vx, vy, wz)` in the base frame, normalized to `[-1, 1]`
  - `trunk` (4): absolute torso joint positions
  - `arm_left` (7): absolute left arm joint positions
  - `gripper_left` (1): left gripper command (`+1` open to `-1` close)
  - `arm_right` (7): absolute right arm joint positions
  - `gripper_right` (1): right gripper command (`+1` open to `-1` close)
- **Optional Hooks**:
  - `bind(executor=, loader=, meta=, out_dir=, tro_path=)`: called once before the first step.
  - `finalize(result)`: called once after the episode ends.

### Running Evaluation

Run an episode or group of instances with `ispatialgym.run_isg`:

```bash
python -m ispatialgym.run_isg \
    --config configs/eval.yaml \
    --task alignment \
    --instance-indices 0 1 \
    --policy my_package.my_module:MyPolicy
```

- `--task` accepts a `task_family`, `task`, or `subtask` name; `--instance-indices` selects instances in dataset order (alternatively, pass `--instance <path/to/tro_state.json>`).
- Command-line flags override `configs/eval.yaml`.
- Each instance runs in an isolated process with a budget of `max_steps` control steps (at 30 Hz) and `1 s` of wall-clock time per step (excluding scene loading). The episode ends upon reaching geometric success, when the policy returns `None`, or when the budget runs out.

### Understanding Evaluation Results

Each evaluated instance writes its outputs to `<out>/<instance_id>/`:

- `meta.json`: episode verdict (`status`, `geometric_success`, `symbolic_success`, `reason`), step count, per-target errors, and timings.
- `config.yaml`: resolved evaluation configuration.
- `init_rgb.png`, `final_rgb.png`: head camera frames from the first and last steps.
- `<view>_rgb.mp4`: rollout videos for any views specified in `--record-view` (`head`, `external`, `left_wrist`, `right_wrist`, `tiled`).

To aggregate metrics across all runs in a directory:

```bash
python -m ispatialgym.eval.summarize runs/ --csv runs/summary.csv
```

Metrics reported per task family and overall:

- **Geometric Success Rate** *(primary)*: the BDDL goal holds and every measured target is within its tolerance (defined in `ispatialgym/eval/tolerance_profile.py`) at the same step.
- **Symbolic Success Rate**: the BDDL goal holds.
- **Mean Position Error (m)** and **Mean Orientation Error (rad)**: mean errors of measured targets across all episodes and across symbolically successful episodes (evaluated at the geometric success step if reached, at the symbolic success step if only symbolic success is reached, otherwise at the final step).

### Example Evaluation Workflow

**1. Smoke test a single instance** with the built-in `HoldPolicy` (holds the robot still):

```bash
python -m ispatialgym.run_isg \
    --task mental_rotation \
    --policy ispatialgym.eval.policy:HoldPolicy \
    --max-steps 60 \
    --record-view head
```

**2. Evaluate a policy on all 13 instances and summarize results:**

```bash
for subtask in $(python -m ispatialgym.eval.instances list --level subtask | cut -f1); do
    python -m ispatialgym.run_isg \
        --config configs/eval.yaml \
        --task "$subtask" \
        --policy my_package.my_module:MyPolicy \
        --out runs/my_policy
done

python -m ispatialgym.eval.summarize runs/my_policy --csv runs/my_policy/summary.csv
```
