# Goal-image conditioning: final proposal

Status: proposal, 2026-09-18. Companion to `goal_image_conditioning_survey.md`,
which holds the literature and the reasoning behind every choice below.

Budget assumption: three goal-conditioned training configurations per model
(ACT, Diffusion Policy, π0.5), one seed each for selection, plus one
language-only floor per model that is a control and does not count against the
three. Backups are pre-specified contingencies with explicit triggers; they are
not run unless a trigger fires.

## 1. Picks at a glance

| Model | Pick | One line | Survey ID | Effort | Role |
| --- | --- | --- | --- | --- | --- |
| ACT | ACT-1 | Goal as extra token stream through the shared ResNet-18, learned per-source embedding | A2 | Low | Late-fusion baseline |
| ACT | ACT-2 | Early fusion: 6-channel head-camera stream (BridgeData V2 goal-conditioned ACT) | A1 | Low | Precision default, single-run bet |
| ACT | ACT-3 | ACT-1 + ACT-2 + relative-goal-pose auxiliary head + multi-view goal tokens | A3 + A4 | Low-medium | Combined bet |
| DP | DP-1 | Late fusion: separate goal encoder appended to `global_cond` (BPP Goal-Image baseline) | D1 | Very low | Late-fusion baseline |
| DP | DP-2 | Early fusion: 6-channel head encoder, separate goal vector kept (ViNT layout) | D2 | Low | Precision default |
| DP | DP-3 | DP-2 + goal masking with classifier-free guidance + aux pose head + multi-view goal vectors | D2 + D4 + D6 | Low-medium | Combined bet |
| π0.5 | PI-1 | Goal head image as a 4th SigLIP slot with a learned goal-slot bias | P1 | Low-medium | Baseline (GoalVLA/ForeAct mechanism) |
| π0.5 | PI-2 | PI-1 + VLM emits a discretised relative goal pose as the subtask text; action expert conditions on it | P1 + P3 | Medium | Forces the VLM to read the goal through its native pathway |
| π0.5 | PI-3 | PI-2 + multi-view goal slots (3 observation + 3 goal images, VISTA layout) | P1 + P3 (+ multi-view) | Medium | Combined bet |

Every pick trains with the shared recipe in Section 2. Hindsight goal sampling,
augmentation, goal rendering, and language conditioning are therefore held
constant across the three runs of a model; only the goal-fusion mechanism and
the extras listed above vary.

## 2. Shared recipe (all runs)

### 2.1 Goal rendering (TRO / dataset side)

- Render goal images from the robot's own sensors at the **settled goal state**
  (after the deterministic settling sequence, from the snapshot the evaluator
  resets to), **with the robot visible** in its final configuration.
- Record three goal views per instance: head, left wrist, right wrist. v1
  picks consume the head view; ACT-3, DP-3, PI-3 consume all three.
- Store the sensor key and calibration in `goal_observation` as already
  specified in `task_definition_and_evaluation.md`.
- Store goal images in the LeRobot dataset as per-frame image features
  (`observation.images.goal_head`, `observation.images.goal_left_wrist`,
  `observation.images.goal_right_wrist`), repeated within an episode.

### 2.2 Hindsight goal sampling

A dataset wrapper replaces the rendered goal with a future frame of the same
episode and camera:

- With probability 0.5 keep the rendered TRO goal (this is the test-time
  distribution).
- Otherwise sample a future index `g > t` from a geometric distribution biased
  toward near frames, clipped to the terminal frame; the terminal frame must
  remain reachable with non-trivial probability.
- Apply the same choice to all goal views of that sample.

Starting point only; the mixing ratio and geometric parameter are tuned on
`D_val`. Precondition: rendered goals and oracle terminal frames must be
visually indistinguishable (same camera, same robot visibility, settled state).

### 2.3 Augmentation

- Identical geometric augmentation (crop / shift / small rotation) for a goal
  view and its matching observation camera, so pixel alignment survives.
- Independent photometric augmentation (brightness, contrast, saturation) for
  goal and observation, so the policy cannot match illumination.
- No augmentation that changes object geometry.

### 2.4 Language conditioning

- **ACT and DP:** if the instruction is constant *per task*, add FiLM
  conditioning on the language embedding to the visual backbone (Octo's
  `task_film_keys` / MT-ACT style) in all runs including the floor. Use a
  fine-tuned text encoder or a learned 7-way task embedding; do not rely on a
  frozen encoder to separate seven similar sentences. If the instruction is
  constant across *all* tasks, omit FiLM entirely.
- **π0.5:** keep the text prompt. Use the task instruction followed by a fixed
  clause naming the goal image's role, e.g. "Reach the configuration shown in
  the goal image."

### 2.5 Controls

- One **language-only floor** per model (no goal input, otherwise identical
  recipe). With FiLM this is an MT-ACT / LCBC-style multi-task baseline.
- Optimiser, schedule, chunk size, resolution and observation history follow
  the existing BEHAVIOR baselines for the same model unless a pick says
  otherwise.

## 3. Diagnostics and decision rules

### 3.1 Goal-swap sensitivity test

Run on every checkpoint, on `D_val` instances, before looking at success rates.

For a fixed initial state with true goal `G_a`, roll out with:

- (a) `G_a` (correct goal),
- (b) `G_b`, the goal image of another instance of the same task whose target
  is reachable from the same initial state,
- (c) a null goal (zeros, or the learned null vector for DP-3).

Report:

- `success(a)`, `success(b)` judged against the goal that was *given*, and
  `success(c)`.
- **Goal-following rate**: fraction of (b) rollouts whose final state is closer
  (in the task's pose metric) to `G_b`'s target than to `G_a`'s target.
- Pose error to the given goal for (a) and (b).

A goal-reading policy has `success(a) ≈ success(b) ≫ success(c)` and a
goal-following rate well above chance. Identical numbers across (a), (b), (c)
mean the goal is ignored (the ViNT failure).

### 3.2 Selection metrics

Per model, rank picks by settled success on `D_val`, then by max pose error
across goals, with the goal-following rate as a gate: a pick that does not read
the goal is not eligible regardless of success.

### 3.3 Triggers for backups

| Trigger | Symptom | Response |
| --- | --- | --- |
| T1: goal ignored | Goal-following rate near chance and `success(a) ≈ success(c)` | ACT: ACT-B1, then ACT-B2. DP: raise guidance `w`, raise hindsight ratio, then DP-B1. π0.5: switch LoRA to full fine-tune and unfreeze the vision encoder, then PI-B1. |
| T2: goal read but imprecise | High goal-following rate, pose errors above tolerance, ordinary success low | Increase aux-pose loss weight, bias hindsight sampling toward near frames, enable multi-view goals if not already on; then ACT-B3 / DP-B3 / PI-B2 (resolution and geometry backups). |
| T3: reaches but does not settle | Ordinary success high, settled success low | Oversample terminal frames as goals, add a "goal reached / hold" flag head, revisit chunk size and temporal ensembling. Not a fusion problem. |
| T4: engineering block | π0.5 code path rejects extra image slots or lacks a text-output loss | PI-B3 (tiling) as a stopgap; do PI-2's text target in the openpi b1k fork, which implements token loss masks. |

## 4. ACT

Code facts (`lerobot/policies/act/modeling_act.py`, LeRobot 0.5.2 in the
`ispatialgym` env): one shared ResNet-18 backbone over `config.image_features`;
per-camera tokens use a shared 2D sinusoidal position embedding with **no camera
identity**; encoder input is `[latent z, robot_state, camera tokens…]`; the cVAE
encoder sees the ground-truth action chunk during training and `z` is zero at
inference.

### ACT-1: goal token stream (late fusion)

- Add `observation.images.goal_head` to `image_features`; it passes through the
  shared backbone like a camera.
- Add `nn.Embedding(n_sources, dim_model)` and add the source vector (head /
  left wrist / right wrist / goal) to every token of that stream. Without it the
  encoder cannot distinguish goal tokens from current-view tokens.
- Tests whether the transformer can perform the goal/observation comparison on
  its own. Expected: works on seen-like goals, at risk of T1.

### ACT-2: early fusion (precision default)

- A second ResNet-18 for the head stream with `conv1` widened to 6 input
  channels: pretrained weights copied into channels 0–2, channels 3–5
  zero-initialised. Input is `concat(head_rgb, goal_head_rgb)`. Wrist cameras
  keep the shared 3-channel backbone.
- Tokens from the fused stream replace the plain head-camera tokens.
- Follows BridgeData V2 App. B.3. Best inductive bias for Camera Relocalization
  and for pixel-level pose precision. Single-run bet for ACT.

### ACT-3: combined bet

- ACT-2's fused head stream, plus ACT-1's goal token streams for the two wrist
  goal views (with source embeddings), so all three goal views are present.
- Relative-goal-pose auxiliary head: an MLP on the encoder output at the latent
  token position predicting the goal-pose delta from privileged TRO targets
  (base pose for CR; object pose for OS, MR, WA, IORS; assigned slot poses for
  CM block design; moving-link pose for articulated opening). Normalised
  targets, loss weight ~0.1 as a starting point.
- Watch the KL term: if goal information leaks through `z`, apply ACT-B1.

### ACT backups

| ID | Backup | Trigger | Change |
| --- | --- | --- | --- |
| ACT-B1 | VAE off, or goal-conditioned VAE encoder | T1 with low KL / evidence that `z` carries task information | `use_vae=False`, or add a pooled goal embedding token to the VAE encoder input so `z` only needs to encode residual style |
| ACT-B2 | FiLM the pooled goal embedding into the backbone alongside language | T1 on ACT-1 | Nearly free once FiLM exists; loses spatial layout, so it is a diagnostic, not a precision fix |
| ACT-B3 | Higher input resolution or goal-centred crop | T2 | Raise head/goal resolution from the baseline setting; crop around the task object using privileged masks at train time only if a test-time crop rule exists |
| ACT-B4 | Chunk size and ensembling sweep | T3 | BridgeData V2 reduced chunk size from 100 to 5 at 5 Hz; tune for R1Pro control rate |

## 5. Diffusion Policy

Code facts (`lerobot/policies/diffusion/modeling_diffusion.py`): per-camera
ResNet-18 + SpatialSoftmax `DiffusionRgbEncoder` (shared, or per-camera with
`use_separate_rgb_encoder_per_camera=True`); image features and state are
concatenated over `n_obs_steps` into `global_cond`, which FiLM-conditions the 1D
U-Net (`DiffusionConditionalUnet1d`, `cond_dim = step_embed + global_cond`).

### DP-1: late fusion (BPP Goal-Image baseline)

- A dedicated `DiffusionRgbEncoder` for `goal_head`; encode once per sample
  (not per observation step) and append to `global_cond`
  (`global_cond_dim += goal_feature_dim` in `DiffusionModel.__init__`,
  concatenate after the flatten in `_prepare_global_conditioning`).
- Because `global_cond` enters through FiLM, this is "FiLM on [language, goal
  embedding]". Reference point; expected to be goal-aware but imprecise.

### DP-2: early fusion (precision default)

- Head-camera encoder with a 6-channel `conv1` over `concat(head_rgb,
  goal_head_rgb)`, wrist encoders unchanged; keep DP-1's separate goal vector as
  well (ViNT layout: relative features in the fused stream, global goal context
  in the vector).

### DP-3: combined bet

- DP-2 plus:
- **Goal masking + classifier-free guidance.** During training replace the goal
  inputs (fused channels and goal vectors) with a learned null with probability
  ~0.2; language is never dropped. At inference,
  `ε = ε_null + w (ε_goal − ε_null)`, sweep `w ∈ {1, 1.5, 2, 3}` on `D_val`. The
  `w = 0` branch is the task-average policy used for control (c) in Section 3.1.
- **Aux pose head** on `global_cond`, same targets and weight as ACT-3.
- **Multi-view goal vectors** for left/right wrist goal views via separate
  encoders appended to `global_cond`.

### DP backups

| ID | Backup | Trigger | Change |
| --- | --- | --- | --- |
| DP-B1 | MDT-style transformer denoiser with a joint goal/observation encoder | T1 or T2 persisting on DP-3 | New model code (original `diffusion_policy` DP-T or MDT repo); MDT's ablation says the joint encoder is what matters |
| DP-B2 | Near-horizon curriculum | T2 | Cap hindsight horizon (`k_max`, SuSIE style) early in training, anneal toward the terminal goal; if still imprecise, consider subgoal generation (SuSIE / Envision) as a later project |
| DP-B3 | Higher resolution or goal-centred crop | T2 | As ACT-B3 |
| DP-B4 | Frozen pretrained goal encoder (CLIP / DINOv2) for the goal vector | T1 on DP-1 only | Low expected value (MDT found CLIP unnecessary); cheap to try |

## 6. π0.5

Code facts (`lerobot/policies/pi05/modeling_pi05.py`, `configuration_pi05.py`):
`prepare_images` iterates the image keys present in `config.image_features`,
resizes to `image_resolution`, normalises to [-1, 1], and pads missing cameras
(`empty_cameras`) with -1 images and zero masks; `embed_prefix` runs SigLIP per
image and appends tokens with bidirectional attention, followed by text tokens
and the action-expert suffix; position ids are cumulative over pad masks, so an
extra image lengthens the prefix by 256 tokens. The openpi JAX/PyTorch paths
hard-code `IMAGE_KEYS` and must be edited (`model.py`,
`preprocessing_pytorch.py`, and the BEHAVIOR fork's
`src/openpi/configs/robots/b1k.py`).

### PI-1: goal head image as a 4th image slot

- Add `observation.images.goal_head` as an image feature so it becomes a fourth
  SigLIP image in the prefix.
- Add a learned per-slot bias `nn.Parameter(zeros(hidden))` to the goal-image
  tokens (zero-initialised so pretrained behaviour is unchanged at step 0).
  PaliGemma has no camera identity beyond sequence position.
- Prompt per Section 2.4.
- Full fine-tune preferred; LoRA on the VLM acceptable. `train_expert_only` is
  excluded because the VLM must learn to compare images.
- Starting point: the b1k π0.5 fine-tuning schedule, 32-step action horizon.

### PI-2: PI-1 + relative-goal-pose text (hierarchical)

- Discretise the relative goal pose (task-specific target as in ACT-3) into
  text tokens, e.g. 256 bins per dimension, mirroring how π0.5 already writes
  discretised state into the prompt.
- Training: VLM text loss on the pose tokens given (observations, goal image,
  instruction); action expert conditioned on the prefix that includes the pose
  tokens, mirroring π0.5's subtask factorisation
  `π(a | o, ℓ̂) π(ℓ̂ | o, ℓ)`.
- Inference: run the high-level text prediction first, append it to the prompt,
  then sample actions, exactly as π0.5 does with subtasks.
- Also log the predicted pose against the TRO ground truth; it is an
  interpretable measure of whether the VLM reads the goal.
- Engineering check: confirm the LeRobot π0.5 port trains text outputs
  (`token_loss_mask`); if not, run PI-2 in the openpi b1k fork (T4).

### PI-3: PI-2 + multi-view goal slots

- Three observation images plus three goal images (head, left wrist, right
  wrist), VISTA's exact layout; prefix grows to ~1536 image tokens.
- Add the goal-slot bias to all three goal streams (separate biases per slot).
- Apply VISTA's temporal-overlap idea to hindsight sampling: sample goals from a
  window around the terminal frame so small early/late goal misalignments do not
  destabilise the final approach.

### π0.5 backups

| ID | Backup | Trigger | Change |
| --- | --- | --- | --- |
| PI-B1 | Goal-image pretraining stage | T1 on PI-1 or PI-2 | Before iSpatialGym fine-tuning, fine-tune π0.5 with PI-1's architecture on hindsight goal images built from the BEHAVIOR challenge demonstrations (20k R1Pro episodes, head + wrist RGB; any later frame is a free goal). This mirrors VISTA's goal-annotated pretraining stage, which our picks otherwise lack. |
| PI-B2 | Reloc3r relative-pose front-end for Camera Relocalization | T2 concentrated on CR | Frozen two-view relative pose regressor on (current head, goal head) fed as extra state tokens, or used as PI-2's text target |
| PI-B3 | Pixel tiling of current + goal head view into one slot | T4 | No model change; halves resolution, so a stopgap and sanity baseline only |
| PI-B4 | Full fine-tune with vision encoder unfrozen, longer schedule | T1 with LoRA | BPP's π0.5 comparison used 100k LoRA steps on LIBERO-Gen; budget similar |

## 7. Run order and budget

| Phase | Runs | Purpose |
| --- | --- | --- |
| 0 | Data: goal renders (3 views), per-frame goal features, hindsight sampler, augmentation pipeline; the three language-only floors | Pipeline validation; the floor numbers |
| 1 | ACT-2, DP-2, PI-1 | Most likely to produce working policies and to expose data bugs; first goal-swap tests |
| 2 | ACT-3, DP-3, PI-2 | Combined bets; guidance sweep for DP-3 |
| 3 | ACT-1, DP-1, PI-3 | Late-fusion baselines needed for the paper; π0.5 multi-view |
| 4 | Two additional seeds for the selected pick per model | Numbers for Tables 1A/1B; `D_test_main_public` stays untouched until the end |
| B | Backups | Only when a trigger in Section 3.3 fires |

Total planned: 3 floors + 9 picks + 6 seed repeats = 18 training runs before
backups. The goal-swap test is evaluation only and runs on every checkpoint.

## 8. Deliberately not in the plan

- MDT-style transformer denoiser as a primary pick (new model code; kept as
  DP-B1).
- Pixel tiling for π0.5 as a primary pick (resolution loss; kept as PI-B3).
- Reloc3r / geometric front-ends as primary picks (task-specific; kept as
  PI-B2).
- Robot-hidden goal rendering (kept as a later ablation; the rendered goal must
  match oracle terminal frames for hindsight sampling to be valid).
- Subgoal generation (SuSIE / Envision / Act2Goal style world models). Heavy;
  revisit only if T2 persists after DP-B2.
- Separate language-vs-goal ablations beyond the floor. With three runs per
  model the floor already measures what language alone gives.

## 9. Prerequisite checklist

- [ ] TRO stores head + wrist goal renders at the settled goal state, robot
      visible, with sensor keys and calibration.
- [ ] Rendered goals verified visually indistinguishable from oracle terminal
      frames on a sample of episodes.
- [ ] LeRobot dataset exposes `observation.images.goal_*` per frame.
- [ ] Hindsight sampler wrapper with configurable mixing ratio and geometric
      parameter; unit test that the terminal frame is sampled.
- [ ] Augmentation pipeline with shared geometric / independent photometric
      transforms across goal and matching camera.
- [ ] Privileged relative-goal-pose targets exported per frame for the aux
      heads and PI-2 text targets.
- [ ] Goal-swap evaluation script over `D_val` producing the Section 3.1
      metrics.
- [ ] Decision on language: per-task instruction (FiLM on) or global
      instruction (FiLM off).
- [ ] π0.5 code path confirmed to accept 4 and 6 image features and to train
      text outputs; otherwise openpi b1k fork prepared.
