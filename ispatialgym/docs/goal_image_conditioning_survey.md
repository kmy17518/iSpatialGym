# Goal-image conditioning for ACT, Diffusion Policy, and π0.5

Status: literature survey and design recommendations, 2026-09-05

This document answers three questions for iSpatialGym:

1. How has prior work injected goal images into ACT-style, Diffusion-Policy-style,
   and VLA-style (π0/π0.5) policies?
2. What did those works learn that transfers to our setting?
3. Which concrete variants should we implement for each of our three
   architectures, and in what order?

## 1. Why our setting is unusual

In iSpatialGym the language instruction is identical across instances of a task.
The goal image is the *only* input that tells the policy which exact end state is
required, and success is judged against numeric pose tolerances
(`pose_reached`, `layout_reached`, `attached`, `open`), not against "did the
right kind of thing happen". Two facts from the literature matter here:

- Most goal-image work uses the goal image to identify *which task* to do.
  RoboCat states this explicitly: "the goal image is an example of the task being
  solved and it does not indicate a specific state that the agent should reach."
  BPP, RT-Sketch, MDT, and GR-MG are in the same regime (LIBERO / CALVIN-style
  tasks where success is a predicate, not a pose).
- A smaller line of work uses the goal image as a *precise target state*:
  GCBC/D-GCBC/SuSIE (reach a future frame), ViNT/NoMaD (drive until the camera
  sees the goal view), BESO on CALVIN (goal images that did not even contain the
  end-effector), Act2Goal, and the recent goal-conditioned VLAs (VISTA/GoalVLA)
  which report that the robot "posture largely aligns with that shown in the goal
  image".

We are squarely in the second regime, and Camera Relocalization is literally
image-goal navigation/visual servoing. The design lessons from the second group
(relative goal features, same-camera goals, hindsight goal sampling, goal
sensitivity checks) are therefore the ones to prioritise.

## 2. Survey: how goal images have been injected

### 2.1 Early fusion: stack goal and current image channel-wise

| Work | Base policy | Fusion | Goal source / sampling | Notes |
| --- | --- | --- | --- | --- |
| GCBC and D-GCBC, [BridgeData V2](https://arxiv.org/abs/2308.12952) (Walke et al., CoRL 2023) | MLP BC / DDPM BC | Observation and goal stacked channel-wise, ResNet-34 | Hindsight future frame of the same over-the-shoulder camera | Random crop/resize/color jitter applied to both. |
| **Goal-conditioned ACT**, BridgeData V2 App. B.3 | ACT (cVAE + transformer) | "The observation and goal are stacked channel-wise before being passed into a ResNet-18 image encoder." | Same as above | Only published goal-image ACT we found. Chunk size reduced from 100 to 5 for 5 Hz control. Same ACT hyper-parameters otherwise. |
| [SuSIE](https://arxiv.org/abs/2310.10639) low-level policy (Black et al., ICLR 2024) | Diffusion MLP head | Observation and goal stacked channel-wise, ResNet-50 | Future frame `k` steps ahead, `k ≤ k_max` | "Oracle GCBC" (same policy, goal horizon to end of trajectory, ground-truth goal image) was *less precise* than the same policy fed generated near-term subgoals. |
| [Octo](https://arxiv.org/abs/2405.12213) (2024) | Transformer, diffusion action head | `task_stack_keys`: goal image concatenated channel-wise with the observation before the conv patch encoder ("early goal fusion") | Hindsight goal from the same camera; zero-padded when absent | Authors' rationale: lets the network do "pixel-wise comparisons between the camera view at the current timestep and the desired goal camera view" ([issue #25](https://github.com/octo-models/octo/issues/25)). |
| [RT-Sketch](https://arxiv.org/abs/2403.02709) / RT-Goal-Image (Sundaresan et al., 2024) | RT-1 (EfficientNet-B3 tokens, transformer, discretised actions) | Goal image (or sketch) concatenated with the image history at the EfficientNet input; FiLM language removed | Last frame of the episode (hindsight) | Motivates sketches because "images can be over-specified": goal images carry distractors and lighting that the policy may try to match. |
| [ViNT](https://arxiv.org/abs/2306.14846) (2023) | Transformer, regression | Separate *goal fusion encoder*: (current, goal) stacked to 6 channels through a second EfficientNet-B0 to produce one goal token; observation history tokens from a separate encoder | Hindsight goal, plus a temporal-distance-to-goal head | Key negative result: "naively extracting features from the goal image ... led to poor performance, often ignoring the goal entirely." They hypothesise good goal features are *relative* (difference between current and goal). |
| [NoMaD](https://arxiv.org/abs/2310.07896) (2023) | ViNT encoder + diffusion action head | Same goal fusion token, plus a binary goal mask sampled Bernoulli(0.5) during training | Hindsight goal | One network serves goal-conditioned and goal-free behaviour; the mask is the same trick as classifier-free guidance dropout. |

### 2.2 Late fusion: encode the goal separately, inject as a token or vector

| Work | Base policy | Fusion | Goal source / sampling | Notes |
| --- | --- | --- | --- | --- |
| **Goal-Image baseline in [Behavior Prompting Policy](https://arxiv.org/abs/2606.30457)** (Patel, Pekarek, Castro Hernandez, Song, 2026) | Diffusion Policy CNN U-Net (151M) | Goal image passed through the *same fine-tuned CLIP vision encoder* as the 3rd-person camera; embedding replaces the prompt-encoder output in the U-Net's global conditioning | "The goal image is the final frame of a demonstration from the 3rd person camera" | Worked on training tasks, degenerated on unseen tasks; ambiguous for chained tasks (no ordering); on real drawing it produced "a recognizable rendition ... but with spatial misalignment or local geometric errors". Evaluation used matched pairs: same reset, same demo used for prompt vs. goal image. Code: [real-stanford/behavior_prompting](https://github.com/real-stanford/behavior_prompting) ("support for language conditioning, goal image conditioning, and in-context behavior prompting"). |
| [MDT / MDT-V](https://arxiv.org/abs/2407.05996) (Reuss et al., RSS 2024) | Transformer encoder-decoder diffusion policy | Frozen CLIP ViT-B/16 embeds the goal image into one token; encoder self-attends over goal + observation tokens; decoder denoises actions with cross-attention to the encoder | Geometric goal-window sampling (p = 0.1, window 49) over future frames; LIBERO goals = final state | Auxiliary losses: Masked Generative Foresight (predict masked future patches) and Contrastive Latent Alignment (image goal vs language goal). Ablations: a ResNet goal encoder trained from scratch was nearly as good as CLIP; removing the transformer encoder (feeding conditions straight into the denoiser as in vanilla DP-T) collapsed CALVIN avg. length from 4.18 to 1.41. |
| [BESO / CFG-BESO](https://arxiv.org/abs/2304.02532) (Reuss et al., RSS 2023) | Score-based diffusion transformer | Goal image embedding as a conditioning token | Future-frame goals from play data | Classifier-free guidance on the goal; evaluated with goal images "that did not contain the end-effector in the correct position", i.e. the policy must reason about scene changes, not copy the arm. |
| [GR-MG](https://arxiv.org/abs/2408.14368) (2024) | GPT-style transformer with **cVAE action-trajectory head** (ACT-like) | MAE tokens of the goal image *prepended* to the input sequence; all later tokens attend to them | Goal images sampled from trajectories; at test time generated by InstructPix2Pix | Also predicts task progress (`[PROG]` token) and future frames. Closest published precedent for "ACT + goal-image tokens". |
| [C-BeT](https://arxiv.org/abs/2210.10047), [VQ-BeT](https://arxiv.org/abs/2403.03181) | Behaviour transformers | Future observation(s) concatenated to the observation sequence as conditioning | Random future frame from the same episode | Simple, effective in play-data settings. |
| [RoboCat](https://arxiv.org/abs/2306.11706) (2023) | Gato-style autoregressive transformer | Goal image tokenised by a frozen VQ-GAN (8×8 tokens) and placed in the sequence | Hindsight goals; every frame is a valid goal for all earlier steps | Predicting future image tokens alongside actions improved performance. |
| Play-LMP / MCIL / HULC lineage ([1903.01973](https://arxiv.org/abs/1903.01973), [2005.07648](https://arxiv.org/abs/2005.07648), [2204.06252](https://arxiv.org/abs/2204.06252)) | Latent-plan policies | Goal image encoded into a latent goal space shared with language | Play data with hindsight goals | Historical baseline for CALVIN; superseded by BESO/MDT. |

### 2.3 VLA-style: goal image as extra image tokens in the VLM prefix

| Work | Base | Fusion | Notes |
| --- | --- | --- | --- |
| **GoalVLA** in [VISTA](https://arxiv.org/abs/2602.10983) (Feb 2026) | π0 architecture: PaliGemma-3B + 0.3B action expert, flow matching | "In each inference step, the model takes six images as input, including three current observation images and three goal images, which are encoded into tokens via SigLIP." Current-observation tokens are concatenated with goal-image tokens in the prefix. | Trained in two stages: 200k AgiBot-Beta trajectories with subtask text + goal images, then 737 in-domain trajectories. Real-world OOD success 69% vs 14% for language-only π0 (the comparison includes the world-model planner). Temporal-overlap augmentation near subtask boundaries so the policy tolerates goal images that are a few frames early/late. Reported failure modes: the policy "overly focuses on matching the robot pose depicted in the goal image"; spatially misaligned wrist goal images caused collisions; goals outside the training spatial distribution were not followed. |
| [ForeAct](https://openaccess.thecvf.com/content/CVPR2026/papers/Zhang_ForeAct_Steering_Your_VLA_with_Efficient_Visual_Foresight_Planning_CVPR_2026_paper.pdf) (CVPR 2026) | VLA | Predicted future head-camera observation fed as an *additional visual input* alongside subtask text | Generates foresight only for the head camera; same mechanism works with a given goal image. |
| [ManualVLA](https://arxiv.org/abs/2512.02013) (Dec 2025) | VLA | Overlays the target object's final position as a mask on the current image (visual prompt) and conditions the action expert on subgoal-image features through a cross-task attention mask | Shows a second injection route: paint the goal onto the current view. |
| [WorldScape Policy 2.0](https://arxiv.org/abs/2607.18840) (Jul 2026) | Video-action DiT + VLM memory | Goal image encoded as a clean VAE latent kept in the causal visual prefill so future video/action chunks attend to it | Distinguishes "first-view" goals (terminal frame from the robot's own camera, "an unambiguous visual target") from "third-view" goals (from a human demo, needs viewpoint abstraction). |
| [CoT-VLA](https://arxiv.org/abs/2503.22020) (2025) | 7B VLA | Predicts a subgoal image `n` frames ahead as visual chain-of-thought, then actions | The reverse direction (generate rather than consume), but it confirms VLM backbones handle image-goal tokens in-sequence. |
| [Act2Goal](https://arxiv.org/abs/2512.23541) (RSS 2026) | Genie-Envisioner-style video world model + action expert | Goal visual condition concatenated with the current observation along the hidden-state sequence; language removed; action expert cross-attends layer-wise to world-model features | Multi-Scale Temporal Hashing; HER relabelling for online improvement. |
| [Envision](https://arxiv.org/abs/2512.22626) (Dec 2025) | Video diffusion | First-and-last-frame conditioned video model interpolates from current observation to the goal image | Visual planner, not a policy; relevant if we later want subgoal generation. |
| [Goal-VLA](https://arxiv.org/abs/2506.23919) (2025) | VLM + registration | Goal image → target object pose via feature matching / point-cloud registration → low-level controller | Object-centric alternative: extract the pose target explicitly. |
| [Track2Act](https://arxiv.org/abs/2405.01527) (2024) | DiT track predictor + residual policy | (initial image, goal image) → point tracks → rigid transforms → EE poses, then residual corrections | Goal-image-driven intermediate representation. |

π0 and π0.5 have no native goal-image input. The open-source code hard-codes
`IMAGE_KEYS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")`
([openpi `model.py`](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/models/model.py));
a maintainer noted in [openpi #620](https://github.com/Physical-Intelligence/openpi/issues/620)
that π0.5 was pretrained with up to four cameras, so a fourth SigLIP image is not
foreign to the backbone even though the released preprocessing rejects it. BPP
compared against a LoRA-finetuned π0.5 driven by *language*, not by goal images.

### 2.4 Lessons that transfer to iSpatialGym

1. **Goal features must be relative, or the goal gets ignored.** ViNT's separate
   goal encoder was ignored by the policy; stacking (current, goal) fixed it.
   With constant language and similar initial scenes, "ignore the goal and emit
   the average behaviour" is our single most likely failure mode. Section 4.6
   gives a diagnostic.
2. **Same-camera goals enable both early fusion and hindsight sampling.** Every
   method in 2.1 and most in 2.2 use the goal from the same camera as the
   observation and treat any later frame as a valid goal. That multiplies
   (observation, goal) pairs by the episode length and teaches the policy to
   *reach* pictured states rather than *recognise* tasks. It only works if our
   rendered goal image is statistically indistinguishable from a real terminal
   observation.
3. **Goal images are over-specified.** RT-Sketch and VISTA both observed the
   policy latching onto irrelevant content (distractors, exact robot posture).
   In our benchmark the background/lighting interventions already re-render the
   goal image; augmentation should decorrelate photometry between goal and
   observation while preserving geometry.
4. **Precision comes from near goals; far goals lose it.** SuSIE's Oracle GCBC
   (ground-truth final goal) was less precise than following generated near-term
   subgoals. Hindsight sampling with a bias toward short horizons (MDT geometric
   sampling, SuSIE `k_max`) is the cheap version of this; subgoal generation is
   the expensive version.
5. **Auxiliary heads ground the goal.** Temporal distance to goal (ViNT/NoMaD),
   progress (GR-MG), masked future prediction (MDT), future image tokens
   (RoboCat), image–language alignment (MDT, [GRIF](https://arxiv.org/abs/2307.00117)).
   We have something none of them had: privileged numeric goal poses in the TRO,
   which can supervise a relative-goal-pose head directly.
6. **Goal masking / classifier-free guidance amplifies goal adherence.** NoMaD and
   CFG-BESO train with the goal dropped some fraction of the time and can then
   extrapolate toward the goal at test time (`w > 1`). Because our unconditional
   policy is the task-average behaviour, guidance has a clear meaning.
7. **A joint encoder over goal and observation tokens matters more than a fancy
   goal encoder.** MDT's ablation: CLIP vs scratch ResNet for the goal was nearly
   a wash; removing the encoder that lets goal and observation tokens interact was
   catastrophic.
8. **Goal images do not encode ordering.** BPP's chained tasks were ambiguous under
   goal images. Our tasks are single-stage except block design, where any order
   is acceptable, so this is not a blocker, but Configuration Matching rollouts
   should be checked for dithering between blocks.

## 3. Decisions that precede the architecture

These are dataset/TRO decisions; every architecture below assumes them.

- **Goal camera.** Render the goal image from the robot's own sensors at the goal
  state, and store the sensor key and calibration in the TRO (`goal_observation`
  already has these fields). For Camera Relocalization this is forced (the goal
  *is* the head-camera view at the target pose). For the other tasks it enables
  early fusion and hindsight sampling. A fixed external camera would force late
  fusion and remove hindsight sampling; a *multi-view* goal (head + both wrists at
  the goal state, as VISTA used) is a strict superset and should be recorded even
  if v1 policies only consume the head view.
- **Robot visibility in the goal image.** Render the goal from the settled goal
  state *with* the robot in its final configuration. Reasons: it matches settled
  success; it is exactly what oracle terminal frames look like, so hindsight
  frames and rendered goals share one distribution; Object Scaling and Mental
  Rotation have the object in hand, so the arm must be there anyway. Keep a
  robot-hidden rendering as an ablation to test the VISTA "copies the arm pose"
  failure, and note that BESO showed goal-conditioned diffusion can work without
  the arm in the goal.
- **Goal state = settled state.** Render after the deterministic settling
  sequence, from the same snapshot the evaluator resets to, so the goal image is
  reachable and stable.
- **Augmentation.** Apply *identical* geometric augmentation (crop/shift) to the
  goal and the matching observation camera so pixel alignment survives, and
  *independent* photometric augmentation (brightness/contrast/saturation) so the
  policy cannot solve tasks by matching illumination. This is also the training
  side of the lighting-shift analysis.
- **Dataset layout.** Store the goal image(s) as per-frame features
  (`observation.images.goal_head`, repeated within an episode) so all three
  LeRobot policies see them as ordinary image features; hindsight sampling then
  needs a small dataset wrapper that replaces the goal feature with a future
  frame of the same episode with some probability.

## 4. Recommended variants per architecture

Effort and risk are relative to the existing LeRobot baselines installed in the
`ispatialgym` env (`lerobot` 0.5.2: `policies/act`, `policies/diffusion`,
`policies/pi05`).

### 4.1 ACT

Code facts (`lerobot/policies/act/modeling_act.py`): one shared ResNet-18
backbone is applied to every entry of `config.image_features`; each camera's
feature map becomes `h*w` tokens with a 2D sinusoidal position embedding that is
**identical across cameras**, so the transformer encoder receives no signal about
which camera a token came from. Tokens are `[latent z, robot_state, cam tokens…]`.

| ID | Variant | Precedent | How | Effort / risk |
| --- | --- | --- | --- | --- |
| A1 | Early fusion on the head camera | BridgeData V2 ACT (App. B.3), Octo | Concatenate goal and head image to 6 channels; widen `backbone.conv1` to 6 input channels (copy pretrained weights into channels 0–2, zero-init 3–5). Wrist cameras unchanged. | Low. Best inductive bias for CR and for pixel-level pose precision. Only the head view gets goal information. |
| A2 | Goal as extra "camera" with a source embedding | GR-MG (goal tokens prepended), MDT | Add the goal key to `image_features`; add a learned per-source embedding (head / left wrist / right wrist / goal) to the camera tokens so the encoder can distinguish goal tokens from current-view tokens. Without this, A2 degenerates to "ACT with a confusing fourth camera". | Low. Naturally extends to multi-view goals. Risk: ViNT-style goal ignoring, so pair with the 4.6 diagnostic. |
| A3 | A1 + A2 | ViNT (goal fusion token *and* observation tokens) | Both routes; the fusion channel supplies relative features, the tokens supply global context. | Low-medium. Recommended default after A1/A2 are individually checked. |
| A4 | Relative-goal-pose auxiliary head | ViNT/NoMaD distance head; our TRO | Read out a token (e.g. the latent-input position) through an MLP to predict the goal pose delta from privileged TRO targets: base pose delta for CR, object pose delta for OS/MR/WA/IORS, per-slot deltas for CM. Add as a small weighted loss. | Low. Directly supervises "where is the pictured state relative to now". |

ACT-specific caution: the cVAE encoder sees the ground-truth action chunk at
training time and `z` is set to the prior mean at inference. If goal features
are weak, the model can route goal information through `z` (the action chunk is a
function of the goal) and never learn to read the goal image, which only shows
up at test time. Monitor the KL term and the goal-sensitivity metric; if it
happens, either condition the VAE encoder on the goal features too, lower the
prior's role (`use_vae=False`), or increase `kl_weight`.

### 4.2 Diffusion Policy

Code facts (`lerobot/policies/diffusion/modeling_diffusion.py`): each camera goes
through a ResNet-18 + SpatialSoftmax `DiffusionRgbEncoder` (shared, or per-camera
with `use_separate_rgb_encoder_per_camera=True`); features and state are
concatenated over `n_obs_steps` into a single `global_cond` vector that FiLM-
conditions the 1D U-Net. There is no token-level conditioning path in this
implementation.

| ID | Variant | Precedent | How | Effort / risk |
| --- | --- | --- | --- | --- |
| D1 | Late fusion into `global_cond` | BPP Goal-Image baseline; BridgeData D-GCBC (late variant) | Encode the goal image with its own `DiffusionRgbEncoder` (or the shared one) and append its feature vector to `global_cond`. Since the goal is static, encode it once rather than per obs step. | Very low. This is the exact baseline BPP beat; expect "right kind of motion, imprecise placement". Necessary as a reference point. |
| D2 | Early fusion on the head camera | BridgeData D-GCBC, SuSIE, Octo | 6-channel head encoder as in A1, wrist encoders unchanged; optionally keep D1's separate goal vector as well (ViNT layout). | Low. Recommended default. |
| D3 | Token-level conditioning (transformer DP or MDT) | MDT/MDT-V; DP-T from Chi et al. | Use a transformer denoiser whose cross-attention consumes goal tokens + observation tokens through a small joint encoder (MDT's key ablation says the joint encoder is what matters). LeRobot has no DP-T; the original `diffusion_policy` repo and the MDT code do. | Medium (new model code). Strongest reported goal-image DP; consider after D2 plateaus. |
| D4 | Goal masking + classifier-free guidance | NoMaD, CFG-BESO | Drop the goal features with probability ~0.1–0.5 during training (replace with a learned null vector). At inference denoise with `ε = ε_uncond + w (ε_goal − ε_uncond)`, sweep `w ∈ [1, 3]`. | Low. Gives a test-time knob for goal adherence and a free "task-average" policy for the goal-sensitivity diagnostic. |
| D5 | Hindsight goal sampling | GCBC, SuSIE, MDT, RoboCat | With probability p replace the rendered goal with a future frame of the same episode (geometric or uniform over `[t+1, T]`, biased toward near frames), with the terminal frame always included. Requires decision "same camera + robot visible" in Section 3. | Low-medium (dataset wrapper). Largest expected gain in precision; also regularises against memorising per-instance goals. |
| D6 | Auxiliary heads | MDT MGF, ViNT distance, our TRO | Relative-goal-pose head as A4; optional masked-future-patch prediction from the `global_cond` features. | Low. |

### 4.3 π0.5

Code facts (`lerobot/policies/pi05/modeling_pi05.py`): `prepare_images` iterates
over the image keys present in `config.image_features`, resizes to
`image_resolution`, normalises to [-1, 1], and pads missing cameras with -1 images
and zero masks (`empty_cameras`); `embed_prefix` runs SigLIP per image and
appends the tokens with bidirectional (`att_mask = 0`) attention, then text
tokens, then the action-expert suffix. Position ids are cumulative over pad
masks, so an extra image simply lengthens the prefix (+256 tokens per 224×224
image). The openpi JAX/PyTorch code paths hard-code the three keys and must be
edited in `model.py` / `preprocessing_pytorch.py` and the robot input transform
(for the BEHAVIOR fork, `src/openpi/configs/robots/b1k.py`).

| ID | Variant | Precedent | How | Effort / risk |
| --- | --- | --- | --- | --- |
| P1 | Goal image(s) as additional SigLIP image slots | VISTA/GoalVLA (3 obs + 3 goal images), ForeAct | Add `goal_head` (and optionally `goal_left_wrist`, `goal_right_wrist`) as image features. Add a learned per-slot bias vector to goal-image tokens (PaliGemma has no camera identity beyond sequence position) so the backbone can tell goal tokens from current-view tokens. Keep the constant task prompt but state the role, e.g. "reach the configuration shown in the goal image". | Low-medium. Prefix grows from 768 to 1024–1536 image tokens; fine on 80 GB. The pretrained model has never seen "goal" semantics, so expect it to be learned entirely during fine-tuning; VISTA needed 200k goal-annotated trajectories before its 737-episode fine-tune. Full fine-tune or LoRA; `train_expert_only` is likely too weak because the VLM must learn to compare images. |
| P2 | Pixel-space composition (no architecture change) | ManualVLA visual prompt | Tile current head view and goal view into one 224×224 slot, or paint the goal into the current view. | Very low, but halves resolution for a precision benchmark. Use only as a sanity baseline. |
| P3 | Hierarchical: goal image → textual subgoal → actions | π0.5's own high-level/low-level factorisation; Goal-VLA (pose from goal image) | Train the VLM head to emit a discretised relative goal pose string (e.g. "base dx +0.42 dy -0.10 yaw -18") from (observation, goal image), supervised from TRO targets; the action expert conditions on that text (plus P1's tokens). At test time run high-level inference first, exactly as π0.5 does with subtasks. | Medium. Uses the language pathway that pretraining optimised; gives an interpretable intermediate that can be evaluated against TRO ground truth. |
| P4 | Geometric front-end for Camera Relocalization | [Reloc3r](https://arxiv.org/abs/2412.08376), map-free relocalisation | Run a frozen two-view relative-pose regressor on (current head, goal head) and feed the predicted SE(3) as extra state tokens, or use it as the P3 supervision target at test time. | Medium. Task-specific; strong prior for CR, irrelevant for in-hand tasks. |
| P5 | Goal-timing augmentation | VISTA temporal overlap window | When hindsight-sampling goals for π0.5, sample from a window around the terminal frame so small goal/observation misalignments do not destabilise the final approach. | Low. |

### 4.4 Cross-cutting: making the goal image *precise*

- Same-camera goal + early fusion + hindsight sampling is the combination every
  precision-oriented prior work converged on (GCBC → SuSIE → Octo; ViNT → NoMaD).
  Treat it as the default for ACT and DP.
- Privileged supervision is our unfair advantage: relative goal pose, per-object
  pose deltas, block-to-slot assignment, articulated joint value, and a "goal
  reached" flag are all in the TRO or computable from the simulator. Cheap
  auxiliary heads on all three architectures; none of the surveyed papers had
  this.
- Multi-view goals (head + wrists at the goal state) help in-hand tasks (OS, MR)
  where the head view under-constrains orientation. VISTA is the precedent.
- Explicit correspondence as an intermediate (Track2Act tracks, DINO feature
  correlation between goal and current view) is a heavier option if the
  end-to-end variants plateau on Mental Rotation / World Alignment.

### 4.5 Suggested experiment ladder

1. Language-only ACT / DP / π0.5 with the constant prompt. This is the floor and
   measures how much a policy can infer from the initial scene alone.
2. D1 and its ACT/π0.5 analogues (late fusion, BPP-style). Cheapest, and the
   comparison everyone will ask for.
3. A1/D2 early fusion; P1 extra slot with source embedding.
4. Add D5 hindsight sampling, D4 goal masking + guidance, A4/D6 auxiliary pose
   head. Re-run the goal-sensitivity diagnostic (4.6) at each step.
5. A3, D3 (token-level joint encoder), P3 (textual subgoal) as the "proposed
   method" candidates; multi-view goals last.

### 4.6 Goal-sensitivity diagnostic (run from step 1 onward)

For a fixed initial state, roll out with (a) the correct goal image, (b) a goal
image from a *different* instance of the same task with the same initial scene,
and (c) a blank/null goal. A goal-conditioned policy must produce different
trajectories and different final poses in (a) vs (b), and (a) must beat (c) by a
wide margin. Log the pose error to the *swapped* goal as well: a policy that
reaches the swapped target is reading the goal correctly, a policy whose error is
the same for (a), (b), (c) is ignoring it (ViNT's failure). This is the same
matched-pair logic as the intervention analysis and reuses its infrastructure;
it also complements the planned "old goal, new live observation" cross-context
test.

## 5. Open questions to settle before implementation

- Resolution: head camera is 720×720 in the BEHAVIOR data; ACT/DP baselines
  typically downsample. Goal precision may need higher input resolution or a
  goal-centred crop; decide per task after the first early-fusion runs.
- Whether Configuration Matching block design needs an ordering-free goal
  representation (goal images cannot express order; check for dithering).
- Whether to render goal images with the robot hidden as an additional training
  augmentation (guards against posture copying) or only as an ablation.

## References (grouped)

Early fusion: BridgeData V2 ([2308.12952](https://arxiv.org/abs/2308.12952)); SuSIE ([2310.10639](https://arxiv.org/abs/2310.10639)); Octo ([2405.12213](https://arxiv.org/abs/2405.12213), [goal fusion discussion](https://github.com/octo-models/octo/issues/25)); RT-Sketch ([2403.02709](https://arxiv.org/abs/2403.02709)); ViNT ([2306.14846](https://arxiv.org/abs/2306.14846)); NoMaD ([2310.07896](https://arxiv.org/abs/2310.07896)).

Late fusion / diffusion: Behavior Prompting Policy ([2606.30457](https://arxiv.org/abs/2606.30457), [code](https://github.com/real-stanford/behavior_prompting)); MDT ([2407.05996](https://arxiv.org/abs/2407.05996)); BESO ([2304.02532](https://arxiv.org/abs/2304.02532)); GR-MG ([2408.14368](https://arxiv.org/abs/2408.14368)); C-BeT ([2210.10047](https://arxiv.org/abs/2210.10047)); VQ-BeT ([2403.03181](https://arxiv.org/abs/2403.03181)); RoboCat ([2306.11706](https://arxiv.org/abs/2306.11706)); GRIF ([2307.00117](https://arxiv.org/abs/2307.00117)); Play-LMP ([1903.01973](https://arxiv.org/abs/1903.01973)); MCIL ([2005.07648](https://arxiv.org/abs/2005.07648)); HULC ([2204.06252](https://arxiv.org/abs/2204.06252)).

VLA / world-model era: VISTA–GoalVLA ([2602.10983](https://arxiv.org/abs/2602.10983)); ForeAct ([CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/papers/Zhang_ForeAct_Steering_Your_VLA_with_Efficient_Visual_Foresight_Planning_CVPR_2026_paper.pdf)); ManualVLA ([2512.02013](https://arxiv.org/abs/2512.02013)); WorldScape Policy 2.0 ([2607.18840](https://arxiv.org/abs/2607.18840)); CoT-VLA ([2503.22020](https://arxiv.org/abs/2503.22020)); Act2Goal ([2512.23541](https://arxiv.org/abs/2512.23541)); Envision ([2512.22626](https://arxiv.org/abs/2512.22626)); Goal-VLA ([2506.23919](https://arxiv.org/abs/2506.23919)); Track2Act ([2405.01527](https://arxiv.org/abs/2405.01527)); CoTDiffusion ([CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/papers/Ni_Generate_Subgoal_Images_before_Act_Unlocking_the_Chain-of-Thought_Reasoning_in_CVPR_2024_paper.pdf)); π0.5 ([2504.16054](https://arxiv.org/abs/2504.16054)); openpi image-key limitation ([issue #620](https://github.com/Physical-Intelligence/openpi/issues/620)).

Geometry: Reloc3r ([2412.08376](https://arxiv.org/abs/2412.08376)); Map-free relocalisation ([2210.05494](https://arxiv.org/abs/2210.05494)).
