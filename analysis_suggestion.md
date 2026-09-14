# iSpatialGym paper experiment plan

This document specifies:

1. The main benchmark experiment.
2. The controlled intervention analyses.
3. The training-data strategy and data-scaling analyses.
4. The exact training and test composition for each experiment.

The tables below are templates for the paper. Values such as `--` are replaced
with measured results.

## 1. Fixed terminology and reporting rules

### Public tasks

| Abbreviation | Public task |
| --- | --- |
| CR | Camera Relocalization |
| OS | Object Scaling |
| MR | Mental Rotation |
| DE | Disembedding |
| CM | Configuration Matching |
| WA | World Alignment |
| IORS | Inter-object Relation Satisfaction |

Configuration Matching contains three kinds of instances:

- Block design.
- Articulated opening.
- Object attachment.

These are not separate benchmark tasks or separately weighted evaluation
strata. Compute one Configuration Matching success rate over all its instances,
just as Inter-object Relation Satisfaction remains one task even though its
instances use different relation predicates. The overall benchmark score is the
equal average of the seven public task scores.

### Instance-factor representation

For analysis, represent an instance as:

```text
task specification + pose/configuration + task object identity + background + lighting
```

The **pose/configuration** factor contains the initial and goal poses of the
agent and task objects, plus relevant joint, grasp, layout, relation, or
attachment configurations. It describes simulator state, not the trajectory.
The required trajectory is an outcome that may change after either a
pose/configuration intervention or an object intervention.

### Variation and shift

| Factor | Variation | Shift |
| --- | --- | --- |
| Pose/configuration | New values inside the training pose/joint/configuration ranges | A reachable, held-out pose/joint/configuration region |
| Object | A held-out model from a category present in training | A held-out category compatible with the same task role |
| Background | Replace 1 or 2 task-irrelevant visible background objects | Replace more than 50% of eligible visible background objects |
| Lighting | New values inside a trained lighting setup and range | A held-out lighting setup or disjoint parameter range |

Variation appears in two places for different reasons:

1. The 20 standard instances cover natural in-range variation and measure
   average benchmark performance.
2. The matched intervention analysis changes one factor slightly while
   holding all others fixed. It measures local sensitivity that cannot be
   recovered from an unpaired 20-instance average.

Variation is therefore not presented as out-of-distribution generalization.
It is a controlled "what if this changed slightly?" analysis. Shift remains the
out-of-distribution analysis.

Object models are the exception in the standard test: it uses training-seen
task-relevant models. Same-category held-out models appear only in the matched
object-variation analysis, and held-out categories appear only in the
object-shift analysis.

For background shift, also require a minimum change in visible background
pixels. Object count alone is insufficient because many replaced objects might
be outside the camera view.

A completely new room geometry or layout is a separate **scene-layout shift**.
It should not be mixed with the background-composition shift above.

### Action-changing versus action-preserving factors

Use **controlled intervention** as the umbrella term in the paper. All four can
be called counterfactuals in a causal sense, but they test two different
properties:

- **Action-changing interventions:** pose/configuration and task-object identity.
  The correct trajectory is expected to change, so generate a new oracle.
- **Action-preserving interventions:** background and lighting. The correct
  trajectory should remain valid, so replay the exact anchor oracle and retain
  the intervention only if replay succeeds without contacting changed objects.

Candidate objects and occluders in Disembedding are task-relevant objects, not
background. Tables, trays, supports, attachment sites, handles, and objects
inside the robot's swept volume are also excluded from the background factor.

### Common split rule

An anchor instance and every intervention-derived sibling must remain in the
same train, validation, or test split. No test anchor, sibling, oracle rollout,
goal image, or frame may appear in training.

Object assets should also be grouped by source mesh or asset lineage before
splitting so that cosmetic duplicates cannot cross splits.

## 2. Dataset partitions

Use the following named partitions consistently throughout the paper.

| Dataset | Purpose |
| --- | --- |
| `D_train_standard` | Diverse but independently sampled training trajectories |
| `D_val` | Model selection, training-strategy selection, and threshold selection |
| `D_test_main_public` | Released benchmark instances: 20 per public task |
| `D_test_main_private` | Hidden final benchmark instances: 20 per public task |
| `D_test_interventions` | Matched one-factor variation and shift families |
| `D_test_crossed` | Instances containing multiple simultaneous shifts |
| `D_test_scene_layout` | Optional held-out room/layout experiment |

Final test categories, object models, pose/configuration regions, background
assets, scene layouts, and lighting regimes must be selected before experiments
begin.
Validation uses different holdouts from the final test.

### Experiment map

| Experiment | Training data | Test data | Paper output |
| --- | --- | --- | --- |
| Main benchmark | `D_train_standard` | Public and private main tests | Tables 1A and 1B |
| Single-factor interventions | Main benchmark checkpoint | `D_test_interventions` | Tables 2A and 2B |
| Combined shifts | Main benchmark checkpoint | `D_test_crossed` | Table 3 |
| Training-data strategy | Equal-budget strategy-specific train sets | All three test sets | Table 4 |
| Data scaling | Nested subsets of the selected train strategy | Fixed main and shift tests | Table 5 |
| Settling behavior | Main benchmark checkpoint | Private main test | Table 6 |
| Disembedding occlusion | Main benchmark checkpoint | Matched Disembedding set | Table 7 |
| Intervention certification | No policy training | Candidate background/light pairs | Table 8 |

## 3. Main benchmark experiment

### Question

How well does each policy solve the seven benchmark tasks on new instances drawn
from the standard benchmark distribution?

### Training dataset: `D_train_standard`

Recommended full-budget composition:

| Component | Composition |
| --- | --- |
| Size | 6,400 oracle trajectories per public task; 44,800 total |
| Task balance | Equal number for each of the seven public tasks |
| CM composition | Approximately balance block design, articulated opening, and object attachment within CM |
| Pose/configuration | Broad independent samples from the designated training ranges |
| Objects | Training categories and training models only |
| Background | Training scenes and background assets only |
| Lighting | Training rigs and parameter ranges only |
| Factor correlation | Sample factors independently so scene or lighting cannot predict the target action |
| Intervention pairing | None required in the standard dataset |

Because trajectory lengths differ, also report the number of action transitions
and frames. During optimization, sample public tasks uniformly; after selecting
CM, sample its three variants uniformly.

All methods in the main same-data comparison must receive exactly this dataset.
If a method uses paired-intervention augmentation at the same transition budget,
label that composition explicitly. If it uses additional data, place it in a
separate table block so it is not presented as a same-data comparison.

### Validation dataset: `D_val`

- 10 instances per public task; 70 total.
- Include all three Configuration Matching variants and all intended IORS
  relation types within their respective task allocations.
- Same factor support as training but with unseen tuples and random seeds.
- Additional validation-only pose/configuration regions, object models/categories,
  backgrounds, and lighting regimes for selecting the final training strategy.
- No final-test holdout value may be used for model or threshold selection.

### Test datasets: `D_test_main_public` and `D_test_main_private`

Recommended composition:

| Component | Composition |
| --- | --- |
| Size | 20 public and 20 private instances per public task |
| Total | 140 public + 140 private = 280 simulator instances |
| CM composition | Approximately 7 block-design, 7 articulated-opening, and 6 object-attachment instances per split; rotate the extra slots between splits |
| IORS composition | Balance the intended relation predicates within the 20 IORS instances |
| Pose/configuration | New continuous samples from the training-supported ranges |
| Objects | Training-seen task-relevant models in new state and factor combinations |
| Goal | Fresh exact goals sampled from training-supported ranges; no reused goal state or goal image |
| Background | Training-supported scene/background distribution with new configurations |
| Lighting | Training-supported rigs/ranges with new parameter samples |
| Variation coverage | Stratify the 20 instances across supported pose/configuration, background, and lighting ranges and across the seen model pool |
| Pairing | No explicit intervention is applied |
| Leakage control | No exact task state, tuple, seed, trajectory, or image appears in training |

The standard split deliberately measures execution on familiar object models.
It must still use new object poses, robot states, goals, backgrounds, lighting,
factor combinations, and random seeds. A standard-test TRO, trajectory, or goal
image may not be reused from training.

Models reserved for object variation must be absent from training, including as
background objects. Models in the public and private object-variation pools
should also be disjoint where asset availability permits.

All exact goals are fresh, but this is not automatically a pose shift. For
continuous poses, a new value sampled from the same allowed range is a normal
test sample. Pose shift is reserved for a goal region or difficulty range that
received no training samples.

The private split is the primary paper result. The public split provides
released evaluation examples and should not be used to select the final
checkpoint. Held-out categories and held-out pose regions used to define shifts
belong in `D_test_interventions`, not in the main test.

### Main paper tables

Run evaluation long enough to compute both ordinary goal success and settled
success from the same rollout.

**Table 1A: Private-test ordinary goal success (%)**

| Method | Train data | CR | OS | MR | DE | CM | WA | IORS | Overall |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Oracle | Privileged | -- | -- | -- | -- | -- | -- | -- | -- |
| Baseline 1 | Standard | -- | -- | -- | -- | -- | -- | -- | -- |
| Baseline 2 | Standard | -- | -- | -- | -- | -- | -- | -- | -- |
| Proposed method | Standard | -- | -- | -- | -- | -- | -- | -- | -- |
| Proposed method + paired data | Paired, matched budget | -- | -- | -- | -- | -- | -- | -- | -- |

**Table 1B: Private-test settled success (%)**

| Method | Train data | CR | OS | MR | DE | CM | WA | IORS | Overall |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Oracle | Privileged | -- | -- | -- | -- | -- | -- | -- | -- |
| Baseline 1 | Standard | -- | -- | -- | -- | -- | -- | -- | -- |
| Baseline 2 | Standard | -- | -- | -- | -- | -- | -- | -- | -- |
| Proposed method | Standard | -- | -- | -- | -- | -- | -- | -- | -- |
| Proposed method + paired data | Paired, matched budget | -- | -- | -- | -- | -- | -- | -- | -- |

Use settled success as the primary benchmark number if the final protocol
requires `PoseSettled`. Keep Table 1A because it distinguishes inability to
reach the goal from inability to stop stably. Report the same two tables for the
public split in the appendix.

## 4. Analysis 1: Single-factor controlled interventions

### Question

How much does performance change when exactly one factor changes slightly
(variation) or moves outside the training-supported distribution (shift)?

This is not redundant with the main test. The main test compares unrelated
instances and estimates average performance. This analysis compares matched
pairs and attributes the change to one controlled intervention.

### Training dataset

Use the exact checkpoint and `D_train_standard` training recipe from the main
table. Do not retrain separately for each factor.

### Test dataset: `D_test_interventions`

Select 5 anchor families per applicable public task from the private main test.
An anchor plus eight applicable siblings produces at most 45 instances per
task. Because the anchors are already evaluated in the main test, this adds at
most 280 policy rollouts; Camera Relocalization's missing object factor reduces
the maximum to 270.

The five CM anchors should collectively cover its three variants; this remains
one CM analysis result.

For every anchor, create each applicable sibling:

```text
anchor
pose/configuration variation
pose/configuration shift
object-model variation
object-category shift
background variation
background shift
lighting variation
lighting shift
```

Composition:

| Sibling | Changed value | Fixed values | Oracle |
| --- | --- | --- | --- |
| Pose/configuration variation | A small but task-meaningful change inside the supported range | Object, background, lighting | Generate new |
| Pose/configuration shift | Held-out reachable pose/configuration region | Object, background, lighting | Generate new |
| Object variation | A size/affordance-matched model from the same category | Pose/configuration difficulty, background, lighting | Generate new |
| Object shift | Held-out compatible category | Pose/configuration difficulty, background, lighting | Generate new |
| Background variation | Replace 1 or 2 eligible visible objects | Pose/configuration, task objects, lighting | Replay anchor |
| Background shift | Replace >50% of eligible visible objects and enough visible pixels | Pose/configuration, task objects, lighting | Replay anchor |
| Lighting variation | A small change within the same lighting setup and range | Pose/configuration, objects, background | Replay anchor |
| Lighting shift | Held-out rig or parameter range | Pose/configuration, objects, background | Replay anchor |

A pose/configuration variation must be larger than the goal tolerance;
otherwise the anchor action may already satisfy both goals and the intervention
is not meaningful. It should still remain inside the training range and have
comparable difficulty.

Camera Relocalization has no task-object intervention. Mark its object cells
`N/A`; its landmarks are part of the scene factor.

For goal-image-conditioned tasks:

- Rerender the goal image when target pose/configuration, object identity/geometry,
  background, or lighting changes it.
- For the primary background/lighting condition, apply the intervention to both
  the live observation and goal image.
- Keep "live observation changed but old goal image retained" as a separate
  cross-context experiment if desired.

### Analysis table

Do not use a signed success delta as the only metric. For binary anchor result
`A` and intervention result `I`, every pair has one of four outcomes:

| Pair outcome | Meaning |
| --- | --- |
| Success to success (`S->S`) | Solved both |
| Success to failure (`S->F`) | Regression after intervention |
| Failure to success (`F->S`) | Improvement after intervention |
| Failure to failure (`F->F`) | Solved neither |

The previously proposed signed delta is:

```text
mean(I - A) = P(F->S) - P(S->F)
```

It is only a **net performance change**. A positive value means improvements
outnumber regressions; it does not necessarily mean the policy is robust.
Regressions and improvements can cancel completely.

For action-preserving background and lighting interventions, both `S->F` and
`F->S` demonstrate sensitivity. Report regression rate and total flip rate
`P(S->F) + P(F->S)` as primary diagnostics.

For action-changing pose/configuration and object interventions, report absolute
intervention success as the primary result. `F->S` may simply mean that the new
required trajectory was easier, while `S->F` may mean it was harder.

**Table 2A: Paired intervention outcomes (%)**

| Class | Factor | Level | Anchor success | Intervention success | S->S | S->F | F->S | F->F | Flip rate |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Action-changing | Pose/config. | Variation | -- | -- | -- | -- | -- | -- | -- |
| Action-changing | Pose/config. | Shift | -- | -- | -- | -- | -- | -- | -- |
| Action-changing | Object | Variation | -- | -- | -- | -- | -- | -- | -- |
| Action-changing | Object | Shift | -- | -- | -- | -- | -- | -- | -- |
| Action-preserving | Background | Variation | -- | -- | -- | -- | -- | -- | -- |
| Action-preserving | Background | Shift | -- | -- | -- | -- | -- | -- | -- |
| Action-preserving | Lighting | Variation | -- | -- | -- | -- | -- | -- | -- |
| Action-preserving | Lighting | Shift | -- | -- | -- | -- | -- | -- | -- |

**Table 2B: Intervention success by task (%)**

| Factor | Level | CR | OS | MR | DE | CM | WA | IORS | Macro |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Pose/config. | Variation | -- | -- | -- | -- | -- | -- | -- | -- |
| Pose/config. | Shift | -- | -- | -- | -- | -- | -- | -- | -- |
| Object | Variation | N/A | -- | -- | -- | -- | -- | -- | -- |
| Object | Shift | N/A | -- | -- | -- | -- | -- | -- | -- |
| Background | Variation | -- | -- | -- | -- | -- | -- | -- | -- |
| Background | Shift | -- | -- | -- | -- | -- | -- | -- | -- |
| Lighting | Variation | -- | -- | -- | -- | -- | -- | -- | -- |
| Lighting | Shift | -- | -- | -- | -- | -- | -- | -- | -- |

Report raw pair counts alongside percentages because five pairs per task make
the task-level estimates coarse. Bootstrap whole matched families rather than
individual frames.

## 5. Analysis 2: Combined shifts

### Question

Do shifts interact, causing a larger failure than single-factor tests suggest?

### Training dataset

Use the same `D_train_standard` checkpoint as in the main experiment.

### Test dataset: `D_test_crossed`

- Select 3 anchors per public task from the controlled-intervention analysis.
- Use one anchor from each Configuration Matching variant while still reporting
  one aggregate CM result.
- Use only final-test shift values.
- Construct the four conditions below.
- Generate new oracles whenever pose/configuration or task objects change.
- Certify action preservation for background and lighting changes.

```text
Anchor
Action-changing shift: pose/configuration + object
Nuisance shift: background + lighting
All applicable shifts: pose/configuration + object + background + lighting
```

For Camera Relocalization, omit the object term.

### Analysis table

**Table 3: Settled success under combined shifts (%)**

| Condition | CR | OS | MR | DE | CM | WA | IORS | Macro |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Anchor | -- | -- | -- | -- | -- | -- | -- | -- |
| Pose/configuration + object shift | -- | -- | -- | -- | -- | -- | -- | -- |
| Background + lighting shift | -- | -- | -- | -- | -- | -- | -- | -- |
| All applicable shifts | -- | -- | -- | -- | -- | -- | -- | -- |

Do not merge this table with Table 2. Table 2 estimates controlled individual
effects; Table 3 measures compositional robustness.

## 6. Analysis 3: Training-data strategy

### Question

At a fixed data budget, is independent diversity sufficient, or does explicit
intervention pairing improve robustness?

### Policy and optimization controls

- Use one policy architecture for every row.
- Match total labeled action transitions within 1%.
- Match public-task proportions and each task's internal instance-type mix.
- Match trajectory-length and difficulty distributions.
- Use only train-pool factors; final-test models, categories,
  pose/configuration regions, scenes, and lighting regimes remain unavailable.
- Use at least three dataset draws and three optimization seeds.

Run the main ablation at 1,600 trajectory-equivalents per public task
(11,200 total), then confirm the strongest strategies at the full budget.

### Training datasets

| Strategy | Training composition |
| --- | --- |
| Independent diversity | 100% independently sampled pose/configuration and object trajectories; independently sampled scene and lighting |
| Reduced-core control | Same number of independent task cores as the paired condition; remaining budget is another rollout/render of the same condition |
| Nuisance-paired | 50% anchors, 25% matched background replays, and 25% matched lighting replays |
| All-factor-paired | 50% anchors and 12.5% siblings for each of pose/configuration, object, background, and lighting |
| Mixed | 75% independent trajectories, 12.5% matched background replays, and 12.5% matched lighting replays |
| Mixed + pair-aware loss | Exactly the Mixed dataset plus a consistency loss for matched background/lighting observations |

Do not force identical actions across pose/configuration or object pairs. Their
correct actions can differ. A pair-aware invariance loss applies only to
certified action-preserving background and lighting pairs.

When a factor is not applicable, redistribute its share evenly among the other
applicable factors for that task.

The Reduced-core control is necessary because paired datasets contain fewer
independent task cores at the same transition budget. Without it, a difference
could be caused by core diversity rather than intervention pairing.

### Test datasets

Evaluate every strategy on the same:

- `D_test_main_private`.
- `D_test_interventions`.
- `D_test_crossed`.

### Analysis table

**Table 4: Effect of training-data strategy on settled success (%)**

| Training strategy | Main | Pose/config. var. | Pose/config. shift | Object var. | Object shift | Background var. | Background shift | Lighting var. | Lighting shift | All-shift |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Independent diversity | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| Reduced-core control | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| Nuisance-paired | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| All-factor-paired | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| Mixed | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| Mixed + pair-aware loss | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- |

Each intervention column reports absolute success on that sibling set. Table 2A
reports the paired outcome breakdown.

This experiment answers whether explicit paired interventions are useful. The
result should determine the final training recipe rather than assuming in
advance that pairing is always better.

## 7. Analysis 4: How much training data is needed?

### Question

How does standard and shifted performance scale with the number of oracle
trajectories?

### Training datasets

Use nested subsets so that every smaller dataset is contained in the next
larger dataset:

| Trajectories per public task | Total trajectories |
| ---: | ---: |
| 100 | 700 |
| 400 | 2,800 |
| 1,600 | 11,200 |
| 6,400 | 44,800 |

Use the training strategy selected using `D_val`. Keep its proportions fixed at
all four budgets. Configuration Matching's allocation should approximately
balance its three internal variants without scoring them as separate tasks.
Match optimization by both:

1. A fixed number of epochs, to measure normal data scaling.
2. A fixed number of gradient updates, to separate data amount from compute.

Use at least three dataset draws and three optimization seeds.

### Test datasets

Every checkpoint uses the exact same:

- `D_test_main_private`.
- `D_test_interventions`.
- All-shift subset of `D_test_crossed`.

### Analysis table

**Table 5: Data scaling**

| Trajectories per task | Total trajectories | Main | Pose/config. shift | Object shift | Background shift | Lighting shift | All-shift |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 100 | 700 | -- | -- | -- | -- | -- | -- |
| 400 | 2,800 | -- | -- | -- | -- | -- | -- |
| 1,600 | 11,200 | -- | -- | -- | -- | -- | -- |
| 6,400 | 44,800 | -- | -- | -- | -- | -- | -- |

Also plot these values as learning curves. Report the smallest dataset whose
confidence interval overlaps the full-data result; this gives a defensible
answer to "how much data is enough."

## 8. Analysis 5: Settling and stopping behavior

### Question

When a policy first reaches the spatial goal, can it stop and hold the complete
goal configuration?

### Training dataset

Use the same trained checkpoints as the main experiment. No special retraining
is needed.

### Test dataset

Use `D_test_main_private`. Record:

- First ordinary-goal success time.
- First settled-goal success time.
- Settling latency.

### Analysis table

**Table 6: Goal achievement and settling**

| Method | Task | Goal success (%) | Settled success (%) | Success gap (pp) | Median settling latency (s) |
| --- | --- | ---: | ---: | ---: | ---: |
| Baseline 1 | CR | -- | -- | -- | -- |
| Baseline 1 | OS | -- | -- | -- | -- |
| ... | ... | ... | ... | ... | ... |
| Proposed method | IORS | -- | -- | -- | -- |

Compute latency only for episodes that eventually achieve settled success.
Also report the fraction that reached an ordinary goal but never settled.

## 9. Analysis 6: Disembedding occlusion

### Question

How does target-object occlusion affect Disembedding performance?

### Training dataset

Use the main checkpoint trained on `D_train_standard`. Balance training
occlusion so target identity, target location, and object category do not
predict the occlusion bin.

### Test dataset

Construct a Disembedding-only matched set:

- Keep the target object, destination, background, lighting, and target role
  fixed within a family.
- Change candidate-object placement to produce the desired external-occlusion
  bin.
- Generate a valid oracle for each arrangement because occluders can affect
  grasp access.
- Require the target to remain in frame and above the minimum visible-pixel
  threshold.

Suggested bins should be finalized after a generation pilot:

```text
0%-20%, 20%-40%, 40%-60%, and 60%-maximum-allowed occlusion
```

Use 5 instances per bin, for 20 Disembedding analysis instances total. These can
be derived from five matched target families rather than introducing 20
unrelated setups.

### Analysis table

**Table 7: Disembedding performance by target occlusion**

| External occlusion | Number of instances | Goal success (%) | Settled success (%) | Pick success (%) | Median completion time (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0%-20% | -- | -- | -- | -- | -- |
| 20%-40% | -- | -- | -- | -- | -- |
| 40%-60% | -- | -- | -- | -- | -- |
| 60%-maximum | -- | -- | -- | -- | -- |

## 10. Dataset-quality validation

This is a required appendix table, not a learned-policy experiment.

### Training dataset

None.

### Test dataset

Use every proposed background and lighting pair before admitting it to
`D_test_interventions` or `D_test_crossed`. Apply the same validation to in-support paired
perturbations used for training augmentation.

For each pair:

1. Replay the exact anchor oracle controls.
2. Require the same task success.
3. Require no contact with changed background objects.
4. Require unchanged task predicates and target states within tolerance.
5. Reject failed pairs and record the reason.

### Validation table

**Table 8: Action-preserving intervention certification**

| Factor | Proposed pairs | Accepted pairs | Acceptance rate | Oracle replay success | Background-contact failures | Other failures |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Background variation | -- | -- | -- | -- | -- | -- |
| Background shift | -- | -- | -- | -- | -- | -- |
| Lighting variation | -- | -- | -- | -- | -- | -- |
| Lighting shift | -- | -- | -- | -- | -- | -- |

## 11. Recommended paper ordering

Main paper:

1. Table 1A/1B: overall benchmark performance.
2. Tables 2A/2B: single-factor controlled interventions.
3. Table 4: training-data strategy.
4. Table 5 plus learning curves: data scaling.
5. Table 3: combined-shift stress test.
6. Table 7: Disembedding occlusion analysis.

Appendix:

1. Table 6: detailed settling and latency results.
2. Table 8: action-preservation validation.
3. Per-task and per-category intervention results with confidence intervals.
4. Rejected intervention counts and rejection reasons.
