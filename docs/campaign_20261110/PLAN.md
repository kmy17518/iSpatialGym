# Fresh collection campaign — 2026-11-10

Source request: [REQUEST.md](REQUEST.md). Baseline: `9949b012` / frozen runtime `trunk_f24ea424`. Campaign branch: `campaign-20261110`. Existing worktrees, datasets, code snapshots, caches and jobs are read-only inputs. No pushes.

## Deliverables and priority

1. Fresh standard evaluation: 100 classes, one pinned template each, ten instances each (1,000 total). Alignment axial accounts for the omitted five classes in the request's first list.
2. Fresh model holdout: 99 classes with globally unseen task-role models. Category/room holdout: 23 tasks with globally unseen task-role categories and rooms. Counts follow the request's per-task table, with unresolved counts explicitly pending.
3. Training: all 100 standard classes plus usable additional classes; target at least 1,000 distinct templates and 100,000 successful instances. Task quotas in the request sum to exactly 100,000.
4. Every template must have one successful demo and one genuine open-loop physics replay before production. Past demos and instances do not count. Reuse compatible grasp/reach resources only.
5. Prepare batched replay before the collection deadline. The deadline is a handoff target, never an automatic shutdown instruction.

## Execution lanes

| Lane | Work | Release gate |
|---|---|---|
| Compute and monitor | Inspect both cluster users/partitions and Vast; isolate ledgers, index ranges, outputs and spending; start durable monitor | Exact job ownership and compatible driver/VRAM/CPU capacity |
| Contracts | Task/class/template/instance IDs; role inventory; split reservations; perturbation and group provenance; validation evidence | CPU invariants pass and sampler/replay integration verified |
| Templates and assets | Pin standard templates; inventory all roles; reserve holdouts globally; registered variants; generate requested custom assets | Geometry/semantic review, ready grasps/maps, one demo and replay |
| Runtime integration | Goal-first groups, persisted scale/scene/light state, failure records, sampling and replay adapters | Fresh GPU canary and replay with exact restored state |

Each ready template advances independently. A missing grasp/map for one template must not hold the others. Validation and data builds use available GPUs while the broader inventory and code are prepared.

## Timeline and control loop

All user-facing timestamps use America/Chicago. Owner confirmed the handoff target: November 12 at 13:00 America/Chicago, 19:00 UTC (literal noon PDT).

- First hour: resolve contradictory counts and spending cap; publish role inventory, specification and live state; inspect cluster/Vast; launch fresh baseline canaries in isolated output roots; implement and test campaign contracts.
- Following hours: release correctly grouped unperturbed standard templates first; reserve held-out models/categories/rooms before train expansion; validate candidates immediately as maps arrive. Pending owner alias updates remain held.
- Remaining collection window: fill standard and holdout evaluation deficits first; expand training to the requested template and instance diversity; shard slow but viable templates across GPUs using globally unique indices and quota claims.
- Before handoff: exercise batched open-loop replay, export manifests, audit split leakage and group siblings, and reconcile collected/accepted/replayed counts.

Progress is measured separately as attempts, sampled instances, successful demos, replay-validated templates, accepted split instances, and used GPUs. No aggregate success label substitutes for the validation gate. Forecasts use measured successful instances per GPU-hour and remaining quotas; feasibility is reported rather than assumed.

## Binding decisions and defaults

- Standard eval remains unperturbed. Its training siblings also use default scale, scene and lighting so they share the goal and observation conditions. Additional train-only groups may be perturbed.
- Every candidate model in every role must have verified `ready` status from the asset catalog. File existence or a successful prior demo does not replace the readiness gate. New generated assets enter candidate manifests only after readiness validation.
- Training buckets are soft balancing targets (owner follow-up). Prefer underfilled buckets, but do not reject an otherwise valid training instance solely for missing a bucket allocation. Eval bucket quotas remain strict.
- Goal groups target at most three distinct successful starts/grasps, with at least two needed to promote a standard-eval member. Timed-out singletons are training-only. Camera relocalization is exempt.
- Owner policy v2 permits repeated canonical goal states across independently generated groups and train/eval splits. Group IDs and instances remain fresh; within-group distinct starts/grasps, maximum three members, successful training siblings, exact physics gates, and model/category/room holdouts remain enforced. A different viewpoint is not required merely to create another group. See [OWNER_DECISIONS.md](OWNER_DECISIONS.md).
- All sampling-time perturbations are shared within a group. Lighting is selected and recorded at collection time and restored at replay; applying a random light only during replay would break reproducibility. Visibility-sensitive scene removals occur before sampling checks.
- Use the maximum feasible layout envelope at supported scales when screening supports. Count usable robot standoff areas without imposing an untucked arm pose; detailed IK still decides feasibility.
- Preloaded candidate pools must be bounded by measured memory. Inactive objects must not collide, affect visibility, or appear in the selected template/TRO. Large pools may be sharded.
- Preserve frozen physics, camera-frame checks, v2-only grasps and alias-memory-off defaults. New behavior is campaign-scoped. Existing 100-template oracle programs remain unchanged.
- Save failed start/goal proposals with phase, reason, template, scale, seed and group ID when available. Sampling failures and execution failures remain distinct.
- All held-out reservations apply across target, partner, support and distractor roles across task families. Category holdout reserves actual categories even when supports share an abstract utility class.
- Room/model/category reservations are frozen before releasing expanded train manifests. Asset derivations retain source provenance; face permutation alone cannot leak a held-out asset identity into training under a renamed ID.
- Three explicit blocked alias families: touching_03; under_large_01/02/03; nextto_04. Await the owner's new code/template versions before those releases.
- Registered variants are candidates for training, subject to their own data dependencies and fresh demo/replay gate; registration alone is not proof of readiness.
- A new oracle candidate has a bounded engineering/validation budget, initially 60 minutes of active investigation plus at most 2 GPU-hours of validation. Reject or defer an unproductive candidate and try another; standard classes are not silently dropped.

## Owner decisions — 2026-11-10

- Vast: $3,000 total campaign budget. Existing campaign spending is separate; include this campaign's storage/transfer/idle costs in its ledger.
- Handoff: November 12 at 19:00 UTC / 13:00 America/Chicago. No automatic shutdown.
- Block design: unseen Kohs permutation for model holdout, 9 instances; Wechsler for category holdout, 1 instance.
- Category articulation: 2 instances per small-scale task (1+1 buckets), 3 per large-scale task (2+1 buckets).
- Articulation single-joint rule concerns the initial-to-goal change. Intermediate movement of a second flap is allowed if it returns; retain the validated carton preparation sequence.
- Structural walls/floors are exempt from global object model/category holdout. The category split's held-out room requirement remains.
- Tray/board categories are exempt from category holdout; model holdout still requires unused tray/board models.
- Final eval quotas: standard 1,000; model holdout 891; category/room holdout 109. Total eval 2,000; target training 100,000.
- Distractors are a category pool: a model-holdout template may select any available globally unseen models from the pool. No unused model is required for every distractor category. disembedding_06 can use the unused soy-milk bottle model without requiring unseen coconut-bottle or soy-carton models.
- Repeated goals: “Don't care about repeats between train and eval when generating for the whole benchmark.” The allocator's prior global canonical-goal uniqueness rule is removed through an audited policy migration. Exact native pair anchor goals may be reused with newly sampled starts/grasps; translation or gap changes are optional proposals, not release requirements.

## Live state

The accessible monitor is `tmp/outputs/campaign_20261110/LIVE.md` and machine-readable `live.json` in the original workspace. It distinguishes this campaign's GPUs from unrelated work, includes probe timestamps/errors, and never treats stale/unknown observations as zero.
