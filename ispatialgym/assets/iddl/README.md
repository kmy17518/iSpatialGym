# iddl — iSpatialGym domain definition language

One `.iddl` file per subtask (21 files, one directory per task family). An iddl file is a BDDL problem skeleton
written over **roles** instead of synset instances: no synsets appear (the agent excepted), group roles have no fixed
size or category, and fixed roles name scene objects only by kind. `ground_isg` grounds an iddl file with the icl
(candidate list) into concrete BDDL problems — one per category combination, named by the `:alias` template — and
into the task customs list; `sample_isg` then binds models per template.

```
(define (iddl <subtask-id>)
    (:family <family>)                       ; camera_relocalization | object_scaling | mental_rotation | disembedding |
                                             ; configuration_matching | alignment | inter_object_relation_satisfaction
    (:alias <template>)                      ; alias of the grounded BDDL; {role} slots are filled with category names
    (:roles
        <role> - <kind>                      ; kind: object | group | fixed | agent
        ...
    )
    (:groups                                 ; optional: composite groups made of other roles (attachment assembly)
        <role> <role> - <group-role>
    )
    (:options ...)                           ; optional: requirements ground_isg / the icl must satisfy (below)
    (:init <literals>)                       ; no ? prefix, roles in place of instances
    (:goal (and <literals>))                 ; ? prefix; pose_settled atoms are direct children of the and
    (:measure <literals>)                    ; optional: pose_reached atoms reported but not gating
)
```

## Role kinds

| kind | grounded to | notes |
|---|---|---|
| `object` | one spawned instance `<synset>_k` | category from the icl (alias slot `{role}`), model per template |
| `group` | a BDDL group `<role>_group.n.01_1` with N spawned members | N and the members' categories are chosen by `ground_isg` (alias slot lists them, e.g. `bottle_of_beer_3_bottle_of_sake_1`); members exchangeable unless `:options` says otherwise |
| `fixed` | a scene object bound per scene | supports (tables, walls, floors), scene articulated objects, big partners; the alias slot carries the scene object's category |
| `agent` | `agent.n.01_1` | always present; the only role with a fixed synset |

`floor` is declared explicitly in every file (the agent stands on it; ontop(large‑scale) may also use it as `support`).
When two fixed roles are grounded to the same scene object (under(large‑scale) with the partner as init surface) the
BDDL declares the object once and both role names map to it.

## Options

```
(grasps <role> <pipeline>)          ; the role's models need library grasps: floating | tabletop | insert | attachment
(push <role>)                       ; the role is pushed, not grasped (no grasp requirement; contacts from contacts/<key>.json)
(articulated <role>)                ; one joint per object is chosen (roles.<role>.joint / handle_key in the customs list)
(attach <male-role> <female-role>)  ; BEHAVIOR attachment meta-links must match
(hinge <role> <parent-role> <axis>) ; sampler installs a COM hinge (TRO constraint); axis: vertical | normal
(scale <role> configured)           ; per-model scale from the customs list (roles.<role>.scale)
(scene <role> <category> ...)       ; fixed role may bind to scene objects of these categories
(group_size <role> <min> <max>)     ; member count range of a group role
(same_model <role>)                 ; group members may share one model (blocks); default: unique models per template
```

## Literal semantics

Identical to `docs/task_bddl_formats.md`: group literals expand to conjunctions over members (negated literals to
"no member"), `pose_reached` / `layout_reached` on a group are aggregate atoms, `pose_settled` atoms never gate by
themselves, every `pose_reached` atom needs a TRO target. Argument order matters and differs between the two `under`
subtasks: `(under ?target ?partner)` for under(large‑scale) (target ends under the partner), `(under ?partner ?target)`
for under(small‑scale) (the partner ends under the manipulated target).

## Grounding rules (what `ground_isg` does with a file)

1. Fill every alias slot from the icl: `{target}`, `{partner}`, `{support}` → category names; `{distractors}` /
   `{targets}` / `{obstacles}` → `<category>_<count>` tokens; the result is the BDDL `(:alias ...)` and the problem's id.
2. Declare `<synset>_k` instances for object and group roles (synset = KB synset of the category), one `:groups` entry
   per group role and per `:groups` composite, the fixed roles' synsets from the bound scene objects, `agent.n.01_1`.
3. Copy `:init`, `:goal`, `:measure` with roles replaced by instances / group names.
4. Write the customs list entry: roles → per‑scene whitelists (models), fixed bindings, options, generation parameters,
   constraints records; representative‑model test → template + instance 0.
