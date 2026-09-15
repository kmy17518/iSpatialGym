# iddl — iSpatialGym domain definition language

One `.iddl` file per subtask (21 files, one directory per task family). An iddl file is a BDDL problem skeleton
written over **roles** instead of synset instances: no synsets appear (the agent excepted), no categories, no models,
no object counts. Everything concrete lives elsewhere:

- **icl** (`ispatialgym/assets/icl/`): `roles/` — which roles a subtask has, their *kind* (`sampleable` / `group` /
  `scene`), the candidate categories and models per role with the `representative` / `test` marks
  (`candidates.csv`), which candidates go together (`constraints_compatible.csv`) and which must not
  (`constraints_conflicting.csv`); `require/` — one compact row per computed grasp / handle pinch / contact
  (`grasps.csv`, `handles.csv`, `contacts.csv`), the artefacts the `:require` block points at, with `lib/` symlinks to
  the libraries.
- **task customs list** (one entry per grounded alias, `docs/instance_sampling_spec.md` §2.3): how models are drawn
  (group sizes, identical vs unique members, spawn scales, chosen joints), the per‑scene whitelists / blacklists and
  every constraint sampling learned.
- **initialization rules** (`docs/instance_sampling_spec.md` §3–6): how the objects and the robot are placed, which
  constraints the sampler installs (hinges), which IK checks gate an instance.

`ground_isg` grounds an iddl file with the icl into concrete BDDL problems — one per admitted category combination,
named by the `:alias` template — and into the task customs list; `sample_isg` then binds models per template.

```
(define (iddl <subtask-id>)
    (:family <family>)                       ; camera_relocalization | object_scaling | mental_rotation | disembedding |
                                             ; configuration_matching | alignment | inter_object_relation_satisfaction
    (:alias <template>)                      ; alias of the grounded BDDL; {role} slots are filled with category names
    (:groups                                 ; optional: composite groups made of other roles (attachment assembly)
        <role> <role> - <group-role>
    )
    (:require ...)                           ; optional: icl artefacts the role's models must have (below)
    (:init <literals>)                       ; no ? prefix, roles in place of instances
    (:goal (and <literals>))                 ; ? prefix; pose_settled atoms are direct children of the and
    (:measure <literals>)                    ; optional: pose_reached atoms reported but not gating
)
```

## Roles

Roles are bare names used in the literals; there is no declaration block. Every role of a subtask except the two
implicit ones must have rows in `roles/candidates.csv` for that subtask, and the row's `kind` says how it is grounded:

| `roles/candidates.csv` kind | grounded to | notes |
|---|---|---|
| `sampleable` | one spawned instance `<synset>_k` | category from the alias slot `{role}`, model drawn per template from the customs whitelist |
| `group` | a BDDL group `<role>_group.n.01_1` with N spawned members | N and identical‑vs‑unique members from the customs list (`generation.groups.<role>`); the alias slot lists `<category>_<count>` tokens (e.g. `bottle_of_beer_3_bottle_of_sake_1`); goal assignment of a group with goal literals is `exchangeable` (one‑to‑one matching, `docs/task_bddl_formats.md`) |
| `scene` | a scene object bound per scene | supports (tables, wall regions), scene articulated objects, big partners; the alias slot carries the scene object's category; `room` rows bound a subtask to one scene / room |

Implicit roles: `agent` (`agent.n.01_1`, the only fixed synset) and `floor` (the task room's floor; bounded by the
subtask's `room` rows where they exist — camera relocalization and the held‑object tasks are `house_single_floor` /
`living_room_0` for now — otherwise every candidate scene). Role names in use: `target`, `targets` (group), `partner`,
`source`, `destination`, `distractors` (group), `obstacles` (group), `support`, `floor`, `agent`. When two scene roles
are grounded to the same scene object (under(large‑scale) with the partner as init surface) the BDDL declares the
object once and both role names map to it.

## `:require` — artefacts a role's models must have

`:require` lists the icl artefacts that initialization and demo collection need for a role. Nothing else belongs
here: *how* an object is manipulated (grasped, pushed, braced), hinges installed at init, spawn scales, joint choice
and group sizes are initialization rules or customs‑list parameters.

```
(grasps <role> floating|tabletop|insert|attachment) ; icl/require/grasps.csv: >= 1 row with require = grasps:<pipeline>
                                                    ; and verdict ok / flag for every model of the role (library keys
                                                    ; __f / __t / __i / __a; the flower variant __fl counts as insert)
(contacts <role>)                                   ; icl/require/contacts.csv: >= 1 push row for every model of the role
                                                    ; (push ends, brace kinds, stop edges; B6 fills the sheet)
(handle <role>)                                     ; icl/require/handles.csv: >= 1 row with verdict handle / edge /
                                                    ; manual for the model (spawned) or the (scene, instance) (scene
                                                    ; objects, B4 scaled annotations); its joint is a roles.<role>.joint
                                                    ; candidate
```

`ground_isg` treats every line as an admissibility filter over the `require/` sheets (the sheets summarise the
libraries they link to under `require/lib/`, `path` column): a candidate model without an admissible row is not
grounded and is written to the customs blacklist with the reason `missing_grasps:<pipeline>` / `missing_contacts` /
`missing_handle`. Contacts that the sampler computes geometrically (small‑scale articulation base‑link brace,
attachment partner brace, touching partner brace, block pushes) need no artefact and have no entry; block design
therefore has no `:require` at all (generic cube grasp set).

Where the former `:options` went: `articulated` → `(handle …)` + customs `roles.<role>.joint` / `handle_key`;
`push` → `(contacts …)`; `attach` → derived from the `(attached …)` literal and `roles/constraints_compatible.csv` (`partner_of`,
`attach_type`); `hinge` → initialization rule (spec §5.5, §6.10), recorded in the TRO constraints; `scale` → customs
`roles.<role>.scale`; `scene <categories>` → `roles/candidates.csv` rows of kind `scene`; `group_size` / `same_model` → customs
`generation.groups.<role>` (`size`, `draw`); role kinds → `roles/candidates.csv` `kind`.

## Literal semantics

Identical to `docs/task_bddl_formats.md`: group literals expand to conjunctions over members (negated literals to
"no member"), `pose_reached` / `layout_reached` on a group are aggregate atoms, `pose_settled` atoms never gate by
themselves, every `pose_reached` atom needs a TRO target. Argument order matters and differs between the two `under`
subtasks: `(under ?target ?partner)` for under(large‑scale) (target ends under the partner), `(under ?partner ?target)`
for under(small‑scale) (the partner ends under the manipulated target).

## Grounding rules (what `ground_isg` does with a file)

1. Collect the roles from the alias slots and the literals (the wall `support` of the two world‑alignment subtasks
   appears only in the alias: the target hangs on it through the sampler's hinge, not through a literal, but the
   BDDL still declares it); look up their kind and candidates in `roles/candidates.csv` (subtask, role); `agent` and `floor` are
   implicit, `room` rows only bound the scene. A role without rows is an error, not a fallback.
2. Enumerate the admitted category combinations from `roles/constraints_compatible.csv` (only correlated pairs;
   `--test-only` keeps the combinations of `test = 1` candidates, `--representative-only` those of `representative = 1`
   candidates — there is no separate combinations sheet); apply the `:require` filters to the candidate models,
   `roles/constraints_conflicting.csv` to the pairs and sets. Fill every alias slot: `{target}`,
   `{partner}`, `{support}` → category names; `{distractors}` / `{targets}` / `{obstacles}` → `<category>_<count>`
   tokens with the count from the customs seed (`generation.groups.<role>.size`); the result is the BDDL
   `(:alias ...)` and the problem's id.
3. Declare `<synset>_k` instances for sampleable roles and group members (synset = KB synset of the category), one
   `:groups` entry per group role and per `:groups` composite, the scene roles' synsets from the bound scene objects,
   the floor, `agent.n.01_1`.
4. Copy `:init`, `:goal`, `:measure` with roles replaced by instances / group names.
5. Write the customs list entry: `roles` (category, `sampleable` flag, `require`, representative), per‑scene
   `bindings` of the scene roles and `whitelist` / `blacklist` (full icl pools minus the `:require` rejects),
   `generation` (seeded from the existing entry when the alias was grounded before, else from the per‑subtask defaults
   in `ispatialgym/sampling/generation_defaults.py`, whose values are the numbers of the spec), `constraints`;
   representative‑model test → template + instance 0.
