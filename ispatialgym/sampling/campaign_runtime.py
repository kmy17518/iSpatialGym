"""Campaign provenance and live scale checks, isolated from legacy runs.

Sampling opts in with ISPATIALGYM_CAMPAIGN_CONFIG. Replaying a campaign TRO
always checks its recorded scale, without depending on that environment flag.
Scene parking is loaded before sampling; lighting is applied before visibility
checks and restored from its recorded settings for demo collection and replay.
"""

from __future__ import annotations

import json
import math
import os
import time
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

from scripts.campaign_20261110.contracts import (
    ObjectRef, Perturbations, ScaleState, Template, ValidationKey,
    attach_tro_metadata, canonical_json, digest, file_digest, require, write_json_atomic,
)
from ispatialgym.sampling import campaign_augmentation as augmentation


def active() -> bool:
    return bool(os.environ.get("ISPATIALGYM_CAMPAIGN_CONFIG"))


@lru_cache(maxsize=32)
def _load(path: str):
    with open(path) as stream:
        config = json.load(stream)
    data = dict(config["template"])
    data["objects"] = tuple(ObjectRef(**obj) for obj in data["objects"])
    template = Template(**data)
    from scripts.campaign_20261110.binding_pools import load_pool_for_template
    from scripts.campaign_20261110.camera_context import load_camera_context
    load_pool_for_template(template, root=os.environ.get("ISGF_ROOT"))
    load_camera_context(template, root=os.environ.get("ISGF_ROOT"))
    p = config["perturbations"]
    perturbations = Perturbations({key: ScaleState(**value) for key, value in p["scales"].items()},
                                 tuple(p.get("removed_objects", [])), p.get("lighting", {}))
    key = ValidationKey(**config["validation_key"])
    require(key.template_sha256 == template.fingerprint, "validation key template mismatch")
    require(key.scale_sha256 == perturbations.scale_fingerprint, "validation key scale mismatch")
    require(config["split"] == template.split or (template.split == "eval_standard" and config["split"] == "train"),
            "campaign configuration split mismatch")
    perturbations.validate(config["split"], (obj.instance for obj in template.objects))
    _require_supported(perturbations)
    from ispatialgym.sampling.campaign_class_routes import resolve
    resolve(config)  # Release registry selects required profiles; no shell flag.
    return config, template, perturbations, key


def configuration():
    return _load(str(Path(os.environ["ISPATIALGYM_CAMPAIGN_CONFIG"]).resolve()))


def training_soft_buckets() -> bool:
    return active() and configuration()[0]["split"] == "train"


#: train_r6 (2026-11-15): the TRAIN clone of an audited canonical eval_standard template (train_gen author_canonical: the
#: eval template unchanged except split, scales fixed at the eval's). OnlineGroups gives every group configuration the
#: config split, so each eval-only recovery scope below refused the canonical template's own train groups. A train
#: template qualifies only when its eval_standard fingerprint and its scale fingerprint equal the reviewed canonical
#: selection's: another scene, room, model, scale, scene_setup or recovery option is a different template and still fails.
#: class_id -> (template_sha256 of the template with split='eval_standard', scale_sha256)
CANONICAL_TRAIN_CLONES = {
    'configuration_matching-attachment_02': ('a5af6405f14944a04d9745b47595c5ab843f3d49a04e6e17c62d7a83c7a3ac8b',
                                             '83e7e6d5e24baa9b6ed67fca084da4d971c71b0c756f6a504184d54e33c2e308'),
    'alignment-axial_04': ('53c2207c15fc6eff94e8fc136287b77e0a1cea07947fc0e94051a14358cc46be',
                           '50c06400a9500b10f9f9c9696cae9593f001dc25d6709e302743154f561d8172'),
    'inter_object_relation_satisfaction-inside_01': ('f4a70919f2edff45825665440f74e07c2b8244c230e3d42f2c742d7e171b0ee2',
                                                     '8ab89408559b39eab98796b79d484cf8a385ca4d0d7b73635c916af82ccaee22'),
    'inter_object_relation_satisfaction-touching_02': ('27295a266d23f7d63cb374e07be2564c8cf0e29b6093f8b3ddafe67bcbba7a91',
                                                       '4079d475fffca23e596a335e183cf12a28adcd96be6972f46c7b7d468b8b7002'),
    'disembedding_01': ('4cf10ec8922438c24bb1cf5ca4cd988126a840bb830d9147d17a86cd4fb03f5f',
                        '732387eadeea6c0d3aa4bda49ca507c58d28653ae9738d2f5371539ba08ac578'),
    'disembedding_10': ('8e83f44a5eb5c65eaa4b05a3a02d1d67af843307d83210ca6eb25e9324338a93',
                        '2a459eba5b7dc7dc0c7a951ab16378b5926f157a4345b00417047c38a7d9b065'),
    'inter_object_relation_satisfaction-nextto_large_scale_01': (
        'b22482e0684bbf35f3e2d20006c1a48d282e9d2b2a8a340153e7a707698afd11',
        '05ec3144e810414a6ef3e2d4f06fe052d05b7fe73a0b5bfd0858c2a179f1e775'),
}


def canonical_scope(template, perturbations=None) -> bool:
    """True for eval_standard, or for the train clone of the exact canonical template (see CANONICAL_TRAIN_CLONES).

    ``perturbations`` defaults to the active configuration's (fail closed without one).
    """
    if template.split == 'eval_standard':
        return True
    expected = CANONICAL_TRAIN_CLONES.get(template.class_id)
    if template.split != 'train' or expected is None:
        return False
    if perturbations is None:
        if not active():
            return False
        perturbations = configuration()[2]
    from dataclasses import replace
    return ((replace(template, split='eval_standard').fingerprint, perturbations.scale_fingerprint) == expected
            and all(s.mode == 'fixed' and list(s.applied) == list(s.default) for s in perturbations.scales.values()))


def inside_frame_recovery() -> bool:
    """Hash-bound ranking/retry policy for the blocked saucepot configuration.

    This changes candidate selection, never camera, IK or collision thresholds.
    It travels with the template into later sibling groups and train instances.
    """
    if not active():
        return False
    _, template, _, _ = configuration()
    value = template.scene_setup.get('recovery_options', {}).get('inside_frame_rank', False)
    require(type(value) is bool, 'inside frame recovery option must be boolean')
    if value:
        require(template.class_id == 'inter_object_relation_satisfaction-inside_04'
                and template.task == 'inter_object_relation_satisfaction-inside_small_scale',
                'inside frame recovery belongs to the audited saucepot class')
    return value


def contact_recovery(option: str) -> bool:
    """Physical contact policies, carried by the exact campaign template hash.

    Unconfigured runs retain their existing programs. These options change
    contact construction or grasp closure, never goal poses or tolerances.
    """
    classes = {
        'kielbasa_cartesian_pregrasp': ('inter_object_relation_satisfaction-ontop_small_scale_01',
                                      'inter_object_relation_satisfaction-ontop_small_scale'),
        'carton_press_slide_envelope': ('configuration_matching-articulation_open_small_scale_02',
                                       'configuration_matching-articulation_open_small_scale'),
        'held_jar_close_brace': ('configuration_matching-articulation_close_small_scale-hinged_jar',
                                 'configuration_matching-articulation_close_small_scale'),
        'milk_clutter_approach': ('disembedding_04', 'disembedding'),
        'tray_grasp_joint_margin': ('disembedding_07', 'disembedding'),
        'axial_spoon_slow_pad': ('alignment-axial_05', 'alignment-axial'),
        'touching_carry_online': ('inter_object_relation_satisfaction-touching_04',
                                  'inter_object_relation_satisfaction-touching_small_scale'),
        'touching_protected_carry': ('inter_object_relation_satisfaction-touching_04',
                                     'inter_object_relation_satisfaction-touching_small_scale'),
        'touching_place_inhand_reaim': ('inter_object_relation_satisfaction-touching_04',
                                        'inter_object_relation_satisfaction-touching_small_scale'),
        'touching_push_mass_center': ('inter_object_relation_satisfaction-touching_04',
                                     'inter_object_relation_satisfaction-touching_small_scale'),
        'touching_push_straight_retreat': ('inter_object_relation_satisfaction-touching_04',
                                         'inter_object_relation_satisfaction-touching_small_scale'),
        'cereal_brace_gap_1mm': ('inter_object_relation_satisfaction-touching_02',
                               'inter_object_relation_satisfaction-touching_small_scale'),
        'cereal_horizontal_push_proposal': ('inter_object_relation_satisfaction-touching_02',
                                           'inter_object_relation_satisfaction-touching_small_scale'),
        'shelf_board_draw': ('inter_object_relation_satisfaction-touching_03',
                             'inter_object_relation_satisfaction-touching_small_scale'),
        'shelf_other_arm': ('inter_object_relation_satisfaction-touching_03',
                            'inter_object_relation_satisfaction-touching_small_scale'),
        'shelf_target_depth_draw': ('inter_object_relation_satisfaction-touching_03',
                                   'inter_object_relation_satisfaction-touching_small_scale'),
        'marker_gentle_pinch': ('inter_object_relation_satisfaction-nextto_01',
                                'inter_object_relation_satisfaction-nextto_small_scale'),
        'close_door_prepose_margin': ('configuration_matching-articulation_close_large_scale_01',
                                     'configuration_matching-articulation_close_large_scale'),
        'cap_capture_margin': ('configuration_matching-attachment_02',
                                'configuration_matching-attachment'),
        'cap_source_stance_proposal': ('configuration_matching-attachment_02', 'configuration_matching-attachment'),
        'cap_geometry_above_pregrasp': ('configuration_matching-attachment_02', 'configuration_matching-attachment'),
        'cap_shallow_start_proposal': ('configuration_matching-attachment_02', 'configuration_matching-attachment'),
        'inside_goal_grasp_proposal': ('inter_object_relation_satisfaction-inside_01',
                                      'inter_object_relation_satisfaction-inside_small_scale'),
        'inside_goal_stance_proposal': ('inter_object_relation_satisfaction-inside_01',
                                       'inter_object_relation_satisfaction-inside_small_scale'),
        'inside_top_grasp_proposal': ('inter_object_relation_satisfaction-inside_01',
                                      'inter_object_relation_satisfaction-inside_small_scale'),
        'axial_magazine_contact_height': ('alignment-axial_04', 'alignment-axial'),
        'axial_magazine_contact_inward_1cm': ('alignment-axial_04', 'alignment-axial'),
        'axial_magazine_radial_escape': ('alignment-axial_04', 'alignment-axial'),
        'nextto_native_base_hulls': ('inter_object_relation_satisfaction-nextto_large_scale_01',
                                   'inter_object_relation_satisfaction-nextto_large_scale'),
        'cap_vertical_final_retreat': ('configuration_matching-attachment_02', 'configuration_matching-attachment'),
        'lid_vertical_final_retreat': ('configuration_matching-attachment_04', 'configuration_matching-attachment'),
    }
    require(option in classes, 'unknown campaign contact recovery option')
    if not active():
        return False
    _, template, _, _ = configuration()
    value = template.scene_setup.get('recovery_options', {}).get(option, False)
    require(type(value) is bool, 'contact recovery option must be boolean')
    if value:
        require((template.class_id, template.task) == classes[option],
                'contact recovery belongs to another audited class')
        if option in ('cereal_brace_gap_1mm', 'cereal_horizontal_push_proposal'):
            expected = {'target': ('box_of_cereal', 'iucshm', .8),
                        'partner': ('box_of_cereal', 'dkhgxn', 1.)}
            require(canonical_scope(template) and all(
                len(objects := [o for o in template.objects if o.role == role]) == 1 and
                (objects[0].category, objects[0].model) == spec[:2] and
                len(objects[0].default_scale) == 3 and
                all(abs(float(v) - spec[2]) <= 1e-7 for v in objects[0].default_scale)
                for role, spec in expected.items()),
                'cereal brace gap belongs to its exact canonical models and scales')
        if option in ('inside_goal_grasp_proposal', 'inside_goal_stance_proposal'):
            require(canonical_scope(template) and all(
                [(o.category, o.model, tuple(o.default_scale)) for o in template.objects if o.role == role] == [spec]
                for role, spec in {'target': ('brussels_sprouts', 'mbkrxe', (1., 1., 1.)),
                                   'partner': ('bowl', 'belcml', (1., 1., 1.))}.items()),
                'inside goal grasp belongs to its exact canonical target and bowl')
        if option == 'held_jar_close_brace':
            require(template.split == 'eval_category' and
                    [(o.category, o.model, tuple(o.default_scale)) for o in template.objects if o.role == 'target'] ==
                    [('hinged_jar', 'jnjtrl', (1.0, 1.0, 1.0))],
                    'held jar brace belongs to its exact ready default-scale category model')
        if option in ('cap_capture_margin', 'cap_vertical_final_retreat', 'cap_source_stance_proposal',
                      'cap_geometry_above_pregrasp', 'cap_shallow_start_proposal'):
            require(canonical_scope(template) and
                    [(o.category, o.model, tuple(o.default_scale)) for o in template.objects if o.role == 'target'] ==
                    [('cap', 'xxqyrt', (1.0, 1.0, 1.0))],
                    'cap capture margin belongs to its exact default-scale canonical model')
        if option in ('cap_geometry_above_pregrasp', 'cap_shallow_start_proposal'):
            require([(o.category, o.model, tuple(o.default_scale)) for o in template.objects if o.role == 'partner'] ==
                    [('lemonade_bottle', 'yprkek', (1.0, 1.0, 1.0))],
                    'cap pregrasp geometry belongs to its exact canonical bottle')
        if option in ('axial_magazine_contact_height', 'axial_magazine_contact_inward_1cm', 'axial_magazine_radial_escape'):
            require(canonical_scope(template) and
                    [(o.category, o.model, tuple(o.default_scale)) for o in template.objects if o.role == 'target'] ==
                    [('magazine', 'tfovwy', (1.0, 1.0, 1.0))],
                    'axial contact height belongs to its exact canonical magazine')
        if option == 'lid_vertical_final_retreat':
            from scripts.campaign_20261110.lid_pipeline_recovery import authorized
            require((template.split == 'eval_standard' or authorized(template)) and
                    [(o.category, o.model) for o in template.objects if o.role == 'target'] == [('lid', 'pbmsji')] and
                    all(len(o.default_scale) == 3 and all(abs(float(v)-.65) <= 1e-7 for v in o.default_scale)
                        for o in template.objects if o.role == 'target'),
                    'vertical lid retreat belongs to its exact canonical model and scale')
    return value


def touching_push_stroke_m():
    """Exact-template opt-in to the existing checked stroke/re-aim program.

    The successful touching04 reference used 3cm strokes along the full original
    push. This changes execution segmentation, not its goal or tolerances.
    """
    if not active():
        return None
    _, template, _, _ = configuration()
    value = template.scene_setup.get('recovery_options', {}).get('touching_push_stroke_m')
    if value is None:
        return None
    require(type(value) in (int, float) and math.isfinite(value) and 0 < value <= .1,
            'campaign push stroke must be finite and within (0, 0.1] meters')
    require((template.class_id, template.task) == ('inter_object_relation_satisfaction-touching_04',
            'inter_object_relation_satisfaction-touching_small_scale'), 'push stroke recovery belongs to the audited spray-bottle class')
    return float(value)


def touching_push_yaw_gain():
    """Audited spray-bottle feedback gain, recorded in each native stroke plan."""
    if not active():
        return None
    _, template, _, _ = configuration()
    value = template.scene_setup.get('recovery_options', {}).get('touching_push_yaw_gain_m_per_rad')
    if value is None:
        return None
    require(type(value) in (int, float) and math.isfinite(value) and 0 < value <= .15,
            'campaign yaw gain must be finite and within (0, 0.15] meters per radian')
    require((template.class_id, template.task) == ('inter_object_relation_satisfaction-touching_04',
            'inter_object_relation_satisfaction-touching_small_scale'),
            'yaw gain recovery belongs to the audited spray-bottle class')
    require(touching_push_stroke_m() is not None, 'yaw gain requires a configured stroke plan')
    return float(value)


def choose_soft_bucket(rng, buckets, counts):
    """Favor underfilled train buckets while keeping every bucket eligible.

    The sampler still checks the physical bounds of its chosen draw. Unlike eval,
    an already-filled bucket stays available when another bucket has low yield.
    """
    require(bool(buckets), "training bucket list is empty")
    maxima = [float(b["max_m"]) for b in buckets]
    targets = [max(0, int(b.get("n", 0))) for b in buckets]
    require(all(math.isfinite(v) and v > 0 for v in maxima), "invalid bucket bounds")
    deficits = [max(0, n - int(counts.get(hi, 0))) for n, hi in zip(targets, maxima)]
    # Reserve 20% of draws for uniform exploration even for large outstanding
    # quotas. A fixed pseudocount would vanish beside a 10,000-instance deficit.
    total = sum(deficits)
    probabilities = ([0.2 / len(buckets) + 0.8 * deficit / total for deficit in deficits]
                     if total else [1.0 / len(buckets)] * len(buckets))
    return int(rng.choice(len(buckets), p=probabilities))


def _require_supported(perturbations):
    augmentation.validate_lighting(perturbations.lighting)


def scene_removals(scene, stable):
    _, template, perturbations, _ = configuration()
    require(scene == template.scene, "campaign template scene differs from requested world")
    names = template.scene_setup.get("inst_to_name", {}).values()
    return augmentation.checked_removals(stable, perturbations.removed_objects,
                                         [name for name in names if isinstance(name, str)])


def prepare(sampler, sim):
    """Run after bindings exist, before any template geometry/visibility checks."""
    _, template, perturbations, _ = configuration()
    from scripts.campaign_20261110.camera_context import verify_live_context
    verify_live_context(template, sampler.world.scene)
    verify_scales(sampler.objects, perturbations.scales)
    relevant = {obj.name for obj in sampler.objects.values()}
    moved = sampler.world.scene_variant.get("moved_objects") or {}
    parking = augmentation.verify_parking(sampler.world.scene, perturbations.removed_objects, moved, relevant)
    lighting = augmentation.apply_lighting(sampler.world.scene, sim, perturbations.lighting)
    sampler._campaign_augmentation = {"moved_objects": {name: moved[name] for name in parking}, "lighting": lighting}


def verify_prepared(sampler, perturbations):
    state = getattr(sampler, "_campaign_augmentation", {})
    require(not perturbations.removed_objects or bool(state.get("moved_objects")), "scene runtime adapter was not applied")
    require(not perturbations.lighting or bool(state.get("lighting")), "lighting runtime adapter was not applied")
    if perturbations.removed_objects:
        augmentation.verify_parking(sampler.world.scene, perturbations.removed_objects,
                                    state["moved_objects"], [obj.name for obj in sampler.objects.values()])
    if perturbations.lighting:
        import omnigibson as og

        augmentation.verify_lighting(sampler.world.scene, og.sim, state["lighting"])
    return state


def _vector(value):
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (float, int)):
        value = [value] * 3
    result = [float(v) for v in value]
    require(len(result) == 3 and all(math.isfinite(v) and v > 0 for v in result), "invalid live object scale")
    return result


def verify_scales(objects, scales):
    observed = {}
    for name, state in scales.items():
        obj = objects.get(name)
        require(obj is not None, f"campaign task object absent: {name}")
        actual = _vector(obj.scale)
        expected = state.applied if isinstance(state, ScaleState) else state["applied"]
        require(all(abs(a - float(b)) <= 1e-6 * max(1.0, abs(float(b))) for a, b in zip(actual, expected)),
                f"campaign scale mismatch for {name}: expected {expected}, loaded {actual}")
        observed[name] = actual
    return observed


def annotate(sampler, tro, draft, seed):
    config, template, perturbations, key = configuration()
    require(sampler.scene == template.scene, "campaign template scene differs from sampler scene")
    for ref in template.objects:
        obj = sampler.objects.get(ref.instance)
        require(obj is not None, f"campaign binding missing: {ref.instance}")
        require(obj.category == ref.category and obj.model == ref.model,
                f"campaign asset identity mismatch: {ref.instance}")
    observed = verify_scales(sampler.objects, perturbations.scales)
    augmentation_state = verify_prepared(sampler, perturbations)
    from scripts.campaign_20261110.camera_context import verify_live_context
    context_objects = verify_live_context(template, sampler.world.scene)
    expected_bddl = template.scene_setup.get("runtime_bddl_sha256", template.scene_setup.get("source_bddl_sha256"))
    if expected_bddl:
        require(sampler.template_meta.get("bddl_sha256") == expected_bddl, "campaign BDDL differs from its pinned definition")
    group = draft.summary.get("campaign_goal_group") or {}
    if group:
        require(group.get("campaign_id") == config["campaign_id"], "goal plan belongs to another campaign")
        require(group.get("split") == config["split"], "goal plan belongs to another split")
        require(group.get("contract_template_sha256") == template.fingerprint, "goal plan belongs to another template contract")
    result = attach_tro_metadata(tro, campaign_id=config["campaign_id"], split=config["split"],
                                 template=template, created_at=time.time(), seed=seed, perturbations=perturbations,
                                 validation_key=key, goal_group_id=group.get("group_id"))
    from ispatialgym.sampling.campaign_bucket_policy import POLICY as BUCKET_POLICY
    result["ispatialgym"]["campaign"]["runtime"] = {
        "bucket_policy": BUCKET_POLICY,
        "observed_scales": observed, "goal_group": group, "planned_goal_poses": _plain(draft.goal_poses),
        "planned_start_poses": _plain(draft.object_poses),
        "planned_start_joints": _plain(draft.object_joints), "planned_goal_joints": _plain(draft.goal_joints),
        "bound_object_poses": _plain({ref.instance: sampler.objects[ref.instance].get_position_orientation() for ref in template.objects}),
        "joint_names": {ref.instance: list(getattr(sampler.objects[ref.instance], "joints", {})) for ref in template.objects},
        "augmentation": augmentation_state,
        "purpose": config.get("purpose", "candidate"),
        "adapters": {"scale": "template_scene_with_live_verification",
                     "scene": "scene_variant_parking" if perturbations.removed_objects else "none",
                     "lighting": "usd_lights_v1" if perturbations.lighting else "none"},
    }
    from ispatialgym.sampling.campaign_class_routes import record
    online = getattr(sampler, '_online_groups', None)
    private_goal_proposal = online is not None and not online.state['current'].get('plan')
    # A private first draw may establish the goal required by a class recipe.
    # It cannot become a member unless that recipe actually executed. The
    # online publication hook enforces this before collectors see the summary.
    route = None if private_goal_proposal else record(config, getattr(sampler, '_campaign_attachment_active', None) or
                   getattr(sampler, '_campaign_pair_active', None),
                   getattr(sampler, '_campaign_class_route_trace', []))
    if route is not None:
        result["ispatialgym"]["campaign"]["runtime"]["class_route"] = route
    if context_objects:
        result["ispatialgym"]["campaign"]["runtime"]["camera_context"] = {
            "scene": template.scene, "room": template.room,
            "scene_sha256": template.scene_setup["camera_context"]["scene"]["sha256"],
            "observed_objects": context_objects,
        }
    return result


def verify_loaded(meta, physical, *, scene=None, sim=None):
    """Check a loaded campaign instance before any recorded action is executed."""
    require(meta.get("schema") == "isg-campaign-v1", "unknown campaign TRO schema")
    p = meta["perturbations"]
    perturbations = Perturbations({name: ScaleState(**value) for name, value in p["scales"].items()},
                                 tuple(p.get("removed_objects", [])), p.get("lighting", {}))
    perturbations.validate(meta["split"], physical)
    _require_supported(perturbations)
    require(meta["validation_key"]["scale_sha256"] == perturbations.scale_fingerprint,
            "TRO validation key scale mismatch")
    state = meta.get("runtime", {}).get("augmentation", {})
    if perturbations.removed_objects:
        require(scene is not None, "scene runtime adapter needs a loaded scene")
        augmentation.verify_parking(scene, perturbations.removed_objects, state.get("moved_objects", {}),
                                    [obj.name for obj in physical.values()])
    # policy eval 2026-11-15 (ISPATIALGYM_EVAL_IGNORE_COLLECTION_LIGHTING=1): the eval set's goal images and training-format renders
    # come from the batched replay, which ignores the collection-time lighting (eval units: --lighting-pert off), so an eval
    # episode is lit as they were - the stock lights - even when the instance's TRO records a collection lighting draw
    ignore_lighting = bool(perturbations.lighting) and os.environ.get("ISPATIALGYM_EVAL_IGNORE_COLLECTION_LIGHTING") == "1"
    if perturbations.lighting and not ignore_lighting:
        require(scene is not None and sim is not None, "lighting runtime adapter needs a loaded scene and simulator")
        require(bool(state.get("lighting")), "recorded lighting runtime adapter state missing")
        augmentation.apply_lighting(scene, sim, perturbations.lighting, state["lighting"])
    elif scene is not None and sim is not None:
        augmentation.apply_lighting(scene, sim, {})
    context = meta.get("scene_setup", {}).get("camera_context")
    context_observed = {}
    if context is not None:
        from scripts.campaign_20261110.camera_context import verify_live_context
        recorded = meta.get("runtime", {}).get("camera_context")
        require(scene is not None and isinstance(recorded, dict), "camera context replay evidence missing")
        require(recorded.get("scene_sha256") == context["scene"]["sha256"], "camera context replay scene hash changed")
        template = SimpleNamespace(task=meta["task"], split=meta["split"], scene=recorded["scene"],
            room=recorded["room"], scene_setup=meta["scene_setup"],
            objects=tuple(SimpleNamespace(instance=name) for name in physical))
        context_observed = verify_live_context(template, scene)
        require(context_observed == recorded.get("observed_objects"), "camera context replay differs from sampling")
    result = {"observed_scales": verify_scales(physical, perturbations.scales), "scale_verified": True,
            "camera_context": context_observed,
            "scene_augmentation": bool(perturbations.removed_objects), "lighting_augmentation": bool(perturbations.lighting)}
    if ignore_lighting:
        result["collection_lighting_ignored"] = True
    from ispatialgym.sampling.campaign_class_routes import verify_recorded
    route = verify_recorded(meta, physical=physical)
    if route is not None:
        result['class_route'] = route
    if os.environ.get('ISPATIALGYM_CAMPAIGN_RESTORATION_PROBE') == '1':
        result['native_observation'] = restoration_observation(meta, scene, sim)
    return result


_RESTORATION_PROBE_SESSION = None
_RESTORATION_PROBE_SEQUENCE = 0


def restoration_observation(meta, scene, sim):
    """Measure actual loaded state without modifying physics or release status."""
    global _RESTORATION_PROBE_SESSION, _RESTORATION_PROBE_SEQUENCE
    require(scene is not None and sim is not None, 'native restoration probe needs a loaded simulator')
    if _RESTORATION_PROBE_SESSION is None:
        import socket
        _RESTORATION_PROBE_SESSION = f'{socket.gethostname()}:{os.getpid()}:{time.time_ns()}'
    _RESTORATION_PROBE_SEQUENCE += 1
    objects = {}
    for obj in scene.objects:
        position, orientation = obj.get_position_orientation(frame='scene')
        observed = {'category': obj.category, 'model': getattr(obj, 'model', None),
            'scale': _vector(obj.scale), 'position': _plain(position), 'orientation': _plain(orientation),
            'fixed_base': bool(obj.fixed_base)}
        observed['joint_positions'] = _plain(obj.get_joint_positions()) if getattr(obj, 'n_joints', 0) else []
        observed['joint_velocities'] = _plain(obj.get_joint_velocities()) if getattr(obj, 'n_joints', 0) else []
        # EntityPrim exposes velocity methods even when its root is a
        # RigidKinematicPrim, which deliberately has no dynamic velocity API.
        # Record that native distinction instead of inventing zero velocities.
        kinematic = bool(getattr(obj, 'kinematic_only', False))
        observed['kinematic_only'] = kinematic
        observed['root_prim_type'] = type(getattr(obj, 'root_link', obj)).__name__
        for name in ('linear_velocity', 'angular_velocity'):
            if kinematic:
                require(observed['root_prim_type'] == 'RigidKinematicPrim',
                        'kinematic restoration observation has an unexpected native root')
                observed[name] = None
            else:
                getter = getattr(obj, 'get_' + name, None)
                require(callable(getter), 'dynamic restoration velocity observation unavailable: ' + obj.name)
                observed[name] = _plain(getter())
                require(observed[name] is not None, 'dynamic restoration velocity observation missing: ' + obj.name)
        objects[obj.name] = observed
    loaded = {'objects': objects,
        'lights': {name: float(attr.Get()) for name, attr in augmentation._lights(scene).items()},
        'skybox': {'intensity': float(sim.skybox.intensity), 'color': _plain(sim.skybox.color)}}
    result = {'schema': 'isg-native-restoration-observation-v1', 'process_session': _RESTORATION_PROBE_SESSION,
        'sequence': _RESTORATION_PROBE_SEQUENCE, 'scene_runtime_id': id(scene), 'sim_runtime_id': id(sim),
        'simulator_module': type(sim).__module__, 'scheduler_job_id': os.environ.get('SLURM_JOB_ID'),
        'validation_key': meta['validation_key'], 'perturbations_sha256': digest(meta['perturbations']),
        'observed_at': time.time(), 'loaded_state': loaded, 'loaded_state_sha256': digest(loaded)}
    if os.environ.get('ISPATIALGYM_CAMPAIGN_RESTORATION_CONTEXT') == '1':
        context = {'scope': 'Read-only sleep and existing contact-cache observations; no solver reset or physics step.',
                   'dynamic_roots': {}, 'physics_steps_added': 0}
        for obj in scene.objects:
            root = getattr(obj, 'root_link', None)
            if obj.fixed_base or getattr(obj, 'kinematic_only', False) or getattr(obj, 'n_joints', 0):
                continue
            if type(root).__name__ != 'RigidDynamicPrim':
                continue
            row = {}
            for field in ('is_asleep', 'mass', 'sleep_threshold',
                          'solver_position_iteration_count', 'solver_velocity_iteration_count'):
                try: row[field] = _plain(getattr(root, field))
                except Exception as error: row[field] = {'unavailable': type(error).__name__ + ': ' + str(error)}
            context['dynamic_roots'][obj.name] = row
        try:
            # Inspect already computed Python caches without refreshing them or
            # requesting a new native contact query.
            from omnigibson.utils.usd_utils import RigidContactAPI
            scene_idx = scene.idx
            pairs = {}
            for key in ('_CONTACT_MATRIX', '_CURRENT_CONTACT_MATRIX'):
                matrix = getattr(RigidContactAPI, key).get(scene_idx)
                if matrix is None:
                    pairs[key] = None
                else:
                    rows = RigidContactAPI._ROW_IDX_TO_PATH[scene_idx]
                    cols = RigidContactAPI._COL_IDX_TO_PATH[scene_idx]
                    pairs[key] = sorted([str(rows[i]), str(cols[j])] for i,j in matrix.nonzero().tolist())
            context['cached_contacts'] = pairs
            context['pending_contact_steps'] = len(RigidContactAPI._PENDING_IMPULSES.get(scene_idx, []))
        except Exception as error:
            context['contact_cache_unavailable'] = type(error).__name__ + ': ' + str(error)
        result['native_dynamics_context'] = context
        result['native_dynamics_context_sha256'] = digest(context)
    return result


def _plain(value):
    if hasattr(value, "tolist"):
        return _plain(value.tolist())
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(v) for v in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    return str(value)


def _append_attempt_event(sampler, record):
    from scripts.campaign_20261110.storage import atomic_directory_lock

    path = Path(sampler.ctx.out_dir) / "campaign_attempts.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_directory_lock(path.with_suffix(".lockdir")):
        with path.open("a") as stream:
            stream.write(canonical_json(record) + "\n")
            stream.flush()
            os.fsync(stream.fileno())


def attempt_started(sampler, *, index, attempt, seed):
    """Durably mark entry to a native draw after activation and deadline checks."""
    require(getattr(sampler, "_campaign_attempt_event", None) is None, "campaign attempt already started")
    config, template, _, key = configuration()
    config_path = Path(os.environ["ISPATIALGYM_CAMPAIGN_CONFIG"]).resolve()
    require(json.loads(config_path.read_text()) == config, "campaign config changed before native attempt")
    group_id = plan_sha = None
    plan_path = os.environ.get("ISPATIALGYM_CAMPAIGN_GOAL_PLAN")
    queued = bool(os.environ.get("ISPATIALGYM_CAMPAIGN_GOAL_QUEUE"))
    state = getattr(sampler, "_campaign_goal_queue", None)
    if queued:
        require(state is not None and not state.get("exhausted"), "native attempt requires an activated queue entry")
        entry = state["entries"][state["index"]]
        require(plan_path and Path(plan_path).resolve() == Path(entry["path"]).resolve(), "active queue plan differs")
    if plan_path:
        plan = json.loads(Path(plan_path).read_text())
        require(plan.get("campaign_id") == config["campaign_id"] and plan.get("split") == config["split"]
                and plan.get("task") == template.task and plan.get("contract_template_sha256") == key.template_sha256,
                "native attempt goal plan differs from campaign config")
        group_id, plan_sha = plan.get("group_id"), file_digest(plan_path)
        require(isinstance(group_id, str) and bool(group_id), "native attempt plan lacks group identity")
        if queued:
            require(group_id == entry["group_id"] and plan_sha == entry["sha256"], "active queue plan changed")
    identity = {"output_dir": str(Path(sampler.ctx.out_dir).resolve()), "group_id": group_id,
                "index": int(index), "attempt": int(attempt), "seed": int(seed),
                "config_sha256": file_digest(config_path), "plan_sha256": plan_sha}
    record = {"schema": "isg-campaign-attempt-v1", "event": "started", "created_at": time.time(),
              "attempt_id": digest(identity), "campaign_id": config["campaign_id"], "class_id": template.class_id,
              "task": template.task, "split": config["split"], "goal_group_id": group_id,
              "index": int(index), "attempt": int(attempt), "seed": int(seed),
              "config_path": str(config_path), "config_sha256": identity["config_sha256"],
              "plan_sha256": plan_sha, "validation_key": config["validation_key"], "phase": "sample"}
    _append_attempt_event(sampler, record)
    sampler._campaign_attempt_event = record


def attempt_finished(sampler, *, outcome, phase, reason=None, tro_path=None):
    started = getattr(sampler, "_campaign_attempt_event", None)
    if started is None:
        return  # Reset failures and pre-draw queue transitions did not enter a native draw.
    require(outcome in ("written", "rejected"), "invalid native attempt outcome")
    if outcome == "written" and phase == "published":
        published = getattr(sampler, "campaign_instance_published", None)
        if published is not None:
            published(tro_path)
    record = {**started, "event": "finished", "created_at": time.time(), "outcome": outcome,
              "phase": phase, "reason": reason, "tro_path": str(Path(tro_path).resolve()) if tro_path else None}
    _append_attempt_event(sampler, record)
    sampler._campaign_attempt_event = None


def record_failure(sampler, draft, *, index, attempt, seed, phase, error):
    config, template, perturbations, _ = configuration()
    group = (draft.summary.get("campaign_goal_group") or {}) if draft is not None else {}
    proposed = getattr(sampler, "_campaign_latest_proposal", None) or {}
    entered = getattr(sampler, "_campaign_attempt_event", None) or {}
    record = {
        "schema": "isg-campaign-failure-v1", "campaign_id": config["campaign_id"], "created_at": time.time(),
        "template_id": template.template_id, "class_id": template.class_id, "task": template.task,
        "index": index, "attempt": attempt, "seed": seed, "failure_kind": "sampling", "phase": phase,
        "reason": error.reason, "detail": str(error), "diagnostics": _plain(getattr(error, "details", {})),
        "scale_sha256": perturbations.scale_fingerprint,
        "goal_group_id": group.get("group_id") or proposed.get("group_id") or entered.get("goal_group_id"),
        "native_attempt_id": entered.get("attempt_id"),
        "proposal_available": draft is not None or bool(proposed),
        "start_poses": _plain(draft.object_poses if draft is not None else proposed.get("start_poses")),
        "goal_poses": _plain(draft.goal_poses if draft is not None else proposed.get("goal_poses")),
        "start_joints": _plain(draft.object_joints if draft is not None else proposed.get("start_joints")),
        "goal_joints": _plain(draft.goal_joints if draft is not None else proposed.get("goal_joints")),
        "base_pose2d": _plain(draft.base_pose2d if draft is not None else proposed.get("base_pose2d")),
    }
    path = Path(sampler.ctx.out_dir) / "campaign_sampling_failures.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    from scripts.campaign_20261110.storage import atomic_directory_lock

    with atomic_directory_lock(path.with_suffix(".lockdir")):
        with path.open("a") as stream:
            stream.write(canonical_json(record) + "\n")
            stream.flush()
            os.fsync(stream.fileno())


def record_execution_failure(tro_path, output, result):
    """Keep the exact failed proposal separately from sampling rejection logs."""
    if result.get("success") is True:
        return
    path = Path(tro_path)
    with path.open() as stream:
        tro = json.load(stream)
    meta = tro.get("ispatialgym", {})
    campaign = meta.get("campaign")
    if not campaign:
        return
    write_json_atomic(Path(output) / "campaign_execution_failure.json", {
        "schema": "isg-campaign-failure-v1", "created_at": time.time(),
        "campaign_id": campaign["campaign_id"], "template_id": campaign["template_id"],
        "instance_id": meta["instance_id"], "goal_group_id": campaign.get("goal_group_id"),
        "failure_kind": "replay" if result.get("mode") == "replay" else "demo_collection",
        "phase": result.get("status"), "reason": result.get("reason"), "result": _plain(result),
        "tro_path": str(path.resolve()), "tro_sha256": file_digest(path), "proposal_tro": tro,
    })
