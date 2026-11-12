"""Opt-in, campaign-scoped persistent task objects for bounded native canaries.

This is a physical pool, separate from conceptual binding/model inventories.
The SimWorld adapter supplies actual preload, reset, parking and export hooks.
No manifest or passing pool test releases training production.
"""
from __future__ import annotations
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import time

ENV = 'ISPATIALGYM_CAMPAIGN_MODEL_POOL_MANIFEST'
SCHEMA = 'isg-native-model-pool-v1'
SELECTED_MODE = 'selected_training_configuration'


def require(condition, message):
    if not condition:
        raise ValueError('model pool: ' + message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def signature(category, model, dataset, scale, fixed_base=False, extra_abilities=None):
    require(isinstance(scale, (list, tuple)) and len(scale) == 3
            and all(type(x) in (int, float) and math.isfinite(x) and x > 0 for x in scale),
            'an explicit exact applied scale is required')
    return {'category': category, 'model': model, 'dataset': dataset, 'scale': [float(x) for x in scale],
            'fixed_base': bool(fixed_base), 'extra_abilities': copy.deepcopy(extra_abilities or {})}


def check_policy(value):
    """Recheck the active reservation, including inactive physical members."""
    from scripts.campaign_20261110.reservation_policy import active_policy
    active = active_policy(value['campaign_root'])
    require(active is not None, 'active reservation policy is unavailable')
    pointer, reservations, hashes = active
    require(value['reservation_policy_sha256'] == pointer['policy_sha256'], 'reservation policy changed')
    held_rooms = {(r['scene'], r['room']) for r in reservations['category_holdout_rooms']}
    require(not {(value['scene'], r) for r in value['rooms']} & held_rooms, 'held-out room in training pool')
    for row in value['entries'] + value.get('context_assets', []):
        if row.get('semantic_kind') in ('wall', 'floor'):
            continue
        require(row['category']+'/'+row['model'] not in reservations['train_exclude_models'], 'held-out model in pool')
        require(row['asset_sha256'] not in hashes['reserved_asset_hashes'], 'held-out asset bytes in pool')
        require(row['category'] not in reservations['category_holdout_categories']
                or row.get('semantic_kind') in ('tray', 'board'), 'held-out category in pool')


def file_reference(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def native_runtime_reference():
    return file_reference(Path(__file__).resolve().parents[2]/'.snapshot_commit')


def authorize_execution(path, *, environment, start, count):
    """Check the current gate once at native entry, never during structural audits.

    The selected manifest cannot grant production authority. Its exact bytes
    are bound either by a bounded canary authorization or by the fully checked
    applicable-pool release and diversity ticket returned by the current gate.
    """
    path = Path(path).resolve(); value = json.loads(path.read_text())
    if value.get('mode') != SELECTED_MODE:
        return None
    require(type(start) is int and start > 0 and type(count) is int and count > 0,
            'selected execution needs an explicit native index range')
    root = Path(value['campaign_root']).resolve()
    require(path.is_relative_to(root), 'selected manifest outside campaign')
    config_path = Path(environment.get('ISPATIALGYM_CAMPAIGN_CONFIG', '')).resolve()
    require(config_path.is_relative_to(root) and config_path.is_file(), 'selected configuration is unavailable')
    config = json.loads(config_path.read_text()); config_ref = file_reference(config_path)
    require(config.get('campaign_id') == value.get('campaign_id') == 'campaign_20261110'
            and config.get('purpose') == 'training_production', 'selected execution needs the exact production configuration')
    runtime = native_runtime_reference()
    require(config['validation_key']['code_sha256'] == runtime['sha256'], 'selected native runtime differs')
    from scripts.campaign_20261110.production_gate import check_environment
    decision = check_environment(root, environment, kind='collection', start=start, count=count)
    require(decision.get('allowed') is True and decision.get('config_sha256') == config_ref['sha256'],
            'current selected execution gate denied or configuration differs')
    reference = file_reference(path)
    if decision.get('category') == 'validation_canary':
        registered = decision['authorization']
        authorization_path = Path(registered['authorization_path']).resolve()
        require(authorization_path.is_relative_to(root) and
                file_reference(authorization_path)['sha256'] == registered['authorization_sha256'],
                'selected canary authorization changed')
        authorization = json.loads(authorization_path.read_text())
        from scripts.campaign_20261110.training_canary_scope import validate_scope
        validate_scope(config, authorization)
        require(authorization.get('config_sha256') == config_ref['sha256']
                and authorization.get('native_runtime_sha256') == runtime['sha256']
                and (authorization.get('start'), authorization.get('count')) == (start, count)
                and authorization.get('pool_manifest') == reference,
                'selected canary pool, runtime, configuration or range differs')
    elif decision.get('category') == 'released_training':
        release = decision['production_release']
        require(release.get('production_authorization') is True and
                release.get('applicable_pool', {}).get('pool_manifest') == reference,
                'selected production pool differs from current exact release')
    else:
        raise ValueError('model pool: selected execution has no bounded canary or scoped production authority')
    return {'schema': 'isg-model-pool-execution-context-v1', 'campaign_id': value['campaign_id'],
            'manifest': reference, 'configuration': config_ref, 'native_runtime': runtime,
            'start': start, 'count': count, 'observed_at': time.time(), 'gate': decision}


def execution_from_args(path, args, environment):
    """Bind one actual native CLI range, without guessing default batch sizes."""
    if not path or json.loads(Path(path).read_text()).get('mode') != SELECTED_MODE:
        return None
    require(len(args.variant) == 1 and args.scene and type(args.templates) is int and args.templates == 1
            and type(args.instances_per_template) is int and args.instances_per_template > 0,
            'selected execution needs one explicit variant, scene, template and instance bound')
    return authorize_execution(path, environment=environment, start=args.start_index,
                               count=args.instances_per_template)


def validate_execution_context(value, reference, context):
    if value.get('mode') != SELECTED_MODE:
        return
    require(context and context.get('schema') == 'isg-model-pool-execution-context-v1'
            and context.get('campaign_id') == value['campaign_id'] and context.get('manifest') == reference,
            'selected activation needs its exact checked execution context')
    require(reference == file_reference(reference['path']) and
            value == json.loads(Path(reference['path']).read_text()), 'selected manifest changed before activation')
    require(context['configuration'] == file_reference(os.environ.get('ISPATIALGYM_CAMPAIGN_CONFIG', ''))
            and context['native_runtime'] == native_runtime_reference(), 'selected execution context changed')
    decision = context['gate']
    require(decision.get('allowed') is True and decision.get('category') in ('validation_canary', 'released_training')
            and decision.get('config_sha256') == context['configuration']['sha256'],
            'selected execution context has no current gate decision')
    require(type(context.get('start')) is int and context['start'] > 0
            and type(context.get('count')) is int and context['count'] > 0,
            'selected execution context range is missing')
    if decision['category'] == 'validation_canary':
        registered = decision['authorization']
        require((registered.get('start'), registered.get('count')) == (context['start'], context['count']),
                'selected execution context range differs from the gate')


def read_manifest(path, *, scene, rooms, fixed_bindings, execution_context=None):
    path = Path(path).resolve(); value = json.loads(path.read_text())
    require(value.get('schema') == SCHEMA and value.get('campaign_id') == 'campaign_20261110', 'wrong schema/campaign')
    require(value.get('benchmark_credit') == 0 and value.get('production_released') is False
            and (value.get('mode') == 'bounded_validation' or
                 (value.get('mode') == SELECTED_MODE and value.get('production_authorization') is False)),
            'pool manifests cannot grant production or benchmark authority')
    require(value.get('scene') == scene and sorted(value.get('rooms', [])) == sorted(rooms or []), 'world scope mismatch')
    require(value.get('fixed_bindings') == fixed_bindings, 'fixed support/partner scope mismatch')
    require(value.get('selected_scope') and value.get('batch_id') and value.get('selection_proof'),
            'selected scope, explicit batch identity and selection proof are required')
    proof = value['selection_proof']; source = Path(proof['path']).resolve()
    require(source.is_file() and hashlib.sha256(source.read_bytes()).hexdigest() == proof['sha256'], 'selection proof changed')
    selected = json.loads(source.read_text())
    require(selected.get('schema') == 'isg-model-pool-selection-v1'
            and selected.get('campaign_id') == value['campaign_id'], 'invalid selection evidence')
    require(selected.get('world_scope') == {k: value[k] for k in ('scene', 'rooms', 'fixed_bindings')},
            'selection world scope differs')
    require(selected.get('selected_entries') == value.get('entries'), 'pool omits or changes selected batch entries')
    entries = value.get('entries', [])
    require(entries and isinstance(value.get('max_objects'), int) and 1 <= value['max_objects'] <= 128,
            'missing explicit preload bound')
    names = set(); keys = set(); count = 0
    for row in entries:
        key = signature(**{k: row[k] for k in ('category', 'model', 'dataset', 'scale')},
                        fixed_base=row.get('fixed_base', False), extra_abilities=row.get('extra_abilities'))
        require(digest(key) not in keys, 'duplicate physical input; use the copies field')
        keys.add(digest(key))
        require(row.get('ready') is True and row.get('readiness_provenance') and row.get('consumers'),
                'ready provenance and consumer mapping are required')
        require(not row.get('accept_default_scale') or row.get('default_scale_provenance'), 'default scale must be measured')
        require(type(row.get('copies')) is int and row['copies'] > 0, 'invalid copies')
        asset = Path(row['asset_path']).resolve()
        require(asset.is_file() and hashlib.sha256(asset.read_bytes()).hexdigest() == row['asset_sha256'], 'asset changed')
        for i in range(row['copies']):
            name = f'campaign_pool_{digest(key)[:16]}_{i}'
            require(name not in names, 'physical name collision'); names.add(name)
        count += row['copies']
    require(count <= value['max_objects'], 'selected batch exceeds explicit bound; do not silently truncate')
    check_policy(value)
    reference = file_reference(path)
    if execution_context is not None:
        validate_execution_context(value, reference, execution_context)
    return value, reference


def active_scene(scene, loaded_names, active_names):
    """Remove only inactive pool objects; fail if any cross-reference survives."""
    loaded, active = set(loaded_names), set(active_names)
    require(active <= loaded, 'export active inventory is not in the pool')
    inactive = loaded - active
    out = copy.deepcopy(scene)
    initial = out['objects_info']['init_info']; registry = out['state']['registry']['object_registry']
    require(loaded <= initial.keys() and loaded <= registry.keys(), 'native scene omitted loaded pool objects')
    for name in inactive:
        del initial[name]; del registry[name]
    def references(value):
        if isinstance(value, str):
            return any(name in value for name in inactive)
        if isinstance(value, dict):
            return any(references(k) or references(v) for k, v in value.items())
        if isinstance(value, (list, tuple)):
            return any(references(x) for x in value)
        return False
    require(not references(out), 'inactive object remains referenced in exported scene')
    return out


def native_properties(obj):
    """Physics flags omitted by OG's object-state JSON, measured per link."""
    import omnigibson.lazy as lazy
    links = {}
    for name, link in obj.links.items():
        prim = link.prim
        gravity = None
        if prim.HasAPI(lazy.pxr.UsdPhysics.RigidBodyAPI):
            gravity = bool(lazy.pxr.PhysxSchema.PhysxRigidBodyAPI(prim).GetDisableGravityAttr().Get()) \
                if prim.HasAPI(lazy.pxr.PhysxSchema.PhysxRigidBodyAPI) else False
        links[name] = {'gravity_disabled':gravity,
            'collisions':{str(api.GetPrim().GetPath().MakeRelativePath(obj.prim_path)):bool(api.GetCollisionEnabledAttr().Get())
                          for api in getattr(link,'_collision_apis',[])}}
    return {'category':obj.category,'model':obj.model,'scale':obj.scale.tolist(),
            'visible':bool(obj.visible),'links':links}


def restore_export(scene, sim, template):
    """Restore an explicit pool export after scene construction, before use.

This template-load hook restores the recorded initial state. A TRO loader must
then install its own instance poses/holds; this function must not overwrite a
subsequently loaded TRO's per-instance state.
"""
    meta=template.get('metadata',{}).get('ispatialgym',{}).get('model_pool')
    if not meta or not meta.get('active_properties'):return None
    from omnigibson.utils.python_utils import recursively_convert_to_torch
    import omnigibson.lazy as lazy
    import torch as th
    require(set(meta['active_properties'])==set(meta['active_names']), 'pool property inventory differs')
    restored={}
    for name,expected in meta['active_properties'].items():
        obj=scene.object_registry('name',name)
        require(obj is not None and obj.category==expected['category'] and obj.model==expected['model'], 'restored object identity differs')
        require(th.allclose(obj.scale,th.tensor(expected['scale']),atol=1e-6,rtol=0), 'restored scale differs')
        require(set(obj.links)==set(expected['links']), 'restored link inventory differs')
        for link_name,flags in expected['links'].items():
            link=obj.links[link_name]
            if flags['gravity_disabled'] is not None:
                (link.disable_gravity if flags['gravity_disabled'] else link.enable_gravity)()
            with sim.editing_usd():
                if flags['gravity_disabled'] is not None:
                    api=lazy.pxr.PhysxSchema.PhysxRigidBodyAPI(link.prim) if link.prim.HasAPI(lazy.pxr.PhysxSchema.PhysxRigidBodyAPI) \
                        else lazy.pxr.PhysxSchema.PhysxRigidBodyAPI.Apply(link.prim)
                    api.GetDisableGravityAttr().Set(flags['gravity_disabled'])
                apis={str(api.GetPrim().GetPath().MakeRelativePath(obj.prim_path)):api for api in getattr(link,'_collision_apis',[])}
                require(set(apis)==set(flags['collisions']), 'restored collision geometry differs')
                for key,enabled in flags['collisions'].items():apis[key].GetCollisionEnabledAttr().Set(enabled)
        obj.load_state(recursively_convert_to_torch(copy.deepcopy(template['state']['registry']['object_registry'][name])),serialized=False)
        obj.visible=expected['visible']
        require(native_properties(obj)==expected, 'pool physical properties were not restored')
        restored[name]=expected
    return {'active_properties':restored,'source_manifest_sha256':meta['manifest_sha256']}


class ModelPool:
    """Exact object leases; the backend performs real physical state changes."""
    def __init__(self, world, manifest, reference, output, *, execution_context=None):
        validate_execution_context(manifest, reference, execution_context)
        self.world, self.manifest, self.reference = world, manifest, reference
        self.execution_context = execution_context
        self.output = Path(output); self.members = {}; self.active = {}; self.events = []
        self.request_names = set(); self.switches = 0
        specs = []
        for row in manifest['entries']:
            key = signature(**{k: row[k] for k in ('category', 'model', 'dataset', 'scale')},
                            fixed_base=row.get('fixed_base', False), extra_abilities=row.get('extra_abilities'))
            for i in range(row['copies']):
                name = f'campaign_pool_{digest(key)[:16]}_{i}'
                specs.append({'name': name, 'key': key, 'source': row})
        self.members = world.preload_pool_objects(specs)
        require(set(self.members) == {s['name'] for s in specs}, 'preload omitted or added objects')
        self.specs = {s['name']: s for s in specs}
        self.record('preloaded')

    def acquire(self, request_name, category, model, dataset, scale, fixed_base=False, extra_abilities=None):
        if scale is None:
            defaults = [spec['key']['scale'] for spec in self.specs.values()
                        if spec['key']['category'] == category and spec['key']['model'] == model
                        and spec['key']['dataset'] == dataset and spec['source'].get('accept_default_scale')]
            require(defaults and all(x == defaults[0] for x in defaults), 'default scale lacks unique measured identity')
            scale = defaults[0]
        key = signature(category, model, dataset, scale, fixed_base, extra_abilities)
        require(request_name not in self.request_names, 'duplicate role request within active template')
        candidates = [name for name, spec in self.specs.items() if spec['key'] == key and name not in self.active]
        require(candidates, 'active request has no unused exact preloaded object; no lazy singleton fallback')
        name = candidates[0]; obj = self.members[name]
        self.world.reset_pool_object(obj)
        self.active[name] = request_name; self.request_names.add(request_name)
        self.world.spawned[name] = obj
        self.record('acquired')
        return obj

    def release_template(self):
        for name in list(self.active):
            self.world.park_pool_object(self.members[name])
            self.world.spawned.pop(name, None)
        self.active.clear(); self.request_names.clear(); self.switches += 1
        self.record('released')

    def filter_scene(self, scene, inst_to_name):
        check_policy(self.manifest)
        require(set(self.active) <= set(inst_to_name.values()), 'active pool object lacks a task binding')
        result = active_scene(scene, self.members, self.active)
        result.setdefault('metadata', {}).setdefault('ispatialgym', {})['model_pool'] = {
            'manifest_sha256': self.reference['sha256'], 'batch_id': self.manifest['batch_id'],
            'active_names': sorted(self.active), 'inactive_omitted': len(self.members) - len(self.active),
            'active_properties':{name:native_properties(self.members[name]) for name in self.active}}
        self.record('exported')
        return result

    def record(self, event):
        observation = self.world.observe_model_pool(self.members, self.active)
        self.events.append({'event': event, 'observed_at_unix': time.time(), 'loaded': sorted(self.members),
                            'active': dict(self.active), 'parked': sorted(set(self.members) - set(self.active)),
                            'template_switches': self.switches, 'native_observation': observation})
        if self.execution_context is not None:
            self.events[-1]['execution_context'] = self.execution_context
        value = {'schema': 'isg-model-pool-execution-v1', 'campaign_id': 'campaign_20261110',
                 'manifest': self.reference, 'benchmark_credit': 0, 'production_released': False,
                 'events': self.events, 'physical_validation_passed': False,
                 'boundary': 'Execution evidence only; native demo, independent replay, rendering and family-path checks remain separate.'}
        if self.execution_context is not None:
            value['execution_context'] = self.execution_context
        self.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.output.with_suffix('.tmp'); temporary.write_text(json.dumps(value, indent=2)+'\n'); temporary.replace(self.output)
