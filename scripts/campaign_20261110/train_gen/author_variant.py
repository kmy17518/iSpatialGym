#!/usr/bin/env python3
"""Author a dispatchable TRAIN source for one non-canonical training sub-template (preparation selection row).

Route B (one Slurm job, one Kit sampler process): `sample_isg` in fresh mode builds the variant's native template from the
authored customs entry + BDDL, and - because ISPATIALGYM_ONLINE_GROUPS is set - SamplerV2.run_instances hands the loaded
sampler to the online grouped worker (online_groups.run), which samples goal groups in the same world. The template.json is
saved at the first screened draw before its publication, under $JOBDIR/inst/<variant>/<template_id>/, which is where the
collectors (`run_isg --queue $JOBDIR/inst`) read it. Job wrapper: train_gen/job_variant.sh (= job_fast.sh with the sampler
line swapped). Run with the frozen snapshot's python (imports its contracts):

    PYTHONPATH=$SNAP $PY author_variant.py author --configuration-id <id> --k 80 [--groups 2] [--kcollect 2]
    PYTHONPATH=$SNAP $PY author_variant.py report <source dir>          # after the run: timings, map lookups, members

Inputs: validity row (gate `valid` manifest) -> registry configuration (prepared artifacts: grasp libraries, reach maps +
anchors) -> canonical eval source of the class (its customs entry, BDDL bytes, campaign template instances). Outputs (source
dir): campaign_config.json (split train, release code_sha256, exact applied scales; lighting {} / removed_objects [] for
the augmentation component), customs.json + bddl/, deps/{glib,maps,anchors.json[,stable_scene]} overlays (symlinks into the
prepared artifacts plus the base v2 release), preparation_binding.json, gate_author.txt (gate `check` must exit 0),
settings.json, job.env, run.sh, job_variant.sh, source.json.

Covered: floor-only held-object classes (object_scaling-*, mental_rotation-*) and tabletop classes (IORS small, alignment-axial)
with a scaled native support when a height probe scene is found. A role whose category differs from the canonical template
(nextto_01 on a desk instead of the breakfast table, ...) gets its BDDL re-rendered from the class IDDL (render_variant: the
canonical categories must re-render the canonical BDDL and alias first); every generation setting of the canonical customs
entry is kept, model-keyed values move to the new model, alias-bound switches are dropped (recorded). An unscaled support
outside the canonical scene takes the height of its prepared reach maps as top_z; a room with several floors binds the first
in scene order (native floor_of_room). Generic reach maps join the overlay without a library identity.
Refused (explicit error): multi-room / null-room rows (gate identity), dict joints, roles other than target/partner/support,
fixture replacements, scaled supports without a probe scene, x/y-scaled "height" supports. Census of what passes:
train_gen/coverage_census.py.
"""
from __future__ import annotations

import argparse
import copy
import csv
import glob
import hashlib
import importlib.util
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

R = Path('/vision/u/minyeong/projects/isg_home/campaign_20261110')
SNAP = Path(os.environ.get('ISG_TRAIN_SNAPSHOT', str(R / 'code/train_699ba262b46e')))
CODE_SHA256 = '70c32df8ef5cdd132ddc37366aed6ebee1bb458ca4588569bea409ba55a3bfe0'
RELEASE_RECORD = R / 'training_generation/releases/release_train_r2.json'
RP = R / 'training_preparation/preparation_vast_20261114T034355Z'
GATE = RP / 'validity/code/train_validity_gate.py'
REGISTRY = RP / 'registry/revisions/rev_20261114T161444Z_7d5cc515cc7e'
SELECTION = RP / 'routes/scope/selection_r14_sources/configurations.json'
VARIANT_SUPPORTS = R / 'training_preparation/model_variant_inventory_r2/variant_supports.csv'
VARIANTS = R / 'training_preparation/model_variant_inventory_r2/variants.json'
EVAL_INDEX = R / 'training_generation/train_online_r1/eval_source_index.json'
PROPOSAL = R / 'training_generation/handoff_20261114/train_class_wallclock_proposal.csv'
OBSERVER = R / 'tools/eval_runtime_observer_5da2a8fc.py'
VA = R / 'training_generation/variant_authoring_r1'
GIVEUPS = VA / 'subtemplate_giveups.jsonl'
#: where --dry-run sources go (coverage_census.py points it at VA/dryrun_cov/<run>)
DRY_ROOT = {'path': VA / 'dryrun'}
#: production mode (--production): sources in the train root, ordinals shared with author_canonical (t2611_<ordinal>)
T = R / 'training_generation/train_online_r1'
PRODUCTION = {'on': False}
#: ordinals whose index ranges the test slots k=80..99 (INDEX_BASE + k*1e5) already use; never allocated
RESERVED_ORDINALS = range(10080, 10100)
BASE_V2 = Path('/vision/u/minyeong/projects/isg_home/oracle_fix/v2maps_int2')
PY = '/vision/u/minyeong/miniconda3/envs/ispatialgym-vectorized-envs/bin/python'
INDEX_BASE = 2000600000000
#: height probe outputs (support_height_probe.py / prep maps-support_height_measurement): <dir>/{height_probe|out/measurement}.json
PROBE_GLOBS = [str(R / 'height_probes/*/height_probe.json'), str(R / 'height_probes_v2/*/height_probe.json'),
               str(R / 'training_preparation/*/native/support_*/height_probe.json'),
               str(R / 'training_preparation/*/secondary_completed_r1/files/native/support_*/height_probe.json'),
               str(RP / 'jobs/maps-support_height_measurement-*/out/measurement.json')]
EVAL_ONLY_GENERATION_KEYS = ('requested_bucket_m', 'requested_direction', 'start_depth_band', 'partner_side', 'destination_side')

sys.path.insert(0, str(SNAP))
from scripts.campaign_20261110.contracts import (  # noqa: E402
    ObjectRef, Perturbations, ScaleState, Template, ValidationKey, digest, file_digest, require, write_json_atomic)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def ref(p):
    return {'path': str(p), 'sha256': sha(p)}


def read(p):
    return json.loads(Path(p).read_text())


def gate_module():
    spec = importlib.util.spec_from_file_location('tvg', GATE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_env(path):
    env = {}
    for line in Path(path).read_text().splitlines():
        if line.strip() and not line.startswith('#') and '=' in line:
            k, v = line.split('=', 1)
            vals = shlex.split(v)
            env[k] = vals[0] if vals else ''
    return env


def dump_env(env):
    return ''.join(f'{k}={shlex.quote(str(v))}\n' for k, v in env.items())


def proposal_row(cid):
    for row in csv.DictReader(PROPOSAL.open()):
        if row['class_id'] == cid:
            return row
    raise SystemExit('class not in the wallclock proposal: ' + cid)


def asset_path(category, model, dataset=None):
    dataset = dataset or 'behavior-1k-assets'
    root = SNAP / 'ispatialgym/assets/datasets' / dataset if dataset == 'ispatialgym-assets' else SNAP / 'Behavior-1K/datasets' / dataset
    return root / 'objects' / category / model / 'usd' / (model + ('.encrypted.usd' if dataset == 'behavior-1k-assets' else '.usd'))


def same_scale(a, b, tol=1e-12):
    return len(a) == len(b) == 3 and all(abs(float(x) - float(y)) <= tol for x, y in zip(a, b))


# ------------------------------------------------------------------------------------------------ inputs

def validity_row(gate, cid):
    m = gate.load_manifest()
    row = m['_by_id'].get(cid)
    require(row is not None, 'configuration not in the validity manifest: ' + cid)
    require(row['status'] == 'valid', f'{cid} is {row["status"]}, not valid')
    rooms = row['rooms']
    require(isinstance(rooms, list) and len(rooms) == 1, 'unsupported: rooms must be one named room (gate defect: null / multi-room)')
    for r in row['identity_roles']:
        require(r['role'] in ('target', 'partner', 'support'), 'unsupported role ' + str(r['role']))
        require(r.get('joint') is None or isinstance(r.get('joint'), str), 'unsupported: dict joint (articulation)')
    require(not row.get('fixture_replacement'), 'unsupported: replacement fixture (no placement mechanism)')
    return m, row


def registry_configuration(cid):
    cfg = next((r for r in read(REGISTRY / 'configurations.json')['rows'] if r['configuration_id'] == cid), None)
    require(cfg is not None and cfg.get('all_mandatory_consumers_resolve'), 'registry configuration missing or unresolved')
    ids = {a for us in cfg['units'].values() for u in us for a in u['artifacts']}
    arts = {r['artifact_id']: r for r in read(REGISTRY / 'artifacts.json')['rows'] if r['artifact_id'] in ids}
    return cfg, arts


def canonical_source(cid):
    src = Path(read(EVAL_INDEX)['sources'][cid])
    cfg = read(src / 'campaign_config.json')
    env = load_env(src / 'job.env')
    tdir = Path(env['TEMPLATE'])
    meta = read(tdir / 'template_meta.json')
    customs_path = Path(meta['customs'])
    entry = read(customs_path)[meta['variant']]
    bddls = [p for p in (tdir / 'activity_definitions').rglob('*.bddl') if sha(p) == meta['bddl_sha256']]
    require(len(bddls) == 1, 'canonical BDDL bytes missing or ambiguous')
    return dict(dir=src, config=cfg, env=env, template_dir=tdir, meta=meta, customs=customs_path, entry=entry, bddl=bddls[0],
                bddl_rel=bddls[0].relative_to(tdir / 'activity_definitions'), run=(src / 'run.sh').read_text())


def model_defaults():
    out = {}
    for row in read(VARIANTS)['rows']:
        for a in row.get('model_assets') or []:
            out[a['model_key']] = a
    return out


def support_instance(row, name_hint=None):
    sel = next((r for r in read(SELECTION)['rows'] if (r.get('inventory_configuration_id') or '') == row['configuration_id']
                or 'train:' + (r.get('source_configuration_id') or '') == row['configuration_id']), None)
    names = set()
    if sel is not None:
        for r in csv.DictReader(VARIANT_SUPPORTS.open()):
            if r['class_id'] == row['class_id'] and r['support_id'] == sel.get('support_id') and r['support_instance']:
                names.add(r['support_instance'])
    if not names and name_hint:
        names.add(name_hint)
    require(len(names) == 1, f'support scene instance unresolved for {row["configuration_id"]}: {sorted(names)}')
    return names.pop(), sel


def find_probe(scene, name, scale):
    """The height-probe scene of (scene, support, applied scale): its private stable scene (scaled support, chairs moved)."""
    hits = []
    for pattern in PROBE_GLOBS:
        for p in glob.glob(pattern):
            try:
                d = read(p)
            except (OSError, ValueError):
                continue
            s = d.get('absolute_scale') or d.get('selected_scale')
            if d.get('scene') == scene and d.get('object') == name and s and same_scale(s, scale, 1e-9):
                scene_path = d.get('stable_scene_path') or (d.get('stable_scene') or {}).get('path')
                if scene_path and Path(scene_path).is_file():
                    hits.append((p, d, scene_path))
    require(hits, f'no height probe scene for {scene}/{name} at {scale} (scaled native support placement unresolved)')
    return hits[0]


# ------------------------------------------------------------------------------------------------ prepared artifacts

def link(dst, src):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() or dst.exists():
        require(Path(os.path.realpath(dst)) == Path(os.path.realpath(src)) or sha(dst) == sha(src), f'overlay collision {dst}')
        return
    os.symlink(src, dst)


def build_dependency_overlay(job, row, cfg, arts, canonical_env):
    """glib/maps roots = the base v2 release + the configuration's prepared artifacts (one consistent library identity per model)."""
    deps = job / 'deps'
    glib, maps = deps / 'glib/models', deps / 'maps'
    glib.mkdir(parents=True)
    maps.mkdir(parents=True)
    used = {'grasp_libraries': [], 'maps': [], 'anchors': None, 'grasp_keys': {}}
    map_arts = [a for a in arts.values() if a.get('native_kind') == 'reach_map' and (a.get('lookup') or {}).get('lookup_ok')]
    lib_arts = [a for a in arts.values() if a.get('native_kind') == 'grasp_library' and (a.get('lookup') or {}).get('lookup_ok')]
    # maps decide the library identity of their model (a map is only found with the library it was built against)
    chosen_libs = {}
    for a in map_arts:
        if not a.get('library'):   # generic (model-free) reach maps carry no library identity
            continue
        lib = Path(a['library']['path'])
        model_key = '/'.join(lib.parent.name.split('__')[:2])
        chosen_libs.setdefault(model_key, (lib, a['lookup']['identity_key']))
        require(chosen_libs[model_key][1] == a['lookup']['identity_key'], 'maps of one model need different library identities')
    for a in lib_arts:
        mk = a['lookup']['model']
        if mk in chosen_libs:
            continue
        lib = Path(a['path'])
        keys = [k for k in (a['lookup'].get('legacy_keys') or []) if k]
        # prefer the entry with a campaign key, then any keyed entry
        rank = (0 if any(k.startswith('campaign_selected__') for k in keys) else 1 if keys else 2)
        prev = chosen_libs.get(mk + '#cand')
        if prev is None or rank < prev[2]:
            chosen_libs[mk + '#cand'] = (lib, a['lookup']['identity_key'], rank)
    for k in [k for k in chosen_libs if k.endswith('#cand')]:
        mk = k[:-5]
        chosen_libs.setdefault(mk, chosen_libs[k][:2])
        del chosen_libs[k]
    for mk, (lib, ident) in chosen_libs.items():
        d = read(lib)
        keys = [k for k in [d.get('legacy_key'), *(d.get('legacy_keys') or [])] if k]
        require(keys, f'prepared library {lib} has no lookup key')
        key = next((k for k in keys if k.startswith('campaign_selected__')), keys[0])
        entry = lib.parent.name
        alias = None
        stem = d['category'] + '__' + d['model']
        if (d.get('constraint') or 'tabletop') == 'floating' and stem not in keys:
            # The floating consumer (held_object.floating_key) only accepts '<category>__<model>' keys; the prepared entry is
            # keyed 'campaign_selected__*' only. Serve it under the stem key: same files (symlinks), library.json copy with the
            # stem appended to legacy_keys (grasps/rests/scale/policy bytes unchanged otherwise; recorded).
            (glib / entry).mkdir()
            for f in lib.parent.iterdir():
                if f.name != 'library.json':
                    os.symlink(f, glib / entry / f.name)
            d2 = dict(d, legacy_keys=list(d.get('legacy_keys') or []) + [stem])
            write_json_atomic(glib / entry / 'library.json', d2)
            alias = {'stem_key': stem, 'source_library': ref(lib), 'served_library': ref(glib / entry / 'library.json'),
                     'change': 'legacy_keys += [stem_key] (held_object.floating_key accepts stem keys only)'}
            key = stem
            used.setdefault('claimed_keys', []).append(stem)
        else:
            link(glib / entry, lib.parent)
        used['grasp_libraries'].append({'model': mk, 'entry': entry, 'library': ref(lib), 'identity_key': ident, 'grasp_key': key,
                                        'key_alias': alias})
        used['grasp_keys'][mk] = key
    seen = set()
    anchors = set()
    for a in map_arts:
        p = Path(a['path'])
        if p.name in seen:
            continue
        seen.add(p.name)
        for f in [p, p.with_suffix('.npz')]:
            if f.exists():
                link(maps / p.parent.name / f.name, f)
        used['maps'].append({'map': ref(p), 'arm': a.get('arm'), 'height': (a.get('height') or {}).get('key'),
                             'height_source': (a.get('height') or {}).get('source'),
                             'identity_key': a['lookup'].get('identity_key'), 'kind': a['lookup'].get('kind')})
        if (a.get('height') or {}).get('anchors'):
            anchors.add(a['height']['anchors']['path'])
    # base release (the canonical's roots) under every name the prepared artifacts do not take
    base_glib = Path(canonical_env.get('ISPATIALGYM_GRASP_LIBRARY_V2', BASE_V2 / 'glib'))
    base_maps = Path(canonical_env.get('ISPATIALGYM_REACH_MAPS', BASE_V2 / 'maps'))
    claimed = set(used.get('claimed_keys', []))
    for e in sorted((base_glib / 'models').iterdir()):
        if (glib / e.name).exists():
            continue
        if claimed and (e / 'library.json').exists():
            bd = read(e / 'library.json')
            if claimed & {k for k in [bd.get('legacy_key'), *(bd.get('legacy_keys') or [])] if k}:
                used.setdefault('base_entries_shadowed', []).append(e.name)   # would win the key at another scale
                continue
        os.symlink(e, glib / e.name)
    for e in sorted(base_maps.iterdir()):
        target = maps / e.name
        if not target.exists():
            os.symlink(e, target)
        elif target.is_dir() and not target.is_symlink() and e.is_dir():
            for f in e.iterdir():
                if not (target / f.name).exists():
                    os.symlink(f, target / f.name)
    for extra in (base_glib / 'profiles',):
        if extra.is_dir():
            os.symlink(extra, deps / 'glib/profiles')
    if anchors:
        # model maps and generic maps of one configuration can name different anchor files: merge them by key (a key named twice
        # must name the same height), then the base anchors of other keys
        prepared, files = {}, []
        for a in sorted(Path(x) for x in anchors):
            files.append(ref(a))
            for x in read(a)['anchors']:
                prev = prepared.setdefault(x['key'], x)
                require(abs(float(prev['height']) - float(x['height'])) <= 1e-9, f'anchor key {x["key"]} names two heights')
        base = read(canonical_env.get('ISPATIALGYM_REACH_MAP_ANCHORS', BASE_V2 / 'anchors.json'))
        merged = dict(base, anchors=list(prepared.values()) + [x for x in base.get('anchors', []) if x['key'] not in prepared])
        write_json_atomic(deps / 'anchors.json', merged)
        used['anchors'] = {'prepared': files[0] if len(files) == 1 else files,
                           'base': ref(canonical_env.get('ISPATIALGYM_REACH_MAP_ANCHORS', BASE_V2 / 'anchors.json')),
                           'merged': ref(deps / 'anchors.json'), 'rule': 'prepared anchors first, base anchors of other keys kept'}
    else:
        used['anchors'] = {'base': ref(canonical_env.get('ISPATIALGYM_REACH_MAP_ANCHORS', BASE_V2 / 'anchors.json'))}
    return used


# ------------------------------------------------------------------------------------------------ category changes

def _bddl_body(text):
    """BDDL text without comments, blank lines and the problem / variant header (what the IDDL decides)."""
    out = []
    for line in text.splitlines():
        t = line.split(';', 1)[0].strip()
        if t and not t.startswith(('(define (problem', '(:variant')):
            out.append(' '.join(t.split()))
    return out


def render_variant(can, categories):
    """The class IDDL rendered with the row's role categories (materialize_model_table_tasks pattern).

    The canonical categories must re-render the canonical BDDL (comments and header aside) and the canonical alias, with or
    without the `ontop -> inside` init rewrite (ground_isg's starts_inside); the same rewrite is then applied to the row's
    categories. Returns (alias, rendered, canonical rendered, init_predicates, iddl path)."""
    from ispatialgym.sampling.bddl_from_iddl import RoleGrounding, render
    from ispatialgym.sampling.iddl import parse_file
    from ispatialgym.sampling.knowledge import synset_of_category
    entry = can['entry']
    ipath = SNAP / 'ispatialgym/assets/iddl' / entry['iddl']
    iddl = parse_file(str(ipath))
    canon_cats = {role: (entry['roles'].get(role) or {}).get('category') for role in iddl.roles}
    require(not iddl.groups, 'IDDL render: composite groups are not supported')

    def groundings(cats):
        out = {}
        for role in iddl.roles:
            if role == 'floor':
                out[role] = RoleGrounding(role, 'scene', same_as='floor')
                continue
            spec = entry['roles'].get(role) or {}
            require(cats.get(role), f'IDDL render: role {role} has no category')
            out[role] = RoleGrounding(role, 'sampleable' if spec.get('sampleable') else 'scene', category=cats[role],
                                      synset=synset_of_category(cats[role]))
        return out

    def alias(cats):
        return iddl.variant_template.format(**{slot: cats[slot] for slot in iddl.variant_slots})

    did, activity = int(entry['definition_id']), can['bddl_rel'].parts[0]
    require(alias(canon_cats) == can['meta']['variant'], f'IDDL render: canonical alias {alias(canon_cats)} != {can["meta"]["variant"]}')
    want = _bddl_body(can['bddl'].read_text())
    supports = [r for r in iddl.roles if not (entry['roles'].get(r) or {}).get('sampleable') and r != 'floor']
    for init_predicates in (None, {('ontop', r): 'inside' for r in supports}):
        canon = render(iddl, groundings(canon_cats), did, alias(canon_cats), activity, init_predicates=init_predicates or None)
        if _bddl_body(canon.text) == want:
            break
    else:
        raise SystemExit('IDDL render failed: the canonical categories do not re-render the canonical BDDL')
    cats = dict(canon_cats, **categories)
    new = render(iddl, groundings(cats), did, alias(cats), activity,
                 comment=f'train variant (author_variant): {alias(cats)} rendered from {entry["iddl"]} with the row categories',
                 init_predicates=init_predicates or None)
    return alias(cats), new, canon, init_predicates, ipath


# ------------------------------------------------------------------------------------------------ authoring

def allocate_production():
    (T / 'ordinals').mkdir(parents=True, exist_ok=True)
    existing = [int(p.name) for p in (T / 'ordinals').iterdir() if p.name.isdigit()]
    o = max(existing + [9999]) + 1
    while True:
        if o in RESERVED_ORDINALS:
            o += 1; continue
        try:
            (T / 'ordinals' / str(o)).mkdir(); return o
        except FileExistsError:
            o += 1


def allocate(k):
    require(80 <= k <= 89, 'test slots are k = 80..89')
    (VA / 'ordinals').mkdir(parents=True, exist_ok=True)
    (VA / 'ordinals' / str(k)).mkdir()   # atomic; a used slot raises
    return k


def author(cid, k, groups, kcollect, dry=False, lighting=None, removed_objects=(), augment_seed=None, force_augment=False):
    gate = gate_module()
    manifest, row = validity_row(gate, cid)
    cls = row['class_id']
    require(not cls.startswith(('disembedding', 'configuration_matching-block_design', 'configuration_matching-articulation',
                                'camera_relocalization', 'configuration_matching-attachment')),
            'unsupported family for this author: ' + cls)
    require(cls not in ('inter_object_relation_satisfaction-under_small_scale_03',), 'profile_for_config rejects non-canonical ' + cls)
    can = canonical_source(cls)
    cconf, ctemplate = can['config'], can['config']['template']
    scene, room = row['scene'], row['rooms'][0]
    reg, arts = registry_configuration(cid)
    defaults = model_defaults()
    from scripts.campaign_20261110.readiness_catalog import ReadinessCatalog
    catalog = ReadinessCatalog(SNAP / 'Behavior-1K/bddl3/bddl/generated_data')

    # ---- the scene: native stable json, or the height probe's private scene for a scaled native support
    supports = [r for r in row['identity_roles'] if r['role'] == 'support']
    require(len(supports) <= 1, 'more than one support role')
    probe = None
    stable = SNAP / 'Behavior-1K/datasets/2026-challenge-task-instances/scenes' / scene / 'json' / (scene + '_stable.json')
    support_name = None
    if supports:
        canon_support = next(o for o in ctemplate['objects'] if o['role'] == 'support' and o['semantic_kind'] == 'object')
        hint = can['meta']['inst_to_name'].get(canon_support['instance']) if (scene == ctemplate['scene']) else None
        support_name, _ = support_instance(row, hint)
        native = read(stable)['objects_info']['init_info'][support_name]['args']
        require((native['category'], native['model']) == (supports[0]['category'], supports[0]['model']), 'support instance model differs')
        if not same_scale(native['scale'], supports[0]['scale'], 1e-9):
            probe = find_probe(scene, support_name, supports[0]['scale'])
            stable = Path(probe[2])
    init = read(stable)['objects_info']['init_info']
    floors = [n for n, v in init.items() if v['args'].get('category') == 'floors' and room in (v['args'].get('in_rooms') or [])]
    require(floors, f'no floor in {scene}/{room}')
    # a room with several floor objects (house_single_floor garden_0): the sampler binds the first in scene order
    # (sim_utils.floor_of_room; the BDDL `ontop agent floor` is judged on the floor under the robot, floor_object_under)
    floor_note = None if len(floors) == 1 else {'candidates': floors, 'bound': floors[0], 'rule': 'first in scene order (sim_utils.floor_of_room)'}

    # ---- role -> canonical object; a role whose category differs re-renders the BDDL from the class IDDL (render_variant)
    objects, scales, inst_to_name, roles_customs = [], {}, {}, {}
    cobj = {(o['role'], o['category']): o for o in ctemplate['objects'] if o['semantic_kind'] != 'floor'}
    by_role = {}
    for o in ctemplate['objects']:
        if o['semantic_kind'] != 'floor':
            by_role.setdefault(o['role'], []).append(o)
    changed = {}
    for r in row['identity_roles']:
        if (r['role'], r['category']) not in cobj:
            cands = by_role.get(r['role'], [])
            require(len(cands) == 1, f'unsupported: {r["role"]} category change with {len(cands)} canonical {r["role"]} objects')
            changed[r['role']] = (cands[0]['category'], r['category'])
    variant = can['meta']['variant']
    rendered = None
    new_inst = {}   # canonical instance -> rendered instance
    if changed:
        variant, rendered, canon_r, init_predicates, ipath = render_variant(can, {role: c2 for role, (c1, c2) in changed.items()})
        for role, inst in canon_r.instances.items():
            if isinstance(inst, str) and role in rendered.instances:
                new_inst[inst] = rendered.instances[role]
    floor_c = next(o for o in ctemplate['objects'] if o['semantic_kind'] == 'floor')
    fa = init[floors[0]]['args']
    st = catalog.status(fa['category'], fa['model'])
    require(st['ready'], 'floor not ready')
    floor_i = new_inst.get(floor_c['instance'], floor_c['instance'])
    objects.append(ObjectRef(instance=floor_i, role='support', category=fa['category'], model=fa['model'],
                             asset_sha256=sha(asset_path(fa['category'], fa['model'])), default_scale=tuple(fa['scale']),
                             semantic_kind='floor', ready=True, readiness_source=st['readiness_source']))
    scales[floor_i] = ScaleState(tuple(fa['scale']), tuple(fa['scale']), 'fixed')
    inst_to_name[floor_i] = floors[0]
    for r in row['identity_roles']:
        c = cobj.get((r['role'], r['category'])) or by_role[r['role']][0]
        inst = new_inst.get(c['instance'], c['instance'])
        require(not changed or inst in rendered.instances.values(), f'{r["role"]}: canonical instance {c["instance"]} not rendered')
        mk = r['category'] + '/' + r['model']
        path = asset_path(r['category'], r['model'])
        require(path.is_file() and sha(path) == r['asset_sha256'], f'asset bytes of {mk} differ from the selection row')
        st = catalog.status(r['category'], r['model'])
        require(st['ready'], mk + ' not ready')
        applied = tuple(float(v) for v in r['scale'])
        if r['role'] == 'support':
            default = tuple(float(v) for v in read(SNAP / 'Behavior-1K/datasets/2026-challenge-task-instances/scenes' / scene / 'json'
                                                    / (scene + '_stable.json'))['objects_info']['init_info'][support_name]['args']['scale'])
            mode = 'fixed' if same_scale(default, applied) else 'height'
            if mode == 'height':
                require(applied[0] == default[0] and applied[1] == default[1], 'unsupported: height scale changes x/y (world-up axis fix §3a)')
            inst_to_name[inst] = support_name
        else:
            md = defaults.get(mk)
            default = tuple(md['default_scale']) if md else tuple(c['default_scale'])
            require(md is not None or r['model'] == c['model'], 'no ready default scale for ' + mk)
            mode = 'fixed' if same_scale(default, applied) else 'uniform'
            cname = can['meta']['inst_to_name'][c['instance']]
            old_tok, new_tok = f'_{c["category"]}_{c["model"]}_', f'_{r["category"]}_{r["model"]}_'
            require(old_tok in cname or (c['category'], c['model']) == (r['category'], r['model']), f'scene name {cname} has no {old_tok}')
            inst_to_name[inst] = cname.replace(old_tok, new_tok)
        scales[inst] = ScaleState(default, applied, mode)
        objects.append(ObjectRef(instance=inst, role=c['role'], category=r['category'], model=r['model'],
                                 asset_sha256=r['asset_sha256'], default_scale=default, joint=r.get('joint'),
                                 semantic_kind=c['semantic_kind'], ready=True, readiness_source=st['readiness_source']))
        roles_customs[r['role']] = (r, c, applied, mode)
    # scene/light augmentation (another component chooses them): removed_objects are parked by sample_isg at the fresh template's
    # world build (campaign_runtime.scene_removals -> scene variant), lighting is applied by campaign_runtime.prepare before screens
    from ispatialgym.sampling import campaign_augmentation as AUG
    if augment_seed is not None:
        # D5 (2026-11-14): per-source draws, each with p = 0.5 (campaign_augmentation helpers); identical for every group of the
        # worker. force_augment (tests only) advances the seed until both draws are non-empty.
        seed = int(augment_seed)
        while True:
            lit = AUG.lighting_draw(seed) if lighting is None else lighting
            rem = (AUG.removal_candidates(read(stable), rooms=[room], protected=list(inst_to_name.values()), seed=seed + 7919)
                   if not removed_objects else list(removed_objects))
            if not force_augment or (lit and rem):
                break
            seed += 1
        lighting, removed_objects = lit, rem
        augment_seed = seed
    removed = tuple(AUG.checked_removals(read(stable), list(removed_objects or ()), inst_to_name.values())) if removed_objects else ()
    lighting = dict(lighting or {})
    AUG.validate_lighting(lighting)
    perturbations = Perturbations(scales, removed, lighting)
    perturbations.validate('train', [o.instance for o in objects])

    # ---- output dir
    if PRODUCTION['on'] and not dry:
        ordinal = allocate_production(); key = f't2611_{ordinal}'; job = T / 'sources' / key; k = ordinal - 10000
    else:
        k = k if dry else allocate(k)
        key = f'va_{k}'
        # dry: no slot, never dispatched; pid + ns keep parallel dry runs (coverage census) apart
        job = (DRY_ROOT['path'] / f'{key}_{int(time.time())}_{os.getpid()}_{time.time_ns() % 10**9}') if dry else VA / 'sources' / key
    require(not job.exists(), 'source exists: ' + str(job))
    job.mkdir(parents=True)
    used = build_dependency_overlay(job, row, reg, arts, can['env'])
    if probe:
        above = probe[1].get('measured_height_above_floor')
        heights = {float(m['height']) for m in used['maps'] if m.get('height') is not None}
        arms = {m.get('arm') for m in used['maps'] if m.get('height') is not None}
        # every arm needs a prepared map at the probe height; maps of other (older native) heights may ride along unused
        hit = {m.get('arm') for m in used['maps'] if m.get('height') is not None and above is not None and abs(float(m['height']) - above) <= 5e-5}
        require(above is not None and heights and hit == arms,
                f'probe height {above} differs from the prepared map heights {sorted(heights)} (placement mismatch)')

    # ---- customs entry + BDDL (canonical bytes, or the IDDL render when a role category changes)
    entry = copy.deepcopy(can['entry'])
    for role, (c1, c2) in changed.items():
        entry['roles'][role]['category'] = c2
    for role, (r, c, applied, mode) in roles_customs.items():
        if role == 'support':
            continue
        spec = entry['roles'][role]
        mk = r['category'] + '/' + r['model']
        spec['representative'] = r['model']
        if mode != 'fixed':
            spec['scale'] = {mk: list(applied)}
        elif isinstance(spec.get('scale'), dict):
            spec['scale'] = {k2: v for k2, v in spec['scale'].items() if k2 == mk}
        if mk in used['grasp_keys']:
            spec['grasp_keys'] = {mk: used['grasp_keys'][mk]}
        elif 'grasp_keys' in spec:
            require(r['model'] == c['model'] and mode == 'fixed', f'no prepared grasp library for {mk} at {applied}')
        if r['model'] != c['model']:
            for name in ('orientation_equivalences', 'rest_pose', 'rest_hint_wxyz'):
                spec.pop(name, None)
            for name in ('frame', 'handle_file'):
                require(name not in spec, f'model-keyed {name} of {role} cannot be transferred to {mk}')
    gen = entry['generation']
    for name in list(gen):
        if name.startswith('_') or (name in EVAL_ONLY_GENERATION_KEYS and not (name == 'destination_side' and cls.startswith('disembedding'))):
            gen.pop(name)
    gen.update(templates_max=1, instances_per_template=1)
    inherited = []
    for role, (r, c, applied, mode) in roles_customs.items():   # model-keyed generation values (e.g. pair_goal_align.front)
        old, new = c['category'] + '/' + c['model'], r['category'] + '/' + r['model']
        if old == new:
            continue
        def walk(d, path=()):
            if isinstance(d, dict):
                if old in d and new not in d:
                    d[new] = d.pop(old)
                    inherited.append('.'.join(path + (new,)))
                for k2, v in d.items():
                    walk(v, path + (k2,))
        walk(gen, ('generation',))
    dropped_generation = []
    # alias/model-bound switches of the canonical source: they reject any other variant or model pair (iors_small)
    if gen.get('crayon_raised_release_gap_fallback') and (variant != can['meta']['variant'] or any(
            (r['category'], r['model']) != (c['category'], c['model']) for role, (r, c, a, m) in roles_customs.items() if role != 'support')):
        gen.pop('crayon_raised_release_gap_fallback')
        dropped_generation.append('crayon_raised_release_gap_fallback (bound to the canonical crayon/ruler source)')
    block = {'rooms': [room], 'bindings': {'floor': {'room': room}}, 'blacklist': {},
             'whitelist': {role: {r['category']: [r['model']]} for role, (r, c, a, m) in roles_customs.items() if role != 'support'}}
    top = top_source = None
    if supports:
        top = probe[1].get('measured_height_world', probe[1].get('support_top_z_world')) if probe else None
        top_source = 'height_probe' if top is not None else None
        if top is None:
            cblock = can['meta']['choice']['bindings']
            surf = [s for s in cblock.get('support_surfaces', []) if s['name'] == support_name]
            if scene == ctemplate['scene'] and surf:
                top, top_source = surf[0]['top_z'], 'canonical_surface'
            else:
                # unscaled native support elsewhere: the height (m above the floor) its prepared reach maps were measured at;
                # the sampler keeps the support top when it is within SURFACE_MISMATCH_M of it (tabletop.support_surface)
                ms = [m for m in used['maps'] if m.get('height') is not None]
                measured = [m for m in ms if m.get('height_source') == 'measured_configuration_height']
                heights = sorted({round(float(m['height']), 4) for m in (measured or ms)})
                require(len(heights) == 1, f'unscaled support outside the canonical scene: top height unknown (prepared map heights {heights})')
                top, top_source = heights[0], 'prepared_map_height'
        block['bindings']['support'] = [support_name]
        block['bindings']['support_surfaces'] = [{'name': support_name, 'init': 'ontop', 'top_z': float(top), 'row': 'train_variant_' + key}]
    entry['scenes'] = {scene: block}
    entry['constraints'] = []
    for name in list(entry):
        if name.startswith('fallback') or name == 'superseded_by':
            entry.pop(name)
    entry['campaign_provenance'] = {'status': 'train_variant_route_b', 'configuration_id': cid,
                                    'canonical_customs': ref(can['customs']), 'canonical_variant': can['meta']['variant'],
                                    'category_changes': {r: list(v) for r, v in changed.items()},
                                    'inherited_model_keyed_generation': inherited, 'dropped_generation': dropped_generation,
                                    'production_released': False}
    write_json_atomic(job / 'customs.json', {variant: entry})
    bddl = job / 'bddl' / can['bddl_rel']
    bddl.parent.mkdir(parents=True)
    if rendered is None:
        shutil.copyfile(can['bddl'], bddl)
        require(sha(bddl) == can['meta']['bddl_sha256'], 'BDDL copy differs')
    else:
        bddl.write_text(rendered.text)
    # the native problem must ground the exact models in this scene (no simulator)
    from ispatialgym.sampling.v2.problem import GroundedProblem, iter_template_choices
    import numpy as np
    gp = GroundedProblem.load(variant, str(job / 'customs.json'), str(job / 'bddl'))
    choices = list(iter_template_choices(gp, scene, np.random.default_rng(0), max_templates=1))
    require(len(choices) == 1, 'native choice unavailable for the variant')
    for role, (r, c, a, m) in roles_customs.items():
        if role != 'support':
            got = choices[0].models[role]
            require(got.model == r['model'], f'native {role} choice differs')
    if probe:
        sdir = job / 'deps/stable_scene'
        sdir.mkdir(parents=True)
        shutil.copyfile(probe[2], sdir / (scene + '_stable.json'))
        live = read(sdir / (scene + '_stable.json'))['objects_info']['init_info'][support_name]['args']
        require(same_scale(live['scale'], supports[0]['scale'], 1e-9), 'probe scene support scale differs from the row')

    # ---- campaign template + config
    setup = {'task_rooms': [room], 'inst_to_name': inst_to_name, 'runtime_bddl_sha256': sha(bddl),
             'preparation_configuration_id': cid, 'preparation_variant_id': row.get('variant_id'),
             'preparation_template_key': row.get('template_key'),
             'validity_manifest': {'path': manifest['_pointer']['path'], 'sha256': manifest['_pointer']['sha256'],
                                   'revision': manifest['_pointer']['revision']},
             'canonical_source': {'job_dir': str(can['dir']), 'campaign_config': ref(can['dir'] / 'campaign_config.json'),
                                  'template_dir': str(can['template_dir'])},
             'authored_customs': ref(job / 'customs.json'), 'authored_bddl': ref(bddl),
             'prepared_dependencies': used, 'route': 'B_fresh_native_online_worker', 'production_released': False}
    if supports:
        setup['support_top'] = {'top_z': float(top), 'source': top_source}
    if floor_note:
        setup['floor_binding'] = floor_note
    if rendered is not None:
        setup['iddl_render'] = {'iddl': ref(ipath), 'variant': variant, 'canonical_variant': can['meta']['variant'],
                                'category_changes': {r: list(v) for r, v in changed.items()},
                                'init_predicates': sorted(f'{a}:{b}->{c}' for (a, b), c in (init_predicates or {}).items()),
                                'pattern': 'materialize_model_table_tasks (RoleGrounding + iter_template_choices)'}
    if probe:
        setup['private_scene'] = {'schema': 'isg-native-private-scene-v1', 'path': str(job / 'deps/stable_scene' / (scene + '_stable.json')),
                                  'sha256': sha(job / 'deps/stable_scene' / (scene + '_stable.json')),
                                  'height_probe': ref(probe[0]), 'source': probe[2]}
    if ctemplate['scene_setup'].get('recovery_options'):
        setup['recovery_options'] = ctemplate['scene_setup']['recovery_options']
    template = Template(template_id='train_variant_' + digest({'cid': cid, 'customs': sha(job / 'customs.json')})[:20],
                        task=ctemplate['task'], class_id=cls, split='train', scene=scene, room=room, objects=tuple(objects),
                        definition_sha256=digest({'customs': sha(job / 'customs.json'), 'bddl': sha(bddl), 'configuration_id': cid}),
                        scene_setup=setup)
    vk = cconf['validation_key']
    key_ = ValidationKey(template_sha256=template.fingerprint, code_sha256=CODE_SHA256, physics_sha256=vk['physics_sha256'],
                         data_sha256=vk['data_sha256'], scale_sha256=perturbations.scale_fingerprint)
    config = {'campaign_id': cconf['campaign_id'], 'perturbations': asdict(perturbations), 'purpose': 'train_variant', 'split': 'train',
              'template': asdict(template), 'validation_key': asdict(key_)}
    write_json_atomic(job / 'campaign_config.json', config)
    # recovery options must pass their own class/model checks for this template, else they are dropped (recorded)
    os.environ['ISPATIALGYM_CAMPAIGN_CONFIG'] = str(job / 'campaign_config.json')
    os.environ['ISGF_ROOT'] = str(R)
    from ispatialgym.sampling import campaign_runtime as CR
    dropped = []
    for opt, on in list((setup.get('recovery_options') or {}).items()):
        if not on:
            continue
        try:
            CR._load.cache_clear()
            CR.contact_recovery(opt) if opt != 'inside_frame_rank' else CR.inside_frame_recovery()
        except Exception as exc:  # noqa: BLE001
            dropped.append({'option': opt, 'reason': str(exc)})
    if dropped:
        for d in dropped:
            setup['recovery_options'].pop(d['option'])
        setup['dropped_recovery_options'] = dropped
        template = Template(**{**asdict(template), 'objects': tuple(objects), 'scene_setup': setup})
        key_ = ValidationKey(**{**asdict(key_), 'template_sha256': template.fingerprint})
        config.update(template=asdict(template), validation_key=asdict(key_))
        write_json_atomic(job / 'campaign_config.json', config)
    CR._load.cache_clear()
    CR._load(str((job / 'campaign_config.json').resolve()))   # native loader: fingerprints, split, scale states, routes
    from ispatialgym.sampling.campaign_class_routes import resolve
    resolve(config)

    # ---- gate (refuse on non-zero)
    write_json_atomic(job / 'preparation_binding.json', {'configuration_id': cid})
    p = subprocess.run(['python3', str(GATE), 'check', str(job / 'campaign_config.json'), '--binding', str(job / 'preparation_binding.json'),
                        '--giveups', str(GIVEUPS)], capture_output=True, text=True)
    (job / 'gate_author.txt').write_text(f'rc={p.returncode}\n{p.stdout}\n{p.stderr}')
    if p.returncode != 0:
        (job / 'AUTHOR_REFUSED.json').write_text(json.dumps({'gate_rc': p.returncode, 'out': p.stdout}))
        raise SystemExit(f'gate check rc={p.returncode}: {p.stdout} {p.stderr}')

    # ---- settings / job.env / run.sh
    prop = proposal_row(cls)
    require(prop['status'] == 'RUN', f'{cls} is {prop["status"]} in the wallclock proposal')
    L, Lm, cap = int(prop['train_group_limit_s']), int(prop['train_member_limit_s']), int(prop['demo_safety_cap_s'])
    seconds = int(min(14400, 1500 + groups * L + cap + 420))
    start = INDEX_BASE + k * 100000
    idx = read(EVAL_INDEX)
    es = read(can['dir'] / 'settings.json')
    settings = {'worker_id': 'og-isgtrain-' + key, 'state_dir': str(job / 'online'), 'results': str(job / 'results.jsonl'),
                'groups': groups, 'target_members': 3, 'minimum_members': 1, 'group_limit_s': L, 'member_limit_s': Lm,
                'demo_safety_cap_s': cap,
                'magnitude_range': idx.get('magnitude_union', {}).get(cls) or es.get('magnitude_range', [0.0, 1.0]),
                'index_count': 100000, 'require_qualifying_groups': False, 'registry': str(job / 'registry_private')}
    write_json_atomic(job / 'settings.json', settings)
    env = dict(can['env'])
    for name in ('TEMPLATE', 'TEMPLATE_ID', 'COUNTED_RUN', 'COUNTED_SNAPSHOT', 'COUNTED_TEMPLATE', 'EVAL_INDEX', 'KEEP_SCENE',
                 'ISGF_WALLCLOCK_POLICY', 'TAG'):
        env.pop(name, None)
    env.update(ISGE_CODE=str(SNAP), ISGF_SCRIPTS=str(SNAP / 'scripts/trainfast'), ISGF_ROOT=str(R),
               ISGF_JB_NFS=str((T if PRODUCTION['on'] else VA) / 'jobs'), ISGF_PREDEMO_ROOT=str(T if PRODUCTION['on'] else VA),
               ISGF_SAMPLING_CACHE=str(job / 'sampling_cache'),
               ISPATIALGYM_CAMPAIGN_CONFIG=str(job / 'campaign_config.json'), ISPATIALGYM_ONLINE_GROUPS=str(job / 'settings.json'),
               ISPATIALGYM_MAP_OVERLAY_SHARDS=str(job / 'map_overlay'),
               ISPATIALGYM_GRASP_LIBRARY_V2=str(job / 'deps/glib'), ISPATIALGYM_REACH_MAPS=str(job / 'deps/maps'),
               ISPATIALGYM_REACH_MAP_ANCHORS=str(job / 'deps/anchors.json') if (job / 'deps/anchors.json').exists()
               else env.get('ISPATIALGYM_REACH_MAP_ANCHORS', str(BASE_V2 / 'anchors.json')),
               ISPATIALGYM_FAST_COLLECTOR_PREBUILD='0', ISPATIALGYM_FAST_BAND_SHARE='0',
               ALIAS=variant, ALIAS_KEY=variant, PICK=variant, ISPATIALGYM_ALIAS_MEMORY_ALIAS=variant,
               VARIANT_CUSTOMS=str(job / 'customs.json'), VARIANT_BDDL_ROOT=str(job / 'bddl'), VARIANT_SCENE=scene,
               START=str(start), JOB_TAG=key, KCOLLECT=str(kcollect), N='1', TIME_LIMIT_S=str(seconds),
               SAMPLE_LIMIT_S=str(seconds - 600), DEMO_LIMIT_S=str(cap))
    if probe:
        env['ISPATIALGYM_STABLE_SCENE_DIR'] = str(job / 'deps/stable_scene')
    # OmniGibson copies its .kit into EXP_PATH on every launch: a private copy of the campaign overlay per source (eval jobs did the same)
    # keep the symlinks: the overlay's .kit files link into the env's isaacsim/apps and Kit resolves extension folders relative to
    # the link target - dereferenced copies broke that (Kit's prebundled cffi 1.17.1 vs the env's _cffi_backend 2.1.1, va_82..84)
    shutil.copytree(R / 'isaacsim_overlay/apps', job / 'isaacsim_overlay/apps', symlinks=True)
    env['EXP_PATH'] = env['ISGF_EXP_PATH'] = str(job / 'isaacsim_overlay/apps')
    extra = re.sub(r'--wallclock \d+(\.\d+)?', f'--wallclock {cap}', env.get('ISGF_COLLECT_EXTRA', ''))
    if '--wallclock' not in extra:
        extra = (extra + f' --wallclock {cap}').strip()
    env['ISGF_COLLECT_EXTRA'] = extra
    leftovers = [k2 for k2, v in env.items() if 'online_grouped_r1/production_r1' in v or 'online_grouped_r1/code_' in v]
    require(not leftovers, f'eval paths left in job.env: {leftovers}')
    (job / 'job.env').write_text(dump_env(env))
    (job / 'sampling_cache').mkdir()
    wrapper = Path(__file__).resolve().parent / 'job_variant.sh'
    shutil.copyfile(wrapper, job / 'job_variant.sh')
    sdk = re.search(r'--expected-sdk-sha256 ([0-9a-f]{64})', can['run']).group(1)
    run = f"""#!/bin/bash
set -euo pipefail
umask 0002
unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_SHLVL _CE_CONDA _CE_M PYTHONHOME
export PATH=/vision/u/minyeong/miniconda3/envs/ispatialgym-vectorized-envs/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
source /vision/u/minyeong/miniconda3/etc/profile.d/conda.sh
conda activate ispatialgym-vectorized-envs
hash -r
export OPTIX_CACHE_PATH=/tmp/isgtrain_optix_${{SLURM_JOB_ID:-local}}
python -c 'import sys,numpy; assert sys.version_info[:2] == (3,11); print("NATIVE_ENV_READY",sys.executable,numpy.__version__,flush=True)'
date +%s > {job}/job_start_unix
# dispatch-time gate (the manifest can move): skip on any non-zero exit
python3 {GATE} check {job}/campaign_config.json --binding {job}/preparation_binding.json --giveups {GIVEUPS} > {job}/gate_dispatch.txt 2>&1 || {{ echo GATE_SKIP; exit 3; }}
exec {PY} {OBSERVER} --output {job}/runtime.json --snapshot {SNAP} --expected-sdk-sha256 {sdk} --class-id {cls} --request-sha256 {sha(job / 'settings.json')} --phase generation -- bash {job}/job_variant.sh {job}
"""
    (job / 'run.sh').write_text(run)
    os.chmod(job / 'run.sh', 0o775)
    latest = read(RP / 'validity/LATEST.json')
    source = {'owner': 'train_gen_20261114/variant_authoring_r1', 'workstream': 'train', 'key': key, 'slot_k': k, 'class_id': cls,
              'configuration_id': cid, 'route': 'B', 'augment_seed': augment_seed, 'groups': groups, 'job_dir': str(job), 'seconds': seconds, 'start': start,
              'index_count': 100000, 'snapshot': str(SNAP), 'code_sha256': CODE_SHA256,
              'release': ref(RELEASE_RECORD), 'native_commit': read(RELEASE_RECORD)['candidate_commit'],
              'author': ref(Path(__file__).resolve()), 'wrapper_job': ref(job / 'job_variant.sh'),
              'settings': ref(job / 'settings.json'), 'campaign_config': ref(job / 'campaign_config.json'),
              'job_env': ref(job / 'job.env'), 'wrapper': ref(job / 'run.sh'), 'customs': ref(job / 'customs.json'),
              'preparation_binding': {**{x: row.get(x) for x in ('configuration_id', 'variant_id', 'template_key', 'identity_sha256')},
                                      'binding': ref(job / 'preparation_binding.json'),
                                      'manifest_revision': latest['revision'], 'manifest_sha256': latest['sha256']},
              'proposal': ref(PROPOSAL), 'limits': {'group_limit_s': L, 'member_limit_s': Lm, 'demo_safety_cap_s': cap,
                                                    'subtemplate_giveup_s': int(prop['subtemplate_giveup_s'])},
              'scales': {n: asdict(s) for n, s in scales.items()}, 'inherited_model_keyed_generation': inherited,
              'dropped_recovery_options': dropped, 'prepared_dependencies': used, 'variant': variant,
              'category_changes': {r: list(v) for r, v in changed.items()}, 'dropped_generation': dropped_generation, 'kcollect': kcollect, 'authored_at': time.time()}
    write_json_atomic(job / 'source.json', source)
    return source


# ------------------------------------------------------------------------------------------------ report

MISS = re.compile(r'no map|find_prefix|inside_surface|missing_handle|map miss|no reach map', re.I)


def report(job):
    job = Path(job)
    src = read(job / 'source.json')
    events = [json.loads(l) for l in (job / 'online/events.jsonl').read_text().splitlines()] if (job / 'online/events.jsonl').exists() else []
    t0 = float((job / 'job_start_unix').read_text()) if (job / 'job_start_unix').exists() else None
    first = lambda name: next((e['at'] for e in events if e['event'] == name), None)
    groups = [e for e in events if e['event'] == 'group_finalized']
    log = (job / 'sample.log').read_text(errors='replace') if (job / 'sample.log').exists() else ''
    misses = sorted({m.group(0).lower() for m in MISS.finditer(log)})
    lines = [l for l in log.splitlines() if MISS.search(l)][:10]
    metas = list((job / 'inst').glob('*/*/template_meta.json'))
    meta = read(metas[0]) if metas else {}
    cfg = read(job / 'campaign_config.json')
    predicted = cfg['template']['scene_setup']['inst_to_name']
    rel = lambda t: None if (t is None or t0 is None) else round(t - t0, 1)
    out = {'key': src['key'], 'class_id': src['class_id'], 'configuration_id': src['configuration_id'],
           'jobid': (job / 'jobid').read_text().strip() if (job / 'jobid').exists() else None,
           'worker_started_s': rel(first('worker_started')), 'first_group_started_s': rel(first('group_started')),
           'first_goal_selected_s': rel(first('goal_selected')), 'first_member_accepted_s': rel(first('member_accepted')),
           'groups_finalized': len(groups), 'members_per_group': [g.get('accepted') for g in groups],
           'members': {s: sum(e['event'] == 'member_' + s for e in events) for s in ('accepted', 'rejected', 'expired')},
           'reject_reasons': sorted({e.get('reason') for e in events if e['event'] == 'member_rejected'}),
           'map_misses': misses, 'map_miss_lines': lines,
           'template_dir': str(metas[0].parent) if metas else None,
           'inst_to_name_matches': ({k: v for k, v in (meta.get('inst_to_name') or {}).items() if k in predicted} == predicted) if meta else None,
           'scene_load_s': meta.get('scene_load_s'), 'sampler_rejections': meta.get('rejections')}
    write_json_atomic(job / 'variant_report.json', out)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    a = sub.add_parser('author')
    a.add_argument('--configuration-id', required=True)
    a.add_argument('--k', type=int, default=0, help='test slot 80..89 (ignored with --production)')
    a.add_argument('--production', action='store_true', help='train root + shared ordinal (t2611_<n>); giveups in the train root')
    a.add_argument('--groups', type=int, default=2)
    a.add_argument('--kcollect', type=int, default=2)
    a.add_argument('--dry-run', action='store_true', help='author under dryrun/ without allocating the slot (never dispatch it)')
    a.add_argument('--lighting-json', default=None, help='campaign lighting dict (default {})')
    a.add_argument('--removed-objects', default='', help='comma-separated scene object names to park (default none)')
    a.add_argument('--augment-seed', type=int, default=None, help='draw lighting + scene removals (p=0.5 each) from this seed')
    a.add_argument('--force-augment', action='store_true', help='tests only: advance the seed until both draws are non-empty')
    r = sub.add_parser('report')
    r.add_argument('job')
    args = ap.parse_args()
    if args.cmd == 'author':
        if args.production:
            PRODUCTION['on'] = True
            global GIVEUPS
            GIVEUPS = T / 'subtemplate_giveups.jsonl'
        s = author(args.configuration_id, args.k, args.groups, args.kcollect, dry=args.dry_run,
                   lighting=json.loads(args.lighting_json) if args.lighting_json else None,
                   removed_objects=[x for x in args.removed_objects.split(',') if x],
                   augment_seed=args.augment_seed, force_augment=args.force_augment)
        print(json.dumps({x: s[x] for x in ('key', 'class_id', 'configuration_id', 'job_dir', 'seconds', 'start')}))
    else:
        print(json.dumps(report(args.job), indent=1))


if __name__ == '__main__':
    main()
