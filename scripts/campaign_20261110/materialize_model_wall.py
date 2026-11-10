#!/usr/bin/env python3
"""Compile model-held wall alignment with reviewed frames and actual scene walls."""
import argparse
import copy
from dataclasses import asdict
import json
from pathlib import Path

from scripts.campaign_20261110.contracts import ObjectRef, Template, digest, file_digest, require, write_json_atomic
from scripts.campaign_20261110.group_lane import check_reservations
from scripts.campaign_20261110.readiness_catalog import ReadinessCatalog


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'snapshot', 'inventory', 'defaults', 'visual-review', 'frame-review', 'ready-source-contracts', 'reservations', 'reserved-asset-hashes', 'out'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--class-id')
    p.add_argument('--context-class')
    p.add_argument('--context-proof', type=Path)
    a = p.parse_args(); require(not a.out.exists() and a.out.resolve().is_relative_to(a.root.resolve()), 'new owned output required')
    policy, hashes, defaults, review, frames = [json.loads(path.read_text()) for path in
        (a.reservations, a.reserved_asset_hashes, a.defaults, a.visual_review, a.frame_review)]
    require(hashes['reservation_sha256'] == file_digest(a.reservations) == defaults['reservation_sha256'], 'reservation/default identity differs')
    require(defaults['status'] == 'native_defaults_and_geometry_recorded' and review['default_geometry_sha256'] == frames['default_geometry_sha256'] == file_digest(a.defaults), 'native defaults proof differs')
    require(frames['status'] == 'passed' and frames['visual_review_sha256'] == file_digest(a.visual_review), 'frame review proof differs')
    measured = {row['category'] + '/' + row['model']: row for row in defaults['objects']}
    b1k = a.snapshot / 'Behavior-1K'; catalog = ReadinessCatalog(b1k / 'bddl3/bddl/generated_data')
    source = a.inventory / 'frozen_customs_f24ea424.json'; originals = json.loads(source.read_text())
    records = {row['key']: row for row in json.loads((a.inventory / 'cluster_templates_with_scales.json').read_text())['records']}
    ready_sources = {row['class_id']: row for row in json.loads(a.ready_source_contracts.read_text())['templates']}
    from ispatialgym.sampling.bddl_from_iddl import RoleGrounding, render
    from ispatialgym.sampling.iddl import parse_file
    from ispatialgym.sampling.knowledge import synset_of_category
    from ispatialgym.sampling.v2.problem import GroundedProblem, iter_template_choices
    import numpy as np
    proposals = [row for row in policy['model_holdout_templates'] if row['task'] == 'alignment-world_vertical']
    require(len(proposals) == 5, 'expected five wall-alignment classes')
    if a.class_id:
        proposals = [row for row in proposals if row['class_id'] == a.class_id]
        require(len(proposals) == 1, 'unknown wall-alignment class')
    context_proof = None
    if a.context_class:
        require(a.class_id and a.context_proof, 'context override requires one class and fresh native context proof')
        matches = [row for row in json.loads(a.context_proof.read_text())['templates'] if row['class_id'] == a.context_class]
        require(len(matches) == 1 and matches[0]['scene_setup'].get('fresh_geometry_tros'), 'new context lacks fresh native geometry')
        context_proof = matches[0]
        for tro in context_proof['scene_setup']['fresh_geometry_tros']:
            require(file_digest(Path(tro['path'])) == tro['sha256'], 'context native evidence changed')
    a.out.mkdir(parents=True); templates, jobs = [], []
    for index, proposal in enumerate(proposals):
        cls = proposal['class_id']; target_role = next(row for row in proposal['roles'] if row['role'] == 'target')
        key = target_role['category'] + '/' + target_role['model']; target = measured[key]; frame = frames['frames'][key]
        require(target['ready'] and review['objects'][key]['status'] == 'passed' and key in policy['model_holdout_models'], 'unready/unreviewed/unreserved target')
        require(target['asset_sha256'] == frame['asset_sha256'] == hashes['ready_assets'][key]['asset_sha256'] == file_digest(Path(target['asset_path'])), 'target asset changed')
        require(target['default_scale'] == frame['default_scale'], 'reviewed target scale differs')
        record = records[cls]
        context_class = a.context_class or cls
        source_meta = records[context_class]['metadata']; scene = source_meta['scene']; rooms = source_meta['task_rooms']
        if context_proof:
            require(context_proof['scene'] == scene and context_proof['room'] == rooms[0], 'fresh context proof differs from scene binding')
        require(len(rooms) == 1, 'wall candidate requires one exact room')
        room = rooms[0]
        scene_path = b1k / 'datasets/2026-challenge-task-instances/scenes' / scene / 'json' / (scene + '_stable.json')
        init = json.loads(scene_path.read_text())['objects_info']['init_info']; roles = {'target': target}
        for role in ('floor', 'support'):
            binding = next(row for row in source_meta['bindings'] if row['role'] == role)
            obj = init[binding['scene_object_name']]['args']
            if not catalog.status(obj['category'], obj['model'])['ready']:
                repaired = ready_sources[context_class]
                require(repaired['scene'] == scene and repaired['room'] == room, 'ready source repair changes task room')
                kind = 'floor' if role == 'floor' else 'wall'
                replacements = [row for row in repaired['objects'] if row['semantic_kind'] == kind and row['ready']]
                require(len(replacements) == 1, 'ready structural replacement is ambiguous')
                replacement = replacements[0]
                obj = init[repaired['scene_setup']['inst_to_name'][replacement['instance']]]['args']
                require((obj['category'], obj['model'], obj['scale']) == (replacement['category'], replacement['model'], replacement['default_scale']), 'ready source structural identity differs')
            # House-wide walls have an explicit empty room tag in the native
            # stable scene. Keep the exact existing source wall binding; the
            # room-scoped floor and native wall-spot geometry still constrain
            # this task. A nonempty incompatible room tag remains an error.
            if role == 'floor' or obj.get('in_rooms'):
                require(room in obj.get('in_rooms', []), 'structural binding outside task room')
            require(obj['category'] == ('floors' if role == 'floor' else 'walls'), 'expected structural support')
            status = catalog.status(obj['category'], obj['model']); require(status['ready'], 'unready structural role')
            asset = b1k / 'datasets/behavior-1k-assets/objects' / obj['category'] / obj['model'] / 'usd' / (obj['model'] + '.encrypted.usd')
            roles[role] = dict(category=obj['category'], model=obj['model'], default_scale=obj['scale'], asset_sha256=file_digest(asset),
                               ready=True, readiness_source=status['readiness_source'], scene_object_name=obj['name'])
        entry = copy.deepcopy(originals[record['row']['alias']]); iddl = parse_file(str(a.snapshot / 'ispatialgym/assets/iddl' / entry['iddl']))
        entry['roles']['target'].update(category=target['category'], representative=target['model'], dataset=target['dataset'], scale={key: target['default_scale']})
        for name in ('orientation_equivalences', 'rest_pose', 'rest_hint_wxyz', 'grasp_keys'): entry['roles']['target'].pop(name, None)
        entry['generation']['front_axis'] = {key: frame['front_axis']}; entry['generation']['up_axis'] = {key: frame['up_axis']}
        for name in list(entry['generation']):
            if name.startswith('_') or name == 'remove_objects': entry['generation'].pop(name)
        for name in list(entry):
            if name.startswith('fallback') or name == 'superseded_by': entry.pop(name)
        entry['generation'].update(templates_max=1, instances_per_template=1); entry['constraints'] = []
        entry['scenes'] = {scene: dict(rooms=rooms, bindings={'floor': {'room': room}, 'support': [roles['support']['scene_object_name']]},
                                           whitelist={'target': {target['category']: [target['model']]}}, blacklist={})}
        entry['roles'] = {name: entry['roles'][name] for name in iddl.roles}
        alias = iddl.variant_template.format(target=target['category'], support='walls')
        did = 31000 + index if not a.context_class else 31100 + int(cls.rsplit('_', 1)[1])
        entry['definition_id'] = did
        grounds = {role: RoleGrounding(role, 'scene', same_as='floor') if role == 'floor' else RoleGrounding(role, 'sampleable' if role == 'target' else 'scene',
                   category=roles[role]['category'], synset=synset_of_category(roles[role]['category'])) for role in iddl.roles}
        rendered = render(iddl, grounds, did, alias, 'alignment'); directory = a.out / cls; directory.mkdir()
        bddl = directory / 'bddl/alignment' / ('problem' + str(did) + '.bddl'); bddl.parent.mkdir(parents=True); bddl.write_text(rendered.text)
        refs, names = [], {}
        for role, row in roles.items():
            instance = rendered.instances[role]
            refs.append(ObjectRef(instance=instance, role='target' if role == 'target' else 'support', category=row['category'], model=row['model'],
                asset_sha256=row['asset_sha256'], default_scale=tuple(row['default_scale']), semantic_kind='object' if role == 'target' else 'floor' if role == 'floor' else 'wall',
                ready=True, readiness_source=row['readiness_source']))
            if row.get('scene_object_name'): names[instance] = row['scene_object_name']
        proof = dict(defaults_sha256=file_digest(a.defaults), visual_review_sha256=file_digest(a.visual_review), frame_review_sha256=file_digest(a.frame_review),
                     native_scene_sha256=file_digest(scene_path), source_customs_sha256=file_digest(source), ready_source_contracts_sha256=file_digest(a.ready_source_contracts),
                     reservation_sha256=file_digest(a.reservations), production_released=False)
        if context_proof:
            proof.update(context_source_class=context_class, context_proof=str(a.context_proof), context_proof_sha256=file_digest(a.context_proof),
                         context_adaptation='Same target/default scale/task program; exact structural bindings from a separately sampled wall context')
        entry['campaign_provenance'] = proof; customs = directory / 'candidate_customs.json'; write_json_atomic(customs, {alias: entry})
        template_id = 'held_model__' + cls + ('__context_' + context_class if a.context_class else '')
        template = Template(template_id, proposal['task'], cls, 'eval_model', scene, room, tuple(refs), digest(dict(entry=entry, bddl_sha256=file_digest(bddl))),
                            {**proof, 'task_rooms': rooms, 'inst_to_name': names, 'runtime_bddl_sha256': file_digest(bddl), 'customs_alias': alias,
                             'reviewed_frame': dict(front_axis=frame['front_axis'], up_axis=frame['up_axis']), 'fresh_geometry': 'pending', 'fresh_demo': 'pending', 'fresh_replay': 'pending'})
        check_reservations(template, policy, hashes)
        gp = GroundedProblem.load(alias, str(customs), str(directory / 'bddl'))
        choices = list(iter_template_choices(gp, scene, np.random.default_rng(20261110), max_templates=1))
        require(len(choices) == 1 and choices[0].models['target'].model == target['model'], 'native target choice differs')
        templates.append({**asdict(template), 'fingerprint': template.fingerprint})
        jobs.append(dict(candidate_id=template.template_id, class_id=cls, task=proposal['task'], alias=alias, scene=scene, room=room, split='eval_model',
                         template_fingerprint=template.fingerprint, source_manifest=str((a.out / 'validation_jobs.json').resolve()), production_released=False,
                         status='native wall spot/visibility/contact geometry pending; retain existing native optional-map fallback',
                         sample_argv=['python', '-m', 'ispatialgym.sampling.sample_isg', '--variant', alias, '--customs', str(customs),
                                      '--bddl-root', str(directory / 'bddl'), '--scene', scene, '--templates', '1', '--instances-per-template', '1',
                                      '--max-attempts', '3', '--seed', '20261110', '--no-write-back', '--out', '${CAMPAIGN_OUT}/validation/' + cls]))
    write_json_atomic(a.out / 'contract_candidates.json', dict(templates=templates)); write_json_atomic(a.out / 'validation_jobs.json', dict(jobs=jobs))
    print(json.dumps(dict(candidates=len(jobs), manifest_sha256=file_digest(a.out / 'validation_jobs.json'))))


if __name__ == '__main__':
    main()
