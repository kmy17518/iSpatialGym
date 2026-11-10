#!/usr/bin/env python3
"""Freeze ready baseline room-family objects into the ten screened train rooms.

Produces runnable private customs/BDDL plus contract candidates, never releases.
All object scales come from source template initialization, including floors.
"""
import copy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
B1K = Path('/home/minyeongk/Desktop/projects/Behavior-1K-grasp-bands-6a6194600')
sys.path[:0] = [str(ROOT), str(B1K/'bddl3')]
from contracts import ObjectRef, Template, digest
from readiness_catalog import ReadinessCatalog
from ispatialgym.sampling.bddl_from_iddl import RoleGrounding, render
from ispatialgym.sampling.iddl import parse_file
from ispatialgym.sampling.knowledge import synset_of_category
from ispatialgym.sampling.v2.problem import GroundedProblem, SUBTASK_FAMILY
from ispatialgym.oracle.sidecar import subtask_of_variant


def file_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    inv = ROOT/'tmp/campaign_20261110/inventory'
    out = ROOT/'tmp/campaign_20261110/room_materialized'
    out.mkdir(parents=True, exist_ok=True)
    reservations = json.loads((inv/'split_reservations.json').read_text())
    excluded = set(reservations['train_exclude_models'])
    excluded_categories = set(reservations['train_exclude_categories'])
    excluded_rooms = {(r['scene'],r['room']) for r in reservations['category_holdout_rooms']}
    plan_path = inv/'train_room_plan.json'
    plan = json.loads(plan_path.read_text())
    records = {r['key']:r for r in json.loads((inv/'cluster_templates_with_scales.json').read_text())['records']}
    standards = json.loads((inv/'standard_contract_ready.json').read_text())['templates']
    customs_path = inv/'frozen_customs_f24ea424.json'
    originals = json.loads(customs_path.read_text())
    generated_path=inv/'generated_project_readiness_r1.json'
    floating_jobs=[]
    if generated_path.is_file():
        generated=json.loads(generated_path.read_text())['objects']
        source_standard=next(t for t in standards if t['class_id']=='mental_rotation-classroom_mock_up')
        source_record=records[source_standard['class_id']]
        for key,asset in sorted(generated.items()):
            if asset['family']!='mental_rotation' or asset['split_reservation']!='train':continue
            if not asset['ready'] or key in excluded or asset['category'] in excluded_categories:
                raise ValueError(f'unready/reserved generated train asset {key}')
            category,model=asset['category'],asset['model'];grasp_key=f'{category}__{model}__f'
            alias=f'mental_rotation-{category}_{model}'
            entry=copy.deepcopy(originals[source_record['row']['alias']])
            entry['roles']['target'].update(representative=model,dataset=asset['dataset'],
                grasp_keys={key:grasp_key},scale={key:asset['default_scale']},orientation_equivalences={key:[]})
            originals[alias]=entry
            record=copy.deepcopy(source_record);record['row']['alias']=alias
            for binding in record['metadata']['bindings']:
                if binding['role']=='target':
                    binding.update(category=category,model=model,dataset=asset['dataset'])
                    record['object_init_args'][binding['scene_object_name']].update(category=category,model=model,dataset_name=asset['dataset'],scale=asset['default_scale'])
            standard=copy.deepcopy(source_standard)
            for obj in standard['objects']:
                if obj['role']=='target':
                    obj.update(category=category,model=model,asset_sha256=asset['asset_sha256'],default_scale=asset['default_scale'],ready=True,readiness_source=asset['readiness_source'])
            standard['_record_override']=record;standard['_new_grasp_key']=grasp_key;standards.append(standard)
            floating_jobs.append({'id':f'{category}__{model}__s1__floating','category':category,'model':model,
                'scale':asset['default_scale'],'dataset':asset['dataset'],'constraint':'floating','rest_hint_wxyz':None,'rest_pose':None,
                'legacy_key':grasp_key,'kind':'floating','source':'campaign_generated_asset','floating_mode':'C','held_tasks':['mental_rotation'],
                'ready':True,'readiness_source':asset['readiness_source'],'asset_sha256':asset['asset_sha256'],
                'legacy_key_note':'New routing alias only; no legacy grasps or model geometry are imported.'})
    readiness = ReadinessCatalog(B1K/'bddl3/bddl/generated_data')
    scene_cache = {}
    definitions, jobs, templates, blocked = {}, [], [], []
    families = {'camera_relocalization-standard','camera_relocalization-obstructed','mental_rotation','object_scaling'}
    for standard in standards:
        task, class_id = standard['task'], standard['class_id']
        if task not in families:
            continue
        record = standard.get('_record_override') or records[class_id]
        alias = record['row']['alias']
        entry = copy.deepcopy(originals[alias])
        iddl = parse_file(str(ROOT/'ispatialgym/assets/iddl'/entry['iddl']))
        if subtask_of_variant(alias) != task:
            raise ValueError(f'non-native source alias {alias}')
        entry['roles'] = {r:entry['roles'][r] for r in iddl.roles if r in entry['roles']} | {
            r:s for r,s in entry['roles'].items() if r not in iddl.roles}
        pool = {}
        for b in record['metadata']['bindings']:
            role, category, model = b['role'], b['category'], b['model']
            if category == 'floors':
                continue
            if category in excluded_categories or category+'/'+model in excluded:
                raise ValueError(f'reserved train object {class_id}/{category}/{model}')
            pool.setdefault(role,{}).setdefault(category,[]).append(model)
            entry['roles'][role].setdefault('scale',{})[category+'/'+model] = record['object_init_args'][b['scene_object_name']]['scale']
        pool = {r:{c:sorted(set(ms)) for c,ms in cats.items()} for r,cats in pool.items()}
        entry['constraints'] = []
        for key in list(entry):
            if key.startswith('fallback') or key == 'superseded_by':
                entry.pop(key)
        for key in list(entry['generation']):
            if key.startswith('_') or key == 'remove_objects':
                entry['generation'].pop(key)
        entry['generation'].update(templates_max=1,instances_per_template=1)
        entry['scenes'] = {}
        for anchor in plan['tasks'][task]:
            scene, room = anchor['scene'], anchor['room']
            if (scene,room) in excluded_rooms:
                raise ValueError(f'reserved train room {scene}/{room}')
            block = entry['scenes'].setdefault(scene,{'rooms':[],'bindings':{},'whitelist':copy.deepcopy(pool),'blacklist':{}})
            block['rooms'].append(room)
            block['bindings']['floor'] = {'room':block['rooms'][0]}
        definition_id = 19000+len(definitions)
        entry['definition_id'] = definition_id
        groundings = {}
        for role in iddl.roles:
            spec = entry['roles'][role]
            group = entry['generation'].get('groups',{}).get(role)
            if group:
                members = [(c,synset_of_category(c)) for c,n in group['members'].items() for _ in range(n)]
                groundings[role] = RoleGrounding(role,'group',members=members)
            elif spec.get('category') == 'floors':
                groundings[role] = RoleGrounding(role,'scene',same_as='floor')
            else:
                category = spec.get('category')
                groundings[role] = RoleGrounding(role,'sampleable',category=category,synset=synset_of_category(category))
        activity = SUBTASK_FAMILY[task][0]
        rendered = render(iddl,groundings,definition_id,alias,activity,comment='Private train-room candidate; geometry/demo/replay pending.')
        bddl_path = out/'bddl'/activity/f'problem{definition_id}.bddl'
        bddl_path.parent.mkdir(parents=True,exist_ok=True)
        bddl_path.write_text(rendered.text)
        entry['campaign_provenance'] = {'source_class':class_id,'source_customs_sha256':file_sha(customs_path),
            'room_plan_sha256':file_sha(plan_path),'reservation_sha256':file_sha(inv/'split_reservations.json'),
            'status':'geometry_and_demo_and_replay_required'}
        definitions[alias] = entry
        for anchor in plan['tasks'][task]:
            scene, room = anchor['scene'], anchor['room']
            if scene not in scene_cache:
                path = B1K/'datasets/behavior-1k-assets/scenes'/scene/'json'/f'{scene}_best.json'
                scene_cache[scene] = (json.loads(path.read_text())['objects_info']['init_info'],file_sha(path))
            scene_info, scene_sha = scene_cache[scene]
            # This is the native floor_of_room iteration order. Fresh metadata
            # must confirm the binding before the contract can be promoted.
            floors = [x['args'] for x in scene_info.values() if x['args'].get('category')=='floors' and room in (x['args'].get('in_rooms') or [])]
            if not floors:
                blocked.append({'class_id':class_id,'scene':scene,'room':room,'reason':'no scene floor'});continue
            floor = floors[0]
            status = readiness.status('floors',floor['model'])
            if not status['ready']:
                blocked.append({'class_id':class_id,'scene':scene,'room':room,'reason':'floor readiness unresolved'});continue
            refs = [ObjectRef(**{**o,'default_scale':tuple(o['default_scale'])}) for o in standard['objects'] if o['semantic_kind']!='floor']
            refs.append(ObjectRef(instance='floor.n.01_1',role='support',category='floors',model=floor['model'],
                default_scale=tuple(floor['scale']),asset_sha256=file_sha(B1K/'datasets/behavior-1k-assets/objects/floors'/floor['model']/'usd'/(floor['model']+'.encrypted.usd')),
                semantic_kind='floor',ready=True,readiness_source=status['readiness_source']))
            candidate_id = 'train_room__'+digest({'class_id':class_id,'scene':scene,'room':room,'objects':[asdict(o) for o in refs]})[:20]
            template = Template(template_id=candidate_id,task=task,class_id=class_id,split='train',scene=scene,room=room,
                objects=tuple(refs),definition_sha256=digest({'entry':entry,'bddl_sha256':file_sha(bddl_path)}),scene_setup={
                    'task_rooms':[room],'inst_to_name':{'floor.n.01_1':floor['name']},'runtime_bddl_sha256':file_sha(bddl_path),
                    'customs_alias':alias,'scene_init_source_sha256':scene_sha,'room_plan':anchor,
                    'release_status':'candidate_only','fresh_metadata_binding_check':'required','fresh_demo':'pending','fresh_replay':'pending'})
            templates.append({**asdict(template),'fingerprint':template.fingerprint})
            jobs.append({'candidate_id':candidate_id,'class_id':class_id,'alias':alias,'scene':scene,'room':room,
                'template_fingerprint':template.fingerprint,'production_released':False,
                'status':'awaiting_fresh_floating_grasp_library' if standard.get('_new_grasp_key') else 'ready_for_geometry_validation',
                'required_new_grasp_key':standard.get('_new_grasp_key'),
                'sample_argv':['python','-m','ispatialgym.sampling.sample_isg','--variant',alias,'--customs','${ROOM_MATERIALIZED}/candidate_customs.json',
                    '--bddl-root','${ROOM_MATERIALIZED}/bddl','--scene',scene,'--room',room,'--templates','1','--instances-per-template','1',
                    '--max-attempts','3','--seed','20261110','--no-write-back','--out','${CAMPAIGN_OUT}/room_validation/'+candidate_id]})
    path = out/'candidate_customs.json'
    path.write_text(json.dumps(definitions,indent=2)+'\n')  # Role order is semantic.
    for alias in definitions:
        gp = GroundedProblem.load(alias,str(path),str(out/'bddl'))
        for anchor in plan['tasks'][gp.subtask]:
            gp.override_scene_room(anchor['scene'],anchor['room'])
            if gp.rooms(anchor['scene']) != [anchor['room']]:
                raise ValueError('native room override did not narrow the room')
    manifest = {'schema':'campaign-room-candidates-v1','reservation_sha256':file_sha(inv/'split_reservations.json'),
                'room_plan_sha256':file_sha(plan_path),'customs_sha256':file_sha(path),'jobs':jobs,'blocked':blocked,'production_released':0}
    (out/'validation_jobs.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    (out/'contract_candidates.json').write_text(json.dumps({'templates':templates},indent=2,sort_keys=True)+'\n')
    (out/'generated_mental_floating_jobs.json').write_text(json.dumps(floating_jobs,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'source_classes':len({j['class_id'] for j in jobs}),'native_variants':len(definitions),'candidates':len(jobs),'new_floating_jobs':len(floating_jobs),'blocked':blocked,'production_released':0}))


if __name__ == '__main__':
    main()
