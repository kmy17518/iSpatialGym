from dataclasses import asdict, replace
from pathlib import Path

import pytest

from scripts.campaign_20261110 import controller_held_sources as sources, contracts as c
from scripts.campaign_20261110.tests.test_controller_admission import anchor_fixture


def fixture(root, monkeypatch):
    f, template, config = anchor_fixture(root, monkeypatch, split='eval_model')
    template = replace(template, task='object_scaling', class_id='object_scaling-apple')
    library = root/'v2_native_release'
    for name in ('glib', 'maps'):
        (library/name).mkdir(parents=True)
    (f.job/'job.env').write_text(f'ISPATIALGYM_GRASP_LIBRARY_V2={library}/glib\nISPATIALGYM_REACH_MAPS={library}/maps\n')
    standards=root/'standards.json'; c.write_json_atomic(standards, {'templates':[asdict(template)]})
    cfg={'root':str(root), 'campaign_id':'campaign', 'started_at':f.since, 'standard_contracts':str(standards),
         'physics_sha256':'a'*64, 'data_sha256':'b'*64, 'seeds':[], 'excluded_classes':[],
         'seed_catalog':str(root/'controllers/models/catalog.json')}
    proof={'template':asdict(template), 'source_job':str(f.job), 'source_gate':{'path':str(f.gate),'sha256':c.file_digest(f.gate)},
           'config':{'path':str(config),'sha256':c.file_digest(config)},
           'validation_key':{'physics_sha256':'a'*64,'data_sha256':'b'*64}}
    allocation={'job_dir':str(f.job),'ledger':str(root/'manifests/allocations.tsv')}
    ledger=Path(allocation['ledger']);ledger.parent.mkdir(exist_ok=True)
    ledger.write_text('job_dir\n'+str(f.job)+'\n')
    return cfg,proof,allocation,library


def test_held_model_seed_keeps_actual_native_library_and_never_becomes_training(tmp_path, monkeypatch):
    cfg,proof,allocation,library=fixture(tmp_path,monkeypatch)
    spec=sources.make_seed(proof,allocation,cfg,tmp_path)
    assert spec['v2_root']==str(library) and spec['groups_per_batch']==1
    assert spec['seed_config']==proof['config']['path'] and spec['source_gate_sha256']==proof['source_gate']['sha256']
    with pytest.raises(c.ContractError,match='another refill controller'):
        sources.make_seed(proof,allocation,{**cfg,'excluded_classes':['object_scaling-apple']},tmp_path)
    proof['template']['split']='train'
    with pytest.raises(c.ContractError,match='requires model evaluation'):
        sources.make_seed(proof,allocation,cfg,tmp_path)


def test_unmatched_map_release_or_changed_campaign_physics_is_rejected(tmp_path, monkeypatch):
    cfg,proof,allocation,library=fixture(tmp_path,monkeypatch)
    with pytest.raises(c.ContractError,match='physics/data'):
        sources.make_seed(proof,allocation,{**cfg,'physics_sha256':'c'*64},tmp_path)
    (Path(allocation['job_dir'])/'job.env').write_text(
        f'ISPATIALGYM_GRASP_LIBRARY_V2={library}/glib\nISPATIALGYM_REACH_MAPS={tmp_path}/different/maps\n')
    with pytest.raises(c.ContractError,match='release differs'):
        sources.make_seed(proof,allocation,cfg,tmp_path)


def test_model_catalog_is_reaudited_then_pinned_against_rewrite(tmp_path, monkeypatch):
    cfg,proof,allocation,_=fixture(tmp_path,monkeypatch)
    spec=sources.make_seed(proof,allocation,cfg,tmp_path)
    sources.append_seed(cfg['seed_catalog'],spec,'campaign')
    seen=[]
    def audit(*args,**kwargs):
        seen.append(kwargs['permitted_split']); return proof
    monkeypatch.setattr(sources,'source_proof',audit)
    monkeypatch.setattr(sources,'allocations',lambda root:[allocation])
    state={}
    assert sources.load_seeds(cfg,state)==[spec] and seen==['eval_model']
    assert sources.load_seeds(cfg,state)==[spec] and seen==['eval_model']
    raw=sources.read(cfg['seed_catalog']);raw['seeds'][0]['v2_root']='unreviewed'
    c.write_json_atomic(cfg['seed_catalog'],raw)
    with pytest.raises(c.ContractError,match='removed or changed'):
        sources.load_seeds(cfg,state)


def test_category_source_requires_explicit_task_and_preserves_held_split(tmp_path, monkeypatch):
    cfg, proof, allocation, library = fixture(tmp_path, monkeypatch)
    cfg.update(source_split='eval_category', admitted_tasks=['object_scaling'])
    proof['template'].update(split='eval_category', class_id='object_scaling-tennis_ball')
    spec = sources.make_seed(proof, allocation, cfg, tmp_path)
    assert spec['id'].startswith('held_category_') and spec['task'] == 'object_scaling'
    assert spec['class_id'] == 'object_scaling-tennis_ball' and spec['v2_root'] == str(library)
    assert spec['seed_config'] == proof['config']['path'] and spec['existing_queues'] == []
    with pytest.raises(c.ContractError, match='explicit held task ownership'):
        sources.make_seed(proof, allocation, {**cfg, 'admitted_tasks': []}, tmp_path)
    with pytest.raises(c.ContractError, match='outside owned tasks'):
        sources.make_seed(proof, allocation, {**cfg, 'admitted_tasks': ['mental_rotation']}, tmp_path)
    proof['template']['split'] = 'train'
    with pytest.raises(c.ContractError, match='requires category evaluation'):
        sources.make_seed(proof, allocation, cfg, tmp_path)


def test_category_catalog_reaudits_category_proof_and_pins_one_source_per_task(tmp_path, monkeypatch):
    cfg, proof, allocation, _ = fixture(tmp_path, monkeypatch)
    cfg.update(source_split='eval_category', admitted_tasks=['object_scaling'])
    proof['template'].update(split='eval_category', class_id='object_scaling-tennis_ball')
    spec = sources.make_seed(proof, allocation, cfg, tmp_path)
    sources.append_seed(cfg['seed_catalog'], spec, 'campaign')
    observed = []
    def audit(*args, **kwargs):
        observed.append(kwargs['permitted_split'])
        return proof
    monkeypatch.setattr(sources, 'source_proof', audit)
    monkeypatch.setattr(sources, 'allocations', lambda root: [allocation])
    assert sources.load_seeds(cfg, {}) == [spec] and observed == ['eval_category']
    second = {**spec, 'id': spec['id'] + '_different', 'class_id': 'object_scaling-another'}
    sources.append_seed(cfg['seed_catalog'], second, 'campaign')
    with pytest.raises(c.ContractError, match='multiple pinned sources'):
        sources.load_seeds(cfg, {})
