import json
import pytest
from scripts.campaign_20261110.contracts import ContractError, file_digest, write_json_atomic
from scripts.campaign_20261110.source_support_inventory import checked_context, KEYS
from scripts.campaign_20261110.controller_held import source_environment


def fixture(root):
    scene=root/'stable/room_stable.json';write_json_atomic(scene,{'scene':'physical private scene'})
    identity={'scene':'room','name':'table_model_0','category':'table','model':'model','scale':1.}
    inventory=root/'measured/top.json';write_json_atomic(inventory,{**identity,'top':{'resolution':.02,'hull_xy':[[0.,0.],[1.,0.],[1.,1.],[0.,1.]]}})
    ref=lambda p:{'path':str(p),'sha256':file_digest(p)}
    measurement=root/'measured/measurement.json';write_json_atomic(measurement,{'native_inventory_record':ref(inventory),'source_scene':ref(scene),'object_name':identity['name'],'actual_scale':1.,'category':'table','model':'model'})
    entry={**identity,'scene_file':ref(scene),'inventory_record':ref(inventory),'measurement':ref(measurement)}
    manifest=root/'manifest.json';write_json_atomic(manifest,{'schema':'isg-private-support-inventory-v1','entries':[entry]})
    return {KEYS[0]:str(manifest),KEYS[1]:file_digest(manifest),'ISPATIALGYM_STABLE_SCENE_DIR':str(scene.parent)},entry


def test_private_measured_inventory_survives_held_refill_with_exact_provenance(tmp_path):
    environment,entry=fixture(tmp_path);job=tmp_path/'job';job.mkdir()
    (job/'job.env').write_text(''.join(k+'='+v+'\n' for k,v in environment.items()))
    extras,_=source_environment(job,root=tmp_path)
    assert all(k+'='+environment[k] in extras for k in KEYS)
    proof=checked_context(environment,tmp_path)
    assert proof['environment']=={k:environment[k]for k in KEYS}
    assert entry['measurement'] in proof['artifacts']


@pytest.mark.parametrize('field',['scene_file','inventory_record','measurement'])
def test_changed_exact_input_is_not_propagated(tmp_path,field):
    env,entry=fixture(tmp_path)
    from pathlib import Path
    Path(entry[field]['path']).write_text('{}')
    with pytest.raises(ContractError,match='evidence changed'):checked_context(env,tmp_path)


@pytest.mark.parametrize('field',['scene','scale','model'])
def test_rehashed_manifest_cannot_rebind_native_measurement(tmp_path,field):
    env,entry=fixture(tmp_path);p=env[KEYS[0]];manifest=json.loads(open(p).read())
    manifest['entries'][0][field]='other' if field!='scale' else 2.
    write_json_atomic(p,manifest);env[KEYS[1]]=file_digest(p)
    with pytest.raises(ContractError,match='differs'):checked_context(env,tmp_path)


def test_missing_half_or_changed_private_directory_fails_closed(tmp_path):
    env,_=fixture(tmp_path)
    with pytest.raises(ContractError,match='manifest and digest'):checked_context({KEYS[0]:env[KEYS[0]]},tmp_path)
    env['ISPATIALGYM_STABLE_SCENE_DIR']=str(tmp_path/'different')
    with pytest.raises(ContractError,match='private scene differs'):checked_context(env,tmp_path)
    assert checked_context({},tmp_path) is None
