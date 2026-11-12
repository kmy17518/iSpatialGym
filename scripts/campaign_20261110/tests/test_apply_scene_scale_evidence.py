from copy import deepcopy
import pytest
from scripts.campaign_20261110.apply_scene_scale_evidence import apply, digest


def fixture():
    role={'role':'target','model_key':'chair/abc','asset_sha256':'asset','applied_scale':None}
    row={'id':'choice','configuration_id':'old','class_id':'class','scene':'room','rooms':['r'],
         'roles':[role], 'dependencies':['dep','cached'],'exact_scales_resolved':False}
    expansion={'rows':[row]}
    source={'path':'scene.json','sha256':'scenehash'}
    proof={'rows':[{'configuration_id':'old','roles':[{**role,'applied_scale':[1,2,3],
          'state':'literal_exact_scene_scale_observed','binding_unambiguous':True,
          'scene_source':source,'matching_instances':['chair_abc_0']}]}]}
    dep={'id':'dep','kind':'query','inputs':{'target':role},'status':'verified','artifact':'oldproof',
         'evidence':['oldproof'],'active_jobs':['job'],'unresolved_inputs':['exact native target default and applied scale'],
         'consumers':['old'],'required_by_classes':['class'],'physical_artifact_id':'oldphysical'}
    cached={**dep,'id':'cached','kind':'asset','inputs':{'asset_sha256':'asset'},'unresolved_inputs':[]}
    jobs={'rows':[dep,cached]}
    check={'verified':True,'sources':{'scene.json':{'sha256':'scenehash'}}}
    return expansion,proof,jobs,check


def test_rekey_invalidates_changed_inputs_preserves_unrelated_cache():
    args=fixture(); original=deepcopy(args)
    expansion,jobs,receipt=apply(*args)
    assert args==original
    row=expansion['rows'][0]
    assert row['configuration_id']!='old' and row['roles'][0]['applied_scale']==[1,2,3]
    changed=next(x for x in jobs['rows'] if x['kind']=='query')
    assert changed['id']!='dep' and changed['status']=='missing'
    assert changed['artifact'] is None and not changed['active_jobs'] and not changed['evidence']
    cached=next(x for x in jobs['rows'] if x['id']=='cached')
    assert cached['artifact']=='oldproof' and cached['status']=='verified'
    assert cached['consumers']==[row['configuration_id']]
    assert receipt['summary']['exact_scales_missing']==0


@pytest.mark.parametrize('bad',['ambiguous','changed_source','wrong_asset','already_scaled'])
def test_fail_closed_on_mismatched_evidence(bad):
    args=fixture(); measured=args[1]['rows'][0]['roles'][0]
    if bad=='ambiguous': measured['binding_unambiguous']=False
    if bad=='changed_source': args[3]['sources']['scene.json']['sha256']='wrong'
    if bad=='wrong_asset': measured['asset_sha256']='wrong'
    if bad=='already_scaled': args[0]['rows'][0]['roles'][0]['applied_scale']=[1,1,1]
    with pytest.raises(ValueError): apply(*args)


def profile_fixture():
    args = fixture(); row = args[0]['rows'][0]
    row.update(class_id='inter_object_relation_satisfaction-under_large_scale_01', object_scale_slot=1)
    row['roles'][0]['scale_profile'] = 'profile'
    proof = args[1]['rows'][0]['roles'][0]
    proof.update(state='literal_persisted_target_scale_draw', native_default_source=proof['scene_source'],
                 profile_source={'path':'profile.json','sha256':'profilehash'}, scale_profile='profile', object_scale_slot=1)
    args[3]['verified_profile_roles'] = [digest(proof)]
    args[2]['rows'][0]['unresolved_inputs'].append('native physical pair fit')
    return args


def test_persisted_target_draw_preserves_physical_pair_fit_gate():
    expansion, jobs, receipt = apply(*profile_fixture())
    role = expansion['rows'][0]['roles'][0]
    assert role['applied_scale'] == [1,2,3] and role['scale_rule'] == 'target_two_persisted_draws'
    assert 'scene_scale_evidence' not in role
    query = next(r for r in jobs['rows'] if r['kind'] == 'query')
    assert query['status'] == 'blocked' and query['unresolved_inputs'] == ['native physical pair fit']
    assert not query['evidence'] and query['physical_artifact_id'] is None


@pytest.mark.parametrize('bad', ['profile', 'slot', 'class', 'unverified'])
def test_target_profile_does_not_resolve_replacement_or_other_scale(bad):
    args = profile_fixture(); row = args[0]['rows'][0]
    if bad == 'profile': row['roles'][0]['scale_profile'] = 'wrong'
    if bad == 'slot': row['object_scale_slot'] = 0
    if bad == 'class': row['class_id'] = 'configuration_matching-articulation_open_large_scale_01'
    if bad == 'unverified': args[3]['verified_profile_roles'] = []
    with pytest.raises(ValueError): apply(*args)


def test_revised_query_reuses_identical_verified_request_independent_of_order():
    args = fixture()
    changed = deepcopy(args[2]['rows'][0]); changed['inputs']['target']['applied_scale'] = [1,2,3]
    changed['inputs']['target']['scale_rule'] = 'literal_exact_scene_initialization'
    changed['id'] = 'request_' + digest({'kind':changed['kind'],'inputs':changed['inputs']})
    changed.update(unresolved_inputs=[], artifact='exact-positive-native-proof', evidence=['exact-positive-native-proof'])
    args[2]['rows'].append(changed)
    jobs = apply(*args)[1]['rows']
    query = next(r for r in jobs if r['kind'] == 'query')
    assert query['id'] == changed['id']
    assert query['status'] == 'verified' and query['artifact'] == 'exact-positive-native-proof'
