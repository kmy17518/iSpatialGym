import pytest

from scripts.campaign_20261110.camera_native import prepare


def test_filtered_manifest_resolves_preserved_inputs_and_optional_blockers(tmp_path):
    source=tmp_path/'v1/model';source.mkdir(parents=True)
    (source/'customs.json').write_text('{}');(source/'bddl').mkdir();(source/'bddl/problem.bddl').write_text('(define (problem p))')
    entry={'task':'camera_relocalization-obstructed','alias':'camera_relocalization-obstructed-chair_1',
        'class_id':'camera_relocalization-obstructed-chair_1','scene':'s','room':'r','environment':{'ISPATIALGYM_CAMPAIGN_OBSTACLE_GOAL':'1'},
        'sample_argv':['python','-m','ispatialgym.sampling.sample_isg','--variant','camera_relocalization-obstructed-chair_1',
        '--scene','s','--room','r','--customs','${MATERIALIZED}/model/customs.json','--bddl-root','${MATERIALIZED}/model/bddl',
        '--templates','1','--instances-per-template','1','--no-write-back','--out','${CAMPAIGN_OUT}/validation/new']}
    options=dict(root=tmp_path,manifest_root=tmp_path/'v2',inputs_root=tmp_path/'v1',output_root=tmp_path/'run',index=123)
    argv,output,env,refs=prepare(entry,**options)
    assert str(source/'customs.json') in argv and argv[-4:]==['--start-index','123','--time-budget-s','1800']
    assert len(refs)==2 and env==entry['environment'] and output==tmp_path/'run/validation/new'
    output.mkdir(parents=True)
    with pytest.raises(ValueError,match='already exists'):prepare(entry,**options)
    output.rmdir();entry['blockers']=['native model unready']
    with pytest.raises(ValueError,match='blocked'):prepare(entry,**options)
