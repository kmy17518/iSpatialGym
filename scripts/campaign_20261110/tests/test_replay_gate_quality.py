import json

import pytest

from scripts.campaign_20261110.replay_gate_quality import assess,digest


def fixture(root,annotated=True):
    tro=root/'source.json';tro.write_text(json.dumps({'ispatialgym':{'instance_id':'same',**({'campaign':{}} if annotated else {})}}))
    report=root/'replay/same/meta.json';report.parent.mkdir(parents=True)
    value={'instance_id':'same','mode':'replay','status':'success','success':True,'goal_override':None,
           'qa':{'enabled':True,'hard_triggered':[],'metrics':{}},'loader':{'campaign_restoration':{'scale_verified':True}}}
    report.write_text(json.dumps(value))
    gate={'success':True,'replay_rc':0,'replay_mode':'physics_action_open_loop','instance':str(tro),'tro_sha256':digest(tro),
          'replay_meta_paths':[str(report)],'result':{'success':True,'qa_hard':[],'goal_override':None},
          'scene_identity':{'verified_replay':True,'replay_report':str(report),'replay_report_sha256':digest(report)}}
    return gate,report,value


@pytest.mark.parametrize('failure',['hard_list','hard_metric','qa_disabled','restoration','override'])
def test_result_summary_success_cannot_override_native_rejection(tmp_path,failure):
    gate,report,value=fixture(tmp_path)
    if failure=='hard_list':value['qa']['hard_triggered']=['goal_occlusion']
    if failure=='hard_metric':value['qa']['metrics']['goal_occlusion']={'mode':'hard','triggered':True}
    if failure=='qa_disabled':value['qa']['enabled']=False
    if failure=='restoration':value['loader']['campaign_restoration']['scale_verified']=False
    if failure=='override':value['goal_override']={'changed':True}
    report.write_text(json.dumps(value));gate['scene_identity']['replay_report_sha256']=digest(report)
    assert not assess(gate,tmp_path/'gate.json')['passed']
    assert gate['success'] is True  # Original evidence is never edited by an audit.


def test_baseline_does_not_require_annotation_restoration_but_report_must_be_exact(tmp_path):
    gate,report,value=fixture(tmp_path,False);value.pop('loader');report.write_text(json.dumps(value))
    gate['scene_identity']['replay_report_sha256']=digest(report)
    assert assess(gate,tmp_path/'gate.json')['passed']
    value['instance_id']='different';report.write_text(json.dumps(value))
    assert not assess(gate,tmp_path/'gate.json')['passed']


def test_unchanged_annotated_success_passes_but_mutated_report_does_not(tmp_path):
    gate,report,value=fixture(tmp_path)
    assert assess(gate,tmp_path/'gate.json')['passed']
    report.write_text(report.read_text()+' ')
    assert not assess(gate,tmp_path/'gate.json')['passed']
