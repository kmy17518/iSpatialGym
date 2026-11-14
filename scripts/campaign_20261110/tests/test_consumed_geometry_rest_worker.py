import json
from types import SimpleNamespace
from scripts.campaign_20261110 import consumed_geometry_rest_worker as worker


def test_geometry_worker_uses_native_rest_writer_without_grasp_trials(tmp_path):
    job = {'id':'exact','category':'magazine','model':'exact','scale':[1.0129]*3};seen={}
    def save(path,value):
        seen['candidate']=value
        open(path,'w').write(json.dumps(value))
    def cfg(**kwargs):seen['config']=kwargs;return kwargs
    def run(world,actual,path):
        assert actual==job
        lib={'category':'magazine','model':'exact','scale':[1.0129]*3,'grasps':[],
             'rests':{'r0':{'stable':True,'contact':True}}}
        open(path+'/library.json','w').write(json.dumps(lib))
        return {'n_reps':0,'n_valid':0}
    result=worker.measure({'job':job},tmp_path,SimpleNamespace(CANDIDATES='native',save_json=save),
        SimpleNamespace(ValidateConfig=cfg,TabletopWorld=lambda c:c,run_model=run))
    assert seen['config']=={'max_trials':0,'max_valid':0}
    assert seen['candidate']['representatives']==seen['candidate']['candidates']==[]
    assert result['stable_rest_ids']==['r0'] and result['grasp_credit'] is False
