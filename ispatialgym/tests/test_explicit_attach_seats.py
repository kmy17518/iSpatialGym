import hashlib
import json

import pytest

from ispatialgym.grasping import candidates as CA


def fixture(tmp_path):
    rec={'male':{'category':'cork','model':'m','scale':[.9]*3,'links':{
        'male_link':{'meta_link_id':'corkM','sub_id':'0','position_root':[0,0,-.01],
                     'orientation_root_xyzw':[0,0,0,1]}}},
        'partners':[{'category':'wine_bottle','model':'f','scale':[.9]*3,
                     'female':[{'child_link':'male_link','parent_link':'female_link','loc':[0,0,1]}]}]}
    path=tmp_path/'native_seats.json';path.write_text(json.dumps({'schema':'isg-attach-seats/1','pairs':[rec]}))
    job={'category':'cork','model':'m','scale':[.9]*3,'attach_seat_proof':{
        'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'partner':{'category':'wine_bottle','model':'f','scale':[.9]*3}}}
    return job,path


def test_exact_native_seat_proof_uses_measured_mating_axis(tmp_path):
    job,_=fixture(tmp_path);rec=CA.attach_seat_record(job)
    assert rec['male']['scale']==[.9]*3
    assert CA.seat_mating_dirs(rec)[('corkM','0')]['locs'][0][0]==[0,0,1]


@pytest.mark.parametrize('mutation',['male_scale','partner_scale','hash','failed_measurement'])
def test_explicit_missing_or_changed_measurement_never_falls_back(tmp_path,mutation):
    job,path=fixture(tmp_path)
    if mutation=='male_scale':job['scale']=[1]*3
    elif mutation=='partner_scale':job['attach_seat_proof']['partner']['scale']=[1]*3
    elif mutation=='hash':path.write_text('{}')
    else:
        data=json.loads(path.read_text());data['pairs'][0]['partners'][0]['error']='no matching female link'
        path.write_text(json.dumps(data));job['attach_seat_proof']['sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError,match='explicit attachment seat proof'):CA.attach_seat_record(job)
