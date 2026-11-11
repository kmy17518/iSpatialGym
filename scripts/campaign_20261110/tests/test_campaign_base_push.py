import math
import pytest
from ispatialgym.sampling.campaign_base_push import settled_endpoint


def test_measured_nudge_extends_original_line_without_slack():
    angle=math.radians(255);d=[math.cos(angle),math.sin(angle),0.]
    start=([15.677,1.674,.193],[0,0,0,1]);goal=([start[0][i]+1.658*d[i] for i in range(3)],start[1])
    result=settled_endpoint(start,goal,d,1.652,[15.445,.811,angle])
    assert result['advance_m']==pytest.approx(.006)
    assert result['distance_m']==pytest.approx(1.658)
    assert result['base_end']==pytest.approx([15.445+.006*d[0],.811+.006*d[1],angle])
    assert result['delta_xyz']==pytest.approx([.006*d[0],.006*d[1],0])


def test_no_goal_motion_requires_no_change():
    assert settled_endpoint(([0,0,0],[0,0,0,1]),([1,0,0],[0,0,0,1]),[1,0,0],1,[1,0,0]) is None


@pytest.mark.parametrize('goal,fixed',[
    (([1.006,0,0],[0,0,0,1]),True),
    (([1.006,.004,0],[0,0,0,1]),False),
    (([1.04,0,0],[0,0,0,1]),False),
    (([.98,0,0],[0,0,0,1]),False),
    (([1.006,0,0],[0,0,.1,.995]),False),
])
def test_different_group_goal_or_noncollinear_motion_is_rejected(goal,fixed):
    with pytest.raises(ValueError):
        settled_endpoint(([0,0,0],[0,0,0,1]),goal,[1,0,0],1,[1,0,0],fixed_goal=fixed)
