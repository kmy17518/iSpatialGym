from dataclasses import replace
from types import SimpleNamespace as NS

import pytest
import torch as th

from ispatialgym.sampling import campaign_block_goals as goals
from ispatialgym.sampling.sim_utils import Reject
from ispatialgym.sampling.tasks.base import Draft
from ispatialgym.sampling.v2.tasks.block_design import BlockDesign


def raw_plan():
    return {"schema": goals.SCHEMA, "task": "configuration_matching-block_design", "campaign_id": "new",
            "group_id": "group", "split": "eval_standard", "template_id": "template", "scene": "scene", "room": "room",
            "contract_template_sha256": "a"*64, "target_names": [f"block{i}" for i in range(4)], "board_name": "board",
            "board_pose": {"position": [.6, .1, .8], "orientation": [0, 0, 0, 1]},
            "goal_slots": [{"position": [.58+.03*(i%2), .12-.03*(i//2), .822], "orientation": [0, 0, 0, 1]} for i in range(4)],
            "pattern_id": "rbbr", "pattern": ["red", "blue", "blue", "red"],
            "initial_up_faces": ["+x", "-x", "+y", "-y"], "initial_orientations": [[0, 0, 0, 1]]*4,
            "flip_directions": ["right", None, None, "right"], "created_at": 100, "budget_seconds": 900, "max_members": 3}


def validated(raw=None):
    return goals.validate_block_plan(raw or raw_plan(), template_id="template", scene="scene", rooms=["room"], activated_at=200, now=201)


def test_block_plan_checks_all_identities_poses_and_time():
    plan = validated()
    assert plan.deadline_at == 1100
    assert plan.summary()["requested_goal"]["board"]["position"] == [.6, .1, .8]
    for key, value, message in (("target_names", ["same"]*4, "distinct"), ("max_members", 4, "group size"),
                               ("initial_up_faces", ["+x"]*4, "distinct"), ("initial_orientations", [[0, 0, 0, 2]]*4, "normalized")):
        with pytest.raises(goals.GoalPlanError, match=message):
            validated({**raw_plan(), key: value})
    with pytest.raises(goals.GoalGroupTimeout):
        goals.validate_block_plan(raw_plan(), template_id="template", scene="scene", rooms=["room"], activated_at=200, now=1100)


def test_exchangeable_block_mapping_keeps_exact_slot_goals_after_settling():
    s = object.__new__(BlockDesign)
    p = s._campaign_block_active = validated()
    s.campaign_group_remaining_s = lambda: 100
    names = list(p.target_names)
    s.names = lambda: names
    s._group_name = lambda names: "blocks"
    d = Draft()
    slot_indices = [2, 0, 3, 1]
    d.groups["blocks"] = {"target_slots": {f"slot_{i}": dict(raw_plan()["goal_slots"][i]) for i in slot_indices}}
    d.object_poses["board"] = s.campaign_tensor_pose(p.board)
    d.summary["campaign_goal_group"] = {**p.summary(), "actual_flips": 2}
    for name, i in zip(names, slot_indices):
        d.goal_poses[name] = s.campaign_tensor_pose(p.slots[i])
    s.objects = {name: NS(get_position_orientation=lambda pose=pose: (pose[0]+th.tensor([.001, 0, 0]), pose[1]))
                 for name, pose in {**d.goal_poses, "board": d.object_poses["board"]}.items()}
    s.campaign_verify_goal(d, settled=True)
    for name, i in zip(names, slot_indices):
        assert d.goal_poses[name][0].tolist() == pytest.approx(p.slots[i][0])
        assert d.groups["blocks"]["target_slots"][f"slot_{i}"]["position"] == list(p.slots[i][0])
    d.goal_poses[names[0]] = (th.tensor([9., 0, 0]), th.tensor([0., 0, 0, 1]))
    with pytest.raises(Reject, match="slot 2 differs"):
        s.campaign_verify_goal(d)


def test_block_train_difficulty_is_soft_while_eval_requires_flips():
    s = object.__new__(BlockDesign)
    p = s._campaign_block_active = validated()
    s.campaign_group_remaining_s = lambda: 100
    s.names = lambda: list(p.target_names)
    s._group_name = lambda names: "blocks"
    d = Draft()
    d.groups["blocks"] = {"target_slots": {f"slot_{i}": dict(raw_plan()["goal_slots"][i]) for i in range(4)}}
    d.goal_poses = {name: s.campaign_tensor_pose(p.slots[i]) for i, name in enumerate(p.target_names)}
    d.object_poses["board"] = s.campaign_tensor_pose(p.board)
    d.summary["campaign_goal_group"] = {"actual_flips": 0}
    with pytest.raises(Reject, match="one to four"):
        s.campaign_verify_goal(d)
    s._campaign_block_active = replace(p, split="train")
    s.campaign_verify_goal(d)


@pytest.mark.parametrize('up_faces,expected', [
    (['+z','-x','+y','+x'], {1,2,3}),
    (['-x','-y','+y','+x'], {0,1,2,3}),
])
def test_fixed_profile_screens_all_required_flips_before_selecting_cells(up_faces, expected):
    s=object.__new__(BlockDesign)
    s._campaign_block_active=NS(initial_up_faces=up_faces,pattern=['red','blue','blue','red'])
    s.faces={'+z':'red','-z':'blue','+x':'white','-x':'yellow','+y':'blue|yellow','-y':'red|white'}
    s.gen=lambda key,default=None: default
    s.g=lambda key: {'rbbr':['red','blue','blue','red']} if key=='patterns' else None
    s._has_board=lambda: True
    cells=[th.tensor([.3*i,0.]) for i in range(4)]
    stance=NS(world_to_base=lambda cell: cell)
    goals_=['+z','-z','-z','+z']
    assert s._n_flips()==len(expected)
    assert s._flips_expected()
    s._flip_cell_ok=lambda cell,*args: float(cell[0])<.8
    assert s._flip_plan(cells,lambda c:float(c[0]),goals_,stance,stance,'right',True) is None
    s._flip_cell_ok=lambda *args: True
    assert s._flip_plan(cells,lambda c:float(c[0]),goals_,stance,stance,'right',True)==expected
    # A nearby block on the approach side still rejects the whole candidate.
    blocked=list(cells);blocked[3]=th.tensor([.01,.1])
    assert s._flip_plan(blocked,lambda c:float(c[0]),goals_,stance,stance,'right',True) is None
    # Ordinary sampling keeps its original repeated-color assignment.
    s._campaign_block_active=None
    assert s._n_flips()==2
    assert s._flip_plan(cells,lambda c:float(c[0]),goals_,stance,stance,'right',True)=={0,1}


def test_board_precheck_uses_the_actual_fixed_pose_and_keeps_geometry_gates():
    s=object.__new__(BlockDesign)
    s._campaign_block_active=NS(board=([.8,.1,.7],[0.,0.,0.,1.]))
    s._table_axes=lambda _: (th.tensor([1.,0.]),th.tensor([0.,1.]),0.)
    s._board_geometry=lambda: {'plate_m':.1}
    s.edge=.025;s.gen=lambda key,default=None: default
    s.names=lambda: ['a','b','c','d'];s._n_flips=lambda: 0;s._flips_expected=lambda: False
    s._flip_directions=lambda: ['right'];s.camera_pose_at=lambda _: None;s.top=NS(z=.7)
    s._board_plan=lambda *args,**kwargs: [(2.,2.,False)]
    cells=[th.tensor([.1*i,0.]) for i in range(4)]
    stance=NS(extras={'focus_info':{'cells':cells}},world_to_base=lambda xy:xy,
              base_xy=th.zeros(2),forward=th.tensor([1.,0.]),left=th.tensor([0.,1.]))
    visited=[]
    def on_top(_m,d,l,*args):
        visited.append((d,l));return d<1
    s._board_on_top=on_top
    s._board_precheck(stance,stance,[],count=False)
    assert len(visited)==1
    assert visited[0]==pytest.approx((.8,.1))
    # A valid alternative position cannot rescue an invalid fixed goal.
    s._campaign_block_active=NS(board=([2.,2.,.7],[0.,0.,0.,1.]))
    s._board_plan=lambda *args,**kwargs: [(.8,.1,False)]
    with pytest.raises(Reject,match='fixed group board pose'):
        s._board_precheck(stance,stance,[],count=False)
    s._campaign_block_active=NS(board=([.8,.1,.7],[0.,0.,1.,0.]))
    with pytest.raises(Reject,match='campaign_board_direction'):
        s._board_precheck(stance,stance,[],count=False)


def test_board_feasibility_cache_and_pattern_are_scoped_to_fixed_goal():
    s=object.__new__(BlockDesign)
    stance=NS(base_pose2d=(0.,0.,0.),translate=.5,tilt=-.45)
    s.manip_candidates=[stance];s.color_to_face=lambda: {'red':'+z','blue':'-z'}
    s.g=lambda _: {'legacy':['blue']}
    seen=[];s._board_precheck=lambda _i,_c,faces,**kwargs: seen.append(faces)
    s._campaign_block_active=NS(sha256='first',pattern=['red'])
    assert s._board_feasible(0,stance);assert s._board_feasible(0,stance)
    assert seen==[['+z']]
    s._campaign_block_active=NS(sha256='second',pattern=['blue','red'])
    assert s._board_feasible(0,stance);assert seen[-1]==['-z','+z']
    s._campaign_block_active=None
    assert s._board_feasible(0,stance);assert seen[-1]==['-z']
    assert len(seen)==3


@pytest.mark.parametrize('count,up', [(3,['+z','-x','+y','+x']), (4,['-x','-y','+y','+x'])])
def test_native_anchor_profile_preserves_flip_geometry_without_freezing_board(count,up):
    s=object.__new__(BlockDesign)
    proposal={'pattern_id':'rbbr','initial_up_faces':up,'required_flips':count}
    s.gen=lambda key,default=None: proposal if key=='campaign_start_profile' else default
    s.g=lambda _: {'rbbr':['red','blue','blue','red']}
    s.faces={'+z':'red','-z':'blue','+x':'white','-x':'yellow','+y':'blue|yellow','-y':'red|white'}
    s._has_board=lambda: True
    assert s._n_flips()==count and s._flips_expected()
    assert s.pattern(None)==('rbbr',['red','blue','blue','red'])
    assert getattr(s,'_campaign_block_active',None) is None
    cells=[th.tensor([.3*i,0.]) for i in range(4)]
    stance=NS(world_to_base=lambda c:c)
    s._flip_cell_ok=lambda c,*args: float(c[0])<.8
    assert s._flip_plan(cells,lambda c:float(c[0]),['+z','-z','-z','+z'],stance,stance,'right',True) is None
    s._flip_cell_ok=lambda *args: True
    assert len(s._flip_plan(cells,lambda c:float(c[0]),['+z','-z','-z','+z'],stance,stance,'right',True))==count
    proposal['required_flips']=count-1
    with pytest.raises(Reject,match='flip count'):
        s._n_flips()
    proposal.update(required_flips=count,initial_up_faces=['+x']*4)
    with pytest.raises(Reject,match='distinct native faces'):
        s.pattern(None)
    proposal['initial_up_faces']=['-z','-x','+y','+x']
    with pytest.raises(Reject,match='opposite-face'):
        s._n_flips()
