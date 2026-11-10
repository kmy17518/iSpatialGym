"""Exercise actual run_instances queue transitions while retaining every core screen."""
import json
import os
from types import SimpleNamespace

import pytest

from ispatialgym.sampling import campaign_goal_queue as queue
from ispatialgym.sampling.v2 import core
from scripts.campaign_20261110.tests import test_goal_queue as fixtures


@pytest.mark.parametrize("family", ["held", "pair"])
def test_two_groups_share_one_sampler_and_do_not_consume_attempts_when_full(tmp_path, monkeypatch, family):
    from scripts.campaign_20261110.tests.test_pair_goal_batches import pair_fixture
    manifest, _, _ = (fixtures.fixture if family == "held" else pair_fixture)(tmp_path, monkeypatch)
    monkeypatch.setattr(core, "fast_on", lambda *args: False)
    monkeypatch.setattr(core, "publish_instance", lambda path: True)
    calls = []
    sampler = SimpleNamespace(ctx=SimpleNamespace(base_seed=9), gp=SimpleNamespace(variant="held"), objects={},
        world=SimpleNamespace(reset_scene_objects=lambda *args,**kwargs: 0),
        kit=SimpleNamespace(stats_snapshot=lambda:{},stats_delta=lambda a,b:{}),
        ikc=SimpleNamespace(solutions={}), counters={}, template_paths=("template",), template_meta={}, instance_seeds=[])
    for name in ("attempt_begin","attempt_end","park_robot","check_early_repeat","apply_draft","settle_and_validate",
                 "check_stances_with_objects","check_goal_feasible","after_goal_check","verify_carry_legs","program_check",
                 "check_witness_reach","check_instance_filter","check_goal_occlusion","constraints_write_back",
                 "write_template_meta","_cap_after_attempt","log_attempt_stages"):
        setattr(sampler,name,lambda *args,_name=name,**kwargs:calls.append(_name))
    sampler.attempt_cap=lambda:(60.,"fixture")
    sampler.campaign_group_remaining_s=lambda:60.
    sampler.instance_id=lambda index:f"fresh_{index}"
    goals=[]
    def sample(rng):
        plan=json.load(open(os.environ[queue.PLAN_ENV]))
        goals.append((plan["group_id"],plan["goal_pose"]))
        return SimpleNamespace(summary={})
    def write(draft,index,seed):
        sampler._campaign_group_written=getattr(sampler,"_campaign_group_written",0)+1
        return str(tmp_path/f"fresh_{index}-tro_state.json")
    sampler.sample,sampler.write_instance=sample,write
    paths=core.SamplerV2.run_instances(sampler,n=9,max_attempts=1,indices=range(9))
    assert len(paths)==6
    assert [group for group,_ in goals] == ["g0"]*3+["g1"]*3
    assert goals[0][1] != goals[3][1]
    assert calls.count("program_check")==calls.count("check_goal_feasible")==calls.count("check_goal_occlusion")==6
    assert sampler.template_meta["n_written"]==6
    closed=[json.loads(line) for line in manifest.with_suffix(".events.jsonl").read_text().splitlines() if json.loads(line)["event"]=="closed"]
    assert [(row["group_id"],row["written"]) for row in closed] == [("g0",3),("g1",3)]
