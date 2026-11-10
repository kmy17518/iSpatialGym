"""``oracle.sleep_assist``: the rigid sleep threshold and the sleep assist both drivers run (run_isg: demos, run_isg_vec: evaluation),
their flags and defaults, and the executor's wiring (after every step, trimmed on a rollback, ``result.sleep_assist``)."""

from __future__ import annotations

import argparse
import sys
import types

import pytest
import torch as th

from ispatialgym.oracle import sleep_assist as SA
from ispatialgym.oracle.executor import ExecutorConfig
from ispatialgym.oracle.programs import expand
from ispatialgym.tests.test_v2_oracle import _place_goal, iors_world, make_executor, success_eval

ASSIST_ENV = ("ISG_SLEEP_ASSIST", "ISG_SLEEP_ASSIST_MODE", "ISG_SLEEP_ASSIST_WAKE_M", "ISG_SLEEP_ASSIST_MAX_KG", SA.RESTART_ON_WAKE_ENV)
PHYSICS_FLAGS = ("rigid_sleep_threshold", "rigid_solver_iterations", "sleep_assist", "sleep_assist_mode", "sleep_assist_max_mass",
                 "sleep_assist_wake_radius", "lean_step")


class Touching:
    """Stands in for ``omnigibson.object_states.Touching``: ``get_value(other)`` from the world's touching pairs."""

    def __init__(self, obj, world):
        self.obj, self.world = obj, world

    def get_value(self, other):
        return frozenset((self.obj.name, other.name)) in self.world.touching


class World:
    def __init__(self):
        self.touching = set()

    def touch(self, a, b, on=True):
        (self.touching.add if on else self.touching.discard)(frozenset((a.name, b.name)))


class Body:
    def __init__(self, world, name, pos, mass=0.2, half=0.03, n_dof=0):
        self.name, self.mass, self.n_dof, self.category = name, mass, n_dof, "apple"
        self.pos, self.quat, self.half = th.tensor(pos, dtype=th.float32), th.tensor([0.0, 0.0, 0.0, 1.0]), half
        self.asleep, self.slept_calls, self.woken_calls = False, 0, 0
        self.states = {Touching: Touching(self, world)}
        self.joints = {}

    def get_position_orientation(self):
        return self.pos.clone(), self.quat.clone()

    @property
    def aabb(self):
        return self.pos - self.half, self.pos + self.half

    @property
    def aabb_center(self):
        return self.pos.clone()

    def is_asleep(self):
        return self.asleep

    def sleep(self):
        self.asleep, self.slept_calls = True, self.slept_calls + 1

    def wake(self):
        self.asleep, self.woken_calls = False, self.woken_calls + 1


class Robot:
    name, category, arm_names = "robot_r1", "agent", ["left", "right"]

    def __init__(self):
        self.eef = {"left": th.tensor([5.0, 0.0, 1.0]), "right": th.tensor([5.0, 1.0, 1.0])}

    def get_eef_position(self, arm):
        return self.eef[arm].clone()


def clean_env(monkeypatch, **values):
    """The ``ISG_SLEEP_ASSIST*`` variables unset (or ``values``) for the test and restored after it, also when the code under test
    writes them (``export_env`` sets ``os.environ`` itself)."""
    for k in ASSIST_ENV:
        monkeypatch.setenv(k, "x")  # recorded, so the teardown restores the value from before the test
        monkeypatch.delenv(k)
    for k, v in values.items():
        monkeypatch.setenv(k, v)


@pytest.fixture
def world(monkeypatch):
    clean_env(monkeypatch)
    monkeypatch.setitem(sys.modules, "omnigibson.object_states", types.SimpleNamespace(Touching=Touching))
    return World()


def run(assist, steps, start=1, each=None):
    for s in range(start, start + steps):
        if each is not None:
            each(s)
        assist.step(s)
    return start + steps


def test_the_flags_and_defaults_of_both_drivers():
    p = argparse.ArgumentParser()
    SA.add_arguments(p)
    a = p.parse_args([])
    assert (a.rigid_sleep_threshold, a.rigid_solver_iterations, a.sleep_assist, a.sleep_assist_mode, a.sleep_assist_max_mass,
            a.sleep_assist_wake_radius) == (0.005, None, "30,1,3", "sleep", 5.0, 0.05)
    assert SA.rigid_overrides(a) == {"rigid_sleep_threshold": 0.005}
    assert SA.rigid_overrides(p.parse_args(["--rigid-sleep-threshold", "0"])) == {}
    assert SA.rigid_overrides(p.parse_args(["--rigid-solver-iterations", "64+8"]))["rigid_solver_position_iterations"] == 64


def test_run_isg_and_run_isg_vec_take_the_same_physics_flags():
    """Demos (run_isg, both entry points) and evaluations (run_isg_vec) run the same physics: the same options, defaults and choices."""
    from ispatialgym import run_isg as PUBLIC
    from ispatialgym.oracle import run_isg as RI
    from ispatialgym.oracle import run_isg_vec as RV

    def flags(p):
        return {a.dest: (a.default, tuple(a.option_strings), a.choices, type(a)) for a in p._actions if a.dest in PHYSICS_FLAGS}

    vec = flags(RV.build_parser())
    assert sorted(vec) == sorted(PHYSICS_FLAGS)
    assert flags(RI.build_parser()) == vec
    assert flags(PUBLIC.build_parser()) == vec


def test_export_env_and_the_config_it_reads(world, monkeypatch):
    p = argparse.ArgumentParser()
    SA.add_arguments(p)
    SA.export_env(p.parse_args(["--sleep-assist", "20,2,4", "--sleep-assist-mode", "clamp", "--sleep-assist-max-mass", "3"]))
    assert SA.sleep_assist_config() == (20, 0.002, 0.004)
    assert (SA.sleep_assist_mode(), SA.sleep_assist_max_mass(), SA.sleep_assist_wake_radius()) == ("clamp", 3.0, 0.05)
    monkeypatch.setenv("ISG_SLEEP_ASSIST", "off")
    assert SA.sleep_assist_config() is None


def test_the_overrides_reach_free_rigid_objects_only(world):
    free, door = Body(world, "apple", [0, 0, 1]), Body(world, "door", [1, 0, 1], n_dof=1)
    fixed, robot = Body(world, "table", [0, 0, 0.5]), Robot()
    fixed.fixed_base = True
    for o in (free, door, fixed, robot):
        o.sleep_threshold = 5e-5
    rep = SA.apply_to_objects([free, door, fixed, robot], {"rigid_sleep_threshold": 0.005})
    assert rep == {"set": ["apple"], "skipped": ["door", "table", "robot_r1"], "threshold": 0.005}
    assert (free.sleep_threshold, door.sleep_threshold, fixed.sleep_threshold, robot.sleep_threshold) == (0.005, 5e-5, 5e-5, 5e-5)
    assert SA.apply_to_objects([free], {}) is None
    scene = types.SimpleNamespace(idx=0, objects=[free, door, fixed])
    assert SA.apply_to_scenes([scene], {"rigid_sleep_threshold": 0.01})["rigid_sleep_threshold"] == {"threshold": 0.01, "objects": 1, "scene0_names": ["apple"]}


def test_an_object_at_rest_sleeps_after_the_window_and_not_while_the_robot_touches_it(world):
    robot, apple = Robot(), Body(world, "apple", [0, 0, 1])
    sa = SA.SleepAssist(robot, {"apple.n.01_1": apple}, "s0", config=(5, 0.001, 0.003))
    world.touch(apple, robot)
    n = run(sa, 10)
    assert not apple.asleep  # every contact with the robot restarts the window
    world.touch(apple, robot, on=False)
    run(sa, 4, start=n)
    assert not apple.asleep
    sa.step(n + 4)
    assert apple.asleep and sa.sleeps == {"apple.n.01_1": [n + 4]}
    rep = sa.report()
    assert (rep["mode"], rep["window_steps"], rep["sleeps"], rep["free_objects"]) == ("sleep", 5, {"apple.n.01_1": [n + 4]}, ["apple.n.01_1"])


def test_a_moving_object_heavy_furniture_and_a_support_under_an_awake_body_stay_awake(world):
    robot = Robot()
    table = Body(world, "table", [0, 0, 0.5], mass=40.0)
    plate, apple = Body(world, "plate", [0, 0, 0.8]), Body(world, "apple", [0, 0, 0.85])
    world.touch(plate, apple)
    sa = SA.SleepAssist(robot, {"table": table, "plate": plate, "apple": apple}, "s0", config=(3, 0.001, 0.003))

    def roll(s):
        apple.pos = apple.pos + th.tensor([0.002, 0.0, 0.0])  # the apple rolls 2 mm per step on the plate

    n = run(sa, 6, each=roll)
    assert not (table.asleep or plate.asleep or apple.asleep)  # table: > 5 kg; plate: the awake apple rests on it
    run(sa, 2, start=n)
    assert apple.asleep and not plate.asleep  # the top of the stack sleeps first ...
    sa.step(n + 2)
    assert plate.asleep and not table.asleep  # ... the support follows


def test_a_moving_gripper_wakes_what_the_assist_slept_a_parked_one_does_not(world):
    robot, apple = Robot(), Body(world, "apple", [0, 0, 1])
    sa = SA.SleepAssist(robot, {"apple": apple}, "s0", config=(3, 0.001, 0.003))
    n = run(sa, 3)
    assert apple.asleep
    robot.eef["right"] = th.tensor([0.0, 0.06, 1.0])  # 3 cm from the box: the jump is a motion, so this step wakes it
    sa.step(n)
    assert not apple.asleep and sa.wakes == {"apple": [n]}
    n = run(sa, 4, start=n + 1)  # the gripper parks 3 cm away: no motion, the apple sleeps again
    assert apple.asleep and apple.slept_calls == 2


def test_a_rollback_drops_the_attempts_events_and_keeps_an_earlier_sleep_the_assists(world):
    robot, apple = Robot(), Body(world, "apple", [0, 0, 1])
    sa = SA.SleepAssist(robot, {"apple": apple}, "s0", config=(3, 0.001, 0.003))
    run(sa, 3)
    assert sa.sleeps == {"apple": [3]}
    sa.trim(2)  # the checkpoint was taken before the sleep: it leaves the record
    assert sa.sleeps == {} and sa.slept == {} and sa.hist == {} and sa.report()["rollbacks"] == [2]
    apple.asleep = False
    run(sa, 3, start=3)
    assert sa.sleeps == {"apple": [5]}
    sa.trim(7)  # slept before the checkpoint: still the assist's to wake when a gripper comes
    assert sa.slept == {"apple": True} and sa.hist == {}
    sa.step(8)
    robot.eef["right"] = th.tensor([0.0, 0.05, 1.0])
    sa.step(9)
    assert not apple.asleep and sa.wakes == {"apple": [9]}


def sleepable(*objs):
    """The fake world's objects with PhysX's sleep flag (``is_asleep`` / ``sleep``; the fake physics ignores it)."""
    for o in objs:
        o.asleep = False
        o.is_asleep = lambda o=o: o.asleep
        o.sleep = lambda o=o: setattr(o, "asleep", True)


def test_the_executor_runs_the_assist_per_step_and_reports_it(tmp_path, monkeypatch):
    clean_env(monkeypatch, ISG_SLEEP_ASSIST="30,1,3")
    sc, env = iors_world()
    env.evaluate = success_eval("apple.n.01_1", _place_goal(sc), tol=0.08)
    plate = env.objects["plate.n.04_1"]
    sleepable(plate, env.objects["apple.n.01_1"])
    ex, motions, backend, rec = make_executor(env, sc, tmp_path, exec_cfg=ExecutorConfig(settle_steps=50))
    result = ex.run_program(expand(sc.variant, sc))
    assert result["success"], result["verdict"]
    rep = result["sleep_assist"]
    assert rep["window_steps"] == 30 and rep["sleeps"]["plate.n.04_1"] == [30]  # the plate never moves: asleep after the first window
    assert plate.asleep and ex.step_costs["sleep_assist_s"] > 0


def test_a_rollback_trims_the_executors_assist(tmp_path, monkeypatch):
    clean_env(monkeypatch, ISG_SLEEP_ASSIST="5,1,3")
    sc, env = iors_world()
    sleepable(*env.objects.values())
    ex, motions, backend, rec = make_executor(env, sc, tmp_path)
    ex._init_sleep_assist()  # run_program builds it; the actions run one by one here
    actions = expand(sc.variant, sc)
    cp = ex.checkpoint(0)
    assert ex.run_action(0, actions[0]) is None and ex.sleep_assist.sleeps  # the plate and the apple rest through the transit
    ex.restore(cp)
    assert ex.sleep_assist.rollbacks == [cp.step] and ex.sleep_assist.sleeps == {} and ex.sleep_assist.slept == {}


def test_the_executor_has_no_assist_without_the_flag(tmp_path, monkeypatch):
    clean_env(monkeypatch, ISG_SLEEP_ASSIST="off")
    sc, env = iors_world()
    env.evaluate = success_eval("apple.n.01_1", _place_goal(sc), tol=0.08)
    ex, motions, backend, rec = make_executor(env, sc, tmp_path, exec_cfg=ExecutorConfig(settle_steps=50))
    result = ex.run_program(expand(sc.variant, sc))
    assert result["sleep_assist"] is None and ex.sleep_assist is None and "sleep_assist_s" not in ex.step_costs


@pytest.mark.parametrize("entry", ["public", "oracle"])
def test_a_config_turns_the_physics_flags_off(tmp_path, entry):
    """YAML reads a bare ``off`` and ``false`` as booleans: the config still turns the sleep assist and the lean step off (config.yaml
    round-trips them)."""
    import yaml

    if entry == "public":
        from ispatialgym import run_isg as RUN
    else:
        from ispatialgym.oracle import run_isg as RUN
    config = tmp_path / "c.yaml"
    config.write_text("sleep_assist: off\nlean_step: false\nrigid_sleep_threshold: 0\n")
    argv = ["--instance", "x-tro_state.json", "--mode", "oracle"]
    args = RUN.parse_args(argv + ["--config", str(config)])
    assert (args.sleep_assist, args.lean_step, args.rigid_sleep_threshold) == ("off", False, 0.0)
    assert SA.rigid_overrides(args) == {}
    (tmp_path / "run.yaml").write_text(yaml.safe_dump(RUN.run_options(args)))
    child = RUN.parse_args(["--config", str(tmp_path / "run.yaml"), "--instance", "x-tro_state.json"])
    assert RUN.run_options(child) == RUN.run_options(args)
    (tmp_path / "on.yaml").write_text("sleep_assist: on\n")
    assert RUN.parse_args(argv + ["--config", str(tmp_path / "on.yaml")]).sleep_assist == SA.SLEEP_ASSIST


class HullLink:
    """A link with OmniGibson's two calls the box needs; counts how often the local hull is computed."""

    def __init__(self, points, pose):
        self.points, self.pose, self.hull_calls = th.tensor(points, dtype=th.float32), pose, 0

    @property
    def collision_boundary_points_local(self):
        self.hull_calls += 1
        return self.points

    def transform_local_points_to_world(self, points):
        return points + self.pose()


def test_the_box_is_omnigibsons_with_each_hull_computed_once(world):
    robot, apple = Robot(), Body(world, "apple", [0, 0, 1])
    links = {"base_link": HullLink([[-0.03, -0.02, -0.01], [0.03, 0.02, 0.01]], lambda: apple.pos),
             "meta": HullLink([[0.0, 0.0, 0.05]], lambda: apple.pos)}
    apple.links = links
    sa = SA.SleepAssist(robot, {"apple": apple}, "s0", config=(3, 0.001, 0.003))
    for _ in range(3):
        apple.pos = apple.pos + th.tensor([0.1, 0.0, 0.0])
        pts = th.cat([l.collision_boundary_points_local + apple.pos for l in links.values()])
        want = (th.min(pts, dim=0).values, th.max(pts, dim=0).values)  # EntityPrim.aabb over collision_boundary_points_world
        got = sa._aabb("apple", apple)
        assert th.equal(got[0], want[0]) and th.equal(got[1], want[1])
    assert [l.hull_calls for l in links.values()] == [4, 4]  # 3 for the reference above, 1 for the assist
    assert th.equal(sa._aabb_center("apple", apple), (want[1] + want[0]) / 2.0)


def test_no_box_is_computed_while_every_gripper_is_parked(world):
    robot, apple = Robot(), Body(world, "apple", [0, 0, 1])
    calls = []
    type(apple).aabb = property(lambda self: calls.append(1) or (self.pos - self.half, self.pos + self.half))
    sa = SA.SleepAssist(robot, {"apple": apple}, "s0", config=(3, 0.001, 0.003))
    run(sa, 3)
    assert apple.asleep and calls == []  # slept with the grippers parked: no box needed
    robot.eef["right"] = th.tensor([0.0, 0.06, 1.0])
    sa.step(4)
    assert not apple.asleep and calls == [1]


def test_restart_on_wake_is_off_by_default_a_physx_wake_is_re_slept_at_once(world):
    """Today's rule (ISPATIALGYM_SLEEP_ASSIST_RESTART_ON_WAKE unset): the window keeps the asleep span's still poses, so a body PhysX woke
    (a slow hand's contact) is re-slept on the next control step although it now moves 0.1 mm per step (the 5w.3(b) pressed sleeper)."""
    assert SA.RESTART_ON_WAKE_DEFAULT == "0" and not SA.restart_on_wake()
    robot, apple = Robot(), Body(world, "apple", [0, 0, 1])
    sa = SA.SleepAssist(robot, {"apple": apple}, "s0", config=(5, 0.001, 0.003))
    n = run(sa, 5)
    assert apple.asleep and sa.sleeps == {"apple": [5]}
    n = run(sa, 10, start=n)  # asleep: the window fills with still poses
    apple.asleep = False  # PhysX wakes it (contact); it creeps 0.1 mm per step

    def creep(s):
        apple.pos = apple.pos + th.tensor([0.0001, 0.0, 0.0])

    run(sa, 1, start=n, each=creep)
    assert apple.asleep and sa.sleeps == {"apple": [5, n]}
    assert "restart_on_wake" not in sa.report() and "wake_restarts" not in sa.report()


def test_restart_on_wake_a_physx_wake_restarts_the_window(world, monkeypatch):
    monkeypatch.setenv(SA.RESTART_ON_WAKE_ENV, "1")
    robot, apple = Robot(), Body(world, "apple", [0, 0, 1])
    sa = SA.SleepAssist(robot, {"apple": apple}, "s0", config=(5, 0.001, 0.003))
    n = run(sa, 15)
    assert apple.asleep and sa.sleeps == {"apple": [5]}
    apple.asleep = False  # PhysX's wake
    w = n

    def creep(s):
        apple.pos = apple.pos + th.tensor([0.0001, 0.0, 0.0])

    n = run(sa, 4, start=n, each=creep)
    assert not apple.asleep  # 4 poses since the wake: the window (5) is not full
    sa.step(n)  # 5 poses spanning 0.4 mm < 1 mm: a real rest of the awake body -> asleep
    assert apple.asleep and sa.sleeps == {"apple": [5, n]}
    rep = sa.report()
    assert rep["restart_on_wake"] is True and rep["wake_restarts"] == {"physx": {"apple": [w]}, "assist": {}}
    assert rep["wake_restarts_n"] == {"physx": 1, "assist": 0}
    apple.asleep = False  # woken again, now pushed 0.3 mm per step: the window never sees a rest
    for s in range(n + 1, n + 20):
        apple.pos = apple.pos + th.tensor([0.0003, 0.0, 0.0])
        sa.step(s)
    assert not apple.asleep and sa.sleeps == {"apple": [5, n]}


def test_restart_on_wake_the_assists_own_wake_restarts_the_window_and_a_rollback_trims_the_record(world, monkeypatch):
    monkeypatch.setenv(SA.RESTART_ON_WAKE_ENV, "1")
    robot, apple = Robot(), Body(world, "apple", [0, 0, 1])
    sa = SA.SleepAssist(robot, {"apple": apple}, "s0", config=(3, 0.001, 0.003))
    n = run(sa, 6)
    assert apple.asleep
    robot.eef["right"] = th.tensor([0.0, 0.06, 1.0])  # a moving gripper 3 cm away: the assist wakes it
    sa.step(n)
    assert not apple.asleep and sa.restarts == {"physx": {}, "assist": {"apple": [n]}}
    sa.step(n + 1)  # the gripper parks: today the full window re-sleeps it here; restarted, 2 poses < 3
    assert not apple.asleep
    sa.step(n + 2)
    assert apple.asleep and apple.slept_calls == 2
    sa.trim(n - 1)
    assert sa.restarts == {"physx": {}, "assist": {}} and sa.was_asleep == {}
