"""BEHAVIOR's replay_obs.py on iSpatialGym demos: LeRobot v3 output of a demo.hdf5 through replay_obs's writer, sensors,
modalities, proprioception keys and playback settings. The playback wrapper is AlignedLeRobotPlaybackWrapper, which pairs
each action with the observation of the state it was taken from as live collection does (the unmodified wrapper puts the
previous state's observation, with its velocities zeroed, next to each action); ``--unmodified`` runs BEHAVIOR's wrapper as is.

    python scripts/isg/replay_obs_isg.py --demo <demo dir>[,<demo dir>...] --root <lerobot root> --repo-id <repo id> --task <task name>
        [--resume]

Only replay_obs's challenge lookups are replaced (the "thin adapter"): the task name is ours instead of the challenge task id
table, the full scene file / room list are None (our demo.hdf5 scene file holds every object and every room of the scene),
and the input path it builds (<data folder>/2026-challenge-rawdata/task-<id>/episode_<demo id>.hdf5) is a symlink to the
demo's demo.hdf5. Several demos go into one dataset with --resume after the first (replay_obs's --resume_lerobot). The
installed lerobot's DatasetInfo rejects the two info.json keys BEHAVIOR's writer adds; ``lerobot_compat.install()`` lets
it carry them.
"""
import argparse
import importlib.util
import os
import sys
import tempfile

import omnigibson as og

from ispatialgym.oracle import lerobot_compat  # noqa: E402
from ispatialgym.tasks import ispatialgym_task  # noqa: F401,E402  (registers ISpatialGymTask, the task of our env configs)

lerobot_compat.install()  # BEHAVIOR's writer adds info keys the installed lerobot's DatasetInfo rejects

B1K_LEARNING = os.path.join(os.path.dirname(os.path.dirname(og.__file__)), "scripts", "learning")
_spec = importlib.util.spec_from_file_location("replay_obs", os.path.join(B1K_LEARNING, "replay_obs.py"))
replay_obs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(replay_obs)

TASK_ID = 1  # any id: replay_obs derives the task id from the demo id (demo_id // 10000) and asks the (replaced) lookups
UnmodifiedPlaybackWrapper = replay_obs.LeRobotPlaybackWrapper


class AlignedLeRobotPlaybackWrapper(UnmodifiedPlaybackWrapper):
    """``LeRobotPlaybackWrapper`` whose episodes pair every action with the observation of the state it was taken from, as
    the writer's live collection does (the frame of action i: the observation of state[i], its velocities included).

    The writer pairs each history entry's observation with the next entry's action; live collection stores the observation
    the action led to, i.e. that of state[i + 1]. The unmodified playback stores the observation after a kinematic step from
    state[i] (nothing moves, ``keep_still`` zeroes the velocities), which puts the observation of state[i - 1] with action i
    and near-zero velocities. Here each entry's observation is replaced by the one of the next recorded state, rendered right
    after that state is loaded (before ``keep_still`` and the step); sensors, modalities and settings are replay_obs's."""

    def playback_episode(self, episode_id, record_data=True, video_keys=None, post_state_update_callback=None):
        sim = og.sim
        loaded = {"flag": False}

        def load_state(state, serialized=False):
            type(sim).load_state(sim, state, serialized=serialized)
            loaded["flag"] = True

        def callback():
            if post_state_update_callback is not None:
                post_state_update_callback()
            if loaded["flag"]:
                loaded["flag"] = False
                # loading state[i], i >= 1: the history ends with the entry of action i - 1
                if record_data and len(self.current_traj_history) > 1:
                    for _ in range(self.n_render_iterations):
                        og.sim.render()
                    obs_list, info = self.env.get_obs()
                    self.current_traj_history[-1]["obs"] = self._process_obs(obs_list[0], info)

        sim.load_state = load_state
        try:
            return super().playback_episode(episode_id, record_data=record_data, video_keys=video_keys, post_state_update_callback=callback)
        finally:
            del sim.load_state


def adapt(task_name: str) -> None:
    replay_obs._get_task_name_from_task_id = lambda task_id: task_name
    replay_obs._load_challenge_available_tasks = lambda: {task_name: [{"scene_model": None}]}
    replay_obs._find_full_scene_file = lambda task_name, scene_model: None
    replay_obs._load_room_instances = lambda task_name: None


def replay(demo_dirs, root: str, repo_id: str, task_name: str, resume: bool = False, flush_every_n_steps: int = 1000,
           aligned: bool = True) -> None:
    adapt(task_name)
    replay_obs.LeRobotPlaybackWrapper = AlignedLeRobotPlaybackWrapper if aligned else UnmodifiedPlaybackWrapper
    data_folder = tempfile.mkdtemp(prefix="replay_obs_isg_")
    raw = os.path.join(data_folder, "2026-challenge-rawdata", f"task-{TASK_ID:04d}")
    os.makedirs(raw)
    for k, demo_dir in enumerate(demo_dirs):
        demo_id = TASK_ID * 10000 + k
        os.symlink(os.path.abspath(os.path.join(demo_dir, "demo.hdf5")), os.path.join(raw, f"episode_{demo_id:08d}.hdf5"))
        replay_obs.replay_hdf5_file(data_folder=data_folder, task_id=TASK_ID, demo_id=demo_id, output_format="lerobot",
                                    flush_every_n_steps=flush_every_n_steps, lerobot_repo_id=repo_id, lerobot_root_dir=root,
                                    overwrite_lerobot=not (resume or k > 0))
        print(f"REPLAY_OBS_DONE {demo_dir}", flush=True)
        og.clear()


def replay_group(demo_dirs, root: str, repo_id: str, task_name: str, resume: bool = False, flush_every_n_steps: int = 1000):
    """Demos of one template (the same scene objects) into one dataset through one playback environment: replay_obs's writer
    and settings (replay_hdf5_file's kwargs) with AlignedLeRobotPlaybackWrapper, built from the first demo; before each next
    demo the wrapper's input file and scene file are switched (playback_episode restores the scene from the scene file)."""
    import json

    import h5py
    from omnigibson.envs.data_wrapper import _align_scene_object_states_with_recorded_schema
    from omnigibson.eval.utils.eval_utils import PROPRIOCEPTION_INDICES
    from omnigibson.macros import gm

    from ispatialgym.oracle.behavior_lerobot import ROBOT_OBS_MODALITIES, ROBOT_SENSOR_CONFIG

    gm.ENABLE_TRANSITION_RULES = False
    env = AlignedLeRobotPlaybackWrapper.create_from_hdf5(
        input_path=os.path.join(demo_dirs[0], "demo.hdf5"), full_scene_file=None, load_room_instances=None, robot_sensor_config=ROBOT_SENSOR_CONFIG,
        n_render_iterations=1, flush_every_n_steps=flush_every_n_steps, flush_every_n_traj=1, include_robot_control=False,
        robot_proprio_keys=list(PROPRIOCEPTION_INDICES["R1Pro"].keys()), robot_obs_modalities=list(ROBOT_OBS_MODALITIES), include_contacts=False,
        output_path=repo_id, root_dir=root, overwrite=not resume, robot_type="R1Pro", task_name=task_name, include_task_obs=False)
    env.load_observation_space()
    done = []
    for k, demo_dir in enumerate(demo_dirs):
        if k > 0:
            env.input_hdf5.close()
            env.input_hdf5 = h5py.File(os.path.join(demo_dir, "demo.hdf5"), "r")
            env.recorded_scene_file = json.loads(env.input_hdf5["data"].attrs["scene_file"])
            env.scene_file = env.recorded_scene_file
            _align_scene_object_states_with_recorded_schema(scene=env.scene, recorded_scene_file=env.recorded_scene_file)
        episodes = sorted(int(key.split("_", 1)[1]) for key in env.input_hdf5["data"].keys() if key.startswith("demo_"))
        env.playback_episode(episode_id=episodes[-1], record_data=True)
        done.append(demo_dir)
        print(f"REPLAY_OBS_DONE {demo_dir}", flush=True)
    env.save_data()
    return done


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--demo", required=True, help="demo dir(s) holding demo.hdf5, comma-separated")
    p.add_argument("--root", required=True, help="LeRobot root dir")
    p.add_argument("--repo-id", required=True)
    p.add_argument("--task", required=True, help="task name written into the dataset (the alias)")
    p.add_argument("--resume", action="store_true", help="append to an existing dataset")
    p.add_argument("--flush-every-n-steps", type=int, default=1000)
    p.add_argument("--unmodified", action="store_true",
                   help="BEHAVIOR's LeRobotPlaybackWrapper as is (the observation of state[i - 1] with action i, velocities zeroed by "
                        "keep_still) instead of AlignedLeRobotPlaybackWrapper")
    p.add_argument("--one-world", action="store_true",
                   help="the demos share a template: one playback environment for all of them (replay_group) instead of replay_obs's "
                        "replay_hdf5_file per demo")
    a = p.parse_args(argv)
    if a.one_world:
        replay_group(a.demo.split(","), a.root, a.repo_id, a.task, a.resume, a.flush_every_n_steps)
    else:
        replay(a.demo.split(","), a.root, a.repo_id, a.task, a.resume, a.flush_every_n_steps, aligned=not a.unmodified)
    return 0


if __name__ == "__main__":
    rc = main()
    og.shutdown()
    sys.exit(rc)
