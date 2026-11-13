"""Explicit per-loader lifecycle for a source-checked fixed actor union.

This is an opt-in native consumer of the campaign's exact pool contract. It
performs the existing measured pool operations at named loader boundaries;
it grants no execution/release authority and never changes source artifacts.
"""
import copy
import json
from pathlib import Path

from ispatialgym.sampling.model_pool import require, digest
from .pool_consumers import PoolConsumers


class PoolWorld:
    def __init__(self, contract, tro_path, output):
        self.contract = copy.deepcopy(contract)
        self.contract_sha256 = digest(contract)
        self.consumers = PoolConsumers(contract, tro_path)
        self.member = self.consumers.member
        self.output = Path(output).resolve()
        require(self.output.is_dir(), 'native pool output must already be private and prepared')
        self.loader = None
        self.cold_anchor = None
        self.expected_scene = None
        self.isolation = None
        self._incoming_state = None
        self.events = []

    @classmethod
    def from_source_sequence(cls, root, source_reference, tro_path, output):
        # This CPU contract rechecks current source reviews, exact fixed-world
        # identity, role mappings and additive pool compatibility. The ordinary
        # loader still independently checks execution authorization.
        from scripts.campaign_20261110.training_pool_world import checked, prepare
        root = Path(root).resolve()
        checked(root, source_reference)
        require(Path(output).resolve().is_relative_to(root), 'native pool output escaped campaign')
        return cls(prepare(root, source_reference['path']), tro_path, output)

    def bind_loader(self, loader):
        require(self.loader is None and loader.tro_path == self.member['tro']['path'],
                'native pool lifecycle belongs to one incoming source')
        require(getattr(loader, '_sequential_augmentation', None) is None,
                'pool plus sequential augmentation requires a combined reviewed contract')
        self.loader = loader
        self.consumers.bind_loader(loader)
        from scripts.campaign_20261110.training_pool_world import isolated_member_scene
        source = json.loads(Path(self.member['scene']['path']).read_text())
        self.expected_scene, self.isolation = isolated_member_scene(
            self.contract, source, self.member['active'])

    def configuration(self, config):
        from scripts.campaign_20261110.training_pool_world import private_member_filename
        require(self.loader is not None and self.expected_scene is not None,
                'pool configuration has no incoming source')
        path = self.output/private_member_filename(self.member['active'])
        require(not path.exists(), 'preserve existing pool initialization artifact')
        with path.open('x') as stream:
            json.dump(self.expected_scene, stream, indent=2)
            stream.write('\n')
        result = copy.deepcopy(config)
        result['scene']['scene_file'] = str(path)
        return result

    def cold_stopped(self, sim):
        from scripts.campaign_20261110.training_pool_probe import _incoming_spawn
        from scripts.campaign_20261110.training_pool_source_anchor_probe import apply_cold_anchor
        require(sim.is_stopped() and self.loader.robot.is_holonomic_base,
                'pool cold anchor requires the existing stopped holonomic-base lifecycle')
        incoming = _incoming_spawn(self.loader, self.loader.robot)
        # The original scene loader already authored this native root. A second
        # setter round-trips its quaternion through the fixed joint and changes
        # the original cold trajectory. Check our own loaded anchor without
        # rewriting it; warm source transitions remain separately measured.
        anchor = apply_cold_anchor(self.loader.robot, incoming, preserve=True)
        self.events.append({'phase':'cold_existing_mass_stop','source_spawn':copy.deepcopy(incoming),
                            'extra_physics_steps':0, **anchor})
        self.observe_reset('cold_stopped_before_existing_play', sim)

    def initialized(self):
        from ispatialgym.sampling.model_pool import restore_export
        from scripts.campaign_20261110.training_pool_adapter import pool_restore_document
        from ispatialgym.oracle.campaign_batch import capture_robot_anchor
        import omnigibson as og
        loader = self.loader
        require(len(self.events)==1, 'pool cold anchor did not join one existing mass lifecycle')
        self.observe_reset('cold_before_pool_restore', og.sim)
        restore_export(loader.env.scene, og.sim, pool_restore_document(self.expected_scene, self.isolation))
        self.observe_reset('cold_after_pool_restore', og.sim)
        loader._scene_state0 = copy.deepcopy(loader.env.scene.dump_state(serialized=False))
        self.cold_anchor = capture_robot_anchor(loader)
        self._observe_task(rebind=False)
        self.observe('world_initialized_before_first_restore')

    def share(self, shared):
        require(self.cold_anchor is not None, 'pool has no initialized native anchor')
        if 'build_report' in shared:
            # InstanceLoader shares build measurements before the first restore
            # and recording. Its shallow report copy otherwise retains the first
            # episode's mutable lifecycle and serialized-state lists. Incoming
            # loaders must publish their own task, isolation and reader evidence.
            shared['build_report'] = {
                key: copy.deepcopy(value) for key, value in shared['build_report'].items()
                if not key.startswith('model_pool_') and key != 'pool_exact_task_binding'
            }
        shared['model_pool_world'] = {'contract_sha256':self.contract_sha256,
                                    'cold_anchor':copy.deepcopy(self.cold_anchor)}

    def compatible(self, shared):
        current = shared.get('model_pool_world')
        require(isinstance(current, dict) and current.get('contract_sha256')==self.contract_sha256,
                'incoming pool does not match this initialized world contract')
        return True

    def prepare_incoming(self, shared):
        from scripts.campaign_20261110.training_pool_adapter import merge_member_state
        from scripts.campaign_20261110.training_pool_source_anchor_probe import source_anchor
        from scripts.campaign_20261110.training_pool_probe import _incoming_spawn
        from omnigibson.utils.python_utils import recursively_convert_to_torch
        import torch
        self.compatible(shared)
        self.cold_anchor = copy.deepcopy(shared['model_pool_world']['cold_anchor'])
        anchor = source_anchor(self.cold_anchor, _incoming_spawn(self.loader, shared['robot']))
        scene = shared['env'].scene
        def local(position, orientation):
            return scene.convert_world_pose_to_scene_relative(
                torch.tensor(position,dtype=torch.float32),torch.tensor(orientation,dtype=torch.float32))
        state = merge_member_state(shared['scene_state0'],self.expected_scene,shared['robot'].name,anchor,local)
        self._incoming_state = {'anchor':anchor,'loaded':False}
        return recursively_convert_to_torch(state)

    def load_incoming(self, state, sim):
        from scripts.campaign_20261110.training_pool_adapter import restart_without_steps, pool_restore_document
        from scripts.campaign_20261110.training_pool_native_anchor_probe import anchored_scene_state
        from ispatialgym.sampling.model_pool import restore_export
        from .pool_robot_anchor import author_native_root_joint
        import torch
        loader = self.loader; incoming = self._incoming_state
        require(incoming is not None and incoming['loaded'] is False,
                'incoming pool state must load exactly once after native cleanup')
        anchor = incoming['anchor']; incoming['loaded'] = True
        joint_event = {}
        self.observe_reset('incoming_before_stop', sim)
        def stopped_anchor():
            self.observe_reset('incoming_stopped_before_anchor', sim)
            joint_event.update(author_native_root_joint(loader.robot, sim, anchor))
            self.observe_reset('incoming_stopped_after_anchor', sim)
        transition = restart_without_steps(sim, stopped_anchor)
        self.observe_reset('incoming_after_play', sim)
        pose = tuple(v.clone() for v in loader.robot.root_link.get_position_orientation())
        prepared = anchored_scene_state(state,loader.robot.name,pose,loader.env.scene.get_position_orientation())
        result = loader.env.scene.load_state(prepared,serialized=False)
        actual = loader.robot.root_link.get_position_orientation()
        require(all(torch.equal(a,b)for a,b in zip(pose,actual)), 'incoming restore changed its own native anchor')
        self.observe_reset('incoming_after_scene_load', sim)
        restore_export(loader.env.scene,sim,pool_restore_document(self.expected_scene,self.isolation))
        self.observe_reset('incoming_after_pool_restore', sim)
        self.events.append({'phase':'incoming_after_native_cleanup','anchor':copy.deepcopy(anchor),
                            'transition':transition,'joint_anchor':joint_event,
                            'native_anchor_bit_identical':True})
        return result

    def attached(self):
        require(self._incoming_state is not None and self._incoming_state['loaded'] is True,
                'incoming pool restoration boundary was omitted')
        self._observe_task(rebind=True)
        self.observe('incoming_scene_loaded_before_task_restore')

    def _observe_task(self, *, rebind):
        from scripts.campaign_20261110.training_pool_task import observe_task, rebind_task
        loader = self.loader
        require(self.member.get('task_definition') is not None,
                'native pool needs an explicit exact role/task contract')
        loader.report['pool_exact_task_binding'] = (rebind_task if rebind else observe_task)(loader,self.member)

    def observe_reset(self, phase, sim):
        """Read native boundary state without setters, steps or reference-state imports.

        Stopped observations intentionally avoid articulation/velocity handles.
        Preserve raw USD quaternion components as well as measured world poses:
        their conversion can distinguish cold construction from a warm setter.
        """
        loader = self.loader
        robot = loader.robot

        def plain(value):
            if isinstance(value, dict):
                return {key: plain(item) for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return [plain(item) for item in value]
            if hasattr(value, 'tolist'):
                return plain(value.tolist())
            return value

        def attributes(prim, names):
            result = {}
            for name in names:
                attribute = prim.GetAttribute(name)
                value = attribute.Get() if attribute else None
                if value is not None and hasattr(value, 'GetReal'):
                    value = [*value.GetImaginary(), value.GetReal()]
                elif value is not None:
                    value = list(value)
                result[name] = value
            return result

        clock = lambda: (float(sim.current_time), int(sim.current_time_step_index))
        before = clock()
        stopped = sim.is_stopped()
        xforms = ('xformOp:translate', 'xformOp:orient', 'xformOp:scale')
        native = {
            'root_world_pose': plain(robot.root_link.get_position_orientation()),
            'base_world_pose': plain(robot.get_position_orientation()),
            'entity_usd': attributes(robot.prim, xforms),
            'root_usd': attributes(robot.root_link.prim, xforms),
            'base_usd': attributes(robot.base_footprint_link.prim, xforms),
            'world_joint_usd': attributes(robot._world_base_fixed_joint_prim,
                                          ('physics:localPos0', 'physics:localRot0',
                                           'physics:localPos1', 'physics:localRot1')),
        }
        active = {}
        if not stopped:
            native['state'] = plain(robot.dump_state(serialized=False))
            for name in self.consumers.active:
                obj = loader.env.scene.object_registry('name', name)
                require(obj is not None, 'reset observation lost active actor: ' + name)
                active[name] = plain(obj.dump_state(serialized=False))
        require(clock() == before, 'reset observation advanced the native simulation clock')
        row = {'phase': phase, 'source_tro': self.member['tro'],
               'sim_time': before[0], 'sim_step': before[1], 'stopped': stopped,
               'robot': native, 'active_states': active,
               'state_writes': 0, 'physics_steps_requested': 0}
        # Detach every mutable value and reject nonfinite/unserializable native
        # evidence now, instead of corrupting the eventual episode report.
        row = json.loads(json.dumps(row, allow_nan=False))
        loader.report.setdefault('model_pool_reset_observations', []).append(row)

    def observe(self, phase):
        from scripts.campaign_20261110.training_pool_adapter import native_isolation
        loader = self.loader
        bindings = {k:v.name for k,v in loader.task.physical_object_scope.items() if v is not None}
        observed = native_isolation(loader.env.scene,loader.robot,self.isolation,bindings,None)
        loader.report.setdefault('model_pool_world',[]).append({
            'phase':phase,'contract_sha256':self.contract_sha256,'source_tro':self.member['tro'],
            'active_names':list(self.consumers.active),'isolation':observed,
            'lifecycle_events':copy.deepcopy(self.events),'production_authorization':False})
