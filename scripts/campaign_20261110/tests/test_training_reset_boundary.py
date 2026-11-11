import unittest
from scripts.campaign_20261110.training_batch_probe import observe_reset_boundary, install_reset_boundary_probe
from types import SimpleNamespace


class ResetBoundaryTests(unittest.TestCase):
    def test_observes_before_same_step_and_preserves_arguments(self):
        class Sim:
            value=0
            def step_physics(self, amount, *, marker):
                self.value+=amount
                return marker
        sim=Sim()
        class Scene:
            def reset(self, marker):
                sim.value=7
                return sim.step_physics(3, marker=marker)
        scene=Scene()
        result, observation=observe_reset_boundary(scene, sim,
            lambda:scene.reset('native-result'), lambda:{'value':sim.value})
        self.assertEqual(result, 'native-result')
        self.assertEqual(observation, {'value':7})
        self.assertEqual(sim.value,10)
        self.assertNotIn('reset',vars(scene))
        self.assertNotIn('step_physics',vars(sim))

    def test_exception_restores_original_instance_overrides(self):
        class Obj: pass
        scene,sim=Obj(),Obj()
        def step(): raise RuntimeError('native failure')
        def reset(): return sim.step_physics()
        scene.reset,sim.step_physics=reset,step
        with self.assertRaisesRegex(RuntimeError,'native failure'):
            observe_reset_boundary(scene,sim,scene.reset,lambda:{'read':True})
        self.assertIs(scene.reset,reset)
        self.assertIs(sim.step_physics,step)

    def test_missing_or_multiple_step_is_not_evidence(self):
        class Obj: pass
        for count in (0,2):
            scene,sim=Obj(),Obj()
            sim.step_physics=lambda:None
            scene.reset=lambda:[sim.step_physics()for _ in range(count)]
            with self.assertRaisesRegex(ValueError,'exactly one'):
                observe_reset_boundary(scene,sim,scene.reset,lambda:{})

    def test_final_checkpoint_follows_complete_loader_without_replacing_early_state(self):
        class Sim:
            value=0
            steps=0
            def step_physics(self):self.steps+=1;self.value+=1
        sim=Sim()
        class Scene:
            def reset(self):sim.value=7;sim.step_physics()
        scene=Scene()
        class Task:
            def load_tro_instance(self):scene.reset()
        task=Task()
        class Loader:
            def restore(self):
                self.task.load_tro_instance()
                self.report['campaign_restoration']={'native_observation':{'value':sim.value}}
                # Model the unchanged constraints/settle/final reset sequence.
                sim.step_physics();sim.step_physics();sim.value=42
                return ('native-observation','native-info')
        runner=SimpleNamespace(InstanceLoader=Loader,og=SimpleNamespace(sim=sim))
        loader=Loader();loader.task=task;loader.env=SimpleNamespace(scene=scene,step=sim.step_physics)
        loader.report={};loader.tro=SimpleNamespace(meta=SimpleNamespace(raw={'campaign':{}}))
        cleanups=[]
        original=install_reset_boundary_probe(runner,final_pre_action=True,next_native_control_step=True,
            pending_cleanups=cleanups,
            observer=lambda *_:{'value':sim.value,'steps':sim.steps})
        try:result=loader.restore()
        finally:Loader.restore=original
        self.assertEqual(result,('native-observation','native-info'))
        report=loader.report['campaign_restoration']
        self.assertEqual(report['native_reset_boundary_observation']['observation'],{'value':7,'steps':0})
        self.assertEqual(report['native_observation'],{'value':8})
        self.assertEqual(report['native_final_pre_action_observation']['observation'],{'value':42,'steps':3})
        self.assertEqual(sim.steps,3)
        self.assertEqual(report['next_native_control_step_observation'],{})
        loader.env.step()
        control=report['next_native_control_step_observation']
        self.assertEqual(control['before'],{'value':42,'steps':3})
        self.assertEqual(control['after'],{'value':43,'steps':4})
        self.assertEqual(len(cleanups),1)
        cleanups.pop()()
        self.assertNotIn('load_tro_instance',vars(task))
        self.assertNotIn('reset',vars(scene))


if __name__=='__main__': unittest.main()
