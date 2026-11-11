import unittest
from scripts.campaign_20261110.training_batch_probe import observe_reset_boundary


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


if __name__=='__main__': unittest.main()
