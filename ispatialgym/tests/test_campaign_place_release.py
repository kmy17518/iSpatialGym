import unittest
from ispatialgym.oracle.campaign_place_release import target_clear


class ReleaseClearanceTest(unittest.TestCase):
    def test_release_requires_current_contact_evidence(self):
        for held, pairs in [(True, []), (False, None)]:
            self.assertFalse(target_clear('/robot', '/soap', 'left', held, pairs))
        self.assertFalse(target_clear('', '/soap', 'left', False, []))

    def test_either_pad_contact_and_pair_orientation_blocks(self):
        for finger in (1, 2):
            pair=('/robot/left_gripper_finger_link'+str(finger), '/soap/base_link')
            self.assertFalse(target_clear('/robot', '/soap', 'left', False, [pair]))
            self.assertFalse(target_clear('/robot', '/soap', 'left', False, [pair[::-1]]))

    def test_support_contact_can_be_relieved_by_retreat(self):
        pairs=[('/robot/left_gripper_finger_link1', '/dish/base_link')]
        self.assertTrue(target_clear('/robot', '/soap', 'left', False, pairs))
        self.assertTrue(target_clear('/robot', '/soap', 'left', False, []))

    def test_motion_keeps_full_open_and_only_releases_after_clear_observations(self):
        import sys
        import torch as th
        from types import SimpleNamespace
        from unittest.mock import patch
        from ispatialgym.oracle.motions import Motions

        def run(enabled, contacts, release=True):
            target=SimpleNamespace(name='soap',prim_path='/soap',n_joints=0,joints={})
            robot=SimpleNamespace(prim_path='/robot',scene=SimpleNamespace(idx=0),gripper_control_idx={'left':[0,1]},
                get_joint_positions=lambda:th.tensor([.014,.014]),get_joint_velocities=lambda:th.ones(2))
            m=SimpleNamespace(robot=robot,cfg=SimpleNamespace(gripper_steps=60),steps=0,gripper_intent_closed={},release_events=[],
                _gripper_limits=lambda arm:(0.,.05))
            m.held_object=lambda arm:target if m.steps<2 or not release else None
            commands=[]
            m.state=SimpleNamespace(stop_base=lambda:None,set_gripper=lambda arm,closed:commands.append((arm,closed)))
            def emit(**kwargs):
                m.steps+=1
                return m.steps
            m._emit=emit
            m.hold=lambda n:(emit() for _ in range(n))
            api=SimpleNamespace(get_contact_pairs=lambda *a,**kw:contacts)
            with patch.dict(sys.modules,{'omnigibson.utils.usd_utils':SimpleNamespace(RigidContactAPI=api)}),patch('ispatialgym.oracle.motions.oracle_still_exit',return_value=False):
                list(Motions.gripper(m,'left',False,release_when_clear=enabled))
            self.assertEqual(commands,[('left',False)])
            return m

        clear=run(True,[])
        self.assertEqual(clear.steps,13)
        self.assertEqual(clear.release_events[-1]['clear_observations'],2)
        self.assertEqual(clear.release_events[-1]['gripper_command'],1.)
        for enabled,contacts,release in [(False,[],True),(True,None,True),(True,[],False),
                (True,[('/robot/left_gripper_finger_link1','/soap/base_link')],True)]:
            with self.subTest(enabled=enabled,contacts=contacts,release=release):
                self.assertEqual(run(enabled,contacts,release).steps,65)
