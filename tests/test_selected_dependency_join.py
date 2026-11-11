import unittest
from scripts.campaign_20261110.join_selected_dependencies import join


class SelectedDependencyJoinTests(unittest.TestCase):
    def fixture(self):
        target=dict(asset_sha256='a'*64,category='cup',model='a',dataset='assets',applied_scale=[1.]*3)
        base=dict(inputs={'target':target,'arm':None},required_by_classes=['object_scaling-cup'],
                  unresolved_inputs=['physical generator component fingerprint'],status='blocked',physical_artifact_id=None)
        requests=[dict(base,id='lib',kind='floating_grasp_library'),
                  dict(base,id='right',kind='floating_native_motion_validation',inputs={'target':target,'arm':'right'}),
                  dict(base,id='left',kind='floating_native_motion_validation',inputs={'target':target,'arm':'left'})]
        job=dict(category='cup',model='a',dataset='assets',scale=[1.]*3,asset_sha256='a'*64,
                 constraint='floating',campaign_classes=['object_scaling-cup'],held_tasks=['object_scaling'])
        audit=dict(observed_at='now',verified=[dict(job=job,validated_arm='right',native_identity_key='key',
                   native_identity={},library={'path':'library','sha256':'b'*64})],failed=[])
        manifest=dict(artifact_requests=requests,configurations=[dict(id='c',class_id='object_scaling-cup',
                      dependencies=['lib','right','left'],status='blocked',configuration_sha256=None)],
                      per_class=[dict(class_id='object_scaling-cup')],summary={})
        return manifest,audit

    def test_right_arm_proof_does_not_clear_left_or_materialization(self):
        m,a=self.fixture();v=join(m,a,{}, {})
        self.assertEqual([r['status'] for r in v['artifact_requests']],['verified','verified','blocked'])
        self.assertEqual(v['summary']['fully_served_configurations'],0)
        self.assertEqual(v['summary']['exact_materialized_configurations'],0)
        self.assertEqual(m['artifact_requests'][0]['status'],'blocked')

    def test_other_scale_and_asset_do_not_match(self):
        m,a=self.fixture();a['verified'][0]['job']['scale']=[1.01]*3
        self.assertFalse(join(m,a,{}, {})['native_library_join']['verified_request_ids'])
        a['verified'][0]['job']['scale']=[1.]*3;a['verified'][0]['job']['asset_sha256']='c'*64
        self.assertFalse(join(m,a,{}, {})['native_library_join']['verified_request_ids'])

    def test_unknown_joint_or_geometry_obligation_stays_open(self):
        m,a=self.fixture();m['artifact_requests'][0]['unresolved_inputs'].append('exact native geometry')
        self.assertEqual(join(m,a,{}, {})['artifact_requests'][0]['status'],'blocked')

    def test_repaired_grasp_does_not_leave_unmeasured_arm_as_a_failed_new_trial(self):
        m,a=self.fixture();m['artifact_requests'][2].update(status='failed',last_failure='old library had zero grasps')
        v=join(m,a,{}, {})['artifact_requests'][2]
        self.assertEqual(v['status'],'blocked')
        self.assertIsNone(v['physical_artifact_id'])
        self.assertIn('left-arm',v['unresolved_inputs'][0])
        self.assertEqual(v['superseded_prerequisite_failure'],'old library had zero grasps')


if __name__=='__main__':unittest.main()
