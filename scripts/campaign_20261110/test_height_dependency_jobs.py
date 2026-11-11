import json
from pathlib import Path
import tempfile
import unittest

from scripts.campaign_20261110.contracts import file_digest
from scripts.campaign_20261110.height_dependency_jobs import measured_probe, positive_library


class DependencyIdentityTest(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        self.scene=self.root/'scene.json';self.scene.write_text(json.dumps({'objects_info':{'init_info':{'table':{'args':{
            'category':'table','model':'a','scale':[1,1,.9]}}}}}))
        self.profiles={'bindings':{'binding':{'profile_id':'profile','scene':'house','name':'table'}},
            'profiles':{'profile':{'ready':True,'category':'table','model':'a','default_scale':[1,1,1],
                                 'absolute_scales':[[1,1,.9],[1,1,1.1]]}}}
        self.probe=dict(schema='isg-support-height-probe-v1',geometry_admitted=True,
            profile_manifest_sha256='profile_hash',binding_id='binding',profile_id='profile',slot=0,
            scene='house',object='table',absolute_scale=[1,1,.9],conservative_overlap_blockers=[],
            measured_height_above_floor=.732123,measured_height_world=.832123,floor_z_world=.1,
            actual_height_key='0.7321',surface_hits=[{}, {}, {}],stable_scene_path=str(self.scene),
            stable_scene_sha256=file_digest(self.scene))
        self.path=self.root/'probe.json'
        self.job=dict(category='cup',model='b',scale=[.9,.9,.9],constraint='tabletop',ready=True,
            readiness_source='readiness_record',asset_sha256='asset_hash')
        self.library=dict(schema='isg-grasp-library/2',category='cup',model='b',scale=[.9,.9,.9],
            constraint='tabletop',provenance={'validator':'isg-tabletop-validator/1'},physics_profile={'physics_hz':120},
            policy_key='contact-policy',rests={'r0':{}},grasps=[{'physics':'tested','rest_id':'r0'}])
        self.reservations={'train_exclude_models':[],'category_holdout_categories':[]}

    def audit_probe(self):
        self.path.write_text(json.dumps(self.probe))
        return measured_probe(self.path,self.profiles,'profile_hash',self.root)

    def audit_library(self):
        self.path.write_text(json.dumps(self.library))
        return positive_library(self.job,self.path,self.reservations)

    def test_measured_height_survives_without_prediction(self):
        self.assertEqual(self.audit_probe()[0]['actual_height_key'],'0.7321')

    def test_changed_scene_is_not_grandfathered(self):
        self.scene.write_text('{}')
        with self.assertRaisesRegex(ValueError,'changed since probe'):self.audit_probe()

    def test_world_height_cannot_replace_floor_relative_height(self):
        self.probe['measured_height_above_floor']=.832123
        with self.assertRaisesRegex(ValueError,'relative to the floor'):self.audit_probe()

    def test_stale_profile_and_failed_geometry_remain_blocked(self):
        for field,value in [('profile_manifest_sha256','other'),('geometry_admitted',False),
                            ('absolute_scale',[.9,1,.9]),('conservative_overlap_blockers',[{'object':'wall'}]),
                            ('actual_height_key','0.7400')]:
            with self.subTest(field=field):
                old=self.probe[field];self.probe[field]=value
                with self.assertRaises(ValueError):self.audit_probe()
                self.probe[field]=old

    def test_positive_exact_scale_library(self):
        self.assertEqual(len(self.audit_library()['grasps']),1)

    def test_default_scale_library_is_not_substituted(self):
        self.library['scale']=[1,1,1]
        with self.assertRaisesRegex(ValueError,'exact target scale'):self.audit_library()

    def test_reserved_or_unready_target_rejected(self):
        self.reservations['train_exclude_models']=['cup/b']
        with self.assertRaisesRegex(ValueError,'reserved'):self.audit_library()
        self.reservations['train_exclude_models']=[];self.job['ready']=False
        with self.assertRaisesRegex(ValueError,'readiness'):self.audit_library()

    def test_untested_grasp_or_missing_rest_rejected(self):
        self.library['grasps'][0]['physics']='untested'
        with self.assertRaisesRegex(ValueError,'physics-tested'):self.audit_library()
        self.library['grasps'][0]['physics']='tested';self.library['rests']={}
        with self.assertRaisesRegex(ValueError,'stable rest'):self.audit_library()


if __name__=='__main__':unittest.main()
