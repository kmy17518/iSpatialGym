import copy
import json
from pathlib import Path
import tempfile
import unittest
from scripts.campaign_20261110.audit_selected_libraries import validate_record
from scripts.campaign_20261110.height_dependency_jobs import selected_libraries
from scripts.campaign_20261110.contracts import file_digest


class SelectedLibraryAuditTests(unittest.TestCase):
    def setUp(self):
        self.job=dict(category='cup',model='abc',dataset='assets',scale=[1.,1.,1.],constraint='floating',
                      ready=True,readiness_source='catalog',asset_sha256='a'*64,held_tasks=['object_scaling'])
        self.policy=dict(train_exclude_models=[],category_holdout_categories=[])
        self.data={**self.job,'schema':'isg-grasp-library/2','provenance':{'validator':'isg-floating-validator/2'},
                   'physics_profile':{'physics_hz':120},'policy_key':'policy','rests':{},
                   'grasps':[{'physics':'tested','rest_id':None,'held_motions':{'object_scaling':{'motions_executed':3}}}]}

    def test_zero_grasp_output_is_not_a_verified_dependency(self):
        self.data['grasps']=[]
        with self.assertRaisesRegex(ValueError,'zero positive'): validate_record(self.job,self.data,self.policy)

    def test_floating_requires_actual_task_motion(self):
        self.assertEqual(validate_record(self.job,self.data,self.policy),1)
        self.data['grasps'][0]['held_motions']={}
        with self.assertRaisesRegex(ValueError,'motion was not executed'): validate_record(self.job,self.data,self.policy)

    def test_exact_scale_and_holdout_are_checked(self):
        self.data['scale']=[1.01]*3
        with self.assertRaisesRegex(ValueError,'mismatch'): validate_record(self.job,self.data,self.policy)
        self.data['scale']=self.job['scale'];self.policy['train_exclude_models']=['cup/abc']
        with self.assertRaisesRegex(ValueError,'reserved'): validate_record(self.job,self.data,self.policy)

    def test_tabletop_cannot_substitute_floating_or_missing_rest(self):
        self.job['constraint']=self.data['constraint']='tabletop'
        with self.assertRaisesRegex(ValueError,'validator mismatch'): validate_record(self.job,self.data,self.policy)
        self.data['provenance']['validator']='isg-tabletop-validator/1'
        with self.assertRaisesRegex(ValueError,'measured rest'): validate_record(self.job,self.data,self.policy)
        self.data['grasps'][0]['rest_id']='r0';self.data['rests']={'r0':{}}
        self.assertEqual(validate_record(self.job,self.data,self.policy),1)

    def test_height_requests_reject_changed_audited_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp);job={**self.job,'constraint':'tabletop','id':'exact'}
            manifest=d/'jobs.json';manifest.write_text(json.dumps([job]))
            library=d/'library.json';library.write_text('{}')
            ref=lambda p:dict(path=str(p),sha256=file_digest(p))
            row=dict(job=job,status='verified',native_reader_passed=True,
                     source_manifest=ref(manifest),source_index=0,library=ref(library))
            audit=d/'audit.json';audit.write_text(json.dumps({'schema':'isg-native-library-harvest-v1','verified':[row]}))
            jobs,paths=selected_libraries(audit)
            self.assertEqual(jobs,[job]);self.assertEqual(paths['exact'],row['library'])
            library.write_text('{"changed":true}')
            with self.assertRaisesRegex(ValueError,'library changed'): selected_libraries(audit)


if __name__=='__main__': unittest.main()
