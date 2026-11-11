import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.campaign_20261110.canonical_validation import audit


class CanonicalValidationTests(unittest.TestCase):
    def observe(self, cls, task, members, group=None, normalize=None):
        catalog={cls:task,**{f'unattempted_{i}':'disembedding' for i in range(99)}}
        groups={'g':group} if group else {}
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for member in members:
                tro=root/(member['instance_id']+'.json')
                tro.write_text(json.dumps({'ispatialgym':{'campaign':{'runtime':{'goal_group':{
                    'requested_goal':{'fixed':1},'requested_start':{'same':1}}}}}}))
                member['tro']=str(tro)
            def normalized(member, **kw):
                if normalize: return normalize(member,**kw)
                return ({'instance_id':member['instance_id'],'split':member['split'],'gate':{},'evidence':[],
                    'validation_key':{},'perturbations':{},'template':{},'artifacts':{'tro':{'path':member['tro']}}},
                    SimpleNamespace(class_id=cls,split='eval_standard'))
            with patch('scripts.campaign_20261110.canonical_validation.snapshot_allocations',return_value=(members,groups,[])), \
                 patch('scripts.campaign_20261110.canonical_validation.active_policy',return_value=({}, {}, {})), \
                 patch('scripts.campaign_20261110.canonical_validation.normalize_record',side_effect=normalized):
                return audit(root,catalog,None,0)

    def record(self, cls, task, iid, split='eval_standard', group='g'):
        return dict(class_id=cls,task=task,instance_id=iid,split=split,group_id=group,validation_key={},contract={'class_id':cls})

    def test_historical_hold_is_never_counted_as_canonical_validation(self):
        cls='inter_object_relation_satisfaction-under_large_scale_01';task='inter_object_relation_satisfaction-under_large_scale'
        data=self.observe(cls,task,[self.record(cls,task,'e')])
        row=next(r for r in data['classes'] if r['class_id']==cls)
        self.assertEqual(row['historical_accepted_standard'],1)
        self.assertFalse(row['validated']);self.assertEqual(row['status'],'parked_owner_artifact')

    def test_released_artifact_requires_chain_instead_of_remaining_parked(self):
        cls='inter_object_relation_satisfaction-nextto_04';task='inter_object_relation_satisfaction-nextto_small_scale'
        row=next(r for r in self.observe(cls,task,[self.record(cls,task,'e')],{'accepted':{'e':'eval_standard'}})['classes'] if r['class_id']==cls)
        self.assertEqual(row['status'],'awaiting_complete_proof')
        self.assertFalse(row['validated']);self.assertIn('lacks2–3',row['errors'][0]['reason'])

    def test_accepted_count_without_sibling_cannot_validate(self):
        cls='disembedding_01';task='disembedding';m=self.record(cls,task,'e')
        row=next(r for r in self.observe(cls,task,[m],{'accepted':{'e':'eval_standard'}})['classes'] if r['class_id']==cls)
        self.assertFalse(row['validated']);self.assertIn('lacks2–3',row['errors'][0]['reason'])

    def test_camera_exemption_still_requires_exact_artifact_audit(self):
        cls='camera_relocalization-standard';task=cls;m=self.record(cls,task,'e',group=None)
        self.assertEqual(self.observe(cls,task,[m])['validated'],1)
        def reject(*args,**kw):raise ValueError('actual replay failed')
        self.assertEqual(self.observe(cls,task,[m],normalize=reject)['validated'],0)

    def test_distinct_ids_do_not_prove_distinct_native_members(self):
        cls='disembedding_01';task='disembedding';members=[self.record(cls,task,'e'),self.record(cls,task,'t','train')]
        group={'accepted':{'e':'eval_standard','t':'train'}}
        row=next(r for r in self.observe(cls,task,members,group)['classes']if r['class_id']==cls)
        self.assertFalse(row['validated']);self.assertIn('not distinct',row['errors'][0]['reason'])


if __name__=='__main__':unittest.main()
