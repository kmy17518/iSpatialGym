"""CPU-only reproduction of D04 publication accounting on exact source methods."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
import unittest
import difflib

ROOT=Path(__file__).resolve().parents[3]
COMMIT='2808cf5ab2abbe1f2e93f00f36b494996b2fb1a8'
REL='ispatialgym/sampling/v2/tasks/disembedding.py'
SOURCE=subprocess.check_output(['git','show',f'{COMMIT}:{REL}'],cwd=ROOT,text=True)
PATCHED=(ROOT/REL).read_text()

class Rejected(Exception):pass
class Base:
    def write_instance(self,draft,index,seed):
        self.writes+=1
        if self.write_failure:raise Rejected('write')
        return str(index)
    def check_goal_occlusion(self,draft,path):
        self.visibility_calls+=1
        if draft.occluded:raise Rejected('goal_occluded')

def load(source):
    tree=ast.parse(source)
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Disembedding')
    cls.bases=[ast.Name(id='Base',ctx=ast.Load())]
    cls.decorator_list=[]
    cls.body=[m for m in cls.body if isinstance(m,ast.FunctionDef) and m.name in ('write_instance','check_goal_occlusion')]
    module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),cls],type_ignores=[])
    scope={'Base':Base}
    exec(compile(ast.fix_missing_locations(module),'<exact-disembedding-methods>','exec'),scope)
    obj=scope['Disembedding']()
    obj._campaign_tray_active=object();obj._campaign_group_written=0
    obj.template_meta={};obj.writes=0;obj.visibility_calls=0;obj.write_failure=False
    obj.campaign_verify_goal=lambda d:None;obj.instance_id=lambda i:str(i)
    return obj

def attempt(s,*,occluded=False,filter_rejected=False):
    d=SimpleNamespace(summary={'campaign_goal_group':{}},extras={},occluded=occluded)
    path=s.write_instance(d,0,42)
    if filter_rejected:raise Rejected('instance_filter')
    s.check_goal_occlusion(d,path)

class Accounting(unittest.TestCase):
    def test_frozen_reproduction_counts_two_visibility_rejects_and_one_success_as_three(self):
        s=load(SOURCE)
        for _ in range(2):
            with self.assertRaisesRegex(Rejected,'goal_occluded'):attempt(s,occluded=True)
        attempt(s)
        self.assertEqual(s._campaign_group_written,3)
        self.assertEqual(s.visibility_calls,3)
    def test_candidate_preserves_rejections_and_only_counts_survivor(self):
        s=load(PATCHED)
        for _ in range(2):
            with self.assertRaisesRegex(Rejected,'goal_occluded'):attempt(s,occluded=True)
        attempt(s)
        self.assertEqual(s._campaign_group_written,1)
        self.assertEqual(s.visibility_calls,3)
    def test_instance_filter_reject_does_not_consume_member(self):
        s=load(PATCHED)
        with self.assertRaisesRegex(Rejected,'instance_filter'):attempt(s,filter_rejected=True)
        self.assertEqual(s._campaign_group_written,0)
        self.assertEqual(s.visibility_calls,0)
    def test_failed_write_does_not_consume_member(self):
        s=load(PATCHED);s.write_failure=True
        with self.assertRaisesRegex(Rejected,'write'):attempt(s)
        self.assertEqual(s._campaign_group_written,0)
    def test_native_goal_rejection_is_preserved(self):
        s=load(PATCHED)
        def fail(d):raise Rejected('physical_goal')
        s.campaign_verify_goal=fail
        with self.assertRaisesRegex(Rejected,'physical_goal'):attempt(s)
        self.assertEqual(s.writes,0)
        self.assertEqual(s._campaign_group_written,0)
    def test_three_survivors_reach_existing_limit(self):
        s=load(PATCHED)
        for _ in range(3):attempt(s)
        self.assertEqual(s._campaign_group_written,3)
    def test_noncampaign_does_not_add_group_accounting(self):
        s=load(PATCHED);s._campaign_tray_active=None
        attempt(s)
        self.assertEqual(s._campaign_group_written,0)
        self.assertEqual(s.visibility_calls,1)

