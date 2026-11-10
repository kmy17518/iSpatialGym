import json
import os
from pathlib import Path
import tempfile
import unittest
from scripts.campaign_20261110.infra_release_floating_aliases import physical_digest, replace_json

class FloatingRouteReleaseTests(unittest.TestCase):
    def test_only_additional_routing_aliases_are_excluded_from_physical_proof(self):
        physical={'legacy_key':'apple__test__s1__floating','grasps':[{'q':[0,1]}],'scale':[1,1,1]}
        self.assertEqual(physical_digest(physical),physical_digest({**physical,'legacy_keys':['apple__test__f']}))
        self.assertNotEqual(physical_digest(physical),physical_digest({**physical,'grasps':[{'q':[0,2]}]}))
        self.assertNotEqual(physical_digest(physical),physical_digest({**physical,'scale':[2,2,2]}))
        self.assertNotEqual(physical_digest(physical),physical_digest({**physical,'legacy_key':'unrelated'}))

    def test_alias_write_detaches_hard_link_and_preserves_old_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            old=Path(tmp)/'old.json';new=Path(tmp)/'new.json'
            old.write_text('{"grasps":[1]}');old.chmod(0o444);os.link(old,new)
            before=old.read_bytes();replace_json(new,{'grasps':[1],'legacy_keys':['native']})
            self.assertEqual(old.read_bytes(),before)
            self.assertNotEqual(old.stat().st_ino,new.stat().st_ino)
            self.assertEqual(json.loads(new.read_text())['grasps'],[1])

if __name__=='__main__':unittest.main()
