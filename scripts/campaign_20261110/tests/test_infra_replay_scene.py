import json
from pathlib import Path
import tempfile
import unittest

import h5py

from scripts.campaign_20261110.infra_replay_scene import prepare, verify


class RecordedSceneTests(unittest.TestCase):
    def fixture(self, root, mode="rooms"):
        template = root / "inst/template"
        demo = root / "demo/i/demo.hdf5"
        template.mkdir(parents=True)
        demo.parent.mkdir(parents=True)
        scene = {"objects_info": {"init_info": {"robot": {}, "target": {}, "lamp": {}}}}
        filename = "template.json" if mode == "full" else "template-partial_rooms.json"
        (template / filename).write_text(json.dumps(scene))
        crop = {"mode": mode, "loaded_scene_file": filename, "loaded_rooms": ["room"]}
        (template / "template_meta.json").write_text(json.dumps({"scene_crop": crop}))
        report = {"template_dir": str(template), "scene_crop": {"crop": mode, "loaded_scene_file": str(template / filename)},
                  "load_rooms": {"rooms": ["room"]}}
        (demo.parent / "meta.json").write_text(json.dumps(report))
        with h5py.File(demo, "w") as stream:
            data = stream.create_group("data")
            data.attrs["scene_file"] = json.dumps(scene)
            if mode != "full":
                data.attrs["isg_scene_crop"] = json.dumps(crop)
        return demo, template, report

    def test_recorded_crop_is_pinned_and_full_replay_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            demo, template, report = self.fixture(root)
            identity = prepare(demo, root)
            self.assertTrue(identity["partial_scene"])
            replay = root / "replay.json"
            replay.write_text(json.dumps(report))
            self.assertTrue(verify(identity, replay)["verified_replay"])
            report["scene_crop"]["crop"] = "full"
            replay.write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, "different scene mode"):
                verify(identity, replay)
            (template / "template-partial_rooms.json").write_text("changed")
            report["scene_crop"]["crop"] = "rooms"
            replay.write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, "scene bytes changed"):
                verify(identity, replay)

    def test_full_recording_stays_full(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            demo, _, _ = self.fixture(root, "full")
            self.assertFalse(prepare(demo, root)["partial_scene"])

    def test_hdf5_scene_expansion_is_rejected_before_gpu_work(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            demo, _, _ = self.fixture(root)
            with h5py.File(demo, "a") as stream:
                stream["data"].attrs["scene_file"] = json.dumps({"objects_info": {"init_info": {"robot": {}}}})
            with self.assertRaisesRegex(ValueError, "object sets differ"):
                prepare(demo, root)


if __name__ == "__main__":
    unittest.main()
