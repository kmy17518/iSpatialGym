import json
from pathlib import Path
import tempfile
import unittest
import shutil

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
            shutil.copyfile(demo, replay.parent / 'demo.hdf5')
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

    def test_native_room_filter_explains_only_exact_subset_and_replay_checks_actual_objects(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            demo, template, report = self.fixture(root, mode='full')
            runtime = root / 'code'
            native = Path(__file__).parent / 'data/native_room_filter_6a619.py'
            for relative in ('scenes/interactive_traversable_scene.py', 'utils/constants.py'):
                destination = runtime / 'Behavior-1K/OmniGibson/omnigibson' / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(native, destination)
            def obj(name, category, rooms):
                return {'class_module': 'omnigibson.objects.dataset_object', 'class_name': 'DatasetObject',
                        'args': {'name': name, 'category': category, 'model': 'ready', 'scale': [1, 1, 1], 'in_rooms': rooms}}
            objects = {'target': obj('target', 'apple', ['room']), 'wall': obj('wall', 'walls', []),
                       'door': obj('door', 'door', ['omitted_room']), 'omitted': obj('omitted', 'chair', ['omitted_room']),
                       'floor': obj('floor', 'floors', ['omitted_room']), 'robot': obj('robot', 'agent', [])}
            objects['robot']['class_module'] = 'omnigibson.robots.robot'
            source = {'objects_info': {'init_info': objects}}
            recorded = {'objects_info': {'init_info': {name: objects[name] for name in ('target', 'wall', 'door', 'robot')}},
                        'init_info': {'class_name': 'InteractiveTraversableScene', 'args': {'load_room_instances': ['room']}}}
            (template / 'template.json').write_text(json.dumps(source))
            report['provenance'] = {'isg_root': str(runtime)}
            (demo.parent / 'meta.json').write_text(json.dumps(report))
            with h5py.File(demo, 'a') as stream:
                stream['data'].attrs['scene_file'] = json.dumps(recorded)
            identity = prepare(demo, root)
            self.assertEqual(identity['objects'], ['door', 'robot', 'target', 'wall'])
            self.assertEqual(identity['room_projection']['rooms'], ['room'])
            replay = root / 'replay/meta.json'
            replay.parent.mkdir()
            replay.write_text(json.dumps(report))
            shutil.copyfile(demo, replay.parent / 'demo.hdf5')
            self.assertTrue(verify(identity, replay)['verified_replay'])
            recorded['objects_info']['init_info'].pop('target')
            with h5py.File(replay.parent / 'demo.hdf5', 'a') as stream:
                stream['data'].attrs['scene_file'] = json.dumps(recorded)
            with self.assertRaisesRegex(ValueError, 'actual replay object set'):
                verify(identity, replay)
            with h5py.File(demo, 'a') as stream:
                stream['data'].attrs['scene_file'] = json.dumps(recorded)
            with self.assertRaisesRegex(ValueError, 'after exact native room filtering'):
                prepare(demo, root)


if __name__ == "__main__":
    unittest.main()
