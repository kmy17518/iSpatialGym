import hashlib

import pytest

from scripts.campaign_20261110.private_scene import private_scene_environment


def fixture(tmp_path):
    path = tmp_path / 'scene' / 'Rs_int_stable.json'
    path.parent.mkdir()
    path.write_text('{}')
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    template = {'scene': 'Rs_int', 'scene_setup': {'native_scene_sha256': sha,
        'private_scene': {'schema': 'isg-native-private-scene-v1', 'path': str(path), 'sha256': sha}}}
    return path, template


def test_exact_private_scene_forwarded(tmp_path):
    path, template = fixture(tmp_path)
    assert private_scene_environment(template, tmp_path) == {'ISPATIALGYM_STABLE_SCENE_DIR': str(path.parent)}


def test_default_scene_has_no_override(tmp_path):
    assert private_scene_environment({'scene_setup': {}}, tmp_path) == {}


def test_private_scene_mutation_is_rejected(tmp_path):
    path, template = fixture(tmp_path)
    path.write_text('{"changed": true}')
    with pytest.raises(ValueError, match='bytes differ'):
        private_scene_environment(template, tmp_path)


def test_private_scene_must_be_owned(tmp_path):
    _, template = fixture(tmp_path)
    with pytest.raises(ValueError, match='belong'):
        private_scene_environment(template, tmp_path / 'other_campaign')
