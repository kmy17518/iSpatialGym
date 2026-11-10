"""Forward only an exact private stable scene already bound by native sampling."""
from pathlib import Path

from scripts.campaign_20261110.contracts import file_digest, require


def private_scene_environment(template, root):
    record = template['scene_setup'].get('private_scene')
    if record is None:
        return {}
    path = Path(record['path']).resolve()
    require(record.get('schema') == 'isg-native-private-scene-v1', 'invalid private scene reference')
    require(path.is_relative_to(Path(root).resolve()), 'private scene must belong to the campaign')
    require(path.name == template['scene'] + '_stable.json', 'private scene name differs from template')
    require(file_digest(path) == record['sha256'] == template['scene_setup']['native_scene_sha256'],
            'private scene bytes differ from native candidate')
    return {'ISPATIALGYM_STABLE_SCENE_DIR': str(path.parent)}
