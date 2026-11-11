"""Hash-bound scene-removal checks, independent of an older native runtime."""
import json
from pathlib import Path

from scripts.campaign_20261110.contracts import file_digest, require


STRUCTURAL = frozenset((
    'walls', 'floors', 'ceilings', 'roof', 'baseboard', 'downlight',
    'room_light', 'electric_switch', 'door', 'sliding_door', 'garage_door',
    'fixed_window', 'window', 'skylight', 'stairs', 'staircase', 'railing',
))


def validate_removal_scope(root, config):
    names = config.get('perturbations', {}).get('removed_objects', [])
    if not names:
        return None
    require(isinstance(names, list) and len(names) == len(set(names)),
            'scene removals must be distinct named objects')
    setup = config['template']['scene_setup']
    reference = setup.get('augmentation_scene') or setup.get('training_height_dependency', {}).get('stable_scene')
    require(isinstance(reference, dict), 'scene removals lack a hash-bound native scene')
    path = Path(reference['path']).resolve()
    require(path.is_relative_to(Path(root).resolve()) and path.is_file()
            and file_digest(path) == reference['sha256'], 'scene-removal source changed or is outside campaign')
    stable = json.loads(path.read_text())['objects_info']['init_info']
    bindings = setup.get('inst_to_name', {})
    require(all(isinstance(bindings.get(obj['instance']), str) and bindings[obj['instance']]
                for obj in config['template']['objects']), 'scene-removal task bindings are incomplete')
    protected = set(bindings.values())
    for name in names:
        require(name in stable, 'unknown scene-removal object: ' + name)
        args = stable[name]['args']
        require(name not in protected, 'cannot park task-relevant object: ' + name)
        require(args.get('category') not in STRUCTURAL, 'cannot park structural object: ' + name)
        require(not args.get('type', '').endswith('Robot'), 'cannot park a robot: ' + name)
    return {'scene': reference, 'removed_objects': sorted(names), 'protected_objects': sorted(protected)}
