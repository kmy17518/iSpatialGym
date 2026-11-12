"""Preserve exact measured private support inputs from a validated source."""
import hashlib
import json
import math
from pathlib import Path

from .contracts import require

KEYS = ('ISPATIALGYM_SUPPORT_INVENTORY_MANIFEST', 'ISPATIALGYM_SUPPORT_INVENTORY_SHA256')


def checked_context(environment, root):
    """Verify evidence bytes and private scene identity; never change support rules."""
    if not any(environment.get(key) for key in KEYS):
        return None
    require(all(environment.get(key) for key in KEYS), 'support inventory requires manifest and digest')
    root = Path(root).resolve()
    artifacts = []

    def checked(reference):
        path = Path(reference['path']).resolve()
        require(path.is_relative_to(root) and path.is_file(), 'support inventory evidence outside campaign')
        payload = path.read_bytes()
        require(hashlib.sha256(payload).hexdigest() == reference['sha256'], 'support inventory evidence changed')
        artifacts.append({'path':str(path), 'sha256':reference['sha256']})
        return json.loads(payload)

    manifest_ref = {'path':environment[KEYS[0]], 'sha256':environment[KEYS[1]]}
    manifest = checked(manifest_ref)
    require(manifest.get('schema') == 'isg-private-support-inventory-v1' and
            isinstance(manifest.get('entries'), list) and manifest['entries'], 'invalid support inventory manifest')
    stable = environment.get('ISPATIALGYM_STABLE_SCENE_DIR')
    require(stable and Path(stable).resolve().is_relative_to(root), 'support inventory lacks private stable scene')
    seen = set()
    for entry in manifest['entries']:
        identity = (entry['scene'], entry['name'])
        require(identity not in seen, 'duplicate support inventory identity')
        seen.add(identity)
        require(Path(entry['scene_file']['path']).resolve() ==
                (Path(stable)/(entry['scene']+'_stable.json')).resolve(), 'support inventory private scene differs')
        checked(entry['scene_file'])
        record = checked(entry['inventory_record'])
        measured = checked(entry['measurement'])
        require(all(record[k] == entry[k] for k in ('scene','name','category','model','scale')),
                'support inventory model/scale identity differs')
        require(measured['native_inventory_record'] == entry['inventory_record'] and
                measured['source_scene'] == entry['scene_file'] and measured['object_name'] == entry['name'] and
                measured['actual_scale'] == entry['scale'] and measured['category'] == entry['category'] and
                measured['model'] == entry['model'], 'support inventory measured provenance differs')
        top = record['top']; hull = top['hull_xy']
        require(top['resolution'] == .02 and isinstance(hull, list) and len(hull) >= 3 and
                all(isinstance(row, list) and len(row) == 2 and
                    all(type(v) in (int,float) and math.isfinite(v) for v in row) for row in hull),
                'support inventory has no finite native outline')
    return {'environment':{key:environment[key] for key in KEYS}, 'artifacts':artifacts,
            'scope':'Unchanged measured inventory/private-scene input only; native rectangularity and own group physics checks remain required.'}
