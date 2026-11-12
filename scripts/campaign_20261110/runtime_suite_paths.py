"""Exact registered attempt locations shared by observational suite owners."""
from pathlib import Path
import json
from .contracts import require, file_digest


def base_for_row(path):
    path=Path(path).resolve()
    if path.parent.name=='rows':return path.parent.parent
    require(path.name=='row.json' and path.parents[2].name=='attempts', 'unknown suite attempt locator')
    return path.parents[3]


def class_rows(root,base,initial):
    root=Path(root).resolve();base=Path(base).resolve();initial=Path(initial).resolve()
    require(initial.parent==base/'rows','initial row outside suite')
    pointer_path=root/'runtime_validation/current.json'
    if not pointer_path.exists():return [initial]
    pointer=json.loads(pointer_path.read_text())
    if not pointer.get('attempt_registry'):return [initial]
    from .runtime_suite import active, registered_rows
    pointer,_,_,_,current=active(root)
    require(current==base,'retry registry belongs to another suite')
    refs=registered_rows(root,pointer,base,initial.stem)
    return [Path(ref['path']).resolve() for ref in refs]


def all_rows(root,base):
    return [p for initial in sorted((Path(base)/'rows').glob('*.json')) for p in class_rows(root,base,initial)]


def paths(base,row_path):
    base=Path(base).resolve();p=Path(row_path).resolve()
    require(base_for_row(p)==base,'attempt belongs to another suite')
    if p.parent==base/'rows':
        return {'allocation':base/'allocations'/p.name,'members':base/'member_replays'/p.name,
                'registration':base/'measurement_registrations'/p.name,'summary':base/'allocation_class_observations'/p.name}
    row=json.loads(p.read_text());initial=base/'rows'/(row['class_id']+'.json')
    require(p in class_rows(base.parents[1],base,initial),'unregistered suite attempt')
    return {'allocation':p.parent/'allocation.json','members':p.parent/'member_replays.json',
            'registration':p.parent/'measurement_registration.json','summary':p.parent/'allocation_observation.json'}
