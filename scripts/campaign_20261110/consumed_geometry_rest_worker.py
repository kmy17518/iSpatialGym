"""Export one exact object's geometry and measure its rest; no grasp trials."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def measure(prepared, directory, schema, validator):
    job = prepared['job']; model = directory / 'glib/models' / job['id']
    model.mkdir(parents=True, exist_ok=False)
    # run_model is the existing native rest writer. An empty representative set
    # and max_trials=0 prevent planner/trial work and yield no grasp credit.
    schema.save_json(str(model / 'candidates.json'), {'schema': schema.CANDIDATES,
        'candidates': [], 'representatives': [], 'families': {}, 'arm': 'right',
        'pipeline': 'consumed_geometry_and_rest_only', 'constraint': 'tabletop'})
    world = validator.TabletopWorld(validator.ValidateConfig(max_trials=0, max_valid=0))
    result = validator.run_model(world, job, str(model))
    library = json.loads((model / 'library.json').read_text())
    assert result['n_reps'] == result['n_valid'] == 0 and library['grasps'] == []
    assert library['category'] == job['category'] and library['model'] == job['model']
    assert library['scale'] == job['scale']
    stable = [k for k, r in library['rests'].items() if r['stable'] and r['contact']]
    return {'native_result': result, 'stable_rest_ids': stable, 'grasp_credit': False,
            'geometry_files': [{'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
                for p in sorted(model.iterdir()) if p.is_file()]}


if __name__ == '__main__':
    directory = Path(sys.argv[1]); path = directory / 'prepared.json'; prepared = json.loads(path.read_text())
    for ref in prepared['input_refs']:
        assert hashlib.sha256(Path(ref['path']).read_bytes()).hexdigest() == ref['sha256'], ref['path']
    from ispatialgym.grasping import schema, validate_tabletop
    started = time.time(); result = measure(prepared, directory, schema, validate_tabletop)
    result.update(started_at=started, finished_at=time.time(), native_metadata_done=True,
                  prepared_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  physical_dedup_key=prepared['physical_dedup_key'], job_id=os.environ.get('SLURM_JOB_ID'),
                  benchmark_credit=0, production_authorized=False)
    with (directory / 'native_result.json').open('x') as stream:
        stream.write(json.dumps(result, indent=2) + '\n'); stream.flush(); os.fsync(stream.fileno())
    print('ISG_CONSUMED_GEOMETRY_REST_DONE', flush=True)
    # All requested native work and its durable result are complete. Avoid
    # the independently observed Kit interpreter-finalization segfault.
    os._exit(0)
