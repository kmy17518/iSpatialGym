#!/usr/bin/env python3
"""Copy completed owned native templates, with a narrowly audited customs locator."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'runs', 'jobs', 'out'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--customs-copy-audit', type=Path)
    a = p.parse_args()
    assert a.out.resolve().is_relative_to(a.root.resolve()) and not a.out.exists()
    jobs = json.loads(a.jobs.read_text())['jobs']
    records = []
    for meta_path in sorted(a.runs.rglob('template_meta.json')):
        original = json.loads(meta_path.read_text())
        if original.get('n_written', 0) < 1:
            continue
        matches = [j for j in jobs if j['alias'] == original['variant'] and j['scene'] == original['scene']]
        assert len(matches) == 1, 'native output must match one exact prepared job'
        job = matches[0]
        source_jobs = json.loads(Path(job['source_manifest']).read_text())['jobs']
        match = [j for j in source_jobs if j['template_fingerprint'] == job['template_fingerprint']]
        assert len(match) == 1
        argv = match[0]['sample_argv']
        customs = Path(argv[argv.index('--customs') + 1]).resolve()
        old_customs = Path(original['customs']).resolve()
        assert customs.is_relative_to(a.root.resolve())
        export = None
        if a.customs_copy_audit:
            audit = json.loads(a.customs_copy_audit.read_text())
            assert audit['schema'] == 'isg-private-source-copy-audit-v1'
            matches = [row for row in audit['records'] if row['source'] == str(old_customs)]
            assert len(matches) == 1
            record = matches[0]
            copied = (a.customs_copy_audit.parent / record['relative_path']).resolve()
            assert copied.is_relative_to(a.customs_copy_audit.parent.resolve())
            assert sha(copied) == sha(customs) == record['sha256']
            export = dict(path=str(a.customs_copy_audit), sha256=sha(a.customs_copy_audit), copied_customs=str(copied))
        else:
            assert sha(customs) == sha(old_customs)
        source = meta_path.parent
        assert sha(source / 'template.json') == original['template_sha256']
        assert list((source / 'instances').glob('*-tro_state.json')), 'fresh native TRO missing'
        dest = a.out / 'templates' / job['class_id']
        assert not dest.exists()
        shutil.copytree(source, dest)
        copied = []
        for old in sorted(source.rglob('*')):
            if not old.is_file():
                continue
            relative = old.relative_to(source)
            new = dest / relative
            assert sha(old) == sha(new), 'native copy bytes differ'
            copied.append(dict(relative_path=str(relative), sha256=sha(old)))
        definitions = [path for path in (dest / 'activity_definitions').rglob('*.bddl') if sha(path) == original['bddl_sha256']]
        assert len(definitions) == 1, 'exact native BDDL copy absent or ambiguous'
        activity_definition = str(definitions[0].relative_to(dest))
        metadata = {**original, 'customs': str(customs), 'activity_definition': activity_definition}
        (dest / 'template_meta.json').write_text(json.dumps(metadata, indent=2, sort_keys=True) + '\n')
        proof = dict(schema='isg-native-source-relocation-v1', original_metadata=str(meta_path),
                     original_metadata_sha256=sha(meta_path), metadata_sha256=sha(dest / 'template_meta.json'),
                     original_customs=str(old_customs), customs=str(customs), customs_sha256=sha(customs),
                     private_source_export=export,
                     changed_fields=['customs', 'activity_definition'], activity_definition=activity_definition,
                     copied_files=copied,
                     source_jobs=str(a.jobs), source_jobs_sha256=sha(a.jobs),
                     native_template_sha256=original['template_sha256'], native_bddl_sha256=original['bddl_sha256'])
        (dest / 'native_source_relocation.json').write_text(json.dumps(proof, indent=2, sort_keys=True) + '\n')
        records.append(dict(class_id=job['class_id'], template=str(dest), proof=str(dest / 'native_source_relocation.json')))
    assert records, 'no completed native sources'
    output = a.out / 'relocation_audit.json'
    output.write_text(json.dumps(dict(records=records, source_runs=str(a.runs), original_sources_unchanged=True), indent=2) + '\n')
    print(json.dumps(dict(copied=len(records), audit=str(output), sha256=sha(output))))


if __name__ == '__main__':
    main()
