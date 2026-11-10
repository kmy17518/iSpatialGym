"""Admit completed positive floating libraries into a new immutable data release."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def release(root, previous, destination, sources):
    from ispatialgym.grasping import schema as S
    from ispatialgym.sampling import grasp_library as G
    root, previous, destination = (Path(p).resolve() for p in (root, previous, destination))
    assert previous.is_relative_to(root) and destination.is_relative_to(root) and not destination.exists()
    before = json.loads((previous / "release_manifest.json").read_text())
    assert all(digest(previous / name) == sha for name, sha in before["files"].items()), "prior release changed"
    records, skipped = [], []
    for output, manifest in sources:
        output, manifest = Path(output).resolve(), Path(manifest).resolve()
        assert output.is_relative_to(root) or output.is_relative_to(Path("/vision/u/shiyuc/minyeong/campaign_20261110"))
        assert manifest.is_relative_to(root)
        for index, job in enumerate(json.loads(manifest.read_text())):
            assert job["kind"] == job["constraint"] == "floating" and job["ready"]
            log = output / "logs" / f"floating_{index}.log"
            if not log.exists() or not os.access(log, os.R_OK):
                skipped.append({"id":job["id"], "reason":"validation log not shared yet"}); continue
            contents = log.read_text()
            if "GB_VALIDATE_DONE" not in contents or "GB_JOB_DONE" not in contents:
                skipped.append({"id":job["id"], "reason":"validation not complete"}); continue
            try:
                lib = S.load_model(job["category"], job["model"], job["scale"], str(output / "glib"), constraint="floating")
            except PermissionError:
                skipped.append({"id":job["id"], "reason":"completed library not shared yet"}); continue
            if lib is None or not lib.grasps():
                skipped.append({"id":job["id"], "reason":"no positive native grasps"}); continue
            source = Path(lib.path).parent
            if not all(os.access(p, os.R_OK) for p in source.rglob("*") if p.is_file()):
                skipped.append({"id":job["id"], "reason":"completed geometry not shared yet"}); continue
            assert source.name == job["id"] and lib.floating and not lib.rests
            assert lib.data["dataset"] == job["dataset"] and lib.data["floating_mode"] == job["floating_mode"]
            assert lib.data["legacy_key"] == job["legacy_key"], "routing alias differs"
            records.append({"id":job["id"], "source":str(source), "library_sha256":digest(lib.path),
                "manifest":str(manifest), "manifest_sha256":digest(manifest), "validation_log":str(log),
                "validation_log_sha256":digest(log), "alias":job["legacy_key"], "scale":job["scale"],
                "positive_grasps":len(lib.grasps()), "asset_sha256":job["asset_sha256"]})
    assert records and len({r["id"] for r in records}) == len(records), "no positive or duplicate entries"
    shutil.copytree(previous, destination, symlinks=False)
    for p in [destination,*destination.rglob("*")]: p.chmod(p.stat().st_mode | 0o200)
    block = destination / "release_block.json"; block.write_text('{"reason":"reader audit pending"}\n')
    for row in records:
        target = destination / "glib/models" / row["id"]
        assert not target.exists(), "fresh entry already present"
        shutil.copytree(row["source"], target)
        assert digest(target / "library.json") == row["library_sha256"]
    os.environ["ISPATIALGYM_GRASP_LIBRARY_V2"] = str(destination / "glib")
    G._v2_index.cache_clear()
    for row in records:
        lib = G.v2_entry(row["alias"])
        assert lib is not None and Path(lib.path).parent.name == row["id"] and lib.floating
        S.check_compatible(lib, scale=row["scale"])
        assert len(lib.grasps()) == row["positive_grasps"]
        assert len(G.grasps_from_v2(lib,row["alias"])) > 0
    audit = {"records":records,"skipped":skipped,"new_positive_floating_libraries":len(records),
             "native_reader_errors":0,"task_geometry_demo_replay_still_required":True}
    (destination / "floating_content_audit.json").write_text(json.dumps(audit,indent=2)+"\n")
    block.unlink()
    before.update(created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),previous_release=str(previous),
                  new_floating_libraries=before.get("new_floating_libraries",0)+len(records))
    before["files"] = {str(p.relative_to(destination)):digest(p) for p in sorted(destination.rglob("*"))
                       if p.is_file() and p.name != "release_manifest.json"}
    (destination / "release_manifest.json").write_text(json.dumps(before,indent=2)+"\n")
    (root / "validation" / (destination.name+"_reader.txt")).write_text("Native floating alias/scale/positive-grasp reader audit\nERRORS 0\n")
    for p in [destination,*destination.rglob("*")]:
        p.chmod((p.stat().st_mode & ~0o222) | (0o055 if p.is_dir() else 0o044))
    print(json.dumps({"release":str(destination),"positive_floating_libraries":len(records),"skipped":len(skipped)}))


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("root","previous","destination"):p.add_argument("--"+name,required=True)
    p.add_argument("--source",nargs=2,action="append",required=True,metavar=("OUTPUT","MANIFEST"))
    a=p.parse_args();release(a.root,a.previous,a.destination,a.source)
