"""Create an immutable alias-only revision; never mutate claims or old inputs."""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import time


def main(args):
    source, out, runtime = map(Path, (args.source, args.output, args.runtime))
    sys.path[:0] = [str(runtime / "Behavior-1K/bddl3"), str(runtime)]
    spec = importlib.util.spec_from_file_location("floating_author", args.author)
    author = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(author)
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    ref = lambda p: {"path": str(p), "sha256": sha(p)}
    read = lambda p: json.loads(Path(p).read_text())

    def write(p, value):
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(value, indent=2) + "\n")

    old = source.parent
    assert not out.exists()
    out.mkdir(parents=True)
    original = read(source)

    def relocate(value):
        if isinstance(value, dict):
            return {k: relocate(v) for k, v in value.items()}
        if isinstance(value, list):
            return [relocate(v) for v in value]
        if isinstance(value, str) and value.startswith(str(old) + "/"):
            return str(out / Path(value).relative_to(old))
        return value

    releases, jobs = {}, []
    for original_job in original["jobs"]:
        job = relocate(deepcopy(original_job))
        source_release = Path(original_job["dependency_release"])
        release = Path(job["dependency_release"])
        if str(source_release) not in releases:
            shutil.copytree(source_release, release)
            libraries = list(release.glob("glib/models/*/library.json"))
            assert len(libraries) == 1
            library = libraries[0]
            before = read(library)
            routed = deepcopy(before)
            routed["legacy_key"] = job["target"]["category"] + "__" + job["target"]["model"] + "__f"
            assert {k: v for k, v in before.items() if k != "legacy_key"} == {k: v for k, v in routed.items() if k != "legacy_key"}
            write(library, routed)
            proof = read(source_release / "release.json")
            proof.update(routed=ref(library), route_revision="native_floating_suffix_f", supersedes=ref(source_release / "release.json"), consumer_lookup=author.verify_floating_consumer(release, job["target"], job["arm"]))
            proof["files"] = [ref(p) for p in release.rglob("*") if p.is_file() and p.name != "release.json"]
            write(release / "release.json", proof)
            releases[str(source_release)] = proof["consumer_lookup"]
        for old_ref in original_job["input_refs"]:
            source_path = Path(old_ref["path"])
            assert sha(source_path) == old_ref["sha256"]
            if source_path.is_relative_to(old) and not source_path.is_relative_to(source_release):
                dest = out / source_path.relative_to(old)
                dest.parent.mkdir(parents=True, exist_ok=True)
                if not dest.exists():
                    shutil.copy2(source_path, dest)
                assert sha(dest) == old_ref["sha256"]
        job["input_refs"] = [ref(r["path"]) for r in job["input_refs"]]
        job["supersedes_request"] = original_job["request"]
        job["route_revision"] = "native_floating_suffix_f"
        job["consumer_lookup"] = releases[str(source_release)]
        request = Path(job.pop("request")["path"])
        write(request, job)
        job["request"] = ref(request)
        jobs.append(job)
    report = {**original, "observed_at": time.time(), "supersedes": ref(source), "route_revision": "native_floating_suffix_f", "jobs": jobs,
              "revision_scope": "Only native floating library alias metadata corrected; exact physical query identities, task constraints, geometry and native runtime retained. Existing claims require dispatch-owner reconciliation before replacement.",
              "consumer_lookup_verified_releases": len(releases), "repair_helper": ref(Path(__file__)), "author_helper": ref(args.author)}
    write(out / "requests.json", report)
    print(json.dumps({"manifest": ref(out / "requests.json"), "jobs": len(jobs), "verified_releases": len(releases), "by_arm": dict(Counter(j["arm"] for j in jobs))}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "output", "runtime", "author"):
        p.add_argument("--" + name, required=True)
    main(p.parse_args())
