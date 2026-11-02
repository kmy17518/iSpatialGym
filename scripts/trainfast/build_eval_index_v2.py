"""trainfast 2026-11-01 (REPEAT_MEASURED): train85's per-alias eval indexes rebuilt with the measured entries (each record's eval TRO
re-read with the train-fast instance_separation; a record whose TRO cannot be read is kept as it was - the old rule then applies to it).
    python build_eval_index_v2.py <train85 eval_index/per_alias> <out dir>"""
import glob, json, os, sys
from ispatialgym.sampling import instance_separation as SEP

src, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
tot = kept_old = with_measured = 0
for p in sorted(glob.glob(os.path.join(src, "*.json"))):
    d = json.load(open(p))
    recs = []
    for r in d.get("instances") or []:
        tot += 1
        try:
            new = SEP.read_instance(r["path"])
            new.split = r.get("split") or new.split
            recs.append(new)
            with_measured += bool(new.measured_goal)
        except (OSError, ValueError, KeyError, TypeError) as e:
            kept_old += 1
            recs.append(SEP.InstanceRecord.from_json(r))
            print(f"kept old record {r.get('path')}: {e!r}")
    SEP.write_index(os.path.join(out, os.path.basename(p)), recs)
print(f"EVAL_INDEX_V2 {len(glob.glob(os.path.join(src, '*.json')))} aliases, {tot} records, {with_measured} with measured entries, {kept_old} kept as before")
