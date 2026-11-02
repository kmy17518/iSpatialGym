"""Packing test per job (owner item 2): host, start, sampler attempts / written / accept share / s per attempt, successes, sample wall,
job wall and drain (job wall - sample wall), the COLLECTOR_PREBUILD source tag and mismatch lines, the collectors' mean demo total_s.
    pack_detail.py [tag ...]"""
import glob, json, os, re, sys
F = "/vision/u/minyeong/projects/isg_home/trainfast"
TAGS = ("pk_k2_titan", "pk_k3_titan", "pk_pair1_titan", "pk_k2_l40s", "pk_k3_l40s", "pk_pair1_l40s", "pk_pair2_l40s")


def lines(p):
    try:
        return open(p).read().splitlines()
    except OSError:
        return []


def main(tags):
    print(f"{'arm':<15} {'key':<8} {'job':<16} {'host':<9} {'start':<6} {'att':>4} {'wr':>3} {'acc%':>5} {'s/att':>5} {'succ':>4} "
          f"{'samp s':>6} {'job s':>6} {'drain':>5} {'demo s':>6}  prebuild src / mismatches")
    for tag in tags:
        for job in sorted(glob.glob(f"{F}/out/{tag}/*/g*")):
            if job.endswith(".failed_submit"):
                continue
            L = lines(f"{job}/job.txt")
            key = os.path.basename(os.path.dirname(job)).replace("inter_object_relation_satisfaction-ontop_small_scale_01", "ontop01") \
                .replace("disembedding_03", "de03").replace("mental_rotation-mug", "mrmug")
            g = lambda rx: next((m for l in L for m in [re.search(rx, l)] if m), None)
            s = g(r"^ISGF_SAMPLE (\S+) .*host=(\S+) .* start=\S*T(\d\d:\d\d)")
            fc = g(r"^FAST_CHECK attempts=(\d+) written=(\d+) attempts_s=(\d+) demos=(\d+) successes=(\d+) sample_wall=(\d+)")
            e = g(r"^ISGF_END \S+ wall_s=(\d+)")
            pb = g(r"^PREBUILD source=\S+/out/([^/]+)/")
            mism = sum("mismatch" in l for p in glob.glob(f"{job}/collect_*.log") for l in lines(p))
            tot = []
            for p in glob.glob(f"{job}/demo/*/meta.json"):
                try:
                    tot.append(float((json.load(open(p)).get("timings") or {}).get("total_s") or 0))
                except Exception:  # noqa: BLE001
                    pass
            if not (s and fc and e):
                print(f"{tag:<15} {key:<8} {os.path.basename(job)} incomplete")
                continue
            att, wr, att_s, demos, succ, sw = map(int, fc.groups())
            jw = int(e.group(1))
            print(f"{tag:<15} {key:<8} {s.group(1):<16} {s.group(2):<9} {s.group(3):<6} {att:4d} {wr:3d} {100 * wr / att if att else 0:5.0f} "
                  f"{att_s / att if att else 0:5.0f} {succ:4d} {sw:6d} {jw:6d} {jw - sw:5d} {sum(tot) / len(tot) if tot else 0:6.0f}  "
                  f"{pb.group(1) if pb else '-'} / {mism}")


if __name__ == "__main__":
    main(sys.argv[1:] or TAGS)
