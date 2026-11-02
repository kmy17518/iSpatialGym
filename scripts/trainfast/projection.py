"""Production-mix speed-up and the 10x projection: per alias key, the fast code's whole-job successes per GPU-hour (pooled over the
given tags) against train85's on the same key, weighted by train85's output (the GPU-hours each key would need for 10x train85's
successes of it). Keys with no fast success are listed apart (not runnable / not measured).
    projection.py <label> <tag>[,<tag>...] [<label> <tags> ...]"""
import collections, sys

import speed_compare as S


def rates(tags):
    by = collections.defaultdict(list)
    for tag in tags:
        for r in S.job_rows(f"{S.F}/out/{tag}"):
            by[r[0]].append(r)
    return {k: S.pooled(v) for k, v in by.items()}


def main(argv):
    base = collections.defaultdict(list)
    for r in S.job_rows(f"{S.H}/train85/out"):
        base[r[0]].append(r)
    b = {k: S.pooled(v) for k, v in base.items()}
    for label, tags in zip(argv[0::2], argv[1::2]):
        f = rates(tags.split(","))
        keys = sorted(k for k in f if k in b and b[k][0] > 0)
        ok = [k for k in keys if f[k][0] > 0]
        zero = [k for k in keys if f[k][0] == 0]
        h85 = sum(b[k][1] for k in ok)  # GPU-h train85 spent on these keys
        s85 = sum(b[k][0] for k in ok)
        hfast = sum(b[k][0] / f[k][2] for k in ok)  # GPU-h the fast code needs for the same successes per key
        print(f"{label}: {len(keys)} keys measured, {len(ok)} with fast successes; train85 on them {s85} successes / {h85:.0f} GPU-h "
              f"= {s85 / h85:.2f} per GPU-h; the fast code would need {hfast:.0f} GPU-h for the same per-key successes -> x{h85 / hfast:.1f} "
              f"(production-mix rate {s85 / hfast:.1f} per GPU-h)")
        worst = sorted(ok, key=lambda k: b[k][0] / f[k][2], reverse=True)[:5]
        print("   GPU-h hogs (share of the fast GPU-h): " + ", ".join(f"{k[-28:]} {100 * b[k][0] / f[k][2] / hfast:.0f} % ({f[k][2]:.1f}/GPU-h)" for k in worst))
        if zero:
            print(f"   no fast success: {', '.join(k[-30:] for k in zero)} (train85: {sum(b[k][0] for k in zero)} successes / {sum(b[k][1] for k in zero):.0f} GPU-h)")


if __name__ == "__main__":
    main(sys.argv[1:])
