"""GPU use of the fast-flow jobs (gpu.log: one nvidia-smi sample per minute: memory used, utilisation, per-process memory): per tag,
the median over jobs of the job's mean utilisation, the share of samples at >= 50 %, and the peak memory.
    gpu_use.py <tag> [<tag> ...]"""
import glob, re, statistics as st, sys

F = "/vision/u/minyeong/projects/isg_home/trainfast"
RX = re.compile(r"^\d+ (\d+)MiB,(\d+)%")


def main(tags):
    for tag in tags:
        mean_u, hi, peak = [], [], []
        for p in glob.glob(f"{F}/out/{tag}/*/g*/gpu.log"):
            s = [tuple(map(int, m.groups())) for m in (RX.match(l) for l in open(p)) if m]
            s = [x for x in s if x[0] > 1000]  # samples with the job's processes up
            if len(s) < 5:
                continue
            mean_u.append(st.mean(u for _, u in s))
            hi.append(sum(u >= 50 for _, u in s) / len(s))
            peak.append(max(m for m, _ in s))
        if mean_u:
            print(f"{tag:<10} jobs {len(mean_u):3d}  mean GPU util: median {st.median(mean_u):4.1f} %, p90 {sorted(mean_u)[(9*len(mean_u))//10]:4.1f} %  "
                  f"samples >= 50 %: median {100*st.median(hi):4.1f} %  peak memory: median {st.median(peak)/1024:4.1f} GB, max {max(peak)/1024:4.1f} GB")


if __name__ == "__main__":
    main(sys.argv[1:] or ["pilot2", "pilot3", "pilot4", "crop2", "crop4", "refine_on", "refine_off", "refine_nearest"])
