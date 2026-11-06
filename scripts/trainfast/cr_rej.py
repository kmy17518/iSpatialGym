import glob, re, collections, ast
F = "/vision/u/minyeong/projects/isg_home/trainfast"
RX = re.compile(r"attempt (\d+) rejected in (\S+) after ([\d.]+)s: \[([^\]]+)\] ?(.*)$")
stage = collections.Counter(); reason = collections.Counter(); secs = collections.defaultdict(float); fails = collections.Counter(); rendered = []
wr = []; stage_s = collections.defaultdict(list)
for f in glob.glob(f"{F}/out/pilot/camera_relocalization-obstructed-toy_car_1/g*/sample.log"):
    for line in open(f, errors="replace"):
        m = RX.search(line)
        if m:
            stage[m.group(2)] += 1; reason[m.group(4)] += 1; secs[m.group(4)] += float(m.group(3))
            mm = re.search(r"\((\d+) rendered; (\{[^}]*\})\)", m.group(5))
            if mm:
                rendered.append(int(mm.group(1)))
                for k, v in ast.literal_eval(mm.group(2)).items(): fails[k] += v
            continue
        m = re.search(r" written \(attempt (\d+), ([\d.]+)s\)", line)
        if m: wr.append(float(m.group(2)))
        m = re.search(r"stages: (.*) total=", line)
        if m:
            for part in m.group(1).split():
                k, v = part.split("="); stage_s[k].append(float(v.rstrip("s")))
n = sum(stage.values())
print(f"rejected attempts {n}, written {len(wr)} (written attempt mean {sum(wr)/len(wr):.0f} s)")
print("by stage:", dict(stage.most_common()))
print("by reason (count, total s, mean s):", {k: (v, round(secs[k]), round(secs[k]/v)) for k, v in reason.most_common()})
print("goal_sampling candidate fails summed:", dict(fails.most_common()), "renders per rejected attempt: mean", round(sum(rendered)/max(1,len(rendered)),1), "max", max(rendered or [0]))
print("per-attempt stage means:", {k: round(sum(v)/len(v),1) for k, v in stage_s.items()})
