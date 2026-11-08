import glob, re, collections
d = glob.glob("/vision/u/minyeong/projects/isg_home/oracle_fix/out/c3_pdc_sh/*touching_04/g*")[0]
rej, kit = {}, {}
for l in open(d + "/sample.log", errors="replace"):
    m = re.search(r"instance (\d+) attempt (\d+) rejected in (\w+) after ([\d.]+)s: \[(\w+)\] (.*)$", l)
    if m:
        rej[(m.group(1), m.group(2))] = (m.group(5), float(m.group(4)), m.group(6)); continue
    m = re.search(r"instance (\d+) attempt (\d+) kit: (.*)$", l)
    if m:
        kit[(m.group(1), m.group(2))] = dict((a, int(b)) for a, b in re.findall(r"(place_depth_cap[\w.]*) \((\d+)\)", m.group(3)))
cls = collections.Counter(); tim = collections.defaultdict(float); rest = collections.Counter()
for k, (r, t, rest_s) in rej.items():
    kk = kit.get(k, {})
    n = kk.get("place_depth_cap.candidates", 0); tried = min(6, n); sw = kk.get("place_depth_cap.shadow.tried", 0)
    if r == "target_placement":
        c = "all tried over the cap" if (tried and sw == tried) else ("some over the cap" if sw else "none over the cap")
    else:
        c = "drafted (%s)" % ("from an over-cap candidate" if kk.get("place_depth_cap.shadow.drafted") else "under the cap")
    cls[(r if r == "target_placement" else "other:" + r, c)] += 1; tim[(r if r == "target_placement" else "other:" + r, c)] += t
    if r == "target_placement" and c != "all tried over the cap":
        for ci, cr, cw in re.findall(r"candidate (\d+): (?:\[(\w+)\] )?(.*?)(?=; candidate \d+: |$)", rest_s):
            w = re.sub(r"-?\d+\.\d+|\b\d+\b", "#", cw)[:110]
            rest[(c, cr, w)] += 1
for k, v in sorted(cls.items()):
    print(f"{v:4d} att {tim[k]:7.0f} s  {k}")
print("-- candidates shown in the target_placement attempts NOT fully over the cap:")
for k, v in rest.most_common(12):
    print(f"{v:4d} {k}")
tot_tried = tot_over = 0
for k, (r, t, s_) in rej.items():
    kk = kit.get(k, {})
    if r == "target_placement":
        tot_tried += min(6, kk.get("place_depth_cap.candidates", 0)); tot_over += kk.get("place_depth_cap.shadow.tried", 0)
print("target_placement attempts: candidates tried", tot_tried, "over the cap", tot_over, "under", tot_tried - tot_over)
