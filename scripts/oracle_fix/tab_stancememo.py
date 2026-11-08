#!/usr/bin/env python3
"""i7-stancememo: per (tag, alias) of the 5x.1 / 5x.3 A/Bs, from each job's sample.log (CPU; run on the cluster):
attempts / written, rejections by reason (incl. no_floor_cell / time_budget / time_cap), sampling seconds, seconds in rejected search
attempts, the memo / redraw / time_cap kit counters summed over the attempts, the B / C search failures, and - for arms without the
redraw - what the attempt after an instance's 2nd B / C search failure did (the offline twin of redraw.third_attempt_*).

    python3 tab_stancememo.py <out root> <tag> [<tag> ...]
Use $OFH/jobs/v2build/tab_i.py -a <tag> for written / ok / GPU-h per success."""
import collections
import glob
import os
import re
import sys

REJ = re.compile(r"instance (\d+) attempt (\d+) rejected in (\w+) after ([\d.]+)s: \[(\w+)\]")
WR = re.compile(r"instance (\d+) written \(attempt (\d+), ([\d.]+)s\)")
KIT = re.compile(r"instance (\d+) attempt (\d+) kit: (.*)$")
CNT = re.compile(r"((?:memo|redraw|time_cap)\.[\w.]+) \((\d+)\)")
SEARCH_FAIL = ("target_placement", "time_budget", "time_cap", "grasp_unreachable", "no_placement_stance")


def one(log):
    att, kit = [], collections.Counter()
    for line in open(log, errors="replace"):
        m = REJ.search(line)
        if m:
            att.append((int(m.group(1)), int(m.group(2)), m.group(3), float(m.group(4)), m.group(5)))
            continue
        m = WR.search(line)
        if m:
            att.append((int(m.group(1)), int(m.group(2)), "written", float(m.group(3)), "written"))
            continue
        m = KIT.search(line)
        if m:
            for k, n in CNT.findall(m.group(3)):
                kit[k] += int(n)
    return att, kit


def main(root, tags):
    for tag in tags:
        for log in sorted(glob.glob(os.path.join(root, "out", tag, "*", "g*", "sample.log"))):
            alias = log.split("/")[-3].replace("inter_object_relation_satisfaction-", "iors-").replace("configuration_matching-", "cm-")
            att, kit = one(log)
            reasons = collections.Counter(a[4] for a in att)
            written = reasons.pop("written", 0)
            s_all = sum(a[3] for a in att)
            s_rej_search = sum(a[3] for a in att if a[2] == "sample")
            search_fails = sum(1 for a in att if a[2] == "sample" and a[4] in SEARCH_FAIL)
            # offline third attempt: per instance, the attempt after the 2nd search failure (no redraw in the arm: one start per instance)
            third = collections.Counter()
            by_inst = collections.defaultdict(list)
            for a in att:
                by_inst[a[0]].append(a)
            for inst, rows in by_inst.items():
                n = 0
                rows.sort(key=lambda a: a[1])
                for i, a in enumerate(rows):
                    if a[2] == "sample" and a[4] in SEARCH_FAIL:
                        n += 1
                        if n == 2 and i + 1 < len(rows):
                            nx = rows[i + 1]
                            third["written" if nx[4] == "written" else "search_failed" if nx[2] == "sample" and nx[4] in SEARCH_FAIL
                                  else "start_failed" if nx[2] == "sample" else "post_failed"] += 1
            print(f"{tag}\t{alias}\tatt {len(att)}\twritten {written}\tsample_s {s_all:.0f}\trejected_in_search_s {s_rej_search:.0f}\t"
                  f"search_fails {search_fails}\trej {dict(reasons.most_common())}\tthird_after_2 {dict(third)}\tkit {dict(sorted(kit.items()))}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
