"""COLLECTOR_PIN CPU layout (trainfast, 2026-11-02): this job's CPUs (sched_getaffinity, ascending) split into K disjoint sets of <c>
CPUs for the collectors, taken from the start of the list as in the E16 pin test (on the svl nodes a job's ascending list is its
physical cores, then their hyperthread siblings), and the rest for the sampler (at least <c>). Prints the sets as comma lists, the
sampler's first, one per line; nothing when the job has fewer than (K + 1) * <c> CPUs (no pinning then).
`halves` (job_pair.sh): this job's physical cores (sysfs thread siblings) split in two, each half with its cores' CPUs in this job, so
the two pair members pin inside CPUs of their own; nothing when the job holds fewer than two cores.
    cpu_split.py <K> <c>
    cpu_split.py halves"""
import os
import sys


def main(k, c):
    cpus = sorted(os.sched_getaffinity(0))
    if k < 1 or c < 1 or len(cpus) < (k + 1) * c:
        return []
    return [cpus[k * c:]] + [cpus[i * c:(i + 1) * c] for i in range(k)]


def halves():
    cores = {}
    for c in sorted(os.sched_getaffinity(0)):
        try:
            sib = open(f"/sys/devices/system/cpu/cpu{c}/topology/thread_siblings_list").read().strip()
        except OSError:
            sib = str(c)
        cores.setdefault(sib, []).append(c)
    cores = sorted(cores.values())
    n = len(cores) // 2
    if n < 1:
        return []
    return [sorted(c for g in cores[:n] for c in g), sorted(c for g in cores[n:] for c in g)]


if __name__ == "__main__":
    for s in halves() if sys.argv[1] == "halves" else main(int(sys.argv[1]), int(sys.argv[2])):
        print(",".join(map(str, s)))
