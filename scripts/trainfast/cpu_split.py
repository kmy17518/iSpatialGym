"""COLLECTOR_PIN CPU layout (trainfast, 2026-11-02): this job's CPUs (sched_getaffinity, ascending) split into K disjoint sets of <c>
CPUs for the collectors, taken from the start of the list as in the E16 pin test (on the svl nodes a job's ascending list is its
physical cores, then their hyperthread siblings), and the rest for the sampler (at least <c>). Prints the sets as comma lists, the
sampler's first, one per line; nothing when the job has fewer than (K + 1) * <c> CPUs (no pinning then).
    cpu_split.py <K> <c>"""
import os
import sys


def main(k, c):
    cpus = sorted(os.sched_getaffinity(0))
    if k < 1 or c < 1 or len(cpus) < (k + 1) * c:
        return []
    return [cpus[k * c:]] + [cpus[i * c:(i + 1) * c] for i in range(k)]


if __name__ == "__main__":
    for s in main(int(sys.argv[1]), int(sys.argv[2])):
        print(",".join(map(str, s)))
