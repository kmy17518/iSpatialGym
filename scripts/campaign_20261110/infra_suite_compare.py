#!/usr/bin/env python3
"""Compare complete eight-shard suites; a wrapper's zero exit is insufficient."""
import argparse
import json
import pathlib
import re


def evidence(root):
    summaries = list(root.glob('summary_*.txt'))
    failures = list(root.glob('fails_*.txt'))
    if len(summaries) != 1 or len(failures) != 1:
        raise ValueError('Missing or ambiguous suite evidence: ' + str(root))
    rows = summaries[0].read_text().splitlines()
    shards = [row for row in rows if re.match(r'^shard [0-7]: ', row)]
    if len(shards) != 8 or {row.split(':')[0] for row in shards} != {'shard ' + str(i) for i in range(8)}:
        raise ValueError('Incomplete suite shards: ' + str(root))
    passed = sum(int(match.group(1)) for row in shards if (match := re.search(r'\b(\d+) passed\b', row)))
    if passed < 3000:
        raise ValueError('Suite collected too few passing tests: ' + str(passed))
    ids = {line.split(' - ', 1)[0] for line in failures[0].read_text().splitlines() if line.startswith(('FAILED ', 'ERROR '))}
    return {'passing_tests': passed, 'failure_ids': sorted(ids), 'summary': str(summaries[0])}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', type=pathlib.Path, required=True)
    parser.add_argument('--candidate', type=pathlib.Path, required=True)
    parser.add_argument('--output', type=pathlib.Path, required=True)
    args = parser.parse_args()
    baseline, candidate = evidence(args.baseline), evidence(args.candidate)
    added = sorted(set(candidate['failure_ids']) - set(baseline['failure_ids']))
    result = {'baseline': baseline, 'candidate': candidate, 'added': added,
              'resolved': sorted(set(baseline['failure_ids']) - set(candidate['failure_ids'])),
              'regression_gate_passed': not added and candidate['passing_tests'] >= baseline['passing_tests']}
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: result[key] for key in ('added', 'resolved', 'regression_gate_passed')}))
    raise SystemExit(0 if result['regression_gate_passed'] else 1)


if __name__ == '__main__':
    main()
