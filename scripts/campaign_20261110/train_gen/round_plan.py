#!/usr/bin/env python3
"""Plan one train round: per-template targets -> worker items -> ~5 h chain jobs.

Inputs (JSON):
  --subtemplates  rows {class_id, configuration_id, template_key, status, authorable(bool)} (gate `valid` + authoring coverage)
  --model         per-class yield model {class: {members_mean, occupancy_mean_s, group_wall_mean_s, L}} (group_yield_model.py,
                  later re-measured from train rounds)
  --proposal      train_class_wallclock_proposal.csv (limits, caps, status)
  --flags         flagged sub-templates / templates from earlier rounds (excluded)
  --round-total   instances per class at 100k (default 1000 per class: final_plan_req train totals / classes per task) x --fraction

Rules (owner 2026-11-14):
  * class target = fraction x the 100k plan's per-class count; split evenly over the class's remaining templates
    (round-robin remainder), then over each template's remaining sub-templates (scales) by groups.
  * groups for a template = ceil(target / E[members per group]); E floored at --min-members-per-group.
  * worker item = (sub-template, groups) with time = startup + groups x group wall + demo tail; an item longer than the
    job budget is split into several items (parallel workers on the same sub-template are fine: private registries).
  * per-class GPU-hour cap per round; what does not fit is reported as deferred, not silently dropped.
  * jobs: first-fit-decreasing packing of items into chains of <= --job-hours (default 5 h) so jobs finish together;
    expensive classes therefore spread over more GPUs, cheap classes pack more groups per job.
"""
import argparse, collections, csv, json, math, random

PER_CLASS_100K = 1000   # every non-camera task's 100k total / its number of classes = 1000 (final_plan_req.md)


def load(p):
    with open(p) as f:
        return json.load(f)


def plan(subs, model, proposal, flags, fraction, job_s, startup_s, min_mpg, class_cap_gpu_h, seed):
    rng = random.Random(seed)
    flagged_sub = set(flags.get('subtemplates', [])); flagged_tpl = set(flags.get('templates', []))
    prop = {r['class_id']: r for r in proposal}
    by_class = collections.defaultdict(lambda: collections.defaultdict(list))
    for s in subs:
        if s.get('status') != 'valid' or not s.get('authorable') or s['configuration_id'] in flagged_sub \
                or s['template_key'] in flagged_tpl:
            continue
        if prop.get(s['class_id'], {}).get('status') != 'RUN':
            continue
        by_class[s['class_id']][s['template_key']].append(s['configuration_id'])
    items, report = [], {}
    for cid, templates in sorted(by_class.items()):
        p = prop[cid]; m = model.get(cid, {})
        L = float(p['train_group_limit_s']); cap = float(p['demo_safety_cap_s'])
        mpg = max(min_mpg, float(m.get('members_mean', 1.0)))
        gwall = float(m.get('group_wall_mean_s', L)); occ = float(m.get('occupancy_mean_s', L))
        target = int(round(PER_CLASS_100K * fraction))
        keys = sorted(templates); rng.shuffle(keys)
        per_tpl = {k: target // len(keys) for k in keys}
        for k in keys[: target % len(keys)]:
            per_tpl[k] += 1
        class_items, gpu_s, deferred = [], 0.0, 0
        # per-group GPU time: sampling occupancy (the sampler moves on while demos finish)
        for k in keys:
            t = per_tpl[k]
            if t <= 0:
                continue
            groups = math.ceil(t / mpg)
            sts = templates[k][:]; rng.shuffle(sts)
            # spread groups over the template's sub-templates (scales), one worker per sub-template
            alloc = collections.Counter(sts[i % len(sts)] for i in range(groups))
            for st, g in alloc.items():
                max_g = max(1, int((job_s - startup_s - cap - 420) // max(occ, 1)))
                while g > 0:
                    gg = min(g, max_g)
                    secs = startup_s + gg * occ + (gwall - occ) + 420
                    if class_cap_gpu_h and gpu_s + secs > class_cap_gpu_h * 3600:
                        deferred += gg * mpg; g -= gg; continue
                    class_items.append({'class_id': cid, 'configuration_id': st, 'template_key': k, 'groups': gg,
                                        'est_seconds': secs, 'expected_members': gg * mpg,
                                        'worker_seconds': int(min(14400, 1500 + gg * L + cap + 420))})
                    gpu_s += secs; g -= gg
        items += class_items
        report[cid] = {'target': target, 'templates': len(keys), 'subtemplates': sum(len(v) for v in templates.values()),
                       'items': len(class_items), 'groups': sum(i['groups'] for i in class_items),
                       'expected_members': round(sum(i['expected_members'] for i in class_items), 1),
                       'gpu_hours': round(gpu_s / 3600, 2), 'deferred_members': round(deferred, 1), 'members_per_group': mpg}
    # first-fit decreasing into chains of <= job_s
    jobs = []
    for it in sorted(items, key=lambda i: -i['est_seconds']):
        for j in jobs:
            if j['est_seconds'] + it['est_seconds'] <= job_s:
                j['items'].append(it); j['est_seconds'] += it['est_seconds']; break
        else:
            jobs.append({'items': [it], 'est_seconds': it['est_seconds']})
    return items, jobs, report


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--subtemplates', required=True); ap.add_argument('--model', required=True)
    ap.add_argument('--proposal', required=True); ap.add_argument('--flags')
    ap.add_argument('--fraction', type=float, default=0.1); ap.add_argument('--job-hours', type=float, default=5.0)
    ap.add_argument('--startup-s', type=float, default=600); ap.add_argument('--min-members-per-group', type=float, default=0.3)
    ap.add_argument('--class-cap-gpu-h', type=float, default=0); ap.add_argument('--seed', type=int, default=20261114)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    items, jobs, report = plan(load(a.subtemplates), load(a.model), list(csv.DictReader(open(a.proposal))),
                               load(a.flags) if a.flags else {}, a.fraction, a.job_hours * 3600, a.startup_s,
                               a.min_members_per_group, a.class_cap_gpu_h, a.seed)
    out = {'fraction': a.fraction, 'job_hours': a.job_hours, 'n_items': len(items), 'n_jobs': len(jobs),
           'gpu_hours': round(sum(j['est_seconds'] for j in jobs) / 3600, 1),
           'expected_members': round(sum(i['expected_members'] for i in items)), 'classes': report, 'jobs': jobs}
    json.dump(out, open(a.out, 'w'), indent=1)
    print(json.dumps({k: out[k] for k in ('n_items', 'n_jobs', 'gpu_hours', 'expected_members')}))
