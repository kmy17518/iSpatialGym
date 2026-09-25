# Paused for resume (coverage paused by the owner 2026-09-16 06:50 PDT)

Everything here resumes without recomputation: the manifests and progress ledgers are in this directory.

## Sampling chunks (wave A, `wave_a/jobs_coverage.json` + `wave_a/jobs_rest.json`)
- Running at the pause (let finish, results pulled by the C2 pull): see `paused_queues_0700.json` (`running` per pod).
- Queued and moved aside on the pods (`/root/samp/queue_paused/`): `rest_009__disembedding` (was on pod `oliefq3ymuexna`, now a pool pod; the chunk file stays under `/root/samp/queue_paused/` there and in `wave_a/jobs_rest.json`).
- 11:15 PDT: no pod keeps the sampling role (all wave-A chunks finished; `pods_samp.txt` empty). Resume = re-dispatch from the manifests to whichever pods get the role back.
- Not yet dispatched: none of wave A (all 23 chunks were dispatched; 21 finished or running at 07:00).
- Resume rule: regenerate the ledger (`scripts/isg/pilot_ledger.py --manifest jobs_sample.json,wave_a/jobs_coverage.json,wave_a/jobs_rest.json`),
  then `scripts/isg/pilot_jobs.py chunks --progress progress.json` (complete aliases excluded) for the coverage wave, then the rest.

## Deferred waves
- Wave B (needs the left / T2 / T3 bands): axial, attachment, block design, touching, small articulation; every T2 / T3 support alias
  (`--exclude-alias-regex` in wave A: countertop_kelker | bar_udatjt | conference_table | desk_mdhelw | commercial_kitchen | bar_xxftww).
- Fallback wave: `scripts/isg/pilot_jobs.py fallbacks --progress progress.json` (33 eligible at 05:40; only after the primaries' re-runs are in).
- Variation / held-out slots and `multiply_isg` per template: not started (discussion after the first-demo pass).

## Band fleet (8 band pods, `runlogs/runpod/pilot/pods_bands.txt`)
- Running: A3 rep wave (156); queued: A3 var (150, wave 9), A3 held (124, wave 10) — being re-ordered per item 7a (left bands per subtask first,
  breakfast-table remainder, then T2/T3); disembedding right-arm bands to recompute with the tray at 1.25 (item 4b / 9).

## Demo collection
- Pre-fix demos (120, `demos/demos.jsonl`): mental_rotation 50/52, object_scaling 42/44, camera standard 2/2, ontop-small 8/10, nextto 6/9 —
  all `pre-fix` under the validity rule (sampled before `7a340e4`); re-demo per subtask through `first_demo.py` (one instance each first).
- Remaining pre-fix instances not yet demoed: none pending (the full set was stopped at the pause).

- 2026-09-16 17:08 PDT pod_scheduler: role change bands -> pool: h7thjdjnyh8qcx (pilot2-l40s-0), bl91tuqop45ila (pilot2-l40s-2), cy0f5t6i58i08v (pilot2-l40s-3), s7m4h2ay2slau4 (pilot2-l40s-4), nfgjwzgjajpr5u (pilot2-l40s-7), usqruny80uunl6 (pilot2-l40s-9), 5gm563m9wgnrwc (pilot2-l40s-10), 6pdrunfn78iofy (pilot2-a40-4), rzcqsre0b89792 (pilot2-a40-5), 4udkfzz52mzpih (pilot2-a40-7), 8lp72savyk8co4 (pilot2-a40-8), gvua02xnwm4ti6 (pilot2-a40-9), 29s7u374iifjbj (pilot2-a40-10), z830mv63m0wjt7 (pilot2-a40-11); queued jobs moved: none (queues were empty); files updated: smoke/pods.txt, rb/pods.txt, pods_bands.txt (owner rule 11:05: one role per pod, nothing finished recomputed); coverage chunks stay paused

- 2026-09-16 17:24 PDT pod_scheduler: role change bands -> pool: 1qzna64k3s9t8s (pilot2-l40s-1); queued jobs moved: none (queues were empty); files updated: smoke/pods.txt, rb/pods.txt, pods_bands.txt (owner rule 11:05: one role per pod, nothing finished recomputed); coverage chunks stay paused

- 2026-09-16 18:08 PDT pod_scheduler: role change pool -> bands: uilwpybjxnekg0 (smoke2-a40-2), xnam1nh3y2706e (smoke-l40s-3), oliefq3ymuexna (pilot2-a40-1), bl91tuqop45ila (pilot2-l40s-2), cy0f5t6i58i08v (pilot2-l40s-3), s7m4h2ay2slau4 (pilot2-l40s-4); queued jobs moved: none (queues were empty); files updated: smoke/pods.txt, rb/pods.txt, pods_bands.txt (owner rule 11:05: one role per pod, nothing finished recomputed); coverage chunks stay paused

- 2026-09-16 18:13 PDT pod_scheduler: role change pool -> bands: 4udkfzz52mzpih (pilot2-a40-7), 8lp72savyk8co4 (pilot2-a40-8), gvua02xnwm4ti6 (pilot2-a40-9), 29s7u374iifjbj (pilot2-a40-10), z830mv63m0wjt7 (pilot2-a40-11), 1qzna64k3s9t8s (pilot2-l40s-1); queued jobs moved: none (queues were empty); files updated: smoke/pods.txt, rb/pods.txt, pods_bands.txt (owner rule 11:05: one role per pod, nothing finished recomputed); coverage chunks stay paused

- 2026-09-16 18:21 PDT pod_scheduler: PREEMPTION bl91tuqop45ila (pilot2-l40s-2): band job(s) none running stopped for first-demo work (started 18:11); files parked in /root/samp/queue_preempted, re-dispatched to a band pod by the next tick via the manifest (no band file yet = still remaining)

- 2026-09-16 18:21 PDT pod_scheduler: role change bands -> pool: bl91tuqop45ila (pilot2-l40s-2); queued jobs moved: none (queues were empty); files updated: smoke/pods.txt, rb/pods.txt, pods_bands.txt (owner rule 11:05: one role per pod, nothing finished recomputed); coverage chunks stay paused

- 2026-09-16 18:44 PDT pod_scheduler: role change pool -> bands: bpu6l9bffwaf1y (pilot2-l40s-11), usqruny80uunl6 (pilot2-l40s-9); queued jobs moved: none (queues were empty); files updated: smoke/pods.txt, rb/pods.txt, pods_bands.txt (owner rule 11:05: one role per pod, nothing finished recomputed); coverage chunks stay paused

- 2026-09-16 18:51 PDT pod_scheduler: role change pool -> bands: 6c5712ggii2ihf (pilot2-a40-2); queued jobs moved: none (queues were empty); files updated: smoke/pods.txt, rb/pods.txt, pods_bands.txt (owner rule 11:05: one role per pod, nothing finished recomputed); coverage chunks stay paused

- 2026-09-16 18:54 PDT pod_scheduler: role change pool -> bands: 88hr6mw4y5i1h8 (smoke2-l40s-1); queued jobs moved: none (queues were empty); files updated: smoke/pods.txt, rb/pods.txt, pods_bands.txt (owner rule 11:05: one role per pod, nothing finished recomputed); coverage chunks stay paused

- 2026-09-16 19:11 PDT pod_scheduler: role change pool -> bands: z2qdfyfyrgcjys (pilot2-a40-0), h7thjdjnyh8qcx (pilot2-l40s-0); queued jobs moved: none (queues were empty); files updated: smoke/pods.txt, rb/pods.txt, pods_bands.txt (owner rule 11:05: one role per pod, nothing finished recomputed); coverage chunks stay paused

- 2026-09-16 19:19 PDT pod_scheduler: role change pool -> bands: nfgjwzgjajpr5u (pilot2-l40s-7), rzcqsre0b89792 (pilot2-a40-5); queued jobs moved: none (queues were empty); files updated: smoke/pods.txt, rb/pods.txt, pods_bands.txt (owner rule 11:05: one role per pod, nothing finished recomputed); coverage chunks stay paused

- 2026-09-16 19:26 PDT pod_scheduler: PREEMPTION oliefq3ymuexna (pilot2-a40-1): band job(s) none running stopped for first-demo work (started 18:30); files parked in /root/samp/queue_preempted, re-dispatched to a band pod by the next tick via the manifest (no band file yet = still remaining)

- 2026-09-16 19:26 PDT pod_scheduler: PREEMPTION s7m4h2ay2slau4 (pilot2-l40s-4): band job(s) none running stopped for first-demo work (started 19:19); files parked in /root/samp/queue_preempted, re-dispatched to a band pod by the next tick via the manifest (no band file yet = still remaining)

- 2026-09-16 19:26 PDT pod_scheduler: PREEMPTION gvua02xnwm4ti6 (pilot2-a40-9): band job(s) none running stopped for first-demo work (started 19:02); files parked in /root/samp/queue_preempted, re-dispatched to a band pod by the next tick via the manifest (no band file yet = still remaining)

- 2026-09-16 19:26 PDT pod_scheduler: PREEMPTION 29s7u374iifjbj (pilot2-a40-10): band job(s) none running stopped for first-demo work (started 19:09); files parked in /root/samp/queue_preempted, re-dispatched to a band pod by the next tick via the manifest (no band file yet = still remaining)

- 2026-09-16 19:26 PDT pod_scheduler: role change bands -> pool: oliefq3ymuexna (pilot2-a40-1), s7m4h2ay2slau4 (pilot2-l40s-4), gvua02xnwm4ti6 (pilot2-a40-9), 29s7u374iifjbj (pilot2-a40-10); queued jobs moved: none (queues were empty); files updated: smoke/pods.txt, rb/pods.txt, pods_bands.txt (owner rule 11:05: one role per pod, nothing finished recomputed); coverage chunks stay paused

- 2026-09-16 19:36 PDT pod_scheduler: role change bands -> pool: xnam1nh3y2706e (smoke-l40s-3), cy0f5t6i58i08v (pilot2-l40s-3), bpu6l9bffwaf1y (pilot2-l40s-11), nfgjwzgjajpr5u (pilot2-l40s-7), rzcqsre0b89792 (pilot2-a40-5); queued jobs moved: none (queues were empty); files updated: smoke/pods.txt, rb/pods.txt, pods_bands.txt (owner rule 11:05: one role per pod, nothing finished recomputed); coverage chunks stay paused

- 2026-09-16 19:40 PDT pod_scheduler: role change bands -> pool: 8lp72savyk8co4 (pilot2-a40-8), 1qzna64k3s9t8s (pilot2-l40s-1), usqruny80uunl6 (pilot2-l40s-9), 6c5712ggii2ihf (pilot2-a40-2), 88hr6mw4y5i1h8 (smoke2-l40s-1), h7thjdjnyh8qcx (pilot2-l40s-0); queued jobs moved: none (queues were empty); files updated: smoke/pods.txt, rb/pods.txt, pods_bands.txt (owner rule 11:05: one role per pod, nothing finished recomputed); coverage chunks stay paused

- 2026-09-16 19:46 PDT pod_scheduler: role change bands -> pool: 4udkfzz52mzpih (pilot2-a40-7), z830mv63m0wjt7 (pilot2-a40-11), z2qdfyfyrgcjys (pilot2-a40-0); queued jobs moved: none (queues were empty); files updated: smoke/pods.txt, rb/pods.txt, pods_bands.txt (owner rule 11:05: one role per pod, nothing finished recomputed); coverage chunks stay paused

- 2026-09-16 21:37 PDT pod_scheduler: role change pool -> bands: m48tuedy95bu05 (smoke2-l40s-0), rzcqsre0b89792 (pilot2-a40-5), 8lp72savyk8co4 (pilot2-a40-8), 1qzna64k3s9t8s (pilot2-l40s-1); queued jobs moved: none (queues were empty); files updated: smoke/pods.txt, rb/pods.txt, pods_bands.txt (owner rule 11:05: one role per pod, nothing finished recomputed); coverage chunks stay paused

- 2026-09-16 21:43 PDT pod_scheduler: role change pool -> bands: z2qdfyfyrgcjys (pilot2-a40-0); queued jobs moved: none (queues were empty); files updated: smoke/pods.txt, rb/pods.txt, pods_bands.txt (owner rule 11:05: one role per pod, nothing finished recomputed); coverage chunks stay paused

- 2026-09-16 21:50 PDT pod_scheduler: role change pool -> bands: xnam1nh3y2706e (smoke-l40s-3), cy0f5t6i58i08v (pilot2-l40s-3); queued jobs moved: none (queues were empty); files updated: smoke/pods.txt, rb/pods.txt, pods_bands.txt (owner rule 11:05: one role per pod, nothing finished recomputed); coverage chunks stay paused

- 2026-09-16 22:01 PDT pod_scheduler: PREEMPTION xnam1nh3y2706e (smoke-l40s-3): band job(s) none running stopped for first-demo work (started 21:50); files parked in /root/samp/queue_preempted, re-dispatched to a band pod by the next tick via the manifest (no band file yet = still remaining)

- 2026-09-16 22:01 PDT pod_scheduler: role change bands -> pool: xnam1nh3y2706e (smoke-l40s-3); queued jobs moved: none (queues were empty); files updated: smoke/pods.txt, rb/pods.txt, pods_bands.txt (owner rule 11:05: one role per pod, nothing finished recomputed); coverage chunks stay paused

- 2026-09-16 22:02 PDT pod_scheduler: role change bands -> pool: cy0f5t6i58i08v (pilot2-l40s-3); queued jobs moved: none (queues were empty); files updated: smoke/pods.txt, rb/pods.txt, pods_bands.txt (owner rule 11:05: one role per pod, nothing finished recomputed); coverage chunks stay paused

- 2026-09-16 22:13 PDT pod_scheduler: PREEMPTION z2qdfyfyrgcjys (pilot2-a40-0): band job(s) ['rb__tabletop__office_cubicles_right__desk_pfsjbc_0__teacup-cpozxi'] stopped for first-demo work (started 21:43); files parked in /root/samp/queue_preempted, re-dispatched to a band pod by the next tick via the manifest (no band file yet = still remaining)

- 2026-09-16 22:13 PDT pod_scheduler: role change bands -> pool: z2qdfyfyrgcjys (pilot2-a40-0); queued jobs moved: ['rb__tabletop__office_cubicles_right__desk_pfsjbc_0__teacup-cpozxi']; files updated: smoke/pods.txt, rb/pods.txt, pods_bands.txt (owner rule 11:05: one role per pod, nothing finished recomputed); coverage chunks stay paused

- 2026-09-16 22:20 PDT pod_scheduler: PREEMPTION uilwpybjxnekg0 (smoke2-a40-2): band job(s) ['rb__tabletop__house_single_floor__countertop_ikwqer_0__shampoo_dispenser-zoyyia'] stopped for first-demo work (started 21:43); files parked in /root/samp/queue_preempted, re-dispatched to a band pod by the next tick via the manifest (no band file yet = still remaining)

- 2026-09-16 22:20 PDT pod_scheduler: role change bands -> pool: uilwpybjxnekg0 (smoke2-a40-2); queued jobs moved: ['rb__tabletop__house_single_floor__countertop_ikwqer_0__shampoo_dispenser-zoyyia']; files updated: smoke/pods.txt, rb/pods.txt, pods_bands.txt (owner rule 11:05: one role per pod, nothing finished recomputed); coverage chunks stay paused
