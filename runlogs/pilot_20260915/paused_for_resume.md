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
