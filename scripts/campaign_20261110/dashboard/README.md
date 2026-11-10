# Campaign 20261110 local dashboard

Open **http://127.0.0.1:8765/** on the workstation. The browser fetches fresh
observations every 30 seconds without navigating or resetting filters. All
displayed observation times use `America/Chicago`.

`server.py` only reads local mirrors. It binds to loopback, serves an exact route
allowlist, rejects mutation methods and unrecognized Host headers, filters
credential fields, and never imports a scheduler or runs a subprocess. It does
not write `LIVE.md`, `live.json`, allocations, job state, or campaign manifests.
The report routes expose selected evidence, not arbitrary local files or raw
cluster logs. There is no public deployment or tunnel.

The separately launched `sync.py` reads two campaign evidence files over the
existing `minyeong@svl11` SSH connection and atomically updates its local mirror.
An independent CPU publisher runs `dashboard_evidence.py` on the cluster every
60 seconds. Neither collector is triggered by a browser request. The mirror also refreshes
`RETURN_REPORT.txt` from these observations and separately dated operator notes. Existing live
monitoring continues to own `LIVE.md`, `live.json`, and cluster observations.

## Data and interpretation

- `dashboard/dashboard_evidence.json`: locked snapshots of the SQL allocation
  ledger and all three camera ledgers. Contains exact committed IDs, 322 class
  and split rows, 23 tasks, source hashes, template identities, and accepted
  count history. A release still requires the immutable release re-audit.
- `live.json` and `infra/cluster_snapshot.json`: current validation, scheduler,
  capacity, budget, and worker observations. GPU job rows are restricted to
  this campaign. All-user partition usage is explicitly labeled because it
  determines shared scheduling headroom.
- `train_candidate_summary.json`: a dated inventory of ready-asset model/room
  combinations. Candidates have not passed runtime gates. Demo/replay counts
  use distinct admitted contract fingerprints; accepted template counts use
  normalized physical and conceptual identities. They are different populations.
- Failed reads preserve the last observation and show its age and error.
  Observations become stale after 3 minutes, or 5 minutes for Vast. Missing
  evidence is not converted into zero accepted instances.
- Throughput uses changes in committed acceptance, with per-task rates. A
  benchmark ETA requires positive measured rates for every unfinished task and
  split; an easy task's rate is never applied to other families.

The deadline is November 12, 2026 at **19:00 UTC / 13:00 Chicago**. It is a
handoff target, not an automatic shutdown. The campaign Vast cap is $3,000.

## Local services and verification

Service PIDs and logs are recorded in
`tmp/outputs/campaign_20261110/dashboard/services.json` in the original workspace.
Screenshots and `browser_check.json` are in the same directory. Collection
continues independently if the dashboard server is stopped.

From the campaign worktree, the launch commands are:

```bash
python scripts/campaign_20261110/dashboard/server.py \
  --data-root /home/minyeongk/Desktop/projects/ispatialgym/tmp/outputs/campaign_20261110 \
  --inventory /home/minyeongk/Desktop/projects/ispatialgym/tmp/worktrees/campaign_20261110/tmp/campaign_20261110/inventory
python scripts/campaign_20261110/dashboard/sync.py \
  --output /home/minyeongk/Desktop/projects/ispatialgym/tmp/outputs/campaign_20261110/dashboard
```

Do not start a second copy when the recorded services are running.

`python -m unittest scripts.campaign_20261110.dashboard.test_server` checks
count preservation, read-only methods, campaign isolation, credential filtering,
and missing/corrupt observations. `browser_check.mjs` uses an isolated local
Chrome session on debug port 9227 to check actual rendering, filters, Chicago
time, a real 30-second refresh without navigation, failed-request recovery,
GET-only browser traffic, and desktop/mobile overflow. QA never submits jobs.
