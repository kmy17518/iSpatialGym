"""Coordinator (cluster) for the 2026-11-15 batched replay on Vast.ai, og-isgrend-20261115 (source $RV/tool/creds.sh first;
adapted from train85/work/vast_replay/bin/vastctl.py). Secrets reach Vast only in REST bodies.

Fleet rules (BATCHED_REPLAY_HANDOFF.md section 0/6): only instances whose label starts with og-isgrend- AND whose id is in this
ledger are listed, reaped or destroyed; R/vast_global/STOP_ALL_RENTALS (never created here) and $RV/STOP_LAUNCH stop every rental;
RTX GPUs only, driver 570-580, < $1.5/GPU-h; Vast API calls are paced (>= 3 s apart). Every command takes --run <name>: the units,
claims, heartbeats and outputs live under $JOB_S3/runs/<run>/.

Four Vast accounts (ACCOUNTS: VAST_API_KEY, VAST_QUERY_API_KEY_1 / _2 / _3) rent, show and destroy instances, so one
account's hourly spend cap does not cap the fleet; searches stay on the query accounts.

    vastctl.py offers [--max 1.5] [--verified]          candidate offers, cheapest effective $/GPU-h first
    vastctl.py launch --offer ID [--procs P] [--prefer u1,u2] [--label L]   rent one offer
    vastctl.py status                                    instances, heartbeats, progress, spend
    vastctl.py reap [--dry-run]                          destroy dead / finished / stuck instances, release their claims
    vastctl.py loop --gpus N [--max 1.5] [--every 120]   keep ~N GPUs working while units are open; reap; ledger
    vastctl.py destroy ID|all                            destroy instances (ledger records it)

Effective $/GPU-h = (dph_total with the disk + (down_gb * inet_down_cost + up_gb * inet_up_cost) / hours) / num_gpus.
Ledger: $W/ledger/events.jsonl (create / destroy / credit snapshots); cost = sum of dph_total x rented hours.
"""
import argparse
import configparser
import json
import math
import os
import sys
import time
import uuid
from pathlib import Path

import boto3
import botocore
from vastai.api import instances as vi
from vastai.api import offers as vo
from vastai.api.client import VastClient

W = Path(os.environ.get("RV", "/vision/u/minyeong/projects/isg_home/campaign_20261110/training_generation/batched_replay_r1"))
assert str(W).endswith("training_generation/batched_replay_r1"), W
LEDGER = W / "ledger" / "events.jsonl"
ROOT = "jobs/og-isgrend-20261115"
PREFIX = ROOT  # set to ROOT/runs/<run> by main()
LABEL_PREFIX = "og-isgrend-"
STOP_ALL = Path("/vision/u/minyeong/projects/isg_home/campaign_20261110/vast_global/STOP_ALL_RENTALS")
ISG_KEY_S3 = "s3://isg-demos/isg-1020-b1k/secrets/omnigibson.key"
_last_api = [0.0]


def pace():
    """Vast API budget (shared main key, 429-limited): >= 3 s between calls of this process."""
    dt = time.time() - _last_api[0]
    if dt < 3.0:
        time.sleep(3.0 - dt)
    _last_api[0] = time.time()


def stop_reason():
    if STOP_ALL.exists():
        return f"{STOP_ALL} exists"
    if (W / "STOP_LAUNCH").exists():
        return f"{W / 'STOP_LAUNCH'} exists"
    return None


def ledger_ids():
    ids = set()
    if LEDGER.exists():
        for line in open(LEDGER):
            e = json.loads(line)
            if e["ev"] == "create" and e.get("instance"):
                ids.add(str(e["instance"]))
    return ids
IMAGE = os.environ.get("ISG_IMAGE", "ghcr.io/kmy17518/isg-replay:v3")
# RTX only (owner): no L40 / L40S / A40 / A10
GPU_OK = ["RTX 3090", "RTX 3090 Ti", "RTX 4090", "RTX 5090", "RTX A5000", "RTX A6000", "RTX 6000Ada",
          "RTX 5880Ada", "RTX 5000Ada", "RTX 4500Ada", "RTX PRO 5000", "RTX PRO 6000 S", "RTX PRO 6000 WS",
          "RTX PRO 6000 Max-Q"]  # no Turing (Q RTX 8000): slow per $
# relative replay throughput per GPU (RTX 3090 = 1); replaced by measured values in $W/ledger/speed.json when present
# 224 px batched replay, 2026-11-15: RTX 5090 ~36-40 fps/GPU (4 procs), RTX 4090 ~22-25 (2 procs), A5000 (cluster) ~9 per proc
SPEED = {"RTX 3090": 1.0, "RTX 3090 Ti": 1.05, "RTX A5000": 0.8, "RTX A6000": 1.0, "RTX 4090": 1.4, "RTX 5090": 2.0,
         "RTX 6000Ada": 1.5, "RTX 5880Ada": 1.4, "RTX 5000Ada": 1.2, "RTX 4500Ada": 1.0, "RTX PRO 5000": 1.5,
         "RTX PRO 6000 S": 2.0, "RTX PRO 6000 WS": 2.0, "RTX PRO 6000 Max-Q": 1.8}
HB_STALE_S = 900
LOADING_MAX_S = 1500


def now():
    return time.time()


def ledger(ev, **kw):
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with open(LEDGER, "a") as f:
        f.write(json.dumps({"ev": ev, "time": now(), **kw}) + "\n")


ACCOUNTS = ("VAST_API_KEY", "VAST_QUERY_API_KEY_1", "VAST_QUERY_API_KEY_2", "VAST_QUERY_API_KEY_3")
SHORT = {"VAST_API_KEY": "main", "VAST_QUERY_API_KEY_1": "q1", "VAST_QUERY_API_KEY_2": "q2", "VAST_QUERY_API_KEY_3": "q3"}
#: Vast error codes about the account (its hourly spend cap, credit, billing), not the offer: another account may rent it
ACCOUNT_ERRORS = ("spend_limit", "credit", "balance", "fund", "billing", "payment", "verif")
#: accounts whose instance list could not be read in the last my_instances() call
SHOW_ERRORS = set()


class AccountsFull(Exception):
    """No account could rent the offer: every one answered with an account error at this price or more."""


def vclients():
    """name -> Vast client for each account with a key (create / show / destroy); dict order = launch preference."""
    return {k: VastClient(api_key=os.environ[k]) for k in ACCOUNTS if os.environ.get(k)}


def owner_of(iid):
    """The account an instance was created with (ledger), else None; instances created before the second and third
    accounts are VAST_API_KEY's."""
    acct = None
    if LEDGER.exists():
        for line in open(LEDGER):
            e = json.loads(line)
            if e["ev"] == "create" and e.get("provider") != "runpod" and str(e.get("instance")) == str(iid):
                acct = e.get("account", "VAST_API_KEY")
    return acct


def r2():
    return boto3.client("s3", endpoint_url=os.environ["S3_ENDPOINT_URL"], region_name="auto")


def r2_key():
    c = configparser.ConfigParser()
    c.read(os.environ["AWS_SHARED_CREDENTIALS_FILE"])
    p = c[os.environ["AWS_PROFILE"]]
    return p["aws_access_key_id"], p["aws_secret_access_key"]


def r2_list(s3, prefix):
    out, token = {}, None
    while True:
        kw = {"Bucket": os.environ["S3_BUCKET"], "Prefix": f"{PREFIX}/{prefix}"}
        if token:
            kw["ContinuationToken"] = token
        r = s3.list_objects_v2(**kw)
        for o in r.get("Contents", []):
            out[o["Key"][len(f"{PREFIX}/{prefix}"):]] = o
        if not r.get("IsTruncated"):
            return out
        token = r["NextContinuationToken"]


def r2_json(s3, k):
    try:
        return json.loads(s3.get_object(Bucket=os.environ["S3_BUCKET"], Key=f"{PREFIX}/{k}")["Body"].read())
    except botocore.exceptions.ClientError:
        return None


def speeds():
    p = W / "ledger" / "speed.json"
    s = dict(SPEED)
    if p.exists():
        s.update(json.load(open(p)))
    return s


#: newest driver branch allowed (580: the cluster's; 595 / 610 / 615 only once one of them has replayed: ISG_DRIVER_MAX)
DRIVER_MAX = int(os.environ.get("ISG_DRIVER_MAX", "580"))


def driver_ok(v):
    try:
        return 570 <= int(str(v).split(".")[0]) <= DRIVER_MAX
    except ValueError:
        return False


def disk_for(n_gpus, procs):
    # env 20 GB + assets ~38 GB (every behavior-1k object) + texture packs (common + house_single_floor 16.3 GB at boot, other scenes
    # on demand, all 54 GB) + 8 GB a process
    return int(150 + 8 * n_gpus * procs)


SEARCH_EVERY_S = 600
SEARCH_LIMIT = 100
QUERY_KEYS = ("VAST_QUERY_API_KEY_1", "VAST_QUERY_API_KEY_2", "VAST_QUERY_API_KEY_3")


def search_keys_state():
    p = W / "market" / "search_state.json"
    return json.load(open(p)) if p.exists() else {}


def save_search_state(st):
    (W / "market" / "search_state.json").write_text(json.dumps(st, indent=1))


def search(c, max_eff, verified_only=True, procs=2, hours=3.0, down_gb=30, up_gb_per_gpu_h=1.0, min_down=700):
    """Quota-safe offer search: at most one search every SEARCH_EVERY_S, done with the search-only accounts
    (VAST_QUERY_API_KEY_1 / _2, rotated on 429; each 429's retry_after is recorded), filtered on Vast's side and capped at
    SEARCH_LIMIT rows (~14,000 rows/day of the 20,000 a key may read); the result is saved as market/fresh_<time>.json and
    reused (cached_offers) until the next search."""
    st = search_keys_state()
    live = {i.get("machine_id") for i in my_instances(c)}
    if now() - st.get("last_search", 0) < SEARCH_EVERY_S:
        return cached_offers(max_eff, live)
    q = {"rentable": {"eq": True}, "rented": {"eq": False}, "external": {"eq": False}, "gpu_name": {"in": list(GPU_OK)},
         "gpu_ram": {"gte": 23000}, "num_gpus": {"gte": 1}, "inet_down": {"gte": min_down}, "reliability2": {"gte": 0.95},
         "cpu_ram": {"gte": 32000}, "cpu_cores_effective": {"gte": 6},
         "driver_version": {"gte": 570000000, "lte": DRIVER_MAX * 1000000 + 999999},
         "dph_total": {"lte": round(max_eff * 8, 2)}}
    if verified_only:
        q["verified"] = {"eq": True}
    for name in QUERY_KEYS:
        if not os.environ.get(name) or st.get(f"blocked_until_{name}", 0) > now():
            continue
        try:
            pace()
            rows = vo.search_offers(VastClient(api_key=os.environ[name]), query=dict(q), order=[["dph_total", "asc"]],
                                    limit=SEARCH_LIMIT, storage=disk_for(1, procs), no_default=True)
        except Exception as e:  # noqa: BLE001
            body = getattr(getattr(e, "response", None), "text", "") or ""
            if "429" in str(e) or "quota" in body:
                retry = json.loads(body).get("retry_after", 3600) if body.startswith("{") else 3600
                st[f"blocked_until_{name}"] = now() + float(retry)
                save_search_state(st)
                print(f"search key {name} rate-limited for {float(retry) / 3600:.1f} h", file=sys.stderr)
                continue
            print(f"search ({name}): {str(e)[:200]}", file=sys.stderr)
            continue
        st["last_search"] = now()
        st["last_key"] = name
        save_search_state(st)
        f = W / "market" / f"fresh_{time.strftime('%Y%m%d_%H%M%S')}.json"
        f.write_text(json.dumps({"time": now(), "key": name, "query": q, "offers": rows}))
        print(f"search ({name}): {len(rows)} offers -> {f.name}", file=sys.stderr)
        break
    else:
        st["last_search"] = now()  # every key blocked: try again in SEARCH_EVERY_S
        save_search_state(st)
    return cached_offers(max_eff, live)


def cached_offers(max_eff, live_machines=None):
    """Offers saved from earlier searches (market/*.json), best $ per replay process first, without blacklisted machines,
    machines in use and offers already tried; their price is checked again once rented (status/reap)."""
    seen = {}
    files = sorted((W / "market").glob("fresh_*.json"), reverse=True)[:3]
    for p in files:
        if p.exists():
            d = json.load(open(p))
            for o in (d["offers"] if isinstance(d, dict) else d):
                seen.setdefault(o["id"], o)
    bad = bad_machines()
    tried, created, destroyed = set(), {}, set()
    if LEDGER.exists():
        for line in open(LEDGER):
            e = json.loads(line)
            if e["ev"] == "tried" and now() - e["time"] < 3600:
                tried.add(e.get("offer"))
            elif e["ev"] == "create":
                created[str(e.get("instance"))] = e.get("machine_id")
            elif e["ev"] == "destroy":
                destroyed.add(str(e.get("instance")))
    if live_machines is None:
        live_machines = {m for i, m in created.items() if i not in destroyed}
    out = []
    for o in seen.values():
        n = int(o.get("num_gpus") or 1)
        if (o["id"] in tried or o.get("machine_id") in bad or o.get("machine_id") in live_machines or o.get("gpu_name") not in GPU_OK
                or not driver_ok(o.get("driver_version")) or str(o.get("geolocation") or "").endswith("CN")
                or n < 1 or o.get("verification") != "verified"  # the same bar as the search (older saved files are looser)
                or float(o.get("reliability2") or 0) < 0.95 or float(o.get("inet_down") or 0) < 700):
            continue
        cpu_g = float(o.get("cpu_cores_effective") or 0) / n
        ram_g = float(o.get("cpu_ram") or 0) / 1024 / n
        p_est = int(min(4, float(o.get("gpu_ram") or 0) // 10500, cpu_g // 3, ram_g // 16))
        eff = float(o["dph_total"]) / n
        if p_est < 2 or eff >= max_eff:
            continue
        o = dict(o)
        o["_eff"], o["_disk"], o["_cpu_g"], o["_ram_g"], o["_procs"] = eff, disk_for(n, p_est), cpu_g, ram_g, p_est
        o["_value"] = eff / speeds().get(o.get("gpu_name"), 0.7)  # $ per unit of measured replay throughput
        out.append(o)
    out.sort(key=lambda o: (o.get("verification") != "verified", -int(o.get("num_gpus") or 1), o["_value"]))
    by_machine = {}
    for o in out:
        by_machine.setdefault(o.get("machine_id") or o["id"], o)
    res = list(by_machine.values())
    res.sort(key=lambda o: (o.get("verification") != "verified", round(o["_value"], 2), -int(o.get("num_gpus") or 1)))
    return res


def bad_machines():
    bad = set()
    if LEDGER.exists():
        for line in open(LEDGER):
            e = json.loads(line)
            if e["ev"] == "bad_machine":
                bad.add(e["machine_id"])
    return bad


def fmt_offer(o):
    return (f"{o['id']:<9} {o['gpu_name']:12s} x{o['num_gpus']:<2} ${o['_eff']:.3f}/GPU-h value ${o['_value']:.3f} "
            f"drv {o.get('driver_version')} cpu/gpu {o['_cpu_g']:.0f} ram/gpu {o['_ram_g']:.0f}G procs/gpu {o['_procs']} "
            f"net {o.get('inet_down', 0):.0f}/{o.get('inet_up', 0):.0f} rel {o.get('reliability2', 0):.3f} "
            f"{o.get('verification')} {o.get('geolocation')} m{o.get('machine_id')}")


def launch(c, o, procs=None, prefer="", label="og-isgrend", full=None, run=None, max_eff=1.5):
    """Rent offer o with the first account that can. An account answering with an account error (spend_limit_reached,
    credit) is marked full at this offer's price (full: name -> $/h, shared over a loop tick; it is tried again only for
    cheaper offers) and the next account is tried. Raises the offer's own error (e.g. no_such_ask), or AccountsFull."""
    full = {} if full is None else full
    why = stop_reason()
    if why:
        raise AccountsFull(f"no rental: {why}")
    if float(o["dph_total"]) / max(1, int(o.get("num_gpus") or 1)) >= max_eff:
        raise AccountsFull(f"offer {o['id']} over ${max_eff}/GPU-h")
    if not str(o.get("gpu_name")) in GPU_OK:
        raise AccountsFull(f"offer {o['id']} {o.get('gpu_name')} is not an RTX GPU")
    key_id, secret = r2_key()
    tag = uuid.uuid4().hex[:8]
    assert label.startswith("og-isgrend"), label
    env = {"ISG_S3": f"s3://{os.environ['S3_BUCKET']}/{ROOT}", "ISG_RUN": run, "ISG_KEY_S3": ISG_KEY_S3,
           "ISG_TEX_S3": f"s3://{os.environ['S3_BUCKET']}/isg-1020-b1k/bundle/textures",
           "S3_ENDPOINT_URL": os.environ["S3_ENDPOINT_URL"],
           "AWS_ACCESS_KEY_ID": key_id, "AWS_SECRET_ACCESS_KEY": secret, "AWS_REGION": "auto", "ISG_TAG": tag,
           "NVIDIA_VISIBLE_DEVICES": "all", "NVIDIA_DRIVER_CAPABILITIES": "all"}
    if procs:
        env["ISG_PROCS_PER_GPU"] = str(procs)
    if prefer:
        env["ISG_PREFER"] = prefer
    for k in ("ISG_PIN", "ISG_WARMUP_S", "ISG_GPU_ID_MODE", "ISG_NO_DESTROY", "ISG_WORKER", "ISG_DRIVER_MAX", "ISG_DIAG_ONLY",
              "ISG_BOOT_TIMEOUT_S", "ISG_STAGGER_S", "ISG_SWEEP"):
        if os.environ.get(k):
            env[k] = os.environ[k]
    login = f"-u {os.environ['GHCR_USERNAME']} -p {os.environ['GHCR_PULL_TOKEN']} ghcr.io"
    price = float(o["dph_total"])
    for name, cl in c.items():
        if price >= full.get(name, math.inf):
            continue
        try:
            pace()
            r = vi.create_instance(cl, o["id"], image=IMAGE, disk=o["_disk"], env=env, label=f"{label}-{tag}", login=login,
                                   runtype="args", args=[], cancel_unavail=True)
        except Exception as e:  # noqa: BLE001
            err = vast_error(e) or ""
            if any(w in err for w in ACCOUNT_ERRORS):
                full[name] = min(full.get(name, math.inf), price)
                print(f"account {SHORT.get(name, name)}: {err} at ${price:.2f}/h (offer {o['id']}); trying the next account")
                continue
            raise
        iid = r.get("new_contract")
        ledger("create", instance=iid, offer=o["id"], machine_id=o.get("machine_id"), gpu=o["gpu_name"], n=o["num_gpus"],
               dph_total=o["dph_total"], eff=o["_eff"], disk=o["_disk"], tag=tag, label=label, procs=procs,
               driver=o.get("driver_version"), geo=o.get("geolocation"), verification=o.get("verification"), account=name, run=run)
        print(f"LAUNCHED instance {iid} [{SHORT.get(name, name)}] tag {tag} {fmt_offer(o)}")
        return iid
    raise AccountsFull(f"offer {o['id']} (${price:.2f}/h): every account full "
                       f"({', '.join(f'{SHORT.get(k, k)} <${v:.2f}/h' for k, v in full.items())})")


def my_instances(c):
    """Our instances (label og-isgrend-* AND created in this ledger) on every account, each with _key_name = its account;
    SHOW_ERRORS names the accounts whose list could not be read (their instances are missing from the result)."""
    out = []
    SHOW_ERRORS.clear()
    mine = ledger_ids()
    for name, cl in c.items():
        try:
            pace()
            for i in vi.show_instances(cl):
                if str(i.get("label", "")).startswith(LABEL_PREFIX) and str(i.get("id")) in mine:
                    i["_key_name"] = name
                    out.append(i)
        except Exception as e:  # noqa: BLE001
            SHOW_ERRORS.add(name)
            print(f"show instances ({SHORT.get(name, name)}) failed: {vast_error(e) or str(e)[:160]}")
    return out


def destroy(c, iid, reason, key_name=None):
    """Destroy with the owning account (key_name, else the ledger's), falling back to the other accounts. Only ids this ledger
    created (og-isgrend- fleet rule)."""
    if str(iid) not in ledger_ids():
        print(f"REFUSE destroy {iid}: not created by this ledger")
        return False
    owner = key_name or owner_of(iid)
    order = ([owner] if owner in c else []) + [k for k in c if k != owner]
    ok, used = False, None
    for name in order:
        try:
            pace()
            r = vi.destroy_instance(c[name], iid)
            if isinstance(r, dict) and r.get("success") is False:
                raise RuntimeError(r.get("error") or r.get("msg") or "success: false")
            ok, used = True, name
            break
        except Exception as e:  # noqa: BLE001
            print(f"destroy {iid} with {SHORT.get(name, name)} failed: {vast_error(e) or str(e)[:160]}")
    ledger("destroy", instance=iid, reason=reason, ok=ok, account=used)
    print(f"DESTROY {iid} [{SHORT.get(used, used)}]: {reason} ({'ok' if ok else 'FAILED'})")
    return ok


def credit(c):
    """Credit and total spend summed over the accounts, with each account's credit."""
    out = {"credit": 0.0, "total_spend": 0.0, "accounts": {}}
    for name, cl in c.items():
        try:
            pace()
            u = cl.get("/users/current/").json()
            out["credit"] += float(u.get("credit") or 0)
            out["total_spend"] += float(u.get("total_spend") or 0)
            out["accounts"][SHORT.get(name, name)] = round(float(u.get("credit") or 0), 2)
        except Exception as e:  # noqa: BLE001
            out["accounts"][SHORT.get(name, name)] = f"error {str(e)[:60]}"
    out["credit"], out["total_spend"] = round(out["credit"], 2), round(out["total_spend"], 2)
    return out


def progress(s3):
    units = json.load(open(W / "runs" / RUN[0] / "units.json"))["units"]
    done = r2_list(s3, "done/")
    claims = r2_list(s3, "claims/")
    dead = r2_list(s3, "dead/")
    frames = {u["unit"]: u["frames"] for u in units}
    ids = set(frames)
    # ids retired by a re-split (dead/, gone from units.json) are not counted
    done_ids = {k[:-5] for k in done} & ids
    claim_ids = {k[:-5] for k in claims} & ids
    dead_ids = {k[:-5] for k in dead} & ids
    running = claim_ids - done_ids
    return {"units": len(units), "done": len(done_ids), "claimed": len(running), "dead": len(dead_ids),
            "frames_total": sum(frames.values()), "frames_done": sum(frames[u] for u in done_ids),
            "frames_open": sum(frames[u] for u in ids - done_ids - claim_ids - dead_ids),
            "frames_running": sum(frames[u] for u in running),
            "open": len(ids - done_ids - claim_ids - dead_ids), "claims": claims, "done_ids": done_ids}


def heartbeats(s3):
    out = {}
    for k in r2_list(s3, "hb/"):
        hb = r2_json(s3, f"hb/{k}")
        if hb:
            out[str(hb.get("instance"))] = hb
    return out


def spend():
    """Ledger estimate: dph_total x hours between create and destroy (or now)."""
    created, ended = {}, {}
    if LEDGER.exists():
        for line in open(LEDGER):
            e = json.loads(line)
            if e.get("provider") == "runpod":
                continue
            if e["ev"] == "create" and e.get("instance"):
                created[e["instance"]] = e
            elif e["ev"] == "destroy" and e.get("ok"):
                ended.setdefault(e["instance"], e["time"])
    tot = 0.0
    for iid, e in created.items():
        tot += e["dph_total"] * (ended.get(iid, now()) - e["time"]) / 3600
    return tot, len(created), len(created) - len(ended)


def status(c, s3, quiet=False):
    inst = my_instances(c)
    hbs = heartbeats(s3)
    pr = progress(s3)
    est, n_created, n_live = spend()
    lines = [f"[{time.strftime('%H:%M:%S')}] units {pr['done']}/{pr['units']} done, {pr['claimed']} claimed, {pr['open']} open, "
             f"{pr['dead']} dead; frames {pr['frames_done']}/{pr['frames_total']} ({100 * pr['frames_done'] / pr['frames_total']:.1f}%); "
             f"instances {len(inst)} (ledger: {n_created} created, est spend ${est:.2f}); {credit(c)}"]
    for i in sorted(inst, key=lambda i: i["id"]):
        hb = hbs.get(str(i["id"])) or {}
        run = hb.get("running") or []
        head = (f"  {i['id']} [{SHORT.get(i.get('_key_name'), '?')}] {i.get('gpu_name')} x{i.get('num_gpus')} "
                f"{i.get('actual_status')}/{i.get('intended_status')} "
                f"${float(i.get('dph_total') or 0):.3f}/h up {(now() - (i.get('start_date') or now())) / 60:.0f}m | ")
        if hb.get("time"):
            head += f"hb {hb.get('phase')} {hb.get('detail', '')} age {now() - hb['time']:.0f}s "
        else:
            head += "no hb "
        lines.append(head)
        if hb.get("phase") in ("run", "template"):
            lines[-1] += (f"done {hb.get('done')} fail {hb.get('failed')} frames {hb.get('frames_done')} "
                          f"gpu {hb.get('gpu_util')} mem {hb.get('mem_used_gb')}/{hb.get('mem_gb')}G cpu {hb.get('cpu_busy')} "
                          f"running {[(r['unit'], r.get('done'), r.get('n')) for r in run]}")
    if not quiet:
        print("\n".join(lines))
    return inst, hbs, pr


def reap(c, s3, dry=False):
    inst, hbs, pr = status(c, s3, quiet=True)
    live = {str(i["id"]): i for i in inst}
    t = now()
    for i in inst:
        iid = str(i["id"])
        hb = hbs.get(iid)
        up = t - (i.get("start_date") or t)
        st = i.get("actual_status")
        reason = None
        msg = str(i.get("status_msg") or "")
        host_bad = None
        n_g = max(1, int(i.get("num_gpus") or 1))
        if float(i.get("dph_total") or 0) / n_g >= 1.5:  # owner: < $1.5/GPU-h
            destroy(c, i["id"], f"price {float(i.get('dph_total') or 0) / n_g:.2f} $/GPU-h over the limit", i.get("_key_name"))
            continue
        if hb and hb.get("phase") in ("destroyed", "done", "failed") and t - hb["time"] > 120:
            reason = f"worker phase {hb.get('phase')} {hb.get('detail', '')}"
        elif not hb and ("Error" in msg or (st == "created" and i.get("intended_status") == "stopped")):
            reason = host_bad = f"container failed to start: {msg[:120]}"
        elif st in ("exited",) and up > 300:
            reason = f"status {st}"
        elif st in ("loading", "created", None) and up > 600 and not hb:
            reason = host_bad = f"container not started after {up / 60:.0f} min ({st}: {msg[:80]})"
        elif hb and t - hb["time"] > HB_STALE_S:
            reason = f"heartbeat stale {(t - hb['time']) / 60:.0f} min"
        elif not hb and st == "running" and up > LOADING_MAX_S:
            reason = f"running {up / 60:.0f} min without a heartbeat"
        if reason:
            if host_bad:
                ledger("bad_machine", machine_id=i.get("machine_id"), instance=i["id"], reason=host_bad)
            detail = f"{(hb or {}).get('phase')} {(hb or {}).get('detail', '')}"
            if (hb or {}).get("phase") == "failed" or any(w in detail for w in ("driver", "nvidia-smi", "GLX", "self-test",
                                                                                 "RT cores")):
                ledger("bad_machine", machine_id=i.get("machine_id"), instance=i["id"], reason=reason)
            if not dry:
                destroy(c, i["id"], reason, i.get("_key_name"))
            else:
                print(f"would destroy {iid}: {reason}")
    # instances gone without a destroy event (they destroyed themselves): record it, at their last heartbeat
    created, ended = {}, set()
    if LEDGER.exists():
        for line in open(LEDGER):
            e = json.loads(line)
            if e["ev"] == "create" and e.get("instance"):
                created[str(e["instance"])] = e
            elif e["ev"] == "destroy":
                ended.add(str(e["instance"]))
    for iid, e in created.items():
        if SHOW_ERRORS:  # an account's list is missing: its instances are not gone
            break
        if iid.isdigit() and iid not in live and iid not in ended and t - e["time"] > 120:
            hb = hbs.get(iid) or {}
            with open(LEDGER, "a") as f:
                f.write(json.dumps({"ev": "destroy", "time": hb.get("time", t), "instance": int(iid), "ok": True,
                                    "reason": f"gone ({hb.get('phase', 'no heartbeat')} {hb.get('detail', '')})".strip()}) + "\n")
    # instances that destroyed themselves for a host problem: never rent that machine again
    machine_of, flagged = {}, set()
    if LEDGER.exists():
        for line in open(LEDGER):
            e = json.loads(line)
            if e["ev"] == "create" and e.get("instance"):
                machine_of[str(e["instance"])] = e.get("machine_id")
            elif e["ev"] == "bad_machine":
                flagged.add(str(e.get("instance")))
    for iid, hb in hbs.items():
        detail = f"{hb.get('phase')} {hb.get('detail', '')}"
        if iid.isdigit() and iid not in flagged and hb.get("phase") in ("destroyed", "failed") and any(
                w in detail for w in ("driver", "nvidia-smi", "GLX", "self-test", "RT cores")):
            ledger("bad_machine", machine_id=machine_of.get(iid), instance=iid, reason=detail)
            print(f"bad machine {machine_of.get(iid)} (instance {iid}): {detail}")
    # claims of instances that are gone or silent go back to the pool
    for k in pr["claims"]:
        uid = k[:-5]
        if uid in pr["done_ids"]:
            continue
        cl = r2_json(s3, f"claims/{k}") or {}
        owner = str(cl.get("instance"))
        if owner.startswith("cluster"):  # cluster lanes (cluster_assign.py) hold their units; never released here
            continue
        hb = hbs.get(owner)
        # other providers' workers (rp-<pod>): only silence counts; nothing is gone while an account's list is missing
        gone = owner.isdigit() and owner not in live and not SHOW_ERRORS
        silent = hb is None or t - hb["time"] > HB_STALE_S
        if (gone or silent) and t - cl.get("time", t) > 600:
            print(f"release claim {uid} of {owner} (gone={gone} silent={silent})")
            if not dry:
                s3.delete_object(Bucket=os.environ["S3_BUCKET"], Key=f"{PREFIX}/claims/{k}")
                ledger("release", unit=uid, instance=owner)


def runpod_tick(s3, hbs, pr, last_try):
    """RunPod (second provider): reap its pods; while $W/target_runpod_gpus asks for more and units are open, try for a
    pod on a 570-580 driver at most every 10 min (create fails at once when RunPod has none)."""
    import runpod
    try:
        runpod.reap(s3, hbs)
        tf = W / "target_runpod_gpus"
        target = int(tf.read_text().split()[0]) if tf.exists() else 0
        have = sum(int(p.get("gpuCount") or 0) for p in runpod.pods())
        if target > have and pr["open"] > 0 and now() - last_try > 600 and not os.path.exists(W / "STOP_LAUNCH"):
            last_try = now()
            for cloud in ("COMMUNITY", "SECURE"):
                try:
                    runpod.launch(["A6000", "A40", "4090", "L40", "6000Ada", "L40S", "5090"], 2, cloud=cloud)
                    break
                except Exception as e:  # noqa: BLE001
                    print(f"runpod {cloud}: {runpod.redact(e)[:160]}")
    except Exception as e:  # noqa: BLE001
        print(f"runpod tick error: {str(e)[:200]}")
    return last_try


def vast_error(e):
    """Vast's error code from a failed request (e.g. no_such_ask, spend_limit_reached), else None."""
    try:
        return e.response.json().get("error")
    except Exception:  # noqa: BLE001
        return None


def loop(c, s3, a):
    rp_last = 0.0
    while True:
        try:
            reap(c, s3)
            inst, hbs, pr = status(c, s3)
            # (no RunPod in this campaign)
            ledger("tick", **{k: v for k, v in pr.items() if k not in ("claims", "done_ids")}, credit=credit(c))
            gpus = sum(int(i.get("num_gpus") or 0) for i in inst)
            tf = W / "target_gpus"
            if tf.exists():
                a.gpus = int(tf.read_text().split()[0])
            remaining = pr["open"]
            if remaining == 0 and pr["claimed"] == 0:
                print("ALL UNITS DONE")
                if not inst:
                    return
            burn = sum(float(i.get("dph_total") or 0) for i in inst)
            if remaining > 0 and gpus < a.gpus and not stop_reason():
                booting = sum(int(i.get("num_gpus") or 0) for i in inst
                              if (hbs.get(str(i["id"])) or {}).get("phase") not in ("run",))
                want = min(a.gpus - gpus, max(0, math.ceil(remaining / (a.procs or 2)) - booting))
                unverified = sum(int(i.get("num_gpus") or 0) for i in inst if str(i.get("verification", "")) != "verified")
                full, refused = {}, 0
                for o in search(c, a.max, verified_only=True, procs=a.procs or 2):
                    if want <= 0:
                        break
                    if int(o["num_gpus"]) > want + 2 or int(o["num_gpus"]) > a.max_gpus_per_instance:
                        continue
                    if o.get("verification") != "verified":
                        if unverified + int(o["num_gpus"]) > 0.25 * a.gpus:
                            continue
                        unverified += int(o["num_gpus"])
                    if len(full) == len(c) and float(o["dph_total"]) >= max(full.values()):
                        continue  # no account has room at this price
                    try:
                        launch(c, o, procs=a.procs, label=a.label, full=full, run=RUN[0], max_eff=a.max)
                        want -= int(o["num_gpus"])
                    except AccountsFull as e:  # every account's hourly cap: not the offer's fault, cheaper ones may fit
                        refused += 1
                        print(f"launch {e} (fleet ${burn:.2f}/h)")
                        if refused >= 8:
                            break
                        continue
                    except Exception as e:  # noqa: BLE001
                        err = vast_error(e)
                        ledger("tried", offer=o["id"], machine_id=o.get("machine_id"), error=(err or str(e))[:120])
                        print(f"launch {o['id']} failed: {(err or str(e))[:120]}")
                    time.sleep(2)
        except Exception as e:  # noqa: BLE001
            print(f"loop error: {e}")
        sys.stdout.flush()
        time.sleep(a.every)


RUN = [None]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run name: $JOB_S3/runs/<run>/ and $RV/runs/<run>/units.json")
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("offers")
    p.add_argument("--max", type=float, default=1.5)
    p.add_argument("--verified", action="store_true")
    p.add_argument("--procs", type=int, default=2)
    p.add_argument("--show", type=int, default=40)
    p = sp.add_parser("launch")
    p.add_argument("--offer", required=True, help="offer id(s), comma-separated: the first available is rented")
    p.add_argument("--procs", type=int)
    p.add_argument("--prefer", default="")
    p.add_argument("--label", default="og-isgrend")
    p.add_argument("--max", type=float, default=1.5)
    sp.add_parser("status")
    p = sp.add_parser("reap")
    p.add_argument("--dry-run", action="store_true")
    p = sp.add_parser("loop")
    p.add_argument("--gpus", type=int, required=True)
    p.add_argument("--max", type=float, default=1.5)
    p.add_argument("--procs", type=int)
    p.add_argument("--verified", action="store_true")
    p.add_argument("--every", type=int, default=120)
    p.add_argument("--max-gpus-per-instance", type=int, default=8)
    p.add_argument("--label", default="og-isgrend")
    p = sp.add_parser("destroy")
    p.add_argument("id")
    p.add_argument("--reason", default="manual")
    a = ap.parse_args()
    global PREFIX
    RUN[0] = a.run
    PREFIX = f"{ROOT}/runs/{a.run}"
    (W / "market").mkdir(parents=True, exist_ok=True)
    c, s3 = vclients(), r2()
    if a.cmd == "offers":
        for o in search(c, a.max, verified_only=a.verified, procs=a.procs)[: a.show]:
            print(fmt_offer(o))
    elif a.cmd == "launch":
        offers = {o["id"]: o for o in search(c, a.max, procs=a.procs or 2)}
        for oid in [int(x) for x in a.offer.split(",")]:
            if oid not in offers:
                print(f"offer {oid} not available under the filters")
                continue
            try:
                launch(c, offers[oid], procs=a.procs, prefer=a.prefer, label=a.label, run=a.run, max_eff=a.max)
                break
            except AccountsFull as e:
                print(f"launch {e}")
            except Exception as e:  # noqa: BLE001
                ledger("tried", offer=oid, machine_id=offers[oid].get("machine_id"), error=(vast_error(e) or str(e))[:120])
                print(f"launch {oid} failed: {(vast_error(e) or str(e))[:200]}")
    elif a.cmd == "status":
        status(c, s3)
    elif a.cmd == "reap":
        reap(c, s3, dry=a.dry_run)
    elif a.cmd == "loop":
        loop(c, s3, a)
    elif a.cmd == "destroy":
        targets = [(i["id"], i["_key_name"]) for i in my_instances(c)] if a.id == "all" else [(int(a.id), None)]
        for iid, key_name in targets:
            destroy(c, iid, a.reason, key_name)


if __name__ == "__main__":
    main()
