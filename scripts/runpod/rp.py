#!/usr/bin/env python3
"""RunPod fleet tooling for iSpatialGym batch jobs (pods = one OmniGibson/cuRobo worker box each).

Credentials come from the repo ``.env`` (``RUNPOD_API_KEY``, ``HF_TOKEN``, ``GH_TOKEN``); nothing is ever printed
or written to disk. The fleet we created is tracked in ``runlogs/runpod/fleet.json`` (pod id, GPU, TTL, purpose),
spend is tracked in ``runlogs/runpod/billing.log`` (balance samples, RECHARGE events = balance jumps).

Guard rails (the account has an auto-recharge card, so runaway pods cost real money):

* every pod gets a TTL; ``watchdog`` terminates pods past their TTL and pods that report idle for too long
  (never for budget reasons unless started with ``--enforce-cap``: the balance / RECHARGE lines are the budget signal),
* the pod itself arms a self-terminate timer in ``bootstrap.sh`` (``runpodctl remove pod`` at TTL),
* ``create`` refuses to start a pod when the projected spend (spent so far + fleet cost until TTL) exceeds ``--cap``.

Usage (run from the repo root, ``python scripts/runpod/rp.py <cmd> ...``):

    balance                              balance, $/h of the running fleet, spend so far
    stock                                RTX-class GPU types with price and stock status
    create --gpu "NVIDIA GeForce RTX 4090" --name bench-4090 --ttl-h 2 [--cloud SECURE|COMMUNITY] [--disk 150]
    list                                 pods of the account (ours are marked)
    wait-ssh <pod>                       block until sshd is reachable, print "ip port"
    run <pod> "<remote command>"         run a command over SSH (direct TCP port)
    push <pod> <local> <remote>          rsync local -> pod   /   pull <pod> <remote> <local>
    bootstrap <pod> [--job bench:2]      copy bootstrap.sh + run it detached on the pod (install, verify, optional job)
    terminate <pod>|--all|--ours         terminate pods (irreversible)
    watchdog [--interval 300]            loop: enforce TTLs, log balance / recharges, terminate idle pods
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shlex
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional

import requests

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ENV_PATH = os.path.join(REPO, ".env")
STATE_DIR = os.path.join(REPO, "runlogs", "runpod")
FLEET_PATH = os.path.join(STATE_DIR, "fleet.json")
BILLING_LOG = os.path.join(STATE_DIR, "billing.log")
SSH_KEY = os.path.expanduser("~/.ssh/id_ed25519")

REST = "https://api.runpod.io/v2"
GQL = "https://api.runpod.io/graphql"
#: RunPod official PyTorch image (Ubuntu 24.04, CUDA 12.8.1) validated in docs/runpod_setup.md
DEFAULT_IMAGE = "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404"
#: RTX-class cards Isaac Sim can render on (RT cores); processes per card = 24 GB -> 2, 48 GB -> 4, 96 GB -> 8
RTX_GPUS = {
    "NVIDIA GeForce RTX 4090": 2,
    "NVIDIA GeForce RTX 5090": 3,
    "NVIDIA RTX A5000": 2,
    "NVIDIA RTX A6000": 4,
    "NVIDIA A40": 4,
    "NVIDIA L40": 4,
    "NVIDIA L40S": 4,
    "NVIDIA RTX 6000 Ada Generation": 4,
    "NVIDIA RTX PRO 6000 Blackwell Server Edition": 8,
    "NVIDIA RTX PRO 6000 Blackwell Workstation Edition": 8,
}
DEFAULT_CAP_USD = 2500.0  # owner raised the budget to $2500 on 2026-09-25 17:05 (pilot waves: speed over cost; $1000 on 09-16 17:10, $500 on 09-15); the cap is never enforced by the watchdog (owner 09-15 21:27)
RECHARGE_MIN_JUMP_USD = 20.0


# ----------------------------------------------------------------------------------------------- credentials

def load_env() -> Dict[str, str]:
    env: Dict[str, str] = {}
    if os.path.exists(ENV_PATH):
        with open(ENV_PATH) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip().strip('"').strip("'")
    for k in ("RUNPOD_API_KEY", "HF_TOKEN", "GH_TOKEN"):
        if k in os.environ:
            env[k] = os.environ[k]
    if "RUNPOD_API_KEY" not in env:
        sys.exit("RUNPOD_API_KEY not found in .env or the environment")
    return env


class Api:
    def __init__(self, key: str):
        self.key = key
        self.h = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

    # -- REST v2
    def _rest(self, method: str, path: str, **kw) -> Any:
        r = requests.request(method, REST + path, headers=self.h, timeout=60, **kw)
        if r.status_code == 429:
            time.sleep(int(r.headers.get("Retry-After", "5")))
            r = requests.request(method, REST + path, headers=self.h, timeout=60, **kw)
        if r.status_code >= 400:
            raise RuntimeError(f"{method} {path} -> {r.status_code}: {r.text[:400]}")
        return None if r.status_code == 204 or not r.text else r.json()

    def list_pods(self) -> List[dict]:
        return self._rest("GET", "/pods").get("pods", [])

    def get_pod(self, pod_id: str) -> dict:
        return self._rest("GET", f"/pods/{pod_id}")

    def create_pod(self, name: str, gpu: str, cloud: str = "SECURE", disk: int = 150, env: Optional[Dict[str, str]] = None,
                   image: str = DEFAULT_IMAGE, gpu_count: int = 1, datacenters: Optional[List[str]] = None,
                   cuda_versions: Optional[List[str]] = None) -> dict:
        gpu_cfg: Dict[str, Any] = {"id": gpu, "count": gpu_count}
        if cuda_versions:  # host driver filter: Isaac Sim 5.1 segfaults at startup on hosts with drivers 595 / 610 (CUDA >= 13.1)
            gpu_cfg["allowedCudaVersions"] = list(cuda_versions)
        body = {
            "name": name,
            "image": image,
            "cloud": cloud,
            "gpu": gpu_cfg,
            "disk": disk,
            "ports": ["22/tcp"],
            "env": env or {},
            "startSsh": True,
        }
        if datacenters:
            body["dataCenterIds"] = datacenters
        return self._rest("POST", "/pods", json=body)

    def terminate_pod(self, pod_id: str):
        return self._rest("DELETE", f"/pods/{pod_id}")

    # -- GraphQL (balance, runtime ports, stock)
    def gql(self, query: str) -> dict:
        r = requests.post(GQL, params={"api_key": self.key}, json={"query": query}, timeout=60)
        r.raise_for_status()
        d = r.json()
        if "errors" in d:
            raise RuntimeError(json.dumps(d["errors"])[:400])
        return d["data"]

    def myself(self) -> dict:
        return self.gql("{ myself { clientBalance currentSpendPerHr spendLimit pods { id name desiredStatus costPerHr uptimeSeconds "
                        "machine { gpuDisplayName } runtime { uptimeInSeconds ports { ip isIpPublic privatePort publicPort type } "
                        "gpus { id gpuUtilPercent memoryUtilPercent } } } } }")["myself"]

    def stock(self) -> List[dict]:
        return self.gql("{ gpuTypes { id displayName memoryInGb securePrice communityPrice lowestPrice(input:{gpuCount:1}) "
                        "{ minimumBidPrice uninterruptablePrice stockStatus } } }")["gpuTypes"]


# ----------------------------------------------------------------------------------------------- state

def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse(ts: str) -> dt.datetime:
    return dt.datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=dt.timezone.utc)


def load_fleet() -> Dict[str, Any]:
    if os.path.exists(FLEET_PATH):
        with open(FLEET_PATH) as f:
            return json.load(f)
    return {"pods": {}, "initial_balance": None, "recharges": []}


def save_fleet(fleet: Dict[str, Any]):
    os.makedirs(STATE_DIR, exist_ok=True)
    with open(FLEET_PATH, "w") as f:
        json.dump(fleet, f, indent=2)


def log_billing(line: str):
    os.makedirs(STATE_DIR, exist_ok=True)
    with open(BILLING_LOG, "a") as f:
        f.write(f"{_now()} {line}\n")


def record_balance(api: Api, fleet: Dict[str, Any], tag: str = "") -> dict:
    """Sample the balance; a jump of >= RECHARGE_MIN_JUMP_USD since the last sample is a recharge (card charged)."""
    me = api.myself()
    bal = float(me["clientBalance"])
    if fleet.get("initial_balance") is None:
        fleet["initial_balance"] = bal
    last = fleet.get("last_balance")
    if last is not None and bal - last >= RECHARGE_MIN_JUMP_USD:
        ev = {"at": _now(), "from": last, "to": bal, "amount": round(bal - last, 2)}
        fleet.setdefault("recharges", []).append(ev)
        log_billing(f"RECHARGE ${ev['amount']:.2f} (balance {last:.2f} -> {bal:.2f})")
        print(f"*** RECHARGE detected: card charged ${ev['amount']:.2f} (balance {last:.2f} -> {bal:.2f}) ***")
    fleet["last_balance"] = bal
    spent = spent_so_far(fleet)
    log_billing(f"balance ${bal:.2f} spend/h ${float(me['currentSpendPerHr']):.3f} running={sum(1 for p in me['pods'] if p['desiredStatus']=='RUNNING')} "
                f"spent_total ${spent:.2f} {tag}".rstrip())
    save_fleet(fleet)
    return me


def spent_so_far(fleet: Dict[str, Any]) -> float:
    if fleet.get("initial_balance") is None or fleet.get("last_balance") is None:
        return 0.0
    return fleet["initial_balance"] + sum(r["amount"] for r in fleet.get("recharges", [])) - fleet["last_balance"]


# ----------------------------------------------------------------------------------------------- ssh helpers

def pod_ssh(api: Api, pod_id: str) -> Optional[tuple]:
    for p in api.myself()["pods"]:
        if p["id"] == pod_id and p.get("runtime") and p["runtime"].get("ports"):
            for port in p["runtime"]["ports"]:
                if port["privatePort"] == 22 and port["type"] == "tcp" and port.get("isIpPublic"):
                    return port["ip"], int(port["publicPort"])
    return None


def ssh_base(ip: str, port: int) -> List[str]:
    return ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null", "-o", "LogLevel=ERROR",
            "-o", "ConnectTimeout=20", "-o", "ServerAliveInterval=30", "-i", SSH_KEY, "-p", str(port), f"root@{ip}"]


def ssh_run(ip: str, port: int, cmd: str, capture: bool = False, timeout: Optional[int] = None, stdin_text: Optional[str] = None) -> subprocess.CompletedProcess:
    """Run ``cmd`` on the pod. Never put secrets in ``cmd`` (it shows up in ``ps`` and in tracebacks); pass them through
    ``stdin_text`` instead. A timeout is reported without echoing the command line."""
    try:
        return subprocess.run(ssh_base(ip, port) + [cmd], text=True, capture_output=capture, timeout=timeout,
                              input=stdin_text, stdin=None if stdin_text is not None else subprocess.DEVNULL)
    except subprocess.TimeoutExpired:
        sys.exit(f"ssh to {ip}:{port} timed out after {timeout}s (command not shown)")


def wait_ssh(api: Api, pod_id: str, timeout_s: int = 900) -> tuple:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        ep = pod_ssh(api, pod_id)
        if ep is not None:
            r = ssh_run(*ep, "echo ok", capture=True, timeout=40)
            if r.returncode == 0 and "ok" in r.stdout:
                return ep
        time.sleep(15)
    raise TimeoutError(f"pod {pod_id}: sshd not reachable after {timeout_s}s")


def rsync(src: str, dst: str, ip: str, port: int, to_pod: bool):
    ssh_cmd = " ".join(shlex.quote(x) for x in ssh_base(ip, port)[:-1])  # everything except root@ip
    remote = f"root@{ip}:{dst if to_pod else src}"
    args = ["rsync", "-az", "--info=progress2", "-e", ssh_cmd] + ([src, remote] if to_pod else [remote, dst])
    subprocess.run(args, check=True)


# ----------------------------------------------------------------------------------------------- commands

def cmd_balance(api: Api, args):
    fleet = load_fleet()
    me = record_balance(api, fleet, tag="(balance cmd)")
    print(f"balance ${float(me['clientBalance']):.2f}   fleet spend/h ${float(me['currentSpendPerHr']):.3f}   "
          f"spent so far ${spent_so_far(fleet):.2f} of cap ${args.cap:.0f}   recharges={len(fleet.get('recharges', []))}")
    for p in me["pods"]:
        print(f"  {p['id']} {p['name']:24s} {p['desiredStatus']:8s} ${float(p['costPerHr']):.2f}/h {p['machine']['gpuDisplayName']} up {int(p.get('uptimeSeconds') or 0)//60} min")


def cmd_stock(api: Api, args):
    print(f"{'GPU':50s} {'mem':>4} {'secure$':>8} {'comm$':>7} {'stock':>7} {'procs':>5}")
    for t in sorted(api.stock(), key=lambda t: (t["memoryInGb"] or 0, t["displayName"])):
        if t["id"] in RTX_GPUS or args.all:
            lp = t["lowestPrice"] or {}
            print(f"{t['id']:50s} {t['memoryInGb'] or 0:>4} {t['securePrice'] or 0:>8} {t['communityPrice'] or 0:>7} {str(lp.get('stockStatus')):>7} {RTX_GPUS.get(t['id'], '-'):>5}")


def projected_fleet_cost(fleet: Dict[str, Any], me: dict) -> float:
    """$ the running fleet will still cost until every pod's TTL (upper bound)."""
    now = dt.datetime.now(dt.timezone.utc)
    cost = 0.0
    for p in me["pods"]:
        ours = fleet["pods"].get(p["id"])
        if ours is None or p["desiredStatus"] != "RUNNING":
            continue
        end = _parse(ours["created_at"]) + dt.timedelta(hours=ours["ttl_h"])
        cost += max(0.0, (end - now).total_seconds() / 3600.0) * float(p["costPerHr"])
    return cost


def cmd_create(api: Api, args):
    fleet = load_fleet()
    me = record_balance(api, fleet, tag="(before create)")
    price = None
    for t in api.stock():
        if t["id"] == args.gpu:
            price = float(t["securePrice"] if args.cloud == "SECURE" else t["communityPrice"] or 0)
    if price is None:
        sys.exit(f"unknown GPU id {args.gpu!r}; see `stock`")
    projected = spent_so_far(fleet) + projected_fleet_cost(fleet, me) + price * args.ttl_h * args.count
    if projected > args.cap:
        sys.exit(f"refusing: projected spend ${projected:.2f} exceeds cap ${args.cap:.2f}")
    env = load_env()
    pod_env = {"OMNIGIBSON_HEADLESS": "1", "OMNI_KIT_ACCEPT_EULA": "YES", "ISG_TTL_SECONDS": str(int(args.ttl_h * 3600))}
    created = []
    for i in range(args.count):
        name = args.name if args.count == 1 else f"{args.name}-{i}"
        try:
            pod = api.create_pod(name, args.gpu, cloud=args.cloud, disk=args.disk, env=pod_env, datacenters=args.dc, cuda_versions=args.cuda)
        except RuntimeError as e:
            print(f"create {name} on {args.gpu} ({args.cloud}) failed: {e}")
            continue
        fleet["pods"][pod["id"]] = {"name": name, "gpu": args.gpu, "cloud": args.cloud, "created_at": _now(), "ttl_h": args.ttl_h,
                                    "purpose": args.purpose, "price_per_h": price, "procs": RTX_GPUS.get(args.gpu)}
        save_fleet(fleet)
        log_billing(f"CREATE {pod['id']} {name} {args.gpu} {args.cloud} ${price}/h ttl {args.ttl_h}h purpose={args.purpose}")
        print(f"created {pod['id']} {name} ({args.gpu}, {args.cloud}, ${price}/h, ttl {args.ttl_h} h)")
        created.append(pod["id"])
    if not created:
        sys.exit(1)


def cmd_list(api: Api, args):
    fleet = load_fleet()
    me = api.myself()
    now = dt.datetime.now(dt.timezone.utc)
    for p in me["pods"]:
        ours = fleet["pods"].get(p["id"])
        ttl = ""
        if ours:
            end = _parse(ours["created_at"]) + dt.timedelta(hours=ours["ttl_h"])
            ttl = f"ttl ends in {int((end - now).total_seconds() // 60)} min"
        ep = None
        if p.get("runtime") and p["runtime"].get("ports"):
            for port in p["runtime"]["ports"]:
                if port["privatePort"] == 22 and port["type"] == "tcp":
                    ep = f"{port['ip']}:{port['publicPort']}"
        gpus = (p.get("runtime") or {}).get("gpus") or []
        util = f"gpu {gpus[0]['gpuUtilPercent']}% mem {gpus[0]['memoryUtilPercent']}%" if gpus else ""
        print(f"{p['id']} {p['name']:24s} {p['desiredStatus']:8s} {p['machine']['gpuDisplayName']:28s} ${float(p['costPerHr']):.2f}/h "
              f"ssh={ep} {util} {ttl} {'(ours)' if ours else ''}")
    if not me["pods"]:
        print("no pods")


def cmd_wait_ssh(api: Api, args):
    ip, port = wait_ssh(api, args.pod, timeout_s=args.timeout)
    print(f"{ip} {port}")


def cmd_run(api: Api, args):
    ip, port = wait_ssh(api, args.pod, timeout_s=120)
    r = ssh_run(ip, port, args.command)
    sys.exit(r.returncode)


def cmd_push(api: Api, args):
    ip, port = wait_ssh(api, args.pod, timeout_s=120)
    rsync(args.local, args.remote, ip, port, to_pod=True)


def cmd_pull(api: Api, args):
    ip, port = wait_ssh(api, args.pod, timeout_s=120)
    rsync(args.remote, args.local, ip, port, to_pod=False)


def cmd_bootstrap(api: Api, args):
    """Copy scripts/runpod/*.sh to the pod and run bootstrap.sh detached (install ~13 min, then the optional job)."""
    env = load_env()
    fleet = load_fleet()
    ours = fleet["pods"].get(args.pod, {})
    ip, port = wait_ssh(api, args.pod, timeout_s=args.timeout)
    ssh_run(ip, port, "mkdir -p /root/isg_boot", timeout=60)
    here = os.path.dirname(os.path.abspath(__file__))
    for fn in ("bootstrap.sh", "bench.sh"):
        subprocess.run(["scp", "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null", "-o", "LogLevel=ERROR", "-i", SSH_KEY,
                        "-P", str(port), os.path.join(here, fn), f"root@{ip}:/root/isg_boot/{fn}"], check=True)
    ttl = int(float(ours.get("ttl_h", args.ttl_h)) * 3600)
    # secrets go to a root-only file over stdin (never on a command line); bootstrap.sh sources and shreds it
    secrets = (f"GH_TOKEN={shlex.quote(env.get('GH_TOKEN', ''))}\nHF_TOKEN={shlex.quote(env.get('HF_TOKEN', ''))}\n"
               f"ISG_TTL_SECONDS={ttl}\nISG_JOB={shlex.quote(args.job or '')}\nISG_REPO_REF={shlex.quote(args.ref)}\n")
    r = ssh_run(ip, port, "umask 077 && cat > /root/isg_boot/env.sh", timeout=60, stdin_text=secrets)
    if r.returncode != 0:
        sys.exit("writing the pod env file failed")
    # setsid + all three streams redirected, otherwise ssh waits for the detached process; flock in bootstrap.sh makes
    # a repeated invocation a no-op
    remote = "chmod +x /root/isg_boot/*.sh && cd /root && setsid nohup /root/isg_boot/bootstrap.sh > /root/isg_bootstrap.log 2>&1 < /dev/null & disown"
    try:  # sshd keeps the session open for a while after the detached launch; the lock file is the confirmation
        subprocess.run(ssh_base(ip, port) + [remote], text=True, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=15)
    except subprocess.TimeoutExpired:
        pass
    time.sleep(3)
    r = ssh_run(ip, port, "test -f /root/isg_boot/.lock && head -1 /root/isg_bootstrap.log", capture=True, timeout=60)
    ok = r.returncode == 0 and bool((r.stdout or "").strip())
    print(f"bootstrap {'started' if ok else 'NOT CONFIRMED'} on {args.pod} ({ip}:{port}): {(r.stdout or '').strip()}; log /root/isg_bootstrap.log; markers BOOTSTRAP_DONE / JOB_DONE; job={args.job or '-'}")
    sys.exit(0 if ok else 1)


def cmd_terminate(api: Api, args):
    fleet = load_fleet()
    me = api.myself()
    targets = []
    if args.all:
        targets = [p["id"] for p in me["pods"]]
    elif args.ours:
        targets = [p["id"] for p in me["pods"] if p["id"] in fleet["pods"]]
    elif args.pod:
        targets = [args.pod]
    if not targets:
        print("nothing to terminate")
        return
    for pid in targets:
        try:
            api.terminate_pod(pid)
            log_billing(f"TERMINATE {pid} {fleet['pods'].get(pid, {}).get('name', '?')} reason={args.reason}")
            print(f"terminated {pid}")
        except RuntimeError as e:
            print(f"terminate {pid} failed: {e}")
        if pid in fleet["pods"]:
            fleet["pods"][pid]["terminated_at"] = _now()
    save_fleet(fleet)
    record_balance(api, fleet, tag="(after terminate)")


def cmd_watchdog(api: Api, args):
    """Enforce TTLs / idleness and log the balance until stopped (or until no pods remain and --exit-when-empty)."""
    print(f"watchdog: interval {args.interval}s, idle-terminate after {args.idle_min} min of <{args.idle_util}% GPU (pods marked ours only); "
          f"spend cap {'ENFORCED at $%.0f' % args.cap if args.enforce_cap else 'not enforced (logging only)'}")
    idle_since: Dict[str, float] = {}
    while True:
        fleet = load_fleet()
        try:
            me = record_balance(api, fleet, tag="(watchdog)")
        except Exception as e:  # network hiccups must not kill the guard
            print(f"watchdog: balance query failed: {e}")
            time.sleep(args.interval)
            continue
        now = dt.datetime.now(dt.timezone.utc)
        running = 0
        for p in me["pods"]:
            ours = fleet["pods"].get(p["id"])
            if p["desiredStatus"] == "RUNNING":
                running += 1
            if ours is None:
                continue
            end = _parse(ours["created_at"]) + dt.timedelta(hours=ours["ttl_h"])
            reason = None
            if now >= end:
                reason = "ttl"
            # idle = the pod reports JOB_DONE and no sampler / bench / install process is alive (the API's GPU
            # utilisation is useless here: the sampler is CPU-bound with short GPU bursts and reads 0 % most of the time)
            up = int(p.get("uptimeSeconds") or 0)
            ep = pod_ssh(api, p["id"]) if up > args.idle_grace_min * 60 else None
            if ep is not None:
                try:
                    r = ssh_run(*ep, "if [ -f /root/JOB_DONE ] && ! pgrep -f 'sampling\\.|isg_boot/bench\\.s[h]|setup\\.s[h]|bootstrap\\.s[h]' >/dev/null; then echo IDLE; else echo BUSY; fi",
                                capture=True, timeout=45)
                    state = (r.stdout or "").strip()
                except SystemExit:
                    state = "UNKNOWN"
                if state == "IDLE":
                    idle_since.setdefault(p["id"], time.time())
                    if time.time() - idle_since[p["id"]] > args.idle_min * 60:
                        reason = reason or "idle"
                else:
                    idle_since.pop(p["id"], None)
            # spend-cap termination is opt-in (`--enforce-cap`): on 2026-09-15 the $200 default cap killed a whole
            # sampling fleet mid-run (owner decision 21:27 PDT: the watchdog never terminates for budget reasons;
            # the balance / RECHARGE lines in billing.log stay the budget signal, `create` still refuses over the cap)
            if args.enforce_cap and spent_so_far(fleet) > args.cap:
                reason = reason or "cap"
            if reason:
                try:
                    api.terminate_pod(p["id"])
                    ours["terminated_at"] = _now()
                    log_billing(f"TERMINATE {p['id']} {ours['name']} reason={reason}")
                    print(f"watchdog: terminated {p['id']} {ours['name']} ({reason})")
                except RuntimeError as e:
                    print(f"watchdog: terminate {p['id']} failed: {e}")
        save_fleet(fleet)
        if args.exit_when_empty and running == 0:
            print("watchdog: no running pods; exiting")
            return
        time.sleep(args.interval)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cap", type=float, default=DEFAULT_CAP_USD, help="hard spend cap in USD for this project")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("balance")
    s = sub.add_parser("stock"); s.add_argument("--all", action="store_true")
    s = sub.add_parser("create")
    s.add_argument("--gpu", required=True); s.add_argument("--name", required=True); s.add_argument("--count", type=int, default=1)
    s.add_argument("--cloud", default="SECURE", choices=["SECURE", "COMMUNITY"]); s.add_argument("--disk", type=int, default=150)
    s.add_argument("--ttl-h", type=float, default=2.0); s.add_argument("--purpose", default=""); s.add_argument("--dc", nargs="*", default=None)
    s.add_argument("--cuda", nargs="*", default=["12.8", "12.9", "13.0"], help="host CUDA versions allowed (driver filter; [] = any)")
    sub.add_parser("list")
    s = sub.add_parser("wait-ssh"); s.add_argument("pod"); s.add_argument("--timeout", type=int, default=900)
    s = sub.add_parser("run"); s.add_argument("pod"); s.add_argument("command")
    s = sub.add_parser("push"); s.add_argument("pod"); s.add_argument("local"); s.add_argument("remote")
    s = sub.add_parser("pull"); s.add_argument("pod"); s.add_argument("remote"); s.add_argument("local")
    s = sub.add_parser("bootstrap"); s.add_argument("pod"); s.add_argument("--job", default=None, help="e.g. bench:2 (K concurrent sampler processes)")
    s.add_argument("--ttl-h", type=float, default=2.0); s.add_argument("--timeout", type=int, default=900); s.add_argument("--ref", default="main", help="git ref of kmy17518/iSpatialGym to check out")
    s = sub.add_parser("terminate"); s.add_argument("pod", nargs="?"); s.add_argument("--all", action="store_true"); s.add_argument("--ours", action="store_true")
    s.add_argument("--reason", default="manual")
    s = sub.add_parser("watchdog"); s.add_argument("--interval", type=int, default=300); s.add_argument("--idle-min", type=int, default=20)
    s.add_argument("--idle-util", type=float, default=3.0); s.add_argument("--idle-grace-min", type=int, default=45, help="no idle check before the install has had time to finish")
    s.add_argument("--exit-when-empty", action="store_true")
    s.add_argument("--enforce-cap", action="store_true", help="terminate pods when spent_total exceeds --cap (OFF by default since 2026-09-15: the cap killed a fleet mid-run)")
    args = ap.parse_args()
    api = Api(load_env()["RUNPOD_API_KEY"])
    {"balance": cmd_balance, "stock": cmd_stock, "create": cmd_create, "list": cmd_list, "wait-ssh": cmd_wait_ssh, "run": cmd_run,
     "push": cmd_push, "pull": cmd_pull, "bootstrap": cmd_bootstrap, "terminate": cmd_terminate, "watchdog": cmd_watchdog}[args.cmd](api, args)


if __name__ == "__main__":
    main()
