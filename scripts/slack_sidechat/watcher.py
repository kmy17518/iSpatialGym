#!/usr/bin/env python3
"""Slack side-chat watcher: answers the owner's ``Q:`` thread replies in the coordinator's Slack channel with a Cursor SDK
agent that has read access to the repo, the pilot artefacts and the main chat's transcripts — one agent per Slack thread,
resumed on every follow-up, so a thread behaves like an IDE side chat — and hands finished discussions to the main
chat's coordinator through ``MAIN:`` posts.

    python scripts/slack_sidechat/watcher.py [--channel C0C2D1X2V8E] [--poll 20] [--once] [--dry-run]

Secrets (environment or ``<repo>/.env``): ``SLACK_BOT_TOKEN`` (bot token of a Slack app installed in the owner's
workspace, scopes ``channels:history groups:history chat:write users:read``; the bot must be invited to the channel),
``CURSOR_API_KEY`` (Cursor SDK). Optional: ``SIDECHAT_MODEL`` (default ``composer-2.5``), ``SLACK_OWNER_ID``
(default ``U0C28KXH5J7``).

Thread reply grammar (owner messages only; everything else is the coordinator's business and is ignored here):
  ``Q: <question>``      -> answered in the thread as ``[SIDE] …`` by the thread's agent (follow-ups keep the context)
  ``HANDOFF`` / ``MAIN: draft``  -> the thread's agent composes a coordinator prompt from the discussion -> ``DRAFT: …``
  ``MAIN: send``         -> the last draft is re-posted top-level as ``MAIN: <prompt>`` for the coordinator's poll
  ``MAIN: <text>``       -> left alone: the coordinator picks it up itself as an owner instruction

State (agent id per thread, last seen timestamps, drafts) lives in ``scripts/slack_sidechat/state.json`` (git-ignored).
The agent is read-only by instruction: it may read files, run ``git log``/``git show``, and run read-only ``rp.py run``
commands on the pods; it never edits, commits, or changes pod / fleet state.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
STATE_PATH = os.path.join(HERE, "state.json")
TRANSCRIPTS = os.path.expanduser("~/.cursor/projects/home-minyeongk-Desktop-projects-ispatialgym/agent-transcripts")
MAIN_SESSION = "2e9b454d-c20e-4c10-bc0b-41b48999e555"
DEFAULT_CHANNEL = "C0C2D1X2V8E"  # #cursor-demos-chat (private; ids only, name search does not see private channels)
DEFAULT_OWNER = "U0C28KXH5J7"

SYSTEM = f"""You are the *side-chat agent* of the iSpatialGym pilot: you answer the project owner's questions posted as
Slack thread replies in #cursor-demos-chat. You are NOT the coordinator and you never act on the project.

Hard rules:
- Read-only. You may read any file under {REPO}, run `git log` / `git show` / `git diff` (read-only), and run read-only
  shell commands on the RunPod pods through `python scripts/runpod/rp.py run <pod_id> '<command>'` (ls, cat, tail, ps,
  nvidia-smi, pgrep). Never edit, create, delete or commit files; never start, stop, kill, dispatch or reassign anything
  on the pods or locally; never change TODO.md or the board. If the owner asks you to do something, explain that the
  coordinator does it and offer to draft a `MAIN:` hand-off instead.
- Sources of truth (read before answering): runlogs/pilot_20260915/first_demo_board.md (per-subtask state),
  runlogs/pilot_20260915/first_demo/ledger.jsonl (every first-demo run: alias, commit, sampling outcome, demo verdict,
  meta path), runlogs/pilot_20260915/fix_log.jsonl, runlogs/pilot_20260915/pending_decisions.json (if present),
  runlogs/pilot_20260915/status.md (if present), runlogs/pilot_20260915/paused_for_resume.md, TODO.md (decision register
  + activity log, newest last), the pod role files runlogs/runpod/smoke/pods.txt (remote_run pool) and
  runlogs/runpod/pilot/pods_bands.txt (band fleet), runlogs/runpod/fleet.json (pod names/ids).
- The main chat session (the coordinator) is fully readable for you: its transcript is
  {TRANSCRIPTS}/{MAIN_SESSION}.jsonl (one JSON object per line: role user/assistant, message.content blocks) and its
  subagents' transcripts are in {TRANSCRIPTS}/{MAIN_SESSION}/subagents/<id>.jsonl. Search them with grep for names,
  aliases, commits or error strings instead of reading them linearly (they are large).
- Live pod state: the pool pods run one job each under a local lock (`runlogs/runpod/smoke/locks/<pod>.lock`); the band
  pods run a queue runner (`/root/samp/queue`, `/root/samp/current.txt`, `/root/samp/done.txt`). Use rp.py run only when
  the artefacts cannot answer (ETA of a running job, what a pod is doing right now).
- Answer in Slack mrkdwn, concise (<= 25 lines unless asked for detail), lead with the answer, then the evidence with
  file paths / commit ids / pod ids. Say plainly when something is unknown or when the owner must decide.
- Keep the thread's context: follow-up questions refer to earlier turns of the same thread."""

HANDOFF_PROMPT = """The owner wants to hand this thread's conclusions to the coordinator (the main chat). Compose a
coordinator prompt from the thread: the decision(s) taken or the request, the concrete subtasks/aliases/commits involved,
the acceptance criterion, and what to log (TODO.md decision register + activity, board, fix log). Owner voice, imperative,
<= 20 lines, no preamble, no questions back. Start it with a one-line title. Output only the prompt text."""


# ---------------------------------------------------------------------------------------------------------------------
def load_env():
    env = dict(os.environ)
    p = os.path.join(REPO, ".env")
    if os.path.exists(p):
        for line in open(p):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    return env


class Slack:
    def __init__(self, token: str):
        self.token = token
        self.bot_user = None

    def api(self, method: str, **params):
        r = requests.post(f"https://slack.com/api/{method}", headers={"Authorization": f"Bearer {self.token}"}, data=params, timeout=30)
        d = r.json()
        if not d.get("ok"):
            raise RuntimeError(f"slack {method}: {d.get('error')}")
        return d

    def whoami(self):
        self.bot_user = self.api("auth.test")["user_id"]
        return self.bot_user

    def history(self, channel: str, oldest: str, limit: int = 200):
        return self.api("conversations.history", channel=channel, oldest=oldest, limit=limit, inclusive="false").get("messages", [])

    def replies(self, channel: str, ts: str, oldest: str = "0"):
        msgs = self.api("conversations.replies", channel=channel, ts=ts, limit=200).get("messages", [])
        return [m for m in msgs if m.get("ts") != ts and float(m.get("ts", 0)) > float(oldest)]

    def post(self, channel: str, text: str, thread_ts: str | None = None):
        params = {"channel": channel, "text": text[:39000], "unfurl_links": "false"}
        if thread_ts:
            params["thread_ts"] = thread_ts
        return self.api("chat.postMessage", **params)["ts"]


# ---------------------------------------------------------------------------------------------------------------------
def load_state():
    if os.path.exists(STATE_PATH):
        return json.load(open(STATE_PATH))
    return {"last_ts": None, "threads": {}}


def save_state(state):
    tmp = STATE_PATH + ".tmp"
    json.dump(state, open(tmp, "w"), indent=1)
    os.replace(tmp, STATE_PATH)


def thread_context(slack: Slack, channel: str, ts: str) -> str:
    parent = slack.api("conversations.replies", channel=channel, ts=ts, limit=200).get("messages", [])
    lines = []
    for m in parent:
        who = "coordinator/owner" if m.get("user") == DEFAULT_OWNER else ("side-agent" if m.get("user") == slack.bot_user else m.get("user", "?"))
        lines.append(f"[{who}] {m.get('text', '')}")
    return "\n".join(lines)


# Built-in tools removed from the side-chat agent (public SDK names = proto field names minus "ToolCall"): no file
# edits/deletes, no subagents, no branch switching. `shell` stays (git log, grep of transcripts, rp.py run) and is
# read-only by instruction. Unknown names make the SDK raise with the valid list; we then retry unrestricted and warn.
DISALLOWED_TOOLS = ["edit", "delete", "task", "setActiveBranch", "switchMode"]


def run_agent(env, agent_id, prompt, cwd=REPO):
    """Create or resume the thread's agent, send the prompt, return (agent_id, text)."""
    from cursor_sdk import Agent, AgentOptions, CursorAgentError, LocalAgentOptions  # lazy: needs the bridge + API key

    api_key = env["CURSOR_API_KEY"]
    model = env.get("SIDECHAT_MODEL", "composer-2.5")
    disallowed = DISALLOWED_TOOLS
    for attempt in range(2):
        try:
            if agent_id:
                opts = AgentOptions(api_key=api_key, disallowed_tools=disallowed)
                with Agent.resume(agent_id, opts) as agent:
                    return agent.agent_id, agent.send(prompt).text()
            opts = AgentOptions(model=model, api_key=api_key, local=LocalAgentOptions(cwd=cwd), disallowed_tools=disallowed)
            with Agent.create(opts) as agent:
                return agent.agent_id, agent.send(SYSTEM + "\n\n" + prompt).text()
        except CursorAgentError as e:
            if attempt == 0 and "Unknown tool name" in str(e):
                print(f"[watcher] tool restriction rejected, retrying unrestricted: {e}", flush=True)
                disallowed = None
                continue
            raise
    raise RuntimeError("unreachable")


# ---------------------------------------------------------------------------------------------------------------------
def handle_reply(slack: Slack, env, state, channel: str, thread_ts: str, msg: dict, dry_run: bool):
    text = (msg.get("text") or "").strip()
    trec = state["threads"].setdefault(thread_ts, {"agent_id": None, "last_reply_ts": "0", "draft": None})
    low = text.lower()
    if low.startswith("q:"):
        question = text[2:].strip()
        if trec["agent_id"] is None:
            prompt = f"Slack thread so far (parent = the coordinator's post):\n{thread_context(slack, channel, thread_ts)}\n\nOwner question: {question}"
        else:
            prompt = f"Owner follow-up in the same thread: {question}"
        print(f"[watcher] Q in thread {thread_ts}: {question[:80]}", flush=True)
        agent_id, answer = run_agent(env, trec["agent_id"], prompt)
        trec["agent_id"] = agent_id
        out = f"[SIDE] {answer.strip()}"
        if dry_run:
            print(out)
        else:
            slack.post(channel, out, thread_ts=thread_ts)
    elif low.startswith("handoff") or low.startswith("main: draft"):
        print(f"[watcher] HANDOFF in thread {thread_ts}", flush=True)
        prompt = HANDOFF_PROMPT if trec["agent_id"] else (f"Slack thread:\n{thread_context(slack, channel, thread_ts)}\n\n" + HANDOFF_PROMPT)
        agent_id, draft = run_agent(env, trec["agent_id"], prompt)
        trec["agent_id"], trec["draft"] = agent_id, draft.strip()
        out = f"DRAFT: (reply `MAIN: send` to post it top-level for the coordinator, or `HANDOFF` again with changes)\n{trec['draft']}"
        if dry_run:
            print(out)
        else:
            slack.post(channel, out, thread_ts=thread_ts)
    elif low.startswith("main: send"):
        if not trec.get("draft"):
            slack.post(channel, "[SIDE] no draft in this thread yet — reply `HANDOFF` first.", thread_ts=thread_ts)
        else:
            out = f"MAIN: {trec['draft']}\n_(hand-off from thread {thread_ts})_"
            print(f"[watcher] MAIN send from thread {thread_ts}", flush=True)
            if dry_run:
                print(out)
            else:
                ts = slack.post(channel, out)
                slack.post(channel, f"[SIDE] posted top-level as MAIN: (ts {ts}); the coordinator picks it up on its next poll.", thread_ts=thread_ts)
            trec["draft"] = None
    # anything else (D-…, RUN:, MAIN: <text>, notes) belongs to the coordinator
    trec["last_reply_ts"] = max(trec["last_reply_ts"], msg.get("ts", "0"), key=float)


def poll_once(slack: Slack, env, state, channel: str, owner: str, dry_run: bool):
    oldest = state["last_ts"] or f"{time.time() - 600:.6f}"  # first start: only the last 10 minutes
    msgs = sorted(slack.history(channel, oldest=oldest), key=lambda m: float(m["ts"]))
    for m in msgs:
        state["last_ts"] = max(state["last_ts"] or "0", m["ts"], key=float)
        # a new top-level owner message that is itself a question -> treat the message as its own thread
        if m.get("user") == owner and (m.get("text") or "").strip().lower().startswith("q:") and not m.get("thread_ts"):
            handle_reply(slack, env, state, channel, m["ts"], m, dry_run)
        # remember every top-level message as a potential thread
        state["threads"].setdefault(m["ts"], {"agent_id": None, "last_reply_ts": "0", "draft": None})
    # threads with new replies: Slack marks parents with latest_reply; we scan the known threads' recent activity
    recent = slack.api("conversations.history", channel=channel, limit=100).get("messages", [])
    for parent in recent:
        ts = parent.get("ts")
        if not ts or not parent.get("reply_count"):
            continue
        trec = state["threads"].setdefault(ts, {"agent_id": None, "last_reply_ts": "0", "draft": None})
        if float(parent.get("latest_reply", "0")) <= float(trec["last_reply_ts"]):
            continue
        for r in sorted(slack.replies(channel, ts, oldest=trec["last_reply_ts"]), key=lambda x: float(x["ts"])):
            if r.get("user") == owner:
                try:
                    handle_reply(slack, env, state, channel, ts, r, dry_run)
                except Exception as e:  # noqa: BLE001
                    traceback.print_exc()
                    if not dry_run:
                        slack.post(channel, f"[SIDE] error while answering: {type(e).__name__}: {str(e)[:300]}", thread_ts=ts)
                    trec["last_reply_ts"] = max(trec["last_reply_ts"], r["ts"], key=float)
            else:
                trec["last_reply_ts"] = max(trec["last_reply_ts"], r["ts"], key=float)
        save_state(state)
    save_state(state)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--channel", default=DEFAULT_CHANNEL)
    ap.add_argument("--poll", type=int, default=20, help="seconds between polls")
    ap.add_argument("--once", action="store_true", help="one poll, then exit")
    ap.add_argument("--dry-run", action="store_true", help="answer to stdout instead of Slack")
    args = ap.parse_args()
    env = load_env()
    owner = env.get("SLACK_OWNER_ID", DEFAULT_OWNER)
    for k in ("SLACK_BOT_TOKEN", "CURSOR_API_KEY"):
        if not env.get(k):
            sys.exit(f"missing {k} (environment or {REPO}/.env)")
    slack = Slack(env["SLACK_BOT_TOKEN"])
    print(f"[watcher] bot user {slack.whoami()}, channel {args.channel}, owner {owner}, repo {REPO}", flush=True)
    state = load_state()
    while True:
        try:
            poll_once(slack, env, state, args.channel, owner, args.dry_run)
        except Exception:  # noqa: BLE001
            traceback.print_exc()
        if args.once:
            break
        time.sleep(args.poll)


if __name__ == "__main__":
    main()
