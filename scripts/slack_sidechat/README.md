# Slack side-chat watcher

A local process that turns the owner's `Q:` thread replies in `#cursor-demos-chat` into answers from a Cursor SDK agent
with read access to the repo, the pilot artefacts (`runlogs/pilot_20260915/*`) and the **main chat's transcripts**
(`~/.cursor/projects/…/agent-transcripts/2e9b454d-…`). One agent per Slack thread, resumed on every follow-up, so a thread
behaves like an IDE side chat; a finished discussion is handed to the coordinator with `HANDOFF` → `DRAFT:` → `MAIN: send`.

## One-time setup

1. **Slack app** (in the personal workspace, once): api.slack.com → Create App (from scratch) → *OAuth & Permissions* →
   bot token scopes `channels:history`, `groups:history`, `chat:write`, `users:read` → Install to workspace → copy the
   `xoxb-…` bot token. In Slack: `/invite @<app name>` into `#cursor-demos-chat`.
2. **Secrets** in `<repo>/.env` (already git-ignored; holds the RunPod key):
   ```
   SLACK_BOT_TOKEN=xoxb-…
   CURSOR_API_KEY=cursor_…          # cursor.com/dashboard → Integrations → API keys
   SIDECHAT_MODEL=composer-2.5      # optional
   ```
3. **Python**: `pip install cursor-sdk` in the `ispatialgym` env (`requests` is already there).

## Run

```
source ~/miniconda3/etc/profile.d/conda.sh && conda activate ispatialgym
nohup python scripts/slack_sidechat/watcher.py > /tmp/slack_sidechat.log 2>&1 &
```
`--once --dry-run` for a first test (answers go to stdout, nothing is posted).

## Message grammar (owner replies inside a thread)

| reply | effect |
|---|---|
| `Q: <question>` | the thread's agent answers as `[SIDE] …`; follow-up `Q:` in the same thread keep the context |
| `HANDOFF` or `MAIN: draft` | the agent composes a coordinator prompt from the thread → `DRAFT: …` |
| `MAIN: send` | the last draft is re-posted **top-level** as `MAIN: <prompt>` (the coordinator polls it) |
| `MAIN: <text>` | untouched — the coordinator treats it as an owner instruction |
| `D-0xx: …`, `RUN: …`, notes | untouched — coordinator business |

The agent is read-only by instruction (files, `git log`, read-only `rp.py run` on the pods); it never edits, commits or
changes pod state. Its answers carry file paths, commit ids and pod ids as evidence.

## Coordinator protocol addendum (send to the main chat)

- `Q:` replies are answered by the side-chat watcher, not by you; ignore messages starting with `Q:`, `[SIDE]`, `DRAFT:`.
- A message starting with `MAIN:` (top-level or in a thread) from the owner (or the watcher's hand-off, which ends with
  `_(hand-off from thread …)_`) is an owner instruction: act on it as if typed in the chat, acknowledge in the same thread
  with `[MAIN] received — <one-line summary>`, and log it in TODO.md.
