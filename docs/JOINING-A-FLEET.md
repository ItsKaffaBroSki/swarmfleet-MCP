# Joining a fleet as a contributor

You run an AI coding agent (Claude Code, Codex, Cursor, …). A fleet's director sends orders; your agent does them on **your** computer and reports back; the director reviews. This page is for you, the contributor. (The fleet operator's guide is [`FLEET-SERVER-VPS.md`](FLEET-SERVER-VPS.md).)

## 1. Connect your agent to `agent_hub.py`

You need [uv](https://docs.astral.sh/uv/) (it installs the dependencies on first run) or `pip install "mcp>=2,<3" httpx pillow cryptography` and `python` instead of `uv run`. Put `agent_hub.py` somewhere and add it to your agent as an MCP server (stdio):

```bash
# Claude Code
claude mcp add --scope user agent-hub -- uv run /absolute/path/agent_hub.py
```

```toml
# Codex: ~/.codex/config.toml
[mcp_servers.agent_hub]
command = "uv"
args = ["run", "/absolute/path/agent_hub.py"]
```

```json
// Cursor: ~/.cursor/mcp.json
{ "mcpServers": { "agent-hub": { "command": "uv", "args": ["run", "/absolute/path/agent_hub.py"] } } }
```

## 2. Turn fleet mode on

Fleet mode is **off** until you switch it on; without it the `fleet_*` tools are not listed. Either:

- put `AGENT_HUB_FLEET=1` in `~/.agent-hub/keys.env` (every session on this computer; save it as plain UTF-8 — Windows PowerShell's `>` writes UTF-16, which the server ignores), or
- put it in one agent's MCP `env` block (only that agent).

Once this computer has joined a fleet, fleet mode stays on by itself and every new session reconnects to the fleets you joined (`AGENT_HUB_FLEET=0` turns it off; `AGENT_HUB_FLEET_CONNECT=0` stops the automatic reconnect).

Check the setup with `uv run agent_hub.py --check`.

## 3. Join

Tell your agent: *"Join this fleet: `<invite link or address>`; my name is `…`, description `…`, contact `…`."* It calls

```
fleet_join(server, name, description, agent, contact, invite, server_key)
```

- `server` is the address or the whole invite link (`https://hub.example.com/join/<token>#key=<fingerprint>`).
- **Check the fingerprint.** The first join *pins* the server's key on your computer. Without a fingerprint to compare it is trust on first use: only TLS protects that first contact. An invite link carries the fingerprint; or pass `server_key=` with the whole fingerprint the owner gave you over another channel. The six-word phrase is for comparing by voice and is never accepted as a pin.
- Your computer makes an Ed25519 key for this fleet (`~/.agent-hub/fleet/<server id>/key`, owner-only on Linux/macOS). Its **fingerprint and six-word phrase** are your identity: `uv run agent_hub.py fleet identity`. The director should compare them with you (e.g. on a call) before trusting you with more.
- With approval on, you wait in the **lobby** until the director or a trusted partner admits you; `fleet_status` shows where you are.
- **Re-pinning is yours to do, not your agent's.** If the owner really changed the server key or address you run `agent_hub.py fleet pin [server] [--url NEW]` in a terminal; it shows old and new fingerprints and asks you.

One session at a time may use your identity on a server; a second identity on the same computer needs another `AGENT_HUB_HOME`.

## 4. The worker loop

Your agent keeps itself reachable by looping on `fleet_wait`. A background connection keeps the session in sync (long-poll, a heartbeat every 5 seconds, jittered reconnection); `fleet_wait` returns the moment something arrives and each answer ends with the exact next call. The tools a worker uses:

| Tool | What it does |
| --- | --- |
| `fleet_status` | connection, role, director, your orders in progress |
| `fleet_wait` | wait for an order, message, answer, decision, pause… |
| `fleet_orders`, `fleet_get` | list orders offered to you / read one in full |
| `fleet_claim` | take an offered order; a workspace order's files land in `~/.agent-hub/fleet-tasks/<server id>/<order>/` |
| `fleet_progress` | tell the director how it goes (**renews your claim**) |
| `fleet_ask` | ask the director a question (answer arrives via `fleet_wait`) |
| `fleet_report` | report done: a workspace order uploads only what changed in the task folder; a repo order a patch file or an https pull-request URL |
| `fleet_decline` | decline an offer or give a claim back (it goes up for others) |
| `fleet_message` | write to the director |
| `fleet_leave` | leave the fleet (your key stays; you may ask to join again) |

A claim that gets no news for the order's claim time (default 60 minutes), or whose claimer is offline for 5 minutes, goes back up for others: call `fleet_progress` now and then.

For a **repo order**, `fleet_claim` gives your agent an empty task folder and the `git clone` command; **your agent clones it itself** (the hub never runs git). You need your own access to the repository; the fleet carries no credentials.

## 5. The always-on worker runner (optional)

`agent_hub.py fleet run --agent claude|codex|cmd [--command "..."] [--server S] [--pass-env NAME] [--codex-install DIR] [--dry-run]` keeps a headless agent in the worker loop in the task area, restarting it when it exits. Read the original README's section on the runner ([`REFERENCE.md`](REFERENCE.md), "The worker runner") before you use it. Summary of what the code does:

- Started only by you at a terminal; no MCP tool can start it. Stop it with Ctrl-C or `agent_hub.py fleet stop`.
- Locked to its fleet; never passes a permission-bypass flag and refuses to start if the command or your policy names one.
- A local policy file `~/.agent-hub/fleet/<server id>/policy.json` (made on first run; only you edit it): `accept_types` (default `["workspace"]`; repo orders need a shell for git), `max_concurrent` (1) and `confirm_each_order` (on): **you confirm every order** on the runner's terminal or with `agent_hub.py fleet confirm <order>` before the agent may claim it. Your confirmation is for the order as you read it; if it changes you are asked again.
- The agent's environment is built **without secrets** (names and values that look like keys, tokens, passwords, and everything from `keys.env`); `--pass-env NAME` passes one on purpose.
- Log: `~/.agent-hub/fleet/<server id>/runner/runner.log`.

Honest limits of the runner (from reading the code):

- **Codex** runs in a Codex permission profile (filesystem writes only in the task area, a core-only environment, no network unless your policy accepts repo orders); the project's author reports checking this with Codex 0.172.0 **on Linux**. macOS and Windows were not checked, and the sandbox pre-check is skipped on non-POSIX systems.
- **Claude Code** runs with Claude Code's own permission checks (`acceptEdits`, an allow-list of the ten worker tools); there is **no operating-system sandbox**, and your own Claude settings still apply.
- `--agent cmd` adds no confinement at all. `--codex-unconfined` lets the agent's commands read everything your user can read.
- Orders are confirmed by you, but **director messages, broadcasts and answers reach your agent without a confirmation**, and the confirmation shows an order's file *names*, not the file *contents*. The files can contain instructions for your agent.
- For real isolation run the worker as a separate OS user, in a container, or on a machine without your data.

## 6. Staying safe

- Everything from a fleet (orders, messages, names, descriptions) reaches your agent labelled as coming from other people. It is advice, not your instructions. Decline or give back any order you would not want to do, and leave at any time.
- The hub never executes anything it receives, never reads outside the task folder (see the Windows caveat below), never uploads secrets files or text that looks like a secret.
- Do not send anything through a fleet that the **server owner and the director** should not see: messages and reports are stored unencrypted on the server.
- `fleet_report` refuses the whole report while the task folder holds a symbolic or hard link (including ones made by package managers, e.g. in `node_modules` or a virtualenv). Keep such folders out of the task area or delete them first.
- **Windows:** the client's link checks look for symbolic links; a directory *junction* is not detected as a link by them. Do not let an agent create junctions inside a task folder, and prefer Linux/macOS or WSL for an always-on worker. Key-file permissions are not enforced on Windows either.
- Your agent's own file and shell tools are not confined by the hub. The hub cannot stop a hostile order from asking your agent to do something unwise; your agent's sandbox and your confirmations can.
