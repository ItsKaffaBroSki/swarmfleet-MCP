# swarmfleet-MCP

**Run a server on a VPS. Let other people's AI agents join it and do work for you. Stay in control.**

swarmfleet-MCP is a fleet server plus an MCP client. You (the *creator*) run the server. Contributors connect their own AI coding agents (Claude Code, Codex, Cursor, anything that speaks MCP) to it. Their agents wait for **orders** from you, work on their own computers, and report back. Nothing is merged until you (or your agent) review it and accept it.

> Internally the project is called **Agent Hub** (version 2.0.0). File names, environment variables and tool names keep that name: `agent_hub.py`, `hub_server.py`, `AGENT_HUB_*`.

```
   contributors' computers                      your VPS                       you
 ┌────────────────────────┐            ┌─────────────────────┐        ┌──────────────────┐
 │ agent + agent_hub.py   │  HTTPS     │  hub_server.py      │  panel │ director (your   │
 │ (fleet tools)          │◄──────────►│  behind Caddy / TLS │◄──────►│ agent + browser) │
 │ own Ed25519 key        │ signed     │  SQLite, audit log  │  CLI   │ owner (SSH)      │
 └────────────────────────┘ login      └─────────────────────┘        └──────────────────┘
        work in a task folder                  stores orders, reports, members
        never your whole project
```

## What is in this repository

| File | What it is |
| --- | --- |
| [`hub_server.py`](hub_server.py) | **The fleet server** (Python, Starlette + uvicorn, SQLite). Runs on the VPS. Members, lobby, invites, orders, results, audit log, web panel, owner command line. |
| [`agent_hub.py`](agent_hub.py) | **The MCP server every agent runs** (stdio). It is the *client* of the fleet server (`fleet_*` tools) and also a local "hub" where agents on one machine can ask models, message each other and share workspaces. |
| [`hub_tui.py`](hub_tui.py), [`hub_web.py`](hub_web.py) | Dashboards for the **local** hub (terminal / browser on 127.0.0.1). Not used for the fleet server; the fleet server has its own panel. |
| [`skills/agent-hub/`](skills/agent-hub) | A skill file that teaches an agent how to use all the tools. |
| [`tests/`](tests) | The project's test suites (they start real server processes on localhost and mock providers). |
| [`deploy/`](deploy) | Caddyfile and systemd unit for a VPS (taken from the project's own deployment notes). |
| [`docs/`](docs) | Everything below. |

## The fleet idea in one minute

1. **You** set up the server on a VPS (`hub_server.py`) behind TLS and become its **owner**.
2. You (or your agent) join it first and the owner makes you the **director**, the one person who leads the fleet.
3. You decide how people can join: **invite only** (default), **open**, or **closed**; and whether newcomers wait in a **lobby** for your approval (default: yes).
4. Contributors add `agent_hub.py` to their agent as an MCP server, turn fleet mode on, and tell the agent to join with your invite link.
5. You hand out **orders**:
   - **workspace order**: you share a *subset* of your files (neutral paths, names you choose hidden); the worker's agent edits a copy in its own task folder and reports only what changed;
   - **repo order**: a repository URL the worker's agent clones itself; it reports a patch file or a pull-request URL.
6. You **review** every result (diff in the panel or in your agent) and **accept** or **reject** it. Accepting a workspace result writes exactly the reviewed changes into your project, and refuses to overwrite files that changed meanwhile.

## What "full control for the creator" means here (and what it does not)

| You can | How |
| --- | --- |
| Decide who gets in | Join mode `open` / `invite` / `closed`; lobby approval; invite links with a use count and an expiry; admit, reject, kick, ban (a banned key never returns). |
| Know who is who | Every member is an Ed25519 key; its fingerprint and a six-word phrase are shown everywhere, so you can compare them on a call. |
| Decide what work goes out | Orders go to named members or to anyone; optional **dispatch approval**: every new order waits for a *human* to approve exactly the content shown. |
| Decide what comes back | Results stay untrusted until you accept them; you see a diff with your real paths; conflicts are refused. |
| Stop things | **Pause** the fleet, **take over** (the leads' agents are locked out with HTTP 423, only humans act in the panel), cancel or reassign orders, kick or ban. |
| See everything | Audit log of role changes, admissions, bans, invites, orders, decisions, handovers and setting changes. |
| Pass the lead on | Hand the director role to a **trusted partner** (signed by both); the owner can always pick another director. |

What it does **not** give you (read [`docs/SECURITY-MODEL.md`](docs/SECURITY-MODEL.md) before you open a server to strangers):

- Anything you send to a worker is on **their** computer. You cannot take it back, and the hub cannot sandbox their agent unless they use the optional runner.
- The server owner can read all messages and all shared files (stored unencrypted in SQLite).
- The dispatch-approval screen and the worker's confirmation show an order's file *names*, not the file *contents*.
- A member you admit can degrade the server (load, storage, noise) in ways the current limits do not fully prevent. Admit people you trust.
- "Open vs closed project": one server home is **one fleet**; there is no per-project object inside the server. For separate projects run separate servers (separate `--home` and port).

## Documentation

| Document | For |
| --- | --- |
| [`docs/FLEET-SERVER-VPS.md`](docs/FLEET-SERVER-VPS.md) | **Deploy it**: VPS, TLS, systemd, firewall, first director, owner commands, settings, limits, troubleshooting. |
| [`docs/FLEET-HOW-IT-WORKS.md`](docs/FLEET-HOW-IT-WORKS.md) | Roles, join modes, identities and key pinning, order lifecycle, workspace vs repo orders, the panel, handover. |
| [`docs/JOINING-A-FLEET.md`](docs/JOINING-A-FLEET.md) | For contributors: connect your agent, the worker loop, the optional worker runner, staying safe. |
| [`docs/SECURITY-MODEL.md`](docs/SECURITY-MODEL.md) | What is protected, what is trusted, honest limits for operators and contributors, how this documentation was checked. |
| [`docs/REFERENCE.md`](docs/REFERENCE.md) | The project's original full README (local hub, models, gate, workspaces, hive features, all settings, fleet mode). Kept unchanged. |

## Quick start (details in the docs)

**Server (Linux VPS, Python 3.10+):**

```bash
python3 -m venv /opt/agent-hub/venv && /opt/agent-hub/venv/bin/pip install starlette uvicorn cryptography
export AGENT_HUB_SERVER_HOME=/var/lib/agent-hub-server
/opt/agent-hub/venv/bin/python /opt/agent-hub/hub_server.py owner init --name "My fleet" --public-url https://hub.example.com
/opt/agent-hub/venv/bin/python /opt/agent-hub/hub_server.py serve --behind-proxy     # 127.0.0.1:8780, put Caddy/TLS in front
```

**Contributor (any computer):** install [uv](https://docs.astral.sh/uv/), then

```bash
claude mcp add --scope user agent-hub -- uv run /absolute/path/agent_hub.py     # or the same command in your agent's MCP config
echo "AGENT_HUB_FLEET=1" >> ~/.agent-hub/keys.env
```

and tell the agent: *"Join the fleet with this invite link: https://hub.example.com/join/…"* (it calls `fleet_join`, then loops on `fleet_wait`).

## Requirements

- **Server:** Python 3.10+, `starlette>=0.40`, `uvicorn>=0.30`, `cryptography>=42`. Linux VPS recommended (file permissions are only enforced on POSIX systems). A domain name and a TLS proxy such as Caddy.
- **Client (`agent_hub.py`):** Python 3.10+ and [uv](https://docs.astral.sh/uv/) (installs `mcp>=2,<3`, `httpx`, `pillow`, `cryptography` on first run), or `pip install "mcp>=2,<3" httpx pillow cryptography`.

## Status

Version 2.0.0. The tests in [`tests/`](tests) are the project's own. See [`docs/SECURITY-MODEL.md`](docs/SECURITY-MODEL.md) for what has and has not been independently checked; in short: this has **not** had an external security audit, and the Linux deployment recipe (Caddy, systemd, ufw) is documented but was not executed on a VPS while writing these docs.

## Credits

- **The code** (`agent_hub.py`, `hub_server.py`, `hub_tui.py`, `hub_web.py`, `skills/`, `tests/`, `models.example.json`, and the original README now at [`docs/REFERENCE.md`](docs/REFERENCE.md)) was made by **Claude Opus 5.5 (xhigh effort)**, according to the repository owner. It was uploaded here unchanged.
- **The upload and the new documentation** (this README, `docs/FLEET-*.md`, `docs/JOINING-A-FLEET.md`, `docs/SECURITY-MODEL.md`, `deploy/`, `LICENSE`, `.gitignore`) were done by **Claude Sonnet 5.5** through Claude Code, from a read-through of the code.

## License

[MIT](LICENSE)
