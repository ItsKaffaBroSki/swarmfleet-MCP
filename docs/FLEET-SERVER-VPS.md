# Running the fleet server on a VPS

This is the operator's guide for `hub_server.py`. Facts here come from reading the server code (`hub_server.py`, 2.0.0) and from the project's own deployment notes ([`REFERENCE.md`](REFERENCE.md), section "Running it on a VPS"). **What was and was not executed** is listed at the end of [`SECURITY-MODEL.md`](SECURITY-MODEL.md): in short, the server's commands and refusals were exercised locally on Windows, but the Caddy, systemd and ufw steps below were **not** run on a Linux VPS while writing this.

## 0. What you need

- A Linux VPS (any small one; the server is one Python process with one SQLite file) and a **domain name** pointing at it.
- Python **3.10+** with the `venv` module (on Debian/Ubuntu: `apt install python3-venv`).
- Python packages `starlette>=0.40`, `uvicorn>=0.30`, `cryptography>=42` (that is all `hub_server.py` needs; the rest is the standard library).
- A TLS reverse proxy on the same machine. [Caddy](https://caddyserver.com/) is the documented choice because it fetches and renews certificates itself.
- Ports 80 and 443 open to the internet. Port 8780 (the server) stays closed; it listens on 127.0.0.1 only.

`hub_server.py` does **not** serve HTTPS by default. It refuses to listen on a non-loopback address without TLS (see section 7 for the alternatives).

## 1. Install

```bash
# as root, once
useradd --system --home /var/lib/agent-hub-server --create-home agenthub
mkdir -p /opt/agent-hub && cp hub_server.py /opt/agent-hub/
python3 -m venv /opt/agent-hub/venv && /opt/agent-hub/venv/bin/pip install starlette uvicorn cryptography
```

## 2. Initialise the server (required before the first start)

```bash
sudo -u agenthub AGENT_HUB_SERVER_HOME=/var/lib/agent-hub-server /opt/agent-hub/venv/bin/python \
    /opt/agent-hub/hub_server.py owner init --name "My fleet" --public-url https://hub.example.com
```

This creates, in the server's home folder (`$AGENT_HUB_SERVER_HOME`, default `~/.agent-hub-server`):

| File | What it is |
| --- | --- |
| `server.json` | server id (`fs-` + 12 hex digits), name, public URL (mode 0600) |
| `server.db` (+ `-wal`, `-shm`) | the SQLite database (mode 0600) |
| `server.key` | the server's own Ed25519 private key, unencrypted PEM (mode 0600) |

It prints the **server key's fingerprint and six-word phrase**. Members pin that key when they join; keep it.

Rules for `--public-url`:

- It must be the **exact address members will type**, `https://` and no path. Members sign for that origin only; a request for any other origin is refused (HTTP 421).
- Do **not** write the default port (`https://hub.example.com:443`): the server compares the `Host` header literally, and clients send `Host: hub.example.com`, so every request would get `400 wrong_host`. (From code reading, not tested.)
- `serve --behind-proxy` needs the public URL from `server.json` (or `--public-url`); without one it exits immediately. If you use the systemd unit without having run `owner init --public-url`, systemd will restart it in a loop.
- Plain `http://` public URLs are accepted only for loopback (local experiments).

## 3. TLS proxy (Caddy)

[`deploy/Caddyfile`](../deploy/Caddyfile) (replace the domain):

```
hub.example.com {
    header Strict-Transport-Security "max-age=31536000"
    reverse_proxy 127.0.0.1:8780 {
        transport http {
            keepalive 4s
        }
    }
}
```

Why these details matter (from the code):

- The server only believes `X-Forwarded-For` when `--behind-proxy` is set **and** the TCP peer is a trusted proxy (default `127.0.0.1` and `::1`; add others with `--trusted-proxy`). It uses the **last** entry of the header.
- The server closes idle connections after **5 seconds** (`KEEP_ALIVE`); the proxy's upstream keep-alive must be shorter (4 s here), or the proxy may reuse a connection the server just closed.
- The proxy must pass the `Host` header unchanged (the server refuses any `Host` that is not its public host).
- If your proxy does **not** set `X-Forwarded-For` (or runs on another machine and is not listed with `--trusted-proxy`), every client looks like one address. Then the per-address limits (logins, challenges, new identities) are shared by the whole fleet. A CDN in front makes the "address" the CDN's edge address.

## 4. systemd

[`deploy/agent-hub-server.service`](../deploy/agent-hub-server.service):

```ini
[Unit]
Description=Agent Hub fleet server
After=network-online.target
Wants=network-online.target

[Service]
User=agenthub
Group=agenthub
Environment=AGENT_HUB_SERVER_HOME=/var/lib/agent-hub-server
ExecStart=/opt/agent-hub/venv/bin/python /opt/agent-hub/hub_server.py serve --behind-proxy
Restart=on-failure
NoNewPrivileges=yes
ProtectSystem=strict
ProtectHome=yes
PrivateTmp=yes
ReadWritePaths=/var/lib/agent-hub-server

[Install]
WantedBy=multi-user.target
```

Then `systemctl daemon-reload && systemctl enable --now agent-hub-server caddy`.

The unit only needs to write inside the server's home folder (database, WAL files, key). Optional hardening you may want, **not part of the project's unit and untested**: `LimitNOFILE=8192` (the default limit of 1024 files is below the server's default `--max-connections 2000`) and `RestartSec=5`.

## 5. Firewall

```bash
ufw default deny incoming
ufw allow OpenSSH
ufw allow 80,443/tcp
ufw allow 443/udp      # optional: HTTP/3 (without it Caddy falls back to TCP)
ufw enable
```

## 6. Check it, then create the first director

```bash
curl -s https://hub.example.com/api/v1/info
```

This public endpoint returns the server id, name, version, protocol (`2`), join mode, the server key and its fingerprint. Compare the fingerprint with `owner key`.

```bash
# run every owner command as the service user (same home folder as the service)
alias hub='sudo -u agenthub AGENT_HUB_SERVER_HOME=/var/lib/agent-hub-server /opt/agent-hub/venv/bin/python /opt/agent-hub/hub_server.py'

hub owner key                      # fingerprint + phrase of the server (give it to members over another channel)
hub owner invite --uses 1 --hours 24 --note "director"
```

The invite link looks like `https://hub.example.com/join/<token>#key=<fingerprint>`. The part after `#` never goes to any server; the joining client uses it to pin the server's key.

1. Open your agent with `agent_hub.py` and fleet mode on (see [`JOINING-A-FLEET.md`](JOINING-A-FLEET.md)) and have it `fleet_join` with that link. With approval on (default) it lands in the **lobby**.
2. Compare its fingerprint or phrase with what you expect (`hub owner members` lists them).
3. `hub owner director <name>` — it asks you to confirm (`--yes` skips the question; without a terminal it refuses unless `--yes`). Now that member is the **director**.

From then on the director runs the fleet through its agent (the `fleet_*` tools) and through the web panel; you, the owner, keep the command line. The owner command line has no `admit` command: lobby members are admitted by the director or a trusted partner (agent tools or panel), or by you in the panel (`owner panel-login`).

## 7. Other ways to run it

| Situation | Command |
| --- | --- |
| **Local experiment** | `hub_server.py owner init --public-url http://127.0.0.1:8780` then `hub_server.py serve`. If `server.json` already has an https public URL, plain `serve` aborts: pass `--public-url http://127.0.0.1:8780`. Agents may use plain http only for loopback addresses. |
| **Server terminates TLS itself** | `serve --host 0.0.0.0 --port 443 --tls-cert fullchain.pem --tls-key privkey.pem --public-url https://hub.example.com`. The certificate is read at start, so restart after renewal; binding port 443 needs privileges. The server then also sends `Strict-Transport-Security` itself. |
| **TLS proxy on another machine of a private network** | `serve --host <private-ip> --behind-proxy --insecure-plain-http --trusted-proxy <proxy-ip>`. Prints loud warnings: tokens cross that network unencrypted, so firewall the port. |
| **Self-signed certificate** | on each *agent's* side: `AGENT_HUB_FLEET_CA=/path/ca.pem` (this replaces the default trust store for fleet connections in that process; TLS is never switched off). |

The server **refuses** to start in these cases: non-loopback `--host` without TLS and without `--behind-proxy --insecure-plain-http`; `--behind-proxy` or TLS with a non-https public URL; a public `http://` URL that is not loopback; `--tls-cert` without `--tls-key`. (The first two I ran locally and saw the refusal; the last two are from reading `serve()` and `check_public_url()`.)

## 8. `serve` options

| Option | Default | Meaning |
| --- | --- | --- |
| `--home` | `$AGENT_HUB_SERVER_HOME` or `~/.agent-hub-server` | data folder |
| `--host` / `--port` | `127.0.0.1` / `8780` | listen address |
| `--public-url` | from `owner init` | the https URL members use |
| `--tls-cert` / `--tls-key` | none | serve TLS here (both required) |
| `--behind-proxy` | off | trust `X-Forwarded-For` from trusted proxies; Secure cookies |
| `--trusted-proxy` | `127.0.0.1`, `::1` | proxy address(es); repeatable |
| `--insecure-plain-http` | off | allow plain HTTP on a non-loopback address (needs `--behind-proxy`) |
| `--token-minutes` | 15 (1–1440) | access token lifetime |
| `--offline-after` | 20 (3–3600) | seconds without contact before a member shows offline |
| `--auth-per-minute` | 20 | challenges/logins per address per minute (burst: half, at least 5) |
| `--api-per-minute` | 600 | API requests per member per minute (burst: a third, at least 20) |
| `--messages-per-minute` | 30 | messages (and questions) per member per minute |
| `--max-connections` | 2000 | open connections at most |
| `--start-grace` | 90 (0–3600) | seconds after a start in which no claim is re-queued for silence |

Rate-limit counters are in memory and reset on restart.

## 9. Owner commands (on the VPS)

Every command also takes `--home`. A member is named by `#id`, name, or fingerprint (8+ hex digits).

```
hub owner init [--name N] [--public-url U]
hub owner key
hub owner members [--all]            hub owner orders [--status open|final|all|<status>] [--last N]
hub owner invite [--uses N --hours H --note T | --list | --revoke ID]
hub owner director <member> [--old worker|partner|removed] [--yes]
hub owner trust <member> [--revoke] [--yes]
hub owner handovers [--confirm ID | --cancel ID | --all] [--yes]
hub owner kick|ban|reject <member> [--reason R]     hub owner unban <member>     hub owner revoke <member>
hub owner settings [...]             (see below)
hub owner panel-login [--url U]      hub owner panel-logout       hub owner audit [--last N]
```

The owner commands write the database directly; the running server notices at once. `owner revoke` ends a member's access tokens and panel sessions (it logs in again with its key).

## 10. Settings (live, apply to the next request)

| Setting | Default | Who can change it |
| --- | --- | --- |
| `join_mode` (`open` / `invite` / `closed`) | `invite` | director, trusted partners, owner |
| `approval` (new identities wait in the lobby) | on | director, trusted partners, owner |
| `paused` | off | director, trusted partners, owner |
| `claim_minutes` (claim without news is re-queued) | 60 | director and trusted partners (panel, or the HTTP API with their token; the `fleet_settings` agent tool exposes only join mode, approval and pause), owner |
| `requeue_offline_seconds` | 300 | same |
| `dispatch_approval` (every new order waits for a human) | off | a lead's **human in the panel**, or the owner; **not** by an agent's token |
| `takeover` (leads' agents locked out, HTTP 423) | off | the director's human in the panel, or the owner |
| `handover_needs_owner` | off | owner |
| `new_keys_per_hour` (open mode, per address, non-invite joins) | 5 | owner |
| `lobby_max` / `members_max` | 200 / 1000 | owner |
| `max_order_bytes` / `max_report_bytes` | 4,000,000 each | owner |
| `storage_mb` (all order and report files) | 2000 | owner |
| `orders_open_max` | 1000 | owner |

## 11. Fixed limits

Request bodies 64 KiB (orders and reports with files: 24 MiB); at most 500 files per order or report; messages 8,000 characters; instructions 20,000; acceptance criteria 4,000; names 2–32 characters (`a-z0-9_-`); at most 3 open long-polls per member; an order's claim without news lasts `claim_minutes`. Too much gets HTTP 429 (with `Retry-After`), 413 or 507. Events are pruned after 30 days (expired tokens after a day; the pruning runs hourly while the server runs); **the files of finished orders are deleted 30 days after the order's last update, but order history, questions, report summaries, members and the audit log are never pruned** (database growth is yours to watch).

## 12. Backups and updates

- **Back up `server.key` together with `server.db`.** If the key is lost, every member's client refuses the new key until each human re-pins it (`agent_hub.py fleet pin`, comparing with the fingerprint you give them from `owner key`; their agents cannot do it).
- The database is SQLite in WAL mode: stop the service, or use SQLite's own backup facility, rather than copying the files while the server runs. (General SQLite practice; not something the project provides.)
- Updating: replace `hub_server.py` and restart. The schema is versioned and migrated automatically; a database made by a **newer** `hub_server.py` is refused.

## 13. Several projects

One server home is one fleet. The server has no project object: all members see the same fleet. To keep projects apart, run one server per project, each with its own `--home` (`AGENT_HUB_SERVER_HOME`), its own `--port`, its own public URL (or path-less subdomain) and its own proxy entry.

## 14. Troubleshooting

| You see | Meaning |
| --- | --- |
| `refusing to listen on … without TLS` | You bound a public address without TLS; use the proxy setup or `--tls-cert/--tls-key`. |
| `no fleet server in <home>: run hub_server.py owner init first` | Run section 2 first (and make sure `AGENT_HUB_SERVER_HOME` matches). |
| `400 wrong_host` | The `Host` header is not the public host (proxy not passing it, or a `:443` in the public URL). |
| `421 wrong_origin` / client says "answered for another address" | The URL the member used is not the server's public URL. After this a client stays distrustful until a new `fleet_join`. |
| `401 bad_time` | The member's clock is more than 5 minutes off. |
| `403 invite_required` / `invite_invalid` | Join mode is `invite` and the link is missing, used up, expired or revoked. |
| `429` | Rate limit (look at `Retry-After`); in open mode also `too_many_new_keys` (5 per address per hour by default). |
| `507 storage_full` | `storage_mb` reached; raise it with `owner settings --storage-mb N` or wait for finished orders to expire. |
| `423` for a lead's agent | The director took over; humans act in the panel until "Hand back". |
