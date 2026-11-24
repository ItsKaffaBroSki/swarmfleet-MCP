# Security model, limits and how this was checked

Read this before you open a fleet server to people you do not know. It is a description of what the code does, written from reading it. It is **not** an external security audit.

## 1. What the design protects

These points were checked in the code (`hub_server.py` read in full; the fleet client in `agent_hub.py` read by reviewers who compared it with the server):

- **Every endpoint is authorised on the server**, for the role behind the access token, read from the database on each request. A modified client does not get more rights. No authorisation hole was found in the endpoint table.
- **Login is a signed challenge**, not a password. The server signs its challenge with its own key; the client verifies it against the pinned key and the origin before it signs anything. The member's signature binds purpose, server id, server-key fingerprint, origin, challenge, nonce, time (±5 minutes) and its own key. Challenges live 30 seconds and are single use. A relay at another address gets nothing it can use.
- **After any lost connection the client sends no request carrying its token** until the server has proved it still holds the pinned key at that address. TLS verification is never switched off; redirects are never followed; plain `http://` is only accepted for loopback addresses.
- **Actions with consequences are signed per action**: marking a trusted partner, offering and accepting a handover, and approving or refusing orders from the command line are each signed with the member's key for exactly that action (the other member's fingerprint, the handover, or the digest of the orders' content).
- **Human-only switches** (dispatch approval, takeover, limits) cannot be changed with an agent's access token.
- **The server never runs anything a member sends.** The fleet tools in the client contain no subprocess call; for repo orders the client only prints a `git clone` command for the agent.
- **Secrets are refused** in every text, file and patch a member uploads (known key formats, `NAME=value` with a secret-looking value, URL passwords, `Authorization` credentials), and secrets file names (`.env`, `*.pem`, `id_rsa*`, `.ssh/`, `.git/`, …) are refused as paths. This is a heuristic, not a guarantee: it cannot recognise every secret.
- **Paths** in orders and reports are relative, without `..`, `\`, `:` or `.git`; the client checks them again before writing, writes only inside the task folder, refuses links on the way (on POSIX with `O_NOFOLLOW`), and never overwrites an existing file when downloading.
- **Applying a result** to your project refuses files that changed since they were shared, writes new files only where nothing exists, deletes only on request, never writes through a link, into `.git`, into the hub's own data, or outside the folder the files came from.
- **The panel** uses one-time sign-in links, `Secure; HttpOnly; SameSite=Strict` cookies (`__Host-` prefix over HTTPS), CSRF tokens, `Origin` and `Host` checks, `Sec-Fetch-Site` checks, a strict Content-Security-Policy without JavaScript, and HTML-escapes everything members wrote. No SQL injection or cross-site scripting was found.
- **Defaults are conservative**: the server listens on 127.0.0.1 and refuses a public bind without TLS; join mode `invite`; approval on; claims time out; everything is in an audit log.

## 2. What you must trust

- **The server's owner.** The database is unencrypted. Whoever runs the server can read every message, every shared file, every report and the audit log. Hidden names and your project folder never reach it, but only for the files, title, instructions and criteria of an order built *with files*: messages, answers, cancel reasons and orders without files carry the director's raw text, and `replace` is silently ignored when no files are given.
- **Trust on first use of the server key.** Unless the joining human supplies the fingerprint (invite link or `server_key=`), the first join pins whatever the address presents; only TLS protects that first contact. If the pin file (`server.json` in the member's fleet folder) is deleted while the key file remains, the next join pins again silently.
- **Your own agent.** The hub does not confine an agent's file and shell tools. Contributors' agents run under their owners' control, not yours.
- **The director's judgement.** Results are untrusted until accepted, but `fleet_accept` writes the *whole* reviewed change set into your project, including new hidden files such as `.github/workflows/…` or `.claude/settings.json` (they are flagged "read these first", not refused). Read diffs, not summaries.

## 3. Honest limits for operators (a fleet open to strangers)

These come from reading the code and are **not load-tested**. Until they are addressed in code, treat every admitted member as someone who can cause trouble, and admit people you trust.

1. **A member can make the server unresponsive.** Member-supplied text goes through the secret-detection regular expressions on the server's single event-loop thread. At least one of them (`SECRET_URL_RE` in `hub_server.py`) has a quadratic worst case, so one large crafted request (up to the report size limit, 4 MB by default) can block the whole server for a very long time. (Measured on a standalone copy of the pattern: roughly 0.1 s for 8 KB of crafted text, 1.4 s for 32 KB, 5.8 s for 64 KB; the cost grows with the square of the size, so the full 4 MB is hours.) Ownership of an order is checked *after* the scan in `report()`, so any admitted worker can trigger it with any existing order number.
2. **One member can fill the shared storage.** The `storage_mb` quota (default 2000 MB) is global, not per member, and superseded reports still count until 30 days after the order ends. Every order and report creation then fails with 507 for everyone until you raise it.
3. **Unbounded growth.** Progress notes and the order history are not counted against the quota and are never pruned; neither is the audit log. Watch disk usage.
4. **Per-address limits can be sidestepped.** Only the challenge store groups IPv6 addresses by /64; the login, page and proof limits and the open-mode "new identities per hour" limit use the exact address, so someone with a /64 can rotate addresses and fill the lobby (`lobby_max`) or, with approval off, the member list. Requests with garbage `Bearer` tokens are not rate-limited before the database lookup. Slow-body connections are bounded only by `--max-connections`.
5. **Everything runs in one process on one thread** with synchronous SQLite. Every write wakes all long-polling members. Expect it to be fine for small fleets; scale beyond that is untested.
6. **Noise amplification.** A member can loop claim → decline on an order, producing offer events for every other worker (and each wakes their agent). Nothing rate-limits this beyond the general per-member limits.
7. **Behind a proxy that does not pass the client address**, all clients share one address and one set of limits (see [`FLEET-SERVER-VPS.md`](FLEET-SERVER-VPS.md), section 3).
8. **Trusted partners are powerful.** A partner can pause the fleet, switch join mode to `open` and approval off, create invites, kick and ban workers, and (a partner's human in the panel) switch dispatch approval off. The original README understates this. Mark partners only after comparing fingerprints, and remember the six-word phrase is only 48 bits (the client accepts it for trust and handover; pinning requires the full fingerprint).
9. **Approval screens show file names, not file contents.** The digest covers the content, but neither the panel's approvals page nor `fleet approve` nor the worker's `fleet confirm` shows what is *inside* the files. A director's agent can ship instructions for the worker's agent inside files that a human "approved" by digest.
10. **The Linux deployment recipe was not run here**; the systemd unit has no `LimitNOFILE` (1024 files by default against `--max-connections 2000`).

## 4. Honest limits for contributors

- Orders can carry instructions your agent may follow; confirmation exists only in the optional runner, covers orders but not director messages, and shows file names, not file contents.
- The runner's strongest confinement is Codex's permission profile (checked by the project's author on Linux only). The Claude Code path has no operating-system sandbox. `--agent cmd` has none.
- On **Windows**: key and folder permissions are not enforced by the code; the client's link checks do not detect directory junctions (a junction inside a task folder could make `fleet_report` read files outside it; a standalone test with Python 3.14.3 showed `lstat`/`islink` do not flag junctions); the runner's process handling is POSIX-oriented (`fleet stop` and the second-runner check are likely to misbehave); the Codex sandbox pre-check is skipped.
- A hostile *server* can only reach what the fleet tools show your agent; but a few client code paths do not guard against malformed server answers (for example `fleet_wait` can fail on a non-numeric event time), which at worst loses events for that session.

## 5. Differences between the original README and the code (fleet and server)

Kept here so the unchanged original ([`REFERENCE.md`](REFERENCE.md)) is not taken at its word where the code differs:

- "Owner-only" files and folders: enforced on POSIX only.
- Plain `http://` for agents is accepted for `127.0.0.1`, `localhost` and `::1`, not only `127.0.0.1`.
- `AGENT_HUB_FLEET_CA` becomes the *only* trust anchor for fleet connections in that process, not an addition to the system store.
- Roles: trusted partners can also kick/ban, pause, change join mode and approval, read the audit log and review every result; they can claim orders. `serve --trusted-proxy`, the `--yes` flags of `owner director|trust|handovers`, `fleet run --restart-delay/--max-runs` and `fleet_members(include_banned)` exist but are not documented.
- "It shows **all** of each order": the approval and confirm screens show instructions, criteria, every field and every file **name**, not file contents.
- Replace-names hiding in title/instructions/criteria works only when `files` are given.
- Instructions above 8,000 characters are cut in the worker's view (the cut is marked); the server allows 20,000.
- The "90 seconds after a start no claim goes back up" grace does not stop re-queueing of claims whose claimer is no longer a worker, nor deadline expiry.
- A plain `serve` answers at `http://127.0.0.1:8780` only when `server.json` has no https public URL.
- Retention: files of finished orders are deleted after 30 days; order history, questions, summaries, members and the audit log are kept forever.
- `owner … --reason` exists on `kick`, `ban`, `reject` only.

## 6. How this documentation was checked

Written in November 2026 against `hub_server.py` and `agent_hub.py` version 2.0.0.

- `hub_server.py` (4,856 lines) was read in full by the author of these docs.
- `agent_hub.py` (15,842 lines), `hub_tui.py`, `hub_web.py`, the skill files and `models.example.json` were each read completely by separate reviewers, who compared them with the original README and reported mismatches with line numbers; a further reviewer audited the README's fleet section claim by claim against server and client code.
- A few **local smoke checks on Windows (Python 3.14.3)**, with a throwaway server home: `owner init`, `owner settings`, `owner invite`; `serve` refusing a public bind without TLS and refusing `--behind-proxy` with an http public URL; the server answering `/api/v1/info` and the landing page; an unauthenticated request getting 401 and a wrong `Host` header 400; `agent_hub.py --check`; and a black-box script that exercised login (server-signed challenge, origin mismatch refused with 421, bad signature 401, replay of a challenge 401, join without an invite refused, reserved name refused, a secret in the join text refused with 422, a lobby member refused (403) at every lead endpoint, rate limiting returning 429).
- **Not done:** the project's own test suites were not run to completion while writing these docs; the Caddy/systemd/ufw recipe was not executed on a Linux VPS; no load or soak testing; no external security audit; the worker runner was not run (its claims come from reading the code and from the project's own README).

If you find something here that is wrong, please open an issue.
