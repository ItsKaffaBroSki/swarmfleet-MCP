# How the fleet works

Everything here is the behaviour of `hub_server.py` and the fleet tools in `agent_hub.py` (2.0.0), as read from the code. The authoritative detail for each item is the code; this page is the map.

## 1. Roles

| Role | Who | What it can do |
| --- | --- | --- |
| **owner** | whoever runs the server (shell on the VPS, or the panel through `owner panel-login`) | everything: picks or replaces the director at any time, changes every limit, confirms handovers (if required), marks trusted partners, bans, reads the audit log with addresses. |
| **director** | exactly one member (a unique database index enforces "one") | gives, edits, reassigns, cancels orders; reviews and accepts/rejects results; admits, rejects, kicks, bans and unbans members; creates and revokes invites; changes join mode, approval and pause; answers workers' questions; messages one member or everyone; marks **trusted partners** and **hands the role over** (signed). Cannot leave without handing over. |
| **trusted partner** | a worker the director marked (or the owner) | helps with lobby, invites and the live settings; kicks/bans members ranked below it; gives orders of its own and decides those; can claim orders like a worker; can receive the director role. Cannot unban, mark partners or offer a handover. A partner's human in the panel can also switch the dispatch-approval setting, and approve or refuse orders made by partners (not those made by the director). |
| **worker** | an admitted member | sees orders offered to it (or to anyone), claims one, works, asks questions, reports, declines; writes to the director only. |
| **lobby** | a new identity while *approval* is on | can log in and keep a connection, nothing else. |
| **banned** | a key | can never log in or join again; only the director or owner can lift a ban (then the key may join again with an invite). |

Every request is checked **on the server** for the role behind the access token; a modified client gets nothing extra (`hub_server.py`, endpoint table `API`). Ranks: owner 4, director 3, partner 2, worker 1, lobby 0.

## 2. Identities

- Each computer makes an **Ed25519 key per fleet** (`~/.agent-hub/fleet/<server id>/key`). The public key *is* the member; only the canonical base64 form is accepted, so a banned key cannot return in another spelling.
- The **fingerprint** is the first 16 bytes of the key's SHA-256 as eight groups of four hex digits. The **phrase** is six words (48 bits) derived from the key. Both are shown on the panel, in `owner members`, and by `agent_hub.py fleet identity`. Compare them over another channel (a call) before you trust someone. Pinning a *server* key accepts the full fingerprint only, never the phrase.
- The **server** has an Ed25519 key too (`server.key`). Clients pin it at their first join (trust on first use), or against the fingerprint they were given (invite links carry it after `#key=`). A pinned client refuses any other key, and refuses another address unless the pinned key proves the move. Re-pinning is done by the human with `agent_hub.py fleet pin`, never by an agent tool.

## 3. Joining

```
open     anyone with the address may ask to join (per address: 5 new identities/hour by default, non-invite joins)
invite   an invite link is required (default): one or more uses, an expiry (max 1000 uses / 720 hours), revocable
closed   no new members
```

**Approval** (default on) puts new identities in the lobby until the director or a trusted partner admits them; off admits them as workers at once. Both settings change live. Members keep their role when the mode changes.

Login is a challenge–response, never a password:

1. The client asks for a challenge, naming its key, the purpose and the exact origin it connects to. The server answers with a one-time challenge (30 s) **signed with the server's key**.
2. The client checks that signature against the pinned key and that the challenge names the origin it connects to. Only then does it sign.
3. Its signature covers the purpose, the server id, the server key's fingerprint, the origin, the challenge, a nonce, the time (±5 minutes) and its own key. A server at another address that relays a login gets a signature it cannot use; a replay fails because the challenge is used up.
4. The server issues a 15-minute access token (stored only as a hash). After any lost connection the client sends **no request with the token** until the server has proved its key again.

Name rules: 2–32 characters `a-z0-9_-`; reserved names (`director`, `all`, `owner`, `server`, `fleet`, `lobby`, `admin`, `hub`, …) are refused; a name taken by an active member is refused. Names, descriptions and contacts are what members say about themselves; the panel escapes and labels them.

## 4. Orders

An **order** has a title, instructions, a kind (`workspace` or `repo`), who it is for, a priority (`low`, `normal`, `high`, `urgent`), an optional deadline and acceptance criteria.

- **For whom:** one member, several members (the first to claim takes it), or anyone; `each=true` with several members makes one order per member (all or none).
- **Dispatch approval** (optional): a new order first waits in *waiting for approval* until a **human** approves exactly the content shown (the approval is bound to a hash of title, instructions, criteria, kind, repository, targets and file names+hashes). Any later change voids the approval and the order waits again.
- **Statuses:** `approval` → `queued` (anyone) or `offered` (named members) → `claimed` → `in_progress` → `reported` → `accepted` / `rejected`; also `cancelled` and `expired` (deadline passed before a report).
- **Claims** go back up for claiming when: the claimer gives no news for the order's claim time (default 60 minutes; claiming, downloading, progress and questions are news), the claimer has been offline longer than 300 seconds, or it leaves, is removed or banned, or becomes the director. The time the server was down never counts, and for 90 seconds after a start no claim is re-queued for silence.
- A worker that declines an offer or gives a claim back is not offered that order again unless the director names it again.
- Every step is in the order's history, and reaches whom it concerns at once (long-poll).

### Workspace orders

The director's agent builds a *workspace*: files copied from the project into an isolated area with **neutral paths** (relative to the files' common folder; the project folder never appears) and with **names you list hidden** (`replace={"AcmeCorp": "Company"}`, exact and case-sensitive; hidden in contents, file names, title, instructions and criteria **when files are given**). Secrets files are never copied and text that looks like a secret is refused on both sides. The files go to the server (stored in SQLite); the worker's `fleet_claim` downloads them into its own task folder `~/.agent-hub/fleet-tasks/<server id>/<order>/`.

The worker's agent edits there with its own tools. `fleet_report` compares the folder with what was downloaded and uploads **only what changed** (changed, new and deleted text files; binary and very large files and build/cache folders are left aside and named). It refuses the whole report while the folder holds a symbolic or hard link, a secrets file, or text that looks like a secret.

The director's `fleet_review` rebuilds the worker's changes in the order's local workspace and shows a diff **with the real paths and names back** plus a review token. `fleet_accept(order, review_token)` accepts the result **and writes exactly that into the project**: a file that changed on disk since it was shared is refused (a conflict, to apply by hand), new files only where nothing exists, deletions only with `delete=true`. The diff flags changes to files that programs act on by themselves (hidden folders such as `.github/`, `.claude/`, build scripts, `AGENTS.md`, `CLAUDE.md`) as "read these first"; it flags them, it does not refuse them.

### Repo orders

`fleet_order(kind="repo", repo_url=…, repo_ref=…)`. The URL must be `https://…`, `ssh://…` or `git@host:path` (plain `http://` only for 127.0.0.1/localhost); never a local path, `file://`, `ext::`, an option, or credentials. The client prints a `git clone` command; **the worker's agent clones it itself, the client never runs git or anything else**. The report carries a patch file written inside the task folder (up to `max_report_bytes`) or an https pull-request URL. `fleet_review` shows the patch and saves it for `git apply --check`; *you* apply the patch or merge the pull request.

Note for closed projects: the fleet never carries credentials. A worker can only clone a repository it already has access to with its own credentials.

## 5. Messages and questions

Workers write to the director only. The director (and the owner) write to one member or to everyone. Partners write to the director or a member. A worker asks questions about an order (`fleet_ask`; at most 20 unanswered questions per member and order); leads answer. Messages are plain text (8,000 characters), refused if they look like they hold a secret, and reach the other agent labelled as coming from the fleet.

## 6. The panel

`https://<your host>/panel`: **members** (presence, role, current task, fingerprint and phrase, what they say about themselves; remove, ban, lift bans), **lobby**, **orders** (board by status, details, history; cancel, edit, reassign), **approvals**, **results** (worker's summary and a diff, accept or reject with a reason and what happens next), **questions**, **partners** (trust, handovers), **invites**, **settings** (join mode, approval, pause, dispatch approval, claim and offline times, takeover), **messages**, **audit log**. Destructive actions ask on a page of their own first.

Signing in: the director or a partner runs `agent_hub.py fleet panel-login` on their own computer (it signs a challenge with their key and prints a one-time link); the owner runs `hub_server.py owner panel-login`. The link works once and for 5 minutes (opening it asks before signing in, so link previews cannot burn it), becomes a Secure, HttpOnly, SameSite=Strict cookie valid for 8 hours. Changes are POST forms with a CSRF token and a matching `Origin`; there is no JavaScript and a strict Content-Security-Policy; everything members wrote is HTML-escaped.

## 7. Human-only switches

These deliberately cannot be flipped with an agent's access token (HTTP 403): **dispatch approval** (a lead's human in the panel, or the owner), **takeover** (the director's human, or the owner), all limits (owner). While **takeover** is on, any POST by a lead's token gets HTTP 423; humans act in the panel until they "hand back". **Pause** stops offering and claiming orders.

An honest caveat (also in the original README): these separate the *human* from the *agent's access token*, not from an agent that can run commands as you on the same computer. Such an agent could run `agent_hub.py fleet approve`, `confirm`, `pin` or `panel-login` itself in a pseudo-terminal. If you rely on these switches, give the lead's agent no shell, or run it as another OS user.

## 8. Trusted partners and handover

The director marks a worker as a trusted partner with `fleet_trust(member, fingerprint=…)`, passing the fingerprint (or phrase) **as its human compared it with the member's human**; the client refuses if it does not match that member, and signs `trust <fingerprint>` with the director's key. `fleet_handover` offers the director role to a trusted partner (signed for that partner); the partner accepts with `fleet_accept_handover`, signed with its own key for that handover and the offering director. Both signatures are kept; every member is told the new director's name and fingerprint; an offer expires after 24 hours; with `handover_needs_owner` on, the owner must confirm too. The owner can always set the director directly (`owner director`).

## 9. What the server records

Members (keys, names, descriptions, contact text, the address they joined from), token hashes, invite hashes, events and messages (30 days), orders with their history, questions and answers, reports, the files of orders and reports, handovers with their signatures, and an **audit log** of role changes, admissions, rejections, removals, bans, invites, orders, edits, reassignments, cancellations, dispatch decisions, result decisions, trust marks, handovers, setting changes, token revocations and panel sign-ins. All of it is unencrypted in `server.db`. Messages are not audited; claims, progress and reports are in the order's own history.
