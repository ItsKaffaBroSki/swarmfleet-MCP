#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "textual>=8,<9",
#     "mcp>=2,<3",
#     "httpx>=0.27,<1",
# ]
# ///
"""Agent Hub dashboard: a live view of the hub's shared database, and the approvals for the gate.

    uv run hub_tui.py           the dashboard
    uv run hub_tui.py --once    print the overview and exit

Everything is read-only, except approving or denying requests at the gate, applying or closing workspaces, and the
hive (its settings, deleting notes, withdrawing or deleting decisions, changing or deleting tasks, resetting ratings),
which go through agent_hub's own functions (mcp and httpx are listed above only for that, and for the Models tab).
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any

from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.css.query import NoMatches
from textual.screen import ModalScreen, Screen
from textual.widgets import DataTable, Footer, Header, Input, Static, TabbedContent, TabPane, TextArea

ONLINE_WINDOW = 50  # seconds, as in agent_hub.py
GATE_DEFAULT_TIMEOUT = 600  # as in agent_hub.py: the deadline of a gate row written before deadlines were recorded
# CSI and OSC sequences, and single-character escapes: agent text must not drive the terminal.
ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)|\x1b[@-Z\\-_]")
TABS = ["overview", "approvals", "agents", "threads", "messages", "reservations", "models", "errors", "help", "workspaces"]
TAB_KEYS = dict(zip(TABS, "1234567890"))  # Workspaces on 0; Help also on ? and F1
# The hive: its settings, the notes, the decision log, the task board, the model track record (scores).
HIVE_TAB_KEYS = {"hive": "h", "memory": "m", "decisions": "l", "tasks": "t", "scores": "s"}
TASK_USER_STATUSES = ("open", "blocked", "done", "dropped")  # as agent_hub's: what the user sets
RATINGS_SHOWN = 500  # the newest ratings a model's view lists (its score counts them all)
KIND_FILTERS = {
    "all": None,
    "requests": ("ask",),
    "replies": ("reply",),
    "errors": ("error", "turn_error"),
    "discussions": ("turn", "summary"),
    "public posts": ("post",),
}


def _keys_file_value(home: Path, name: str) -> str:
    """A setting from keys.env, which the server also reads settings from (the environment wins)."""
    try:
        lines = (home / "keys.env").read_text(encoding="utf-8-sig").splitlines()
    except (OSError, UnicodeDecodeError):
        return ""
    for line in lines:
        line = line.strip()
        if line.startswith("export "):
            line = line[7:].lstrip()
        key, sep, value = line.partition("=")
        if sep and key.strip() == name:
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            return value
    return ""


def db_path() -> Path:
    """The same database the server uses: AGENT_HUB_DB, else hub.db in AGENT_HUB_HOME (default ~/.agent-hub)."""
    home = Path(os.environ.get("AGENT_HUB_HOME") or "~/.agent-hub").expanduser()
    chosen = os.environ.get("AGENT_HUB_DB") or _keys_file_value(home, "AGENT_HUB_DB")
    return Path(chosen).expanduser() if chosen else home / "hub.db"


def refresh_seconds() -> float:
    try:
        value = float(os.environ.get("AGENT_HUB_TUI_REFRESH") or 2)
    except ValueError:
        value = 2.0
    return max(0.5, min(60.0, value))


def clean(text: Any) -> str:
    """Agent text for display: no escape sequences or control characters."""
    text = ANSI_RE.sub("", "" if text is None else str(text))
    return "".join(c if c in "\n\t" or c.isprintable() else "?" for c in text)


def one_line(text: Any, width: int = 120) -> str:
    line = " ".join(clean(text).split())
    return line if len(line) <= width else line[: width - 1] + "…"


def lit(text: Any, style: str = "") -> Text:
    """Literal text: never parsed as markup (strings in a DataTable cell would be)."""
    return Text(clean(text), style=style)


def ago(ts: float | None, now: float | None = None) -> str:
    if not ts:
        return "never"
    secs = max(0, int((now or time.time()) - ts))
    if secs < 90:
        return f"{secs}s ago"
    if secs < 5400:
        return f"{secs // 60}m ago"
    if secs < 172800:
        return f"{secs // 3600}h ago"
    return f"{secs // 86400}d ago"


def meta_of(raw: Any) -> dict[str, Any]:
    try:
        meta = json.loads(raw or "{}")
    except (TypeError, ValueError):
        return {}
    return meta if isinstance(meta, dict) else {}


def agent_status(agent: dict[str, Any], now: float) -> str:
    if now - (agent.get("last_seen") or 0) >= ONLINE_WINDOW:
        return "offline"
    if (agent.get("listening_until") or 0) > now:
        return "listening"
    return "busy" if agent.get("last_inbox") else "online"


# --------------------------------------------------------------------------
# Reading the database (read-only connection, in a worker thread)
# --------------------------------------------------------------------------


def connect(path: Path) -> sqlite3.Connection:
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=1.0)
    con.row_factory = sqlite3.Row
    return con


def read_snapshot(path: Path, scan: bool = False, query: str = "") -> dict[str, Any] | None:
    """Everything the tabs show, or None when there is no database yet. `scan`: also compare each open workspace's
    files with what was shared (it reads them, so only while its tab is shown). `query`: the filter's text, looked
    for in the whole text of the hive's notes and decisions (their tables show only the start)."""
    if not path.exists():
        return None
    con = connect(path)
    try:
        tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}

        def columns(table: str) -> set[str]:
            return {r[1] for r in con.execute(f"PRAGMA table_info({table})")} if table in tables else set()

        now = time.time()
        snap: dict[str, Any] = {"now": now, "path": str(path), "agents": [], "messages": [], "threads": [],
                                "reservations": [], "usage": {}, "errors": [], "gate_pending": [], "gate_done": [],
                                "counts": {}, "recent": [], "workspaces": [], "settings": {}, "memories": [],
                                "decisions": [], "memory_hits": None, "decision_hits": None, "tasks": [],
                                "task_hits": None, "scores": []}  # fmt: skip
        if "agents" in tables:
            snap["agents"] = [dict(r) for r in con.execute("SELECT * FROM agents ORDER BY name")]
        if "messages" in tables:
            cols = columns("messages")
            held = "held" if "held" in cols else "0"
            public = "public" if "public" in cols else "0"
            snap["messages"] = [dict(r) for r in con.execute(
                "SELECT id, ts, thread, sender, recipient, kind, rtype, substr(body, 1, 400) AS preview,"
                f" length(attachments) AS att, meta, reply_to, {held} AS held, {public} AS public FROM messages"
                " ORDER BY id DESC LIMIT 500"
            )]  # fmt: skip
            threads = [dict(r) for r in con.execute(
                "SELECT thread, COUNT(*) AS n, MAX(ts) AS last, MIN(id) AS first_id, GROUP_CONCAT(DISTINCT sender)"
                " AS senders FROM messages GROUP BY thread ORDER BY last DESC LIMIT 300"
            )]  # fmt: skip
            firsts = {r["id"]: r["preview"] for r in con.execute(
                f"SELECT id, substr(body, 1, 300) AS preview FROM messages WHERE id IN"
                f" ({','.join(str(int(t['first_id'])) for t in threads) or '0'})"
            )}  # fmt: skip
            for t in threads:
                t["first"] = firsts.get(t["first_id"], "")
            snap["threads"] = threads
            usage: dict[str, dict[str, Any]] = {}
            for r in con.execute(
                "SELECT sender, kind, ts, meta FROM messages WHERE ts > ? AND kind IN ('reply','turn','error',"
                "'turn_error') ORDER BY id DESC LIMIT 20000", (now - 7 * 86400,)
            ):  # fmt: skip
                u = usage.setdefault(r["sender"], {"calls": 0, "in": 0, "out": 0, "secs": 0.0, "errors": 0,
                                                   "in24": 0, "out24": 0})  # fmt: skip
                if r["kind"] in ("error", "turn_error"):
                    u["errors"] += 1
                    continue
                meta = meta_of(r["meta"])
                u["calls"] += 1
                for key in ("in", "out"):
                    value = meta.get(key) if isinstance(meta.get(key), int) else 0
                    u[key] += value
                    if r["ts"] > now - 86400:
                        u[f"{key}24"] += value
                u["secs"] += float(meta.get("secs") or 0) if isinstance(meta.get("secs"), (int, float)) else 0
            snap["usage"] = usage
            snap["errors"] = [dict(r) for r in con.execute(
                "SELECT id, ts, thread, sender, recipient, substr(body, 1, 2000) AS body FROM messages"
                " WHERE kind IN ('error','turn_error') ORDER BY id DESC LIMIT 200"
            )]  # fmt: skip
            day = con.execute("SELECT COUNT(DISTINCT thread), COUNT(*) FROM messages WHERE ts > ?",
                              (now - 86400,)).fetchone()  # fmt: skip
            open_requests = con.execute(
                "SELECT COUNT(*) FROM messages q WHERE q.kind='ask' AND q.ts > ? AND NOT EXISTS (SELECT 1 FROM"
                " messages a WHERE a.reply_to = q.id AND a.kind IN ('reply','error','summary'))", (now - 86400,)
            ).fetchone()[0]  # fmt: skip
            snap["counts"].update(threads24=day[0], messages24=day[1], open=open_requests)
        if "reservations" in tables and "agents" in tables:
            snap["reservations"] = [dict(r) for r in con.execute(
                "SELECT r.path, r.reason, r.expires, a.name, a.last_seen FROM reservations r JOIN agents a"
                " ON a.instance = r.instance WHERE r.expires > ? ORDER BY r.path", (now,)
            ) if now - r["last_seen"] < ONLINE_WINDOW]  # fmt: skip
        if "gate" in tables:
            # A request past its deadline is expired, whether or not a session has written that yet (this
            # connection only reads: the next decision, peers or session start writes it).
            deadline = f"COALESCE(expires, ts + {GATE_DEFAULT_TIMEOUT})" if "expires" in columns("gate") else (
                f"ts + {GATE_DEFAULT_TIMEOUT}")  # fmt: skip
            snap["gate_pending"] = [dict(r) for r in con.execute(
                "SELECT id, ts, kind, origin, target, substr(content, 1, 300) AS preview, length(content) AS size,"
                f" editable, {deadline} AS deadline FROM gate WHERE status='pending' AND {deadline} > ?"
                " ORDER BY id LIMIT 500", (now,)
            )]  # fmt: skip
            snap["gate_done"] = [dict(r) for r in con.execute(
                "SELECT id, ts, kind, origin, target, status, note, decided_at FROM gate WHERE status != 'pending'"
                f" UNION ALL SELECT id, ts, kind, origin, target, 'expired', 'nobody decided before the deadline',"
                f" {deadline} FROM gate WHERE status='pending' AND {deadline} <= ?"
                " ORDER BY decided_at DESC LIMIT 50", (now,)
            )]  # fmt: skip
        if "workspaces" in tables:
            snap["workspaces"] = [dict(r) for r in con.execute(
                "SELECT w.id, w.ts, w.updated, w.owner, w.parent, w.root, w.label, w.status, (SELECT COUNT(*) FROM"
                " workspace_files f WHERE f.workspace = w.id) AS files, (SELECT GROUP_CONCAT(a.name || ' (' || a.kind"
                " || ')', ', ') FROM workspace_access a WHERE a.workspace = w.id) AS recipients FROM workspaces w"
                " ORDER BY w.ts DESC LIMIT 200"
            )]  # fmt: skip
            for w in snap["workspaces"]:
                w["changed"] = None
                if scan and w["status"] == "open":
                    w["changed"] = workspace_changes(con, path, w["id"])
        read_hive(con, tables, columns, snap, query)
        snap["counts"].update(
            online=sum(1 for a in snap["agents"] if now - (a.get("last_seen") or 0) < ONLINE_WINDOW),
            agents=len(snap["agents"]),
            reservations=len(snap["reservations"]),
            waiting=len(snap["gate_pending"]),
            workspaces=sum(1 for w in snap["workspaces"] if w["status"] == "open"),
        )
        snap["recent"] = snap["messages"][:10]
        return snap
    finally:
        con.close()


def read_hive(con: sqlite3.Connection, tables: set[str], columns: Any, snap: dict[str, Any], query: str) -> None:
    """The hive: the settings the user chose, the newest notes and decisions (the start of their text), with the
    latest gate decision about each, and which of them hold the filter's text anywhere."""
    if "settings" in tables:
        snap["settings"] = {r["key"]: dict(r) for r in con.execute("SELECT key, value, updated_at, updated_by FROM"
                                                                    " settings")}  # fmt: skip
    ref = "ref" in columns("gate")

    def review(kind: str, alias: str) -> str:
        return (f"(SELECT status FROM gate g WHERE g.ref = '{kind}:' || {alias}.id ORDER BY g.id DESC LIMIT 1)" if ref
                else "NULL")  # fmt: skip

    like = f"%{query.lower()}%" if query else ""
    if "memories" in tables:
        snap["memories"] = [dict(r) for r in con.execute(
            "SELECT m.id, m.ts, m.author, m.abliterated, m.cross_hold, m.cleared, m.project, m.scope, m.tags,"
            f" substr(m.text, 1, 300) AS preview, length(m.text) AS size, {review('memory', 'm')} AS review"
            " FROM memories m ORDER BY m.id DESC LIMIT 1000"
        )]  # fmt: skip
        if like:
            snap["memory_hits"] = {r[0] for r in con.execute(
                "SELECT id FROM memories WHERE lower(text) LIKE ? OR lower(tags) LIKE ? OR lower(author) LIKE ?"
                " OR lower(project) LIKE ?", (like,) * 4)}  # fmt: skip
    states: dict[int, str] = {}
    if "decisions" in tables and "decision_stances" in tables:
        # Superseded or not is worked out from the chain of decisions in force (agent_hub.decision_links: what you
        # see; an agent of the other group sees a replacement only once you cleared it). "status" below is that.
        links = decision_links(con)
        snap["decisions"] = [dict(r) for r in con.execute(
            "SELECT d.id, d.ts, d.author, d.abliterated, d.cross_hold, d.cleared, d.project, d.title, d.status,"
            " d.supersedes, substr(d.decision, 1, 300) AS preview, (SELECT COUNT(*) FROM"
            f" decision_stances s WHERE s.decision = d.id) AS stances, {review('decision', 'd')} AS review"
            " FROM decisions d ORDER BY d.id DESC LIMIT 1000"
        )]  # fmt: skip
        for d in snap["decisions"]:
            d["status"], d["superseded_by"] = decision_status(d["status"], d["id"], links), links.get(d["id"], [])
        states = {r[0]: decision_status(r[1], r[0], links) for r in con.execute("SELECT id, status FROM decisions")}
        if like:
            snap["decision_hits"] = {r[0] for r in con.execute(
                "SELECT id FROM decisions WHERE lower(title) LIKE ? OR lower(decision) LIKE ? OR lower(why) LIKE ?"
                " OR lower(alternatives) LIKE ? OR lower(author) LIKE ? OR lower(project) LIKE ?",
                (like,) * 6)} | {i for i, state in states.items() if query.lower() in state}  # fmt: skip
    if "memories" in tables and "decisions" in tables:
        snap["counts"].update(
            memories=con.execute("SELECT COUNT(*) FROM memories").fetchone()[0],
            active_decisions=sum(1 for state in states.values() if state == "active"),
        )
    if "tasks" in tables and "task_notes" in tables:
        online = {a["instance"] for a in snap["agents"] if snap["now"] - (a.get("last_seen") or 0) < ONLINE_WINDOW}
        snap["tasks"] = [dict(r) for r in con.execute(
            "SELECT t.id, t.ts, t.author, t.abliterated, t.cross_hold, t.cleared, t.project, t.title, t.status,"
            " t.assignee, t.depends, t.claimer, t.claimer_name, t.renewed_at, t.changed_at, (SELECT COUNT(*) FROM"
            f" task_notes n WHERE n.task = t.id) AS notes, {review('task', 't')} AS review FROM tasks t"
            " ORDER BY CASE WHEN t.status IN ('done','dropped') THEN 1 ELSE 0 END, t.id DESC LIMIT 1000"
        )]  # fmt: skip
        for t in snap["tasks"]:  # a claim whose session is gone returns to open the next time an agent looks
            t["claim_gone"] = bool(t["claimer"]) and t["status"] in ("claimed", "in_progress", "blocked") and (
                t["claimer"] not in online)  # fmt: skip
        if like:
            snap["task_hits"] = {r[0] for r in con.execute(
                "SELECT id FROM tasks WHERE lower(title) LIKE ? OR lower(details) LIKE ? OR lower(author) LIKE ? OR"
                " lower(project) LIKE ? OR lower(assignee) LIKE ? OR lower(claimer_name) LIKE ? OR status LIKE ? OR"
                " id IN (SELECT task FROM task_notes WHERE lower(note) LIKE ?)", (like,) * 8)}  # fmt: skip
        snap["counts"]["unfinished_tasks"] = con.execute(
            "SELECT COUNT(*) FROM tasks WHERE status NOT IN ('done','dropped')").fetchone()[0]  # fmt: skip
    if "ratings" in tables:
        # The scores per kind of request and model, as agent_hub computes them for agents (useful 1, partly 0.5,
        # wrong 0, averaged), best first within each kind.
        snap["scores"] = hub_module().rating_summary(con)
        snap["counts"]["ratings"] = sum(s["count"] for s in snap["scores"])


def read_hive_item(path: Path, kind: str, item_id: Any) -> Text:
    """One note, decision or task in full, for the user: who wrote it, where it stands at the gate, and for a
    decision the stances with their notes, for a task its history; or (kind "ratings", item [model, kind of request])
    the ratings of a model."""
    if kind == "task":
        return read_task(path, int(item_id))
    if kind == "ratings":
        return read_ratings(path, *json.loads(item_id))
    con = connect(path)
    try:
        table = "memories" if kind == "memory" else "decisions"
        row = con.execute(f"SELECT * FROM {table} WHERE id=?", (item_id,)).fetchone()
        if row is None:
            return lit(f"{'Note' if kind == 'memory' else 'Decision'} #{item_id} no longer exists.")

        def gate_of(ref: str) -> str:
            try:
                g = con.execute("SELECT id, status, note FROM gate WHERE ref=? ORDER BY id DESC LIMIT 1", (ref,)).fetchone()
            except sqlite3.Error:  # an older gate table without ref
                return ""
            if g is None:
                return ""
            note = f": {one_line(g['note'], 200)}" if g["note"] else ""
            return f"gate request #{g['id']} {g['status']}{note}"

        def other_group(r: Any) -> str:
            if r["cleared"] == 1:
                return "cleared by you for both groups"
            request = gate_of(f"{kind}:{r['id']}")
            if r["cleared"] == -1:
                return "denied by you: the other group never reads it" + (f" ({request})" if request else "")
            return request or (
                "the writer's gate holds it for the other group" if r["cross_hold"] else "no gate held it when written")

        out = Text()
        if kind == "memory":
            scope = "every project (hub scope)" if row["scope"] == "hub" else f"project {row['project']}"
            out.append(f"Note #{row['id']}  by {clean(row['author'])} ({group_of(dict(row))}), {ago(row['ts'])}\n",
                       style="bold")  # fmt: skip
            out.append(f"for    {clean(scope)}\n")
            tags = ", ".join(t for t in (row["tags"] or "").split(",") if t)
            if tags:
                out.append(f"tags   {clean(tags)}\n")
            out.append(f"other group: {clean(other_group(row))}\n")
            out.append("x delete · Esc back\n\n", style="bold yellow")
            out.append(clean(row["text"]) + "\n")
            return out
        links = decision_links(con, row["project"])
        state = decision_status(row["status"], row["id"], links)
        out.append(f"Decision #{row['id']} [{state}]  {clean(row['title'])}\n", style="bold")
        out.append(f"by     {clean(row['author'])} ({group_of(dict(row))}), {ago(row['ts'])}, project"
                   f" {clean(row['project'])}\n")  # fmt: skip
        out.append(f"other group: {clean(other_group(row))}\n")
        if row["supersedes"]:
            out.append(f"supersedes decision #{row['supersedes']}\n")
        if state == "superseded":
            out.append(f"superseded by decision {replaced_by(con, links[row['id']])}\n")
        elif state == "withdrawn":
            reason = f": {clean(row['status_note'])}" if row["status_note"] else ""
            out.append(f"withdrawn by {clean(row['changed_by'])}, {ago(row['changed_at'])}{reason}\n")
        keys = "w withdraw · x delete · Esc back" if state == "active" else "x delete · Esc back"
        out.append(keys + "\n\n", style="bold yellow")
        for head, column in (("Decision", "decision"), ("Why", "why"), ("Alternatives considered", "alternatives")):
            if row[column]:
                out.append(head + "\n", style="bold")
                out.append(clean(row[column]) + "\n\n")
        stances = con.execute("SELECT * FROM decision_stances WHERE decision=? ORDER BY ts", (item_id,)).fetchall()
        out.append(f"Stances ({len(stances)})\n", style="bold")
        for st in stances:
            state = {1: "cleared by you for both groups", -1: "denied by you for the other group"}.get(
                st["cleared"]) or gate_of(f"stance:{st['id']}")  # fmt: skip
            out.append(f"  {clean(st['name'])} ({group_of(dict(st))}) {clean(st['stance'])}, {ago(st['ts'])}"
                       + (f"  [{clean(state)}]" if state else "") + "\n", style="cyan")  # fmt: skip
            out.append("    " + clean(st["note"]).replace("\n", "\n    ") + "\n")
        return out
    finally:
        con.close()


def decision_links(con: sqlite3.Connection, project: str | None = None) -> dict[int, list[int]]:
    """Which decisions are superseded, and by which decisions in force (agent_hub.decision_links, as you see them:
    all of them)."""
    return hub_module().decision_links(con, project)


def decision_status(stored: str, decision_id: int, links: dict[int, list[int]]) -> str:
    if stored == "withdrawn":
        return "withdrawn"
    return "superseded" if decision_id in links else "active"


def replaced_by(con: sqlite3.Connection, ids: list[int]) -> str:
    """The decisions that replace one, and for whom: a replacement the writer's gate holds for the other group
    supersedes for that group only once you approve it (or never, if you denied it)."""
    parts = []
    for r in con.execute(f"SELECT id, cleared, cross_hold FROM decisions WHERE id IN ({','.join('?' * len(ids))})"
                         " ORDER BY id", ids):  # fmt: skip
        if r["cleared"] == -1:
            parts.append(f"#{r['id']} (for its writer's group only: you denied it to the other group)")
        elif r["cross_hold"] and r["cleared"] != 1:
            parts.append(f"#{r['id']} (for its writer's group; for the other group once you approve it)")
        else:
            parts.append(f"#{r['id']}")
    return ", ".join(parts)


def gate_state(con: sqlite3.Connection, ref: str) -> str:
    """The latest gate request about a hive item, as the user reads it ('' none)."""
    try:
        g = con.execute("SELECT id, status, note FROM gate WHERE ref=? ORDER BY id DESC LIMIT 1", (ref,)).fetchone()
    except sqlite3.Error:  # an older gate table without ref
        return ""
    if g is None:
        return ""
    return f"gate request #{g['id']} {g['status']}" + (f": {one_line(g['note'], 200)}" if g["note"] else "")


def read_task(path: Path, task_id: int) -> Text:
    """A task in full: who added it, for whom, who claims it, what it waits for, where it stands at the gate, its
    details and its whole history (with each note's state at the gate)."""
    con = connect(path)
    try:
        row = con.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()
        if row is None:
            return lit(f"Task #{task_id} no longer exists.")
        out = Text()
        out.append(f"Task #{row['id']} [{row['status']}]  {clean(row['title'])}\n", style="bold")
        out.append(f"by     {clean(row['author'])} ({group_of(dict(row))}), {ago(row['ts'])}, project"
                   f" {clean(row['project'])}\n")  # fmt: skip
        if row["assignee"]:
            out.append(f"for    {clean(row['assignee'])}\n")
        if row["claimer"]:
            agent = con.execute("SELECT last_seen FROM agents WHERE instance=?", (row["claimer"],)).fetchone()
            gone = row["status"] in ("claimed", "in_progress", "blocked") and (
                agent is None or time.time() - agent["last_seen"] >= ONLINE_WINDOW)  # fmt: skip
            news = f", last news {ago(row['renewed_at'])}" if row["renewed_at"] else ""
            out.append(f"claim  {clean(row['claimer_name'])}, {ago(row['claimed_at'])}{news}"
                       + (" (its session is gone: open again the next time an agent looks)" if gone else "") + "\n")  # fmt: skip
        deps = [int(p) for p in (row["depends"] or "").split(",") if p.strip().isdigit()]
        if deps:
            states = {r["id"]: r["status"] for r in con.execute(
                f"SELECT id, status FROM tasks WHERE id IN ({','.join('?' * len(deps))})", deps)}  # fmt: skip
            out.append("waits  " + ", ".join(f"#{d} ({states.get(d, 'gone')})" for d in deps) + "\n")
        state = {1: "cleared by you for both groups", -1: "denied by you: the other group never reads it"}.get(
            row["cleared"]) or gate_state(con, f"task:{row['id']}") or (
            "the writer's gate holds it for the other group" if row["cross_hold"] else "no gate held it when written")
        out.append(f"other group: {clean(state)}\n")
        out.append("s status · a assign · x delete · Esc back\n\n", style="bold yellow")
        if row["details"]:
            out.append("Details\n", style="bold")
            out.append(clean(row["details"]) + "\n\n")
        notes = con.execute("SELECT * FROM task_notes WHERE task=? ORDER BY id", (task_id,)).fetchall()
        out.append(f"History ({len(notes)})\n", style="bold")
        for n in notes:
            who = clean(n["name"]) + (f" ({group_of(dict(n))})" if n["instance"] else "")
            state = ("" if not n["note"] or n["cleared"] == 1 else "denied by you for the other group"
                     if n["cleared"] == -1 else gate_state(con, f"task_note:{n['id']}"))  # fmt: skip
            out.append(f"  {ago(n['ts'])}, {who}" + (f": {clean(n['what'])}" if n["what"] else "")
                       + (f"  [{clean(state)}]" if state else "") + "\n", style="cyan")  # fmt: skip
            if n["note"]:
                out.append("    " + clean(n["note"]).replace("\n", "\n    ") + "\n")
        return out
    finally:
        con.close()


def read_ratings(path: Path, model: str, category: str) -> Text:
    """The ratings of a model for one kind of request, newest first, with the raters' notes (only the user and the
    rater see those)."""
    con = connect(path)
    try:
        # The score and the counts from every rating, as the Scores list and agents get them (rating_summary); only
        # the listing below stops at the newest RATINGS_SHOWN.
        summary = next((s for s in hub_module().rating_summary(con, category) if s["model"] == model), None)
        rows = con.execute("SELECT * FROM ratings WHERE model=? AND category=? ORDER BY id DESC LIMIT ?",
                           (model, category, RATINGS_SHOWN)).fetchall()  # fmt: skip
        out = Text()
        if summary is None or not rows:
            out.append(f"No ratings of {clean(model)} for {clean(category)} (any more).\n", style="bold")
            return out
        out.append(f"{clean(model)} for {clean(category)}: {summary['score']:.2f} from {summary['count']} rating(s):"
                   f" {summary['useful']} useful, {summary['partly']} partly, {summary['wrong']} wrong\n",
                   style="bold")  # fmt: skip
        out.append("x reset these ratings · Esc back\n\n", style="bold yellow")
        if summary["count"] > len(rows):
            out.append(f"The newest {len(rows)} of them:\n", style="bold")
        for r in rows:
            out.append(f"  {ago(r['ts'])}  {r['verdict']:<7} by {clean(r['rater'])} ({group_of(dict(r))}), answer"
                       f" #{r['message_id']}\n", style="cyan")  # fmt: skip
            if r["note"]:
                out.append("    " + clean(r["note"]).replace("\n", "\n    ") + "\n")
        return out
    finally:
        con.close()


def workspace_changes(con: sqlite3.Connection, path: Path, ws_id: str) -> str:
    """How many files the workers changed, added and deleted in a workspace (agent_hub compares them)."""
    try:
        hub = hub_module(path)
        row = con.execute("SELECT * FROM workspaces WHERE id=?", (ws_id,)).fetchone()
        files = con.execute("SELECT * FROM workspace_files WHERE workspace=?", (ws_id,)).fetchall()
        changes = hub.ws_changes(row, files)[0]
    except Exception as exc:  # e.g. its folder is gone
        return clean(str(exc))[:60]
    return f"{len(changes)} change(s)" if changes else "none"


def read_workspace(path: Path, ws_id: str, shared: bool = False) -> tuple[Text, str | None]:
    """A workspace as the user sees it: who shared what from where, with whom, the hidden names, the notes, and
    the diff with the real paths and names (what apply would write), with its review token (apply writes exactly
    that state, or nothing); `shared`: instead of the diff, the files as the workers got them (neutral paths, names
    hidden), and no token."""
    con = connect(path)
    try:
        row = con.execute("SELECT * FROM workspaces WHERE id=?", (ws_id,)).fetchone()
        if row is None:
            return lit(f"Workspace {ws_id} no longer exists."), None
        files = con.execute("SELECT * FROM workspace_files WHERE workspace=? ORDER BY path", (ws_id,)).fetchall()
        who = con.execute("SELECT name, kind FROM workspace_access WHERE workspace=? ORDER BY id", (ws_id,)).fetchall()
        notes = con.execute(
            "SELECT n.ts, n.name, n.summary, m.body, m.kind, m.held, (SELECT status FROM gate WHERE message_id ="
            " n.message_id ORDER BY id DESC LIMIT 1) AS gate_status FROM workspace_notes n LEFT JOIN messages m ON"
            " m.id = n.message_id WHERE n.workspace=? ORDER BY n.id", (ws_id,)
        ).fetchall()  # fmt: skip
        hub = hub_module(path)
        out = Text()
        label = f' "{row["label"]}"' if row["label"] else ""
        out.append(f"Workspace {row['id']}{clean(label)}  ({row['status']}, shared {ago(row['ts'])} by"
                   f" {clean(row['owner'])})\n", style="bold")  # fmt: skip
        out.append(f"from   {clean(hub.ws_real(row, ''))}\n")
        out.append("with   " + (", ".join(f"{clean(w['name'])} ({w['kind']})" for w in who) or "nobody yet") + "\n")
        pairs = hub._ws_pair_rows(row)
        if pairs:
            out.append("hidden " + ", ".join(f"{clean(h)} as {clean(s)}" + (" (the project folder)" if f else "")
                                             for h, s, f in pairs) + "\n")  # fmt: skip
        out.append(f"task   {clean(row['task'])}\n")
        for n in notes:
            text = n["body"] if n["body"] is not None else n["summary"]
            state = (f" [{n['gate_status']} at the gate]" if n["held"] and n["gate_status"] in ("denied", "expired")
                     else " [waiting at the gate]" if n["held"] else "")  # fmt: skip
            kind = " (done)" if n["kind"] == "done" else ""
            out.append(f"note   {ago(n['ts'])}, {clean(n['name'])}{kind}: {one_line(text, 300)}{state}\n")
        view = "f the diff" if shared else "f the files as shared"
        keys = f"a apply · d apply with deletions · c close · {view} · Esc back" if row["status"] == "open" else "Esc back"
        out.append(keys + "\n\n", style="bold yellow" if row["status"] == "open" else "dim")
        token = None
        if shared:
            out.append("The files as the workers got them (or as last applied):\n\n", style="bold")
            for f in files[:200]:
                out.append(f"===== {clean(f['path'])}  <-  {clean(hub.ws_real(row, f['origin']))} =====\n", style="cyan")
                body = f["base"] if len(f["base"]) <= 20_000 else f["base"][:20_000] + "\n[... cut]"
                out.append(clean(body) + "\n\n")
        elif row["status"] == "open":
            try:
                review = hub.ws_review(row, files, None, family(con, ws_id))
                out.append(clean(review.text))
                token = review.token
            except Exception as exc:
                out.append(f"(cannot compare the files: {clean(exc)})")
        return out, token
    finally:
        con.close()


def family(con: sqlite3.Connection, ws_id: str) -> str:
    """The top workspace a workspace was passed on from (their shared lock), as agent_hub's Store.ws_family."""
    seen, top = {ws_id}, ws_id
    row = con.execute("SELECT parent FROM workspaces WHERE id=?", (ws_id,)).fetchone()
    while row is not None and row["parent"] and row["parent"] not in seen:
        seen.add(row["parent"])
        top = row["parent"]
        row = con.execute("SELECT parent FROM workspaces WHERE id=?", (top,)).fetchone()
    return top


def read_thread(path: Path, thread: str) -> list[dict[str, Any]]:
    con = connect(path)
    try:
        cols = {r[1] for r in con.execute("PRAGMA table_info(messages)")}
        held = "held" if "held" in cols else "0"
        public = "public" if "public" in cols else "0"
        return [dict(r) for r in con.execute(
            f"SELECT id, ts, sender, recipient, kind, body, length(attachments) AS att, meta, {held} AS held,"
            f" {public} AS public FROM messages WHERE thread=? ORDER BY id", (thread,)
        )]  # fmt: skip
    finally:
        con.close()


def read_gate_item(path: Path, gate_id: int) -> dict[str, Any] | None:
    """One request at the gate; a pending one past its deadline reads as expired."""
    con = connect(path)
    try:
        row = con.execute("SELECT * FROM gate WHERE id=?", (gate_id,)).fetchone()
    finally:
        con.close()
    if row is None:
        return None
    item = dict(row)
    item["deadline"] = float(item.get("expires") or (item["ts"] + GATE_DEFAULT_TIMEOUT))
    if item["status"] == "pending" and item["deadline"] <= time.time():
        item["status"], item["note"] = "expired", "nobody decided before the deadline"
    return item


def left(deadline: float, now: float | None = None) -> str:
    secs = max(0, int(deadline - (now or time.time())))
    return f"{secs // 60}m {secs % 60:02d}s"


def hub_module(path: Path | None = None) -> Any:
    """agent_hub.py from this folder, for gate decisions, workspaces and the model list (it uses the same
    environment, so the same database). `path`: the database this dashboard shows; its workspaces are next to it."""
    here = str(Path(__file__).resolve().parent)
    if here not in sys.path:
        sys.path.insert(0, here)
    if "agent_hub" not in sys.modules:
        os.environ["AGENT_HUB_LAZY_STORE"] = "1"  # no shared connection, no database created just by importing
    import agent_hub  # noqa: PLC0415

    if path is not None:
        agent_hub.WORKSPACES_DIR = Path(path).parent / "workspaces"
    return agent_hub


def overview_text(snap: dict[str, Any] | None, path: Path) -> Text:
    out = Text()
    if snap is None:
        out.append(f"No hub database yet at {path}.\n", style="bold")
        out.append("It appears when the first agent starts Agent Hub; this view refreshes on its own.")
        return out
    c, now = snap["counts"], snap["now"]
    if c.get("waiting"):
        out.append(f"{c['waiting']} request(s) waiting for your approval at the gate: press 2.\n\n", style="bold red")
    out.append("Hub\n", style="bold")
    out.append(f"  agents online      {c.get('online', 0)} of {c.get('agents', 0)}\n")
    out.append(f"  last 24 hours      {c.get('threads24', 0)} threads, {c.get('messages24', 0)} messages\n")
    out.append(f"  unanswered         {c.get('open', 0)} request(s) from the last 24 hours\n")
    out.append(f"  reservations       {c.get('reservations', 0)} active\n")
    out.append(f"  workspaces         {c.get('workspaces', 0)} open (press 0)\n")
    switches = {v.setting.key: v.shown for v in snap.get("hive") or []}
    out.append(f"  hive               memory {switches.get('memory.enabled', '?')}, decisions"
               f" {switches.get('decisions.enabled', '?')}; {c.get('memories', 0)} note(s),"
               f" {c.get('active_decisions', 0)} active decision(s) (press h, m, l)\n")
    out.append(f"  tasks, scores      task board {switches.get('tasks.enabled', '?')}, track record"
               f" {switches.get('track.enabled', '?')} (auto-route {switches.get('track.auto_route', '?')});"
               f" {c.get('unfinished_tasks', 0)} unfinished task(s), {c.get('ratings', 0)} rating(s) (press t, s)\n\n")
    tokens = [(name, u) for name, u in snap["usage"].items() if u["in24"] or u["out24"]]
    out.append("Tokens, last 24 hours\n", style="bold")
    for name, u in sorted(tokens, key=lambda x: -(x[1]["in24"] + x[1]["out24"]))[:12]:
        out.append(f"  {clean(name)[:28]:<30}{u['in24']:>12,} in {u['out24']:>12,} out\n")
    if not tokens:
        out.append("  (none)\n")
    out.append("\nRecent\n", style="bold")
    for m in snap["recent"]:
        out.append(f"  {ago(m['ts'], now):>9}  #{m['id']} {one_line(m['sender'], 24)} -> {one_line(m['recipient'], 24)}"
                   f" ({m['kind']}): {one_line(m['preview'], 70)}\n")  # fmt: skip
    if not snap["recent"]:
        out.append("  (no messages yet)\n")
    return out


# --------------------------------------------------------------------------
# Screens
# --------------------------------------------------------------------------


class ThreadScreen(Screen[None]):
    BINDINGS = [Binding("escape", "app.pop_screen", "Back")]

    def __init__(self, path: Path, thread: str) -> None:
        super().__init__()
        self.path, self.thread = path, thread

    def compose(self) -> ComposeResult:
        yield Header()
        yield VerticalScroll(Static(lit(f"Loading thread {self.thread}..."), id="transcript"))
        yield Footer()

    def on_mount(self) -> None:
        self.sub_title = f"thread {clean(self.thread)}"
        self.load()

    @work(thread=True)
    def load(self) -> None:
        try:
            rows = read_thread(self.path, self.thread)
            text = Text()
            for r in rows:
                head = f"#{r['id']}  {clean(r['sender'])} -> {clean(r['recipient'])}  ({r['kind']})  {ago(r['ts'])}"
                if r["held"]:
                    head += "  [held at the gate]"
                head += "  [public]" if r["public"] else "  [private]"
                text.append(head + "\n", style="bold cyan")
                if r["att"]:
                    text.append(f"(plus {r['att']:,} characters of attached files, not shown)\n", style="dim")
                text.append(clean(r["body"]) + "\n\n")
            if not rows:
                text = lit(f"Thread {self.thread} has no messages.")
        except sqlite3.Error as exc:
            text = lit(f"Could not read the thread: {exc}")
        self.app.call_from_thread(self.query_one("#transcript", Static).update, text)


class ReasonScreen(ModalScreen[str | None]):
    BINDINGS = [Binding("escape", "cancel", "Cancel")]

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog"):
            yield Static(lit("Reason for denying (optional, shown to the requester). Enter to deny, Esc to cancel."))
            yield Input(placeholder="reason", id="reason")

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self.dismiss(event.value.strip())

    def action_cancel(self) -> None:
        self.dismiss(None)


class AskScreen(ModalScreen[str | None]):
    """One line of text from the user: Enter gives it, Esc cancels."""

    BINDINGS = [Binding("escape", "cancel", "Cancel")]

    def __init__(self, question: str, value: str = "") -> None:
        super().__init__()
        self.question, self.value = question, value

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog"):
            yield Static(lit(self.question), id="ask-text")
            yield Input(value=self.value, id="answer")

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self.dismiss(event.value.strip())

    def action_cancel(self) -> None:
        self.dismiss(None)


class EditScreen(ModalScreen[str | None]):
    BINDINGS = [Binding("ctrl+s", "save", "Approve with this text"), Binding("escape", "cancel", "Cancel")]

    def __init__(self, content: str) -> None:
        super().__init__()
        self.content = content

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog"):
            yield Static(lit("Edit what will be sent. Ctrl+S approves with this text, Esc cancels."))
            yield TextArea(self.content, id="editor")
            yield Footer()

    def action_save(self) -> None:
        text = self.query_one("#editor", TextArea).text.strip()
        self.dismiss(text or None)

    def action_cancel(self) -> None:
        self.dismiss(None)


class GateScreen(ModalScreen[None]):
    BINDINGS = [
        Binding("a", "approve", "Approve"),
        Binding("d", "deny", "Deny"),
        Binding("e", "edit", "Edit and approve"),
        Binding("escape", "close", "Back"),
    ]

    def __init__(self, path: Path, gate_id: int) -> None:
        super().__init__()
        self.path, self.gate_id = path, gate_id
        self.item: dict[str, Any] | None = None

    def compose(self) -> ComposeResult:
        with Vertical(id="gate-dialog"):
            yield Static(lit(f"Request #{self.gate_id}"), id="gate-head")
            yield VerticalScroll(Static(lit("Loading..."), id="gate-content"))
            yield Footer()

    def on_mount(self) -> None:
        self.load()

    @work(thread=True)
    def load(self) -> None:
        try:
            item = read_gate_item(self.path, self.gate_id)
        except sqlite3.Error as exc:
            item, error = None, str(exc)
        else:
            error = "it no longer exists"
        self.app.call_from_thread(self.show, item, error)

    def show(self, item: dict[str, Any] | None, error: str) -> None:
        self.item = item
        head, body = self.query_one("#gate-head", Static), self.query_one("#gate-content", Static)
        if item is None:
            head.update(lit(f"Request #{self.gate_id}: {error}"))
            return
        text = Text()
        status = item["status"] + (f", expires in {left(item['deadline'])}" if item["status"] == "pending" else "")
        text.append(f"#{item['id']}  {item['kind']}  ({status}, {ago(item['ts'])})\n", style="bold")
        text.append("from  ", style="dim")
        text.append(clean(item["origin"]) + "\n")
        text.append("to    ", style="dim")
        text.append(clean(item["target"]) + "\n")
        keys = "a approve · d deny" + (" · e edit and approve" if item["editable"] else "") + " · Esc back"
        text.append(keys, style="bold yellow" if item["status"] == "pending" else "dim")
        head.update(text)
        body.update(lit(item["content"]))

    def action_close(self) -> None:
        self.dismiss(None)

    def _pending(self) -> bool:
        if self.item is not None and self.item["status"] == "pending" and self.item["deadline"] <= time.time():
            self.item["status"] = "expired"  # the deadline passed while it was open here
        if self.item is None or self.item["status"] != "pending":
            status = self.item["status"] if self.item is not None else "gone"
            self.app.notify(f"This request is {clean(status)}: it can no longer be decided.", severity="warning",
                            markup=False)  # fmt: skip
            return False
        return True

    def action_approve(self) -> None:
        if self._pending():
            self.app.decide(self.gate_id, "approved")
            self.dismiss(None)

    def action_deny(self) -> None:
        if not self._pending():
            return

        def done(reason: str | None) -> None:
            if reason is not None:
                self.app.decide(self.gate_id, "denied", reason)
                self.dismiss(None)

        self.app.push_screen(ReasonScreen(), done)

    def action_edit(self) -> None:
        if not self._pending():
            return
        if not self.item["editable"]:
            self.app.notify("This request cannot be edited, only approved or denied.", severity="warning", markup=False)
            return

        def done(text: str | None) -> None:
            if text is not None:
                self.app.decide(self.gate_id, "approved", "", text)
                self.dismiss(None)

        self.app.push_screen(EditScreen(self.item["content"]), done)


class ConfirmScreen(ModalScreen[str | None]):
    """A question answered with one key: y (or the other keys offered), anything else cancels."""

    BINDINGS = [Binding("escape", "cancel", "Cancel")]

    def __init__(self, question: str, keys: str = "y") -> None:
        super().__init__()
        self.question, self.keys = question, keys

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog"):
            yield Static(lit(self.question), id="confirm-text")

    def on_key(self, event: Any) -> None:
        event.stop()
        self.dismiss(event.key if event.key in self.keys else None)

    def action_cancel(self) -> None:
        self.dismiss(None)


class WorkspaceScreen(Screen[None]):
    """One workspace: what was shared from where, with whom, and the workers' changes with the real paths."""

    BINDINGS = [
        Binding("a", "apply(False)", "Apply"),
        Binding("d", "apply(True)", "Apply with deletions"),
        Binding("c", "close_workspace", "Close"),
        Binding("f", "toggle_files", "Diff / files as shared"),
        Binding("escape", "app.pop_screen", "Back"),
    ]

    def __init__(self, path: Path, ws_id: str) -> None:
        super().__init__()
        self.path, self.ws_id = path, ws_id
        self.shared = False
        self.token: str | None = None  # the review token of the diff last shown here: apply writes exactly that

    def compose(self) -> ComposeResult:
        yield Header()
        yield VerticalScroll(Static(lit(f"Loading workspace {self.ws_id}..."), id="workspace-text"))
        yield Footer()

    def on_mount(self) -> None:
        self.sub_title = f"workspace {clean(self.ws_id)}"
        self.load()

    @work(thread=True)
    def load(self) -> None:
        try:
            text, token = read_workspace(self.path, self.ws_id, self.shared)
        except Exception as exc:
            text, token = lit(f"Could not read the workspace: {exc}"), None
        self.app.call_from_thread(self.show, text, token)

    def show(self, text: Text, token: str | None) -> None:
        if token is not None or not self.shared:  # the files as shared change nothing about what was reviewed
            self.token = token
        self.query_one("#workspace-text", Static).update(text)

    def action_toggle_files(self) -> None:
        self.shared = not self.shared
        self.load()

    def action_apply(self, delete: bool) -> None:
        if self.token is None:
            why = ("These are the files as shared: press f for the diff, read it, then apply." if self.shared else
                   "There is no diff to apply: it is still loading, or it could not be computed (the screen says why).")  # fmt: skip
            self.app.notify(why, severity="warning", markup=False)
            return
        self.app.confirm_workspace(self.ws_id, "apply", delete, self.load, self.token)

    def action_close_workspace(self) -> None:
        self.app.confirm_workspace(self.ws_id, "close", False, self.load)


class HiveItemScreen(Screen[None]):
    """One note, decision or task of the hive in full, or the ratings of a model: x deletes it (resets the ratings),
    w withdraws a decision, s changes a task's status and a assigns it, each after a yes."""

    BINDINGS = [
        Binding("x", "delete", "Delete / reset"),
        Binding("w", "withdraw", "Withdraw"),
        Binding("s", "task('status')", "Status"),
        Binding("a", "task('assign')", "Assign"),
        Binding("escape", "app.pop_screen", "Back"),
    ]

    def __init__(self, path: Path, kind: str, item_id: Any) -> None:
        super().__init__()
        self.path, self.kind, self.item_id = path, kind, item_id

    def compose(self) -> ComposeResult:
        yield Header()
        yield VerticalScroll(Static(lit("Loading..."), id="hive-item"))
        yield Footer()

    def on_mount(self) -> None:
        if self.kind == "ratings":
            model, category = json.loads(self.item_id)
            self.sub_title = clean(f"ratings of {model} for {category}")
        else:
            what = "note" if self.kind == "memory" else self.kind
            self.sub_title = f"{what} #{self.item_id}"
        self.load()

    @work(thread=True)
    def load(self) -> None:
        try:
            text = read_hive_item(self.path, self.kind, self.item_id)
        except sqlite3.Error as exc:
            text = lit(f"Could not read it: {exc}")
        self.app.call_from_thread(self.query_one("#hive-item", Static).update, text)

    def action_delete(self) -> None:
        self.app.confirm_hive(self.kind, "reset" if self.kind == "ratings" else "delete", self.item_id, self.load)

    def action_withdraw(self) -> None:
        if self.kind == "decision":
            self.app.confirm_hive(self.kind, "withdraw", self.item_id, self.load)

    def action_task(self, what: str) -> None:
        if self.kind == "task":
            self.app.change_task(int(self.item_id), what, self.load)


# --------------------------------------------------------------------------
# The dashboard
# --------------------------------------------------------------------------

HELP = """Agent Hub dashboard (database: {path})

Keys
  1-9, 0     switch tabs: overview, approvals, agents, threads, messages, reservations, models, errors, help (also
             ? and F1), workspaces (0)
  h, m, l    the hive: its settings (h), shared memory (m), the decision log (l)
  t, s       the hive: the task board (t), the model track record's scores (s)
  /          filter the current table (Esc clears); in memory, decisions and tasks it looks in the whole text
  Enter      open: a thread, a request at the gate, a message's thread, a workspace, a note, a decision, a task,
             a model's ratings; on a hive setting: change it (asks first)
  k          messages: cycle the kind filter (all, requests, replies, errors, discussions, public posts)
  a / d / c  workspaces: apply the changes / apply them with deletions (from the list, these open the diff first:
             apply writes exactly what you read there) / close it (each asks first)
  x          hive: drop your value for the selected setting / delete the selected note, decision or task / reset
             the selected ratings (asks first)
  w          decisions: withdraw the selected decision (asks first)
  s / a      in a task: change its status / who it is meant for (asks first)
  r          refresh now (it also refreshes every {refresh:g}s; AGENT_HUB_TUI_REFRESH changes that)
  q          quit

Tabs
  Overview      counts, tokens per model in the last 24 hours, the latest messages
  Approvals     requests waiting at the gate. Enter shows exactly what would be sent; a approves, d denies
                (with a reason the requester sees), e edits and approves (web searches, screenshots, agent
                messages, posts and replies). Nothing held at the gate is sent before you decide. "publish"
                requests ask whether every agent may read something public (a post, a public answer, a
                discussion's transcript). "memory", "decision" and "stance" requests ask whether agents of the
                other gate group may read a note, a decision or a stance (notes and stances can be edited; a
                denial stands). A request nobody decides on in time expires: then it can no longer be approved.
  Agents        live sessions and their status; the panel below shows the selected agent's reservations and
                messages
  Threads       conversations, newest first; Enter opens the transcript
  Messages      the newest messages of every kind, private and public (you see everything; agents see
                only their own exchanges and public messages)
  Reservations  files agents said they are editing
  Models        configured models and their use in the last 7 days
  Errors        failed requests and discussion turns
  Workspaces    pieces of projects agents shared for isolated work (share): who shared what from where, with
                whom, and how many files the workers changed. Enter shows the diff with the real paths and names
                (what apply writes), the hidden names and the notes; f there shows the files as the workers got
                them (also while a share waits at the gate). a applies exactly the diff on screen (if a worker
                changed the workspace since, nothing is applied and the diff is refreshed; files changed on disk
                since they were shared are refused as conflicts; deletions only with d), c closes it (its folder
                and unapplied changes are deleted). You can apply a workspace whose agent session has ended.
  Hive (h)      the optional hive features (all off until you turn them on) and their options: the value in force
                and where it comes from (your choice, made here or with `agent_hub.py settings`; else the
                environment or keys.env, as this dashboard sees them; else the default). Only you change them:
                agents cannot. Enter changes the selected one (a switch flips, an option asks for its new value;
                each asks for a yes); x drops your choice. Every agent session uses a change at its next tool call.
                A session's own environment may differ from this dashboard's; your choice wins over both.
  Memory (m)    notes agents saved (remember): for which project (or every project), by whom and which gate
                group, tags, and whether the other group may read it. Enter shows one in full; x deletes it.
  Decisions (l) the decision log: status, title, author, how many stances. Enter shows one in full with the stances
                and their notes; w withdraws it, x deletes it (the confirmation says what that does to the decision
                it superseded). A decision is superseded while a later one in force replaces it, for each agent
                only once it can read that one (another group's replacement: once you approved it).
  Tasks (t)     the task board: status, project, title, who added it, for whom, who claims it (a claim whose
                session is gone returns to open the next time an agent looks), what it waits for, how many notes,
                whether the other group may read it. Enter shows one in full with its history; there s sets its
                status (open drops the claim; blocked, done, dropped), a assigns it to an agent name (empty: nobody),
                x deletes it (tasks that waited for it no longer do). "task" and "task_note" requests at the gate
                ask whether the other group may read a task or a note.
  Scores (s)    the model track record: per kind of request and model, the score (useful 1, partly 0.5, wrong 0,
                averaged) and the counts. Enter shows the ratings with the raters' notes (only you see those);
                x resets the selected model's ratings for that kind.

The dashboard only reads the database, except for your decisions at the gate, applying or closing workspaces, and the
hive (settings, deleting notes, withdrawing or deleting decisions, changing or deleting tasks, resetting ratings).
"""


HIVE_HINT = ("The hive's optional features and their options. Only you change them (agents cannot): Enter changes the"
             " selected one, asking first; x drops your value (then the environment, keys.env or the default"
             " applies). Every agent session uses a change at its next tool call.")


def review_state(item: dict[str, Any]) -> str:
    """Whether agents of the other gate group may read a note or a decision."""
    if item.get("cleared") == 1:
        return "cleared by you"
    if item.get("cleared") == -1:
        return "denied by you"
    review = item.get("review")
    if review == "pending":
        return "waiting at the gate"
    if review in ("denied", "expired"):
        return review
    return "held for it" if item.get("cross_hold") else "-"


def group_of(item: dict[str, Any]) -> str:
    value = item.get("abliterated")
    return "?" if value is None else "abliterated" if value else "standard"


class HubDashboard(App[None]):
    TITLE = "Agent Hub"
    CSS = """
    #filter { display: none; }
    #filter.shown { display: block; }
    DataTable { height: 1fr; }
    #agent-detail { height: 12; border: tall $panel; }
    #approvals-done { height: 12; }
    #messages-kind { height: 1; }
    #gate-dialog, #dialog { width: 92%; height: 90%; border: thick $warning; background: $surface; padding: 0 1; }
    #gate-head { height: auto; }
    #editor { height: 1fr; }
    GateScreen, ReasonScreen, EditScreen, ConfirmScreen, AskScreen { align: center middle; }
    ConfirmScreen #dialog, AskScreen #dialog { height: auto; }
    #hive-hint { height: auto; }
    """
    BINDINGS = [
        *[Binding(TAB_KEYS[tab], f"tab('{tab}')", tab.title(), show=False) for tab in TABS],
        *[Binding(key, f"tab('{tab}')", tab.title(), show=False) for tab, key in HIVE_TAB_KEYS.items()],
        Binding("question_mark", "tab('help')", "Help", show=False),
        Binding("f1", "tab('help')", "Help", show=False),
        Binding("slash", "filter", "Filter"),
        Binding("k", "kind", "Kind", show=False),
        Binding("a", "workspace('apply', False)", "Apply", show=False),
        Binding("d", "workspace('apply', True)", "Apply with deletions", show=False),
        Binding("c", "workspace('close', False)", "Close", show=False),
        Binding("x", "hive_remove", "Delete / reset", show=False),
        Binding("w", "withdraw", "Withdraw", show=False),
        Binding("r", "refresh_now", "Refresh"),
        Binding("q", "quit", "Quit"),
    ]

    def __init__(self, path: Path | None = None) -> None:
        super().__init__()
        self.path = path or db_path()
        self.snapshot: dict[str, Any] | None = None
        self.filter_text = ""
        self.kind = "all"
        self.models: list[dict[str, Any]] | None = None
        self.models_error = ""

    def compose(self) -> ComposeResult:
        yield Header()
        # Disabled while hidden: otherwise it can take the first focus and swallow the tab keys.
        yield Input(placeholder="filter the table (Esc clears)", id="filter", disabled=True)
        with TabbedContent(initial="overview", id="tabs"):
            with TabPane("1 Overview", id="overview"):
                yield VerticalScroll(Static(lit("Loading..."), id="overview-text"))
            with TabPane("2 Approvals", id="approvals"):
                with Vertical():
                    yield DataTable(id="approvals-table")
                    yield DataTable(id="approvals-done")
            with TabPane("3 Agents", id="agents"):
                with Vertical():
                    yield DataTable(id="agents-table")
                    yield Static(lit("Select an agent to see its reservations and messages."), id="agent-detail")
            with TabPane("4 Threads", id="threads"):
                yield DataTable(id="threads-table")
            with TabPane("5 Messages", id="messages"):
                with Vertical():
                    yield Static(lit("kind: all (press k to change)"), id="messages-kind")
                    yield DataTable(id="messages-table")
            with TabPane("6 Reservations", id="reservations"):
                yield DataTable(id="reservations-table")
            with TabPane("7 Models", id="models"):
                yield DataTable(id="models-table")
            with TabPane("8 Errors", id="errors"):
                yield DataTable(id="errors-table")
            with TabPane("9 Help", id="help"):
                yield VerticalScroll(Static(lit(HELP.format(path=self.path, refresh=refresh_seconds())), id="help-text"))
            with TabPane("0 Workspaces", id="workspaces"):
                yield DataTable(id="workspaces-table")
            with TabPane("h Hive", id="hive"):
                with Vertical():
                    yield Static(lit(HIVE_HINT), id="hive-hint")
                    yield DataTable(id="hive-table")
            with TabPane("m Memory", id="memory"):
                yield DataTable(id="memory-table")
            with TabPane("l Decisions", id="decisions"):
                yield DataTable(id="decisions-table")
            with TabPane("t Tasks", id="tasks"):
                yield DataTable(id="tasks-table")
            with TabPane("s Scores", id="scores"):
                yield DataTable(id="scores-table")
        yield Footer()

    def on_mount(self) -> None:
        columns = {
            "approvals-table": ("#", "waiting", "left", "kind", "from", "to", "what would be sent"),
            "approvals-done": ("#", "decided", "status", "kind", "from", "to", "note"),
            "agents-table": ("name", "status", "description", "folder", "last seen"),
            "threads-table": ("thread", "participants", "messages", "last activity", "first prompt"),
            "messages-table": ("#", "time", "kind", "from", "to", "thread", "text"),
            "reservations-table": ("path", "held by", "minutes left", "reason"),
            "models-table": ("name", "label", "provider", "model id", "status", "vision", "calls 7d", "tokens in",
                             "tokens out", "avg s", "errors"),  # fmt: skip
            "errors-table": ("#", "time", "from", "to", "thread", "error"),
            "workspaces-table": ("id", "status", "owner", "with", "files", "changed", "shared", "from"),
            "hive-table": ("setting", "value", "from", "allowed", "what it does"),
            "memory-table": ("#", "saved", "for", "by", "tags", "other group", "note"),
            "decisions-table": ("#", "status", "decided", "project", "by", "title", "stances", "other group"),
            "tasks-table": ("#", "status", "project", "title", "by", "for", "claimed by", "waits for", "notes",
                            "other group", "changed"),  # fmt: skip
            "scores-table": ("kind", "model", "score", "useful", "partly", "wrong", "ratings", "last rated"),
        }
        for table_id, names in columns.items():
            table = self.query_one(f"#{table_id}", DataTable)
            table.cursor_type = "row"
            table.zebra_stripes = True
            table.add_columns(*names)
        self.sub_title = str(self.path)
        self.load_models()
        self.refresh_now()
        self.set_interval(refresh_seconds(), self.refresh_now)

    # -- data -------------------------------------------------------------

    def refresh_now(self) -> None:
        try:
            scan = self.query_one(TabbedContent).active == "workspaces"
        except Exception:
            scan = False
        self.load_snapshot(scan)

    def action_refresh_now(self) -> None:
        self.load_models()
        self.refresh_now()

    @work(thread=True, exclusive=True, group="snapshot")
    def load_snapshot(self, scan: bool = False) -> None:
        query = self.filter_text.strip()
        try:
            snap, error = read_snapshot(self.path, scan, query), ""
            if snap is not None:
                snap["query"] = query
                try:  # the effective values (agent_hub knows the settings, their defaults and the environment)
                    snap["hive"], snap["hive_error"] = hub_module().hive_values(snap["settings"]), ""
                except Exception as exc:
                    snap["hive"], snap["hive_error"] = [], f"settings unavailable: {type(exc).__name__}: {exc}"
        except sqlite3.Error as exc:  # e.g. locked for a moment: try again next tick
            snap, error = self.snapshot, str(exc)
        if snap is not None and self.snapshot is not None and not scan:  # keep the last counts while hidden
            known = {w["id"]: w.get("changed") for w in self.snapshot.get("workspaces", [])}
            for w in snap.get("workspaces", []):
                w["changed"] = known.get(w["id"])
        self.call_from_thread(self.show, snap, error)

    @work(thread=True, exclusive=True, group="models")
    def load_models(self) -> None:
        try:
            registry = hub_module().Registry()
            models = [{"alias": alias, "label": m.label, "provider": m.provider, "model": m.model,
                       "status": "ready" if m.ready else f"unavailable: {m.key_problem}",
                       "vision": getattr(m, "vision", None)} for alias, m in registry.models.items()]  # fmt: skip
            error = ""
        except Exception as exc:  # the model list is a nice-to-have
            models, error = [], f"model list unavailable: {type(exc).__name__}: {exc}"
        self.call_from_thread(self._set_models, models, error)

    def _set_models(self, models: list[dict[str, Any]], error: str) -> None:
        self.models, self.models_error = models, error
        self.show(self.snapshot)

    def decide(self, gate_id: int, status: str, note: str = "", edited: str | None = None) -> None:
        self._decide(gate_id, status, note, edited)

    @work(thread=True, group="decide")
    def _decide(self, gate_id: int, status: str, note: str, edited: str | None) -> None:
        try:
            store = hub_module().Store(self.path)  # a connection of this thread's own
            try:
                done = store.gate_decide(gate_id, status, note, edited)
                settled = store.gate_get(gate_id)
            finally:
                store.db.close()
            now = settled["status"] if settled is not None else "gone"
            text = (f"Request #{gate_id} {status}." if done else f"Request #{gate_id} is {now} (decided elsewhere,"
                    " or past its deadline); nothing changed.")  # fmt: skip
            severity = "information" if done else "warning"
        except Exception as exc:
            text = (f"Approvals unavailable: {type(exc).__name__}: {exc}. Use `uv run agent_hub.py gate` in a"
                    " terminal instead.")  # fmt: skip
            severity = "error"
        # Every notification is literal text: markup (and its click actions) in a name an agent chose does nothing.
        self.call_from_thread(self.notify, clean(text), severity=severity, markup=False)
        self.call_from_thread(self.refresh_now)

    def confirm_workspace(self, ws_id: str, what: str, delete: bool, then: Any = None, token: str | None = None) -> None:
        """Ask, then apply (exactly the reviewed diff: `token`) or close a workspace (agent_hub does it, in a worker
        thread)."""
        if what == "close":
            question = (f"Close workspace {ws_id}? Its folder, and any changes not applied, are deleted; nobody can use"
                        " it any more. y closes, any other key cancels.")  # fmt: skip
        else:
            question = (f"Apply the changes shown here (workspace {ws_id}) to the files they came from"
                        + (", deleting the files they deleted" if delete else " (no deletions: d applies them too)")
                        + "? Files changed on disk since they were shared are not touched; if a worker changed the"
                        " workspace since the diff was shown, nothing is applied. y applies, any other key cancels.")  # fmt: skip

        def answered(key: str | None) -> None:
            if key == "y":
                self._workspace_act(ws_id, what, delete, then, token)

        self.push_screen(ConfirmScreen(question), answered)

    @work(thread=True, group="workspace")
    def _workspace_act(self, ws_id: str, what: str, delete: bool, then: Any, token: str | None = None) -> None:
        try:
            hub = hub_module(self.path)
            store = hub.Store(self.path)  # a connection of this thread's own
            try:
                if what == "close":
                    text = f"Closed {', '.join(store.ws_close(ws_id))}."
                else:
                    text = hub.ws_apply(store, ws_id, None, delete, token or "")
            finally:
                store.db.close()
            severity = "warning" if "NOT APPLIED" in text else "information"
        except Exception as exc:
            stale = type(exc).__name__ == "WsStale"
            text = (f"{exc}. The diff on screen is refreshed: read it, then apply again." if stale
                    else f"Workspace {ws_id}: {type(exc).__name__}: {exc}")  # fmt: skip
            severity = "warning" if stale else "error"
        self.call_from_thread(self.notify, clean(text)[:1500], severity=severity, timeout=12, markup=False)
        self.call_from_thread(self.refresh_now)
        if then is not None:
            self.call_from_thread(then)

    def action_workspace(self, what: str, delete: bool) -> None:
        """a / d / c on the Workspaces tab, for the selected workspace: close asks first; apply opens the diff first
        (apply writes exactly what you read there)."""
        try:  # on another screen (a transcript) there is no tab
            if self.query_one(TabbedContent).active != "workspaces":
                return
            table = self.query_one("#workspaces-table", DataTable)
            ws_id = table.coordinate_to_cell_key(table.cursor_coordinate).row_key.value if table.row_count else None
        except Exception:
            return
        if ws_id is None:
            return
        row = next((w for w in (self.snapshot or {}).get("workspaces", []) if w["id"] == ws_id), None)
        if row is None or row["status"] != "open":
            self.notify(f"Workspace {clean(ws_id)} is closed.", severity="warning", markup=False)
            return
        if what == "apply":
            self.push_screen(WorkspaceScreen(self.path, ws_id))
            self.notify("Read the changes, then press a (or d, with deletions) here to apply exactly them.",
                        markup=False)  # fmt: skip
            return
        self.confirm_workspace(ws_id, what, delete)

    # -- the hive: settings, notes, decisions (agent_hub's Store writes, in a worker thread) ----------------

    def selected(self, table_id: str) -> str | None:
        try:
            table = self.query_one(f"#{table_id}", DataTable)
            return table.coordinate_to_cell_key(table.cursor_coordinate).row_key.value if table.row_count else None
        except Exception:
            return None

    def change_setting(self, key: str) -> None:
        """Enter on a setting: a switch flips, an option asks for its new value; either way, only after a yes."""
        current = next((v for v in (self.snapshot or {}).get("hive") or [] if v.setting.key == key), None)
        if current is None:
            self.notify("The settings are not loaded (yet).", severity="warning", markup=False)
            return
        setting = current.setting
        now = f"It is {current.shown} now ({current.detail})."
        tail = " Every agent session uses it from its next tool call. y confirms, any other key cancels."

        def confirm(value: Any) -> None:
            def answered(answer: str | None) -> None:
                if answer == "y":
                    self._hive_act("setting", "set", key, None, value)

            verb = f"Turn {key} {setting.show(value)}" if setting.kind == "bool" else f"Set {key} to {setting.show(value)}"
            self.push_screen(ConfirmScreen(f"{verb}? {now} {setting.description}{tail}"), answered)

        if setting.kind == "bool":
            confirm(not current.value)
            return

        def typed(text: str | None) -> None:
            if not text:
                return
            try:
                value = setting.parse(text)
            except ValueError as exc:
                self.notify(clean(str(exc)), severity="warning", markup=False)
                return
            confirm(value)

        self.push_screen(AskScreen(f"New value for {key}: {setting.allowed()}. {now} {setting.description} Enter"
                                   " continues (you confirm next), Esc cancels.", str(current.value)), typed)  # fmt: skip

    def confirm_hive(self, kind: str, what: str, item: Any, then: Any = None) -> None:
        """Ask, then drop a setting's value, delete a note, or withdraw or delete a decision."""
        if kind == "setting":
            # By the settings table itself (as the control panel and the command line do): a stored value that cannot
            # be used (hand-written, or from another version) is dropped as well.
            current = next((v for v in (self.snapshot or {}).get("hive") or [] if v.setting.key == item), None)
            if item not in ((self.snapshot or {}).get("settings") or {}):
                where = f": it comes from {current.detail}" if current is not None else ""
                self.notify(f"{item} has no value of yours to drop{where}.", severity="warning", markup=False)
                return
            unusable = current is not None and current.problem and current.source != "user"
            question = (f"Drop your value for {item}" + (" (stored in the settings table, but it cannot be used)"
                        if unusable else "") + ", so the environment, keys.env or the default applies again?"
                        " y confirms, any other key cancels.")  # fmt: skip
        elif kind == "memory":
            question = f"Delete note #{item} for good? No agent can recall it after that. y deletes, any other key cancels."
        elif kind == "task":
            question = (f"Delete task #{item} and its history for good? Tasks that wait for it no longer do. y deletes,"
                        " any other key cancels.")  # fmt: skip
        elif kind == "ratings":
            model, category = json.loads(item)
            question = clean(f"Reset the ratings of {model} for {category}? Its score there starts again from nothing"
                             " (auto-route passes it over until it has enough ratings again). y resets, any other key"
                             " cancels.")  # fmt: skip
        elif what == "withdraw":
            question = (f"Withdraw decision #{item}? Agents see it as withdrawn by you.{self._chain_effect(item, True)}"
                        " y withdraws, any other key cancels.")  # fmt: skip
        else:
            question = (f"Delete decision #{item} and its stances for good?{self._chain_effect(item, False)} y deletes,"
                        " any other key cancels.")  # fmt: skip

        def answered(answer: str | None) -> None:
            if answer == "y":
                self._hive_act(kind, what, item, then)

        self.push_screen(ConfirmScreen(question), answered)

    def _chain_effect(self, item: Any, withdraw: bool) -> str:
        """What withdrawing or deleting a decision does to the chain of decisions it is in."""
        row = next((d for d in (self.snapshot or {}).get("decisions", []) if str(d["id"]) == str(item)), None)
        if row is None or not row["supersedes"]:
            return ""
        newer = row.get("superseded_by") or []
        if newer and not withdraw:
            return (f" Decision #{', #'.join(map(str, newer))}, which replaces it, then replaces #{row['supersedes']}"
                    " directly: that one stays superseded.")  # fmt: skip
        if newer or row["status"] == "withdrawn":
            return ""
        return (f" Decision #{row['supersedes']}, which it replaced, is in force again (unless another decision"
                " replaces it).")  # fmt: skip

    @work(thread=True, group="hive")
    def _hive_act(self, kind: str, what: str, item: Any, then: Any = None, value: Any = None) -> None:
        try:
            hub = hub_module()
            store = hub.Store(self.path)  # a connection of this thread's own
            try:
                if kind == "setting" and what == "set":
                    stored = store.setting_set(item, value, "dashboard")
                    text = f"{item} is now {hub.SETTINGS[item].show(stored)} for every session (from its next tool call)."
                elif kind == "setting":
                    done = store.setting_reset(item)
                    text = f"Your value for {item} is dropped." if done else f"{item} had no value of yours."
                elif kind == "memory":
                    done = store.memory_delete(int(item), "the user deleted it")
                    text = f"Note #{item} deleted." if done else f"Note #{item} was already gone."
                elif kind == "task" and what == "status":
                    done = store.task_set_status(int(item), value)
                    text = f"Task #{item} is {value} now." if done else f"Task #{item} is gone, or already {value}."
                elif kind == "task" and what == "assign":
                    done = store.task_assign(int(item), value)
                    whom = f"meant for {value}" if value else "meant for anyone"
                    text = f"Task #{item} is {whom} now." if done else f"Task #{item} is gone, or already {whom}."
                elif kind == "task":
                    done = store.task_delete(int(item))
                    text = f"Task #{item} deleted." if done else f"Task #{item} was already gone."
                elif kind == "ratings":
                    model, category = json.loads(item)
                    count = store.ratings_reset(model, category)
                    text = f"{count} rating(s) of {model} for {category} deleted."
                elif what == "withdraw":
                    done = store.decision_withdraw(int(item), "the user")
                    text = f"Decision #{item} withdrawn." if done else f"Decision #{item} is not active: nothing changed."
                else:
                    done = store.decision_delete(int(item))
                    text = f"Decision #{item} deleted." if done else f"Decision #{item} was already gone."
            finally:
                store.db.close()
            severity = "information"
        except Exception as exc:
            text, severity = f"Could not change it: {type(exc).__name__}: {exc}", "error"
        self.call_from_thread(self.notify, clean(text), severity=severity, markup=False)
        self.call_from_thread(self.refresh_now)
        if then is not None:
            self.call_from_thread(then)

    def action_hive_remove(self) -> None:
        """x: on the Hive tab, drop your value for the selected setting; on Memory or Decisions, delete the selected
        note or decision (each asks first)."""
        try:
            tab = self.query_one(TabbedContent).active
        except Exception:
            return
        table, kind = {"hive": ("hive-table", "setting"), "memory": ("memory-table", "memory"),
                       "decisions": ("decisions-table", "decision"), "tasks": ("tasks-table", "task"),
                       "scores": ("scores-table", "ratings")}.get(tab, (None, None))  # fmt: skip
        key = self.selected(table) if table else None
        if key is not None and key != "error":
            self.confirm_hive(kind, "reset" if kind == "ratings" else "delete", key)

    def change_task(self, task_id: int, what: str, then: Any = None) -> None:
        """s / a in a task: ask for its new status (or the agent name it is meant for), then for a yes."""
        if what == "status":
            question = (f"New status for task #{task_id}: one of {', '.join(TASK_USER_STATUSES)} (open drops the"
                        " claim, so another agent can take it). Enter continues (you confirm next), Esc cancels.")  # fmt: skip
        else:
            question = (f"The agent name task #{task_id} is meant for (empty: anyone). Enter continues (you confirm"
                        " next), Esc cancels.")  # fmt: skip

        def typed(text: str | None) -> None:
            if text is None:
                return
            value = text.strip().lower()
            if what == "status" and value not in TASK_USER_STATUSES:
                self.notify(f"The status must be one of {', '.join(TASK_USER_STATUSES)}.", severity="warning",
                            markup=False)  # fmt: skip
                return
            if what == "assign" and value and not re.match(r"^[a-z0-9][a-z0-9_-]{1,31}$", value):
                self.notify("An agent name is 2-32 characters: lowercase letters, digits, - and _.", severity="warning",
                            markup=False)  # fmt: skip
                return

            def answered(answer: str | None) -> None:
                if answer == "y":
                    self._hive_act("task", what, task_id, then, value)

            change = f"Set task #{task_id} to {value}" if what == "status" else (
                f"Assign task #{task_id} to {value}" if value else f"Assign task #{task_id} to anyone")  # fmt: skip
            self.push_screen(ConfirmScreen(f"{change}? Every agent sees the change in its history. y confirms, any"
                                           " other key cancels."), answered)  # fmt: skip

        self.push_screen(AskScreen(question), typed)

    def action_withdraw(self) -> None:
        try:
            if self.query_one(TabbedContent).active != "decisions":
                return
        except Exception:
            return
        key = self.selected("decisions-table")
        row = next((d for d in (self.snapshot or {}).get("decisions", []) if str(d["id"]) == key), None)
        if row is not None and row["status"] != "active":
            self.notify(f"Decision #{key} is {row['status']}: only an active one can be withdrawn.", severity="warning",
                        markup=False)  # fmt: skip
        elif key is not None:
            self.confirm_hive("decision", "withdraw", key)

    # -- display ------------------------------------------------------------

    def fill(self, table_id: str, rows: list[tuple[str, list[Any]]], searched: bool = False) -> None:
        """`searched`: the rows are already what the filter found (in the whole text of the hive's notes)."""
        table = self.query_one(f"#{table_id}", DataTable)
        needle = self.filter_text.lower()
        if needle and not searched:
            rows = [r for r in rows if needle in " ".join(str(c) for c in r[1]).lower()]
        current = None
        if table.row_count:
            try:
                current = table.coordinate_to_cell_key(table.cursor_coordinate).row_key.value
            except Exception:
                current = None
        table.clear()
        for key, cells in rows:
            table.add_row(*[c if isinstance(c, Text) else lit(c) for c in cells], key=key)
        if current is not None:
            try:
                table.move_cursor(row=table.get_row_index(current))
            except Exception:
                pass

    def show(self, snap: dict[str, Any] | None, error: str = "") -> None:
        try:
            box = self.query_one("#overview-text", Static)
        except NoMatches:  # the dashboard is closing: a snapshot read meanwhile has nowhere to go
            return
        self.snapshot = snap
        overview = overview_text(snap, self.path)
        if error:
            overview = Text(f"(could not refresh: {clean(error)})\n\n", style="yellow") + overview
        box.update(overview)
        waiting = (snap or {}).get("counts", {}).get("waiting", 0)
        self.sub_title = f"{waiting} waiting at the gate · {self.path}" if waiting else str(self.path)
        try:
            self.query_one(TabbedContent).get_tab("approvals").label = (
                f"2 Approvals ({waiting})" if waiting else "2 Approvals"
            )
        except Exception:
            pass
        snap = snap or {"now": time.time(), "agents": [], "messages": [], "threads": [], "reservations": [],
                        "usage": {}, "errors": [], "gate_pending": [], "gate_done": [], "workspaces": []}  # fmt: skip
        self.show_hive(snap)
        now = snap["now"]
        self.fill("approvals-table", [
            (str(g["id"]), [g["id"], ago(g["ts"], now), left(g["deadline"], now), g["kind"], one_line(g["origin"], 40),
                            one_line(g["target"], 40),
                            one_line(g["preview"], 90) + (f" ({g['size']:,} chars)" if g["size"] > 300 else "")])
            for g in snap["gate_pending"]])  # fmt: skip
        self.fill("approvals-done", [
            (str(g["id"]), [g["id"], ago(g["decided_at"], now), g["status"], g["kind"], one_line(g["origin"], 30),
                            one_line(g["target"], 30), one_line(g["note"], 60)]) for g in snap["gate_done"]])  # fmt: skip
        self.fill("agents-table", [
            (a["name"], [a["name"], agent_status(a, now), one_line(a.get("description"), 50), one_line(a.get("cwd"), 40),
                         ago(a.get("last_seen"), now)]) for a in snap["agents"]])  # fmt: skip
        self.fill("threads-table", [
            (t["thread"], [t["thread"], one_line(t["senders"], 40), t["n"], ago(t["last"], now), one_line(t["first"], 80)])
            for t in snap["threads"]])  # fmt: skip
        kinds = KIND_FILTERS[self.kind]
        self.fill("messages-table", [
            (str(m["id"]), [m["id"], ago(m["ts"], now),
                            m["kind"] + (" (held)" if m["held"] else "") + (" (public)" if m["public"] else ""),
                            one_line(m["sender"], 24), one_line(m["recipient"], 24), m["thread"], one_line(m["preview"], 90)])
            for m in snap["messages"] if kinds is None or m["kind"] in kinds])  # fmt: skip
        self.fill("reservations-table", [
            (f"{r['path']}|{r['name']}", [r["path"], r["name"], max(1, int((r["expires"] - now + 59) // 60)),
                                          one_line(r["reason"], 60)]) for r in snap["reservations"]])  # fmt: skip
        rows = []
        for m in self.models or []:
            u = snap["usage"].get(m["alias"], {})
            vision = "yes" if m["vision"] else "no" if m["vision"] is not None else "?"
            avg = f"{u['secs'] / u['calls']:.1f}" if u.get("calls") else ""
            rows.append((m["alias"], [m["alias"], m["label"], m["provider"], m["model"], m["status"], vision,
                                      u.get("calls", 0), f"{u.get('in', 0):,}", f"{u.get('out', 0):,}", avg,
                                      u.get("errors", 0)]))  # fmt: skip
        if not rows and self.models_error:
            rows = [("error", [self.models_error, "", "", "", "", "", "", "", "", "", ""])]
        self.fill("models-table", rows)
        self.fill("errors-table", [
            (str(e["id"]), [e["id"], ago(e["ts"], now), one_line(e["sender"], 24), one_line(e["recipient"], 24),
                            e["thread"], one_line(e["body"], 100)]) for e in snap["errors"]])  # fmt: skip
        self.fill("workspaces-table", [
            (w["id"], [w["id"] + (f' "{one_line(w["label"], 30)}"' if w["label"] else ""), w["status"],
                       one_line(w["owner"], 24), one_line(w["recipients"] or "nobody yet", 50), w["files"],
                       w["changed"] if w["changed"] is not None else "-" if w["status"] != "open" else "...",
                       ago(w["ts"], now), one_line(f"{w['parent']}:{w['root'] or '.'}" if w["parent"] else w["root"], 60)])
            for w in snap.get("workspaces", [])])  # fmt: skip
        self.query_one("#messages-kind", Static).update(lit(f"kind: {self.kind} (press k to change)"))
        self.show_agent_detail()

    def show_hive(self, snap: dict[str, Any]) -> None:
        now = snap["now"]
        rows = [(v.setting.key, [v.setting.key, v.shown, v.detail + (f" [{v.problem}]" if v.problem else ""),
                                 v.setting.allowed(), v.setting.description]) for v in snap.get("hive") or []]  # fmt: skip
        if not rows and snap.get("hive_error"):
            rows = [("error", [snap["hive_error"], "", "", "", ""])]
        self.fill("hive-table", rows)
        # The filter looks in the whole text of notes and decisions (the database search the snapshot made for it).
        fresh = snap.get("query", "") == self.filter_text.strip()
        hits = snap.get("memory_hits") if fresh else None
        self.fill("memory-table", [
            (str(m["id"]), [m["id"], ago(m["ts"], now), "every project" if m["scope"] == "hub" else one_line(m["project"], 40),
                            f"{one_line(m['author'], 24)} ({group_of(m)})", one_line(", ".join(
                                t for t in (m["tags"] or "").split(",") if t), 30), review_state(m),
                            one_line(m["preview"], 90) + (" …" if m["size"] > 300 else "")])
            for m in snap.get("memories", []) if hits is None or m["id"] in hits], searched=hits is not None)  # fmt: skip
        hits = snap.get("decision_hits") if fresh else None
        self.fill("decisions-table", [
            (str(d["id"]), [d["id"], d["status"], ago(d["ts"], now), one_line(d["project"], 40),
                            f"{one_line(d['author'], 24)} ({group_of(d)})", one_line(d["title"], 60), d["stances"],
                            review_state(d)])
            for d in snap.get("decisions", []) if hits is None or d["id"] in hits], searched=hits is not None)  # fmt: skip
        hits = snap.get("task_hits") if fresh else None
        self.fill("tasks-table", [
            (str(t["id"]), [t["id"], t["status"], one_line(t["project"], 30), one_line(t["title"], 50),
                            f"{one_line(t['author'], 20)} ({group_of(t)})", one_line(t["assignee"], 20) or "-",
                            (one_line(t["claimer_name"], 20) + (" (gone)" if t["claim_gone"] else ""))
                            if t["claimer"] and t["status"] in ("claimed", "in_progress", "blocked") else "-",
                            ", ".join(f"#{d}" for d in (t["depends"] or "").split(",") if d) or "-", t["notes"],
                            review_state(t), ago(t["changed_at"], now)])
            for t in snap.get("tasks", []) if hits is None or t["id"] in hits], searched=hits is not None)  # fmt: skip
        self.fill("scores-table", [
            (json.dumps([s["model"], s["category"]]), [s["category"], s["model"], f"{s['score']:.2f}", s["useful"],
                                                        s["partly"], s["wrong"], s["count"], ago(s["last"], now)])
            for s in snap.get("scores", [])])  # fmt: skip

    def show_agent_detail(self) -> None:
        try:
            table = self.query_one("#agents-table", DataTable)
            detail = self.query_one("#agent-detail", Static)
        except NoMatches:  # the dashboard is closing (a highlight event can arrive after its tables are gone)
            return
        if not table.row_count or self.snapshot is None:
            detail.update(lit("Select an agent to see its reservations and messages."))
            return
        try:
            name = table.coordinate_to_cell_key(table.cursor_coordinate).row_key.value
        except Exception:
            return
        snap = self.snapshot
        text = Text(f"{clean(name)}\n", style="bold")
        held = [r for r in snap["reservations"] if r["name"] == name]
        text.append("reservations: " + (", ".join(clean(r["path"]) for r in held) or "none") + "\n")
        for m in [m for m in snap["messages"] if name in (m["sender"], m["recipient"])][:8]:
            text.append(f"  #{m['id']} {one_line(m['sender'], 20)} -> {one_line(m['recipient'], 20)} ({m['kind']}):"
                        f" {one_line(m['preview'], 70)}\n")  # fmt: skip
        detail.update(text)

    # -- events and actions ------------------------------------------------------

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        if event.data_table.id == "agents-table":
            self.show_agent_detail()

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        key = event.row_key.value
        table = event.data_table.id
        if table in ("approvals-table", "approvals-done") and key is not None:
            self.push_screen(GateScreen(self.path, int(key)))
        elif table == "threads-table" and key is not None:
            self.push_screen(ThreadScreen(self.path, key))
        elif table == "workspaces-table" and key is not None:
            self.push_screen(WorkspaceScreen(self.path, key))
        elif table == "hive-table" and key is not None:
            self.change_setting(key)
        elif table in ("memory-table", "decisions-table", "tasks-table") and key is not None:
            kind = {"memory-table": "memory", "decisions-table": "decision", "tasks-table": "task"}[table]
            self.push_screen(HiveItemScreen(self.path, kind, int(key)))
        elif table == "scores-table" and key is not None:
            self.push_screen(HiveItemScreen(self.path, "ratings", key))
        elif table in ("messages-table", "errors-table") and key is not None and self.snapshot:
            rows = self.snapshot["messages"] + self.snapshot["errors"]
            thread = next((m["thread"] for m in rows if str(m["id"]) == key), None)
            if thread:
                self.push_screen(ThreadScreen(self.path, thread))

    def action_tab(self, tab: str) -> None:
        self.query_one(TabbedContent).active = tab
        if tab == "workspaces":
            self.refresh_now()  # count the workers' changes now

    def action_filter(self) -> None:
        box = self.query_one("#filter", Input)
        box.disabled = False
        box.add_class("shown")
        box.focus()

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "filter":
            self.filter_text = event.value
            self.show(self.snapshot)
            if self.query_one(TabbedContent).active in ("memory", "decisions", "tasks"):
                self.refresh_now()  # look for it in the whole text

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "filter":
            self._focus_table()

    def on_key(self, event: Any) -> None:
        box = self.query_one("#filter", Input)
        if event.key == "escape" and box.has_focus:
            box.value = ""
            box.remove_class("shown")
            box.disabled = True
            self.filter_text = ""
            self.show(self.snapshot)
            self._focus_table()
            event.stop()

    def _focus_table(self) -> None:
        pane = self.query_one(TabbedContent).active_pane
        if pane is not None:
            tables = pane.query(DataTable)
            if tables:
                tables.first().focus()

    def action_kind(self) -> None:
        names = list(KIND_FILTERS)
        self.kind = names[(names.index(self.kind) + 1) % len(names)]
        self.show(self.snapshot)


def print_once(path: Path) -> int:
    try:
        snap = read_snapshot(path)
    except sqlite3.Error as exc:
        print(f"could not read {path}: {exc}", file=sys.stderr)
        return 1
    if snap is not None:
        try:
            snap["hive"] = hub_module().hive_values(snap["settings"])
        except Exception:  # the overview says "?" for the switches
            pass
    print(overview_text(snap, path).plain)
    return 0


def main() -> None:
    args = sys.argv[1:]
    if args == ["--once"]:
        sys.exit(print_once(db_path()))
    if args:
        print("usage: hub_tui.py [--once]", file=sys.stderr)
        sys.exit(2)
    HubDashboard().run()


if __name__ == "__main__":
    main()
