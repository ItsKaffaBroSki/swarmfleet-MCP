#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "mcp>=2,<3",
#     "httpx>=0.27,<1",
# ]
# ///
"""Agent Hub control panel: the hive's settings, shared memory, decision log, task board and model scores in your
browser.

    uv run hub_web.py              serves the panel on http://127.0.0.1:8777 and prints a sign-in link
    uv run hub_web.py --port N     another port (0: any free one)

Open the printed link in your browser: it works once and signs that browser in. The panel reads the hub's database
with a read-only connection; changes (the hive's settings, deleting notes, withdrawing or deleting decisions, changing
or deleting tasks, resetting ratings) go through agent_hub's Store, as in the terminal dashboard (mcp and httpx are
listed above only because agent_hub.py needs them; the panel itself uses the standard library). Approvals at the gate
stay in the terminal dashboard (hub_tui.py) and `agent_hub.py gate`.

The panel can change what every agent session does, so:
- it listens on 127.0.0.1 only (there is no option to listen elsewhere);
- the printed link holds a random one-time token, exchanged for an HttpOnly, SameSite=Strict session cookie;
  every other request needs that cookie;
- the Host header must be 127.0.0.1:<port> or localhost:<port> (DNS rebinding);
- changes are POST only, with the session's CSRF token and a matching Origin, and each asks for a confirmation;
  requests another site starts (Sec-Fetch-Site) are refused; no CORS headers;
- no JavaScript: Content-Security-Policy default-src 'none' (the stylesheet allowed by its hash), no framing;
- every string from the database is HTML-escaped, with control characters shown as "?".
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import hmac
import html
import os
import re
import secrets
import sqlite3
import sys
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Iterable
from urllib.parse import parse_qs, urlencode, urlsplit

DEFAULT_PORT = 8777
ONLINE_WINDOW = 50  # seconds, as in agent_hub.py
GATE_DEFAULT_TIMEOUT = 600  # as in agent_hub.py: the deadline of a gate row written before deadlines were recorded
MAX_BODY = 64 * 1024  # the largest form the panel accepts
MAX_FIELDS = 60
LIST_LIMIT = 200  # notes, decisions or rows shown on one page
BY = "control panel"  # the settings table's updated_by: "set in the control panel 3m ago"
# CSI and OSC sequences, and single-character escapes (as in hub_tui.py)
ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)|\x1b[@-Z\\-_]")
GATE_HINT = ("Approvals stay in the terminal: the dashboard (uv run hub_tui.py, tab 2) or uv run agent_hub.py gate."
             " This panel cannot approve or deny anything.")  # fmt: skip


def log(message: str) -> None:
    print("[hub-web]", clean(message), file=sys.stderr, flush=True)


def hub_module() -> Any:
    """agent_hub.py from this folder: the settings registry and the Store the panel writes with. Imported with
    AGENT_HUB_LAZY_STORE=1, so importing it neither creates the database nor opens a connection."""
    here = str(Path(__file__).resolve().parent)
    if here not in sys.path:
        sys.path.insert(0, here)
    if "agent_hub" not in sys.modules:
        os.environ["AGENT_HUB_LAZY_STORE"] = "1"
    import agent_hub  # noqa: PLC0415

    return agent_hub


# --------------------------------------------------------------------------
# Literal text in HTML
# --------------------------------------------------------------------------


def clean(text: Any) -> str:
    """Text from the database for display: no escape sequences, control or format characters (shown as ?)."""
    text = ANSI_RE.sub("", "" if text is None else str(text)).replace("\r\n", "\n")
    return "".join(c if c in "\n\t" or c.isprintable() else "?" for c in text)


def one_line(text: Any, width: int = 200) -> str:
    line = " ".join(clean(text).split())
    return line if len(line) <= width else line[: width - 1] + "…"


class Safe(str):
    """HTML built here, in which every string from outside is escaped. Adding a plain string escapes it."""

    def __add__(self, other: Any) -> Safe:
        return Safe(str.__add__(self, esc(other)))

    def __radd__(self, other: Any) -> Safe:
        return Safe(str.__add__(esc(other), self))


def esc(value: Any) -> Safe:
    return value if isinstance(value, Safe) else Safe(html.escape(clean(value), quote=True))


def fmt(template: str, *args: Any, **kwargs: Any) -> Safe:
    """`template` (written here, never from outside) with every argument HTML-escaped, unless it is Safe."""
    return Safe(template.format(*map(esc, args), **{k: esc(v) for k, v in kwargs.items()}))


def cat(parts: Iterable[Any]) -> Safe:
    return Safe("".join(esc(p) for p in parts))


def ago(ts: Any, now: float | None = None) -> str:
    if not isinstance(ts, (int, float)) or not ts:
        return "never"
    secs = max(0, int((now or time.time()) - ts))
    if secs < 90:
        return f"{secs}s ago"
    if secs < 5400:
        return f"{secs // 60}m ago"
    if secs < 172800:
        return f"{secs // 3600}h ago"
    return f"{secs // 86400}d ago"


def when(ts: Any) -> Safe:
    """"3m ago", with the date and time on hover."""
    if not isinstance(ts, (int, float)) or not ts:
        return esc("never")
    return fmt('<span title="{}">{}</span>', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts)), ago(ts))


def group_of(row: Any) -> str:
    value = row["abliterated"]
    return "?" if value is None else "abliterated" if value else "standard"


def plural(n: int, word: str) -> str:
    return f"{n} {word}" + ("" if n == 1 else "s")


# --------------------------------------------------------------------------
# Reading the database (read-only)
# --------------------------------------------------------------------------


def quote(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


class View:
    """A read-only look at the hub's database: a connection per request, opened with mode=ro and query_only, so a
    page cannot write even by mistake."""

    def __init__(self, path: Path) -> None:
        self.con = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=5.0)
        try:
            self.con.row_factory = sqlite3.Row
            self.con.execute("PRAGMA query_only=ON")
            self.tables = {r[0] for r in self.con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        except BaseException:
            self.con.close()
            raise
        self._columns: dict[str, list[str]] = {}

    def __enter__(self) -> View:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.con.close()

    def columns(self, table: str) -> list[str]:
        if table not in self._columns:
            self._columns[table] = ([r[1] for r in self.con.execute(f"PRAGMA table_info({quote(table)})")]
                                    if table in self.tables else [])  # fmt: skip
        return self._columns[table]

    def rows(self, sql: str, params: Iterable[Any] = ()) -> list[dict[str, Any]]:
        return [dict(r) for r in self.con.execute(sql, tuple(params))]

    def one(self, sql: str, params: Iterable[Any] = ()) -> Any:
        row = self.con.execute(sql, tuple(params)).fetchone()
        return None if row is None else row[0]

    def count(self, table: str, where: str = "", params: Iterable[Any] = ()) -> int:
        if table not in self.tables:
            return 0
        return int(self.one(f"SELECT COUNT(*) FROM {quote(table)}" + (f" WHERE {where}" if where else ""), params))

    def settings(self) -> dict[str, dict[str, Any]]:
        """The settings table: the values the user chose."""
        if "settings" not in self.tables:
            return {}
        return {r["key"]: r for r in self.rows("SELECT key, value, updated_at, updated_by FROM settings")}

    def gate_states(self, refs: list[str]) -> dict[str, str]:
        """The latest gate request about each hive item ("memory:12"), as words."""
        cols = self.columns("gate")
        if "ref" not in cols or not refs:
            return {}
        deadline = f"COALESCE(expires, ts + {GATE_DEFAULT_TIMEOUT})" if "expires" in cols else f"ts + {GATE_DEFAULT_TIMEOUT}"
        now, out = time.time(), {}
        for start in range(0, len(refs), 500):
            part = refs[start:start + 500]
            for r in self.rows(f"SELECT id, ref, status, note, {deadline} AS deadline FROM gate WHERE ref IN"
                               f" ({','.join('?' * len(part))}) ORDER BY id", part):  # fmt: skip
                status = "expired" if r["status"] == "pending" and r["deadline"] <= now else r["status"]
                note = f": {one_line(r['note'])}" if r["note"] else ""
                out[r["ref"]] = f"gate request #{r['id']} {status}{note}"
        return out


def like(columns: Iterable[str], text: str) -> tuple[str, list[str]]:
    """SQL: any of `columns` holds `text` (no wildcards of the user's)."""
    pattern = "%" + text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"
    columns = list(columns)
    return "(" + " OR ".join(f"{quote(c)} LIKE ? ESCAPE '\\'" for c in columns) + ")", [pattern] * len(columns)


def other_group(row: dict[str, Any], kind: str, states: dict[str, str]) -> str:
    """Whether the other gate group (abliterated or standard) may read a hive item."""
    if row.get("cleared") == 1:
        return "cleared by you for both groups"
    if row.get("cleared") == -1:
        request = states.get(f"{kind}:{row['id']}")
        return "denied by you: the other group never reads it" + (f" ({request})" if request else "")
    return states.get(f"{kind}:{row['id']}") or (
        "the writer's gate holds it for the other group" if row.get("cross_hold") else "no gate held it when written")


def stance_state(stance: dict[str, Any], states: dict[str, str]) -> str:
    """A stance's gate state, if it has one worth showing."""
    if stance.get("cleared") == 1:
        return "cleared by you for both groups"
    if stance.get("cleared") == -1:
        return "denied by you for the other group"
    return states.get(f"stance:{stance['id']}", "")


# --------------------------------------------------------------------------
# The task board and the model track record
# --------------------------------------------------------------------------
#
# Both pages are driven by this table. A page is shown when agent_hub registers its feature (register_feature) or its
# Store has one of its `methods` (an older agent_hub.py next to this file has neither, and the page says so). It lists
# its rows through its `read` function, which uses agent_hub's readers (task_rows, task_history, rating_summary,
# ratings_of) on the read-only connection, with the search and the filter; and it offers each action whose Store
# method exists, after a confirmation. An action runs in a Store of its own (a connection per change, closed after it).


@dataclass(frozen=True)
class Action:
    name: str  # in the form: what=<name>
    label: str  # the button
    method: str  # the Store method it needs: the button shows only if agent_hub's Store has it
    question: str | Callable[[Any, dict[str, str]], str]  # asked first: a text with {item} and {<field>}, or a function
    run: Callable[[Any, Any, dict[str, str]], Any]  # (store, item, fields) -> a result; falsy: nothing changed
    fields: tuple[tuple[str, str, tuple[str, ...]], ...] = ()  # (name, label, choices: a list, or () for text)
    only: tuple[str, tuple[str, ...]] | None = None  # offered only for rows whose column holds one of these
    whole: bool = False  # once for the page (item None), not per row
    danger: bool = False
    note: str = ""  # what it does, on the confirmation page
    check: Callable[[Any, dict[str, str]], str] | None = None  # (item, fields) -> why it cannot be done ('' it can)
    done: Callable[[Any, dict[str, str], Any], str] | None = None  # (item, fields, result) -> what happened


@dataclass(frozen=True)
class Extra:
    key: str  # the page: /<key>
    title: str
    noun: str  # what its table holds, for counts
    features: tuple[str, ...]  # the feature keys it is registered under (its switch: "<feature>.enabled")
    methods: tuple[str, ...]  # Store methods that show it is installed
    table: str
    item: str  # the key of a row that names it for the actions
    read: Callable[[View, Extra, dict[str, str]], tuple[list[str], list[dict[str, Any]], int]]  # (columns, rows, all)
    filter: tuple[str, tuple[str, ...]] | None  # (parameter, values): a choice above the list
    missing: str  # the page while it is not installed
    actions: tuple[Action, ...] = ()
    placeholder: str = "Search"
    shown: str = "{n} shown"  # what the rows on the page are
    intro: Callable[[dict[str, Any]], str] | None = None  # (the hive's settings by key) -> a line above the list


# As in agent_hub.py (tests/test_web.py checks they agree)
TASK_STATUSES = ("open", "claimed", "in_progress", "blocked", "done", "dropped")
TASK_USER_STATUSES = ("open", "blocked", "done", "dropped")  # what the user sets: open and blocked drop the claim
TASK_HELD = ("claimed", "in_progress", "blocked")  # with a claimer
RATING_CATEGORIES = ("ask", "ask_many", "discuss", "review", "debug", "web_search", "screenshot")
HISTORY_SHOWN = 5  # history entries per task on the page (the dashboard shows them all)
NOTES_SHOWN = 3  # raters' notes per model and kind of request on the page (the dashboard shows them all)


def history_cell(entries: list[dict[str, Any]], states: dict[str, str]) -> Safe:
    """A task's latest history entries: when, who, what changed, and the writer's note with its state at the gate."""
    shown = entries[-HISTORY_SHOWN:]
    parts: list[Safe] = []
    if len(entries) > len(shown):
        parts.append(fmt('<div class="meta">{} earlier entr{} (the dashboard shows them all)</div>',
                         len(entries) - len(shown), "y" if len(entries) - len(shown) == 1 else "ies"))  # fmt: skip
    for n in shown:
        who = one_line(n["name"]) + (f" ({group_of(n)})" if n["instance"] else "")
        state = "" if not n["note"] or n["cleared"] == 1 else "denied by you" if n["cleared"] == -1 else states.get(
            f"task_note:{n['id']}") or (
            "the writer's gate holds it for the other group" if n["cross_hold"] else "")  # fmt: skip
        parts.append(fmt('<div class="meta">{}, {}{}{}</div>', when(n["ts"]), who,
                         fmt(": {}", one_line(n["what"], 300)) if n["what"] else "",
                         fmt(" · other group: {}", state) if state else ""))  # fmt: skip
        if n["note"]:
            parts.append(fmt('<div class="text">{}</div>', n["note"]))
    return cat(parts) if parts else Safe("-")


def task_list(view: View, extra: Extra, params: dict[str, str]) -> tuple[list[str], list[dict[str, Any]], int]:
    """Every project's tasks (agent_hub.task_rows: unfinished ones first, oldest first): who added it, for whom, its
    claim (marked when its session is gone), what it waits for, its latest history and the other group's state."""
    hub = hub_module()
    assert extra.filter is not None
    status = params.get("status") if params.get("status") in extra.filter[1] else None
    tasks = hub.task_rows(view.con, status, params.get("q", "").strip(), LIST_LIMIT)
    deps = sorted({d for t in tasks for d in t["depends"]})
    known: dict[int, str] = {}
    for start in range(0, len(deps), 500):
        part = deps[start:start + 500]
        known.update({r["id"]: r["status"] for r in view.rows(f"SELECT id, status FROM tasks WHERE id IN"
                                                              f" ({','.join('?' * len(part))})", part)})  # fmt: skip
    histories = {t["id"]: hub.task_history(view.con, t["id"]) for t in tasks}
    states = view.gate_states([f"task:{t['id']}" for t in tasks] + [
        f"task_note:{n['id']}" for h in histories.values() for n in h[-HISTORY_SHOWN:] if n["note"]])  # fmt: skip
    rows = []
    for t in tasks:
        if t["claimer"] and t["status"] in TASK_HELD:
            held_by = {"abliterated": t["claimer_abliterated"]}
            group = "" if t["claimer_abliterated"] is None else f" ({group_of(held_by)})"
            claim = (f"{one_line(t['claimer_name'])}{group}, since {ago(t['claimed_at'])}, last news"
                     f" {ago(t['renewed_at'])}" + (" (its session is gone: the task is open again when an agent next"
                                                    " looks)" if t["claim_gone"] else ""))  # fmt: skip
        else:
            claim = f"last claimed by {one_line(t['claimer_name'])}" if t["claimer"] else "-"
        rows.append({
            "id": t["id"], "status": t["status"], "#": t["id"],
            "task": fmt("<b>{}</b>{}", one_line(t["title"], 300),
                        fmt('<div class="text">{}</div>', t["details"]) if t["details"] else ""),
            "project": one_line(t["project"], 300), "by": f"{one_line(t['author'])} ({group_of(t)})",
            "for": one_line(t["assignee"]) or "anyone", "claim": claim,
            "waits for": ", ".join(f"#{d} ({known.get(d, 'gone')})" for d in t["depends"]) or "-",
            "history": history_cell(histories[t["id"]], states), "other group": other_group(t, "task", states),
            "changed": when(t["changed_at"]),
        })  # fmt: skip
    return (["#", "status", "task", "project", "by", "for", "claim", "waits for", "history", "other group", "changed"],
            rows, view.count(extra.table))  # fmt: skip


def score_list(view: View, extra: Extra, params: dict[str, str]) -> tuple[list[str], list[dict[str, Any]], int]:
    """The scores per kind of request and model (agent_hub.rating_summary: useful 1, partly 0.5, wrong 0, averaged;
    the best first) with the counts, and the raters' latest notes (agents never see those)."""
    hub = hub_module()
    category = params.get("category") if params.get("category") in RATING_CATEGORIES else None
    q = params.get("q", "").strip().lower()
    summary = [s for s in hub.rating_summary(view.con, category)
               if not q or q in str(s["model"]).lower() or q in str(s["category"]).lower()][:LIST_LIMIT]  # fmt: skip
    notes: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for r in hub.ratings_of(view.con, None, category, 5000):  # newest first
        said = notes.setdefault((r["model"], r["category"]), [])
        if r["note"] and len(said) < NOTES_SHOWN:
            said.append(r)
    rows = []
    for s in summary:
        said = cat(fmt('<div class="meta">{} by {} ({}), {}</div><div class="text">{}</div>', r["verdict"],
                       one_line(r["rater"]), group_of(r), when(r["ts"]), r["note"])
                   for r in notes.get((s["model"], s["category"]), []))  # fmt: skip
        rows.append({"key": f"{s['category']}:{s['model']}", "kind": s["category"], "model": s["model"],
                     "score": f"{s['score']:.2f}", "useful": s["useful"], "partly": s["partly"], "wrong": s["wrong"],
                     "ratings": s["count"], "last rated": when(s["last"]), "raters' notes": said or "-"})  # fmt: skip
    return (["kind", "model", "score", "useful", "partly", "wrong", "ratings", "last rated", "raters' notes"], rows,
            view.count(extra.table))  # fmt: skip


def task_item(item: Any, fields: dict[str, str]) -> str:
    """Why the form names no task ('' it does)."""
    ok = isinstance(item, str) and item.isascii() and item.isdigit() and len(item) < 19
    return "" if ok else "no such task"


def assignee_check(item: Any, fields: dict[str, str]) -> str:
    name = fields.get("assignee", "").strip().lower()
    if task_item(item, fields):
        return task_item(item, fields)
    if name and not hub_module().AGENT_RE.match(name):
        return "an agent name is 2-32 characters: lowercase letters, digits, - and _ (empty: anyone)"
    return ""


def score_item(item: Any) -> tuple[str, str] | None:
    """A row of the scores ("<kind>:<model>") as (model, kind), or None."""
    category, sep, model = str(item or "").partition(":")
    return (model, category) if sep and model and category else None


def setting_shown(values: dict[str, Any], key: str) -> str:
    return values[key].shown if key in values else "?"


EXTRAS: tuple[Extra, ...] = (
    Extra(
        key="tasks", title="Task board", noun="task", features=("tasks",),
        methods=("task_set_status", "task_assign", "task_delete"), table="tasks", item="id", read=task_list,
        filter=("status", ("unfinished", *TASK_STATUSES)),
        missing="The task board is available once the task board is installed (a newer agent_hub.py next to this"
                " file). There is nothing to show yet.",
        placeholder="Search the title, details, names, project and notes", shown="{n} shown",
        intro=lambda v: (f"Every project's tasks. A claim lapses when its session ends, or after tasks.claim_minutes"
                         f" ({setting_shown(v, 'tasks.claim_minutes')}) without news from its agent; the task is open"
                         " again the next time an agent looks. Your changes go into the task's history, which every"
                         " agent of its project reads."),
        actions=(
            # The new status is "to", not "status": that is the page's filter, which the forms carry along.
            Action("status", "Set status", "task_set_status", "Set the status of task #{item} to {to}?",
                   lambda store, item, f: store.task_set_status(int(item), f["to"], f.get("note", ""), "the user"),
                   fields=(("to", "new status", TASK_USER_STATUSES),
                           ("note", "your note for the agents (optional)", ())),
                   note="open and blocked drop the claim (another agent can take an open task); done and dropped"
                        " finish it (tasks that wait for it can then be claimed).",
                   check=task_item, done=lambda item, f, r: f"Task #{item} is {f['to']} now." if r else
                   f"Task #{item} is gone, or already {f['to']}: nothing changed."),
            Action("assign", "Assign", "task_assign",
                   lambda item, f: (f"Make task #{item} meant for {f['assignee'].strip().lower()}?"
                                    if f["assignee"].strip() else f"Make task #{item} meant for anyone?"),
                   lambda store, item, f: store.task_assign(int(item), f["assignee"], "the user"),
                   fields=(("assignee", "agent name (empty: anyone)", ()),), only=("status", TASK_STATUSES[:4]),
                   note="While an agent holds that name in the hub, the task is left to it; its inbox mentions it.",
                   check=assignee_check, done=lambda item, f, r: (
                       f"Task #{item} is meant for {f['assignee'].strip().lower() or 'anyone'} now." if r else
                       f"Task #{item} is gone, or already meant for {f['assignee'].strip().lower() or 'anyone'}:"
                       " nothing changed.")),
            Action("delete", "Delete", "task_delete", "Delete task #{item} and its history for good?",
                   lambda store, item, f: store.task_delete(int(item)), danger=True,
                   note="Tasks that wait for it no longer do (their history says so); requests at the gate about it"
                        " expire.", check=task_item,
                   done=lambda item, f, r: f"Task #{item} deleted." if r else f"Task #{item} was already gone."),
        ),
    ),
    Extra(
        key="scores", title="Model scores", noun="rating", features=("track",),
        methods=("ratings_reset", "rating_summary"), table="ratings", item="key", read=score_list,
        filter=("category", RATING_CATEGORIES),
        missing="Model scores are available once the track record is installed (a newer agent_hub.py next to this"
                " file). There is nothing to show yet.",
        placeholder="Search a model or a kind of request", shown="{n} shown (a model in a kind of request each)",
        intro=lambda v: ("Score: useful 1, partly 0.5, wrong 0, averaged over the ratings agents gave the answers they"
                         " got. Agents see the scores and counts; the raters' notes are for you. Auto-route"
                         f" (track.auto_route) is {setting_shown(v, 'track.auto_route')}: it picks among models with"
                         f" at least {setting_shown(v, 'track.min_ratings')} rating(s) for a kind of request"
                         " (track.min_ratings)."),
        actions=(
            Action("reset", "Reset",
                   "ratings_reset", lambda item, f: "Reset the ratings of {} for {}?".format(*score_item(item) or
                                                                                             (item, "?")),
                   lambda store, item, f: store.ratings_reset(*score_item(item)), danger=True,  # type: ignore[misc]
                   note="Its score for that kind starts again from nothing; auto-route passes it over until it has"
                        " enough ratings again.",
                   check=lambda item, f: "" if score_item(item) else "no such model and kind of request",
                   done=lambda item, f, r: "{} rating(s) of {} for {} deleted.".format(
                       r, *score_item(item) or ("?", "?"))),
            Action("reset_all", "Reset all ratings", "ratings_reset", "Reset every model's ratings?",
                   lambda store, item, f: store.ratings_reset(), whole=True, danger=True,
                   note="Every score starts again from nothing.",
                   done=lambda item, f, r: f"{r} rating(s) deleted: every score starts again."),
        ),
    ),
)  # fmt: skip


def extra_feature(hub: Any, extra: Extra) -> str | None:
    return next((f for f in extra.features if f in hub.FEATURES), None)


def extra_installed(hub: Any, extra: Extra) -> bool:
    return extra_feature(hub, extra) is not None or any(hasattr(hub.Store, m) for m in extra.methods)


# --------------------------------------------------------------------------
# Pages
# --------------------------------------------------------------------------

CSS = """
:root { color-scheme: light dark; --fg: #1d1d1f; --bg: #f7f7f5; --card: #ffffff; --muted: #6a6a6a; --line: #e2e2de;
  --accent: #2f5bd3; --ok: #1e7a3c; --warn: #9a5b00; --bad: #b3261e; }
@media (prefers-color-scheme: dark) { :root { --fg: #e8e8e6; --bg: #141414; --card: #1c1c1c; --muted: #a2a2a2;
  --line: #2e2e2e; --accent: #8fb0ff; --ok: #7bd88f; --warn: #e0a84a; --bad: #ff8a80; } }
* { box-sizing: border-box; }
body { margin: 0; font: 15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; color: var(--fg);
  background: var(--bg); }
header { background: var(--card); border-bottom: 1px solid var(--line); }
nav { max-width: 1120px; margin: 0 auto; padding: .55rem 1rem; display: flex; gap: 1.1rem; align-items: center;
  flex-wrap: wrap; }
nav a { color: var(--fg); text-decoration: none; border-bottom: 2px solid transparent; padding: .15rem 0; }
nav a.here { border-color: var(--accent); font-weight: 600; }
nav .brand { font-weight: 700; }
main { max-width: 1120px; margin: 0 auto; padding: 1rem; }
h1 { font-size: 1.35rem; margin: .3rem 0 .8rem; }
h2 { font-size: 1.05rem; margin: 1.5rem 0 .5rem; }
h3 { font-size: 1rem; margin: 0 0 .25rem; }
p { margin: .4rem 0; }
a { color: var(--accent); }
code { font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace; font-size: .9em; }
.card { background: var(--card); border: 1px solid var(--line); border-radius: 8px; padding: .75rem 1rem;
  margin: .6rem 0; }
.meta { color: var(--muted); font-size: .87rem; }
.text { white-space: pre-wrap; overflow-wrap: anywhere; max-height: 24em; overflow: auto; margin: .35rem 0;
  font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace; font-size: .88rem; }
.label { font-weight: 600; font-size: .87rem; margin-top: .5rem; }
table { border-collapse: collapse; width: 100%; background: var(--card); }
th, td { text-align: left; vertical-align: top; padding: .45rem .55rem; border-bottom: 1px solid var(--line); }
th { color: var(--muted); font-size: .84rem; font-weight: 600; }
td.cell { white-space: pre-wrap; overflow-wrap: anywhere; max-width: 28rem; }
.flash { border: 1px solid var(--line); border-left: 4px solid var(--ok); background: var(--card); border-radius: 6px;
  padding: .55rem .9rem; margin: .5rem 0; }
.flash.error { border-left-color: var(--bad); }
.on { color: var(--ok); font-weight: 600; }
.off { color: var(--muted); }
.problem { color: var(--warn); font-size: .87rem; }
.badge { display: inline-block; border: 1px solid var(--line); border-radius: 999px; padding: 0 .5rem;
  font-size: .8rem; color: var(--muted); }
.stats { display: flex; gap: 2rem; flex-wrap: wrap; }
.stat b { display: block; font-size: 1.45rem; }
form.inline { display: inline-flex; gap: .4rem; align-items: center; margin: .15rem .4rem .15rem 0; flex-wrap: wrap; }
nav form.inline { margin: 0 0 0 auto; }
table.settings { table-layout: fixed; }
table.settings th:nth-child(2) { width: 6.5rem; }
table.settings th:nth-child(3) { width: 14rem; }
table.settings th:nth-child(4) { width: 17rem; }
.actions { margin-top: .5rem; }
button { font: inherit; font-size: .9rem; padding: .2rem .75rem; border: 1px solid var(--line); border-radius: 6px;
  background: var(--bg); color: var(--fg); cursor: pointer; }
button.primary { border-color: var(--accent); color: var(--accent); }
button.danger { border-color: var(--bad); color: var(--bad); }
input, select { font: inherit; font-size: .9rem; padding: .18rem .45rem; border: 1px solid var(--line);
  border-radius: 6px; background: var(--bg); color: var(--fg); }
input[type=search] { min-width: 24rem; max-width: 100%; }
input[type=number] { width: 8rem; }
"""
CSS_HASH = base64.b64encode(hashlib.sha256(CSS.encode()).digest()).decode()
CSP = (f"default-src 'none'; style-src 'sha256-{CSS_HASH}'; form-action 'self'; frame-ancestors 'none';"
       " base-uri 'none'")  # fmt: skip
SECURITY_HEADERS = {
    "Content-Security-Policy": CSP,
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    # Not no-referrer: with it, browsers send "Origin: null" on the panel's own forms, which the Origin check refuses.
    # same-origin still sends no referrer to any other site.
    "Referrer-Policy": "same-origin",
    "Cross-Origin-Opener-Policy": "same-origin",
    "Cross-Origin-Resource-Policy": "same-origin",
    "Cache-Control": "no-store",
}
PAGE = ('<!DOCTYPE html>\n<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,'
        ' initial-scale=1"><meta name="referrer" content="same-origin"><title>{title} · Agent Hub</title>'
        '<style>{css}</style></head><body>{nav}<main>{flash}{body}</main></body></html>\n')  # fmt: skip
NAV = (("/", "Overview"), ("/settings", "Settings"), ("/memory", "Memory"), ("/decisions", "Decisions"),
       ("/tasks", "Tasks"), ("/scores", "Model scores"))  # fmt: skip
SIGN_IN_HELP = ("Open the link that hub_web.py printed when it started, in this browser. Each link works once:"
                " restart hub_web.py for a new one.")  # fmt: skip


@dataclass
class Session:
    csrf: str
    flash: list[tuple[str, str]] = field(default_factory=list)


@dataclass
class Request:
    panel: Panel
    sid: str
    session: Session
    path: str
    params: dict[str, str]

    @property
    def hub(self) -> Any:
        return self.panel.hub

    def get(self, name: str) -> str:
        return self.params.get(name, "")

    def flash(self, kind: str, text: str) -> None:
        with self.panel.lock:
            self.session.flash.append((kind, text))


@dataclass
class Page:
    body: Safe
    status: int = 200
    headers: dict[str, str] = field(default_factory=dict)


@dataclass
class Redirect:
    location: str  # a path of this panel
    headers: dict[str, str] = field(default_factory=dict)


def csrf_field(req: Request) -> Safe:
    return fmt('<input type="hidden" name="csrf" value="{}">', req.session.csrf)


def layout(req: Request | None, title: str, body: Safe) -> Safe:
    nav, flash = Safe(""), Safe("")
    if req is not None:
        links = cat(fmt('<a href="{}"{}>{}</a>', href, Safe(' class="here" aria-current="page"') if href == req.path
                        else "", label) for href, label in NAV)  # fmt: skip
        out = fmt('<form method="post" action="/logout" class="inline">{}<button type="submit">Sign out</button>'
                  "</form>", csrf_field(req))  # fmt: skip
        nav = fmt('<header><nav><span class="brand">Agent Hub</span>{}{}</nav></header>', links, out)
        with req.panel.lock:
            flashes, req.session.flash[:] = list(req.session.flash), []
        flash = cat(fmt('<div class="flash{}" role="status">{}</div>', " error" if kind == "error" else "", text)
                    for kind, text in flashes)  # fmt: skip
    return fmt(PAGE, title=title, css=Safe(CSS), nav=nav, flash=flash, body=body)


def message_page(title: str, text: str, status: int) -> Page:
    return Page(layout(None, title, fmt("<h1>{}</h1><p>{}</p>", title, text)), status)


def no_database(req: Request) -> Safe:
    return fmt('<div class="card"><p><b>No hub database yet</b> at <code>{}</code>.</p><p class="meta">It is created'
               " when the first agent session starts (or when you change a setting here, as <code>agent_hub.py"
               " settings set</code> does).</p></div>", req.panel.db)  # fmt: skip


def user_rows(req: Request) -> dict[str, dict[str, Any]]:
    """The settings table (the values the user chose), read-only; empty with no database yet."""
    if not req.panel.db.exists():
        return {}
    with View(req.panel.db) as view:
        return view.settings()


def hive_values(req: Request, rows: dict[str, dict[str, Any]] | None = None) -> list[Any]:
    """Every registered hive setting with its effective value and where it comes from (agent_hub's registry; the
    environment and keys.env as this panel's process sees them)."""
    return req.hub.hive_values(user_rows(req) if rows is None else rows)


def switch(value: Any) -> Safe:
    return fmt('<span class="{}">{}</span>', "on" if value else "off", "on" if value else "off")


def back_to(path: str, req: Request, *names: str) -> str:
    """A page of the panel with the filters the form carried (q, status, ...)."""
    kept = {n: req.get(n) for n in names if req.get(n)}
    return path + ("?" + urlencode(kept) if kept else "")


def hidden(fields: dict[str, str]) -> Safe:
    return cat(fmt('<input type="hidden" name="{}" value="{}">', k, v) for k, v in fields.items() if v != "")


def button_form(req: Request, action: str, fields: dict[str, str], label: str, cls: str = "", extra: Safe = Safe("")
                ) -> Safe:  # fmt: skip
    return fmt('<form method="post" action="{}" class="inline">{}{}{}<button type="submit"{}>{}</button></form>',
               action, csrf_field(req), hidden(fields), extra, Safe(f' class="{cls}"') if cls else "", label)  # fmt: skip


def confirm_page(req: Request, action: str, fields: dict[str, str], question: str, details: list[str], yes: str,
                 back: str, danger: bool = False) -> Page:  # fmt: skip
    """Every change asks first: the same form again, with confirm=yes."""
    body = fmt('<h1>{}</h1><div class="card">{}<form method="post" action="{}" class="inline">{}{}<input'
               ' type="hidden" name="confirm" value="yes"><button type="submit" class="{}">{}</button></form>'
               ' <a href="{}">Cancel</a></div>', question, cat(fmt("<p>{}</p>", d) for d in details if d), action,
               csrf_field(req), hidden(fields), "danger" if danger else "primary", yes, back)  # fmt: skip
    return Page(layout(req, "Confirm", body))


def with_store(req: Request, act: Callable[[Any], Any]) -> Any:
    """A change through agent_hub's Store: a connection of this request's own, closed after it."""
    store = req.hub.Store(req.panel.db)
    try:
        return act(store)
    finally:
        store.db.close()


# -- overview ------------------------------------------------------------------


def page_overview(req: Request) -> Page:
    hub = req.hub
    values = {v.setting.key: v for v in hive_values(req)}
    parts = [fmt("<h1>Overview</h1><p class=\"meta\">Agent Hub {} · database <code>{}</code></p>",
                 getattr(hub, "__version__", "?"), req.panel.db)]  # fmt: skip
    counts: dict[str, Any] = {}
    if not req.panel.db.exists():
        parts.append(no_database(req))
    else:
        with View(req.panel.db) as view:
            now = time.time()
            online = view.count("agents", "last_seen > ?", (now - ONLINE_WINDOW,)) if "last_seen" in view.columns(
                "agents") else 0  # fmt: skip
            gate_cols = view.columns("gate")
            deadline = f"COALESCE(expires, ts + {GATE_DEFAULT_TIMEOUT})" if "expires" in gate_cols else (
                f"ts + {GATE_DEFAULT_TIMEOUT}")  # fmt: skip
            waiting = view.count("gate", f"status = 'pending' AND {deadline} > ?", (now,)) if gate_cols else 0
            counts = {"memory": f"{plural(view.count('memories'), 'note')}",
                      "decisions": f"{view.count('decisions', 'status = ?', ('active',))} active of"
                                   f" {plural(view.count('decisions'), 'decision')}"}  # fmt: skip
            for extra in EXTRAS:
                if extra_installed(hub, extra) and extra.table in view.tables:
                    counts[extra.key] = plural(view.count(extra.table), extra.noun)
            parts.append(fmt(
                '<div class="card stats"><div class="stat"><b>{}</b>of {} agent session(s) online</div>'
                '<div class="stat"><b>{}</b>request(s) waiting at the gate</div></div><p class="meta">{}</p>',
                online, view.count("agents"), waiting, GATE_HINT))  # fmt: skip
    rows = []
    for feature, name in hub.FEATURES.items():
        value = values.get(f"{feature}.enabled")
        extra = next((e for e in EXTRAS if feature in e.features), None)
        holds = counts.get(extra.key if extra else feature, "")
        rows.append(fmt("<tr><td>{}</td><td>{}</td><td class=\"meta\">{}</td><td>{}</td></tr>", name,
                        switch(value.value if value else False), value.detail if value else "", holds))  # fmt: skip
    for extra in EXTRAS:
        if not extra_installed(hub, extra):
            rows.append(fmt('<tr><td>{}</td><td class="off">not installed</td><td class="meta" colspan="2">{}</td>'
                            "</tr>", extra.title, extra.missing.split(" (")[0] + "."))  # fmt: skip
    parts.append(fmt('<h2>Hive features</h2><table><tr><th>Feature</th><th>State</th><th>From</th><th>Holds</th></tr>'
                     '{}</table><p class="meta">Every feature is off until you turn it on in <a href="/settings">'
                     "Settings</a>.</p>", cat(rows)))  # fmt: skip
    return Page(layout(req, "Overview", cat(parts)))


# -- settings ------------------------------------------------------------------


def setting_row(req: Request, v: Any, stored: bool) -> Safe:
    s = v.setting
    if s.kind == "bool":
        change = button_form(req, "/settings/set", {"key": s.key, "value": "off" if v.value else "on"},
                             "Turn off" if v.value else "Turn on", "primary")  # fmt: skip
    elif s.kind == "int":
        change = button_form(req, "/settings/set", {"key": s.key}, "Set", "primary", fmt(
            '<input type="number" name="value" min="{}" max="{}" step="1" value="{}" required aria-label="{}">',
            s.lo, s.hi, v.value, s.key))  # fmt: skip
    else:
        options = cat(fmt('<option value="{}"{}>{}</option>', c, Safe(" selected") if c == v.value else "", c)
                      for c in s.choices)  # fmt: skip
        change = button_form(req, "/settings/set", {"key": s.key}, "Set", "primary",
                             fmt('<select name="value" aria-label="{}">{}</select>', s.key, options))  # fmt: skip
    # Reset whenever the table holds a row for it, also one that cannot be used (hand-written, another version's)
    reset = button_form(req, "/settings/reset", {"key": s.key}, "Reset") if stored else Safe("")
    problem = fmt('<div class="problem">{}</div>', v.problem) if v.problem else Safe("")
    shown = switch(v.value) if s.kind == "bool" else fmt("<b>{}</b>", v.shown)
    return fmt('<tr><td><code>{}</code><div class="meta">{}</div><div class="meta">Allowed: {}; default {}.</div></td>'
               '<td>{}</td><td class="meta">{}{}</td><td>{}{}</td></tr>', s.key, s.description, s.allowed(),
               s.show(s.default), shown, v.detail, problem, change, reset)  # fmt: skip


def page_settings(req: Request) -> Page:
    hub = req.hub
    rows = user_rows(req)
    values = hive_values(req, rows)
    groups: dict[str, list[Any]] = {}
    for v in values:
        groups.setdefault(v.setting.group, []).append(v)
    parts = [fmt("<h1>Settings</h1><p>Only you change these: here, in the dashboard's Hive tab, or with <code>agent_hub"
                 ".py settings</code>. No agent can. Every agent session uses a change from its next tool call. A value"
                 " set here wins over the environment and <code>keys.env</code>, which win over the default.</p>"
                 '<p class="meta">Values from the environment or keys.env are shown as this panel\'s process sees'
                 " them; a session started with other variables has its own, but your value here wins over both."
                 "</p>")]  # fmt: skip
    if not req.panel.db.exists():
        parts.append(no_database(req))
    for group, settings in groups.items():
        parts.append(fmt('<h2>{}</h2><table class="settings"><tr><th>Setting</th><th>Value</th><th>From</th><th>Change</th></tr>{}'
                         "</table>", hub.GROUP_NAMES.get(group, group), cat(setting_row(req, v, v.setting.key in rows) for v in settings)))  # fmt: skip
    return Page(layout(req, "Settings", cat(parts)))


def act_setting_set(req: Request) -> Page | Redirect:
    hub, key, raw = req.hub, req.get("key"), req.get("value")
    setting = hub.SETTINGS.get(key)
    if setting is None:
        req.flash("error", f"There is no setting {key!r}; nothing changed.")
        return Redirect("/settings")
    try:
        value = setting.parse(raw)
    except ValueError as exc:
        req.flash("error", f"Not changed: {exc}.")
        return Redirect("/settings")
    if req.get("confirm") != "yes":
        current = next((v for v in hive_values(req) if v.setting.key == key), None)
        verb = (f"Turn {key} {setting.show(value)}" if setting.kind == "bool" else
                f"Set {key} to {setting.show(value)}")  # fmt: skip
        now = f"It is {current.shown} now ({current.detail})." if current else ""
        return confirm_page(req, "/settings/set", {"key": key, "value": raw}, verb + "?",
                            [now, setting.description, "Every agent session uses it from its next tool call."],
                            verb, "/settings")  # fmt: skip
    try:
        stored = with_store(req, lambda store: store.setting_set(key, value, BY))
        req.flash("ok", f"{key} is now {setting.show(stored)} for every session (from its next tool call).")
    except (ValueError, sqlite3.Error) as exc:
        req.flash("error", f"Not changed: {exc}")
    return Redirect("/settings")


def act_setting_reset(req: Request) -> Page | Redirect:
    hub, key = req.hub, req.get("key")
    if key not in hub.SETTINGS:
        req.flash("error", f"There is no setting {key!r}; nothing changed.")
        return Redirect("/settings")
    if not req.panel.db.exists():
        req.flash("error", f"{key} has no value of yours: there is no hub database yet.")
        return Redirect("/settings")
    if req.get("confirm") != "yes":
        return confirm_page(req, "/settings/reset", {"key": key}, f"Drop your value for {key}?",
                            ["The environment, keys.env or the default applies again, for every session from its"
                             " next tool call."], "Drop it", "/settings")  # fmt: skip
    try:
        done = with_store(req, lambda store: store.setting_reset(key))
        req.flash("ok", f"Your value for {key} is dropped." if done else f"{key} had no value of yours.")
    except sqlite3.Error as exc:
        req.flash("error", f"Not changed: {exc}")
    return Redirect("/settings")


# -- shared memory ---------------------------------------------------------------


def feature_line(req: Request, feature: str) -> Safe:
    value = next((v for v in hive_values(req) if v.setting.key == f"{feature}.enabled"), None)
    if value is None:
        return Safe("")
    return fmt('<p class="meta">{} is {} ({}). <a href="/settings">Settings</a></p>', req.hub.FEATURES.get(feature,
               feature), switch(value.value), value.detail)  # fmt: skip


def search_form(path: str, q: str, placeholder: str, extra: Safe = Safe("")) -> Safe:
    clear = fmt(' <a href="{}">Clear</a>', path) if q else Safe("")
    return fmt('<form method="get" action="{}" class="inline"><input type="search" name="q" value="{}"'
               ' placeholder="{}" aria-label="Search">{}<button type="submit">Search</button>{}</form>', path, q,
               placeholder, extra, clear)  # fmt: skip


def page_memory(req: Request) -> Page:
    q = req.get("q").strip()
    parts = [Safe("<h1>Shared memory</h1>"), feature_line(req, "memory"),
             search_form("/memory", q, "Search the text, tags, author and project")]  # fmt: skip
    if not req.panel.db.exists():
        return Page(layout(req, "Memory", cat(parts + [no_database(req)])))
    with View(req.panel.db) as view:
        if "memories" not in view.tables:
            parts.append(Safe('<p class="meta">No notes yet.</p>'))
            return Page(layout(req, "Memory", cat(parts)))
        where, args = like(("text", "tags", "author", "project"), q) if q else ("1", [])
        rows = view.rows(f"SELECT * FROM memories WHERE {where} ORDER BY id DESC LIMIT {LIST_LIMIT}", args)
        total = view.count("memories")
        found = view.count("memories", where, args) if q else total
        states = view.gate_states([f"memory:{r['id']}" for r in rows])
    shown = f"{plural(total, 'note')} in all" + (f"; {found} match" if q else "") + (
        f"; the newest {LIST_LIMIT} are shown" if found > LIST_LIMIT else "")  # fmt: skip
    parts.append(fmt('<p class="meta">{}. Notes are other agents\' words: shown here exactly as they were saved.</p>',
                     shown))  # fmt: skip
    for r in rows:
        tags = ", ".join(t for t in (r["tags"] or "").split(",") if t)
        scope = "every project (hub scope)" if r["scope"] == "hub" else f"project {r['project']}"
        parts.append(fmt(
            '<article class="card"><h3>Note #{}</h3><div class="meta">by {} ({}) · {} · for {}{}</div>'
            '<div class="meta">Other group: {}</div><div class="text">{}</div><div class="actions">{}</div></article>',
            r["id"], one_line(r["author"]), group_of(r), when(r["ts"]), one_line(scope),
            fmt(" · tags: {}", one_line(tags)) if tags else "", other_group(r, "memory", states), r["text"],
            button_form(req, "/memory/delete", {"id": str(r["id"]), "q": q}, "Delete", "danger")))  # fmt: skip
    return Page(layout(req, "Memory", cat(parts)))


def item_id(req: Request) -> int | None:
    raw = req.get("id")
    return int(raw) if raw.isascii() and raw.isdigit() and len(raw) < 19 else None


def act_memory_delete(req: Request) -> Page | Redirect:
    back = back_to("/memory", req, "q")
    note = item_id(req)
    if note is None:
        req.flash("error", "No such note; nothing changed.")
        return Redirect(back)
    if not req.panel.db.exists():
        req.flash("error", "No hub database yet: nothing to delete.")
        return Redirect(back)
    if req.get("confirm") != "yes":
        return confirm_page(req, "/memory/delete", {"id": str(note), "q": req.get("q")}, f"Delete note #{note}?",
                            ["It is deleted for good: no agent can recall it after that. A request at the gate about"
                             " it expires."], "Delete it", back, danger=True)  # fmt: skip
    try:
        done = with_store(req, lambda store: store.memory_delete(note, "the user deleted it"))
        req.flash("ok", f"Note #{note} deleted." if done else f"Note #{note} was already gone.")
    except sqlite3.Error as exc:
        req.flash("error", f"Not deleted: {exc}")
    return Redirect(back)


# -- decisions -------------------------------------------------------------------

DECISION_STATUSES = ("all", "active", "superseded", "withdrawn")


def page_decisions(req: Request) -> Page:
    q, status = req.get("q").strip(), req.get("status") or "all"
    status = status if status in DECISION_STATUSES else "all"
    choose = fmt('<select name="status" aria-label="Status">{}</select>', cat(
        fmt('<option value="{}"{}>{}</option>', s, Safe(" selected") if s == status else "", s)
        for s in DECISION_STATUSES))  # fmt: skip
    parts = [Safe("<h1>Decision log</h1>"), feature_line(req, "decisions"),
             search_form("/decisions", q, "Search the title, text, author and project", choose)]  # fmt: skip
    if not req.panel.db.exists():
        return Page(layout(req, "Decisions", cat(parts + [no_database(req)])))
    with View(req.panel.db) as view:
        if "decisions" not in view.tables:
            parts.append(Safe('<p class="meta">No decisions yet.</p>'))
            return Page(layout(req, "Decisions", cat(parts)))
        where, args = like(("title", "decision", "why", "alternatives", "author", "project"), q) if q else ("1", [])
        # Superseded is worked out from the chain of decisions in force (agent_hub.decision_links), as you see them
        # all; an agent of the other group sees a replacement (and so the decision as superseded) once you cleared it.
        hub = hub_module()
        links = hub.decision_links(view.con)
        if status == "withdrawn":
            where = f"{where} AND status = 'withdrawn'"
        elif status != "all":
            line, line_args = hub.superseded_sql(None)
            where, args = (f"{where} AND status != 'withdrawn' AND id {'NOT ' if status == 'active' else ''}IN ({line})",
                           [*args, *line_args])  # fmt: skip
        rows = view.rows(f"SELECT * FROM decisions WHERE {where} ORDER BY id DESC LIMIT {LIST_LIMIT}", args)
        newer = sorted({n for r in rows for n in links.get(r["id"], [])})
        replacing = {r["id"]: r for r in view.rows(f"SELECT id, cleared, cross_hold FROM decisions WHERE id IN"
                                                   f" ({','.join('?' * len(newer))})", newer)} if newer else {}  # fmt: skip
        total, found = view.count("decisions"), view.count("decisions", where, args)
        ids = [r["id"] for r in rows]
        stances: dict[int, list[dict[str, Any]]] = {}
        if ids and "decision_stances" in view.tables:
            for s in view.rows(f"SELECT * FROM decision_stances WHERE decision IN ({','.join('?' * len(ids))})"
                               " ORDER BY ts", ids):  # fmt: skip
                stances.setdefault(s["decision"], []).append(s)
        states = view.gate_states([f"decision:{i}" for i in ids] + [f"stance:{s['id']}" for ss in stances.values()
                                                                     for s in ss])  # fmt: skip
    shown = f"{plural(total, 'decision')} in all" + (f"; {found} shown" if found != total else "") + (
        f"; the newest {LIST_LIMIT} are listed" if found > LIST_LIMIT else "")  # fmt: skip
    parts.append(fmt('<p class="meta">{}. Decisions are other agents\' words: shown here exactly as they were'
                     " recorded.</p>", shown))  # fmt: skip
    keep = {"q": q, "status": status if status != "all" else ""}
    for r in rows:
        state = "withdrawn" if r["status"] == "withdrawn" else "superseded" if r["id"] in links else "active"
        chain = []
        if r["supersedes"]:
            chain.append(f"supersedes decision #{r['supersedes']}")
        if state == "superseded":
            chain.append("superseded by decision " + ", ".join(replacement(replacing.get(n), n) for n in links[r["id"]]))
        if r["status"] == "withdrawn":
            chain.append(f"withdrawn by {one_line(r['changed_by'])} {ago(r['changed_at'])}"
                         + (f": {one_line(r['status_note'], 400)}" if r["status_note"] else ""))  # fmt: skip
        body = cat(fmt('<div class="label">{}</div><div class="text">{}</div>', head, r[column])
                   for head, column in (("Decision", "decision"), ("Why", "why"),
                                        ("Alternatives considered", "alternatives")) if r[column])  # fmt: skip
        st = cat(fmt('<li><b>{}</b> ({}) {}, {}{}<div class="text">{}</div></li>', one_line(s["name"]), group_of(s),
                     one_line(s["stance"]), when(s["ts"]), fmt(" · {}", stance_state(s, states)) if stance_state(
                     s, states) else "", s["note"]) for s in stances.get(r["id"], []))  # fmt: skip
        actions = []
        if state == "active":
            actions.append(button_form(req, "/decisions/withdraw", {"id": str(r["id"]), **keep}, "Withdraw", "", Safe(
                '<input type="text" name="note" maxlength="2000" placeholder="Reason (optional)"'
                ' aria-label="Reason">')))  # fmt: skip
        actions.append(button_form(req, "/decisions/delete", {"id": str(r["id"]), **keep}, "Delete", "danger"))
        parts.append(fmt(
            '<article class="card"><h3>#{} {} <span class="badge">{}</span></h3><div class="meta">by {} ({}) · {} ·'
            ' project {}</div><div class="meta">Other group: {}</div>{}{}<div class="label">Stances ({})</div><ul>{}'
            '</ul><div class="actions">{}</div></article>', r["id"], one_line(r["title"], 300), state,
            one_line(r["author"]), group_of(r), when(r["ts"]), one_line(r["project"]),
            other_group(r, "decision", states), cat(fmt('<div class="meta">{}</div>', x) for x in chain), body,
            len(stances.get(r["id"], [])), st, cat(actions)))  # fmt: skip
    return Page(layout(req, "Decisions", cat(parts)))


def replacement(row: dict[str, Any] | None, decision_id: int) -> str:
    """A decision that replaces another, and for whom: one the writer's gate holds for the other group supersedes for
    that group only once you approve it (or never, if you denied it)."""
    if row is not None and row["cleared"] == -1:
        return f"#{decision_id} (for its writer's group only: you denied it to the other group)"
    if row is not None and row["cross_hold"] and row["cleared"] != 1:
        return f"#{decision_id} (for its writer's group; for the other group once you approve it)"
    return f"#{decision_id}"


def chain_effect(req: Request, decision: int, withdraw: bool) -> str:
    """What withdrawing or deleting a decision does to the chain of decisions it is in ('' nothing)."""
    try:
        with View(req.panel.db) as view:
            row = next(iter(view.rows("SELECT id, status, supersedes FROM decisions WHERE id=?", [decision])), None)
            newer = hub_module().decision_links(view.con).get(decision, []) if row is not None else []
    except sqlite3.Error:
        return ""
    if row is None or not row["supersedes"]:
        return ""
    if newer and not withdraw:
        return (f" Decision #{', #'.join(map(str, newer))}, which replaces it, then replaces #{row['supersedes']}"
                " directly: that one stays superseded.")  # fmt: skip
    if newer or row["status"] == "withdrawn":
        return ""
    return f" Decision #{row['supersedes']}, which it replaced, is in force again (unless another decision replaces it)."


def act_decision(req: Request, what: str) -> Page | Redirect:
    back = back_to("/decisions", req, "q", "status")
    decision = item_id(req)
    if decision is None:
        req.flash("error", "No such decision; nothing changed.")
        return Redirect(back)
    if not req.panel.db.exists():
        req.flash("error", "No hub database yet: nothing to change.")
        return Redirect(back)
    note = req.get("note").strip()[:2000]
    if req.get("confirm") != "yes":
        fields = {"id": str(decision), "q": req.get("q"), "status": req.get("status"), "note": note}
        if what == "withdraw":
            said = "Agents see it as withdrawn by you" + (f", with your reason: {note}" if note else
                                                          " (no reason given)") + "."  # fmt: skip
            return confirm_page(req, "/decisions/withdraw", fields, f"Withdraw decision #{decision}?",
                                [said + chain_effect(req, decision, True)], "Withdraw it", back)
        return confirm_page(req, "/decisions/delete", fields, f"Delete decision #{decision} and its stances?",
                            ["It is deleted for good; requests at the gate about it expire."
                             + chain_effect(req, decision, False)], "Delete it", back, danger=True)  # fmt: skip
    try:
        if what == "withdraw":
            done = with_store(req, lambda store: store.decision_withdraw(decision, "the user", note))
            req.flash("ok", f"Decision #{decision} withdrawn." if done else
                      f"Decision #{decision} is not active (or gone): nothing changed.")  # fmt: skip
        else:
            done = with_store(req, lambda store: store.decision_delete(decision))
            req.flash("ok", f"Decision #{decision} deleted." if done else f"Decision #{decision} was already gone.")
    except sqlite3.Error as exc:
        req.flash("error", f"Not changed: {exc}")
    return Redirect(back)


# -- the task board and the model scores (EXTRAS) -----------------------------------


def cell(value: Any) -> Safe:
    return fmt('<td class="cell">{}</td>', "" if value is None else value)


def action_form(req: Request, extra: Extra, action: Action, item: Any, keep: dict[str, str]) -> Safe:
    inputs = cat(fmt('<select name="{}" aria-label="{}">{}</select>', name, label, cat(
        fmt('<option value="{}">{}</option>', c, c) for c in choices)) if choices else fmt(
        '<input type="text" name="{}" maxlength="2000" placeholder="{}" aria-label="{}">', name, label, label)
        for name, label, choices in action.fields)  # fmt: skip
    fields = {"what": action.name, **({} if item is None else {"item": str(item)}), **keep}
    return button_form(req, f"/{extra.key}/act", fields, action.label, "danger" if action.danger else "", inputs)


def page_extra(req: Request, extra: Extra) -> Page:
    hub = req.hub
    parts = [fmt("<h1>{}</h1>", extra.title)]
    if not extra_installed(hub, extra):
        parts.append(fmt('<div class="card"><p>{}</p></div>', extra.missing))
        return Page(layout(req, extra.title, cat(parts)))
    feature = extra_feature(hub, extra)
    if feature:
        parts.append(feature_line(req, feature))
    if extra.intro is not None:
        parts.append(fmt('<p class="meta">{}</p>', extra.intro({v.setting.key: v for v in hive_values(req)})))
    q = req.get("q").strip()
    choose, keep = Safe(""), {"q": q}
    if extra.filter:
        name, values = extra.filter
        picked = req.get(name) if req.get(name) in values else ""
        keep[name] = picked
        choose = fmt('<select name="{}" aria-label="{}"><option value="">every {}</option>{}</select>', name, name,
                     name, cat(fmt('<option value="{}"{}>{}</option>', v, Safe(" selected") if v == picked else "", v)
                               for v in values))  # fmt: skip
    parts.append(search_form(f"/{extra.key}", q, extra.placeholder, choose))
    if not req.panel.db.exists():
        return Page(layout(req, extra.title, cat(parts + [no_database(req)])))
    with View(req.panel.db) as view:
        if extra.table not in view.tables:
            parts.append(fmt('<p class="meta">Nothing yet: no {} has been recorded.</p>', extra.noun))
            return Page(layout(req, extra.title, cat(parts)))
        columns, rows, total = extra.read(view, extra, req.params)
    actions = [a for a in extra.actions if hasattr(hub.Store, a.method)]
    parts.append(fmt('<p class="meta">{} in all; {}. What agents wrote is shown exactly as written.</p>',
                     plural(total, extra.noun), extra.shown.format(n=len(rows))))  # fmt: skip
    whole = [a for a in actions if a.whole]
    if whole and rows:
        parts.append(fmt('<div class="actions">{}</div>', cat(action_form(req, extra, a, None, keep) for a in whole)))
    per_row = [a for a in actions if not a.whole]
    head = cat(fmt("<th>{}</th>", c) for c in columns) + (Safe("<th></th>") if per_row else Safe(""))
    body = []
    for r in rows:
        offered = [a for a in per_row if extra.item in r and (a.only is None or r.get(a.only[0]) in a.only[1])]
        tail = fmt("<td>{}</td>", cat(action_form(req, extra, a, r[extra.item], keep) for a in offered)) if per_row else ""
        body.append(fmt("<tr>{}{}</tr>", cat(cell(r.get(c)) for c in columns), tail))
    parts.append(fmt("<table><tr>{}</tr>{}</table>", head, cat(body)))
    return Page(layout(req, extra.title, cat(parts)))


def act_extra(req: Request, extra: Extra) -> Page | Redirect:
    hub = req.hub
    keep = [n for n in ("q", extra.filter[0] if extra.filter else "") if n]
    back = back_to(f"/{extra.key}", req, *keep)
    action = next((a for a in extra.actions if a.name == req.get("what")), None)
    if not extra_installed(hub, extra) or action is None or not hasattr(hub.Store, action.method):
        req.flash("error", f"{extra.title}: that cannot be done here (not installed); nothing changed.")
        return Redirect(back)
    item = None if action.whole else req.get("item")
    if not action.whole and not item:
        req.flash("error", f"{extra.title}: which {extra.noun}? Nothing changed.")
        return Redirect(back)
    fields = {name: req.get(name) for name, _, _ in action.fields}
    for name, label, choices in action.fields:
        if choices and fields[name] not in choices:
            req.flash("error", f"{extra.title}: {label} must be one of {', '.join(choices)}; nothing changed.")
            return Redirect(back)
    why = action.check(item, fields) if action.check is not None else ""
    if why:
        req.flash("error", f"{extra.title}: {why}; nothing changed.")
        return Redirect(back)
    if not req.panel.db.exists():
        req.flash("error", "No hub database yet: nothing to change.")
        return Redirect(back)
    try:
        question = (action.question(item, fields) if callable(action.question)
                    else action.question.format(item=item, **fields))  # fmt: skip
    except (KeyError, IndexError, ValueError):
        question = f"{action.label}?"
    if req.get("confirm") != "yes":
        sent = {"what": action.name, "item": item or "", **fields, **{n: req.get(n) for n in keep}}
        said = [f"{label}: {fields[name]}" for name, label, choices in action.fields if not choices and fields[name]]
        return confirm_page(req, f"/{extra.key}/act", sent, question, [action.note, *said], action.label, back,
                            action.danger)  # fmt: skip
    try:
        done = with_store(req, lambda store: action.run(store, item, fields))
        if action.done is not None:
            req.flash("ok", action.done(item, fields, done))
        else:
            req.flash("ok", f"{extra.title}: {action.label.lower()}, done." if done or done is None else
                      f"{extra.title}: nothing changed (already done, or gone).")  # fmt: skip
    except Exception as exc:  # whatever the Store refuses
        req.flash("error", f"{extra.title}: not done: {type(exc).__name__}: {exc}")
    return Redirect(back)


# -- signing out -----------------------------------------------------------------


def act_logout(req: Request) -> Page:
    req.panel.sign_out(req.sid)
    page = message_page("Signed out", "This browser is signed out of the panel. Restart hub_web.py for a new link.", 200)
    page.headers["Set-Cookie"] = f"{req.panel.cookie}=; Max-Age=0; HttpOnly; SameSite=Strict; Path=/"
    return page


GET_ROUTES: dict[str, Callable[[Request], Page | Redirect]] = {
    "/": page_overview, "/settings": page_settings, "/memory": page_memory, "/decisions": page_decisions,
}  # fmt: skip
POST_ROUTES: dict[str, Callable[[Request], Page | Redirect]] = {
    "/settings/set": act_setting_set, "/settings/reset": act_setting_reset, "/memory/delete": act_memory_delete,
    "/decisions/withdraw": lambda req: act_decision(req, "withdraw"),
    "/decisions/delete": lambda req: act_decision(req, "delete"), "/logout": act_logout,
}  # fmt: skip
for _extra in EXTRAS:
    GET_ROUTES[f"/{_extra.key}"] = lambda req, extra=_extra: page_extra(req, extra)
    POST_ROUTES[f"/{_extra.key}/act"] = lambda req, extra=_extra: act_extra(req, extra)


# --------------------------------------------------------------------------
# The server
# --------------------------------------------------------------------------


def cookie_value(header: str, name: str) -> str:
    for part in header.split(";"):
        key, sep, value = part.strip().partition("=")
        if sep and key.strip() == name:
            return value.strip()
    return ""


def first_values(text: str) -> dict[str, str]:
    """A query string or a form: the first value of each field. ValueError for too many fields."""
    return {k: v[0] for k, v in parse_qs(text, keep_blank_values=True, max_num_fields=MAX_FIELDS).items()}


def same(a: str, b: str) -> bool:
    return hmac.compare_digest(a.encode("utf-8", "replace"), b.encode("utf-8", "replace"))


class PanelServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = os.name != "nt"  # on Windows it would let another program take over the port
    panel: Panel


class Panel:
    """The control panel's server on 127.0.0.1:<port>, with its one-time sign-in token and the signed-in browser
    sessions (kept in memory: they end with the server)."""

    def __init__(self, db: Path | str | None = None, port: int = DEFAULT_PORT) -> None:
        self.hub = hub_module()
        self.db = Path(db) if db is not None else Path(self.hub.DB_PATH)
        self.token: str | None = secrets.token_urlsafe(32)
        self.sessions: dict[str, Session] = {}
        self.lock = threading.Lock()
        self.httpd = PanelServer(("127.0.0.1", port), Handler)
        self.httpd.panel = self
        self.port = int(self.httpd.server_address[1])
        self.cookie = f"agent_hub_panel_{self.port}"  # cookies do not tell ports apart: one name per panel
        self.hosts = (f"127.0.0.1:{self.port}", f"localhost:{self.port}")
        self.login_url = f"http://127.0.0.1:{self.port}/login?token={self.token}"

    def sign_in(self, token: str) -> str | None:
        """A new browser session for the one-time token (which is then used up), or None."""
        with self.lock:
            if self.token is None or not same(token, self.token):
                return None
            self.token = None
            sid = secrets.token_urlsafe(32)
            self.sessions[sid] = Session(secrets.token_urlsafe(32))
            return sid

    def session(self, cookies: str) -> tuple[str, Session] | None:
        value = cookie_value(cookies, self.cookie)
        if not value:
            return None
        with self.lock:
            for sid, session in self.sessions.items():
                if same(value, sid):
                    return sid, session
        return None

    def sign_out(self, sid: str) -> None:
        with self.lock:
            self.sessions.pop(sid, None)

    def serve_in_thread(self) -> threading.Thread:
        thread = threading.Thread(target=self.httpd.serve_forever, name="hub-web", daemon=True)
        thread.start()
        return thread

    def stop(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


class Handler(BaseHTTPRequestHandler):
    server: PanelServer
    server_version = "AgentHubPanel"
    sys_version = ""
    timeout = 30  # a client that stops sending is dropped

    def version_string(self) -> str:
        return self.server_version  # no Python version

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        pass  # requests are not logged (the sign-in link holds the token); refusals are (refuse)

    def end_headers(self) -> None:
        for name, value in SECURITY_HEADERS.items():
            self.send_header(name, value)
        super().end_headers()

    def do_GET(self) -> None:  # noqa: N802
        self.answer("GET")

    def do_POST(self) -> None:  # noqa: N802
        self.answer("POST")

    def not_allowed(self) -> None:
        self.send_page(message_page("Not allowed", "This panel answers GET and POST requests only.", 405),
                       {"Allow": "GET, POST"})  # fmt: skip

    do_HEAD = do_PUT = do_DELETE = do_PATCH = do_OPTIONS = do_TRACE = do_CONNECT = not_allowed  # noqa: N815

    def send_page(self, page: Page, headers: dict[str, str] | None = None) -> None:
        data = page.body.encode("utf-8")
        self.send_response(page.status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        for name, value in {**page.headers, **(headers or {})}.items():
            self.send_header(name, value)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(data)

    def redirect(self, location: str, headers: dict[str, str] | None = None) -> None:
        self.send_response(303)
        self.send_header("Location", location)
        self.send_header("Content-Length", "0")
        for name, value in (headers or {}).items():
            self.send_header(name, value)
        self.end_headers()

    def refuse(self, status: int, title: str, text: str, why: str) -> None:
        log(f"refused {self.command} {urlsplit(self.path).path[:200]} ({status}): {why}")
        self.send_page(message_page(title, text, status), {"Allow": "POST"} if status == 405 else None)

    def answer(self, method: str) -> None:
        panel = self.server.panel
        hosts = self.headers.get_all("Host") or []
        if len(hosts) != 1 or hosts[0] not in panel.hosts:  # DNS rebinding: a page of another name at this address
            return self.refuse(400, "Wrong address", f"This panel answers only at http://127.0.0.1:{panel.port}/.",
                               f"Host header {hosts!r}")  # fmt: skip
        host = hosts[0]
        site = (self.headers.get("Sec-Fetch-Site") or "").strip().lower()
        if site and site not in (("same-origin",) if method == "POST" else ("same-origin", "none")):
            return self.refuse(403, "Refused", "Requests that another site or page starts are refused.",
                               f"Sec-Fetch-Site {site!r}")  # fmt: skip
        url = urlsplit(self.path)
        if url.scheme or url.netloc or not url.path.startswith("/"):
            return self.refuse(400, "Bad request", "Not a path of this panel.", "absolute request target")
        try:
            query = first_values(url.query)
        except ValueError as exc:  # too many fields
            return self.refuse(400, "Bad request", "The request could not be read.", str(exc))
        if method == "GET" and url.path == "/login":
            return self.login(query)
        found = panel.session(";".join(self.headers.get_all("Cookie") or []))
        if found is None:
            return self.refuse(403, "Not signed in", SIGN_IN_HELP, "no session cookie")
        sid, session = found
        if method == "GET":
            if url.path in POST_ROUTES:
                return self.refuse(405, "Not allowed", "Changes are made with the panel's buttons only (POST).",
                                   "GET of a change")  # fmt: skip
            route, params = GET_ROUTES.get(url.path), query
        else:
            origins = self.headers.get_all("Origin") or []
            if origins != [f"http://{host}"]:
                return self.refuse(403, "Refused", "A change must come from this panel's own page.",
                                   f"Origin {origins!r}")  # fmt: skip
            kind = (self.headers.get("Content-Type") or "").split(";")[0].strip().lower()
            if kind != "application/x-www-form-urlencoded":
                return self.refuse(415, "Refused", "Changes are sent as the panel's forms.", f"Content-Type {kind!r}")
            try:
                length = int(self.headers.get("Content-Length") or "-1")
            except ValueError:
                length = -1
            if not 0 <= length <= MAX_BODY:
                return self.refuse(413, "Refused", "That form is too large (or has no length).", f"length {length}")
            try:
                params = first_values(self.rfile.read(length).decode("utf-8", "replace"))
            except ValueError as exc:  # too many fields
                return self.refuse(400, "Bad request", "The form could not be read.", str(exc))
            if not same(params.get("csrf", ""), session.csrf):
                return self.refuse(403, "Refused", "This form is out of date or not from this panel: reload the page"
                                   " and try again.", "CSRF token")  # fmt: skip
            route = POST_ROUTES.get(url.path)
        if route is None:
            return self.send_page(message_page("Not found", "There is no such page here.", 404))
        try:
            result = route(Request(panel, sid, session, url.path, params))
        except sqlite3.Error as exc:
            log(f"database error on {method} {url.path}: {exc}")
            return self.send_page(message_page("Database unavailable", f"Could not read the hub's database: {exc}."
                                               " Reload in a moment.", 503))  # fmt: skip
        except Exception as exc:
            log(f"error on {method} {url.path}: {type(exc).__name__}: {exc}")
            return self.send_page(message_page("Error", f"{type(exc).__name__}: {exc}", 500))
        if isinstance(result, Redirect):
            return self.redirect(result.location, result.headers)
        return self.send_page(result)

    def login(self, params: dict[str, str]) -> None:
        panel = self.server.panel
        sid = panel.sign_in(params.get("token", ""))
        if sid is None:
            return self.refuse(403, "Link not valid", "This sign-in link is not valid, or it was used already. "
                               + SIGN_IN_HELP, "sign-in token")  # fmt: skip
        log("a browser signed in; the link is used up")
        self.redirect("/", {"Set-Cookie": f"{panel.cookie}={sid}; HttpOnly; SameSite=Strict; Path=/"})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="hub_web.py", description="Agent Hub control panel, in your browser on this"
                                     " computer (127.0.0.1 only).")  # fmt: skip
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"the port (default {DEFAULT_PORT}; 0: any free"
                        " one)")  # fmt: skip
    args = parser.parse_args(argv)
    if not 0 <= args.port <= 65535:
        parser.error("--port must be from 0 to 65535")
    try:
        panel = Panel(port=args.port)
    except ImportError as exc:
        print(f"hub_web.py needs agent_hub.py next to it, and its dependencies (uv run hub_web.py installs them): {exc}",
              file=sys.stderr)  # fmt: skip
        return 1
    except OSError as exc:
        print(f"hub_web.py: cannot listen on 127.0.0.1:{args.port}: {exc.strerror or exc}. Choose another port with"
              " --port N (0: any free one).", file=sys.stderr)  # fmt: skip
        return 1
    print(f"Agent Hub control panel {getattr(panel.hub, '__version__', '')}, database {clean(panel.db)}", flush=True)
    print("Open this link in your browser. It works once, and signs that browser in:", flush=True)
    print(f"\n    {panel.login_url}\n", flush=True)
    print(f"It answers on 127.0.0.1:{panel.port} only. Ctrl+C stops it.", flush=True)
    try:
        panel.httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.", flush=True)
    finally:
        panel.httpd.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
