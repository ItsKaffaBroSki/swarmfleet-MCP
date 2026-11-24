"""Tests for the dashboard (hub_tui.py), driven headlessly with Textual's test pilot.

    python tests/test_tui.py            (needs: pip install "textual>=8,<9" "mcp>=2,<3" httpx)
"""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any

PROJECT = Path(__file__).resolve().parent.parent
TMP = Path(tempfile.mkdtemp())
os.environ["AGENT_HUB_HOME"] = str(TMP)  # before agent_hub or hub_tui read it
os.environ.pop("AGENT_HUB_DB", None)
os.environ["AGENT_HUB_TUI_REFRESH"] = "0.5"
sys.path.insert(0, str(PROJECT))

import agent_hub  # noqa: E402  (here, in the main thread: its connection belongs to this thread)
import hub_tui  # noqa: E402
from textual.widgets import DataTable, Static  # noqa: E402

PASSED: list[str] = []
FAILED: list[str] = []
DB = TMP / "hub.db"


def check(name: str, condition: Any, detail: Any = "") -> None:
    if condition:
        PASSED.append(name)
    else:
        FAILED.append(name)
        print(f"  FAIL {name}: {str(detail)[:600]}")


def text_of(app: Any, selector: str) -> str:
    content = app.query_one(selector, Static).content
    return getattr(content, "plain", str(content))


def table_text(app: Any, table_id: str) -> str:
    table = app.query_one(f"#{table_id}", DataTable)
    return "\n".join(" | ".join(getattr(c, "plain", str(c)) for c in table.get_row_at(i)) for i in range(table.row_count))


def fill_database() -> None:
    now = time.time()
    db = agent_hub.store.db
    agents = [("alpha", "inst-a", "frontend work", "/w/a", now, now, 0, 0), ("ghost", "inst-g", "gone", "/w/g", now - 999,
              now - 999, 0, 0)]  # fmt: skip
    db.executemany("INSERT INTO agents (name, instance, description, cwd, joined_at, last_seen, last_inbox,"
                   " listening_until) VALUES (?,?,?,?,?,?,?,?)", agents)  # fmt: skip
    rows = [
        (now - 50, "t-one", "alpha", "opus-5.5", "ask", "model", "QUESTION-ONE [bold red]x[/] [link=https://evil]y"
         "\x1b[31mRED\nsecond line", '<file path="a">' + "f" * 1000 + "</file>", "{}", None),
        (now - 40, "t-one", "opus-5.5", "alpha", "reply", None, "ANSWER-ONE", "", json.dumps(
            {"model": "claude-opus-5-5", "in": 1200, "out": 300, "secs": 2.5, "reasoning": "SECRET-REASONING"}), 1),
        (now - 30, "t-two", "alpha", "glm-5.3", "ask", "model", "QUESTION-TWO", "", "{}", None),
        (now - 20, "t-two", "glm-5.3", "alpha", "error", None, "HTTP 401: bad key", "", "{}", 3),
        (now - 10, "t-disc", "alpha", "discussion", "ask", "discussion", "TOPIC", "", "{}", None),
        (now - 9, "t-disc", "opus-5.5", "discussion", "turn", None, "TURN-ONE", "", json.dumps({"round": 1, "in": 5,
         "out": 6}), None),  # fmt: skip
    ]
    db.executemany("INSERT INTO messages (ts, thread, sender, recipient, kind, rtype, body, attachments, meta,"
                   " reply_to) VALUES (?,?,?,?,?,?,?,?,?,?)", rows)  # fmt: skip
    db.execute("INSERT INTO messages (ts, thread, sender, recipient, kind, rtype, body, held) VALUES"
               " (?, 't-held', 'alpha', 'beta', 'ask', 'agent', 'HELD-BODY', 1)", (now,))  # fmt: skip
    reservations = [("inst-a", "/w/a/src", "refactor", now, now + 600), ("inst-a", "/w/a/old", "expired", now - 900,
                    now - 60), ("inst-g", "/w/g/lib", "offline holder", now, now + 600)]  # fmt: skip
    db.executemany("INSERT INTO reservations (instance, path, reason, created, expires) VALUES (?,?,?,?,?)",
                   reservations)  # fmt: skip
    agent_hub.store.gate_open("web_search", "glm-abl (abliterated)", "sonnet-5.5 web search (standard)",
                              "SEARCH-QUERY", True)  # fmt: skip
    agent_hub.store.gate_open("model_call", "alpha (standard)", "glm-abl (abliterated)",
                              "=== user ===\nFULL-PROMPT \x1b[2J[bold]not bold[/]", False)  # fmt: skip
    agent_hub.store.gate_open("screenshot", "glm-abl (abliterated)", "sonnet-5.5 (standard)", "IS IT OK?", True)
    agent_hub.store.gate_open("model_call", "x (standard)", "y (abliterated)", "RACE", False)


def gate_rows() -> dict[int, tuple]:
    with sqlite3.connect(DB) as con:
        return {r[0]: r[1:] for r in con.execute("SELECT id, status, note, edited, content FROM gate")}


async def wait_for(predicate: Any, pilot: Any, seconds: float = 6.0) -> bool:
    for _ in range(int(seconds / 0.1)):
        if predicate():
            return True
        await pilot.pause(0.1)
    return bool(predicate())


async def test_no_database() -> None:
    later = TMP / "later" / "hub.db"
    app = hub_tui.HubDashboard(later)
    async with app.run_test(size=(160, 50)) as pilot:
        ok = await wait_for(lambda: "No hub database yet" in text_of(app, "#overview-text"), pilot)
        check("no database yet: says so", ok, text_of(app, "#overview-text"))
        await pilot.pause(1.2)
        check("starting the dashboard does not create the database", not later.exists(), "")
        agent_hub.Store(later).db.close()
        ok = await wait_for(lambda: "agents online" in text_of(app, "#overview-text"), pilot)
        check("picks the database up once it exists", ok, text_of(app, "#overview-text"))
    fill_database()


async def test_dashboard() -> None:
    app = hub_tui.HubDashboard(DB)
    async with app.run_test(size=(180, 50)) as pilot:
        await wait_for(lambda: "agents online" in text_of(app, "#overview-text"), pilot)
        overview = text_of(app, "#overview-text")
        check("overview counts", "1 of 2" in overview and "4 request(s) waiting" in overview, overview)
        check("overview tokens", "opus-5.5" in overview and "1,205" in overview, overview)
        check("pending count in the subtitle", "4 waiting at the gate" in app.sub_title, app.sub_title)

        await pilot.press("3")
        agents = table_text(app, "agents-table")
        check("agents tab", "alpha | online" in agents and "ghost | offline" in agents, agents)
        check("agent detail panel", "/w/a/src" in text_of(app, "#agent-detail"), text_of(app, "#agent-detail"))

        await pilot.press("4")
        threads = table_text(app, "threads-table")
        check("threads tab", "t-one" in threads and "t-disc" in threads, threads)
        table = app.query_one("#threads-table", DataTable)
        table.focus()
        table.move_cursor(row=table.get_row_index("t-one"))
        await pilot.press("enter")
        await pilot.pause(0.6)
        screen = app.screen
        transcript = text_of(screen, "#transcript")
        check("thread transcript opens", "QUESTION-ONE" in transcript and "ANSWER-ONE" in transcript, transcript)
        check("markup is shown literally", "[bold red]x[/]" in transcript and "[link=https://evil]" in transcript,
              transcript)  # fmt: skip
        check("escape sequences are removed", "\x1b" not in transcript and "RED" in transcript, repr(transcript))
        check("attachments are not shown", "fff" not in transcript and "1,0" in transcript, transcript)
        check("reasoning is never shown", "SECRET-REASONING" not in transcript, "")
        await pilot.press("escape")
        await pilot.pause(0.2)
        check("escape closes the transcript", app.screen is not screen, "")

        await pilot.press("5")
        messages = table_text(app, "messages-table")
        check("messages tab", "QUESTION-TWO" in messages and "TURN-ONE" in messages, messages)
        check("held messages are marked", "ask (held)" in messages, messages)
        check("message previews are literal and one line", "[bold red]x[/]" in messages and "\x1b" not in messages,
              messages)  # fmt: skip
        await pilot.press("k", "k", "k")
        await pilot.pause(0.2)
        errors_only = table_text(app, "messages-table")
        check("kind filter (errors)", "HTTP 401" in errors_only and "QUESTION-TWO" not in errors_only, errors_only)
        await pilot.press("k", "k", "k")  # discussions, public posts, back to all
        await pilot.press("slash")
        await pilot.press(*"QUESTION-TWO")
        await pilot.pause(0.3)
        filtered = table_text(app, "messages-table")
        check("the filter narrows the table", "QUESTION-TWO" in filtered and "TURN-ONE" not in filtered, filtered)
        await pilot.press("escape")
        await pilot.pause(0.3)
        check("escape clears the filter", "TURN-ONE" in table_text(app, "messages-table"), "")

        await pilot.press("6")
        reservations = table_text(app, "reservations-table")
        check("active reservation listed", "/w/a/src | alpha" in reservations, reservations)
        check("expired and offline reservations are not", "/w/a/old" not in reservations
              and "/w/g/lib" not in reservations, reservations)  # fmt: skip

        await pilot.press("7")
        await wait_for(lambda: "opus-5.5" in table_text(app, "models-table"), pilot)
        models = table_text(app, "models-table")
        check("models tab with usage (a reply and a discussion turn)", "opus-5.5 | Claude Opus 5.5" in models
              and "| 2 | 1,205 | 306 |" in models, models)  # fmt: skip
        check("models tab shows vision", "glm-5.3 | GLM 5.3" in models and "| no |" in models, models)

        await pilot.press("8")
        check("errors tab", "HTTP 401" in table_text(app, "errors-table"), table_text(app, "errors-table"))
        await pilot.press("9")
        check("help tab", "Approvals" in text_of(app, "#help-text"), "")

        # --- approvals ------------------------------------------------------------
        await pilot.press("2")
        pending = table_text(app, "approvals-table")
        check("approvals list", "SEARCH-QUERY" in pending and "FULL-PROMPT" in pending, pending)
        table = app.query_one("#approvals-table", DataTable)
        table.focus()
        ids = sorted(gate_rows())
        table.move_cursor(row=table.get_row_index(str(ids[1])))
        await pilot.press("enter")
        await pilot.pause(0.5)
        content = text_of(app.screen, "#gate-content")
        check("a request opens in full, literally", "FULL-PROMPT" in content and "[bold]not bold[/]" in content
              and "\x1b" not in content, repr(content))  # fmt: skip
        await pilot.press("e")
        await pilot.pause(0.2)
        check("a model call cannot be edited", "GateScreen" in type(app.screen).__name__, type(app.screen).__name__)
        await pilot.press("d")
        await pilot.pause(0.3)
        await pilot.press(*"too risky", "enter")
        ok = await wait_for(lambda: gate_rows()[ids[1]][0] == "denied", pilot)
        check("deny with a reason", ok and gate_rows()[ids[1]][1] == "too risky", gate_rows()[ids[1]])

        table.focus()
        await wait_for(lambda: str(ids[0]) in table_text(app, "approvals-table"), pilot)
        table.move_cursor(row=table.get_row_index(str(ids[0])))
        await pilot.press("enter")
        await pilot.pause(0.5)
        await pilot.press("e")
        await pilot.pause(0.4)
        editor = app.screen.query_one("#editor")
        editor.text = "EDITED-QUERY"
        await pilot.press("ctrl+s")
        ok = await wait_for(lambda: gate_rows()[ids[0]][0] == "approved", pilot)
        check("edit and approve", ok and gate_rows()[ids[0]][2] == "EDITED-QUERY", gate_rows()[ids[0]])

        table.focus()
        await wait_for(lambda: str(ids[2]) in table_text(app, "approvals-table"), pilot)
        table.move_cursor(row=table.get_row_index(str(ids[2])))
        await pilot.press("enter")
        await pilot.pause(0.5)
        await pilot.press("a")
        ok = await wait_for(lambda: gate_rows()[ids[2]][0] == "approved", pilot)
        check("approve", ok and gate_rows()[ids[2]][2] is None, gate_rows()[ids[2]])

        # someone else decides while the request is open here
        table.focus()
        await wait_for(lambda: str(ids[3]) in table_text(app, "approvals-table"), pilot)
        table.move_cursor(row=table.get_row_index(str(ids[3])))
        await pilot.press("enter")
        await pilot.pause(0.5)
        agent_hub.store.gate_decide(ids[3], "approved", "decided elsewhere")
        await pilot.press("d")
        await pilot.pause(0.3)
        await pilot.press(*"late", "enter")
        await pilot.pause(1.0)
        check("a decision made elsewhere is not overwritten", gate_rows()[ids[3]][:2] == ("approved",
              "decided elsewhere"), gate_rows()[ids[3]])  # fmt: skip
        await pilot.press("3")
        await pilot.pause(0.6)
        check("pending count updates", "waiting at the gate" not in app.sub_title, app.sub_title)
        done = table_text(app, "approvals-done")
        check("decided list", "denied" in done and "too risky" in done, done)
    # Apart from the decisions, nothing was written: the decision function is the only writer.
    with sqlite3.connect(DB) as con:
        others = con.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
    check("the dashboard wrote no messages", others == 7, others)


async def test_deadlines() -> None:
    """Requests past their deadline show as expired and cannot be decided; publication requests are listed."""
    now = time.time()
    overdue = agent_hub.store.gate_open("agent_message", "x (standard)", "y (abliterated)", "OVERDUE-REQUEST", True)
    soon = agent_hub.store.gate_open("model_call", "x (standard)", "y (abliterated)", "EXPIRES-SOON", False)
    publish = agent_hub.store.gate_open("publish", "opus-5.5 (standard), answering x's public request #1",
                                        "every agent, both groups (PUBLIC)", "PUBLISH-THIS", False)  # fmt: skip
    with sqlite3.connect(DB) as con:
        con.execute("UPDATE gate SET expires=? WHERE id=?", (now - 5, overdue))
    app = hub_tui.HubDashboard(DB)
    async with app.run_test(size=(180, 50)) as pilot:
        await wait_for(lambda: "agents online" in text_of(app, "#overview-text"), pilot)
        await pilot.press("2")
        pending = table_text(app, "approvals-table")
        check("a request past its deadline is not listed as waiting", "OVERDUE-REQUEST" not in pending
              and "EXPIRES-SOON" in pending, pending)  # fmt: skip
        check("a publication request is listed, with the time left", "publish" in pending and "PUBLISH-THIS" in pending
              and "m " in pending, pending)  # fmt: skip
        done = table_text(app, "approvals-done")
        check("... it is shown as expired", any(str(overdue) in line and "expired" in line and "deadline" in line
                                                for line in done.splitlines()), done)  # fmt: skip
        check("... and not counted", "2 waiting at the gate" in app.sub_title, app.sub_title)
        table = app.query_one("#approvals-done", DataTable)
        table.focus()
        table.move_cursor(row=table.get_row_index(str(overdue)))
        await pilot.press("enter")
        await pilot.pause(0.5)
        head = text_of(app.screen, "#gate-head")
        check("an expired request opens as expired", "expired" in head, head)
        await pilot.press("a")
        await pilot.pause(0.5)
        check("... and cannot be approved", gate_rows()[overdue][0] == "pending", gate_rows()[overdue])
        await pilot.press("escape")
        await pilot.pause(0.2)
        # Open while still pending, decide after the deadline: the decision does not apply.
        table = app.query_one("#approvals-table", DataTable)
        table.focus()
        table.move_cursor(row=table.get_row_index(str(soon)))
        with sqlite3.connect(DB) as con:
            con.execute("UPDATE gate SET expires=? WHERE id=?", (time.time() + 1.5, soon))
        await pilot.press("enter")
        await pilot.pause(0.3)
        await asyncio.sleep(1.6)
        await pilot.press("a")
        await pilot.pause(0.5)
        check("a request that expired while open is not approved", gate_rows()[soon][0] == "pending", gate_rows()[soon])
        await pilot.press("escape")
        app.decide(soon, "approved")
        ok = await wait_for(lambda: gate_rows()[soon][0] == "expired", pilot)
        check("deciding it anyway marks it expired, not approved", ok, gate_rows()[soon])
    check("the gate's own decision path expires an overdue request", agent_hub.store.gate_decide(overdue, "approved")
          is False and gate_rows()[overdue][0] == "expired", gate_rows()[overdue])  # fmt: skip
    agent_hub.store.gate_decide(publish, "denied", "not public")


async def test_publication_denied() -> None:
    """Denying the publication of a model's answer from the dashboard: the reason reaches the requester (a notice
    in its inbox and thread), and the answer stays private to it."""
    store = agent_hub.store
    ask = store.add("t-pub", "alpha", "opus-5.5", "ask", "PUBLIC-REQUEST", rtype="model", public=True)
    answer = store.add("t-pub", "opus-5.5", "alpha", "reply", "UNPUBLISHED-ANSWER", reply_to=ask, public=True,
                       unpublished=True)  # fmt: skip
    gate_id = store.gate_open("publish", f"opus-5.5 (standard), answering alpha's public request #{ask}",
                              "every agent, both groups (PUBLIC)", "UNPUBLISHED-ANSWER", False, answer)  # fmt: skip
    app = hub_tui.HubDashboard(DB)
    async with app.run_test(size=(180, 50)) as pilot:
        await wait_for(lambda: "agents online" in text_of(app, "#overview-text"), pilot)
        await pilot.press("2")
        table = app.query_one("#approvals-table", DataTable)
        await wait_for(lambda: str(gate_id) in table_text(app, "approvals-table"), pilot)
        table.focus()
        table.move_cursor(row=table.get_row_index(str(gate_id)))
        await pilot.press("enter")
        await pilot.pause(0.5)
        await pilot.press("d")
        await pilot.pause(0.3)
        await pilot.press(*"NAMES-A-CUSTOMER", "enter")
        ok = await wait_for(lambda: gate_rows()[gate_id][0] == "denied", pilot)
    with sqlite3.connect(DB) as con:
        row = con.execute("SELECT held, public FROM messages WHERE id=?", (answer,)).fetchone()
        notice = con.execute("SELECT recipient, body, public FROM messages WHERE kind='notice' AND reply_to=?",
                             (answer,)).fetchone()  # fmt: skip
    check("denying a publication in the dashboard: the answer stays private", ok and row == (0, 0), row)
    check("... and the requester is told the reason", notice is not None and notice[0] == "alpha"
          and "NAMES-A-CUSTOMER" in notice[1] and notice[2] == 0, notice)  # fmt: skip


async def test_workspaces() -> None:
    """The Workspaces tab (key 0): the list with what the workers changed, the diff with the real paths and names
    (Enter), apply (a, d with deletions) and close (c), each only after a yes."""
    proj = TMP / "WsProj-77"
    (proj / "lib").mkdir(parents=True)
    (proj / "lib" / "acme_core.py").write_text("NAME = 'AcmeCorp'\nVALUE = 1\n")
    (proj / "lib" / "old.py").write_text("x = 1\n")
    here = os.getcwd()
    os.chdir(proj)  # a session's project folder is its working folder
    try:
        ws, _, _ = agent_hub._ws_create(None, ["lib"], [("AcmeCorp", "Company"), ("acme_core", "core")],
                                        "Raise VALUE", "tui test")  # fmt: skip
        agent_hub.store.ws_grant(ws.id, "agent", "helper", None)
    finally:
        os.chdir(here)
    folder = TMP / "workspaces" / ws.id
    (folder / "core.py").write_text("NAME = 'Company'\nVALUE = 2\n")  # what a worker did
    (folder / "old.py").unlink()
    original = proj / "lib" / "acme_core.py"
    app = hub_tui.HubDashboard(DB)
    async with app.run_test(size=(180, 50)) as pilot:
        await wait_for(lambda: "agents online" in text_of(app, "#overview-text"), pilot)
        await pilot.press("0")
        ok = await wait_for(lambda: "2 change(s)" in table_text(app, "workspaces-table"), pilot)
        listed = table_text(app, "workspaces-table")
        check("workspaces tab: id, owner, recipients, files, changes, origin", ok and ws.id in listed
              and "helper (agent)" in listed and "| 2 |" in listed and str(proj / "lib") in listed, listed)  # fmt: skip
        await pilot.press("question_mark")
        check("? opens the help", "Workspaces" in text_of(app, "#help-text")
              and app.query_one(hub_tui.TabbedContent).active == "help", "")  # fmt: skip
        await pilot.press("0")
        table = app.query_one("#workspaces-table", DataTable)
        table.focus()
        table.move_cursor(row=table.get_row_index(ws.id))
        await pilot.press("enter")
        ok = await wait_for(lambda: "workspace-text" in [w.id for w in app.screen.query(Static)]
                            and "VALUE = 2" in text_of(app.screen, "#workspace-text"), pilot)  # fmt: skip
        shown = text_of(app.screen, "#workspace-text")
        check("Enter: the diff with the real paths and the hidden names back", ok
              and f"{original} (workspace)" in shown and "NAME = 'AcmeCorp'" in shown and "+VALUE = 2" in shown
              and "hidden AcmeCorp as Company" in shown and "deleted:" in shown, shown)  # fmt: skip
        await pilot.press("f")
        ok = await wait_for(lambda: "as the workers got them" in text_of(app.screen, "#workspace-text"), pilot)
        shared = text_of(app.screen, "#workspace-text")
        check("f: the files as the workers got them", ok and "===== core.py" in shared and "NAME = 'Company'" in shared
              and "VALUE = 1" in shared, shared)  # fmt: skip
        await pilot.press("f")
        await wait_for(lambda: "+VALUE = 2" in text_of(app.screen, "#workspace-text"), pilot)
        await pilot.press("a")
        await pilot.pause(0.3)
        check("apply asks first", type(app.screen).__name__ == "ConfirmScreen", type(app.screen).__name__)
        await pilot.press("n")
        await pilot.pause(0.5)
        check("... and does nothing without a yes", original.read_text() == "NAME = 'AcmeCorp'\nVALUE = 1\n", "")
        await pilot.press("a")
        await pilot.pause(0.3)
        await pilot.press("y")
        ok = await wait_for(lambda: original.read_text() == "NAME = 'AcmeCorp'\nVALUE = 2\n", pilot)
        check("a + y applies, with the real names", ok and (proj / "lib" / "old.py").exists(), original.read_text())
        await pilot.press("d")
        await pilot.pause(0.3)
        await pilot.press("y")
        ok = await wait_for(lambda: not (proj / "lib" / "old.py").exists(), pilot)
        check("d + y applies the deletions too", ok, "")
        await pilot.press("c")
        await pilot.pause(0.3)
        await pilot.press("y")
        ok = await wait_for(lambda: not folder.exists(), pilot)
        check("c + y closes it: its folder is gone", ok, "")
        await pilot.press("escape")
        await wait_for(lambda: "closed" in table_text(app, "workspaces-table"), pilot)
        check("... and the list says closed", "closed" in table_text(app, "workspaces-table"),
              table_text(app, "workspaces-table"))  # fmt: skip
    with sqlite3.connect(DB) as con:
        status = con.execute("SELECT status FROM workspaces WHERE id=?", (ws.id,)).fetchone()
        files = con.execute("SELECT COUNT(*) FROM workspace_files WHERE workspace=?", (ws.id,)).fetchone()
    check("closed in the database, its contents gone", status == ("closed",) and files == (0,), (status, files))


async def test_workspace_markup() -> None:
    """A worker's file names are literal everywhere in the dashboard: markup in them styles nothing and its click
    actions do nothing (a click on the notification once approved a request). And apply writes exactly the diff on
    screen: a worker's later change means nothing is applied."""
    proj = TMP / "MarkupProj"
    (proj / "lib").mkdir(parents=True)
    (proj / "lib" / "core.py").write_text("VALUE = 1\n")
    here = os.getcwd()
    os.chdir(proj)
    try:
        ws, _, _ = agent_hub._ws_create(None, ["lib"], [], "Raise VALUE", "markup test")
    finally:
        os.chdir(here)
    names = ["[on red]styled.py", "[@click=app.quit]CLICK.py", "0[@click=screen.approve]notes.md"]
    for name in names:
        agent_hub._ws_write_text(ws, name, "y = 2\n")
    gate_id = agent_hub.store.gate_open("agent_message", "abl-agent (abliterated)", "std-agent (standard)",
                                        "NOT APPROVED BY A CLICK", True)  # fmt: skip
    folder = TMP / "workspaces" / ws.id
    app = hub_tui.HubDashboard(DB)
    async with app.run_test(size=(180, 50), notifications=True) as pilot:
        await wait_for(lambda: "agents online" in text_of(app, "#overview-text"), pilot)
        await pilot.press("0")
        table = app.query_one("#workspaces-table", DataTable)
        await wait_for(lambda: ws.id in table_text(app, "workspaces-table"), pilot)
        table.focus()
        table.move_cursor(row=table.get_row_index(ws.id))
        await pilot.press("a")
        ok = await wait_for(lambda: type(app.screen).__name__ == "WorkspaceScreen"
                            and "CLICK.py" in text_of(app.screen, "#workspace-text"), pilot)  # fmt: skip
        check("a on the list opens the diff first", ok, type(app.screen).__name__)
        (folder / "core.py").write_text("VALUE = 'unreviewed'\n")  # a worker writes after the diff was shown
        await pilot.press("a")
        await pilot.pause(0.3)
        await pilot.press("y")
        ok = await wait_for(lambda: any("nothing was applied" in str(n.message) for n in app._notifications), pilot)
        check("a change after the diff: nothing is applied", ok and not (proj / "lib" / names[0]).exists()
              and (proj / "lib" / "core.py").read_text() == "VALUE = 1\n", [n.message for n in app._notifications])  # fmt: skip
        await wait_for(lambda: "unreviewed" in text_of(app.screen, "#workspace-text"), pilot)
        await pilot.press("a")
        await pilot.pause(0.3)
        await pilot.press("y")
        ok = await wait_for(lambda: all((proj / "lib" / n).exists() for n in names), pilot)
        check("the refreshed diff applies", ok and "unreviewed" in (proj / "lib" / "core.py").read_text(), "")
        await pilot.pause(0.5)
        toasts = list(app.screen.query("Toast"))
        report = next((t for t in toasts if "with the hidden names back" in str(t._notification.message)), None)
        shown = report.render().plain if report is not None else ""
        check("the notification shows the names literally", report is not None and not report._notification.markup
              and all(n in shown or n[:20] in shown for n in names[:2]), shown)  # fmt: skip
        spots = [(x, y) for t in toasts for y in range(t.region.y, t.region.y + t.region.height)
                 for x in range(t.region.x, t.region.x + t.region.width) if app.screen.get_style_at(x, y).meta.get("@click")]
        check("... and carries no click action", not spots, spots[:3])
        if report is not None:
            await pilot.click(offset=(report.region.x + 2, report.region.y + report.region.height - 1))
            await pilot.pause(0.5)
        check("clicking it quits nothing and approves nothing", app.is_running and gate_rows()[gate_id][0] == "pending",
              gate_rows().get(gate_id))  # fmt: skip
        agent_hub.store.ws_close(ws.id)  # meanwhile, from elsewhere: there is no diff any more
        await pilot.press("f")
        await pilot.press("f")
        await wait_for(lambda: "(closed" in text_of(app.screen, "#workspace-text"), pilot)
        await pilot.press("a")
        ok = await wait_for(lambda: any("no diff to apply" in str(n.message) for n in app._notifications), pilot)
        check("a without a diff says why (and names no key that would not help)", ok and not any(
            "f shows it" in str(n.message) for n in app._notifications), [n.message for n in app._notifications])  # fmt: skip
    agent_hub.store.gate_decide(gate_id, "denied", "test over")


def stored(key: str) -> tuple | None:
    with sqlite3.connect(DB) as con:
        return con.execute("SELECT value, updated_by FROM settings WHERE key=?", (key,)).fetchone()


def hive_counts(item_id: int) -> tuple:
    with sqlite3.connect(DB) as con:
        return (con.execute("SELECT COUNT(*) FROM memories WHERE id=?", (item_id,)).fetchone()[0],
                con.execute("SELECT status FROM decisions WHERE id=?", (item_id,)).fetchone())  # fmt: skip


async def test_hive() -> None:
    """The hive tabs: settings (h) change only after a yes, with the value checked, and x drops the user's value;
    shared memory (m) lists notes, searches their whole text, deletes one after a yes; the decision log (l) shows
    a decision with its stances, withdraws and deletes after a yes. What agents wrote is shown literally."""
    from textual.widgets import Input, TabbedContent

    store = agent_hub.store
    note = {"author": "[bold]agent[/]", "cross_hold": False, "project": "/w/proj", "scope": "project", "tags": ",db,"}
    first = store.memory_add({**note, "text": "MEM-ONE [@click=app.quit]x[/] " + "pad " * 100 + "DEEP-WORD"}, 1000, 0)
    second = store.memory_add({**note, "text": "MEM-TWO the cache lives in /var/cache", "tags": ""}, 1000, 0)
    decision = store.decision_add({"author": "planner", "cross_hold": True, "project": "/w/proj",
                                   "title": "DEC-TITLE [red]x[/]", "decision": "DEC-BODY", "why": "DEC-WHY",
                                   "alternatives": "DEC-ALT"}, None, 5000)  # fmt: skip
    store.db.execute("INSERT INTO decision_stances (decision, ts, instance, name, abliterated, stance, note) VALUES"
                     " (?, ?, 'inst-x', 'critic', 1, 'disagree', 'STANCE-NOTE')", (decision, time.time()))  # fmt: skip
    review = store.gate_open("decision", "planner (standard), decision", "every abliterated agent", "Title: DEC-TITLE",
                             False, ref=f"decision:{decision}")  # fmt: skip
    app = hub_tui.HubDashboard(DB)
    async with app.run_test(size=(180, 50), notifications=True) as pilot:
        await wait_for(lambda: "agents online" in text_of(app, "#overview-text"), pilot)
        ok = await wait_for(lambda: "memory off, decisions off; 2 note(s), 1 active decision(s)" in text_of(
            app, "#overview-text"), pilot)  # fmt: skip
        check("overview: the hive's switches and counts", ok, text_of(app, "#overview-text"))
        await pilot.press("h")
        ok = await wait_for(lambda: "memory.enabled" in table_text(app, "hive-table"), pilot)
        settings = table_text(app, "hive-table")
        check("h: every setting, its value, where it comes from", ok and app.query_one(TabbedContent).active == "hive"
              and "memory.enabled | off | default | on or off" in settings and "context.max_chars | 4000" in settings,
              settings)  # fmt: skip
        table = app.query_one("#hive-table", DataTable)
        table.focus()
        table.move_cursor(row=table.get_row_index("memory.enabled"))
        await pilot.press("enter")
        await pilot.pause(0.3)
        check("a switch asks first", type(app.screen).__name__ == "ConfirmScreen", type(app.screen).__name__)
        await pilot.press("n")
        await pilot.pause(0.5)
        check("... no means no", stored("memory.enabled") is None, stored("memory.enabled"))
        await pilot.press("enter")
        await pilot.pause(0.3)
        await pilot.press("y")
        ok = await wait_for(lambda: stored("memory.enabled") == ("true", "dashboard"), pilot)
        check("... y turns it on", ok, stored("memory.enabled"))
        ok = await wait_for(lambda: "memory.enabled | on | set in the dashboard" in table_text(app, "hive-table"), pilot)
        check("... and the table says so", ok, table_text(app, "hive-table"))
        check("... as agent_hub reads it", agent_hub.hive("memory.enabled", agent_hub.store.settings()) is True, "")

        table.focus()
        table.move_cursor(row=table.get_row_index("memory.max_chars"))
        await pilot.press("enter")
        await pilot.pause(0.3)
        check("an option asks for its value", type(app.screen).__name__ == "AskScreen", type(app.screen).__name__)
        app.screen.query_one("#answer", Input).value = "5"
        await pilot.press("enter")
        ok = await wait_for(lambda: any("from 100 to 20000" in str(n.message) for n in app._notifications), pilot)
        check("... a value that is not allowed is refused", ok and stored("memory.max_chars") is None
              and type(app.screen).__name__ != "ConfirmScreen", [n.message for n in app._notifications])  # fmt: skip
        table.focus()
        await pilot.press("enter")
        await pilot.pause(0.3)
        app.screen.query_one("#answer", Input).value = "3000"
        await pilot.press("enter")
        await pilot.pause(0.3)
        check("... an allowed one asks for a yes", type(app.screen).__name__ == "ConfirmScreen", type(app.screen).__name__)
        await pilot.press("y")
        ok = await wait_for(lambda: stored("memory.max_chars") == ("3000", "dashboard"), pilot)
        check("... and is stored", ok, stored("memory.max_chars"))
        await wait_for(lambda: "memory.max_chars | 3000" in table_text(app, "hive-table"), pilot)
        table.focus()
        table.move_cursor(row=table.get_row_index("memory.max_chars"))
        await pilot.press("x")
        await pilot.pause(0.3)
        await pilot.press("y")
        ok = await wait_for(lambda: stored("memory.max_chars") is None, pilot)
        check("x drops the user's value, after a yes", ok, stored("memory.max_chars"))
        table.focus()
        table.move_cursor(row=table.get_row_index("memory.hub_scope"))
        await pilot.press("x")
        ok = await wait_for(lambda: any("has no value of yours" in str(n.message) for n in app._notifications), pilot)
        check("... and says when there is none", ok and type(app.screen).__name__ != "ConfirmScreen", "")
        # A stored value that cannot be used (hand-written, or from another version): x drops it too.
        store.db.execute("INSERT INTO settings (key, value, updated_at, updated_by) VALUES ('memory.max_entries',"
                         " '\"lots\"', 0, 'hand')")  # fmt: skip
        ok = await wait_for(lambda: "cannot be used" in table_text(app, "hive-table"), pilot)
        table.focus()
        table.move_cursor(row=table.get_row_index("memory.max_entries"))
        await pilot.press("x")
        await pilot.pause(0.3)
        check("x on an unusable stored value asks to drop it", ok and type(app.screen).__name__ == "ConfirmScreen"
              and "cannot be used" in text_of(app.screen, "#confirm-text"), type(app.screen).__name__)  # fmt: skip
        await pilot.press("y")
        ok = await wait_for(lambda: stored("memory.max_entries") is None, pilot)
        check("... and y drops it", ok, stored("memory.max_entries"))

        await pilot.press("m")
        ok = await wait_for(lambda: "MEM-ONE" in table_text(app, "memory-table"), pilot)
        notes = table_text(app, "memory-table")
        check("m: the notes, literally", ok and "[bold]agent[/] (standard)" in notes and "[@click=app.quit]x[/]" in notes
              and "| db |" in notes and "MEM-TWO" in notes, notes)  # fmt: skip
        await pilot.press("slash")
        await pilot.press(*"DEEP-WORD")
        ok = await wait_for(lambda: "MEM-ONE" in table_text(app, "memory-table")
                            and "MEM-TWO" not in table_text(app, "memory-table"), pilot)  # fmt: skip
        check("the filter looks in a note's whole text", ok, table_text(app, "memory-table"))
        await pilot.press("escape")
        await wait_for(lambda: "MEM-TWO" in table_text(app, "memory-table"), pilot)
        table = app.query_one("#memory-table", DataTable)
        table.focus()
        table.move_cursor(row=table.get_row_index(str(first)))
        await pilot.press("enter")
        ok = await wait_for(lambda: type(app.screen).__name__ == "HiveItemScreen"
                            and "DEEP-WORD" in text_of(app.screen, "#hive-item"), pilot)  # fmt: skip
        shown = text_of(app.screen, "#hive-item")
        check("Enter: the note in full, literally", ok and "[@click=app.quit]x[/]" in shown and "tags   db" in shown,
              shown)  # fmt: skip
        await pilot.press("x")
        await pilot.pause(0.3)
        await pilot.press("n")
        await pilot.pause(0.5)
        check("... x asks first; no means no", hive_counts(first)[0] == 1, hive_counts(first))
        await pilot.press("x")
        await pilot.pause(0.3)
        await pilot.press("y")
        ok = await wait_for(lambda: hive_counts(first)[0] == 0 and "no longer exists" in text_of(app.screen, "#hive-item"),
                            pilot)  # fmt: skip
        check("... y deletes it", ok, text_of(app.screen, "#hive-item"))
        await pilot.press("escape")
        await pilot.pause(0.3)
        check("nothing an agent wrote ran an action", app.is_running, "")

        await pilot.press("l")
        ok = await wait_for(lambda: "DEC-TITLE" in table_text(app, "decisions-table"), pilot)
        listed = table_text(app, "decisions-table")
        check("l: the decision log, literally, with where it stands at the gate", ok and "DEC-TITLE [red]x[/]" in listed
              and "active" in listed and "planner (standard)" in listed and "waiting at the gate" in listed, listed)  # fmt: skip
        table = app.query_one("#decisions-table", DataTable)
        table.focus()
        table.move_cursor(row=table.get_row_index(str(decision)))
        await pilot.press("enter")
        ok = await wait_for(lambda: type(app.screen).__name__ == "HiveItemScreen"
                            and "STANCE-NOTE" in text_of(app.screen, "#hive-item"), pilot)  # fmt: skip
        shown = text_of(app.screen, "#hive-item")
        check("Enter: the decision in full with its stances", ok and "DEC-WHY" in shown and "DEC-ALT" in shown
              and "critic (abliterated) disagree" in shown and f"gate request #{review} pending" in shown, shown)  # fmt: skip
        await pilot.press("w")
        await pilot.pause(0.3)
        await pilot.press("y")
        ok = await wait_for(lambda: hive_counts(decision)[1] == ("withdrawn",), pilot)
        check("w withdraws it, after a yes", ok, hive_counts(decision))
        await wait_for(lambda: "withdrawn by the user" in text_of(app.screen, "#hive-item"), pilot)
        await pilot.press("escape")
        await pilot.pause(0.3)
        table.focus()
        table.move_cursor(row=table.get_row_index(str(decision)))
        await pilot.press("x")
        await pilot.pause(0.3)
        await pilot.press("y")
        ok = await wait_for(lambda: hive_counts(decision)[1] is None, pilot)
        check("x deletes it, after a yes; its request at the gate expires", ok and gate_rows()[review][0] == "expired",
              gate_rows().get(review))  # fmt: skip
        # A replacement its writer's gate holds for the other group: superseded for the writer's group only.
        base = {"author": "planner", "project": "/w/proj", "decision": "D", "why": "W", "alternatives": ""}
        older = store.decision_add({**base, "cross_hold": False, "title": "OLD-RULE"}, None, 5000)
        newer = store.decision_add({**base, "cross_hold": True, "title": "NEW-RULE"}, older, 5000)
        ok = await wait_for(lambda: "OLD-RULE" in table_text(app, "decisions-table"), pilot)
        rows = {r.split(" | ")[0]: r for r in table_text(app, "decisions-table").splitlines()}
        check("the list works out which decisions are superseded", ok and "superseded" in rows.get(str(older), "")
              and "active" in rows.get(str(newer), ""), rows)  # fmt: skip
        table.focus()
        table.move_cursor(row=table.get_row_index(str(older)))
        await pilot.press("enter")
        ok = await wait_for(lambda: type(app.screen).__name__ == "HiveItemScreen"
                            and "OLD-RULE" in text_of(app.screen, "#hive-item"), pilot)  # fmt: skip
        shown = text_of(app.screen, "#hive-item")
        check("... and for whom", ok and "[superseded]" in shown and f"superseded by decision #{newer} (for its"
              " writer's group; for the other group once you approve it)" in shown, shown)  # fmt: skip
        await pilot.press("escape")
        await pilot.pause(0.3)
        table.focus()
        table.move_cursor(row=table.get_row_index(str(newer)))
        await pilot.press("x")
        await pilot.pause(0.3)
        question = text_of(app.screen, "#confirm-text") if type(app.screen).__name__ == "ConfirmScreen" else ""
        check("deleting a replacement says the one it replaced is in force again", f"Decision #{older}, which it"
              " replaced, is in force again" in question, question)  # fmt: skip
        await pilot.press("y")
        def older_row() -> str:
            return {r.split(" | ")[0]: r for r in table_text(app, "decisions-table").splitlines()}.get(str(older), "")

        ok = await wait_for(lambda: hive_counts(newer)[1] is None and "active" in older_row(), pilot)
        check("... y deletes it, and the other is active again", ok, table_text(app, "decisions-table"))
        store.decision_delete(older)
    store.memory_delete(second, "test over")
    store.setting_reset("memory.enabled")


def task_row(task_id: int) -> tuple | None:
    with sqlite3.connect(DB) as con:
        return con.execute("SELECT status, assignee, depends FROM tasks WHERE id=?", (task_id,)).fetchone()


def rating_count(model: str) -> int:
    with sqlite3.connect(DB) as con:
        return con.execute("SELECT COUNT(*) FROM ratings WHERE model=?", (model,)).fetchone()[0]


async def test_hive_board() -> None:
    """The task board (t) and the model track record (s): tasks shown literally, a claim whose session is gone
    marked; the filter looks in a task's whole text (its notes too); a task in full with its history and the gate
    states; s sets its status and a assigns it, each after a value check and a yes; x deletes it after a yes. The
    scores per kind and model; a model's ratings with the raters' notes, literally; x resets them after a yes."""
    from textual.widgets import Input, TabbedContent

    store = agent_hub.store
    now = time.time()
    base = {"author": "[b]planner[/b]", "cross_hold": True, "project": "/w/proj", "details": "", "depends": [],
            "assignee": ""}  # fmt: skip
    first = store.task_add({**base, "title": "TASK-ONE [@click=app.quit]x[/]", "details": "TASK-DETAILS"}, 200)
    second = store.task_add({**base, "title": "TASK-TWO", "depends": [first], "assignee": "beta"}, 200)
    store.db.execute("UPDATE tasks SET status='in_progress', claimer='inst-gone', claimer_name='worker', claimed_at=?,"
                     " renewed_at=? WHERE id=?", (now, now, first))  # fmt: skip
    store.db.execute("INSERT INTO task_notes (task, ts, instance, name, abliterated, cross_hold, what, note) VALUES"
                     " (?, ?, 'inst-x', 'critic', 1, 1, 'status claimed -> in_progress', 'NOTE-DEEP-WORD [red]x[/]')",
                     (first, now))  # fmt: skip
    note = store.db.execute("SELECT MAX(id) FROM task_notes").fetchone()[0]
    review = store.gate_open("task_note", "critic (abliterated), note on task", "every standard agent", "NOTE-DEEP-WORD",
                             True, ref=f"task_note:{note}")  # fmt: skip
    for n, (model, category, verdict, why) in enumerate([("opus-5.5", "ask", "useful", "RATING-NOTE [bold]x[/]"),
                                                         ("opus-5.5", "ask", "wrong", ""),
                                                         ("astra", "review", "useful", "")]):  # fmt: skip
        store.rate({"message_id": 9000 + n, "rater": "[i]rater[/i]", "model": model, "category": category,
                    "verdict": verdict, "note": why})  # fmt: skip
    con = hub_tui.connect(DB)  # read-only, as a GUI reads
    try:
        rows = {r["id"]: r for r in agent_hub.task_rows(con, "unfinished")}
        history = agent_hub.task_history(con, first)
        summary = agent_hub.rating_summary(con)
        found = [r["id"] for r in agent_hub.task_rows(con, query="note-deep-word")]
        ratings = agent_hub.ratings_of(con, "opus-5.5", "ask")
    finally:
        con.close()
    check("the GUIs' readers work on a read-only connection: tasks", rows[first]["claim_gone"]
          and rows[second]["depends"] == [first] and rows[first]["notes"] == 1 and found == [first], rows)  # fmt: skip
    check("... a task's history with the gate state of each note", history[0]["review"] == "pending"
          and history[0]["note"].startswith("NOTE-DEEP-WORD"), history)  # fmt: skip
    check("... the scores and the ratings", summary[0]["model"] == "opus-5.5" and summary[0]["score"] == 0.5
          and summary[0]["count"] == 2 and len(ratings) == 2 and ratings[1]["note"] == "RATING-NOTE [bold]x[/]",
          (summary, ratings))  # fmt: skip
    app = hub_tui.HubDashboard(DB)
    async with app.run_test(size=(200, 50), notifications=True) as pilot:
        ok = await wait_for(lambda: "task board off, track record off (auto-route off); 2 unfinished task(s), 3"
                            " rating(s)" in text_of(app, "#overview-text"), pilot)  # fmt: skip
        check("overview: the task board's and the track record's switches and counts", ok,
              text_of(app, "#overview-text"))  # fmt: skip
        await pilot.press("t")
        ok = await wait_for(lambda: "TASK-ONE" in table_text(app, "tasks-table"), pilot)
        listed = table_text(app, "tasks-table")
        check("t: the task board, literally; a claim whose session is gone is marked", ok
              and app.query_one(TabbedContent).active == "tasks" and "TASK-ONE [@click=app.quit]x[/]" in listed
              and "[b]planner[/b] (standard)" in listed and "worker (gone)" in listed and f"#{first}" in listed
              and "beta" in listed and "held for it" in listed, listed)  # fmt: skip
        await pilot.press("slash")
        await pilot.press(*"NOTE-DEEP-WORD")
        ok = await wait_for(lambda: "TASK-ONE" in table_text(app, "tasks-table")
                            and "TASK-TWO" not in table_text(app, "tasks-table"), pilot)  # fmt: skip
        check("the filter looks in a task's notes too", ok, table_text(app, "tasks-table"))
        await pilot.press("escape")
        await wait_for(lambda: "TASK-TWO" in table_text(app, "tasks-table"), pilot)
        table = app.query_one("#tasks-table", DataTable)
        table.focus()
        table.move_cursor(row=table.get_row_index(str(first)))
        await pilot.press("enter")
        ok = await wait_for(lambda: type(app.screen).__name__ == "HiveItemScreen"
                            and "TASK-DETAILS" in text_of(app.screen, "#hive-item"), pilot)  # fmt: skip
        shown = text_of(app.screen, "#hive-item")
        check("Enter: the task in full, its history and the gate states, literally", ok and "History (1)" in shown
              and "critic (abliterated): status claimed -> in_progress" in shown and f"gate request #{review} pending"
              in shown and "NOTE-DEEP-WORD [red]x[/]" in shown and "its session is gone" in shown, shown)  # fmt: skip
        await pilot.press("s")
        await pilot.pause(0.3)
        check("s asks for the status", type(app.screen).__name__ == "AskScreen", type(app.screen).__name__)
        app.screen.query_one("#answer", Input).value = "in_progress"
        await pilot.press("enter")
        ok = await wait_for(lambda: any("The status must be one of open, blocked, done, dropped" in str(n.message)
                                        for n in app._notifications), pilot)  # fmt: skip
        check("... a status the user cannot set is refused", ok and type(app.screen).__name__ == "HiveItemScreen"
              and task_row(first)[0] == "in_progress", task_row(first))  # fmt: skip
        await pilot.press("s")
        await pilot.pause(0.3)
        app.screen.query_one("#answer", Input).value = "done"
        await pilot.press("enter")
        await pilot.pause(0.3)
        check("... an allowed one asks for a yes", type(app.screen).__name__ == "ConfirmScreen", type(app.screen).__name__)
        await pilot.press("n")
        await pilot.pause(0.5)
        check("... no means no", task_row(first)[0] == "in_progress", task_row(first))
        await pilot.press("s")
        await pilot.pause(0.3)
        app.screen.query_one("#answer", Input).value = "done"
        await pilot.press("enter")
        await pilot.pause(0.3)
        await pilot.press("y")
        ok = await wait_for(lambda: task_row(first)[0] == "done" and "the user: status in_progress -> done" in text_of(
            app.screen, "#hive-item"), pilot)  # fmt: skip
        check("... y sets it, and its history says so", ok, (task_row(first), text_of(app.screen, "#hive-item")))
        await pilot.press("a")
        await pilot.pause(0.3)
        app.screen.query_one("#answer", Input).value = "Not A Name!"
        await pilot.press("enter")
        ok = await wait_for(lambda: any("An agent name is 2-32 characters" in str(n.message)
                                        for n in app._notifications), pilot)  # fmt: skip
        check("a assigns: a name no agent could use is refused", ok and task_row(first)[1] == "", task_row(first))
        await pilot.press("a")
        await pilot.pause(0.3)
        app.screen.query_one("#answer", Input).value = "gamma"
        await pilot.press("enter")
        await pilot.pause(0.3)
        await pilot.press("y")
        ok = await wait_for(lambda: task_row(first)[1] == "gamma", pilot)
        check("... a name is set after a yes", ok, task_row(first))
        await pilot.press("x")
        await pilot.pause(0.3)
        await pilot.press("y")
        ok = await wait_for(lambda: task_row(first) is None and "no longer exists" in text_of(app.screen, "#hive-item"),
                            pilot)  # fmt: skip
        check("x deletes it after a yes; the task that waited for it no longer does", ok
              and task_row(second)[2] == "", task_row(second))  # fmt: skip
        await pilot.press("escape")
        await pilot.pause(0.3)
        check("nothing an agent wrote ran an action", app.is_running, "")

        await pilot.press("s")
        ok = await wait_for(lambda: "opus-5.5" in table_text(app, "scores-table"), pilot)
        scores = table_text(app, "scores-table")
        check("s: the scores per kind and model, with the counts", ok and app.query_one(TabbedContent).active == "scores"
              and "ask | opus-5.5 | 0.50 | 1 | 0 | 1 | 2" in scores and "review | astra | 1.00 | 1 | 0 | 0 | 1" in scores,
              scores)  # fmt: skip
        table = app.query_one("#scores-table", DataTable)
        table.focus()
        table.move_cursor(row=table.get_row_index(json.dumps(["opus-5.5", "ask"])))
        await pilot.press("enter")
        ok = await wait_for(lambda: type(app.screen).__name__ == "HiveItemScreen"
                            and "RATING-NOTE" in text_of(app.screen, "#hive-item"), pilot)  # fmt: skip
        shown = text_of(app.screen, "#hive-item")
        check("Enter: the ratings with the raters' notes, literally", ok and "opus-5.5 for ask: 0.50 from 2 rating(s)"
              in shown and "RATING-NOTE [bold]x[/]" in shown and "[i]rater[/i]" in shown, shown)  # fmt: skip
        await pilot.press("x")
        await pilot.pause(0.3)
        await pilot.press("n")
        await pilot.pause(0.5)
        check("... x asks first; no means no", rating_count("opus-5.5") == 2, rating_count("opus-5.5"))
        await pilot.press("x")
        await pilot.pause(0.3)
        await pilot.press("y")
        ok = await wait_for(lambda: rating_count("opus-5.5") == 0 and "No ratings of opus-5.5" in text_of(
            app.screen, "#hive-item"), pilot)  # fmt: skip
        check("... y resets them", ok, rating_count("opus-5.5"))
        await pilot.press("escape")
        await pilot.pause(0.3)
        await wait_for(lambda: "opus-5.5" not in table_text(app, "scores-table"), pilot)
        table.focus()
        table.move_cursor(row=table.get_row_index(json.dumps(["astra", "review"])))
        await pilot.press("x")
        await pilot.pause(0.3)
        await pilot.press("y")
        ok = await wait_for(lambda: rating_count("astra") == 0, pilot)
        check("x on the Scores tab resets the selected ratings after a yes", ok, rating_count("astra"))
    store.task_delete(second)
    store.gate_decide(review, "denied", "test over")


def test_ratings_view() -> None:
    """A model's ratings view takes its score and counts from every rating (as the Scores list and agents do), and
    lists only the newest."""
    now = time.time()
    agent_hub.store.db.executemany(
        "INSERT INTO ratings (ts, message_id, instance, rater, abliterated, model, category, verdict) VALUES"
        " (?,?,?,?,?,?,?,?)", [(now + i, 200000 + i, "inst-r", "rater", 0, "many-5.0", "ask",
                                "wrong" if i < 100 else "useful") for i in range(600)])  # fmt: skip
    summary = next(s for s in agent_hub.rating_summary(agent_hub.store.db, "ask") if s["model"] == "many-5.0")
    shown = hub_tui.read_ratings(DB, "many-5.0", "ask").plain
    check("the ratings view agrees with the Scores list past 500 ratings", f"{summary['score']:.2f} from 600" in shown
          and "many-5.0 for ask: 0.83 from 600 rating(s): 500 useful, 0 partly, 100 wrong" in shown, shown[:300])  # fmt: skip
    check("... and lists the newest 500", "The newest 500 of them" in shown and shown.count(" by rater ") == 500, "")
    agent_hub.store.ratings_reset("many-5.0")


def test_once() -> None:
    done = subprocess.run([sys.executable, str(PROJECT / "hub_tui.py"), "--once"], capture_output=True, text=True,
                          env={**os.environ}, timeout=60)  # fmt: skip
    check("--once prints the overview", done.returncode == 0 and "agents online" in done.stdout, done.stdout + done.stderr)
    empty = TMP / "empty"
    empty.mkdir()
    done = subprocess.run([sys.executable, str(PROJECT / "hub_tui.py"), "--once"], capture_output=True, text=True,
                          env={**os.environ, "AGENT_HUB_HOME": str(empty)}, timeout=60)  # fmt: skip
    check("--once without a database", done.returncode == 0 and "No hub database yet" in done.stdout, done.stdout)
    check("--once does not create one", not (empty / "hub.db").exists(), "")


def test_paths() -> None:
    home = TMP / "paths"
    home.mkdir()
    (home / "keys.env").write_text('# settings\nexport AGENT_HUB_DB="~/elsewhere/hub.db"\n')
    saved = {k: os.environ.get(k) for k in ("AGENT_HUB_HOME", "AGENT_HUB_DB")}
    try:
        os.environ["AGENT_HUB_HOME"] = str(home)
        os.environ.pop("AGENT_HUB_DB", None)
        check("database path from keys.env", hub_tui.db_path() == Path("~/elsewhere/hub.db").expanduser(),
              hub_tui.db_path())  # fmt: skip
        os.environ["AGENT_HUB_DB"] = str(home / "env.db")
        check("the environment wins over keys.env", hub_tui.db_path() == home / "env.db", hub_tui.db_path())
        (home / "keys.env").unlink()
        os.environ.pop("AGENT_HUB_DB")
        check("default database path", hub_tui.db_path() == home / "hub.db", hub_tui.db_path())
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    check("cleaning removes escapes and controls", hub_tui.clean("a\x1b[31mb\x1b]8;;http://x\x07c\x07d\x00e")
          == "abc?d?e", repr(hub_tui.clean("a\x1b[31mb\x1b]8;;http://x\x07c\x07d\x00e")))  # fmt: skip


async def main() -> None:
    print("== paths")
    test_paths()
    for name, test in [("no database", test_no_database), ("dashboard", test_dashboard),
                       ("deadlines", test_deadlines), ("publication denied", test_publication_denied),
                       ("workspaces", test_workspaces), ("workspace names are literal", test_workspace_markup),
                       ("hive", test_hive), ("task board and scores", test_hive_board)]:  # fmt: skip
        print(f"== {name}")
        try:
            await asyncio.wait_for(test(), 120)
        except Exception:
            FAILED.append(f"{name}: crashed")
            traceback.print_exc()
    print("== ratings view")
    test_ratings_view()
    print("== once")
    test_once()
    print(f"\n{len(PASSED)} passed, {len(FAILED)} failed")
    for name in FAILED:
        print(f"  FAILED: {name}")
    sys.exit(1 if FAILED else 0)


if __name__ == "__main__":
    asyncio.run(main())
