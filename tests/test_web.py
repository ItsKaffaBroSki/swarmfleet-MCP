"""Tests for the browser control panel (hub_web.py): panels started on free ports, driven with http.client the way a
browser would drive them, and the way another page or program would try to.

    python tests/test_web.py            (needs: pip install "mcp>=2,<3" httpx)
"""

from __future__ import annotations

import base64
import hashlib
import http.client
import json
import os
import queue
import re
import shutil
import socket
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

PROJECT = Path(__file__).resolve().parent.parent
TMP = Path(tempfile.mkdtemp())
for _name in [n for n in os.environ if n.startswith("AGENT_HUB_")]:
    os.environ.pop(_name)  # the hive settings of whoever runs the tests must not leak in
os.environ["AGENT_HUB_HOME"] = str(TMP / "home")  # before agent_hub or hub_web read it
os.environ["AGENT_HUB_LAZY_STORE"] = "1"
sys.path.insert(0, str(PROJECT))

import agent_hub  # noqa: E402
import hub_web  # noqa: E402

PASSED: list[str] = []
FAILED: list[str] = []
RESPONSES: list[Response] = []  # every answer the panels gave, for the header checks
DB = TMP / "home" / "hub.db"

XSS = '<script>alert("x")</script>'
ATTR = '"><img src=x onerror=alert(1)>'
QUOTE = "' onmouseover='alert(2)"
MARK = "<b>bold</b>"
XSS_SEEN = "&lt;script&gt;alert(&quot;x&quot;)&lt;/script&gt;"
ATTR_SEEN = "&quot;&gt;&lt;img src=x onerror=alert(1)&gt;"
QUOTE_SEEN = "&#x27; onmouseover=&#x27;alert(2)"
MARK_SEEN = "&lt;b&gt;bold&lt;/b&gt;"


def check(name: str, condition: Any, detail: Any = "") -> None:
    if condition:
        PASSED.append(name)
    else:
        FAILED.append(name)
        print(f"  FAIL {name}: {str(detail)[:700]}")


class Response:
    def __init__(self, status: int, headers: list[tuple[str, str]], text: str) -> None:
        self.status, self.headers, self.text = status, headers, text
        RESPONSES.append(self)

    def header(self, name: str) -> str:
        return next((v for k, v in self.headers if k.lower() == name.lower()), "")

    def __repr__(self) -> str:
        return f"<{self.status} {self.text[:300]!r}>"


class Browser:
    """What a browser sends: the Host, the panel's cookie, and with a form the Origin and the CSRF token of the last
    page it loaded. Every part can be left out or replaced, as another page or program might."""

    def __init__(self, panel: hub_web.Panel) -> None:
        self.panel, self.port = panel, panel.port
        self.cookie, self.csrf = "", ""
        self.host = f"127.0.0.1:{self.port}"

    def raw(self, method: str, path: str, body: bytes = b"", headers: dict[str, Any] | None = None,
            host: str | None | list[str] = "") -> Response:  # fmt: skip
        con = http.client.HTTPConnection("127.0.0.1", self.port, timeout=15)
        try:
            con.putrequest(method, path, skip_host=True, skip_accept_encoding=True)
            hosts = [self.host] if host == "" else [] if host is None else [host] if isinstance(host, str) else host
            for value in hosts:
                con.putheader("Host", value)
            for name, value in (headers or {}).items():
                for one in value if isinstance(value, list) else [value]:
                    con.putheader(name, one)
            if body or method == "POST":
                con.putheader("Content-Length", str(len(body)))
            con.endheaders(body or None)
            r = con.getresponse()
            return Response(r.status, r.getheaders(), r.read().decode("utf-8", "replace"))
        finally:
            con.close()

    def cookie_header(self) -> dict[str, str]:
        return {"Cookie": f"{self.panel.cookie}={self.cookie}"} if self.cookie else {}

    def get(self, path: str, headers: dict[str, Any] | None = None, host: Any = "", cookie: bool = True) -> Response:
        r = self.raw("GET", path, headers={**(self.cookie_header() if cookie else {}), **(headers or {})}, host=host)
        found = re.search(r'name="csrf" value="([^"]+)"', r.text)
        if found:
            self.csrf = found.group(1)
        return r

    def post(self, path: str, fields: dict[str, str], csrf: Any = True, origin: Any = "",
             ctype: str = "application/x-www-form-urlencoded", headers: dict[str, Any] | None = None, host: Any = "",
             cookie: bool = True, body: bytes | None = None) -> Response:  # fmt: skip
        data = dict(fields)
        if csrf is True:
            data["csrf"] = self.csrf
        elif csrf is not None:
            data["csrf"] = csrf
        sent: dict[str, Any] = {"Content-Type": ctype} if ctype else {}
        if origin is not None:
            sent["Origin"] = origin or f"http://{self.host}"
        sent.update(self.cookie_header() if cookie else {})
        sent.update(headers or {})
        return self.raw("POST", path, urlencode(data).encode() if body is None else body, sent, host)

    def sign_in(self) -> Response:
        r = self.raw("GET", "/login?token=" + (self.panel.token or ""))
        found = re.match(rf"{self.panel.cookie}=([^;]+)", r.header("Set-Cookie"))
        self.cookie = found.group(1) if found else ""
        self.get("/")
        return r


class Tags(HTMLParser):
    """Every tag and attribute of a page, to look for anything that could run or load."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tags: list[str] = []
        self.attrs: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.tags.append(tag)
        self.attrs += [name for name, _ in attrs]


def inert(text: str) -> tuple[bool, str]:
    """No script, image, frame, object or inline handler or style in the page."""
    parser = Tags()
    parser.feed(text)
    bad_tags = {"script", "img", "svg", "iframe", "object", "embed", "link", "base", "frame", "video", "audio"}
    bad = [t for t in parser.tags if t in bad_tags] + [a for a in parser.attrs if a.startswith("on") or a in (
        "style", "srcdoc", "formaction")]  # fmt: skip
    return not bad, str(bad)


def snapshot(path: Path) -> dict[str, list[str]]:
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        names = [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
        return {n: sorted(map(repr, con.execute(f'SELECT * FROM "{n}"').fetchall())) for n in names}
    finally:
        con.close()


def stored(key: str, db: Path = DB) -> tuple[str, str] | None:
    with sqlite3.connect(db) as con:
        row = con.execute("SELECT value, updated_by FROM settings WHERE key=?", (key,)).fetchone()
    return None if row is None else (row[0], row[1])


def one(sql: str, params: tuple[Any, ...] = (), db: Path = DB) -> Any:
    with sqlite3.connect(db) as con:
        row = con.execute(sql, params).fetchone()
    return None if row is None else row[0] if len(row) == 1 else row


def flashes(text: str) -> str:
    return " | ".join(re.findall(r'<div class="flash[^"]*" role="status">(.*?)</div>', text, re.S))


IDS: dict[str, int] = {}


def fill_database() -> None:
    store = agent_hub.Store(DB)
    try:
        now = time.time()
        store.db.execute("INSERT INTO agents (name, instance, description, cwd, joined_at, last_seen) VALUES"
                         " (?,?,?,?,?,?)", ("alpha" + XSS, "inst-a", "d", "/w", now, now))  # fmt: skip
        store.db.execute("INSERT INTO agents (name, instance, description, cwd, joined_at, last_seen) VALUES"
                         " (?,?,?,?,?,?)", ("ghost", "inst-g", "d", "/w", now - 999, now - 999))  # fmt: skip
        note = {"author": ATTR, "cross_hold": True, "project": "/w/proj" + QUOTE, "scope": "project",
                "tags": f",{MARK},db,"}  # fmt: skip
        IDS["m1"] = store.memory_add({**note, "text": "MEM-ONE " + XSS + "\nline two " + ATTR + " \x1b[31mRED‮RTL"},
                                     1000, 0)  # fmt: skip
        IDS["m2"] = store.memory_add({**note, "author": "beta", "cross_hold": False, "text": "MEM-TWO the cache lives"
                                      " in /var/cache", "tags": ""}, 1000, 0)  # fmt: skip
        IDS["m3"] = store.memory_add({**note, "author": "gamma", "scope": "hub", "project": "", "text": "MEM-THREE hub"
                                      " wide", "tags": ",ops,"}, 1000, 0)  # fmt: skip
        IDS["g_m1"] = store.gate_open("memory", "x (standard), note", "every abliterated agent", "MEM-ONE", True,
                                      ref=f"memory:{IDS['m1']}")  # fmt: skip
        decision = {"author": "planner" + MARK, "cross_hold": False, "project": "/w/proj", "why": "WHY " + QUOTE,
                    "alternatives": "ALT " + ATTR}  # fmt: skip
        IDS["d0"] = store.decision_add({**decision, "title": "DEC-ZERO old way", "decision": "old"}, None, 5000)
        IDS["d1"] = store.decision_add({**decision, "title": "DEC-ONE " + XSS, "decision": "DEC-BODY " + MARK}, None,
                                       5000)  # fmt: skip
        IDS["d2"] = store.decision_add({**decision, "title": "DEC-TWO new way", "decision": "new"}, IDS["d0"], 5000)
        store.db.execute("INSERT INTO decision_stances (decision, ts, instance, name, abliterated, stance, note) VALUES"
                         " (?, ?, 'inst-x', ?, 1, 'disagree', ?)", (IDS["d1"], now, "critic" + ATTR,
                                                                     "STANCE-NOTE " + XSS))  # fmt: skip
        store.db.execute("INSERT INTO decision_stances (decision, ts, instance, name, abliterated, stance, note) VALUES"
                         " (?, ?, 'inst-y', 'fan', 0, 'agree', 'yes')", (IDS["d2"], now))  # fmt: skip
        IDS["g_d2"] = store.gate_open("decision", "planner (standard), decision", "every abliterated agent",
                                      "Title: DEC-TWO", False, ref=f"decision:{IDS['d2']}")  # fmt: skip
        store.gate_open("model_call", "alpha (standard)", "glm (abliterated)", "PROMPT", False)
    finally:
        store.db.close()


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


# --------------------------------------------------------------------------
# The command line
# --------------------------------------------------------------------------


def test_cli() -> None:
    home = TMP / "cli-home"
    env = {**os.environ, "AGENT_HUB_HOME": str(home)}
    env.pop("AGENT_HUB_LAZY_STORE", None)
    port = free_port()
    proc = subprocess.Popen([sys.executable, str(PROJECT / "hub_web.py"), "--port", str(port)], env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)  # fmt: skip
    lines: queue.Queue[str] = queue.Queue()
    threading.Thread(target=lambda: [lines.put(line) for line in proc.stdout], daemon=True).start()  # type: ignore[union-attr]
    try:
        url, seen = "", []
        deadline = time.time() + 60
        while not url and time.time() < deadline:
            try:
                line = lines.get(timeout=1)
            except queue.Empty:
                if proc.poll() is not None:
                    break
                continue
            seen.append(line)
            found = re.search(r"http://\S+", line)
            url = found.group(0) if found else ""
        check("prints a sign-in link on 127.0.0.1 with a long random token",
              re.fullmatch(rf"http://127\.0\.0\.1:{port}/login\?token=[A-Za-z0-9_-]{{40,}}", url), seen)  # fmt: skip
        while not any(f"127.0.0.1:{port} only" in line for line in seen) and time.time() < deadline:
            try:
                seen.append(lines.get(timeout=1))
            except queue.Empty:
                break
        check("... and says it works once and listens on 127.0.0.1 only", "works once" in "".join(seen) and any(
            f"127.0.0.1:{port} only" in line for line in seen), seen)  # fmt: skip
        con = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
        con.request("GET", url.split(str(port), 1)[1], headers={"Host": f"127.0.0.1:{port}"})
        r = con.getresponse()
        r.read()
        cookie = (r.getheader("Set-Cookie") or "").split(";")[0]
        check("the link signs in", r.status == 303 and cookie.startswith(f"agent_hub_panel_{port}="), r.status)
        con = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
        con.request("GET", "/", headers={"Host": f"localhost:{port}", "Cookie": cookie})
        r = con.getresponse()
        page = r.read().decode()
        check("the panel works with no database yet", r.status == 200 and "No hub database yet" in page, page[:300])
        check("... and does not create it", not (home / "hub.db").exists(), list(home.iterdir()) if home.exists() else "")
        busy = subprocess.run([sys.executable, str(PROJECT / "hub_web.py"), "--port", str(port)], env=env,
                              capture_output=True, text=True, timeout=60)  # fmt: skip
        check("a port in use: says so and stops", busy.returncode == 1 and "cannot listen" in busy.stderr
              and "--port" in busy.stderr, (busy.returncode, busy.stderr))  # fmt: skip
        bad = subprocess.run([sys.executable, str(PROJECT / "hub_web.py"), "--port", "70000"], env=env,
                             capture_output=True, text=True, timeout=60)  # fmt: skip
        check("a port out of range is refused", bad.returncode == 2 and "--port" in bad.stderr, bad.stderr)
        host = subprocess.run([sys.executable, str(PROJECT / "hub_web.py"), "--host", "0.0.0.0"], env=env,
                              capture_output=True, text=True, timeout=60)  # fmt: skip
        check("there is no option to listen on another address", host.returncode == 2, host.stderr)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
    err = proc.stderr.read() if proc.stderr else ""  # type: ignore[union-attr]
    check("the token is never logged", "token=" not in err, err)


# --------------------------------------------------------------------------
# No database yet
# --------------------------------------------------------------------------


def test_no_database() -> None:
    later = TMP / "later" / "hub.db"
    panel = hub_web.Panel(later, 0)
    panel.serve_in_thread()
    try:
        b = Browser(panel)
        b.sign_in()
        for path in ("/", "/settings", "/memory", "/decisions", "/tasks", "/scores", "/memory?q=x",
                     "/decisions?status=active"):  # fmt: skip
            r = b.get(path)
            ok = r.status == 200 and "No hub database yet" in r.text
            check(f"no database: {path} works and says so", ok, r)
        settings = b.get("/settings").text
        check("no database: every setting is listed, with its default", all(k in settings for k in agent_hub.SETTINGS)
              and "default" in settings, settings[-2000:])  # fmt: skip
        check("no database: viewing creates nothing", not later.parent.exists(), "")
        r = b.post("/memory/delete", {"id": "1", "confirm": "yes"})
        check("no database: a delete changes nothing and creates nothing", r.status == 303 and not later.parent.exists()
              and "No hub database yet" in flashes(b.get("/memory").text), r)  # fmt: skip
        b.post("/decisions/withdraw", {"id": "1", "confirm": "yes"})
        b.post("/settings/reset", {"key": "memory.enabled", "confirm": "yes"})
        check("no database: withdraw and reset create nothing", not later.parent.exists(), "")
        r = b.post("/settings/set", {"key": "memory.enabled", "value": "on", "confirm": "yes"})
        check("no database: a setting you change creates it (as agent_hub.py settings set does)", r.status == 303
              and later.exists() and stored("memory.enabled", later) == ("true", hub_web.BY), r)  # fmt: skip
        page = b.get("/settings").text
        check("... and the panel reads it from then on", "set in the control panel" in page, page[-1500:])
    finally:
        panel.stop()


# --------------------------------------------------------------------------
# Signing in
# --------------------------------------------------------------------------


def test_sign_in(panel: hub_web.Panel) -> Browser:
    b = Browser(panel)
    token = panel.token or ""
    check("listens on 127.0.0.1 only", panel.httpd.server_address[0] == "127.0.0.1", panel.httpd.server_address)
    try:  # another address of this machine, if it has one, does not reach the panel
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))
            mine = s.getsockname()[0]
    except OSError:
        mine = ""
    if mine and not mine.startswith("127."):
        try:
            socket.create_connection((mine, panel.port), timeout=3).close()
            reached = True
        except OSError:
            reached = False
        check("... not reachable on this machine's other address", not reached, mine)
    r = b.raw("GET", "/login")
    check("sign-in without a token: refused", r.status == 403 and "Set-Cookie" not in str(r.headers), r)
    r = b.raw("GET", "/login?token=" + "A" * len(token))
    check("sign-in with a wrong token: refused", r.status == 403 and not r.header("Set-Cookie"), r)
    r = b.raw("GET", "/login?token=" + token + "x")
    check("sign-in with a longer token: refused", r.status == 403 and not r.header("Set-Cookie"), r)
    r = b.raw("GET", "/login?token=" + token, host=f"evil.example:{panel.port}")
    check("sign-in through another host name: refused", r.status == 400 and not r.header("Set-Cookie"), r)
    r = b.raw("GET", "/login?token=" + token, headers={"Sec-Fetch-Site": "cross-site"})
    check("sign-in started by another site: refused", r.status == 403 and not r.header("Set-Cookie"), r)
    check("... and the refused attempts did not use the link up", panel.token == token, panel.token)
    r = b.raw("GET", "/login?token=" + token, headers={"Sec-Fetch-Site": "none", "Sec-Fetch-Mode": "navigate"})
    cookie = r.header("Set-Cookie")
    parts = [p.strip() for p in cookie.split(";")]
    check("the link signs in and goes to the overview", r.status == 303 and r.header("Location") == "/", r)
    check("... with an HttpOnly, SameSite=Strict session cookie for this panel only", parts[0].startswith(
        f"agent_hub_panel_{panel.port}=") and "HttpOnly" in parts and "SameSite=Strict" in parts and "Path=/" in parts
        and not any(p.lower().startswith(("domain", "expires", "max-age")) for p in parts), cookie)  # fmt: skip
    check("... whose value is not the token", token not in cookie and len(parts[0].split("=", 1)[1]) >= 40, cookie)
    b.cookie = parts[0].split("=", 1)[1]
    r = b.raw("GET", "/login?token=" + token)
    check("the link works once", r.status == 403 and "used already" in r.text and not r.header("Set-Cookie"), r)
    r = b.get("/")
    check("signed in: the overview", r.status == 200 and "Overview" in r.text and b.csrf, r)
    cut = b.cookie[:-1] + ("A" if b.cookie[-1] != "A" else "B")
    r = b.raw("GET", "/", headers={"Cookie": f"{panel.cookie}={cut}"})
    check("a changed cookie: refused", r.status == 403, r)
    r = b.raw("GET", "/", headers={"Cookie": f"agent_hub_panel_1={b.cookie}"})
    check("the right value under another panel's name: refused", r.status == 403, r)
    r = b.raw("GET", "/", headers={"Cookie": f"{panel.cookie}=éé"})
    check("a cookie that is not ASCII: refused, no crash", r.status == 403, r)
    r = b.get("/", headers={"Cookie": f"other=1; {panel.cookie}={b.cookie}; more=2"})
    check("the cookie among others: works", r.status == 200, r)
    return b


def test_no_cookie(panel: hub_web.Panel, b: Browser) -> None:
    for path in ("/", "/settings", "/memory", "/decisions", "/tasks", "/scores", "/nothing", "/favicon.ico"):
        r = b.get(path, cookie=False)
        check(f"without the cookie: {path} refused, nothing shown", r.status == 403 and "MEM-ONE" not in r.text
              and "memory.enabled" not in r.text and "csrf" not in r.text and "Open the link" in r.text, r)  # fmt: skip
    before = snapshot(DB)
    r = b.post("/settings/set", {"key": "memory.hub_scope", "value": "on", "confirm": "yes"}, cookie=False)
    check("without the cookie: a change (with a valid CSRF token) refused", r.status == 403, r)
    r = b.post("/memory/delete", {"id": str(IDS["m2"]), "confirm": "yes"}, cookie=False)
    check("without the cookie: a delete refused", r.status == 403, r)
    check("... and nothing changed", snapshot(DB) == before, "")


def test_host(panel: hub_web.Panel, b: Browser) -> None:
    port = panel.port
    for host in (f"evil.example:{port}", f"127.0.0.1:{port + 1}", "127.0.0.1", "localhost", f"[::1]:{port}",
                 f"127.0.0.1:{port}.evil.example", f"LOCALHOST:{port}", f"127.0.0.1:{port} ", "", None,
                 [f"127.0.0.1:{port}", f"evil.example:{port}"]):  # fmt: skip
        r = b.get("/memory", host=host if host != "" else " ")
        check(f"Host {host!r}: refused (DNS rebinding)", r.status == 400 and "MEM-ONE" not in r.text, r)
    r = b.get("/memory", host=f"localhost:{port}")
    check("Host localhost:<port>: works", r.status == 200 and "MEM-ONE" in r.text, r)
    before = snapshot(DB)
    r = b.post("/settings/set", {"key": "memory.hub_scope", "value": "on", "confirm": "yes"},
               origin=f"http://evil.example:{port}", host=f"evil.example:{port}")  # fmt: skip
    check("a change through another host name: refused", r.status == 400 and snapshot(DB) == before, r)


def test_post_guards(panel: hub_web.Panel, b: Browser) -> None:
    port = panel.port
    change = {"key": "memory.hub_scope", "value": "on", "confirm": "yes"}
    b.get("/settings")
    before = snapshot(DB)
    cases: list[tuple[str, dict[str, Any], int]] = [
        ("no CSRF token", {"csrf": None}, 403),
        ("an empty CSRF token", {"csrf": ""}, 403),
        ("a wrong CSRF token", {"csrf": "x" * len(b.csrf)}, 403),
        ("the CSRF token with one character changed", {"csrf": b.csrf[:-1] + ("A" if b.csrf[-1] != "A" else "B")}, 403),
        ("a CSRF token that is not ASCII", {"csrf": "é" * 10}, 403),
        ("the session cookie as the CSRF token", {"csrf": b.cookie}, 403),
        ("no Origin", {"origin": None}, 403),
        ("Origin of another site", {"origin": "http://evil.example"}, 403),
        ("Origin of another port", {"origin": f"http://127.0.0.1:{port + 1}"}, 403),
        ("Origin null", {"origin": "null"}, 403),
        ("Origin https", {"origin": f"https://127.0.0.1:{port}"}, 403),
        ("Origin of the other local name", {"origin": f"http://localhost:{port}"}, 403),
        ("Origin with a path", {"origin": f"http://127.0.0.1:{port}/"}, 403),
        ("two Origins", {"headers": {"Origin": [f"http://127.0.0.1:{port}", "http://evil.example"]}, "origin": None},
         403),
        ("Sec-Fetch-Site same-site (another local port)", {"headers": {"Sec-Fetch-Site": "same-site"}}, 403),
        ("Sec-Fetch-Site cross-site", {"headers": {"Sec-Fetch-Site": "cross-site"}}, 403),
        ("Sec-Fetch-Site none", {"headers": {"Sec-Fetch-Site": "none"}}, 403),
        ("a text/plain body", {"ctype": "text/plain"}, 415),
        ("a JSON body", {"ctype": "application/json"}, 415),
        ("a multipart body", {"ctype": "multipart/form-data; boundary=x"}, 415),
        ("no Content-Type", {"ctype": ""}, 415),
        ("a body over 64 KB", {"body": urlencode({**change, "csrf": b.csrf, "pad": "x" * 70_000}).encode()}, 413),
        ("too many fields", {"body": urlencode({**change, "csrf": b.csrf, **{f"f{i}": "1" for i in range(80)}}).encode()},
         400),
    ]  # fmt: skip
    for name, kwargs, status in cases:
        r = b.post("/settings/set", change, **kwargs)
        check(f"a change with {name}: refused ({status})", r.status == status, r)
    check("... and none of them changed anything", snapshot(DB) == before and stored("memory.hub_scope") is None, "")
    r = b.post("/settings/set", change, headers={"Sec-Fetch-Site": "same-origin"})
    check("the same change from the panel's own page: done", r.status == 303 and stored("memory.hub_scope") == (
        "true", hub_web.BY), r)  # fmt: skip
    b.get("/settings")
    b.post("/settings/reset", {"key": "memory.hub_scope", "confirm": "yes"})
    check("... and reset", stored("memory.hub_scope") is None, stored("memory.hub_scope"))
    r = b.post("/nothing/here", {"confirm": "yes"})
    check("a POST to no page: 404", r.status == 404, r)
    r = b.post("/memory", {"q": "x"})
    check("a POST to a page that only shows: 404", r.status == 404, r)


def test_get_changes_nothing(panel: hub_web.Panel, b: Browser) -> None:
    b.get("/")
    before = snapshot(DB)
    token = b.csrf
    for path, fields in (("/settings/set", {"key": "memory.hub_scope", "value": "on"}),
                         ("/settings/reset", {"key": "memory.enabled"}), ("/memory/delete", {"id": str(IDS["m2"])}),
                         ("/decisions/withdraw", {"id": str(IDS["d0"])}), ("/decisions/delete", {"id": str(IDS["d0"])}),
                         ("/tasks/act", {"what": "delete", "item": "1"}), ("/scores/act", {"what": "reset_all"}),
                         ("/logout", {})):  # fmt: skip
        r = b.get(path + "?" + urlencode({**fields, "confirm": "yes", "csrf": token}))
        check(f"GET {path}: not allowed, nothing done", r.status == 405 and "POST" in r.header("Allow"), r)
    for method in ("PUT", "DELETE", "PATCH", "HEAD", "OPTIONS", "TRACE"):
        r = b.raw(method, "/settings/set", urlencode({"key": "memory.hub_scope", "value": "on", "confirm": "yes",
                                                      "csrf": token}).encode(),
                  {**b.cookie_header(), "Origin": f"http://{b.host}", "Content-Type":
                   "application/x-www-form-urlencoded"})  # fmt: skip
        check(f"{method}: not allowed", r.status == 405, r)
    check("... nothing changed in the database", snapshot(DB) == before, "")
    r = b.get("/")
    check("... and GET /logout did not sign out", r.status == 200, r)
    r = b.raw("OPTIONS", "/settings/set", headers={"Origin": "http://evil.example", "Access-Control-Request-Method":
                                                   "POST", **b.cookie_header()})  # fmt: skip
    check("a CORS preflight: refused, no CORS headers", r.status == 405 and not any(
        k.lower().startswith("access-control") for k, _ in r.headers), r.headers)  # fmt: skip


def test_read_only(panel: hub_web.Panel, b: Browser) -> None:
    before = snapshot(DB)
    made: list[Any] = []
    original = agent_hub.Store.__init__

    def counting(self: Any, path: Path) -> None:
        made.append(path)
        original(self, path)

    agent_hub.Store.__init__ = counting  # type: ignore[method-assign]
    try:
        for path in ("/", "/settings", "/memory", "/memory?q=cache", "/memory?q=%25", "/decisions",
                     "/decisions?status=active", "/decisions?status=withdrawn&q=DEC", "/decisions?status=nonsense",
                     "/tasks", "/scores", "/tasks?status=open&q=x", "/memory?q=" + "x" * 3000):  # fmt: skip
            r = b.get(path)
            check(f"view {path[:40]}: shown", r.status == 200, r)
        check("views never open a writable connection (no Store)", not made, made)
        b.get("/settings")
        b.post("/settings/set", {"key": "context.max_items", "value": "6"})  # asks first: still only reads
        check("... nor does a confirmation page", not made and stored("context.max_items") is None, made)
    finally:
        agent_hub.Store.__init__ = original  # type: ignore[method-assign]
    check("views changed nothing in the database", snapshot(DB) == before, "")
    with hub_web.View(DB) as view:
        try:
            view.con.execute("CREATE TABLE intruder (x)")
            wrote = True
        except sqlite3.Error:
            wrote = False
    check("the views' connection cannot write", not wrote, "")
    r = b.get("/memory?q=" + "%25")
    check("a search for % matches it literally, not everything", r.status == 200 and "MEM-TWO" not in r.text, r)
    r = b.get("/memory?q=cache")
    check("memory search finds the whole text", "MEM-TWO" in r.text and "MEM-ONE" not in r.text, r)
    r = b.get("/decisions?status=superseded")
    check("decisions by status", "DEC-ZERO" in r.text and "DEC-TWO" not in r.text, r)


def test_headers(panel: hub_web.Panel, b: Browser) -> None:
    r = b.get("/settings")
    csp = r.header("Content-Security-Policy")
    style = re.search(r"<style>(.*?)</style>", r.text, re.S)
    digest = base64.b64encode(hashlib.sha256(style.group(1).encode()).digest()).decode() if style else ""
    check("CSP: nothing may load or run, no framing", "default-src 'none'" in csp and "frame-ancestors 'none'" in csp
          and "unsafe" not in csp and "script-src" not in csp and "form-action 'self'" in csp, csp)  # fmt: skip
    check("CSP: the stylesheet is allowed by its hash", style is not None and f"'sha256-{digest}'" in csp, csp)
    check("no script anywhere", "<script" not in r.text.lower(), "")
    for name, value in (("X-Frame-Options", "DENY"), ("X-Content-Type-Options", "nosniff"), ("Cache-Control",
                        "no-store"), ("Referrer-Policy", "same-origin"), ("Cross-Origin-Resource-Policy",
                        "same-origin")):  # fmt: skip
        check(f"header {name}: {value}", r.header(name) == value, r.header(name))
    check("the server does not name its Python", "python" not in r.header("Server").lower(), r.header("Server"))
    every = [x for x in RESPONSES]
    check(f"no CORS header in any of the {len(every)} answers so far", not any(
        k.lower().startswith("access-control") for x in every for k, _ in x.headers), "")  # fmt: skip
    check("every answer so far carries the CSP and no-store, refusals too", all(
        x.header("Content-Security-Policy") == hub_web.CSP and x.header("Cache-Control") == "no-store"
        for x in every), [x for x in every if not x.header("Content-Security-Policy")][:2])  # fmt: skip
    check("every page so far is inert (no script, handler, image or inline style)", all(
        inert(x.text)[0] for x in every), next((inert(x.text)[1] for x in every if not inert(x.text)[0]), ""))  # fmt: skip


# --------------------------------------------------------------------------
# What agents wrote is shown literally
# --------------------------------------------------------------------------


def test_escaping(panel: hub_web.Panel, b: Browser) -> None:
    store = agent_hub.Store(DB)
    try:  # values a hand-edited (or another version's) settings table may hold
        store.db.execute("INSERT INTO settings (key, value, updated_at, updated_by) VALUES ('context.max_items', '7', ?,"
                         " ?)", (time.time(), '"><script>alert(4)</script>'))  # fmt: skip
        store.db.execute("INSERT INTO settings (key, value, updated_at, updated_by) VALUES ('decisions.enabled', ?, ?,"
                         " 'hand')", (json.dumps("maybe " + XSS), time.time()))  # fmt: skip
    finally:
        store.db.close()
    memory = b.get("/memory").text
    check("memory: the note's text, author, tags and project are escaped", all(s in memory for s in (
        XSS_SEEN, ATTR_SEEN, QUOTE_SEEN, MARK_SEEN)), memory[-3000:])  # fmt: skip
    check("memory: no markup of the agents' gets through", XSS not in memory and ATTR not in memory and MARK not in
          memory and "\x1b" not in memory and "‮" not in memory, "")  # fmt: skip
    check("memory: control and direction characters are shown as ?", "RED?RTL" in memory, "")
    check("memory: the page is inert", *inert(memory))
    check("memory: where each note stands for the other group", f"gate request #{IDS['g_m1']} pending" in memory
          and "no gate held it when written" in memory and "every project (hub scope)" in memory, "")  # fmt: skip
    decisions = b.get("/decisions").text
    check("decisions: title, decision, reasons, alternatives, author and stances escaped", all(s in decisions for s in (
        XSS_SEEN, ATTR_SEEN, QUOTE_SEEN, MARK_SEEN, "STANCE-NOTE " + XSS_SEEN)), decisions[-3000:])  # fmt: skip
    check("decisions: no markup of the agents' gets through", XSS not in decisions and ATTR not in decisions
          and MARK not in decisions, "")  # fmt: skip
    check("decisions: the page is inert", *inert(decisions))
    check("decisions: supersede links and gate state", f"supersedes decision #{IDS['d0']}" in decisions
          and f"superseded by decision #{IDS['d2']}" in decisions and f"gate request #{IDS['g_d2']} pending"
          in decisions, "")  # fmt: skip
    settings = b.get("/settings").text
    check("settings: a hand-written updated_by is escaped", "&quot;&gt;&lt;script&gt;alert(4)&lt;/script&gt;" in
          settings and "<script>alert(4)" not in settings, "")  # fmt: skip
    check("settings: an unusable value is shown as a problem, escaped", "cannot be used" in settings and
          "maybe &lt;script&gt;" in settings and XSS not in settings, "")  # fmt: skip
    check("settings: the page is inert", *inert(settings))
    check("settings: a row that cannot be used can be reset", re.search(r'action="/settings/reset" class="inline">'
          r'<input type="hidden" name="csrf" value="[^"]+"><input type="hidden" name="key" value="decisions\.enabled">',
          settings), "")  # fmt: skip
    overview = b.get("/").text
    check("overview: counts, the gate and the hint", "1</b>of 2 agent session(s) online" in overview
          and "3</b>request(s) waiting at the gate" in overview and "agent_hub.py gate" in overview
          and "3 notes" in overview, overview[-2500:])  # fmt: skip
    check("overview: the page is inert, no agent name in it", inert(overview)[0] and "alpha" not in overview, "")
    reflected = b.get("/memory?q=" + "%22%3E%3Cscript%3Ealert(9)%3C%2Fscript%3E").text
    check("a search's text is escaped where it is shown again", 'value="&quot;&gt;&lt;script&gt;alert(9)' in
          reflected and "<script>alert(9)" not in reflected, "")  # fmt: skip
    b.post("/settings/set", {"key": XSS, "value": ATTR, "confirm": "yes"})
    page = b.get("/settings").text
    check("an error message about your own input is escaped", "There is no setting" in flashes(page) and XSS_SEEN in
          flashes(page) and XSS not in page, flashes(page))  # fmt: skip
    b.get("/decisions")
    r = b.post("/decisions/withdraw", {"id": str(IDS["d1"]), "note": ATTR + XSS, "q": QUOTE, "status": "active"})
    check("a confirmation page escapes what it carries", r.status == 200 and ATTR_SEEN in r.text and XSS not in r.text
          and inert(r.text)[0] and one("SELECT status FROM decisions WHERE id=?", (IDS["d1"],)) == "active", r)  # fmt: skip
    store = agent_hub.Store(DB)
    try:
        store.setting_reset("context.max_items")
        store.setting_reset("decisions.enabled")
    finally:
        store.db.close()


# --------------------------------------------------------------------------
# Settings, notes and decisions
# --------------------------------------------------------------------------


def test_settings(panel: hub_web.Panel, b: Browser) -> None:
    page = b.get("/settings").text
    check("every registered setting is listed", all(f"<code>{k}</code>" in page for k in agent_hub.SETTINGS), page[-800:])
    check("... grouped under its feature's heading", all(f"<h2>{agent_hub.GROUP_NAMES[g]}</h2>" in page for g in
                                                         ("memory", "decisions", "context")), "")  # fmt: skip
    check("... with its value and where it comes from", "Turn on" in page and "default" in page
          and 'name="value" min="100" max="20000"' in page, "")  # fmt: skip
    r = b.post("/settings/set", {"key": "memory.enabled", "value": "on"})
    check("a switch asks first", r.status == 200 and "Turn memory.enabled on?" in r.text and "It is off now (default)"
          in r.text and 'name="confirm" value="yes"' in r.text and stored("memory.enabled") is None, r)  # fmt: skip
    b.csrf = re.search(r'name="csrf" value="([^"]+)"', r.text).group(1)  # type: ignore[union-attr]
    r = b.post("/settings/set", {"key": "memory.enabled", "value": "on", "confirm": "yes"})
    check("... yes turns it on", r.status == 303 and r.header("Location") == "/settings" and stored("memory.enabled")
          == ("true", "control panel"), r)  # fmt: skip
    page = b.get("/settings").text
    check("... the page says so", "memory.enabled is now on for every session" in flashes(page) and
          "set in the control panel" in page and "Turn off" in page, flashes(page))  # fmt: skip
    check("... once (the message is not shown again)", not flashes(b.get("/settings").text), "")
    with sqlite3.connect(DB) as con:
        rows = {r[0]: {"value": r[1], "updated_at": r[2], "updated_by": r[3]} for r in con.execute(
            "SELECT key, value, updated_at, updated_by FROM settings")}  # fmt: skip
    check("... as agent_hub reads it", agent_hub.hive("memory.enabled", rows) is True, rows)
    overview = b.get("/").text
    check("... and the overview shows it", re.search(r"Shared memory</td><td><span class=\"on\">on", overview), "")
    for value, words in (("5", "from 100 to 20000"), ("abc", "from 100 to 20000"), ("", "from 100 to 20000"),
                         ("3000.5", "whole number"), ("99999999999999999999", "from 100 to 20000")):  # fmt: skip
        r = b.post("/settings/set", {"key": "memory.max_chars", "value": value, "confirm": "yes"})
        page = b.get("/settings").text
        check(f"a value that is not allowed ({value[:12]!r}) is refused, with what is allowed", r.status == 303 and
              words in flashes(page) and stored("memory.max_chars") is None, flashes(page))  # fmt: skip
    r = b.post("/settings/set", {"key": "memory.enabled", "value": "maybe", "confirm": "yes"})
    check("a switch takes on or off only", "must be on or off" in flashes(b.get("/settings").text)
          and stored("memory.enabled") == ("true", "control panel"), r)  # fmt: skip
    r = b.post("/settings/set", {"key": "memory.nothing", "value": "1", "confirm": "yes"})
    check("an unknown setting is refused", "There is no setting" in flashes(b.get("/settings").text), r)
    r = b.post("/settings/set", {"key": "memory.max_chars", "value": " 3000 "})
    check("an option asks first, with the value it will set", "Set memory.max_chars to 3000?" in r.text
          and stored("memory.max_chars") is None, r)  # fmt: skip
    b.post("/settings/set", {"key": "memory.max_chars", "value": " 3000 ", "confirm": "yes"})
    check("... yes sets it", stored("memory.max_chars") == ("3000", "control panel"), stored("memory.max_chars"))
    page = b.get("/settings").text
    check("... and Reset is offered for your values only", page.count('action="/settings/reset"') == 2, "")
    r = b.post("/settings/reset", {"key": "memory.max_chars"})
    check("reset asks first", r.status == 200 and "Drop your value for memory.max_chars?" in r.text
          and stored("memory.max_chars") is not None, r)  # fmt: skip
    b.post("/settings/reset", {"key": "memory.max_chars", "confirm": "yes"})
    check("... yes drops it", stored("memory.max_chars") is None and "dropped" in flashes(b.get("/settings").text), "")
    b.post("/settings/reset", {"key": "memory.max_chars", "confirm": "yes"})
    check("... a second time: nothing to drop", "had no value of yours" in flashes(b.get("/settings").text), "")
    b.post("/settings/set", {"key": "memory.enabled", "value": "off", "confirm": "yes"})
    check("a switch turns off again", stored("memory.enabled") == ("false", "control panel"), stored("memory.enabled"))
    b.get("/settings")
    b.post("/settings/reset", {"key": "memory.enabled", "confirm": "yes"})


def test_delete_withdraw(panel: hub_web.Panel, b: Browser) -> None:
    m1 = IDS["m1"]
    b.get("/memory")
    r = b.post("/memory/delete", {"id": str(m1), "q": "MEM"})
    check("deleting a note asks first", r.status == 200 and f"Delete note #{m1}?" in r.text and one(
        "SELECT COUNT(*) FROM memories WHERE id=?", (m1,)) == 1, r)  # fmt: skip
    r = b.post("/memory/delete", {"id": str(m1), "q": "MEM", "confirm": "yes"})
    check("... yes deletes it, back to the same search", r.status == 303 and r.header("Location") == "/memory?q=MEM"
          and one("SELECT COUNT(*) FROM memories WHERE id=?", (m1,)) == 0, r)  # fmt: skip
    check("... its request at the gate expires", one("SELECT status FROM gate WHERE id=?", (IDS["g_m1"],)) == "expired",
          one("SELECT status FROM gate WHERE id=?", (IDS["g_m1"],)))  # fmt: skip
    page = b.get("/memory?q=MEM").text
    check("... the page says so and lists the others", f"Note #{m1} deleted." in flashes(page) and "MEM-ONE" not in
          page and "MEM-TWO" in page, flashes(page))  # fmt: skip
    b.post("/memory/delete", {"id": str(m1), "confirm": "yes"})
    check("deleting it again: already gone", "already gone" in flashes(b.get("/memory").text), "")
    for bad in ("abc", "-1", "1.5", "", "١", "9" * 30):
        b.post("/memory/delete", {"id": bad, "confirm": "yes"})
        check(f"a note id {bad!r} is refused", "No such note" in flashes(b.get("/memory").text), "")
    d1, d2, d0 = IDS["d1"], IDS["d2"], IDS["d0"]
    b.get("/decisions")
    r = b.post("/decisions/withdraw", {"id": str(d1), "note": "REASON <i>stale</i>", "status": "active"})
    check("withdrawing asks first, with your reason", r.status == 200 and f"Withdraw decision #{d1}?" in r.text
          and "REASON &lt;i&gt;stale&lt;/i&gt;" in r.text and one("SELECT status FROM decisions WHERE id=?", (d1,))
          == "active", r)  # fmt: skip
    r = b.post("/decisions/withdraw", {"id": str(d1), "note": "REASON <i>stale</i>", "status": "active",
                                       "confirm": "yes"})  # fmt: skip
    row = one("SELECT status, changed_by, status_note FROM decisions WHERE id=?", (d1,))
    check("... yes withdraws it, by the user, with the reason", r.status == 303 and r.header("Location") ==
          "/decisions?status=active" and row == ("withdrawn", "the user", "REASON <i>stale</i>"), (r, row))  # fmt: skip
    page = b.get("/decisions?status=withdrawn").text
    check("... and the page shows it", f"Decision #{d1} withdrawn." in flashes(page) and
          "withdrawn by the user" in page and "REASON &lt;i&gt;stale&lt;/i&gt;" in page, flashes(page))  # fmt: skip
    check("... a withdrawn decision offers no Withdraw", 'action="/decisions/withdraw"' not in page, "")
    b.post("/decisions/withdraw", {"id": str(d1), "confirm": "yes"})
    check("withdrawing it again: not active", "is not active" in flashes(b.get("/decisions").text), "")
    with sqlite3.connect(DB) as con:  # the replacement's writer's gate held it for the other group
        con.execute("UPDATE decisions SET cross_hold=1 WHERE id=?", (d2,))
    page = b.get("/decisions?status=superseded").text
    check("a replacement held for the other group: superseded for its writer's group only, until approved",
          f"superseded by decision #{d2} (for its writer&#x27;s group; for the other group once you approve it)" in page,
          page[-2500:])  # fmt: skip
    r = b.post("/decisions/delete", {"id": str(d2)})
    check("deleting a decision asks first", r.status == 200 and f"Delete decision #{d2} and its stances?" in r.text
          and one("SELECT COUNT(*) FROM decisions WHERE id=?", (d2,)) == 1, r)  # fmt: skip
    check("... and says what it does to the one it replaced", f"Decision #{d0}, which it replaced, is in force again"
          in r.text, r.text[-1500:])  # fmt: skip
    b.post("/decisions/delete", {"id": str(d2), "confirm": "yes"})
    check("... yes deletes it and its stances", one("SELECT COUNT(*) FROM decisions WHERE id=?", (d2,)) == 0 and
          one("SELECT COUNT(*) FROM decision_stances WHERE decision=?", (d2,)) == 0, "")  # fmt: skip
    check("... the decision it superseded is active again", one("SELECT status FROM decisions WHERE id=?", (d0,)) ==
          "active" and "DEC-ZERO" in b.get("/decisions?status=active").text and "DEC-ZERO" not in b.get(
              "/decisions?status=superseded").text, one("SELECT status FROM decisions WHERE id=?", (d0,)))  # fmt: skip
    check("... its request at the gate expires", one("SELECT status FROM gate WHERE id=?", (IDS["g_d2"],)) == "expired",
          "")  # fmt: skip
    b.post("/decisions/delete", {"id": str(d2), "confirm": "yes"})
    check("deleting it again: already gone", "already gone" in flashes(b.get("/decisions").text), "")


# --------------------------------------------------------------------------
# The task board and the model scores
# --------------------------------------------------------------------------


def fill_board() -> None:
    """Tasks and ratings as agents and the user leave them (agent_hub's Store), with markup and quotes everywhere."""
    store = agent_hub.Store(DB)
    try:
        now = time.time()
        base = {"cross_hold": True, "project": "/w/proj" + QUOTE, "depends": [], "assignee": "", "details": ""}
        IDS["t1"] = store.task_add({**base, "author": "planner" + ATTR, "title": "TASK-ONE " + XSS,
                                    "details": "TASK-DETAILS " + MARK}, 200)  # fmt: skip
        IDS["t2"] = store.task_add({**base, "author": "beta", "cross_hold": False, "title": "TASK-TWO 50%_off",
                                    "depends": [IDS["t1"]], "assignee": "gamma" + XSS}, 200)  # fmt: skip
        IDS["t3"] = store.task_add({**base, "author": "gamma", "title": "TASK-THREE"}, 200)
        # TASK-ONE is claimed by a session that is gone, and has a note of the other group waiting at the gate.
        store.db.execute("UPDATE tasks SET status='in_progress', claimer='inst-gone', claimer_name=?,"
                         " claimer_abliterated=1, claimed_at=?, renewed_at=? WHERE id=?",
                         ("worker" + MARK, now, now, IDS["t1"]))  # fmt: skip
        store.db.execute("INSERT INTO task_notes (task, ts, instance, name, abliterated, cross_hold, what, note) VALUES"
                         " (?, ?, 'inst-x', ?, 1, 1, 'status claimed -> in_progress', ?)",
                         (IDS["t1"], now, "critic" + QUOTE, "TASK-NOTE deepword " + XSS + " " + ATTR))  # fmt: skip
        note = store.db.execute("SELECT MAX(id) FROM task_notes").fetchone()[0]
        IDS["g_tn"] = store.gate_open("task_note", "critic (abliterated), note on task", "every standard agent",
                                      "TASK-NOTE", True, ref=f"task_note:{note}")  # fmt: skip
        IDS["g_t1"] = store.gate_open("task", "planner (standard), task", "every abliterated agent", "Title: TASK-ONE",
                                      False, ref=f"task:{IDS['t1']}")  # fmt: skip
        store.task_set_status(IDS["t3"], "done", "", "the user")
        ratings = [("opus-5.5", "ask", v, f"OPUS-NOTE-{n}") for n, v in enumerate(
            ("useful", "useful", "partly", "wrong", "wrong"))]  # fmt: skip
        ratings += [(XSS, "review", "useful", "XSS-MODEL-NOTE " + ATTR), ("astra", "review", "partly", ""),
                    ("anthropic:claude-x", "ask", "useful", "")]  # fmt: skip
        for n, (model, category, verdict, why) in enumerate(ratings):
            store.rate({"message_id": 9000 + n, "rater": "rater" + QUOTE, "model": model, "category": category,
                        "verdict": verdict, "note": why})  # fmt: skip
    finally:
        store.db.close()


def rows_of(page: str) -> list[str]:
    return re.findall(r"<tr><td class=\"cell\">.*?</tr>", page, re.S)


def row_with(page: str, text: str) -> str:
    return next((r for r in rows_of(page) if text in r), "")


def test_tasks(panel: hub_web.Panel, b: Browser) -> None:
    check("the panel's task statuses are agent_hub's", hub_web.TASK_STATUSES == agent_hub.TASK_STATUSES
          and hub_web.TASK_USER_STATUSES == agent_hub.TASK_USER_STATUSES and hub_web.TASK_HELD == agent_hub.TASK_HELD,
          "")  # fmt: skip
    reserved = {"q", "what", "item", "csrf", "confirm"} | {e.filter[0] for e in hub_web.EXTRAS if e.filter}
    check("no action's field shares its name with the page's filter or the form's own fields", not [
        (e.key, a.name, n) for e in hub_web.EXTRAS for a in e.actions for n, _, _ in a.fields if n in reserved], "")  # fmt: skip
    fill_board()
    t1, t2, t3 = IDS["t1"], IDS["t2"], IDS["t3"]
    made: list[Any] = []
    original = agent_hub.Store.__init__

    def counting(self: Any, path: Path) -> None:
        made.append(path)
        original(self, path)

    before = snapshot(DB)
    agent_hub.Store.__init__ = counting  # type: ignore[method-assign]
    try:
        page = b.get("/tasks").text
        for path in ("/tasks?status=unfinished&q=task", "/tasks?status=done", "/scores?category=review&q=a", "/"):
            b.get(path)
    finally:
        agent_hub.Store.__init__ = original  # type: ignore[method-assign]
    check("tasks: the views read only (no Store, nothing changed)", not made and snapshot(DB) == before, made)
    check("tasks: every task, with the feature's state and the claim timeout", all(f"TASK-{n}" in page for n in (
        "ONE", "TWO", "THREE")) and "Task board is" in page and "tasks.claim_minutes (60)" in page, page[-3000:])  # fmt: skip
    one_row, two_row, three_row = row_with(page, "TASK-ONE"), row_with(page, "TASK-TWO"), row_with(page, "TASK-THREE")
    check("tasks: title, details, author, project, assignee, claimer, note and names escaped", all(s in page for s in (
        XSS_SEEN, ATTR_SEEN, QUOTE_SEEN, MARK_SEEN, "TASK-DETAILS " + MARK_SEEN, "gamma" + XSS_SEEN,
        "worker" + MARK_SEEN, "critic" + QUOTE_SEEN, "planner" + ATTR_SEEN)), page[-4000:])  # fmt: skip
    check("tasks: no markup of the agents' gets through", XSS not in page and ATTR not in page and MARK not in page, "")
    check("tasks: the page is inert", *inert(page))
    check("tasks: a claim whose session is gone is marked; the other group's state", "its session is gone" in one_row
          and "(abliterated)" in one_row and f"gate request #{IDS['g_t1']} pending" in one_row, one_row)  # fmt: skip
    check("tasks: the history, with the gate state of a note", "TASK-NOTE deepword" in one_row
          and "status claimed -&gt; in_progress" in one_row and f"other group: gate request #{IDS['g_tn']} pending"
          in one_row, one_row)  # fmt: skip
    check("tasks: what a task waits for, with its status", f"#{t1} (in_progress)" in two_row
          and "no gate held it when written" in two_row, two_row)  # fmt: skip
    check("tasks: the user's change in the history", f'<td class="cell">{t3}</td>' in three_row
          and "the user: status open -&gt; done" in three_row, three_row)  # fmt: skip
    check("tasks: unfinished first, then finished", page.index("TASK-ONE") < page.index("TASK-TWO")
          < page.index("TASK-THREE"), "")  # fmt: skip
    check("tasks: the actions per task; no assigning a finished task", 'value="status"' in three_row
          and 'value="assign"' not in three_row and 'value="assign"' in one_row and 'value="delete"' in three_row, "")
    picks = re.search(r'<select name="to" aria-label="new status">(.*?)</select>', page, re.S)
    check("tasks: the user sets open, blocked, done or dropped only", picks and re.findall(r'value="(\w+)"',
          picks.group(1)) == list(agent_hub.TASK_USER_STATUSES), picks.group(1) if picks else page[-2000:])  # fmt: skip

    page = b.get("/tasks?status=unfinished").text
    check("tasks: the unfinished ones", "TASK-ONE" in page and "TASK-TWO" in page and "TASK-THREE" not in page, "")
    page = b.get("/tasks?status=done").text
    check("tasks: one status", "TASK-THREE" in page and "TASK-ONE" not in page and "TASK-TWO" not in page, "")
    page = b.get("/tasks?status=nonsense").text
    check("tasks: an unknown status filters nothing", all(f"TASK-{n}" in page for n in ("ONE", "TWO", "THREE")), "")
    page = b.get("/tasks?q=DEEPWORD").text
    check("tasks: the search looks in the notes too, in any case", "TASK-ONE" in page and "TASK-TWO" not in page, "")
    page = b.get("/tasks?q=" + "50%25_").text
    check("tasks: % and _ are searched for literally", "TASK-TWO" in page and "TASK-ONE" not in page
          and "TASK-THREE" not in page, "")  # fmt: skip
    page = b.get("/tasks?q=" + "%22%3E%3Cscript%3Ealert(9)").text
    check("tasks: the search's text is escaped where it is shown again", 'value="&quot;&gt;&lt;script&gt;alert(9)'
          in page and "<script>alert(9)" not in page, "")  # fmt: skip

    b.get("/tasks?status=unfinished")
    form = {"what": "status", "item": str(t2), "to": "blocked", "note": "USER-NOTE " + XSS, "status": "unfinished"}
    r = b.post("/tasks/act", form)
    check("set status: asks first, with your note, escaped", r.status == 200 and f"Set the status of task #{t2} to"
          " blocked?" in r.text and "USER-NOTE " + XSS_SEEN in r.text and XSS not in r.text and inert(r.text)[0]
          and one("SELECT status FROM tasks WHERE id=?", (t2,)) == "open", r)  # fmt: skip
    r = b.post("/tasks/act", {**form, "confirm": "yes"})
    check("... yes sets it, back to the same filter", r.status == 303 and r.header("Location") ==
          "/tasks?status=unfinished" and one("SELECT status FROM tasks WHERE id=?", (t2,)) == "blocked", r)  # fmt: skip
    check("... with your note in its history, by the user, read by every agent", one(
        "SELECT name, what, note, cleared FROM task_notes WHERE task=? ORDER BY id DESC LIMIT 1", (t2,)) == (
        "the user", "status open -> blocked", "USER-NOTE " + XSS, 1), "")  # fmt: skip
    page = b.get("/tasks?status=unfinished").text
    check("... and says so", f"Task #{t2} is blocked now." in flashes(page) and "USER-NOTE " + XSS_SEEN in page, "")
    b.post("/tasks/act", {"what": "status", "item": str(t2), "to": "blocked", "confirm": "yes"})
    check("the same status without a note: nothing changed", "already blocked: nothing changed" in flashes(
        b.get("/tasks").text), "")  # fmt: skip
    for value in ("in_progress", "claimed", "", "Done"):
        b.post("/tasks/act", {"what": "status", "item": str(t2), "to": value, "confirm": "yes"})
        check(f"a status the user does not set ({value!r}) is refused", "new status must be one of open, blocked,"
              " done, dropped" in flashes(b.get("/tasks").text) and one("SELECT status FROM tasks WHERE id=?", (t2,))
              == "blocked", "")  # fmt: skip
    for bad in ("x1", "-1", "1.5", "١", "9" * 30):
        r = b.post("/tasks/act", {"what": "delete", "item": bad})
        check(f"a task id {bad!r} is refused before anything is asked", r.status == 303 and "no such task" in flashes(
            b.get("/tasks").text), r)  # fmt: skip
    b.post("/tasks/act", {"what": "delete", "item": "", "confirm": "yes"})
    check("a task action needs its task", "which task?" in flashes(b.get("/tasks").text), "")
    b.post("/tasks/act", {"what": "status", "item": "999999", "to": "done", "confirm": "yes"})
    check("a task that is not there: nothing changed", "is gone, or already done" in flashes(b.get("/tasks").text), "")

    r = b.post("/tasks/act", {"what": "assign", "item": str(t2), "assignee": "Not A Name" + XSS})
    check("assign: a name no agent could use is refused before anything is asked", r.status == 303
          and "an agent name is 2-32 characters" in flashes(b.get("/tasks").text), r)  # fmt: skip
    r = b.post("/tasks/act", {"what": "assign", "item": str(t2), "assignee": " Delta "})
    check("assign: asks first", r.status == 200 and f"Make task #{t2} meant for delta?" in r.text, r)
    b.post("/tasks/act", {"what": "assign", "item": str(t2), "assignee": " Delta ", "confirm": "yes"})
    check("... yes assigns it, and its history says so", one("SELECT assignee FROM tasks WHERE id=?", (t2,)) == "delta"
          and f"Task #{t2} is meant for delta now." in flashes(b.get("/tasks").text) and one(
          "SELECT what FROM task_notes WHERE task=? ORDER BY id DESC LIMIT 1", (t2,)) == "meant for delta", "")  # fmt: skip
    r = b.post("/tasks/act", {"what": "assign", "item": str(t2), "assignee": ""})
    check("... empty: anyone", f"Make task #{t2} meant for anyone?" in r.text, r)
    b.post("/tasks/act", {"what": "assign", "item": str(t2), "assignee": "", "confirm": "yes"})
    check("... done", one("SELECT assignee FROM tasks WHERE id=?", (t2,)) == "", "")

    r = b.post("/tasks/act", {"what": "delete", "item": str(t1), "q": "TASK"})
    check("delete: asks first", r.status == 200 and f"Delete task #{t1} and its history for good?" in r.text
          and "no longer do" in r.text and one("SELECT COUNT(*) FROM tasks WHERE id=?", (t1,)) == 1, r)  # fmt: skip
    r = b.post("/tasks/act", {"what": "delete", "item": str(t1), "q": "TASK", "confirm": "yes"})
    check("... yes deletes it and its notes, back to the same search", r.status == 303 and r.header("Location") ==
          "/tasks?q=TASK" and one("SELECT COUNT(*) FROM tasks WHERE id=?", (t1,)) == 0 and one(
          "SELECT COUNT(*) FROM task_notes WHERE task=?", (t1,)) == 0, r)  # fmt: skip
    check("... a task that waited for it no longer does", one("SELECT depends FROM tasks WHERE id=?", (t2,)) == "", "")
    check("... its requests at the gate expire", one("SELECT status FROM gate WHERE id=?", (IDS["g_t1"],)) == "expired"
          and one("SELECT status FROM gate WHERE id=?", (IDS["g_tn"],)) == "expired", "")  # fmt: skip
    page = b.get("/tasks?q=TASK").text
    check("... the page says so", f"Task #{t1} deleted." in flashes(page) and "TASK-ONE" not in page, flashes(page))
    b.post("/tasks/act", {"what": "delete", "item": str(t1), "confirm": "yes"})
    check("deleting it again: already gone", f"Task #{t1} was already gone." in flashes(b.get("/tasks").text), "")

    hidden = agent_hub.Store.task_assign
    del agent_hub.Store.task_assign
    try:
        page = b.get("/tasks").text
        check("an action whose Store method is missing is not offered", 'value="assign"' not in page
              and 'value="delete"' in page, "")  # fmt: skip
        b.post("/tasks/act", {"what": "assign", "item": str(t2), "assignee": "delta", "confirm": "yes"})
        check("... and refused", "cannot be done here" in flashes(b.get("/tasks").text)
              and one("SELECT assignee FROM tasks WHERE id=?", (t2,)) == "", "")  # fmt: skip
    finally:
        agent_hub.Store.task_assign = hidden  # type: ignore[method-assign]
    overview = b.get("/").text
    check("the overview: the task board and the track record, with what they hold", "Task board</td>" in overview
          and "2 tasks" in overview and "Model track record</td>" in overview and "8 ratings" in overview
          and "not installed" not in overview, overview[-1500:])  # fmt: skip


def test_scores(panel: hub_web.Panel, b: Browser) -> None:
    check("the panel's kinds of request are agent_hub's", hub_web.RATING_CATEGORIES == agent_hub.RATING_CATEGORIES, "")
    page = b.get("/scores").text
    check("scores: the feature's state and auto-route's", "Model track record is" in page and "Auto-route"
          " (track.auto_route) is off" in page and "at least 5 rating(s)" in page, page[-3000:])  # fmt: skip
    check("scores: per kind and model, with the counts", re.search(
        r"<td class=\"cell\">ask</td><td class=\"cell\">opus-5\.5</td><td class=\"cell\">0\.50</td>"
        r"<td class=\"cell\">2</td><td class=\"cell\">1</td><td class=\"cell\">2</td><td class=\"cell\">5</td>",
        page), page[-3000:])  # fmt: skip
    check("scores: the best first within a kind", page.index("anthropic:claude-x") < page.index("opus-5.5"), "")
    opus = row_with(page, "opus-5.5")
    check("scores: the raters' latest notes, for the user", all(f"OPUS-NOTE-{n}" in opus for n in (2, 3, 4))
          and "OPUS-NOTE-1" not in opus and "rater" + QUOTE_SEEN in opus, opus)  # fmt: skip
    check("scores: a model's name, a rater and a note escaped", XSS_SEEN in page and "XSS-MODEL-NOTE " + ATTR_SEEN in
          page and XSS not in page and ATTR not in page, "")  # fmt: skip
    check("scores: the page is inert", *inert(page))
    page = b.get("/scores?category=review").text
    check("scores: one kind of request", "astra" in page and "opus-5.5" not in page, "")
    page = b.get("/scores?q=ASTRA").text
    check("scores: search a model", "astra" in page and "opus-5.5" not in page, "")
    r = b.post("/scores/act", {"what": "reset", "item": "ask:opus-5.5", "category": "ask"})
    check("reset: asks first, naming the model and the kind", r.status == 200 and "Reset the ratings of opus-5.5 for"
          " ask?" in r.text and one("SELECT COUNT(*) FROM ratings WHERE model='opus-5.5'") == 5, r)  # fmt: skip
    r = b.post("/scores/act", {"what": "reset", "item": "ask:opus-5.5", "category": "ask", "confirm": "yes"})
    check("... yes resets that model for that kind only", r.status == 303 and r.header("Location") ==
          "/scores?category=ask" and one("SELECT COUNT(*) FROM ratings WHERE model='opus-5.5'") == 0
          and one("SELECT COUNT(*) FROM ratings") == 3, r)  # fmt: skip
    check("... and says so", "5 rating(s) of opus-5.5 for ask deleted." in flashes(b.get("/scores").text), "")
    b.post("/scores/act", {"what": "reset", "item": "ask:anthropic:claude-x", "confirm": "yes"})
    check("... a model whose name holds a colon", one("SELECT COUNT(*) FROM ratings WHERE model=?",
                                                      ("anthropic:claude-x",)) == 0, "")  # fmt: skip
    for bad in ("opus-5.5", ":opus-5.5", "ask:"):
        b.post("/scores/act", {"what": "reset", "item": bad, "confirm": "yes"})
        check(f"a row {bad!r} is refused", "no such model and kind of request" in flashes(b.get("/scores").text), "")
    b.post("/scores/act", {"what": "reset", "confirm": "yes"})
    check("a per-row action needs its row", "which rating" in flashes(b.get("/scores").text), "")
    b.post("/scores/act", {"what": "nonsense", "confirm": "yes"})
    check("an unknown action is refused", "cannot be done here" in flashes(b.get("/scores").text), "")
    check("nothing else was reset", one("SELECT COUNT(*) FROM ratings") == 2, "")
    r = b.post("/scores/act", {"what": "reset_all"})
    check("reset all: asks first", r.status == 200 and "Reset every model's ratings?" in r.text.replace("&#x27;", "'")
          and one("SELECT COUNT(*) FROM ratings") == 2, r)  # fmt: skip
    b.post("/scores/act", {"what": "reset_all", "confirm": "yes"})
    page = b.get("/scores").text
    check("... yes resets them all", one("SELECT COUNT(*) FROM ratings") == 0 and "2 rating(s) deleted" in flashes(page)
          and "Reset all ratings" not in page, flashes(page))  # fmt: skip


def test_extras_adapter(panel: hub_web.Panel, b: Browser) -> None:
    """The pages next to an older agent_hub.py (no task board or track record), and settings registered later."""
    features = dict(agent_hub.FEATURES)
    methods = {name: getattr(agent_hub.Store, name) for e in hub_web.EXTRAS for name in e.methods}
    for e in hub_web.EXTRAS:
        for f in e.features:
            agent_hub.FEATURES.pop(f, None)
    for name in methods:
        delattr(agent_hub.Store, name)
    try:
        tasks, scores = b.get("/tasks").text, b.get("/scores").text
        check("an older agent_hub: tasks are available once the task board is installed", "available once the task"
              " board is installed" in tasks and 'action="/tasks/act"' not in tasks, tasks[-600:])  # fmt: skip
        check("... scores once the track record is installed", "available once the track record is installed" in
              scores and 'action="/scores/act"' not in scores, scores[-600:])  # fmt: skip
        check("... the overview says they are not installed", b.get("/").text.count("not installed") == 2, "")
        before = snapshot(DB)
        b.post("/tasks/act", {"what": "delete", "item": str(IDS["t2"]), "confirm": "yes"})
        b.post("/scores/act", {"what": "reset_all", "confirm": "yes"})
        check("... and their actions are refused", "cannot be done here" in flashes(b.get("/tasks").text)
              and snapshot(DB) == before, "")  # fmt: skip
    finally:
        agent_hub.FEATURES.clear()
        agent_hub.FEATURES.update(features)
        for name, method in methods.items():
            setattr(agent_hub.Store, name, method)
    check("installed again", "TASK-TWO" in b.get("/tasks").text and "not installed" not in b.get("/").text, "")

    agent_hub.register_setting(agent_hub.Setting("tasks.order", "choice", "newest", "Which tasks come first (test).",
                                                 choices=("newest", "oldest")))  # fmt: skip
    try:
        settings = b.get("/settings").text
        check("a setting registered later appears in Settings by itself, under its feature",
              "<code>tasks.order</code>" in settings and "<h2>Task board</h2>" in settings, "")  # fmt: skip
        check("... a choice as a list of its values", '<select name="value" aria-label="tasks.order"><option'
              ' value="newest" selected>newest</option><option value="oldest">oldest</option></select>' in settings,
              settings[-2500:])  # fmt: skip
        r = b.post("/settings/set", {"key": "tasks.order", "value": "Oldest"})
        check("... asks first", "Set tasks.order to oldest?" in r.text and stored("tasks.order") is None, r)
        b.post("/settings/set", {"key": "tasks.order", "value": "Oldest", "confirm": "yes"})
        check("... and stores it as agent_hub parses it", stored("tasks.order") == ('"oldest"', hub_web.BY),
              stored("tasks.order"))  # fmt: skip
        b.post("/settings/set", {"key": "tasks.order", "value": "random", "confirm": "yes"})
        check("... a value not among its choices is refused", "must be one of newest, oldest" in flashes(
            b.get("/settings").text) and stored("tasks.order") == ('"oldest"', hub_web.BY), "")  # fmt: skip
        b.post("/settings/reset", {"key": "tasks.order", "confirm": "yes"})
        check("... and reset", stored("tasks.order") is None, "")
    finally:
        agent_hub.SETTINGS.pop("tasks.order", None)
    settings = b.get("/settings").text
    check("the hive's own settings: the task board and the track record", all(f"<code>{k}</code>" in settings for k in (
        "tasks.enabled", "tasks.claim_minutes", "tasks.max_open", "track.enabled", "track.auto_route",
        "track.min_ratings")) and "<h2>Model track record</h2>" in settings, "")  # fmt: skip
    b.post("/settings/set", {"key": "track.auto_route", "value": "on", "confirm": "yes"})
    b.post("/settings/set", {"key": "track.min_ratings", "value": "3", "confirm": "yes"})
    with sqlite3.connect(DB) as con:
        rows = {r[0]: {"value": r[1], "updated_at": r[2], "updated_by": r[3]} for r in con.execute(
            "SELECT key, value, updated_at, updated_by FROM settings")}  # fmt: skip
    check("... set here, read by agent_hub", agent_hub.hive("track.auto_route", rows) is True
          and agent_hub.hive("track.min_ratings", rows) == 3 and stored("track.min_ratings") == ("3", hub_web.BY),
          rows)  # fmt: skip
    page = b.get("/scores").text
    check("... and the scores page says so", "Auto-route (track.auto_route) is on" in page and "at least 3 rating(s)"
          in page, "")  # fmt: skip
    for key in ("track.auto_route", "track.min_ratings"):
        b.post("/settings/reset", {"key": key, "confirm": "yes"})


def test_sign_out(panel: hub_web.Panel, b: Browser) -> None:
    b.get("/")
    r = b.post("/logout", {})
    check("sign out: ends the session and clears the cookie", r.status == 200 and "Signed out" in r.text and
          "Max-Age=0" in r.header("Set-Cookie"), r)  # fmt: skip
    r = b.get("/")
    check("... the old cookie no longer works", r.status == 403, r)
    r = b.raw("GET", "/login?token=" + (panel.token or "nothing"))
    check("... and the used link does not sign in again", r.status == 403, r)


def test_real_browser() -> None:
    """The whole flow in a real browser, when Playwright and Chrome or Chromium are here (else skipped): the link signs
    in, the stylesheet passes the CSP, and changes go through (the Origin a browser really sends must pass)."""
    try:
        from playwright.sync_api import sync_playwright  # noqa: PLC0415
    except ImportError:
        print("  skipped: Playwright is not installed")
        return
    exe = next((shutil.which(n) for n in ("google-chrome", "chromium", "chromium-browser", "chrome") if shutil.which(n)),
               None)  # fmt: skip
    db = TMP / "browser" / "hub.db"
    store = agent_hub.Store(db)
    try:
        note = store.memory_add({"author": "alpha", "cross_hold": False, "project": "/w", "scope": "project", "tags": "",
                                 "text": "BROWSER-NOTE " + XSS}, 1000, 0)  # fmt: skip
        task = store.task_add({"author": "alpha", "cross_hold": False, "project": "/w", "title": "BROWSER-TASK " + XSS,
                               "details": "", "depends": [], "assignee": ""}, 200)  # fmt: skip
    finally:
        store.db.close()
    panel = hub_web.Panel(db, 0)
    panel.serve_in_thread()
    try:
        with sync_playwright() as p:
            try:
                browser = p.chromium.launch(executable_path=exe, args=["--no-sandbox"]) if exe else p.chromium.launch()
            except Exception as exc:
                print(f"  skipped: no browser to run ({str(exc).splitlines()[0][:120]})")
                return
            try:
                page = browser.new_page()
                page.set_default_timeout(15_000)
                problems: list[str] = []
                page.on("console", lambda m: problems.append(m.text) if m.type in ("error", "warning") else None)
                page.on("dialog", lambda d: problems.append("a dialog opened: " + d.message))
                page.goto(panel.login_url)
                check("browser: the link signs in", page.url == f"http://127.0.0.1:{panel.port}/" and
                      page.title().startswith("Overview"), page.url)  # fmt: skip
                background = page.evaluate("getComputedStyle(document.body).backgroundColor")
                check("browser: the stylesheet applies under the CSP", background not in ("rgba(0, 0, 0, 0)",
                      "transparent"), background)  # fmt: skip
                check("browser: the cookie is out of reach of scripts", page.evaluate("document.cookie") == "", "")
                page.goto(f"http://127.0.0.1:{panel.port}/settings")
                page.click('form:has(input[name="key"][value="memory.enabled"]) button')
                check("browser: a change asks first", page.locator("h1").inner_text() == "Turn memory.enabled on?", "")
                page.click("button.primary")
                check("browser: ... and goes through (Origin, CSRF token, cookie)", stored("memory.enabled", db) == (
                    "true", hub_web.BY) and "is now on" in page.locator(".flash").inner_text(), page.url)  # fmt: skip
                page.goto(f"http://127.0.0.1:{panel.port}/memory")
                check("browser: a note's markup stays text", XSS in page.locator(".text").first.inner_text(), "")
                page.click("button.danger")
                page.click("button.danger")
                check("browser: a note is deleted after the confirmation", one(
                    "SELECT COUNT(*) FROM memories WHERE id=?", (note,), db) == 0, "")  # fmt: skip
                page.goto(f"http://127.0.0.1:{panel.port}/tasks?status=unfinished")
                check("browser: a task's markup stays text", XSS in page.locator("td.cell b").first.inner_text(), "")
                page.select_option('select[name="to"]', "done")
                page.fill('input[name="note"]', "BROWSER-USER-NOTE")
                page.click('form:has(input[name="what"][value="status"]) button')
                check("browser: a task's new status asks first", page.locator("h1").inner_text() ==
                      f"Set the status of task #{task} to done?", page.locator("h1").inner_text())  # fmt: skip
                page.click("button.primary")
                check("browser: ... and is set, with the note, back to the same filter", one(
                    "SELECT status FROM tasks WHERE id=?", (task,), db) == "done" and one(
                    "SELECT note FROM task_notes WHERE task=? ORDER BY id DESC LIMIT 1", (task,), db) ==
                    "BROWSER-USER-NOTE" and page.url.endswith("/tasks?status=unfinished")
                    and "is done now" in page.locator(".flash").inner_text(), page.url)  # fmt: skip
                page.click('nav button')
                check("browser: sign out", page.locator("h1").inner_text() == "Signed out" and page.context.cookies()
                      == [], page.context.cookies())  # fmt: skip
                check("browser: no error, CSP violation or dialog", not problems, problems)
            finally:
                browser.close()
    finally:
        panel.stop()


def main() -> None:
    print("== command line")
    try:
        test_cli()
    except Exception:
        FAILED.append("command line: crashed")
        traceback.print_exc()
    print("== no database yet")
    try:
        test_no_database()
    except Exception:
        FAILED.append("no database: crashed")
        traceback.print_exc()
    fill_database()
    panel = hub_web.Panel(DB, 0)
    panel.serve_in_thread()
    try:
        print("== sign-in")
        b = test_sign_in(panel)
        for name, test in [("without the cookie", test_no_cookie), ("Host header", test_host),
                           ("changes: CSRF, Origin, fetch metadata", test_post_guards),
                           ("GET changes nothing", test_get_changes_nothing), ("read-only views", test_read_only),
                           ("escaping", test_escaping), ("settings", test_settings),
                           ("delete and withdraw", test_delete_withdraw), ("task board", test_tasks),
                           ("model scores", test_scores), ("an older agent_hub; settings registered later",
                                                           test_extras_adapter),
                           ("headers", test_headers), ("sign out", test_sign_out)]:  # fmt: skip
            print(f"== {name}")
            try:
                test(panel, b)
            except Exception:
                FAILED.append(f"{name}: crashed")
                traceback.print_exc()
    except Exception:
        FAILED.append("sign-in: crashed")
        traceback.print_exc()
    finally:
        panel.stop()
    print("== a real browser")
    try:
        test_real_browser()
    except Exception:
        FAILED.append("a real browser: crashed")
        traceback.print_exc()
    shutil.rmtree(TMP, ignore_errors=True)
    print(f"\n{len(PASSED)} passed, {len(FAILED)} failed")
    for name in FAILED:
        print(f"  FAILED: {name}")
    sys.exit(1 if FAILED else 0)


if __name__ == "__main__":
    main()
