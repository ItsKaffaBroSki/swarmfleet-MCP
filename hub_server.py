#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "starlette>=0.40",
#     "uvicorn>=0.30",
#     "cryptography>=42",
# ]
# ///
"""Agent Hub fleet server: a hub on a VPS or website that people's agents join, report in to and take orders from a
director (fleet mode, Agent Hub 2.0).

    uv run hub_server.py owner init --name "My fleet" --public-url https://hub.example.com
    uv run hub_server.py serve --behind-proxy          (127.0.0.1:8780, behind Caddy or another TLS proxy)
    uv run hub_server.py owner members | director <member> | settings ... | invite ... | panel-login | audit ...

Agents connect with agent_hub.py's fleet tools (fleet_join, fleet_wait, ...). Every member has an Ed25519 key made
on its own computer; it logs in by signing a one-time challenge and gets a short-lived access token. Every request is
authorized here, on the server, for the identity that made it: clients are untrusted. Roles: the owner (whoever runs
this server: the `owner` commands on the VPS), one director (picked by the owner), trusted partners, workers, the
lobby (waiting for approval) and banned keys.

The director (and trusted partners) hand out orders: workspace tasks (files the director shares, with hidden
paths and names, which the worker's client downloads into a task folder; the report carries only the diff of that
folder) and repo tasks (a repository URL the worker's agent clones itself; the report carries a patch or a pull-request
URL). Results are untrusted until the director or its human accepts them. Trust marking, handover and dispatch
approval from the command line are signed with the member's key for that one purpose.

Security defaults: listens on 127.0.0.1 only unless it serves TLS itself (--tls-cert/--tls-key) or is explicitly
told it sits behind a TLS reverse proxy; rate and size limits on every endpoint; secrets refused in every text and
file members upload; an audit log of every role change, admission, ban, invite, order, result decision, handover and
setting change; the director's web panel (/panel) signs in with one-time links only, uses Secure, HttpOnly,
SameSite=Strict cookies, CSRF tokens and Origin checks, confirmation pages for destructive actions, a strict
Content-Security-Policy, no JavaScript and HTML-escapes everything members wrote. The server never runs anything a
member sends.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import contextlib
import difflib
import fnmatch
import hashlib
import hmac
import html
import ipaddress
import json
import os
import re
import secrets
import sqlite3
import sys
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable
from urllib.parse import parse_qs, unquote, urlsplit

__version__ = "2.0.0"
# The fleet protocol's version (agent_hub.py checks it). 2: the server has a key of its own that signs its login
# challenges, and every member signature names that key's fingerprint and the exact origin the member connects to.
PROTOCOL = 2

DEFAULT_PORT = 8780
DEFAULT_HOME = "~/.agent-hub-server"
SERVER_ID_RE = re.compile(r"^fs-[0-9a-f]{12}$")
NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{1,31}$")  # a member's name: as agent names in the local hub
RESERVED_NAMES = {"director", "all", "everyone", "owner", "server", "fleet", "me", "you", "lobby", "admin", "hub"}

# Limits (every one of them is enforced here, whatever a client does)
MAX_BODY = 64 * 1024  # bytes of one API request or panel form
MAX_TEXT = 8000  # characters of one message
MAX_DESCRIPTION = 300
MAX_AGENT = 100
MAX_CONTACT = 200
MAX_REASON = 500
MAX_NOTE = 200
MAX_TITLE = 200  # an order's title
MAX_INSTRUCTIONS = 20_000  # an order's instructions
MAX_CRITERIA = 4000  # its acceptance criteria
MAX_WORK_TEXT = 4000  # a progress note, a question
MAX_SUMMARY = 8000  # a report's summary, an answer
MAX_UPLOAD = 24 * 1024 * 1024  # bytes of one request that carries files (an order, a report)
MAX_FILES = 500  # files in one order or report
MAX_PATH = 1000
MAX_URL = 500
EVENTS_PER_POLL = 100
POLL_MAX = 30  # seconds one long-poll may wait (below the usual 60 s idle timeout of proxies)
POLLS_PER_MEMBER = 3  # long-polls one member may hold open at once
KEEP_ALIVE = 5  # seconds an idle connection stays open (clients drop theirs sooner: agent_hub.py's 4)
CHALLENGE_SECONDS = 30  # a challenge is used at once (signed and sent back): short-lived
CHALLENGES_MAX = 5000  # outstanding challenges in all (when full, the address holding the most loses its oldest)
CHALLENGES_PER_ADDRESS = 10  # outstanding challenges of one address (an IPv6 /64): a new one replaces its oldest
CLOCK_SKEW = 300  # seconds a signed timestamp may differ from the server's clock
PANEL_LINK_SECONDS = 300
PANEL_SESSION_SECONDS = 8 * 3600
INVITE_MAX_HOURS = 24 * 30
INVITE_MAX_USES = 1000
EVENT_KEEP_DAYS = 30
JOIN_MODES = ("open", "invite", "closed")
STATES = ("waiting", "working", "idle", "offline")  # what a member's client reports
ORDER_KINDS = ("workspace", "repo")
ORDER_WAITING = ("queued", "offered")  # up for claiming
ORDER_WORKING = ("claimed", "in_progress")
ORDER_OPEN = ("approval", *ORDER_WAITING, *ORDER_WORKING, "reported")
ORDER_FINAL = ("accepted", "rejected", "cancelled", "expired")
ORDER_STATUSES = ORDER_OPEN + ORDER_FINAL
STATUS_NAMES = {"approval": "waiting for dispatch approval", "queued": "queued (for anyone)", "offered": "offered",
                "claimed": "claimed", "in_progress": "in progress", "reported": "reported (waiting for review)",
                "accepted": "accepted", "rejected": "rejected", "cancelled": "cancelled", "expired": "expired"}  # fmt: skip
PRIORITIES = {"low": -1, "normal": 0, "high": 1, "urgent": 2}
PRIORITY_NAMES = {v: k for k, v in PRIORITIES.items()}
HANDOVER_SECONDS = 24 * 3600  # a handover offer stays open this long
SIGNED_PURPOSES = ("trust", "handover", "accept-handover", "approve")  # actions signed with the member's key
START_GRACE = 90  # seconds after a start in which no claim goes back up for silence (clients are reconnecting)
MAX_ID = 2**63 - 1  # ids are SQLite integers

ROLES = ("director", "partner", "worker", "lobby", "banned", "removed")
ACTIVE = ("director", "partner", "worker", "lobby")  # roles that may log in
MEMBERS = ("director", "partner", "worker")  # admitted
LEADS = ("director", "partner")
RANK = {"owner": 4, "director": 3, "partner": 2, "worker": 1, "lobby": 0, "banned": -1, "removed": -1}
ROLE_NAMES = {"director": "director", "partner": "trusted partner", "worker": "worker", "lobby": "in the lobby",
              "banned": "banned", "removed": "not a member"}  # fmt: skip

# Settings, changed live (they apply to the next request): who may change which is in Fleet.change_settings.
SETTING_DEFAULTS: dict[str, Any] = {
    "join_mode": "invite",
    "approval": True,
    "paused": False,
    "claim_minutes": 60,  # a claim without news (claim, download, progress, question) for this long is re-queued
    "requeue_offline_seconds": 300,  # ... and so is one whose claimer is offline for this long
    "dispatch_approval": False,  # orders wait for a human's approval (panel, owner, signed CLI) before dispatch
    "takeover": False,  # the director's human took over: lead actions only through the panel (humans), not tokens
    "handover_needs_owner": False,  # an accepted handover also waits for the owner's confirmation
    "new_keys_per_hour": 5,  # new identities one address may create per hour in open mode
    "lobby_max": 200,
    "members_max": 1000,
    "max_order_bytes": 4_000_000,  # the files of one workspace order
    "max_report_bytes": 4_000_000,  # the files or the patch of one report
    "storage_mb": 2000,  # all order and report files kept on this server
    "orders_open_max": 1000,  # orders not yet finished
}
BOOL_SETTINGS = ("approval", "paused", "dispatch_approval", "takeover", "handover_needs_owner")
SETTING_LIMITS = {"new_keys_per_hour": (0, 10_000), "lobby_max": (0, 100_000), "members_max": (1, 1_000_000),
                  "claim_minutes": (1, 10_080), "requeue_offline_seconds": (5, 86_400),
                  "max_order_bytes": (1000, 100_000_000), "max_report_bytes": (1000, 100_000_000),
                  "storage_mb": (1, 1_000_000), "orders_open_max": (1, 100_000)}  # fmt: skip
LIVE_SETTINGS = ("join_mode", "approval", "paused", "claim_minutes", "requeue_offline_seconds")  # leads' agents too
HUMAN_SETTINGS = ("dispatch_approval", "takeover")  # a lead's human in the panel (takeover: the director's) or owner

# Ed25519 identities: the fingerprint and word phrase every member and the panel show (agent_hub.py has the same
# code; tests/test_fleet.py checks they agree).
WORDS = """
acorn actor agent album alley amber anchor angle apple arch arena arrow aspen atlas autumn avenue badge bagel baker
bamboo banjo barley barn basil beach beacon beam bean beaver bench berry birch bison blade blanket bloom board
border bottle boulder brain branch brass bread breeze brick bridge brook broom bubble bucket buffalo bundle butter
button cabin cable cactus camel camera candle canoe canvas canyon carbon cargo carpet carrot castle cedar chalk
cheese cherry chess circle citrus clam clock cloud clover cocoa comet compass copper coral cotton coyote crane creek
crown crystal cup daisy dancer delta desert diamond dolphin domino donkey dragon drum eagle ember emerald engine
falcon feather ferry fiddle field finch flame flute forest fossil fountain fox galaxy garden garlic gazelle geyser
ginger glacier globe granite grape gravel guitar hammer harbor harvest hazel helmet heron honey hornet husky iceberg
indigo island ivory jacket jaguar jasmine jelly jungle kayak kettle kiwi koala ladder lagoon lantern lemon lily
lizard llama lobster lotus lynx magnet mango maple marble meadow melon meteor mint mirror mitten moose moss muffin
needle nickel oasis ocean olive onion orange orbit orchid otter oyster paddle panda panther paper parrot peach
pebble pelican pepper piano pigeon pillow pine planet plum pocket pony poppy prairie pumpkin puzzle quartz rabbit
raven reef ribbon river robin rocket saddle salmon scarf seal shadow shell shovel silver sketch sled spider spruce
stable statue summit sunset swan thunder tiger timber toast tomato tractor trumpet tulip tundra turtle umbrella
valley velvet violin volcano walnut walrus wagon whale willow window wizard yarn zebra
""".split()
assert len(WORDS) == 256 and len(set(WORDS)) == 256


def key_bytes(key_b64: Any) -> bytes | None:
    """The 32 raw bytes of an Ed25519 public key sent as standard base64, or None. Only the canonical encoding (the
    one base64.b64encode gives): a 32-byte key has 2 unused bits in its last character, so 4 strings decode to the
    same key, and a member is one key, however it is written."""
    if not isinstance(key_b64, str) or len(key_b64) != 44:
        return None
    try:
        raw = base64.b64decode(key_b64, validate=True)
    except ValueError:
        return None
    return raw if len(raw) == 32 and base64.b64encode(raw).decode() == key_b64 else None


def origin_of(url: str) -> str:
    """scheme://host[:port] of a URL: lower-case, without the scheme's default port (agent_hub.py does the same)."""
    parts = urlsplit(url)
    host = (parts.hostname or "").lower()
    netloc = f"[{host}]" if ":" in host else host
    port = parts.port
    if port and (parts.scheme, port) not in (("https", 443), ("http", 80)):
        netloc += f":{port}"
    return f"{parts.scheme.lower()}://{netloc}"


def fingerprint(raw: bytes) -> str:
    """SHA-256 of the raw public key, first 16 bytes, as 8 groups of 4 hex digits."""
    digest = hashlib.sha256(raw).hexdigest()
    return ":".join(digest[i:i + 4] for i in range(0, 32, 4))


def phrase(raw: bytes) -> str:
    """Six words derived from the public key (48 bits), easy to read out on a call to compare identities."""
    digest = hashlib.sha256(b"agent-hub-fleet phrase\n" + raw).digest()
    return " ".join(WORDS[b] for b in digest[:6])


def signed_text(purpose: str, server_id: str, server_fp: str, origin: str, challenge: str, nonce: str, timestamp: int,
                key_b64: str) -> bytes:  # fmt: skip
    """What a member signs to log in (purpose "login"), to get a panel link ("panel") or for one action ("trust
    <fp>", ...): bound to this server's id and key (its fingerprint), to the exact origin the member connects to, to
    one challenge (single use) and to a time. A signature made for one server is useless at any other, and a server
    that relays a member's login to another address gets a signature that names its own address, which the real
    server refuses."""
    return (f"agent-hub-fleet/{PROTOCOL}\n{purpose}\n{server_id}\n{server_fp}\n{origin}\n{challenge}\n{nonce}\n"
            f"{int(timestamp)}\n{key_b64}").encode()  # fmt: skip


def challenge_text(server_id: str, origin: str, challenge: str, nonce: str, purpose: str, key_b64: str) -> bytes:
    """What the server signs with its own key when it gives out a challenge: the member checks it with the key it
    pinned when it joined (and that the origin is the one it connects to) before it signs anything."""
    return (f"agent-hub-fleet/{PROTOCOL} server challenge\n{server_id}\n{origin}\n{challenge}\n{nonce}\n{purpose}\n"
            f"{key_b64}").encode()  # fmt: skip


def proof_text(server_id: str, origin: str, nonce: str) -> bytes:
    """What this server signs to prove to a member that kept its access token over a lost connection that it still
    holds its key at this origin (the member's own fresh nonce; nothing a member signs starts this way)."""
    return f"agent-hub-fleet/{PROTOCOL} server proof\n{server_id}\n{origin}\n{nonce}".encode()


def load_server_key(home: Path) -> tuple[Any, bool]:
    """This server's own Ed25519 key (home/server.key, owner-only; back it up with the database): made on first use.
    Members pin its public key when they join. (The key, whether it was made now.)"""
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    path = home / "server.key"
    made = False
    if not path.exists():
        key = Ed25519PrivateKey.generate()
        pem = key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                serialization.NoEncryption())  # fmt: skip
        tmp = home / f".server.key.{secrets.token_hex(6)}"
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as f:
            f.write(pem)
            f.flush()
            os.fsync(f.fileno())
        try:
            os.link(tmp, path)  # never replaces a key another process made meanwhile
            made = True
        except FileExistsError:
            pass
        except OSError:
            if not path.exists():
                os.replace(tmp, path)
                made = True
        finally:
            with contextlib.suppress(OSError):
                os.unlink(tmp)
    with contextlib.suppress(OSError):
        os.chmod(path, 0o600)
    key = serialization.load_pem_private_key(path.read_bytes(), None)
    if not isinstance(key, Ed25519PrivateKey):
        raise SystemExit(f"hub_server.py: {path} is not an Ed25519 key")
    return key, made


def invite_link(base: str, token: str, server_fp: str) -> str:
    """An invite link: the server's address, the one-time token, and (after #, never sent to any server) the server
    key's fingerprint, which the joining member's client pins and checks."""
    return f"{base.rstrip('/')}/join/{token}#key={server_fp.replace(':', '')}"


def public_raw(key: Any) -> bytes:
    from cryptography.hazmat.primitives import serialization

    return key.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)


def verify(raw_key: bytes, message: bytes, signature_b64: Any) -> bool:
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

    if not isinstance(signature_b64, str) or len(signature_b64) > 200:
        return False
    try:
        signature = base64.b64decode(signature_b64, validate=True)
        Ed25519PublicKey.from_public_bytes(raw_key).verify(signature, message)
        return True
    except (ValueError, InvalidSignature):
        return False


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def same(a: str, b: str) -> bool:
    return hmac.compare_digest(a.encode("utf-8", "replace"), b.encode("utf-8", "replace"))


# --------------------------------------------------------------------------
# Text from members
# --------------------------------------------------------------------------

ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)|\x1b[@-Z\\-_]")
# A secret in a message (as agent_hub.py refuses in the hive): a private key or a well-known key or token format.
SECRET_TEXT_RE = re.compile(
    r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY(?: BLOCK)?-----"
    r"|\bsk-(?:ant-|proj-|or-v1-)?[A-Za-z0-9_-]{20,}"
    r"|\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{30,}|\bgithub_pat_[A-Za-z0-9_]{30,}|\bglpat-[A-Za-z0-9_-]{20,}"
    r"|\bxox[abposr]-[A-Za-z0-9-]{10,}|\b(?:AKIA|ASIA)[0-9A-Z]{16}\b|\bAIza[0-9A-Za-z_-]{35}"
    r"|\b[sr]k_(?:live|test)_[0-9A-Za-z]{16,}|\btvly-[A-Za-z0-9_-]{16,}|\bhf_[A-Za-z0-9]{30,}"
    r"|\beyJ[A-Za-z0-9_-]{8,}\.eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}"
)


def clean(text: Any) -> str:
    """Text for display: no escape sequences, control or format characters (shown as ?)."""
    text = ANSI_RE.sub("", "" if text is None else str(text)).replace("\r\n", "\n")
    return "".join(c if c in "\n\t" or c.isprintable() else "?" for c in text)


def one_line(text: Any, width: int = 200) -> str:
    line = " ".join(clean(text).split())
    return line if len(line) <= width else line[: width - 1] + "…"


class Refused(Exception):
    """A request this server refuses: an HTTP status, a short code clients act on, and a text for people."""

    def __init__(self, status: int, code: str, message: str, retry_after: int | None = None) -> None:
        super().__init__(message)
        self.status, self.code, self.message, self.retry_after = status, code, message, retry_after


def text_field(raw: Any, what: str, limit: int, *, line: bool = True, required: bool = False) -> str:
    if raw is None:
        raw = ""
    if not isinstance(raw, str):
        raise Refused(400, "invalid", f"{what} must be text")
    text = "".join(c for c in raw if c.isprintable() or c in "\n\t").strip()
    if line:
        text = " ".join(text.split())
    if required and not text:
        raise Refused(400, "invalid", f"{what} is empty")
    if len(text) > limit:
        raise Refused(413, "too_long", f"{what} is {len(text):,} characters long; the limit is {limit:,}")
    return text


def int_id(raw: Any, what: str, lowest: int = 1) -> int:
    """A number a member sent (an id, an event cursor): a whole number from `lowest` to 2**63-1 (SQLite's range),
    else a 400 that says so (never a server error)."""
    text = str(raw if raw is not None else "").strip().lstrip("#")
    if isinstance(raw, bool) or not re.fullmatch(r"-?\d{1,30}", text) or not lowest <= int(text) <= MAX_ID:
        raise Refused(400, "invalid", f"{what} must be a whole number from {lowest} to {MAX_ID}")
    return int(text)


# The rest of the client's full check (agent_hub.py's _refuse_secrets), so a modified client cannot get past it:
# NAME=value / "name": "value" where the name says it holds a secret and the value looks like one (a placeholder such
# as $VAR, <key> or ... does not); a password in a URL; the credentials of an Authorization header or a bearer token.
# (The client also refuses the values of the secrets its own process knows; the server cannot know those.)
SECRET_ASSIGN_RE = re.compile(r"(?=(?<![\w.-])([A-Za-z_][\w.-]*)[\"']?[ \t]*[:=][ \t]*[\"']?([^\s\"'`,;:=]+))")
SECRET_NAME_RE = re.compile(r"(?i)api[_-]?key|secret|token|passw(?:or)?d|pwd|private[_-]?key|access[_-]?key"
                            r"|credential|auth[_-]?key")  # fmt: skip
SECRET_ENV_RE = re.compile(r"(_KEY|_TOKEN)$|_SECRET|PASSWORD|^AGENT_HUB_", re.IGNORECASE)
SECRET_URL_RE = re.compile(r"(?i)\b[a-z][a-z0-9+.-]*://[^\s/:@]*:([^\s/@]+)@")
SECRET_HEADER_RE = re.compile(r"(?i)\b(?:authorization\s*[:=]\s*[\"']?(?:(?:bearer|basic|token|digest)\s+)?|bearer\s+)"
                              r"([^\s\"'`,;]+)")  # fmt: skip
PLACEHOLDERS = {"changeme", "password", "redacted", "placeholder", "example", "required", "optional", "none", "null"}
URL_PLACEHOLDERS = PLACEHOLDERS | {"pass", "passwd", "pwd", "secret", "user", "username", "xxx", "xxxx"}


def secret_like(value: str) -> bool:
    """A value that looks like a real secret, not a placeholder or code (os.environ["X"], getenv(...))."""
    if len(value) < 6 or value[0] in "$<{%([" or re.search(r"[(){}\[\]]", value):
        return False
    if value.lower() in PLACEHOLDERS or set(value) <= set(".*xX_-"):
        return False
    return bool(re.search(r"\d", value)) or len(value) >= 16


def credential_like(value: str, least: int = 8) -> bool:
    """The password of a URL or the credentials of an Authorization header that look real."""
    value = value.rstrip(".")
    if len(value) < least or value[0] in "$<{%([" or re.search(r"[(){}\[\]<>]", value):
        return False
    if value.lower() in URL_PLACEHOLDERS or set(value) <= set(".*xX_-"):
        return False
    return not (least >= 8 and re.fullmatch(r"[A-Za-z]+(?:[_-][A-Za-z]+)*", value) and len(value) < 24)


def has_secret(text: str) -> bool:
    """The same patterns as the client's full check (a secret in a known format, NAME=value, URL passwords,
    Authorization / Bearer credentials)."""
    return bool(SECRET_TEXT_RE.search(text) or any(
        (SECRET_NAME_RE.search(name) or (SECRET_ENV_RE.search(name) and not name.upper().startswith("AGENT_HUB_")))
        and secret_like(value) for name, value in SECRET_ASSIGN_RE.findall(text)) or any(
        credential_like(value, 4) for value in SECRET_URL_RE.findall(text)) or any(
        credential_like(value) for value in SECRET_HEADER_RE.findall(text)))  # fmt: skip


def check_secrets(text: str, what: str) -> None:
    if has_secret(text):  # never repeat the value
        raise Refused(422, "secret", f"refusing {what}: it looks like it contains a secret (an API key, a token, a"
                      " password or a private key). Never send secrets through the fleet.")  # fmt: skip


# Secrets files, by name (as agent_hub.py's _secret_name judges them): never uploaded to a fleet.
SENSITIVE_DIRS = {".ssh", ".gnupg", ".aws", ".kube", ".docker", ".agent-hub", ".git"}
SENSITIVE_FILES = (
    ".env", ".env.*", "*.env", ".envrc", "*.pem", "*.key", "*.p12", "*.pfx", "*.keystore", "id_rsa*", "id_dsa*",
    "id_ecdsa*", "id_ed25519*", ".netrc", "_netrc", ".npmrc", ".pypirc", ".git-credentials", ".pgpass", ".vault-token",
    "credentials", "*.secret", ".credentials.json", "auth.json", "application_default_credentials.json",
    "mcp.json", ".mcp.json", ".claude.json", "opencode.json", "opencode.jsonc", "claude_desktop_config.json",
)  # fmt: skip
SECRET_STEMS = ("credentials", "secrets")
SOURCE_SUFFIXES = {
    "py", "pyi", "ts", "tsx", "js", "jsx", "mjs", "cjs", "java", "kt", "kts", "gradle", "groovy", "go", "rb", "rs",
    "cs", "swift", "php", "scala", "c", "cc", "cpp", "h", "hpp", "dart", "ex", "exs", "lua", "vue", "svelte", "m",
    "mm", "html", "htm", "css", "scss", "sass", "less", "razor", "cshtml", "proto", "graphql", "gql", "md", "mdx",
    "rst",
}  # fmt: skip
BACKUP_RE = re.compile(r"(~|\.~\d+~|\.(bak|backup|old|orig|save|swp|tmp)(\.\d+)*)$")
SENSITIVE_PATHS = {("gh", "hosts.yml"), (".codex", "config.toml"), ("git", "credentials")}
HARMLESS_SUFFIXES = (".example", ".sample", ".template", ".dist", ".pub")


def secret_file(parts: list[str]) -> bool:
    """Whether a relative path (its folder and file names) looks like a secrets file."""
    if SENSITIVE_DIRS & {part.lower() for part in parts[:-1]}:
        return True
    name = parts[-1].lower()
    while BACKUP_RE.sub("", name) not in ("", name):
        name = BACKUP_RE.sub("", name)
    if name.endswith(HARMLESS_SUFFIXES):
        return False
    bare = name.removeprefix(".")
    if any(fnmatch.fnmatch(n, pattern) for n in (name, bare) for pattern in SENSITIVE_FILES):
        return True
    stem, _, rest = bare.partition(".")
    if stem in SECRET_STEMS and rest.rpartition(".")[2] not in SOURCE_SUFFIXES:
        return True
    return (parts[-2].lower() if len(parts) > 1 else "", name) in SENSITIVE_PATHS


def neutral_path(raw: Any) -> str:
    """A file's path in a workspace task, as members send it: relative, "/" between folders, nothing that leads
    out of the task folder or into .git, never a secrets file (the same rules as agent_hub.py's workspaces)."""
    if not isinstance(raw, str):
        raise Refused(400, "invalid", "a file's path must be text")
    text = raw.strip()
    shown = one_line(text, 80)
    if not text or len(text) > MAX_PATH or not text.isprintable() or "\\" in text or ":" in text:
        raise Refused(400, "invalid_path", f"{shown!r} is not a usable path: relative, \"/\" between folders, no \\ or"
                      f" :, at most {MAX_PATH} characters")  # fmt: skip
    if text.startswith(("/", "~")):
        raise Refused(400, "invalid_path", f"{shown!r}: paths are relative to the task folder, never absolute")
    parts = [p for p in text.split("/") if p not in ("", ".")]
    if not parts:
        raise Refused(400, "invalid_path", f"{shown!r} names no file")
    for part in parts:
        if part == "..":
            raise Refused(400, "invalid_path", f"{shown!r}: .. is not allowed: nothing above the task folder")
        if len(part) > 255 or part.endswith((".", " ")):
            raise Refused(400, "invalid_path", f"{shown!r}: a name in it is too long or ends with a dot or a space")
        if part.casefold() == ".git" or re.fullmatch(r"(?i)git~\d+", part):
            raise Refused(400, "invalid_path", f"{shown!r}: .git folders are never part of a task")
        if part.endswith(".agent-hub-tmp"):
            raise Refused(400, "invalid_path", f"{shown!r}: a temporary file of the hub")
    if secret_file(parts):
        raise Refused(422, "secret_file", f"refusing {shown!r}: it looks like a secrets file; secrets files never go"
                      " through the fleet")  # fmt: skip
    return "/".join(parts)


REPO_SCP_RE = re.compile(r"^(?:[A-Za-z0-9._-]+@)?([A-Za-z0-9.-]+):(?!/)([A-Za-z0-9._~/-]+)$")  # git@host:org/repo.git
REF_RE = re.compile(r"^(?!-)(?!.*\.\.)(?!.*@\{)[A-Za-z0-9._/+-]{1,200}$")


def repo_url(raw: Any) -> str:
    """A repository a worker's agent may clone: https or ssh (git@host:path too); plain http only on this computer
    (127.0.0.1, localhost: tests and development). Never file://, a local path, ext:: or anything git would read as
    an option."""
    text = text_field(raw, "the repository URL", MAX_URL, required=True)
    shown = one_line(text, 100)
    bad = Refused(400, "invalid_repo", f"{shown!r} is not a repository URL a worker may clone: give https://host/path"
                  " or ssh://host/path (or git@host:path); never a local path, file:// or other transports")  # fmt: skip
    if text.startswith(("-", "/", ".", "~")) or any(c.isspace() for c in text) or "::" in text or "\\" in text:
        raise bad
    parts = urlsplit(text)
    if "://" in text:
        host = (parts.hostname or "").lower()
        local = host in ("127.0.0.1", "localhost", "::1")
        if parts.scheme not in ("https", "ssh") and not (parts.scheme == "http" and local):
            raise bad
        if not host or parts.password or (parts.scheme != "ssh" and parts.username) or parts.query or parts.fragment:
            raise Refused(400, "invalid_repo", f"{shown!r}: no credentials, query or fragment in a repository URL")
        # ssh would read a host or user starting with "-" as an option; git decodes %XX first, so the check does too
        if unquote(host).startswith("-") or unquote(parts.username or "").startswith("-"):
            raise bad
        if parts.path in ("", "/") or ".." in parts.path.split("/"):
            raise bad
        return text
    found = REPO_SCP_RE.match(text)
    if not found or ("." not in found.group(1) and "@" not in text) or ".." in found.group(2).split("/"):
        raise bad
    if found.group(1).startswith("-") or text.startswith("-"):
        raise bad
    return text


def repo_ref(raw: Any) -> str:
    text = text_field(raw, "the branch or commit", 200)
    if text and not REF_RE.match(text):
        raise Refused(400, "invalid_ref", "the branch, tag or commit: letters, digits and . _ / + - (no .., not"
                      " starting with -)")  # fmt: skip
    return text


def pr_url(raw: Any) -> str:
    text = text_field(raw, "the pull-request URL", MAX_URL)
    if not text:
        return ""
    parts = urlsplit(text)
    local = (parts.hostname or "") in ("127.0.0.1", "localhost")
    if (parts.scheme != "https" and not (parts.scheme == "http" and local)) or not parts.hostname or parts.username \
            or parts.password or any(c.isspace() for c in text):  # fmt: skip
        raise Refused(400, "invalid_url", "the pull-request URL must be https://host/... without credentials")
    return text


# --------------------------------------------------------------------------
# The database
# --------------------------------------------------------------------------

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS members (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    key TEXT NOT NULL UNIQUE,              -- the Ed25519 public key, standard base64 of the 32 raw bytes
    fingerprint TEXT NOT NULL,
    role TEXT NOT NULL,                    -- director | partner | worker | lobby | banned | removed
    name TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    agent TEXT NOT NULL DEFAULT '',        -- what agent / model it runs, as given
    contact TEXT NOT NULL DEFAULT '',      -- its human's contact, as given
    address TEXT NOT NULL DEFAULT '',      -- where it joined from (the owner sees it)
    joined_at REAL NOT NULL,
    admitted_at REAL,
    invite INTEGER,
    since INTEGER NOT NULL DEFAULT 0,      -- events for everyone up to this id are from before it was admitted
    lead_since INTEGER NOT NULL DEFAULT 0, -- events for the leads up to this id are from before it became one
    cursor INTEGER NOT NULL DEFAULT 0,     -- the last event its client confirmed it showed to its agent
    last_seen REAL,
    state TEXT NOT NULL DEFAULT '',
    removed_at REAL,
    removed_reason TEXT NOT NULL DEFAULT '',
    removed_by TEXT NOT NULL DEFAULT ''
);
CREATE UNIQUE INDEX IF NOT EXISTS members_one_director ON members(role) WHERE role = 'director';
CREATE TABLE IF NOT EXISTS tokens (
    hash TEXT PRIMARY KEY,                 -- sha256 of the access token (the token itself is never stored)
    member INTEGER NOT NULL,
    created REAL NOT NULL,
    expires REAL NOT NULL,
    revoked REAL
);
CREATE INDEX IF NOT EXISTS tokens_member ON tokens(member);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    at REAL NOT NULL,
    audience TEXT NOT NULL,                -- member (one member) | all (every admitted member) | leads
    member INTEGER,
    kind TEXT NOT NULL,
    sender INTEGER,                        -- a member's id, 0 for the owner, NULL for the server
    text TEXT NOT NULL DEFAULT '',
    data TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS events_member ON events(member, id);
CREATE TABLE IF NOT EXISTS invites (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    hash TEXT NOT NULL UNIQUE,             -- sha256 of the invite token (shown once, in the link)
    created REAL NOT NULL,
    created_by TEXT NOT NULL,
    expires REAL NOT NULL,
    max_uses INTEGER NOT NULL,
    uses INTEGER NOT NULL DEFAULT 0,
    revoked REAL,
    note TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS audit (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    at REAL NOT NULL,
    actor TEXT NOT NULL,
    actor_id INTEGER,
    action TEXT NOT NULL,
    target INTEGER,
    detail TEXT NOT NULL DEFAULT '',
    address TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL, updated_at REAL, updated_by TEXT);
CREATE TABLE IF NOT EXISTS joins (at REAL NOT NULL, address TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS panel_links (hash TEXT PRIMARY KEY, member INTEGER, created REAL NOT NULL,
                                        expires REAL NOT NULL);
CREATE TABLE IF NOT EXISTS panel_sessions (hash TEXT PRIMARY KEY, member INTEGER, csrf TEXT NOT NULL,
                                           created REAL NOT NULL, expires REAL NOT NULL, address TEXT NOT NULL DEFAULT '');
"""
# Schema migrations: the database records its version (meta.schema); a newer version's tables are added in order,
# each step in one transaction (SQL, or a function given the connection). A database from a newer hub_server.py is
# refused. Versions: 1 = 2.0.0-dev (phase 1), 2 = 2.0.0, 3 = 2.0.0 after review round 1.
MIGRATIONS: dict[int, Any] = {
    2: """
CREATE TABLE IF NOT EXISTS orders (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    instructions TEXT NOT NULL,
    criteria TEXT NOT NULL DEFAULT '',     -- acceptance criteria
    kind TEXT NOT NULL,                    -- workspace | repo
    repo_url TEXT NOT NULL DEFAULT '',
    repo_ref TEXT NOT NULL DEFAULT '',     -- branch, tag or commit
    priority INTEGER NOT NULL DEFAULT 0,   -- -1 low, 0 normal, 1 high, 2 urgent
    deadline REAL,                         -- expires if not reported by then
    claim_minutes INTEGER NOT NULL,        -- a claim without news for this long is re-queued
    targets TEXT NOT NULL DEFAULT '[]',    -- member ids it is offered to ([]: any member, the first to claim)
    status TEXT NOT NULL,                  -- approval | queued | offered | claimed | in_progress | reported |
                                           -- accepted | rejected | cancelled | expired
    announced INTEGER NOT NULL DEFAULT 0,  -- members were told it is there (not while paused or before approval)
    created REAL NOT NULL,
    created_by INTEGER NOT NULL,           -- the member who made it (0: the owner)
    creator TEXT NOT NULL,
    approved_by TEXT NOT NULL DEFAULT '',
    claimer INTEGER,
    claimed_at REAL,
    news_at REAL,                          -- the claimer's last news (claim, download, progress, question, report)
    updated REAL NOT NULL,
    closed_reason TEXT NOT NULL DEFAULT '',
    files INTEGER NOT NULL DEFAULT 0,
    bytes INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS orders_status ON orders(status);
CREATE INDEX IF NOT EXISTS orders_claimer ON orders(claimer);
CREATE TABLE IF NOT EXISTS order_files (
    order_id INTEGER NOT NULL, path TEXT NOT NULL, content TEXT NOT NULL, sha TEXT NOT NULL,
    PRIMARY KEY (order_id, path)
);
CREATE TABLE IF NOT EXISTS order_log (     -- an order's history
    id INTEGER PRIMARY KEY AUTOINCREMENT, order_id INTEGER NOT NULL, at REAL NOT NULL, member INTEGER,
    who TEXT NOT NULL, kind TEXT NOT NULL, text TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS order_log_order ON order_log(order_id, id);
CREATE TABLE IF NOT EXISTS order_declines (
    order_id INTEGER NOT NULL, member INTEGER NOT NULL, at REAL NOT NULL, reason TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (order_id, member)
);
CREATE TABLE IF NOT EXISTS questions (
    id INTEGER PRIMARY KEY AUTOINCREMENT, order_id INTEGER NOT NULL, member INTEGER NOT NULL, at REAL NOT NULL,
    text TEXT NOT NULL, answer TEXT, answered_by TEXT NOT NULL DEFAULT '', answered_at REAL
);
CREATE INDEX IF NOT EXISTS questions_order ON questions(order_id);
CREATE TABLE IF NOT EXISTS reports (
    id INTEGER PRIMARY KEY AUTOINCREMENT, order_id INTEGER NOT NULL, member INTEGER NOT NULL, at REAL NOT NULL,
    summary TEXT NOT NULL, patch TEXT NOT NULL DEFAULT '', pr_url TEXT NOT NULL DEFAULT '',
    files INTEGER NOT NULL DEFAULT 0, bytes INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'pending',  -- pending | accepted | rejected | superseded
    reason TEXT NOT NULL DEFAULT '', decided_by TEXT NOT NULL DEFAULT '', decided_at REAL
);
CREATE INDEX IF NOT EXISTS reports_order ON reports(order_id, id);
CREATE TABLE IF NOT EXISTS report_files (
    report_id INTEGER NOT NULL, path TEXT NOT NULL,
    kind TEXT NOT NULL,                    -- changed | new | deleted
    content TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (report_id, path)
);
CREATE TABLE IF NOT EXISTS handovers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    from_member INTEGER NOT NULL, to_member INTEGER NOT NULL,
    from_fp TEXT NOT NULL, to_fp TEXT NOT NULL,
    created REAL NOT NULL, expires REAL NOT NULL,
    status TEXT NOT NULL,                  -- offered | accepted (waits for the owner) | done | cancelled | expired
    offer_sig TEXT NOT NULL,               -- the director's signature (and what it signed)
    accept_sig TEXT NOT NULL DEFAULT '',   -- the partner's
    decided_at REAL, decided_by TEXT NOT NULL DEFAULT ''
);
""",
}


def migrate_3(con: sqlite3.Connection) -> None:
    """Schema 3 (2.0.0 review round 1): what a human approved (the order's digest), the order an event is about (so
    a member who gave an order back is not offered it again), and members identified by their key's bytes: every
    stored key in its canonical encoding. Rows that held the same key in another encoding (a banned key that came
    back that way) are no longer members: banned if any of them was banned, else removed."""
    def columns(table: str) -> set[str]:
        return {r[1] for r in con.execute(f"PRAGMA table_info({table})")}

    if "approved_digest" not in columns("orders"):
        con.execute("ALTER TABLE orders ADD COLUMN approved_digest TEXT NOT NULL DEFAULT ''")
    if "order_id" not in columns("events"):
        con.execute("ALTER TABLE events ADD COLUMN order_id INTEGER")
    groups: dict[bytes, list[Any]] = {}
    for r in con.execute("SELECT id, key, role FROM members ORDER BY id").fetchall():
        try:
            raw = base64.b64decode(r[1])
        except ValueError:
            continue
        if len(raw) == 32:
            groups.setdefault(raw, []).append(r)
    now = time.time()
    for raw, rows in groups.items():
        canonical = base64.b64encode(raw).decode()
        if len(rows) == 1 and rows[0][1] == canonical:
            continue
        banned = [r for r in rows if r[2] == "banned"]
        keep = (banned or rows)[0]  # a ban wins; else the identity that joined first
        alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
        low = alphabet.index(canonical[42]) & ~3
        spare = [canonical[:42] + alphabet[low | b] + "=" for b in range(4)]
        held = {r[1] for r in rows}
        for r in rows:
            if r is keep:
                continue
            key = r[1]
            if key == canonical:  # it gives the canonical string to the one kept, keeping another encoding
                key = next(s for s in spare if s not in held and s != canonical)
                held.add(key)
            con.execute("UPDATE members SET key = ?, role = ?, removed_at = COALESCE(removed_at, ?), removed_reason = ?,"
                        " removed_by = 'the server' WHERE id = ?", (key, "banned" if banned else "removed", now,
                                                                    f"the same key as #{keep[0]}, in another encoding",
                                                                    r[0]))  # fmt: skip
            con.execute("UPDATE tokens SET revoked = ? WHERE member = ? AND revoked IS NULL", (now, r[0]))
        con.execute("UPDATE members SET key = ?, fingerprint = ? WHERE id = ?", (canonical, fingerprint(raw), keep[0]))


MIGRATIONS[3] = migrate_3
SCHEMA_VERSION = str(max(MIGRATIONS))


def private_file(path: Path) -> None:
    """Create `path` owner-only (0600) if it does not exist."""
    if not path.exists():
        fd = os.open(path, os.O_WRONLY | os.O_CREAT, 0o600)
        os.close(fd)
    with contextlib.suppress(OSError):
        os.chmod(path, 0o600)


class Db:
    def __init__(self, home: Path) -> None:
        self.path = home / "server.db"
        private_file(self.path)
        self.con = sqlite3.connect(self.path, timeout=15, isolation_level=None, check_same_thread=False)
        self.con.row_factory = sqlite3.Row
        self.con.execute("PRAGMA journal_mode=WAL")
        self.con.execute("PRAGMA synchronous=NORMAL")
        self.con.execute("PRAGMA busy_timeout=15000")
        self.con.executescript(SCHEMA)
        self.con.execute("INSERT OR IGNORE INTO meta VALUES ('schema', '1')")
        self.migrate()
        for suffix in ("-wal", "-shm"):
            extra = Path(str(self.path) + suffix)
            if extra.exists():
                with contextlib.suppress(OSError):
                    os.chmod(extra, 0o600)

    def close(self) -> None:
        self.con.close()

    def schema(self) -> int:
        row = self.one("SELECT value FROM meta WHERE key = 'schema'")
        try:
            return int(row["value"]) if row is not None else 1
        except ValueError:
            return 1

    def migrate(self) -> None:
        """Bring the database to this version's schema, one step (one transaction) at a time."""
        version = self.schema()
        if version > int(SCHEMA_VERSION):
            raise SystemExit(f"hub_server.py: {self.path} has schema version {version}, made by a newer hub_server.py"
                             f" (this one knows {SCHEMA_VERSION}): update hub_server.py")  # fmt: skip
        for step in range(version + 1, int(SCHEMA_VERSION) + 1):
            migration = MIGRATIONS[step]
            try:
                if callable(migration):
                    self.con.execute("BEGIN IMMEDIATE")
                    migration(self.con)
                else:
                    self.con.executescript(f"BEGIN IMMEDIATE;\n{migration}")  # the transaction stays open
                self.con.execute("UPDATE meta SET value = ? WHERE key = 'schema'", (str(step),))
                self.con.execute("COMMIT")
            except Exception:
                if self.con.in_transaction:
                    self.con.execute("ROLLBACK")
                raise

    @contextlib.contextmanager
    def tx(self):
        self.con.execute("BEGIN IMMEDIATE")
        try:
            yield self.con
        except BaseException:
            self.con.execute("ROLLBACK")
            raise
        self.con.execute("COMMIT")

    def one(self, sql: str, args: Iterable[Any] = ()) -> sqlite3.Row | None:
        return self.con.execute(sql, tuple(args)).fetchone()

    def all(self, sql: str, args: Iterable[Any] = ()) -> list[sqlite3.Row]:
        return self.con.execute(sql, tuple(args)).fetchall()

    def data_version(self) -> int:
        return int(self.con.execute("PRAGMA data_version").fetchone()[0])


# --------------------------------------------------------------------------
# Who acts: the owner (CLI on the VPS, or the panel through an owner link) or a member
# --------------------------------------------------------------------------


@dataclass
class Actor:
    member: sqlite3.Row | None  # None: the owner
    address: str = ""

    @property
    def owner(self) -> bool:
        return self.member is None

    @property
    def role(self) -> str:
        return "owner" if self.member is None else self.member["role"]

    @property
    def id(self) -> int:
        return 0 if self.member is None else int(self.member["id"])

    @property
    def label(self) -> str:
        if self.member is None:
            return "the owner"
        return f"{self.member['name']} (#{self.member['id']}, {ROLE_NAMES.get(self.member['role'], '?')})"


def member_label(row: Any) -> str:
    return f"{row['name']} (#{row['id']})"


def short_fp(fp: str) -> str:
    return fp[:9]


# --------------------------------------------------------------------------
# The fleet's rules: shared by the API, the panel and the owner's command line
# --------------------------------------------------------------------------


class Fleet:
    def __init__(self, home: Path, conf: dict[str, Any]) -> None:
        self.home = home
        self.conf = conf
        self.server_id: str = conf["server_id"]
        self.db = Db(home)
        self.started = 0.0  # when the running server started (0: the owner's command line, which never sweeps)
        self.start_grace = START_GRACE
        self.key, made = load_server_key(home)
        self.key_raw = public_raw(self.key)
        self.key_b64 = base64.b64encode(self.key_raw).decode()
        self.key_fp = fingerprint(self.key_raw)
        if made:
            self.audit("the server", "server_key", None, f"this server's key was made: fingerprint {self.key_fp}, phrase"
                       f" {phrase(self.key_raw)!r} (members pin it when they join)")  # fmt: skip
        self._bind_old_approvals()

    def _bind_old_approvals(self) -> None:
        """Dispatch approvals given before approvals named the order's content (2.0.0, schema 2) were given for the
        orders as they are now: their digest is that content's, so only a later change of content voids them (not a
        priority, a deadline or a claim time)."""
        rows = self.db.all("SELECT * FROM orders WHERE approved_by != '' AND approved_digest = ''")
        if not rows:
            return
        with self.db.tx() as con:
            for row in rows:
                digest = self.digest(row)
                con.execute("UPDATE orders SET approved_digest = ? WHERE id = ? AND approved_digest = ''",
                            (digest, row["id"]))  # fmt: skip
                self._log(row["id"], None, "approval_bound", f"its dispatch approval (given before the update) is"
                          f" bound to its content as it is now ({digest[:12]})")  # fmt: skip
            self.audit("the server", "approvals_bound", None, f"{len(rows)} dispatch approval(s) given before the"
                       " update bound to their orders' content as it is now: "
                       + ", ".join(f"#{r['id']}" for r in rows[:50]))  # fmt: skip

    # -- hooks the running server fills in ------------------------------------

    def changed(self) -> None:
        """Something members may be waiting for was written (the server wakes its long-polls)."""

    def presence(self, row: Any) -> tuple[str, float | None]:
        """A member's presence: online (waiting for orders), working, idle or offline, and when it was last seen."""
        seen = row["last_seen"]
        if not seen or time.time() - seen > 60:
            return "offline", seen
        return {"waiting": "online", "working": "working", "idle": "idle"}.get(row["state"], "online"), seen

    def tokens_revoked(self, member_id: int) -> None:
        """A member's tokens were revoked (the server ends its long-polls)."""

    # -- settings ------------------------------------------------------------

    def settings(self) -> dict[str, Any]:
        values = dict(SETTING_DEFAULTS)
        for row in self.db.all("SELECT key, value FROM settings"):
            if row["key"] in values:
                with contextlib.suppress(ValueError):
                    values[row["key"]] = json.loads(row["value"])
        return values

    @staticmethod
    def check_setting(key: str, value: Any) -> Any:
        if key == "join_mode":
            value = str(value).strip().lower()
            if value not in JOIN_MODES:
                raise Refused(400, "invalid", f"join mode must be one of {', '.join(JOIN_MODES)}")
            return value
        if key in BOOL_SETTINGS:
            if isinstance(value, str):
                words = {"on": True, "true": True, "yes": True, "1": True, "off": False, "false": False, "no": False,
                         "0": False}  # fmt: skip
                if value.strip().lower() not in words:
                    raise Refused(400, "invalid", f"{key} must be on or off")
                return words[value.strip().lower()]
            if not isinstance(value, bool):
                raise Refused(400, "invalid", f"{key} must be on or off")
            return value
        if key in SETTING_LIMITS:
            try:
                number = int(value)
            except (TypeError, ValueError):
                raise Refused(400, "invalid", f"{key} must be a whole number") from None
            lo, hi = SETTING_LIMITS[key]
            if not lo <= number <= hi:
                raise Refused(400, "invalid", f"{key} must be from {lo} to {hi}")
            return number
        raise Refused(400, "invalid", f"unknown setting {key!r}")

    def change_settings(self, actor: Actor, changes: dict[str, Any], human: bool = False) -> list[str]:
        """Change settings (live: they apply to the next request); what changed, for people. Leads' agents (tokens)
        change the live switches; a lead's human in the panel (`human`) also dispatch approval, and the director's
        human the takeover; the owner everything (limits, and whether a handover needs the owner)."""
        if RANK[actor.role] < RANK["partner"]:
            raise Refused(403, "forbidden", "only the director, a trusted partner or the owner changes settings")
        if not actor.owner:
            allowed = set(LIVE_SETTINGS) | ({"dispatch_approval"} if human else set()) | (
                {"takeover"} if human and actor.role == "director" else set())  # fmt: skip
            extra = set(changes) - allowed
            if extra & set(HUMAN_SETTINGS):
                raise Refused(403, "forbidden", "dispatch approval and the takeover are switched by a human: the"
                              " director's human in the panel, or the owner (hub_server.py owner settings)")  # fmt: skip
            if extra:
                raise Refused(403, "forbidden", "only the owner changes the limits (on the server: hub_server.py owner"
                              " settings)")  # fmt: skip
        checked = {k: self.check_setting(k, v) for k, v in changes.items()}
        done: list[str] = []
        with self.db.tx():
            now = self.settings()
            for key, value in checked.items():
                if now.get(key) == value:
                    continue
                self.db.con.execute("INSERT INTO settings VALUES (?, ?, ?, ?) ON CONFLICT(key) DO UPDATE SET"
                                    " value = excluded.value, updated_at = excluded.updated_at, updated_by ="
                                    " excluded.updated_by", (key, json.dumps(value), time.time(), actor.label))  # fmt: skip
                shown = ("on" if value else "off") if isinstance(value, bool) else str(value)
                done.append(f"{key} {shown}")
                self.audit(actor, "setting", None, f"{key}: {now.get(key)!r} -> {value!r}")
                sender = actor.id if actor.member else 0
                if key == "paused":
                    self.event("all", None, "paused" if value else "resumed", sender, "", {"by": actor.label})
                    if not value:
                        self._announce_waiting()
                elif key == "takeover":
                    self.event("leads", None, "takeover", sender, "", {"on": value, "by": actor.label})
                else:
                    self.event("leads", None, "settings", sender, "", {"key": key, "value": value, "by": actor.label})
        if done:
            self.changed()
        return done

    # -- records -------------------------------------------------------------

    def audit(self, actor: Actor | str, action: str, target: int | None, detail: str = "", address: str = "") -> None:
        if isinstance(actor, Actor):
            label, actor_id, address = actor.label, (actor.id if actor.member else None), address or actor.address
        else:
            label, actor_id = actor, None
        self.db.con.execute("INSERT INTO audit (at, actor, actor_id, action, target, detail, address) VALUES"
                            " (?, ?, ?, ?, ?, ?, ?)", (time.time(), label, actor_id, action, target, detail[:2000],
                                                       address))  # fmt: skip

    def event(self, audience: str, member: int | None, kind: str, sender: int | None, text: str = "",
              data: dict[str, Any] | None = None) -> int:
        order = (data or {}).get("order")
        order_id = order.get("id") if isinstance(order, dict) and isinstance(order.get("id"), int) else None
        cur = self.db.con.execute("INSERT INTO events (at, audience, member, kind, sender, text, data, order_id) VALUES"
                                  " (?, ?, ?, ?, ?, ?, ?, ?)", (time.time(), audience, member, kind, sender, text,
                                                                json.dumps(data or {}), order_id))  # fmt: skip
        return int(cur.lastrowid or 0)

    def last_event(self) -> int:
        row = self.db.one("SELECT MAX(id) AS id FROM events")
        return int(row["id"] or 0) if row else 0

    def member(self, member_id: int) -> sqlite3.Row | None:
        return self.db.one("SELECT * FROM members WHERE id = ?", (member_id,))

    def director(self) -> sqlite3.Row | None:
        return self.db.one("SELECT * FROM members WHERE role = 'director'")

    def find(self, ref: Any, roles: Iterable[str] = ROLES) -> sqlite3.Row:
        """A member by "#12" / "12", name, or fingerprint (8 or more hex digits, colons optional)."""
        roles = tuple(roles)
        text = str(ref if ref is not None else "").strip()
        marks = ",".join("?" * len(roles))
        row = None
        if re.fullmatch(r"#?\d{1,9}", text):
            row = self.db.one(f"SELECT * FROM members WHERE id = ? AND role IN ({marks})", (int(text.lstrip("#")),
                                                                                         *roles))  # fmt: skip
        elif NAME_RE.match(text.lower()) and not re.fullmatch(r"[0-9a-f:]{8,}", text.lower()):
            rows = self.db.all(f"SELECT * FROM members WHERE name = ? AND role IN ({marks}) ORDER BY"
                               " CASE WHEN role IN ('banned', 'removed') THEN 1 ELSE 0 END, id DESC",
                               (text.lower(), *roles))  # fmt: skip
            row = rows[0] if rows else None
        else:
            hexes = text.lower().replace(":", "")
            if re.fullmatch(r"[0-9a-f]{8,32}", hexes):
                rows = [r for r in self.db.all(f"SELECT * FROM members WHERE role IN ({marks})", roles)
                        if r["fingerprint"].replace(":", "").startswith(hexes)]  # fmt: skip
                if len(rows) > 1 and len({r["fingerprint"] for r in rows}) == 1:  # one key: its member
                    rows = sorted(rows, key=lambda r: (key_bytes(r["key"]) is None, r["id"]))[:1]
                if len(rows) > 1:
                    raise Refused(409, "ambiguous", f"{len(rows)} members have fingerprints starting {text}; give more")
                row = rows[0] if rows else None
        if row is None:
            raise Refused(404, "no_member", f"no such member: {one_line(text, 60)!r} (give #id, name or fingerprint)")
        return row

    def snapshot(self, member: Any) -> dict[str, Any]:
        """What a member's client keeps in sync: its role, the director, whether the fleet is paused."""
        settings = self.settings()
        director = self.director()
        out: dict[str, Any] = {
            "server": {"id": self.server_id, "name": self.conf.get("name", ""), "version": __version__,
                       "protocol": PROTOCOL},
            "me": self.member_info(member, full=True),
            "director": self.member_info(director) if director is not None else None,
            "paused": bool(settings["paused"]),
        }
        if member["role"] in LEADS:
            out["join_mode"], out["approval"] = settings["join_mode"], bool(settings["approval"])
            out["lobby"] = int(self.db.one("SELECT COUNT(*) AS n FROM members WHERE role = 'lobby'")["n"])
            out["dispatch_approval"], out["takeover"] = bool(settings["dispatch_approval"]), bool(settings["takeover"])
            counts = {r["status"]: r["n"] for r in self.db.all("SELECT status, COUNT(*) AS n FROM orders WHERE status IN"
                                                                  " ('approval', 'reported') GROUP BY status")}  # fmt: skip
            out["to_approve"], out["to_review"] = counts.get("approval", 0), counts.get("reported", 0)
            out["questions"] = int(self.db.one("SELECT COUNT(*) AS n FROM questions WHERE answer IS NULL")["n"])
        if member["role"] in ("worker", "partner"):
            rows = self.db.all("SELECT id, claim_minutes, news_at, claimed_at FROM orders WHERE claimer = ? AND status"
                               " IN ('claimed', 'in_progress')", (member["id"],))  # fmt: skip
            out["working"] = [r["id"] for r in rows]
            now = time.time()
            # how long each claim lasts without news (fleet_progress renews it), for the member's warnings
            out["claims"] = [{"id": r["id"], "claim_minutes": r["claim_minutes"],
                              "lapses_in": max(0, int(self.lapses_at(r) - now))} for r in rows]  # fmt: skip
        return out

    def lapses_at(self, row: Any) -> float:
        """When a claim goes back up for lack of news (measured from the server's start at the earliest: the time it
        was down does not count)."""
        news = max(float(row["news_at"] or row["claimed_at"] or time.time()), self.started)
        return news + int(row["claim_minutes"]) * 60

    def member_info(self, row: Any, full: bool = False) -> dict[str, Any]:
        raw = base64.b64decode(row["key"])
        info: dict[str, Any] = {"id": row["id"], "name": row["name"], "role": row["role"],
                                "fingerprint": row["fingerprint"], "phrase": phrase(raw)}  # fmt: skip
        if full:
            presence, seen = self.presence(row)
            info.update(description=row["description"], agent=row["agent"], contact=row["contact"],
                        joined_at=row["joined_at"], admitted_at=row["admitted_at"], presence=presence,
                        last_seen=seen)  # fmt: skip
        return info

    # -- tokens --------------------------------------------------------------

    def issue_token(self, member_id: int, seconds: int) -> tuple[str, float]:
        token = "fh1_" + secrets.token_urlsafe(32)
        expires = time.time() + seconds
        self.db.con.execute("INSERT INTO tokens VALUES (?, ?, ?, ?, NULL)", (sha(token), member_id, time.time(),
                                                                             expires))  # fmt: skip
        return token, expires

    def revoke_tokens(self, member_id: int) -> int:
        now = time.time()
        cur = self.db.con.execute("UPDATE tokens SET revoked = ? WHERE member = ? AND revoked IS NULL", (now, member_id))
        self.db.con.execute("DELETE FROM panel_sessions WHERE member = ?", (member_id,))
        self.db.con.execute("DELETE FROM panel_links WHERE member = ?", (member_id,))
        self.tokens_revoked(member_id)
        return cur.rowcount

    # -- joining -------------------------------------------------------------

    def join(self, raw_key: bytes, key_b64: str, join: Any, address: str) -> sqlite3.Row:
        """A new identity (or one that left or was removed) asks to join; the join rules as they are right now."""
        if not isinstance(join, dict):
            raise Refused(403, "not_member", "this key is not a member of this fleet: join first (fleet_join with a"
                          " name)")  # fmt: skip
        name = text_field(join.get("name"), "name", 32, required=True).lower()
        if not NAME_RE.match(name):
            raise Refused(400, "invalid", "name must be 2-32 characters: lowercase letters, digits, - and _ (start with"
                          " a letter or digit)")  # fmt: skip
        if name in RESERVED_NAMES:
            raise Refused(400, "invalid", f'"{name}" is reserved; pick another name')
        description = text_field(join.get("description"), "description", MAX_DESCRIPTION)
        agent = text_field(join.get("agent"), "agent", MAX_AGENT)
        contact = text_field(join.get("contact"), "contact", MAX_CONTACT)
        for value, what in ((description, "the description"), (agent, "the agent"), (contact, "the contact")):
            check_secrets(value, what)
        invite_token = join.get("invite") or ""
        if not isinstance(invite_token, str) or len(invite_token) > 200:
            raise Refused(400, "invalid", "invite must be the token of an invite link")
        fp = fingerprint(raw_key)
        now = time.time()
        with self.db.tx() as con:
            old = self.db.one("SELECT * FROM members WHERE key = ?", (key_b64,))
            if old is not None and old["role"] == "banned":
                raise Refused(403, "banned", "this key is banned from this fleet")
            if old is not None and old["role"] in ACTIVE:
                return old
            settings = self.settings()
            mode = settings["join_mode"]
            if mode == "closed":
                raise Refused(403, "closed", "this fleet takes no new members right now (join mode: closed)")
            invite = None
            if invite_token:
                invite = self.db.one("SELECT * FROM invites WHERE hash = ?", (sha(invite_token),))
                problem = self.invite_problem(invite, now)
                if problem:
                    raise Refused(403, "invite_invalid", f"this invite {problem}; ask the director for a new one")
            elif mode == "invite":
                raise Refused(403, "invite_required", "this fleet takes new members by invite only: ask the director"
                              " for an invite link")  # fmt: skip
            if mode == "open" and invite is None:
                recent = self.db.one("SELECT COUNT(*) AS n FROM joins WHERE address = ? AND at > ?", (address,
                                                                                                  now - 3600))["n"]  # fmt: skip
                if recent >= int(settings["new_keys_per_hour"]):
                    raise Refused(429, "too_many_new_keys", "too many new identities from this address in the last"
                                  " hour; try again later or ask the director for an invite", retry_after=600)  # fmt: skip
            taken = self.db.one("SELECT id FROM members WHERE name = ? AND role IN ('director', 'partner', 'worker',"
                                " 'lobby') AND key != ?", (name, key_b64))  # fmt: skip
            if taken is not None:
                raise Refused(409, "name_taken", f'the name "{name}" is taken in this fleet; pick another name')
            approval = bool(settings["approval"])
            if approval:
                waiting = self.db.one("SELECT COUNT(*) AS n FROM members WHERE role = 'lobby'")["n"]
                if waiting >= int(settings["lobby_max"]):
                    raise Refused(429, "lobby_full", "the lobby is full; try again later", retry_after=600)
            else:
                count = self.db.one("SELECT COUNT(*) AS n FROM members WHERE role IN ('director', 'partner',"
                                    " 'worker')")["n"]  # fmt: skip
                if count >= int(settings["members_max"]):
                    raise Refused(403, "full", "this fleet is full")
            role = "lobby" if approval else "worker"
            since = self.last_event()
            values = (fp, role, name, description, agent, contact, address, now, None if approval else now,
                      invite["id"] if invite is not None else None, since, since)  # fmt: skip
            if old is None:
                con.execute("INSERT INTO members (fingerprint, role, name, description, agent, contact, address,"
                            " joined_at, admitted_at, invite, since, cursor, key) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?,"
                            " ?, ?, ?, ?)", (*values, key_b64))  # fmt: skip
            else:
                con.execute("UPDATE members SET fingerprint = ?, role = ?, name = ?, description = ?, agent = ?,"
                            " contact = ?, address = ?, joined_at = ?, admitted_at = ?, invite = ?, since = ?, cursor"
                            " = ?, removed_at = NULL, removed_reason = '', removed_by = '' WHERE key = ?",
                            (*values, key_b64))  # fmt: skip
            row = self.db.one("SELECT * FROM members WHERE key = ?", (key_b64,))
            assert row is not None
            if invite is not None:
                con.execute("UPDATE invites SET uses = uses + 1 WHERE id = ?", (invite["id"],))
            if mode == "open" and invite is None:
                con.execute("INSERT INTO joins VALUES (?, ?)", (now, address))
            how = "into the lobby (approval on)" if approval else "admitted at once (approval off)"
            via = f" with invite #{invite['id']}" if invite is not None else f" (join mode {mode})"
            self.audit(f"{name} (#{row['id']}, new)", "join", row["id"], f"{how}{via}; fingerprint {fp}", address)
            self.event("leads", None, "lobby" if approval else "joined", row["id"], "",
                       {"member": self.member_info(row), "description": description, "agent": agent})  # fmt: skip
        self.changed()
        return row

    @staticmethod
    def invite_problem(invite: Any, now: float) -> str:
        if invite is None:
            return "is not valid"
        if invite["revoked"]:
            return "was revoked"
        if invite["expires"] <= now:
            return "has expired"
        if invite["uses"] >= invite["max_uses"]:
            return "is used up"
        return ""

    # -- the director's and partners' actions ----------------------------------

    def _target(self, actor: Actor, ref: Any, roles: Iterable[str], what: str) -> sqlite3.Row:
        """The member an action is for: only one ranked below the actor, and only in the roles the action takes."""
        if RANK[actor.role] < RANK["partner"]:
            raise Refused(403, "forbidden", f"only the director, a trusted partner or the owner may {what}")
        target = self.find(ref)
        if not actor.owner and target["id"] == actor.id:
            raise Refused(409, "self", f"you cannot {what} yourself")
        if RANK[actor.role] <= RANK.get(target["role"], -1) and target["role"] not in ("banned", "removed"):
            raise Refused(403, "forbidden", f"{'a ' + ROLE_NAMES[actor.role] if actor.member else 'the owner'} cannot"
                          f" {what} the {ROLE_NAMES[target['role']]}" if target["role"] == "director" else
                          f"{'a ' + ROLE_NAMES[actor.role] if actor.member else 'the owner'} cannot {what} a"
                          f" {ROLE_NAMES[target['role']]}")  # fmt: skip
        if target["role"] not in tuple(roles):
            raise Refused(409, "wrong_role", f"{member_label(target)} is {ROLE_NAMES[target['role']]}: you cannot"
                          f" {what} it")  # fmt: skip
        return target

    def admit(self, actor: Actor, ref: Any) -> sqlite3.Row:
        with self.db.tx() as con:
            target = self._target(actor, ref, ("lobby",), "admit")
            count = self.db.one("SELECT COUNT(*) AS n FROM members WHERE role IN ('director', 'partner',"
                                " 'worker')")["n"]  # fmt: skip
            if count >= int(self.settings()["members_max"]):
                raise Refused(403, "full", "this fleet is full (members_max)")
            since = self.last_event()
            con.execute("UPDATE members SET role = 'worker', admitted_at = ?, since = ?, cursor = MAX(cursor, ?)"
                        " WHERE id = ?", (time.time(), since, since, target["id"]))  # fmt: skip
            self.audit(actor, "admit", target["id"], f"{member_label(target)} admitted as a worker")
            self.event("member", target["id"], "admitted", actor.id, "", {"by": actor.label, "role": "worker"})
            self.event("leads", None, "joined", target["id"], "", {"member": self.member_info(target), "by": actor.label})
        self.changed()
        return target

    def remove(self, actor: Actor, ref: Any, how: str, reason: str = "") -> sqlite3.Row:
        """reject (a lobby member), kick (it may join again) or ban (its key never again)."""
        reason = text_field(reason, "reason", MAX_REASON)
        check_secrets(reason, "the reason")
        roles = {"reject": ("lobby",), "kick": ("partner", "worker", "lobby"),
                 "ban": ("partner", "worker", "lobby", "removed")}[how]  # fmt: skip
        with self.db.tx() as con:
            target = self._target(actor, ref, roles, how)
            role = "banned" if how == "ban" else "removed"
            said = {"reject": "rejected", "kick": "removed from the fleet", "ban": "banned"}[how]
            con.execute("UPDATE members SET role = ?, removed_at = ?, removed_reason = ?, removed_by = ? WHERE id = ?",
                        (role, time.time(), f"{said}{': ' + reason if reason else ''}", actor.label, target["id"]))
            self.revoke_tokens(target["id"])
            self.release_claims(target["id"], f"its claimer was {said} by {actor.label}")
            self._void_handovers("to_member = ?", (target["id"],), actor, f"{member_label(target)} was {said}")
            self.audit(actor, how, target["id"], f"{member_label(target)} ({target['role']}) {said}"
                       f"{'; reason: ' + reason if reason else ''}")  # fmt: skip
            self.event("leads", None, "removed", target["id"], "", {"member": self.member_info(target), "how": how,
                                                                    "by": actor.label})  # fmt: skip
        self.changed()
        return target

    def unban(self, actor: Actor, ref: Any) -> sqlite3.Row:
        if RANK[actor.role] < RANK["director"]:
            raise Refused(403, "forbidden", "only the director or the owner lifts a ban")
        with self.db.tx() as con:
            target = self.find(ref, ("banned",))
            con.execute("UPDATE members SET role = 'removed', removed_reason = 'ban lifted', removed_by = ? WHERE id ="
                        " ?", (actor.label, target["id"]))  # fmt: skip
            self.audit(actor, "unban", target["id"], f"{member_label(target)}: ban lifted (it may join again)")
        self.changed()
        return target

    def set_director(self, actor: Actor, ref: Any, old_role: str = "worker") -> tuple[sqlite3.Row, sqlite3.Row | None]:
        """The owner picks the director ("the website/VPS decides the director"), or replaces it at any time."""
        if not actor.owner:
            raise Refused(403, "forbidden", "only the owner sets the director this way; the director hands over to a"
                          " trusted partner with a signed handover")  # fmt: skip
        if old_role not in ("worker", "partner", "removed"):
            raise Refused(400, "invalid", "the old director becomes a worker, a partner or is removed")
        with self.db.tx() as con:
            target = self.find(ref, ("partner", "worker", "lobby", "director"))
            old = self.director()
            if old is not None and old["id"] == target["id"]:
                return target, None
            now = time.time()
            if old is not None:
                con.execute("UPDATE members SET role = ?, removed_at = ?, removed_reason = ?, removed_by = ? WHERE id"
                            " = ?", (old_role, now if old_role == "removed" else None, "replaced as director by the"
                                     " owner" if old_role == "removed" else "", actor.label if old_role == "removed"
                                     else "", old["id"]))  # fmt: skip
                if old_role == "removed":
                    self.revoke_tokens(old["id"])
                else:
                    con.execute("DELETE FROM panel_sessions WHERE member = ?", (old["id"],))
                self.event("member", old["id"], "role", 0, "", {"role": old_role, "by": actor.label})
            self.release_claims(target["id"], "its claimer became the director")
            self._void_handovers("1 = 1", (), actor, f"the owner made {member_label(target)} the director")
            since = self.last_event()
            con.execute("UPDATE members SET role = 'director', admitted_at = COALESCE(admitted_at, ?), since = CASE"
                        " WHEN role = 'lobby' THEN ? ELSE since END, lead_since = CASE WHEN role = 'partner' THEN"
                        " lead_since ELSE ? END WHERE id = ?", (now, since, since, target["id"]))  # fmt: skip
            new = self.member(target["id"])
            assert new is not None
            self.audit(actor, "director", target["id"], f"{member_label(target)} is the director now (fingerprint"
                       f" {target['fingerprint']})" + (f"; the old director {member_label(old)} is now"
                       f" {ROLE_NAMES[old_role]}" if old is not None else ""))  # fmt: skip
            self.event("member", target["id"], "role", 0, "", {"role": "director", "by": actor.label})
            self.event("all", None, "director", 0, "", {"director": self.member_info(new), "by": actor.label})
        self.changed()
        return new, old

    def leave(self, member: Any) -> None:
        if member["role"] == "director":
            raise Refused(409, "director", "the director cannot leave the fleet: hand over to a trusted partner first"
                          " (fleet_handover), or the owner picks another director")  # fmt: skip
        with self.db.tx() as con:
            con.execute("UPDATE members SET role = 'removed', removed_at = ?, removed_reason = 'left', removed_by = ?"
                        " WHERE id = ?", (time.time(), member["name"], member["id"]))  # fmt: skip
            self.revoke_tokens(member["id"])
            self.release_claims(member["id"], "its claimer left the fleet")
            self._void_handovers("to_member = ?", (member["id"],), Actor(member), f"{member_label(member)} left the fleet")
            self.audit(Actor(member), "leave", member["id"], f"{member_label(member)} left the fleet")
            self.event("leads", None, "left", member["id"], "", {"member": self.member_info(member)})
        self.changed()

    # -- invites -------------------------------------------------------------

    def create_invite(self, actor: Actor, uses: Any = 1, hours: Any = 24, note: Any = "") -> tuple[str, sqlite3.Row]:
        if RANK[actor.role] < RANK["partner"]:
            raise Refused(403, "forbidden", "only the director, a trusted partner or the owner creates invites")
        try:
            uses, hours = int(uses), float(hours)
        except (TypeError, ValueError):
            raise Refused(400, "invalid", "uses must be a whole number and hours a number") from None
        if not 1 <= uses <= INVITE_MAX_USES:
            raise Refused(400, "invalid", f"uses must be from 1 to {INVITE_MAX_USES}")
        if not 0.05 <= hours <= INVITE_MAX_HOURS:
            raise Refused(400, "invalid", f"hours must be from 0.05 to {INVITE_MAX_HOURS}")
        note = text_field(note, "note", MAX_NOTE)
        check_secrets(note, "the note")
        token = secrets.token_urlsafe(24)
        with self.db.tx() as con:
            cur = con.execute("INSERT INTO invites (hash, created, created_by, expires, max_uses, note) VALUES (?, ?, ?,"
                              " ?, ?, ?)", (sha(token), time.time(), actor.label, time.time() + hours * 3600, uses,
                                            note))  # fmt: skip
            row = self.db.one("SELECT * FROM invites WHERE id = ?", (cur.lastrowid,))
            self.audit(actor, "invite", None, f"invite #{cur.lastrowid}: {uses} use(s), {hours:g} hour(s)"
                       f"{'; note: ' + note if note else ''}")  # fmt: skip
        assert row is not None
        return token, row

    def revoke_invite(self, actor: Actor, invite_id: Any) -> sqlite3.Row:
        if RANK[actor.role] < RANK["partner"]:
            raise Refused(403, "forbidden", "only the director, a trusted partner or the owner revokes invites")
        invite_id = int_id(invite_id, "the invite's number")
        with self.db.tx() as con:
            row = self.db.one("SELECT * FROM invites WHERE id = ?", (invite_id,))
            if row is None:
                raise Refused(404, "no_invite", f"no invite #{invite_id}")
            if not row["revoked"]:
                con.execute("UPDATE invites SET revoked = ? WHERE id = ?", (time.time(), invite_id))
                self.audit(actor, "invite_revoke", None, f"invite #{invite_id} revoked")
        return row

    def invites(self) -> list[dict[str, Any]]:
        now = time.time()
        return [{"id": r["id"], "created": r["created"], "created_by": r["created_by"], "expires": r["expires"],
                 "max_uses": r["max_uses"], "uses": r["uses"], "note": r["note"],
                 "state": self.invite_problem(r, now).replace("is not valid", "unknown") or "valid"}
                for r in self.db.all("SELECT * FROM invites ORDER BY id DESC LIMIT 200")]  # fmt: skip

    # -- messages ------------------------------------------------------------

    def send(self, actor: Actor, to: Any, text: Any) -> tuple[int, str]:
        """A message: members to the director; the director (and the owner) to a member or to everyone; trusted
        partners to the director or a member."""
        text = text_field(text, "the message", MAX_TEXT, line=False, required=True)
        check_secrets(text, "the message")
        target_ref = str(to or "").strip().lower()
        role = actor.role
        if role not in ("owner", *MEMBERS):
            raise Refused(403, "forbidden", "members in the lobby cannot send messages; wait to be admitted")
        if target_ref in ("all", "everyone"):
            if role not in ("owner", "director"):
                raise Refused(403, "forbidden", "only the director messages everyone")
            with self.db.tx():
                event_id = self.event("all", None, "broadcast", actor.id, text, {"by": actor.label})
            self.changed()
            return event_id, "everyone"
        if target_ref in ("", "director"):
            target = self.director()
            if target is None:
                raise Refused(409, "no_director", "this fleet has no director yet")
        else:
            target = self.find(target_ref, ACTIVE)
        if role == "worker" and target["role"] != "director":
            raise Refused(403, "forbidden", "workers message the director only (leave `to` empty)")
        if role == "partner" and target["role"] not in ("director", "worker", "partner", "lobby"):
            raise Refused(403, "forbidden", "not a member you can message")
        if target["role"] == "lobby" and role not in ("owner", "director", "partner"):
            raise Refused(403, "forbidden", "members in the lobby get messages from the director only")
        if target["id"] == actor.id:
            raise Refused(409, "self", "that is you")
        with self.db.tx():
            event_id = self.event("member", target["id"], "message", actor.id, text, {"by": actor.label})
        self.changed()
        return event_id, member_label(target)

    # -- orders ------------------------------------------------------------------
    # An order is a request from a lead (the director or a trusted partner) to the fleet: a workspace task (files the
    # lead shared, as neutral paths) or a repo task (a repository the worker's agent clones). It is offered to named
    # members or to anyone (the first to claim it takes it), claimed by one member, reported, and accepted or rejected
    # by a human or agent who may manage it. Every step is in its history (order_log) and reaches whom it concerns as
    # an event.

    def order(self, ref: Any) -> sqlite3.Row:
        text = str(ref if ref is not None else "").strip().lstrip("#")
        if not re.fullmatch(r"\d{1,12}", text):
            raise Refused(400, "invalid", "give the order's number")
        row = self.db.one("SELECT * FROM orders WHERE id = ?", (int(text),))
        if row is None:
            raise Refused(404, "no_order", f"no order #{text}")
        return row

    @staticmethod
    def targets(row: Any) -> list[int]:
        try:
            return [int(t) for t in json.loads(row["targets"] or "[]")]
        except (ValueError, TypeError):
            return []

    def can_manage(self, actor: Actor, row: Any) -> bool:
        """The owner and the director manage every order; a trusted partner the orders it made."""
        return actor.owner or actor.role == "director" or (actor.role == "partner" and row["created_by"] == actor.id)

    def _manage(self, actor: Actor, row: Any, what: str) -> None:
        if not self.can_manage(actor, row):
            raise Refused(403, "forbidden", f"only the director, the owner or the partner who made order #{row['id']}"
                          f" may {what} it")  # fmt: skip

    def can_see(self, actor: Actor, row: Any) -> bool:
        """Leads and the owner see every order; a worker sees what is offered to it (or to anyone), what it claimed
        and what it reported."""
        if RANK[actor.role] >= RANK["partner"]:
            return True
        if row["claimer"] == actor.id:
            return True
        if row["status"] in ORDER_WAITING and row["announced"]:
            targets = self.targets(row)
            return not targets or actor.id in targets
        return self.db.one("SELECT 1 FROM reports WHERE order_id = ? AND member = ?", (row["id"], actor.id)) is not None

    def _log(self, row_id: int, actor: Actor | None, kind: str, text: str = "") -> None:
        who = actor.label if actor is not None else "the server"
        member = (actor.id if actor.member is not None else 0) if actor is not None else None
        self.db.con.execute("INSERT INTO order_log (order_id, at, member, who, kind, text) VALUES (?, ?, ?, ?, ?, ?)",
                            (row_id, time.time(), member, who, kind, text[:4000]))  # fmt: skip

    def order_info(self, row: Any, full: bool = False) -> dict[str, Any]:
        names = {r["id"]: r["name"] for r in self.db.all(
            "SELECT id, name FROM members WHERE id IN (" + ",".join("?" * (len(self.targets(row)) + 1)) + ")",
            (*self.targets(row), row["claimer"] or -1))}  # fmt: skip
        info: dict[str, Any] = {
            "id": row["id"], "title": row["title"], "kind": row["kind"], "status": row["status"],
            "priority": row["priority"], "deadline": row["deadline"], "created": row["created"],
            "creator": row["creator"], "created_by": row["created_by"],
            "to": [names.get(t, f"#{t}") for t in self.targets(row)],
            "claimer": names.get(row["claimer"]) if row["claimer"] else None, "claimer_id": row["claimer"],
            "files": row["files"], "bytes": row["bytes"], "updated": row["updated"],
            "repo_url": row["repo_url"], "repo_ref": row["repo_ref"], "closed_reason": row["closed_reason"],
            "claim_minutes": row["claim_minutes"],
        }  # fmt: skip
        if full:
            info.update(instructions=row["instructions"], criteria=row["criteria"], claim_minutes=row["claim_minutes"],
                        claimed_at=row["claimed_at"], news_at=row["news_at"], approved_by=row["approved_by"],
                        digest=self.digest(row))  # fmt: skip
            if row["claimer"] and row["status"] in ORDER_WORKING:
                info["lapses_in"] = max(0, int(self.lapses_at(row) - time.time()))
        return info

    def digest(self, row: Any) -> str:
        """What a human approves (dispatch approval): a hash of the order's content (title, instructions, criteria,
        kind, repository and ref, who it is for, its files). Any change makes another digest, and voids the
        approval."""
        files = [[r["path"], r["sha"]] for r in self.db.all("SELECT path, sha FROM order_files WHERE order_id = ? ORDER"
                                                             " BY path", (row["id"],))]  # fmt: skip
        content = {"title": row["title"], "instructions": row["instructions"], "criteria": row["criteria"],
                   "kind": row["kind"], "repo_url": row["repo_url"], "repo_ref": row["repo_ref"],
                   "targets": sorted(self.targets(row)), "files": files}  # fmt: skip
        return hashlib.sha256(json.dumps(content, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
                              .encode("utf-8")).hexdigest()  # fmt: skip

    def _order_targets(self, to: Any) -> list[sqlite3.Row]:
        """The members an order is for: [] for anyone; else admitted workers or partners by #id, name or fingerprint."""
        if to is None or to == "" or to == [] or (isinstance(to, str) and to.strip().lower() in ("any", "anyone")):
            return []
        refs = [to] if isinstance(to, str) else to
        if not isinstance(refs, list) or len(refs) > 100:
            raise Refused(400, "invalid", "to: a member, a list of members (at most 100) or \"any\"")
        out: list[sqlite3.Row] = []
        for ref in refs:
            if isinstance(ref, str) and "," in ref:
                raise Refused(400, "invalid", "to: give members as a list, not separated by commas")
            row = self.find(ref, ("worker", "partner"))
            if row["id"] not in [r["id"] for r in out]:
                out.append(row)
        return out

    @staticmethod
    def _priority(raw: Any) -> int:
        if raw is None or raw == "":
            return 0
        if isinstance(raw, str) and raw.strip().lower() in PRIORITIES:
            return PRIORITIES[raw.strip().lower()]
        if isinstance(raw, int) and not isinstance(raw, bool) and raw in PRIORITIES.values():
            return raw
        raise Refused(400, "invalid", f"priority must be one of {', '.join(PRIORITIES)}")

    @staticmethod
    def _deadline(raw: Any) -> float | None:
        if raw is None or raw == "":
            return None
        try:
            hours = float(raw)
        except (TypeError, ValueError):
            raise Refused(400, "invalid", "deadline_hours must be a number of hours") from None
        if not 0.01 <= hours <= 24 * 365:
            raise Refused(400, "invalid", "deadline_hours must be from 0.01 to 8760")
        return time.time() + hours * 3600

    def _files(self, raw: Any, limit: int, what: str, base: dict[str, str] | None = None
               ) -> list[tuple[str, str, str]]:  # fmt: skip
        """Uploaded files, checked: (path, kind, content). For an order (base None) every file is new; for a report
        the kind is worked out against the order's files (changed, new, deleted; unchanged ones are dropped)."""
        if raw is None:
            return []
        if not isinstance(raw, list) or len(raw) > MAX_FILES:
            raise Refused(400, "invalid", f"{what}: a list of at most {MAX_FILES} files")
        out: list[tuple[str, str, str]] = []
        seen: set[str] = set()
        total = 0
        for item in raw:
            if not isinstance(item, dict):
                raise Refused(400, "invalid", f"{what}: each file is {{\"path\": ..., \"content\": ...}}")
            path = neutral_path(item.get("path"))
            if path.casefold() in seen:
                raise Refused(400, "invalid", f"{what}: {path!r} is there twice")
            seen.add(path.casefold())
            deleted = bool(item.get("deleted"))
            content = item.get("content")
            if deleted:
                if base is None or path not in base:
                    raise Refused(400, "invalid", f"{what}: {path!r} cannot be deleted: it is not one of the order's"
                                  " files")  # fmt: skip
                out.append((path, "deleted", ""))
                continue
            if not isinstance(content, str) or "\x00" in content:
                raise Refused(400, "invalid", f"{what}: {path!r}: content must be text (text files only)")
            size = len(content.encode("utf-8"))
            total += size
            if total > limit:
                raise Refused(413, "too_large", f"{what}: more than {limit:,} bytes in all (a limit the owner sets)")
            check_secrets(content, f"{path!r}")
            if base is None:
                out.append((path, "new", content))
            elif path in base:
                if base[path] != content:
                    out.append((path, "changed", content))
            else:
                out.append((path, "new", content))
        return out

    def _storage_check(self, adding: int) -> None:
        used = self.db.one("SELECT (SELECT COALESCE(SUM(bytes), 0) FROM orders) + (SELECT COALESCE(SUM(bytes), 0)"
                           " FROM reports) AS n")["n"]  # fmt: skip
        if used + adding > int(self.settings()["storage_mb"]) * 1_000_000:
            raise Refused(507, "storage_full", "this fleet server keeps no more files (storage_mb, set by the owner);"
                          f" the files of finished orders are deleted {EVENT_KEEP_DAYS} days after they finish")  # fmt: skip

    def create_orders(self, actor: Actor, spec: dict[str, Any]) -> list[sqlite3.Row]:
        """New orders (one, or one per member with `each`): offered at once unless the fleet is paused or orders
        wait for a human's dispatch approval."""
        if RANK[actor.role] < RANK["partner"]:
            raise Refused(403, "forbidden", "only the director, a trusted partner or the owner gives orders")
        title = text_field(spec.get("title"), "the title", MAX_TITLE, required=True)
        instructions = text_field(spec.get("instructions"), "the instructions", MAX_INSTRUCTIONS, line=False,
                                  required=True)  # fmt: skip
        criteria = text_field(spec.get("criteria"), "the acceptance criteria", MAX_CRITERIA, line=False)
        kind = str(spec.get("kind") or "workspace").strip().lower()
        if kind not in ORDER_KINDS:
            raise Refused(400, "invalid", "kind must be workspace or repo")
        url = repo_url(spec.get("repo_url")) if kind == "repo" else ""
        ref = repo_ref(spec.get("repo_ref")) if kind == "repo" else ""
        if kind == "workspace" and (spec.get("repo_url") or spec.get("repo_ref")):
            raise Refused(400, "invalid", "a workspace order takes files, not a repository")
        if kind == "repo" and spec.get("files"):
            raise Refused(400, "invalid", "a repo order takes no files: the worker's agent clones the repository")
        priority, deadline = self._priority(spec.get("priority")), self._deadline(spec.get("deadline_hours"))
        settings = self.settings()
        minutes = spec.get("claim_minutes") or settings["claim_minutes"]
        minutes = self.check_setting("claim_minutes", minutes)
        for text, what in ((title, "the title"), (instructions, "the instructions"), (criteria, "the criteria")):
            check_secrets(text, what)
        files = self._files(spec.get("files"), int(settings["max_order_bytes"]), "files")
        size = sum(len(c.encode("utf-8")) for _p, _k, c in files)
        targets = self._order_targets(spec.get("to"))
        groups = [[t] for t in targets] if spec.get("each") and len(targets) > 1 else [targets]
        made: list[sqlite3.Row] = []
        with self.db.tx() as con:
            open_now = self.db.one("SELECT COUNT(*) AS n FROM orders WHERE status IN (" + ",".join("?" * len(ORDER_OPEN))
                                   + ")", ORDER_OPEN)["n"]  # fmt: skip
            if open_now + len(groups) > int(settings["orders_open_max"]):
                raise Refused(429, "too_many_orders", "too many unfinished orders on this server (orders_open_max)")
            self._storage_check(size * len(groups))
            for group in groups:
                ids = [t["id"] for t in group]
                status = "approval" if settings["dispatch_approval"] else ("offered" if ids else "queued")
                now = time.time()
                cur = con.execute(
                    "INSERT INTO orders (title, instructions, criteria, kind, repo_url, repo_ref, priority, deadline,"
                    " claim_minutes, targets, status, created, created_by, creator, updated, files, bytes) VALUES"
                    " (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (title, instructions, criteria, kind, url, ref, priority, deadline, minutes, json.dumps(ids),
                     status, now, actor.id, actor.label, now, len(files), size))  # fmt: skip
                order_id = int(cur.lastrowid or 0)
                con.executemany("INSERT INTO order_files VALUES (?, ?, ?, ?)",
                                [(order_id, p, c, sha(c)) for p, _k, c in files])  # fmt: skip
                row = self.order(order_id)
                who = ", ".join(member_label(t) for t in group) or "anyone"
                self._log(order_id, actor, "created", f"{kind} order for {who}" + (f" ({len(files)} file(s))" if files
                                                                                    else ""))  # fmt: skip
                self.audit(actor, "order", None, f"order #{order_id} {title!r}: {kind}, for {who}"
                           + ("; waits for dispatch approval" if status == "approval" else ""))  # fmt: skip
                if status == "approval":
                    self.event("leads", None, "order_approval", actor.id, "", {"order": self.order_info(row)})
                else:
                    self._announce(row)
                made.append(self.order(order_id))
        self.changed()
        return made

    def _announce(self, row: Any) -> None:
        """Tell the members an order is there for them (not while the fleet is paused: at the resume)."""
        if row["status"] not in ORDER_WAITING or self.settings()["paused"]:
            return
        info = self.order_info(row)
        targets = self.targets(row)
        data = {"order": info, "excerpt": row["instructions"][:1500]}
        if targets:
            declined = {r["member"] for r in self.db.all("SELECT member FROM order_declines WHERE order_id = ?",
                                                         (row["id"],))}  # fmt: skip
            for t in targets:
                if t not in declined:  # it declined or gave it back: not offered again unless named again
                    self.event("member", t, "order", row["created_by"], "", data)
        else:  # events_for leaves it out for those who declined it
            self.event("workers", None, "order", row["created_by"], "", data)
        self.db.con.execute("UPDATE orders SET announced = 1 WHERE id = ?", (row["id"],))
        self._log(row["id"], None, "offered", "to " + (", ".join(info["to"]) if targets else "anyone"))

    def _announce_waiting(self) -> None:
        for row in self.db.all("SELECT * FROM orders WHERE status IN ('queued', 'offered') AND announced = 0"):
            self._announce(row)

    @staticmethod
    def approval_refs(raw: Any) -> list[tuple[int, str]]:
        """The orders a human approves or refuses, as shown to it: "12:<digest>,13:<digest>" (or a list of such)."""
        items = raw if isinstance(raw, list) else str(raw or "").split(",")
        out: list[tuple[int, str]] = []
        for item in items:
            text = f"{item[0]}:{item[1]}" if isinstance(item, tuple) and len(item) == 2 else str(item).strip()
            found = re.fullmatch(r"#?(\d{1,19}):([0-9a-f]{64})", text)
            if not found:
                raise Refused(400, "invalid", "orders: each one as <number>:<digest> (the digest of the content that"
                              " was shown, from the order's details), separated by commas")  # fmt: skip
            out.append((int_id(found.group(1), "an order's number"), found.group(2)))
        if not out or len(out) > 500:
            raise Refused(400, "invalid", "orders: from 1 to 500 orders")
        return out

    def creator_rank(self, row: Any) -> int:
        if not row["created_by"]:
            return RANK["owner"]
        creator = self.member(row["created_by"])
        return RANK.get(creator["role"], -1) if creator is not None else -1

    def may_decide(self, actor: Actor, row: Any) -> bool:
        """Who approves or refuses an order's dispatch: the owner, the director (their humans), or a partner's human
        for an order made by a partner (never one of someone ranked higher)."""
        return actor.owner or actor.role == "director" or (actor.role == "partner" and self.creator_rank(row) <=
                                                            RANK["partner"])  # fmt: skip

    def approve(self, actor: Actor, refs: Any, yes: bool, reason: str = "") -> list[sqlite3.Row]:
        """A human lets orders waiting for dispatch approval go out (or refuses them: cancelled): exactly the orders
        named, each only as it was shown (its digest); all of them or none."""
        if RANK[actor.role] < RANK["partner"]:
            raise Refused(403, "forbidden", "only the director's or a partner's human, or the owner, approves orders")
        reason = text_field(reason, "the reason", MAX_REASON)
        check_secrets(reason, "the reason")
        items = self.approval_refs(refs)
        done = []
        with self.db.tx() as con:
            for oid, digest in items:
                row = self.order(oid)
                if row["status"] != "approval":
                    raise Refused(409, "wrong_status", f"order #{row['id']} is {STATUS_NAMES[row['status']]}, not"
                                  " waiting for approval (nothing was decided)")  # fmt: skip
                if not self.may_decide(actor, row):
                    raise Refused(403, "forbidden", f"order #{row['id']} was made by {row['creator']}: a trusted"
                                  " partner approves or refuses only orders of partners; the director's human or the"
                                  " owner decides this one (nothing was decided)")  # fmt: skip
                if not same(digest, self.digest(row)):
                    raise Refused(409, "changed", f"order #{row['id']} changed since it was shown to you: look at it"
                                  " again (reload); nothing was decided")  # fmt: skip
                now = time.time()
                if yes:
                    status = "offered" if self.targets(row) else "queued"
                    con.execute("UPDATE orders SET status = ?, approved_by = ?, approved_digest = ?, updated = ? WHERE"
                                " id = ?", (status, actor.label, digest, now, row["id"]))  # fmt: skip
                    self._log(row["id"], actor, "approved", f"dispatch approved (content {digest[:12]})")
                    self.audit(actor, "order_approve", None, f"order #{row['id']} approved for dispatch (content"
                               f" {digest[:12]})")  # fmt: skip
                    self._announce(self.order(row["id"]))
                else:
                    con.execute("UPDATE orders SET status = 'cancelled', closed_reason = ?, updated = ? WHERE id = ?",
                                (f"dispatch refused{': ' + reason if reason else ''}", now, row["id"]))  # fmt: skip
                    self._log(row["id"], actor, "refused", reason or "dispatch refused")
                    self.audit(actor, "order_refuse", None, f"order #{row['id']} refused for dispatch"
                               f"{'; reason: ' + reason if reason else ''}")  # fmt: skip
                self.event("leads", None, "order_decided", actor.id if actor.member else 0, "",
                           {"order": self.order_info(self.order(row["id"])), "approved": yes, "by": actor.label})  # fmt: skip
                done.append(self.order(row["id"]))
        self.changed()
        return done

    def orders(self, actor: Actor, status: Any = None, mine: bool = False, limit: int = 100) -> list[dict[str, Any]]:
        status = str(status or "").strip().lower()
        if status and status not in (*ORDER_STATUSES, "open", "all", "final"):
            raise Refused(400, "invalid", f"status: one of {', '.join(ORDER_STATUSES)}, open, final or all")
        wanted = (ORDER_OPEN if status in ("", "open") else ORDER_FINAL if status == "final" else ORDER_STATUSES
                  if status == "all" else (status,))  # fmt: skip
        marks = ",".join("?" * len(wanted))
        if mine:  # what it holds now: claimed by it (and with "all" or "final": finished)
            rows = self.db.all(f"SELECT * FROM orders WHERE status IN ({marks}) AND claimer = ? ORDER BY priority DESC,"
                               " id DESC LIMIT ?", (*wanted, actor.id, limit))  # fmt: skip
        elif RANK[actor.role] >= RANK["partner"]:
            rows = self.db.all(f"SELECT * FROM orders WHERE status IN ({marks}) ORDER BY CASE WHEN status IN ('queued',"
                               " 'offered', 'approval') THEN 0 ELSE 1 END, priority DESC, id DESC LIMIT ?",
                               (*wanted, limit))  # fmt: skip
        else:
            rows = self.db.all(f"SELECT * FROM orders WHERE status IN ({marks}) AND (claimer = ? OR (status IN"
                               " ('queued', 'offered') AND announced = 1) OR id IN (SELECT order_id FROM reports WHERE"
                               " member = ?)) ORDER BY priority DESC, id DESC LIMIT ?",
                               (*wanted, actor.id, actor.id, 5 * limit))  # fmt: skip
            rows = [r for r in rows if self.can_see(actor, r)][:limit]
        return [self.order_info(r) for r in rows]

    def order_detail(self, actor: Actor, ref: Any) -> dict[str, Any]:
        row = self.order(ref)
        if not self.can_see(actor, row):
            raise Refused(404, "no_order", f"no order #{row['id']} for you")
        info = self.order_info(row, full=True)
        lead = RANK[actor.role] >= RANK["partner"]
        info["paths"] = [r["path"] for r in self.db.all("SELECT path FROM order_files WHERE order_id = ? ORDER BY path",
                                                        (row["id"],))]  # fmt: skip
        questions = self.db.all("SELECT q.*, m.name FROM questions q LEFT JOIN members m ON m.id = q.member WHERE"
                                " q.order_id = ?" + ("" if lead else " AND q.member = ?") + " ORDER BY q.id",
                                (row["id"],) if lead else (row["id"], actor.id))  # fmt: skip
        info["questions"] = [{"id": q["id"], "member": q["name"], "at": q["at"], "text": q["text"],
                              "answer": q["answer"], "answered_by": q["answered_by"]} for q in questions]  # fmt: skip
        reports = self.db.all("SELECT r.*, m.name FROM reports r LEFT JOIN members m ON m.id = r.member WHERE"
                              " r.order_id = ?" + ("" if lead else " AND r.member = ?") + " ORDER BY r.id",
                              (row["id"],) if lead else (row["id"], actor.id))  # fmt: skip
        info["reports"] = [{"id": r["id"], "member": r["name"], "at": r["at"], "summary": r["summary"],
                            "files": r["files"], "pr_url": r["pr_url"], "patch_bytes": len(r["patch"]),
                            "status": r["status"], "reason": r["reason"], "decided_by": r["decided_by"]}
                           for r in reports]  # fmt: skip
        if lead:
            info["history"] = [{"at": r["at"], "who": r["who"], "kind": r["kind"], "text": r["text"]} for r in
                               self.db.all("SELECT * FROM order_log WHERE order_id = ? ORDER BY id", (row["id"],))]  # fmt: skip
            info["declines"] = [{"member": r["name"], "reason": r["reason"], "at": r["at"]} for r in self.db.all(
                "SELECT d.*, m.name FROM order_declines d LEFT JOIN members m ON m.id = d.member WHERE d.order_id = ?",
                (row["id"],))]  # fmt: skip
            info["manage"] = self.can_manage(actor, row)
        return info

    def order_files(self, actor: Actor, ref: Any) -> tuple[sqlite3.Row, list[dict[str, str]]]:
        """An order's files: for its claimer (the first download starts the work: in progress) and for leads."""
        with self.db.tx() as con:
            row = self.order(ref)
            if RANK[actor.role] < RANK["partner"] and row["claimer"] != actor.id:
                raise Refused(403, "forbidden", f"order #{row['id']} is not yours: claim it first (fleet_claim)")
            if row["claimer"] == actor.id and row["status"] == "claimed":
                con.execute("UPDATE orders SET status = 'in_progress', news_at = ?, updated = ? WHERE id = ?",
                            (time.time(), time.time(), row["id"]))  # fmt: skip
                self._log(row["id"], actor, "started", "downloaded the order's files")
                self.event("leads", None, "started", actor.id, "", {"order": self.order_info(self.order(row["id"]))})
        files = [{"path": r["path"], "content": r["content"], "sha": r["sha"]} for r in
                 self.db.all("SELECT * FROM order_files WHERE order_id = ? ORDER BY path", (row["id"],))]  # fmt: skip
        self.changed()
        return self.order(row["id"]), files

    def _working(self, member: Any, ref: Any, what: str, statuses: tuple[str, ...] = ORDER_WORKING) -> sqlite3.Row:
        row = self.order(ref)
        if row["claimer"] != member["id"]:
            ended = self.db.one("SELECT kind, text FROM order_log WHERE order_id = ? AND kind IN ('requeued',"
                                " 'reassigned', 'declined', 'void') AND id > (SELECT MAX(id) FROM order_log WHERE"
                                " order_id = ? AND kind = 'claimed' AND member = ?) ORDER BY id DESC LIMIT 1",
                                (row["id"], row["id"], member["id"]))  # fmt: skip
            if ended is not None:
                free = row["status"] in ORDER_WAITING and self.can_see(Actor(member), row)
                raise Refused(403, "claim_ended", f"order #{row['id']} is no longer yours: your claim ended"
                              f" ({one_line(ended['text'], 200) or ended['kind']}); it is {STATUS_NAMES[row['status']]}"
                              " now" + ("; it is still up for claiming: claim it again (your task folder is kept),"
                                       " then report" if free else ""))  # fmt: skip
            raise Refused(403, "forbidden", f"order #{row['id']} is not yours: only its claimer can {what}")
        if row["status"] not in statuses:
            raise Refused(409, "wrong_status", f"order #{row['id']} is {STATUS_NAMES[row['status']]}: you cannot {what}"
                          " now")  # fmt: skip
        return row

    def claim(self, member: Any, ref: Any) -> sqlite3.Row:
        actor = Actor(member)
        if member["role"] not in ("worker", "partner"):
            raise Refused(403, "forbidden", "the director hands out orders; workers and trusted partners claim them")
        with self.db.tx() as con:
            row = self.order(ref)
            if row["claimer"] == member["id"] and row["status"] in ORDER_WORKING:
                return row
            targets = self.targets(row)
            offered = row["announced"] or row["claimer"] is not None
            if row["status"] in (*ORDER_WORKING, "reported") and offered and (not targets or member["id"] in targets):
                raise Refused(409, "taken", f"order #{row['id']} is already claimed by another member")
            if not self.can_see(actor, row):
                raise Refused(404, "no_order", f"no order #{row['id']} for you")
            if self.settings()["paused"]:
                raise Refused(409, "paused", "the fleet is paused: no new work until it resumes")
            if row["status"] not in ORDER_WAITING or not row["announced"]:
                taken = row["status"] in ORDER_WORKING or row["status"] == "reported"
                raise Refused(409, "taken" if taken else "wrong_status", f"order #{row['id']} is"
                              f" {'already claimed by another member' if taken else STATUS_NAMES[row['status']]}")  # fmt: skip
            if targets and member["id"] not in targets:
                raise Refused(403, "forbidden", f"order #{row['id']} is offered to other members")
            now = time.time()
            con.execute("UPDATE orders SET status = 'claimed', claimer = ?, claimed_at = ?, news_at = ?, updated = ?"
                        " WHERE id = ?", (member["id"], now, now, now, row["id"]))  # fmt: skip
            con.execute("DELETE FROM order_declines WHERE order_id = ? AND member = ?", (row["id"], member["id"]))
            self._log(row["id"], actor, "claimed")
            row = self.order(row["id"])
            info = self.order_info(row)
            self.event("leads", None, "claimed", member["id"], "", {"order": info})
            others = [t for t in targets if t != member["id"]]
            for t in others:  # the others it was offered to: nothing to do for them any more
                self.event("member", t, "taken", None, "", {"order": info})
            if not targets:
                self.event("workers", None, "taken", None, "", {"order": info})
        self.changed()
        return row

    def progress(self, member: Any, ref: Any, note: Any) -> sqlite3.Row:
        note = text_field(note, "the note", MAX_WORK_TEXT, line=False, required=True)
        check_secrets(note, "the note")
        with self.db.tx() as con:
            row = self._working(member, ref, "report progress")
            con.execute("UPDATE orders SET status = 'in_progress', news_at = ?, updated = ? WHERE id = ?",
                        (time.time(), time.time(), row["id"]))  # fmt: skip
            self._log(row["id"], Actor(member), "progress", note)
            self.event("leads", None, "progress", member["id"], note, {"order": self.order_info(self.order(row["id"]))})
        self.changed()
        return self.order(row["id"])

    def ask(self, member: Any, ref: Any, text: Any) -> int:
        text = text_field(text, "the question", MAX_WORK_TEXT, line=False, required=True)
        check_secrets(text, "the question")
        with self.db.tx() as con:
            row = self._working(member, ref, "ask about it", ORDER_WORKING + ("reported",))
            open_q = self.db.one("SELECT COUNT(*) AS n FROM questions WHERE order_id = ? AND member = ? AND answer IS"
                                 " NULL", (row["id"], member["id"]))["n"]  # fmt: skip
            if open_q >= 20:
                raise Refused(429, "too_many_questions", "20 questions about this order wait for answers already")
            cur = con.execute("INSERT INTO questions (order_id, member, at, text) VALUES (?, ?, ?, ?)",
                              (row["id"], member["id"], time.time(), text))  # fmt: skip
            con.execute("UPDATE orders SET news_at = ?, updated = ? WHERE id = ?", (time.time(), time.time(), row["id"]))
            qid = int(cur.lastrowid or 0)
            self._log(row["id"], Actor(member), "question", f"#{qid}: {text}")
            self.event("leads", None, "question", member["id"], text, {"question": qid, "order": self.order_info(row)})
        self.changed()
        return qid

    def answer(self, actor: Actor, qid: Any, text: Any) -> sqlite3.Row:
        text = text_field(text, "the answer", MAX_SUMMARY, line=False, required=True)
        check_secrets(text, "the answer")
        qid = int_id(qid, "the question's number")
        with self.db.tx() as con:
            q = self.db.one("SELECT * FROM questions WHERE id = ?", (qid,))
            if q is None:
                raise Refused(404, "no_question", f"no question #{qid}")
            row = self.order(q["order_id"])
            self._manage(actor, row, "answer questions about")
            con.execute("UPDATE questions SET answer = ?, answered_by = ?, answered_at = ? WHERE id = ?",
                        (text, actor.label, time.time(), qid))  # fmt: skip
            self._log(row["id"], actor, "answer", f"#{qid}: {text}")
            self.event("member", q["member"], "answer", actor.id if actor.member else 0, text,
                       {"question": qid, "asked": q["text"][:2000], "order": self.order_info(row), "by": actor.label})  # fmt: skip
        self.changed()
        return self.db.one("SELECT * FROM questions WHERE id = ?", (qid,))

    def report(self, member: Any, ref: Any, summary: Any, files: Any, patch: Any, url: Any) -> int:
        summary = text_field(summary, "the summary", MAX_SUMMARY, line=False, required=True)
        check_secrets(summary, "the summary")
        limit = int(self.settings()["max_report_bytes"])
        row = self.order(ref)
        if row["kind"] == "workspace":
            if patch or url:
                raise Refused(400, "invalid", "a workspace order is reported with the task folder's changed files")
            base = {r["path"]: r["content"] for r in self.db.all("SELECT path, content FROM order_files WHERE"
                                                                 " order_id = ?", (row["id"],))}  # fmt: skip
            changes = self._files(files, limit, "the report", base)
            patch_text, link = "", ""
        else:
            if files:
                raise Refused(400, "invalid", "a repo order is reported with a patch or a pull-request URL")
            patch_text = patch if isinstance(patch, str) else ""
            if patch is not None and not isinstance(patch, str) or "\x00" in patch_text:
                raise Refused(400, "invalid", "the patch must be text")
            if len(patch_text.encode("utf-8")) > limit:
                raise Refused(413, "too_large", f"the patch is larger than {limit:,} bytes (a limit the owner sets)")
            check_secrets(patch_text, "the patch")
            link, changes = pr_url(url), []
        size = sum(len(c.encode("utf-8")) for _p, _k, c in changes) + len(patch_text.encode("utf-8"))
        with self.db.tx() as con:
            row = self._working(member, ref, "report it")
            self._storage_check(size)
            con.execute("UPDATE reports SET status = 'superseded' WHERE order_id = ? AND status = 'pending'", (row["id"],))
            cur = con.execute("INSERT INTO reports (order_id, member, at, summary, patch, pr_url, files, bytes) VALUES"
                              " (?, ?, ?, ?, ?, ?, ?, ?)", (row["id"], member["id"], time.time(), summary, patch_text,
                                                            link, len(changes), size))  # fmt: skip
            report_id = int(cur.lastrowid or 0)
            con.executemany("INSERT INTO report_files VALUES (?, ?, ?, ?)", [(report_id, p, k, c) for p, k, c in changes])
            con.execute("UPDATE orders SET status = 'reported', news_at = ?, updated = ? WHERE id = ?",
                        (time.time(), time.time(), row["id"]))  # fmt: skip
            what = (f"{len(changes)} file(s) changed" if row["kind"] == "workspace" else
                    "a patch" if patch_text else f"a pull request {link}" if link else "no patch")  # fmt: skip
            self._log(row["id"], Actor(member), "report", f"report #{report_id}: {what}")
            self.event("leads", None, "report", member["id"], summary, {"order": self.order_info(self.order(row["id"])),
                                                                         "report": report_id, "what": what})  # fmt: skip
        self.changed()
        return report_id

    def _release(self, row: Any, actor: Actor | None, kind: str, why: str, tell: bool = True) -> None:
        """Put a claimed order back up for claiming (re-queued), telling its claimer and the leads."""
        status = "offered" if self.targets(row) else "queued"
        self.db.con.execute("UPDATE orders SET status = ?, claimer = NULL, claimed_at = NULL, news_at = NULL, announced"
                            " = 0, updated = ? WHERE id = ?", (status, time.time(), row["id"]))  # fmt: skip
        self._log(row["id"], actor, kind, why)
        info = self.order_info(self.order(row["id"]))
        if tell and row["claimer"]:
            self.event("member", row["claimer"], "released", None if actor is None else actor.id if actor.member else 0,
                       why, {"order": info, "how": kind})  # fmt: skip
        self.event("leads", None, "requeued", row["claimer"], why, {"order": info, "how": kind})
        self._announce(self.order(row["id"]))

    def decline(self, member: Any, ref: Any, reason: Any) -> sqlite3.Row:
        reason = text_field(reason, "the reason", MAX_REASON)
        check_secrets(reason, "the reason")
        actor = Actor(member)
        with self.db.tx() as con:
            row = self.order(ref)
            if row["claimer"] == member["id"] and row["status"] in ORDER_WORKING:
                con.execute("INSERT OR REPLACE INTO order_declines VALUES (?, ?, ?, ?)", (row["id"], member["id"],
                                                                                         time.time(), reason))  # fmt: skip
                self.event("leads", None, "declined", member["id"], reason, {"order": self.order_info(row),
                                                                             "gave_back": True})  # fmt: skip
                self._release(row, actor, "declined", f"{member['name']} gave it back"
                              f"{': ' + reason if reason else ''}", tell=False)  # fmt: skip
            elif row["status"] in ORDER_WAITING and self.can_see(actor, row):
                con.execute("INSERT OR REPLACE INTO order_declines VALUES (?, ?, ?, ?)", (row["id"], member["id"],
                                                                                         time.time(), reason))  # fmt: skip
                self._log(row["id"], actor, "declined", reason)
                self.event("leads", None, "declined", member["id"], reason, {"order": self.order_info(row)})
            else:
                raise Refused(409, "wrong_status", f"order #{row['id']} is {STATUS_NAMES[row['status']]}"
                              + (" (you reported it)" if row["claimer"] == member["id"] else "")
                              + ": nothing to decline")  # fmt: skip
        self.changed()
        return self.order(row["id"])

    def edit(self, actor: Actor, ref: Any, spec: dict[str, Any]) -> sqlite3.Row:
        changes: dict[str, Any] = {}
        if spec.get("title") is not None:
            changes["title"] = text_field(spec["title"], "the title", MAX_TITLE, required=True)
        if spec.get("instructions") is not None:
            changes["instructions"] = text_field(spec["instructions"], "the instructions", MAX_INSTRUCTIONS,
                                                 line=False, required=True)  # fmt: skip
        if spec.get("criteria") is not None:
            changes["criteria"] = text_field(spec["criteria"], "the acceptance criteria", MAX_CRITERIA, line=False)
        if spec.get("priority") is not None:
            changes["priority"] = self._priority(spec["priority"])
        if spec.get("deadline_hours") is not None:
            changes["deadline"] = self._deadline(spec["deadline_hours"])
        if spec.get("claim_minutes") is not None:
            changes["claim_minutes"] = self.check_setting("claim_minutes", spec["claim_minutes"])
        if not changes:
            raise Refused(400, "invalid", "nothing to change: give title, instructions, criteria, priority,"
                          " deadline_hours or claim_minutes")  # fmt: skip
        for key in ("title", "instructions", "criteria"):
            if key in changes:
                check_secrets(changes[key], f"the {key}")
        with self.db.tx() as con:
            row = self.order(ref)
            self._manage(actor, row, "edit")
            if row["status"] in ORDER_FINAL:
                raise Refused(409, "wrong_status", f"order #{row['id']} is {STATUS_NAMES[row['status']]}")
            con.execute(f"UPDATE orders SET {', '.join(k + ' = ?' for k in changes)}, updated = ? WHERE id = ?",
                        (*changes.values(), time.time(), row["id"]))  # fmt: skip
            self._log(row["id"], actor, "edited", ", ".join(changes))
            self.audit(actor, "order_edit", None, f"order #{row['id']}: {', '.join(changes)} changed")
            new = self.order(row["id"])
            if not self._recheck_approval(new, actor) and row["claimer"]:
                self.event("member", row["claimer"], "edited", actor.id if actor.member else 0, "",
                           {"order": self.order_info(new), "fields": list(changes), "by": actor.label})  # fmt: skip
            new = self.order(row["id"])
        self.changed()
        return new

    def _recheck_approval(self, row: Any, actor: Actor) -> bool:
        """After a change (inside the caller's transaction): a dispatch approval given for other content is void.
        While dispatch approval is on, the order waits for a human's approval again: its claimer (if any) is released
        and told, a pending report is set aside. True then."""
        if not row["approved_by"] or same(row["approved_digest"], self.digest(row)):
            return False
        con, now = self.db.con, time.time()
        con.execute("UPDATE orders SET approved_by = '', approved_digest = '', updated = ? WHERE id = ?", (now, row["id"]))
        if not self.settings()["dispatch_approval"] or row["status"] in (*ORDER_FINAL, "approval"):
            self._log(row["id"], actor, "approval_void", "it changed after its dispatch approval")
            return False
        why = "it changed after its dispatch approval: it waits for a human's approval again"
        con.execute("UPDATE orders SET status = 'approval', claimer = NULL, claimed_at = NULL, news_at = NULL, announced"
                    " = 0, updated = ? WHERE id = ?", (now, row["id"]))  # fmt: skip
        con.execute("UPDATE reports SET status = 'superseded' WHERE order_id = ? AND status = 'pending'", (row["id"],))
        self._log(row["id"], actor, "void", why)
        self.audit(actor, "order_void", None, f"order #{row['id']}: {why}")
        info = self.order_info(self.order(row["id"]))
        sender = actor.id if actor.member else 0
        if row["claimer"]:
            self.event("member", row["claimer"], "released", sender, why, {"order": info, "how": "void"})
        self.event("leads", None, "order_approval", sender, "", {"order": info, "again": True})
        return True

    def reassign(self, actor: Actor, ref: Any, to: Any) -> sqlite3.Row:
        targets = self._order_targets(to)
        with self.db.tx() as con:
            row = self.order(ref)
            self._manage(actor, row, "reassign")
            if row["status"] in ORDER_FINAL or row["status"] == "reported":
                raise Refused(409, "wrong_status", f"order #{row['id']} is {STATUS_NAMES[row['status']]}: accept or"
                              " reject its result first" if row["status"] == "reported" else
                              f"order #{row['id']} is {STATUS_NAMES[row['status']]}")  # fmt: skip
            ids = [t["id"] for t in targets]
            con.execute("UPDATE orders SET targets = ?, updated = ? WHERE id = ?", (json.dumps(ids), time.time(),
                                                                                   row["id"]))  # fmt: skip
            who = ", ".join(member_label(t) for t in targets) or "anyone"
            self.audit(actor, "order_reassign", None, f"order #{row['id']} reassigned to {who}")
            if ids:  # named again: offered to them again, even if they declined it before
                con.execute(f"DELETE FROM order_declines WHERE order_id = ? AND member IN ({','.join('?' * len(ids))})",
                            (row["id"], *ids))  # fmt: skip
            if row["status"] == "approval":
                self._log(row["id"], actor, "reassigned", f"to {who}")
            elif self._recheck_approval(self.order(row["id"]), actor):
                self._log(row["id"], actor, "reassigned", f"to {who}")
            else:
                self._release(self.order(row["id"]), actor, "reassigned", f"reassigned to {who} by {actor.label}")
                con.execute("UPDATE orders SET status = ? WHERE id = ? AND status IN ('queued', 'offered')",
                            ("offered" if ids else "queued", row["id"]))  # fmt: skip
        self.changed()
        return self.order(row["id"])

    def cancel(self, actor: Actor, ref: Any, reason: Any) -> sqlite3.Row:
        reason = text_field(reason, "the reason", MAX_REASON)
        check_secrets(reason, "the reason")
        with self.db.tx() as con:
            row = self.order(ref)
            self._manage(actor, row, "cancel")
            if row["status"] in ORDER_FINAL:
                raise Refused(409, "wrong_status", f"order #{row['id']} is {STATUS_NAMES[row['status']]} already")
            con.execute("UPDATE orders SET status = 'cancelled', closed_reason = ?, updated = ? WHERE id = ?",
                        (reason or "cancelled", time.time(), row["id"]))  # fmt: skip
            con.execute("UPDATE reports SET status = 'superseded' WHERE order_id = ? AND status = 'pending'", (row["id"],))
            self._log(row["id"], actor, "cancelled", reason)
            self.audit(actor, "order_cancel", None, f"order #{row['id']} cancelled{'; reason: ' + reason if reason else ''}")
            info = self.order_info(self.order(row["id"]))
            if row["claimer"]:
                self.event("member", row["claimer"], "cancelled", actor.id if actor.member else 0, reason,
                           {"order": info, "by": actor.label})  # fmt: skip
            self.event("leads", None, "order_closed", actor.id if actor.member else 0, reason,
                       {"order": info, "how": "cancelled", "by": actor.label})  # fmt: skip
        self.changed()
        return self.order(row["id"])

    def review(self, actor: Actor, ref: Any) -> dict[str, Any]:
        """An order's latest report with its files or patch, and the order's own files (for a diff), for leads."""
        if RANK[actor.role] < RANK["partner"]:
            raise Refused(403, "forbidden", "only the director, a trusted partner or the owner reviews results")
        row = self.order(ref)
        report = self.db.one("SELECT r.*, m.name, m.fingerprint FROM reports r LEFT JOIN members m ON m.id = r.member"
                             " WHERE r.order_id = ? ORDER BY r.id DESC LIMIT 1", (row["id"],))  # fmt: skip
        out: dict[str, Any] = {"order": self.order_info(row, full=True), "report": None,
                               "manage": self.can_manage(actor, row)}  # fmt: skip
        if report is not None:
            out["report"] = {"id": report["id"], "member": report["name"], "fingerprint": report["fingerprint"],
                             "at": report["at"], "summary": report["summary"], "patch": report["patch"],
                             "pr_url": report["pr_url"], "status": report["status"], "reason": report["reason"],
                             "files": [{"path": r["path"], "kind": r["kind"], "content": r["content"]} for r in
                                       self.db.all("SELECT * FROM report_files WHERE report_id = ? ORDER BY path",
                                                   (report["id"],))]}  # fmt: skip
        out["base"] = [{"path": r["path"], "content": r["content"], "sha": r["sha"]} for r in
                       self.db.all("SELECT * FROM order_files WHERE order_id = ? ORDER BY path", (row["id"],))]  # fmt: skip
        return out

    def decide(self, actor: Actor, ref: Any, report_id: Any, accept: bool, reason: Any = "", then: str = "rework"
               ) -> sqlite3.Row:  # fmt: skip
        """Accept a reported result (the order is done) or reject it: back to the same worker to fix (rework), back
        up for anyone (queue), or closed as rejected (close)."""
        reason = text_field(reason, "the reason", MAX_SUMMARY, line=False, required=not accept)
        if then not in ("rework", "queue", "close"):
            raise Refused(400, "invalid", "then: rework, queue or close")
        check_secrets(reason, "the reason")
        with self.db.tx() as con:
            row = self.order(ref)
            self._manage(actor, row, "accept or reject results of")
            report = self.db.one("SELECT * FROM reports WHERE order_id = ? AND status = 'pending' ORDER BY id DESC"
                                 " LIMIT 1", (row["id"],))  # fmt: skip
            if row["status"] != "reported" or report is None:
                raise Refused(409, "wrong_status", f"order #{row['id']} is {STATUS_NAMES[row['status']]}: no result"
                              " waits for a decision")  # fmt: skip
            if report_id not in (None, "") and str(report_id).lstrip("#") != str(report["id"]):
                raise Refused(409, "stale", f"order #{row['id']} has a newer report (#{report['id']}) than the one"
                              " reviewed: review it again")  # fmt: skip
            now, sender = time.time(), actor.id if actor.member else 0
            con.execute("UPDATE reports SET status = ?, reason = ?, decided_by = ?, decided_at = ? WHERE id = ?",
                        ("accepted" if accept else "rejected", reason, actor.label, now, report["id"]))  # fmt: skip
            if accept:
                con.execute("UPDATE orders SET status = 'accepted', closed_reason = ?, updated = ? WHERE id = ?",
                            (reason or "accepted", now, row["id"]))  # fmt: skip
                self._log(row["id"], actor, "accepted", f"report #{report['id']}" + (f": {reason}" if reason else ""))
                self.audit(actor, "result_accept", report["member"], f"order #{row['id']}: report #{report['id']}"
                           " accepted")  # fmt: skip
                self.event("member", report["member"], "accepted", sender, reason,
                           {"order": self.order_info(self.order(row["id"])), "report": report["id"], "by": actor.label})  # fmt: skip
            else:
                self._log(row["id"], actor, "rejected", f"report #{report['id']} ({then}): {reason}")
                self.audit(actor, "result_reject", report["member"], f"order #{row['id']}: report #{report['id']}"
                           f" rejected ({then}); reason: {reason}")  # fmt: skip
                if then == "rework" and row["claimer"]:
                    con.execute("UPDATE orders SET status = 'in_progress', news_at = ?, updated = ? WHERE id = ?",
                                (now, now, row["id"]))  # fmt: skip
                    self.event("member", report["member"], "rework", sender, reason,
                               {"order": self.order_info(self.order(row["id"])), "report": report["id"],
                                "by": actor.label})  # fmt: skip
                elif then == "queue" or (then == "rework" and not row["claimer"]):
                    self.event("member", report["member"], "rejected", sender, reason,
                               {"order": self.order_info(row), "report": report["id"], "by": actor.label,
                                "then": "queue"})  # fmt: skip
                    self._release(row, actor, "requeued", f"result rejected by {actor.label}; back in the queue",
                                  tell=False)  # fmt: skip
                else:
                    con.execute("UPDATE orders SET status = 'rejected', closed_reason = ?, updated = ? WHERE id = ?",
                                (reason, now, row["id"]))  # fmt: skip
                    self.event("member", report["member"], "rejected", sender, reason,
                               {"order": self.order_info(self.order(row["id"])), "report": report["id"],
                                "by": actor.label, "then": "close"})  # fmt: skip
            self.event("leads", None, "decided", sender, reason, {"order": self.order_info(self.order(row["id"])),
                                                                   "report": report["id"], "accepted": accept,
                                                                   "by": actor.label, "then": then})  # fmt: skip
        self.changed()
        return self.order(row["id"])

    def offline_for(self, member_id: int) -> float:
        """How long a member has been offline (0 while online)."""
        row = self.member(member_id)
        if row is None:
            return float("inf")
        presence, seen = self.presence(row)
        return 0.0 if presence != "offline" else time.time() - max(float(seen or 0), self.started)

    def sweep(self) -> bool:
        """Re-queue claims without news for their claim time or whose claimer is offline too long or gone; expire
        orders past their deadline; expire handovers. True if anything changed."""
        now = time.time()
        settings = self.settings()
        changed = False
        starting = now - self.started < self.start_grace  # just started: clients are reconnecting, claims stay
        rows = self.db.all("SELECT * FROM orders WHERE status IN ('claimed', 'in_progress') OR (deadline IS NOT NULL AND"
                           " deadline < ? AND status IN ('approval', 'queued', 'offered', 'claimed', 'in_progress'))",
                           (now,))  # fmt: skip
        for row in rows:
            with self.db.tx():
                row = self.order(row["id"])
                if row["deadline"] is not None and row["deadline"] < now and row["status"] in (
                        "approval", "queued", "offered", "claimed", "in_progress"):  # fmt: skip
                    self.db.con.execute("UPDATE orders SET status = 'expired', closed_reason = 'deadline passed',"
                                        " updated = ? WHERE id = ?", (now, row["id"]))  # fmt: skip
                    self._log(row["id"], None, "expired", "the deadline passed")
                    info = self.order_info(self.order(row["id"]))
                    if row["claimer"]:
                        self.event("member", row["claimer"], "expired", None, "", {"order": info})
                    self.event("leads", None, "order_closed", None, "the deadline passed", {"order": info,
                                                                                           "how": "expired"})  # fmt: skip
                    changed = True
                    continue
                if row["status"] not in ORDER_WORKING or not row["claimer"]:
                    continue
                claimer = self.member(row["claimer"])
                why = ""
                if claimer is None or claimer["role"] not in ("worker", "partner"):
                    why = "its claimer is no longer a worker in the fleet"
                elif starting:
                    continue
                elif self.lapses_at(row) < now:
                    why = f"no news from its claimer for {row['claim_minutes']} minute(s)"
                elif self.offline_for(row["claimer"]) > int(settings["requeue_offline_seconds"]):
                    why = f"its claimer has been offline for more than {settings['requeue_offline_seconds']} seconds"
                if why:
                    self._release(row, None, "requeued", why)
                    changed = True
        for h in self.db.all("SELECT * FROM handovers WHERE status IN ('offered', 'accepted') AND expires < ?", (now,)):
            with self.db.tx() as con:
                con.execute("UPDATE handovers SET status = 'expired', decided_at = ? WHERE id = ? AND status IN"
                            " ('offered', 'accepted')", (now, h["id"]))  # fmt: skip
                self.audit("the server", "handover_expired", h["to_member"], f"handover #{h['id']} expired")
                self.event("leads", None, "handover_closed", None, "", {"handover": h["id"], "how": "expired"})
            changed = True
        if changed:
            self.changed()
        return changed

    def release_claims(self, member_id: int, why: str) -> None:
        """A member left or was removed: its claims go back up (inside the caller's transaction)."""
        for row in self.db.all("SELECT * FROM orders WHERE claimer = ? AND status IN ('claimed', 'in_progress')",
                               (member_id,)):  # fmt: skip
            self._release(row, None, "requeued", why, tell=False)

    # -- trusted partners and handover --------------------------------------------------

    def trust(self, actor: Actor, ref: Any, revoke: bool = False) -> sqlite3.Row:
        """Mark a worker as a trusted partner (the director signs it; the owner may too), or take the trust back."""
        if not (actor.owner or actor.role == "director"):
            raise Refused(403, "forbidden", "only the director (signed with its key) or the owner marks trusted"
                          " partners")  # fmt: skip
        with self.db.tx() as con:
            target = self._target(actor, ref, ("partner",) if revoke else ("worker",),
                                  "take the trust back from" if revoke else "trust")  # fmt: skip
            if revoke:
                con.execute("UPDATE members SET role = 'worker' WHERE id = ?", (target["id"],))
                con.execute("DELETE FROM panel_sessions WHERE member = ?", (target["id"],))
                self._void_handovers("to_member = ?", (target["id"],), actor, f"{member_label(target)} is no longer a"
                                     " trusted partner")  # fmt: skip
            else:
                con.execute("UPDATE members SET role = 'partner', lead_since = ? WHERE id = ?", (self.last_event(),
                                                                                                target["id"]))  # fmt: skip
            role = "worker" if revoke else "partner"
            self.audit(actor, "untrust" if revoke else "trust", target["id"], f"{member_label(target)} is"
                       f" {'no longer a trusted partner' if revoke else 'a trusted partner now'} (fingerprint"
                       f" {target['fingerprint']})")  # fmt: skip
            self.event("member", target["id"], "role", actor.id if actor.member else 0, "", {"role": role,
                                                                                           "by": actor.label})  # fmt: skip
            self.event("leads", None, "trusted", target["id"], "", {"member": self.member_info(self.member(target["id"])),
                                                                     "on": not revoke, "by": actor.label})  # fmt: skip
        self.changed()
        new = self.member(target["id"])
        assert new is not None
        return new

    def handover_offer(self, actor: Actor, ref: Any, signature: str) -> sqlite3.Row:
        if actor.role != "director":
            raise Refused(403, "forbidden", "only the director hands over the director role")
        with self.db.tx() as con:
            target = self.find(ref)
            if target["id"] == actor.id:
                raise Refused(409, "self", "you are the director already")
            if target["role"] != "partner":
                raise Refused(409, "not_trusted", f"{member_label(target)} is {ROLE_NAMES[target['role']]}: hand over"
                              " only to a trusted partner (mark it first, after comparing its fingerprint)")  # fmt: skip
            now = time.time()
            self._void_handovers("from_member = ?", (actor.id,), actor, "replaced by a new offer")
            assert actor.member is not None
            cur = con.execute("INSERT INTO handovers (from_member, to_member, from_fp, to_fp, created, expires, status,"
                              " offer_sig) VALUES (?, ?, ?, ?, ?, ?, 'offered', ?)",
                              (actor.id, target["id"], actor.member["fingerprint"], target["fingerprint"], now,
                               now + HANDOVER_SECONDS, signature))  # fmt: skip
            hid = int(cur.lastrowid or 0)
            self.audit(actor, "handover_offer", target["id"], f"handover #{hid}: {member_label(actor.member)} offers the"
                       f" director role to {member_label(target)} (fingerprint {target['fingerprint']}); signed")  # fmt: skip
            data = {"handover": hid, "from": self.member_info(actor.member), "to": self.member_info(target),
                    "expires": now + HANDOVER_SECONDS,
                    "needs_owner": bool(self.settings()["handover_needs_owner"])}  # fmt: skip
            self.event("member", target["id"], "handover", actor.id, "", data)
            self.event("leads", None, "handover_started", actor.id, "", data)
        self.changed()
        row = self.db.one("SELECT * FROM handovers WHERE id = ?", (hid,))
        assert row is not None
        return row

    def handover_row(self, hid: Any) -> sqlite3.Row:
        hid = int_id(hid, "the handover's number")
        row = self.db.one("SELECT * FROM handovers WHERE id = ?", (hid,))
        if row is None:
            raise Refused(404, "no_handover", f"no handover #{hid}")
        return row

    def handover_accept(self, actor: Actor, hid: Any, signature: str) -> sqlite3.Row:
        """The partner's signed accept. (Every role change that would make it stale voids the handover first, so the
        last check below only refuses.)"""
        with self.db.tx() as con:
            h = self.handover_row(hid)
            if h["to_member"] != actor.id:
                raise Refused(403, "forbidden", f"handover #{h['id']} is not offered to you")
            if h["status"] != "offered" or h["expires"] < time.time():
                raise Refused(409, "wrong_status", f"handover #{h['id']} is {h['status'] if h['expires'] >= time.time() else 'expired'}")
            director = self.director()
            if director is None or director["id"] != h["from_member"] or actor.role != "partner":
                raise Refused(409, "stale", "the roles changed since the offer (another director, or you are no longer"
                              " a trusted partner): the handover is void")  # fmt: skip
            con.execute("UPDATE handovers SET accept_sig = ?, decided_at = ? WHERE id = ?", (signature, time.time(),
                                                                                           h["id"]))  # fmt: skip
            if self.settings()["handover_needs_owner"]:
                con.execute("UPDATE handovers SET status = 'accepted' WHERE id = ?", (h["id"],))
                self.audit(actor, "handover_accept", actor.id, f"handover #{h['id']} accepted (signed); waits for the"
                           " owner's confirmation")  # fmt: skip
                self.event("leads", None, "handover_accepted", actor.id, "", {"handover": h["id"], "needs_owner": True})
            else:
                self.audit(actor, "handover_accept", actor.id, f"handover #{h['id']} accepted (signed)")
                self._handover_done(h, actor.label)
        self.changed()
        return self.handover_row(h["id"])

    def handover_confirm(self, actor: Actor, hid: Any) -> sqlite3.Row:
        if not actor.owner:
            raise Refused(403, "forbidden", "only the owner confirms a handover")
        with self.db.tx():
            h = self.handover_row(hid)
            if h["status"] != "accepted":
                raise Refused(409, "wrong_status", f"handover #{h['id']} is {h['status']}, not waiting for the owner")
            director, partner = self.director(), self.member(h["to_member"])
            if director is None or director["id"] != h["from_member"] or partner is None or partner["role"] != "partner":
                raise Refused(409, "stale", "the roles changed since the offer: the handover is void")
            self._handover_done(h, actor.label)
        self.changed()
        return self.handover_row(h["id"])

    def _handover_done(self, h: Any, by: str) -> None:
        con = self.db.con
        old, new = self.member(h["from_member"]), self.member(h["to_member"])
        assert old is not None and new is not None
        con.execute("UPDATE members SET role = 'partner' WHERE id = ?", (old["id"],))
        con.execute("UPDATE members SET role = 'director' WHERE id = ?", (new["id"],))
        con.execute("UPDATE handovers SET status = 'done', decided_at = ?, decided_by = ? WHERE id = ?",
                    (time.time(), by, h["id"]))  # fmt: skip
        new = self.member(new["id"])
        assert new is not None
        self.audit(by, "handover", new["id"], f"handover #{h['id']}: {member_label(new)} is the director now"
                   f" (fingerprint {new['fingerprint']}); {member_label(old)} is a trusted partner")  # fmt: skip
        self.event("member", old["id"], "role", None, "", {"role": "partner", "by": f"handover #{h['id']}"})
        self.event("member", new["id"], "role", None, "", {"role": "director", "by": f"handover #{h['id']}"})
        self.event("all", None, "director", None, "", {"director": self.member_info(new),
                                                        "by": f"handover from {old['name']}"})  # fmt: skip

    def handover_cancel(self, actor: Actor, hid: Any) -> sqlite3.Row:
        with self.db.tx():
            h = self.handover_row(hid)
            if not (actor.owner or actor.id in (h["from_member"], h["to_member"]) or actor.role == "director"):
                raise Refused(403, "forbidden", "only the director, the partner it is offered to or the owner cancels"
                              " a handover")  # fmt: skip
            if h["status"] not in ("offered", "accepted"):
                raise Refused(409, "wrong_status", f"handover #{h['id']} is {h['status']}")
            why = "declined by the partner" if actor.id == h["to_member"] and actor.member else "cancelled"
            self._void_handovers("id = ?", (h["id"],), actor, why)
        self.changed()
        return self.handover_row(h["id"])

    def _void_handovers(self, where: str, args: tuple[Any, ...], by: Actor | str, why: str) -> None:
        """Cancel the open handovers `where` names (inside the caller's transaction): each one audited and announced
        once: to the leads, and to the partner it was offered to only when that one is no longer a lead (a lead hears
        it with the leads)."""
        label = by.label if isinstance(by, Actor) else by
        sender = (by.id if by.member else 0) if isinstance(by, Actor) else None
        for h in self.db.all(f"SELECT * FROM handovers WHERE status IN ('offered', 'accepted') AND ({where})", args):
            self.db.con.execute("UPDATE handovers SET status = 'cancelled', decided_at = ?, decided_by = ? WHERE id = ?",
                                (time.time(), label, h["id"]))  # fmt: skip
            self.audit(by, "handover_cancel", h["to_member"], f"handover #{h['id']} cancelled: {why}")
            data = {"handover": h["id"], "how": "cancelled", "by": label, "why": why}
            self.event("leads", None, "handover_closed", sender, "", data)
            partner = self.member(h["to_member"])
            if partner is not None and partner["role"] in ACTIVE and partner["role"] not in LEADS:
                self.event("member", partner["id"], "handover_closed", sender, "", data)

    def handovers(self, open_only: bool = True) -> list[dict[str, Any]]:
        rows = self.db.all("SELECT h.*, a.name AS a_name, b.name AS b_name FROM handovers h LEFT JOIN members a ON a.id ="
                           " h.from_member LEFT JOIN members b ON b.id = h.to_member" + (" WHERE h.status IN"
                           " ('offered', 'accepted')" if open_only else "") + " ORDER BY h.id DESC LIMIT 100")  # fmt: skip
        return [{"id": r["id"], "from": r["a_name"], "to": r["b_name"], "from_fp": r["from_fp"], "to_fp": r["to_fp"],
                 "status": r["status"], "created": r["created"], "expires": r["expires"],
                 "decided_by": r["decided_by"]} for r in rows]  # fmt: skip

    # -- events for a member ---------------------------------------------------

    def events_for(self, member: Any, after: int, limit: int = EVENTS_PER_POLL) -> list[dict[str, Any]]:
        """What a member is told: to it, to all, to the leads (a lead), to the workers (a worker or partner: not the
        orders it declined). A lead gets no "taken": its "claimed" says it (one line for one claim)."""
        role = member["role"]
        rows = self.db.all(
            "SELECT e.*, m.name AS s_name, m.role AS s_role, m.fingerprint AS s_fp FROM events e LEFT JOIN members m"
            " ON m.id = e.sender AND e.sender > 0 WHERE e.id > ? AND ((e.audience = 'member' AND e.member = ?)"
            " OR (e.audience = 'all' AND ? AND e.id > ?) OR (e.audience = 'leads' AND ? AND e.id > ?)"
            " OR (e.audience = 'workers' AND ? AND e.id > ? AND NOT EXISTS (SELECT 1 FROM order_declines d WHERE"
            " d.order_id = e.order_id AND d.member = ?))) AND NOT (e.kind = 'taken' AND ?) ORDER BY e.id LIMIT ?",
            (after, member["id"], role in MEMBERS, member["since"], role in LEADS,
             max(member["since"], member["lead_since"]), role in ("worker", "partner"), member["since"], member["id"],
             role in LEADS, limit))  # fmt: skip
        out = []
        for r in rows:
            if r["sender"] is None:
                sender = None
            elif r["sender"] == 0:
                sender = {"owner": True, "name": "the owner"}
            else:
                sender = {"id": r["sender"], "name": r["s_name"], "role": r["s_role"], "fingerprint": r["s_fp"]}
            try:
                data = json.loads(r["data"] or "{}")
            except ValueError:
                data = {}
            out.append({"id": r["id"], "at": r["at"], "kind": r["kind"], "from": sender, "text": r["text"],
                        "to": "you" if r["audience"] == "member" else r["audience"], "data": data})  # fmt: skip
        return out

    def prune(self) -> None:
        """Old events, tokens, sign-in links; and the files of orders finished more than EVENT_KEEP_DAYS ago (their
        history, questions and report summaries stay)."""
        now = time.time()
        with self.db.tx() as con:
            done = (f"SELECT id FROM orders WHERE status IN ({','.join(repr(x) for x in ORDER_FINAL)}) AND updated < ?"
                    " AND (bytes > 0 OR id IN (SELECT order_id FROM reports WHERE bytes > 0))")  # fmt: skip
            old = [r["id"] for r in self.db.all(done, (now - EVENT_KEEP_DAYS * 86400,))]
            for i in range(0, len(old), 500):
                ids = old[i:i + 500]
                marks = ",".join("?" * len(ids))
                con.execute(f"DELETE FROM order_files WHERE order_id IN ({marks})", ids)
                con.execute(f"DELETE FROM report_files WHERE report_id IN (SELECT id FROM reports WHERE order_id IN"
                            f" ({marks}))", ids)  # fmt: skip
                con.execute(f"UPDATE reports SET patch = '', bytes = 0 WHERE order_id IN ({marks})", ids)
                con.execute(f"UPDATE orders SET bytes = 0 WHERE id IN ({marks})", ids)
            con.execute("DELETE FROM events WHERE at < ?", (now - EVENT_KEEP_DAYS * 86400,))
            con.execute("DELETE FROM tokens WHERE expires < ?", (now - 86400,))
            con.execute("DELETE FROM joins WHERE at < ?", (now - 86400,))
            con.execute("DELETE FROM panel_links WHERE expires < ?", (now,))
            con.execute("DELETE FROM panel_sessions WHERE expires < ?", (now,))

    # -- the panel's one-time links ----------------------------------------------

    def panel_link(self, member_id: int | None) -> str:
        token = secrets.token_urlsafe(32)
        with self.db.tx() as con:
            con.execute("INSERT INTO panel_links VALUES (?, ?, ?, ?)", (sha(token), member_id, time.time(),
                                                                       time.time() + PANEL_LINK_SECONDS))  # fmt: skip
        return token


# --------------------------------------------------------------------------
# Rate limits
# --------------------------------------------------------------------------


class Buckets:
    """Token buckets: `rate` requests per minute with bursts of up to `burst`, per key (an address or a member)."""

    def __init__(self, rate: float, burst: float) -> None:
        self.rate, self.burst = rate / 60.0, burst
        self.state: dict[Any, tuple[float, float]] = {}

    def take(self, key: Any, cost: float = 1.0) -> int:
        """0 if allowed, else the seconds to wait."""
        now = time.monotonic()
        level, at = self.state.get(key, (self.burst, now))
        level = min(self.burst, level + (now - at) * self.rate)
        if level < cost:
            self.state[key] = (level, now)
            return max(1, int((cost - level) / self.rate) + 1) if self.rate > 0 else 3600
        self.state[key] = (level - cost, now)
        if len(self.state) > 50_000:  # forget the full ones
            self.state = {k: v for k, v in self.state.items() if v[0] < self.burst}
        return 0


# --------------------------------------------------------------------------
# The running server
# --------------------------------------------------------------------------


@dataclass
class Options:
    host: str = "127.0.0.1"
    port: int = DEFAULT_PORT
    public_url: str = ""
    tls_cert: str = ""
    tls_key: str = ""
    behind_proxy: bool = False
    trusted_proxies: tuple[str, ...] = ("127.0.0.1", "::1")
    token_seconds: int = 900
    offline_after: int = 20
    auth_per_minute: int = 20
    api_per_minute: int = 600
    messages_per_minute: int = 30
    start_grace: int = START_GRACE


@dataclass
class Challenge:
    key: str
    purpose: str
    nonce: str
    expires: float
    origin: str
    address: str = ""  # who asked for it: its use (the login, a panel link) is not charged again


def address_group(address: str) -> str:
    """Whose share of the challenge store a request counts against: its address; for IPv6 its /64 (one network,
    which can use any of 2^64 addresses)."""
    with contextlib.suppress(ValueError):
        ip = ipaddress.ip_address(address)
        if ip.version == 6 and ip.ipv4_mapped is None:
            return str(ipaddress.ip_network(f"{ip}/64", strict=False))
    return address


class Challenges:
    """The outstanding login challenges. Nobody can block everyone's logins by filling the store: one address (an
    IPv6 /64) holds at most CHALLENGES_PER_ADDRESS (a new one replaces its oldest), each lives CHALLENGE_SECONDS, and
    when CHALLENGES_MAX are outstanding the oldest challenge of the address holding the most goes, never a refusal."""

    def __init__(self) -> None:
        self.items: dict[str, Challenge] = {}  # in the order given (so in the order they expire)
        self.held: dict[str, list[str]] = {}  # address group -> its challenges, oldest first

    def __len__(self) -> int:
        return len(self.items)

    def get(self, cid: str) -> Challenge | None:
        return self.items.get(cid)

    def pop(self, cid: str) -> Challenge | None:
        challenge = self.items.pop(cid, None)
        if challenge is not None:
            group = address_group(challenge.address)
            ids = self.held.get(group, [])
            with contextlib.suppress(ValueError):
                ids.remove(cid)
            if not ids:
                self.held.pop(group, None)
        return challenge

    def expire(self, now: float) -> None:
        while self.items:
            cid, challenge = next(iter(self.items.items()))
            if challenge.expires >= now:
                break
            self.pop(cid)

    def add(self, cid: str, challenge: Challenge) -> None:
        self.expire(time.time())
        group = address_group(challenge.address)
        while len(self.held.get(group, [])) >= CHALLENGES_PER_ADDRESS:
            self.pop(self.held[group][0])
        while len(self.items) >= CHALLENGES_MAX:
            most = max(self.held, key=lambda g: len(self.held[g]))
            self.pop(self.held[most][0])
        self.items[cid] = challenge
        self.held.setdefault(group, []).append(cid)


@dataclass
class Live:
    """A member's presence, kept in memory (written to the database every few seconds)."""

    seen: float = 0.0
    state: str = "idle"
    polls: int = 0


class Server(Fleet):
    def __init__(self, home: Path, conf: dict[str, Any], options: Options) -> None:
        super().__init__(home, conf)
        self.options = options
        self.secure = options.public_url.startswith("https://")
        self.public = urlsplit(options.public_url)
        loopback = [f"127.0.0.1:{options.port}", f"localhost:{options.port}", f"[::1]:{options.port}"]
        self.hosts = {self.public.netloc.lower(), *(loopback if is_loopback(options.host) else [])}
        # The origins a member may sign for: this server's public origin (and, for a server whose public URL is on
        # this computer, the other names of the loopback address). Never another address: a signature that names
        # the address of a relay is refused.
        self.origin = origin_of(options.public_url)
        self.origins = {self.origin}
        if is_loopback(self.public.hostname or ""):
            port = f":{self.public.port}" if self.public.port else ""
            self.origins |= {f"{self.public.scheme}://{h}{port}" for h in ("127.0.0.1", "localhost", "[::1]")}
        self.started = time.time()
        self.start_grace = options.start_grace
        self.wake = asyncio.Event()
        self.challenges = Challenges()
        self.live: dict[int, Live] = {}
        # A login is charged once, at its challenge (its use is not: no refusal can fall between the two); the proof a
        # member that kept its token asks for after a lost connection has a generous limit of its own (a restart
        # reconnects many members behind one address at once).
        self.auth_limit = Buckets(options.auth_per_minute, max(5, options.auth_per_minute // 2))
        self.proof_limit = Buckets(max(120, options.auth_per_minute * 6), max(60, options.auth_per_minute * 3))
        self.api_limit = Buckets(options.api_per_minute, max(20, options.api_per_minute // 3))
        self.message_limit = Buckets(options.messages_per_minute, max(5, options.messages_per_minute // 3))
        self.page_limit = Buckets(120, 60)
        self.flashes: dict[str, list[tuple[str, str]]] = {}
        self.version = self.db.data_version()
        # Presence written to the database before a restart (shown as it was; the time the server was down never
        # counts as a member's offline time, see offline_for).
        for row in self.db.all("SELECT id, last_seen, state FROM members WHERE last_seen IS NOT NULL"):
            self.live[row["id"]] = Live(seen=row["last_seen"] or 0.0, state=row["state"] or "idle")

    def changed(self) -> None:
        self.wake.set()
        self.wake = asyncio.Event()
        self.version = self.db.data_version()

    def tokens_revoked(self, member_id: int) -> None:
        self.changed()

    def presence(self, row: Any) -> tuple[str, float | None]:
        live = self.live.get(row["id"])
        if live is None or live.state == "offline":  # never seen, or its client said goodbye
            return "offline", live.seen if live is not None else row["last_seen"]
        if live.polls > 0 or time.time() - live.seen <= self.options.offline_after:
            return {"waiting": "online", "working": "working", "idle": "idle"}.get(live.state, "online"), live.seen
        return "offline", live.seen

    def touch(self, member_id: int, state: str | None = None) -> None:
        live = self.live.setdefault(member_id, Live())
        live.seen = time.time()
        if state in STATES:  # "offline": the client says goodbye (its last poll may still be open)
            live.state = state

    def flush_presence(self) -> None:
        rows = [(live.seen or None, live.state, member_id) for member_id, live in self.live.items()]
        if rows:
            with self.db.tx() as con:
                con.executemany("UPDATE members SET last_seen = ?, state = ? WHERE id = ?", rows)

    async def background(self) -> None:
        """Wake the long-polls when another process (the owner's command line) changed the database; write
        presence; re-queue stale claims and expire orders (every second); prune."""
        last_flush = last_prune = last_sweep = time.monotonic()
        while True:
            await asyncio.sleep(0.25)
            try:
                version = self.db.data_version()
                if version != self.version:
                    self.version = version
                    self.wake.set()
                    self.wake = asyncio.Event()
                now = time.monotonic()
                if now - last_flush > 5:
                    last_flush = now
                    self.flush_presence()
                    self.version = self.db.data_version()
                if now - last_sweep > 1:
                    last_sweep = now
                    self.sweep()
                if now - last_prune > 3600:
                    last_prune = now
                    self.prune()
                    self.challenges.expire(time.time())
            except sqlite3.Error as exc:
                log(f"background: {exc}")


def log(message: str) -> None:
    print(f"[hub-server] {clean(message)}", file=sys.stderr, flush=True)


def is_loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host.strip("[]")).is_loopback
    except ValueError:
        return False


# --------------------------------------------------------------------------
# HTTP: the API
# --------------------------------------------------------------------------

JSON_HEADERS = {"X-Content-Type-Options": "nosniff", "Cache-Control": "no-store", "X-Frame-Options": "DENY",
                "Referrer-Policy": "no-referrer", "Content-Security-Policy": "default-src 'none'; frame-ancestors"
                " 'none'"}  # fmt: skip


@dataclass
class Ctx:
    server: Server
    request: Any
    address: str
    body: dict[str, Any] = field(default_factory=dict)
    query: dict[str, str] = field(default_factory=dict)
    member: sqlite3.Row | None = None
    token_hash: str = ""

    @property
    def actor(self) -> Actor:
        assert self.member is not None
        return Actor(self.member, self.address)


def client_address(server: Server, request: Any) -> str:
    peer = request.client.host if request.client else ""
    if server.options.behind_proxy and peer in server.options.trusted_proxies:
        forwarded = request.headers.get("x-forwarded-for", "")
        last = forwarded.split(",")[-1].strip() if forwarded else ""
        with contextlib.suppress(ValueError):
            return str(ipaddress.ip_address(last))
    return peer


async def read_body(request: Any, limit: int = MAX_BODY) -> bytes:
    length = request.headers.get("content-length")
    if length is not None:
        try:
            if int(length) > limit:
                raise Refused(413, "too_large", f"the request is larger than {limit:,} bytes")
        except ValueError:
            raise Refused(400, "invalid", "bad Content-Length") from None
    data = b""
    async for chunk in request.stream():
        data += chunk
        if len(data) > limit:
            raise Refused(413, "too_large", f"the request is larger than {limit:,} bytes")
    return data


PUBLIC = None
ANY = ACTIVE  # every member that may log in, the lobby included
ADMITTED = MEMBERS
WORKERS = ("partner", "worker")  # those who claim and do orders


async def api_info(ctx: Ctx) -> dict[str, Any]:
    server = ctx.server
    settings = server.settings()
    return {"server_id": server.server_id, "name": server.conf.get("name", ""), "version": __version__,
            "protocol": PROTOCOL, "join_mode": settings["join_mode"], "approval": bool(settings["approval"]),
            "server_key": server.key_b64, "fingerprint": server.key_fp, "phrase": phrase(server.key_raw),
            "origin": server.origin}  # fmt: skip


async def api_challenge(ctx: Ctx) -> dict[str, Any]:
    server = ctx.server
    key = ctx.body.get("key")
    if key_bytes(key) is None:
        raise Refused(400, "invalid", "key must be an Ed25519 public key (standard base64 of 32 bytes)")
    purpose = ctx.body.get("purpose", "login")
    if purpose not in ("login", "panel", *SIGNED_PURPOSES):
        raise Refused(400, "invalid", f"purpose must be one of login, panel, {', '.join(SIGNED_PURPOSES)}")
    origin = ctx.body.get("origin")
    if not isinstance(origin, str) or not origin:
        raise Refused(400, "invalid", "origin: the scheme://host[:port] you connect to (an agent_hub.py older than"
                      " fleet protocol 2? update it)")  # fmt: skip
    if origin not in server.origins:
        raise Refused(421, "wrong_origin", f"this fleet server is {server.origin}, not {one_line(origin, 100)}: a"
                      " member signs only for the address it connects to, so a relay at another address gets nothing"
                      " it can use here")  # fmt: skip
    challenge_id, nonce = secrets.token_urlsafe(16), secrets.token_urlsafe(32)
    server.challenges.add(challenge_id, Challenge(key, purpose, nonce, time.time() + CHALLENGE_SECONDS, origin,
                                                  ctx.address))  # fmt: skip
    signature = server.key.sign(challenge_text(server.server_id, origin, challenge_id, nonce, purpose, key))
    return {"challenge": challenge_id, "nonce": nonce, "server_id": server.server_id, "origin": origin,
            "server_key": server.key_b64, "signature": base64.b64encode(signature).decode(),
            "expires_in": CHALLENGE_SECONDS, "now": int(time.time())}  # fmt: skip


async def api_proof(ctx: Ctx) -> dict[str, Any]:
    """A member that kept its access token over a lost connection checks, before it sends the token again, that this
    server still holds the key it pinned, at the address it connects to: this server signs the member's fresh nonce
    with its key (no login, no token: cheap, with a limit of its own)."""
    server = ctx.server
    nonce, origin = ctx.body.get("nonce"), ctx.body.get("origin")
    if not isinstance(nonce, str) or not re.fullmatch(r"[A-Za-z0-9_-]{16,128}", nonce):
        raise Refused(400, "invalid", "nonce: 16 to 128 characters of base64url")
    if not isinstance(origin, str) or origin not in server.origins:
        raise Refused(421, "wrong_origin", f"this fleet server is {server.origin}, not {one_line(origin, 100)}")
    signature = server.key.sign(proof_text(server.server_id, origin, nonce))
    return {"server_id": server.server_id, "origin": origin, "signature": base64.b64encode(signature).decode()}


def check_signature(ctx: Ctx, purpose: str, detail: str = "") -> tuple[bytes, str]:
    """The signed answer to a challenge: single use, for this key and purpose, in time. The key's raw bytes.
    `detail`: what the action is about (a member's fingerprint, an order), signed with the purpose ("trust <fp>"),
    so a signature for one action can never be used for another. The signed text also names this server's key and
    the origin the member connected to (checked when the challenge was given out)."""
    server, body = ctx.server, ctx.body
    challenge_id, key = body.get("challenge"), body.get("key")
    challenge = server.challenges.pop(challenge_id) if isinstance(challenge_id, str) else None
    if challenge is None or challenge.expires < time.time():
        raise Refused(401, "bad_challenge", "unknown, used or expired challenge; ask for a new one")
    raw = key_bytes(key)
    if raw is None or not same(key, challenge.key) or challenge.purpose != purpose:
        raise Refused(401, "bad_challenge", "the challenge was issued for another key or purpose")
    timestamp = body.get("timestamp")
    if not isinstance(timestamp, int) or abs(timestamp - time.time()) > CLOCK_SKEW:
        raise Refused(401, "bad_time", "the signed time is too far from the server's clock; check this computer's"
                      " clock")  # fmt: skip
    message = signed_text(f"{purpose} {detail}".strip(), server.server_id, server.key_fp, challenge.origin,
                          challenge_id, challenge.nonce, timestamp, key)  # fmt: skip
    if not verify(raw, message, body.get("signature")):
        raise Refused(401, "bad_signature", "the signature does not match the key")
    return raw, key


async def api_login(ctx: Ctx) -> dict[str, Any]:
    server = ctx.server
    raw, key = check_signature(ctx, "login")
    row = server.db.one("SELECT * FROM members WHERE key = ?", (key,))
    if row is not None and row["role"] == "banned":
        raise Refused(403, "banned", "this key is banned from this fleet")
    if row is None or row["role"] not in ACTIVE:
        if not ctx.body.get("join"):
            reason = row["removed_reason"] if row is not None else ""
            raise Refused(403, "not_member", f"this key is not a member of this fleet{' (' + reason + ')' if reason else ''}"
                          "; join again with a name")  # fmt: skip
        row = server.join(raw, key, ctx.body.get("join"), ctx.address)
    with server.db.tx():
        token, expires = server.issue_token(row["id"], server.options.token_seconds)
    server.touch(row["id"])
    return {"token": token, "expires_in": int(expires - time.time()), "cursor": max(row["cursor"], row["since"]),
            **server.snapshot(row)}  # fmt: skip


async def api_logout(ctx: Ctx) -> dict[str, Any]:
    with ctx.server.db.tx() as con:
        con.execute("UPDATE tokens SET revoked = ? WHERE hash = ?", (time.time(), ctx.token_hash))
    assert ctx.member is not None
    ctx.server.touch(ctx.member["id"], "offline")
    return {"ok": True}


async def api_panel_link(ctx: Ctx) -> dict[str, Any]:
    server = ctx.server
    _raw, key = check_signature(ctx, "panel")
    row = server.db.one("SELECT * FROM members WHERE key = ?", (key,))
    if row is None or row["role"] not in LEADS:
        raise Refused(403, "forbidden", "the panel is for the director and trusted partners")
    token = server.panel_link(row["id"])
    with server.db.tx():
        server.audit(Actor(row, ctx.address), "panel_link", row["id"], "a one-time panel link was made (signed)")
    return {"link": f"{server.options.public_url}/panel/login?t={token}", "expires_in": PANEL_LINK_SECONDS}


def _ack(ctx: Ctx) -> None:
    member = ctx.member
    assert member is not None
    raw = ctx.query.get("ack") or ctx.body.get("ack")
    ack = int_id(raw, "ack (an event id)", 0) if raw not in (None, "") else -1
    if ack > member["cursor"]:
        ack = min(ack, ctx.server.last_event())
        with ctx.server.db.tx() as con:
            con.execute("UPDATE members SET cursor = MAX(cursor, ?) WHERE id = ?", (ack, member["id"]))


async def api_events(ctx: Ctx) -> dict[str, Any]:
    """The long-poll: returns as soon as there is something for this member, or after `wait` seconds."""
    server = ctx.server
    member = ctx.member
    assert member is not None
    after = int_id(ctx.query.get("after") or 0, "after (an event id)", 0)
    try:
        wait = max(0.0, min(float(ctx.query.get("wait") or 0), POLL_MAX))
    except ValueError:
        raise Refused(400, "invalid", "wait must be a number of seconds") from None
    _ack(ctx)
    state = ctx.query.get("state")
    live = server.live.setdefault(member["id"], Live())
    if live.polls >= POLLS_PER_MEMBER:
        raise Refused(429, "too_many_polls", "too many open polls for this member", retry_after=5)
    server.touch(member["id"], state)
    deadline = time.monotonic() + wait
    live.polls += 1
    gone = asyncio.ensure_future(client_gone(ctx.request)) if wait > 0 else None
    try:
        while True:
            wake = server.wake
            row = server.member(member["id"])
            token = server.db.one("SELECT revoked, expires FROM tokens WHERE hash = ?", (ctx.token_hash,))
            if row is None or row["role"] not in ACTIVE or token is None or token["revoked"]:
                refuse_member(row)
                raise Refused(401, "bad_token", "the access token was revoked; log in again")
            events = server.events_for(row, after)
            remaining = deadline - time.monotonic()
            if events or remaining <= 0:
                return {"events": events, "cursor": events[-1]["id"] if events else after, **server.snapshot(row)}
            woken = asyncio.ensure_future(wake.wait())
            assert gone is not None
            await asyncio.wait({woken, gone}, timeout=min(remaining, 5.0), return_when=asyncio.FIRST_COMPLETED)
            woken.cancel()
            if gone.done():  # the client went away: its slot is free at once, and it no longer counts as seen
                return {"events": [], "cursor": after}
            server.touch(member["id"])
    finally:
        live.polls -= 1
        dropped = gone is not None and gone.done() and not gone.cancelled()
        if gone is not None:
            gone.cancel()
        if not dropped:
            server.touch(member["id"])


async def client_gone(request: Any) -> bool:
    """Wait until the client of a GET request has gone away (http.disconnect); True then."""
    while True:
        message = await request.receive()
        if message.get("type") == "http.disconnect":
            return True


def refuse_member(row: Any) -> None:
    """Refuse a key that is no longer a member, telling it why."""
    if row is None:
        raise Refused(403, "not_member", "this key is not a member of this fleet")
    if row["role"] == "banned":
        raise Refused(403, "banned", f"this key is banned from this fleet ({row['removed_reason'] or 'banned'})")
    if row["role"] == "removed":
        raise Refused(403, "removed", f"no longer a member of this fleet: {row['removed_reason'] or 'removed'}")


async def api_heartbeat(ctx: Ctx) -> dict[str, Any]:
    assert ctx.member is not None
    state = ctx.body.get("state")
    if state is not None and state not in STATES:
        raise Refused(400, "invalid", f"state must be one of {', '.join(STATES)}")
    _ack(ctx)
    ctx.server.touch(ctx.member["id"], state)
    return {"ok": True, **ctx.server.snapshot(ctx.member)}


async def api_status(ctx: Ctx) -> dict[str, Any]:
    assert ctx.member is not None
    ctx.server.touch(ctx.member["id"])
    return ctx.server.snapshot(ctx.member)


async def api_leave(ctx: Ctx) -> dict[str, Any]:
    ctx.server.leave(ctx.member)
    return {"ok": True}


async def api_message(ctx: Ctx) -> dict[str, Any]:
    assert ctx.member is not None
    wait = ctx.server.message_limit.take(ctx.member["id"])
    if wait:
        raise Refused(429, "rate_limited", "too many messages; slow down", retry_after=wait)
    event_id, to = ctx.server.send(ctx.actor, ctx.body.get("to"), ctx.body.get("text"))
    return {"id": event_id, "to": to}


async def api_members(ctx: Ctx) -> dict[str, Any]:
    server = ctx.server
    roles = ("director", "partner", "worker", "lobby", "banned") if ctx.query.get("all") in ("1", "true") else ACTIVE
    marks = ",".join("?" * len(roles))
    rows = server.db.all(f"SELECT * FROM members WHERE role IN ({marks}) ORDER BY CASE role WHEN 'director' THEN 0"
                         " WHEN 'partner' THEN 1 WHEN 'worker' THEN 2 WHEN 'lobby' THEN 3 ELSE 4 END, name", roles)  # fmt: skip
    return {"members": [server.member_info(r, full=True) for r in rows]}


async def api_lobby(ctx: Ctx) -> dict[str, Any]:
    server = ctx.server
    rows = server.db.all("SELECT * FROM members WHERE role = 'lobby' ORDER BY joined_at")
    return {"lobby": [server.member_info(r, full=True) for r in rows]}


def _member_action(how: str) -> Callable[[Ctx], Any]:
    async def act(ctx: Ctx) -> dict[str, Any]:
        server, ref = ctx.server, ctx.body.get("member")
        if how == "admit":
            row = server.admit(ctx.actor, ref)
        elif how == "unban":
            row = server.unban(ctx.actor, ref)
        else:
            row = server.remove(ctx.actor, ref, how, ctx.body.get("reason") or "")
        return {"ok": True, "member": server.member_info(row)}

    return act


async def api_invites(ctx: Ctx) -> dict[str, Any]:
    return {"invites": ctx.server.invites()}


async def api_invite_create(ctx: Ctx) -> dict[str, Any]:
    server = ctx.server
    token, row = server.create_invite(ctx.actor, ctx.body.get("uses", 1), ctx.body.get("hours", 24),
                                      ctx.body.get("note", ""))  # fmt: skip
    return {"id": row["id"], "token": token, "link": invite_link(server.options.public_url, token, server.key_fp),
            "expires": row["expires"], "max_uses": row["max_uses"]}  # fmt: skip


async def api_invite_revoke(ctx: Ctx) -> dict[str, Any]:
    row = ctx.server.revoke_invite(ctx.actor, ctx.body.get("id"))
    return {"ok": True, "id": row["id"]}


SHOWN_SETTINGS = (*LIVE_SETTINGS, *HUMAN_SETTINGS, "handover_needs_owner")


async def api_settings(ctx: Ctx) -> dict[str, Any]:
    settings = ctx.server.settings()
    return {"settings": {k: settings[k] for k in SHOWN_SETTINGS}}


async def api_settings_set(ctx: Ctx) -> dict[str, Any]:
    """The live switches a lead's agent may change; dispatch approval and the takeover are for humans (the panel,
    the owner), the limits for the owner."""
    unknown = set(ctx.body) - set(SETTING_DEFAULTS)
    if unknown:
        raise Refused(400, "invalid", f"unknown setting(s): {', '.join(sorted(map(str, unknown)))[:200]}")
    changes = {k: v for k, v in ctx.body.items() if v is not None}
    done = ctx.server.change_settings(ctx.actor, changes)
    settings = ctx.server.settings()
    return {"changed": done, "settings": {k: settings[k] for k in SHOWN_SETTINGS}}


async def api_audit(ctx: Ctx) -> dict[str, Any]:
    try:
        limit = max(1, min(int(ctx.query.get("limit") or 50), 500))
    except ValueError:
        raise Refused(400, "invalid", "limit must be a number") from None
    rows = ctx.server.db.all("SELECT * FROM audit ORDER BY id DESC LIMIT ?", (limit,))
    return {"audit": [{"id": r["id"], "at": r["at"], "actor": r["actor"], "action": r["action"],
                       "target": r["target"], "detail": r["detail"]} for r in rows]}  # fmt: skip


def _signed_by_member(ctx: Ctx, purpose: str, detail: str) -> str:
    """A signed action of the logged-in member: the signature must be by the token's own key, for this purpose and
    detail. Returns what is kept as the record (signature and signed purpose)."""
    assert ctx.member is not None
    if not isinstance(ctx.body.get("key"), str) or not same(ctx.body["key"], ctx.member["key"]):
        raise Refused(401, "bad_signature", "this action must be signed with your own key")
    check_signature(ctx, purpose, detail)
    return json.dumps({"purpose": f"{purpose} {detail}".strip(), "signature": ctx.body.get("signature"),
                       "timestamp": ctx.body.get("timestamp"), "key": ctx.body.get("key")})  # fmt: skip


def _limit(ctx: Ctx) -> int:
    try:
        return max(1, min(int(ctx.query.get("limit") or 100), 500))
    except ValueError:
        raise Refused(400, "invalid", "limit must be a number") from None


async def api_orders(ctx: Ctx) -> dict[str, Any]:
    return {"orders": ctx.server.orders(ctx.actor, ctx.query.get("status"), ctx.query.get("mine") in ("1", "true"),
                                        _limit(ctx))}  # fmt: skip


async def api_order_create(ctx: Ctx) -> dict[str, Any]:
    assert ctx.member is not None
    wait = ctx.server.message_limit.take(("orders", ctx.member["id"]))
    if wait:
        raise Refused(429, "rate_limited", "too many orders; slow down", retry_after=wait)
    rows = ctx.server.create_orders(ctx.actor, ctx.body)
    return {"orders": [ctx.server.order_info(r) for r in rows]}


async def api_order_get(ctx: Ctx) -> dict[str, Any]:
    return {"order": ctx.server.order_detail(ctx.actor, ctx.query.get("id")), "now": time.time()}


async def api_order_files(ctx: Ctx) -> dict[str, Any]:
    row, files = ctx.server.order_files(ctx.actor, ctx.query.get("id"))
    return {"order": ctx.server.order_info(row), "files": files}


async def api_order_claim(ctx: Ctx) -> dict[str, Any]:
    row = ctx.server.claim(ctx.member, ctx.body.get("id"))
    return {"order": ctx.server.order_detail(ctx.actor, row["id"])}


async def api_order_progress(ctx: Ctx) -> dict[str, Any]:
    row = ctx.server.progress(ctx.member, ctx.body.get("id"), ctx.body.get("note"))
    return {"order": ctx.server.order_info(row)}


async def api_order_ask(ctx: Ctx) -> dict[str, Any]:
    assert ctx.member is not None
    wait = ctx.server.message_limit.take(ctx.member["id"])
    if wait:
        raise Refused(429, "rate_limited", "too many questions; slow down", retry_after=wait)
    return {"question": ctx.server.ask(ctx.member, ctx.body.get("id"), ctx.body.get("question"))}


async def api_order_report(ctx: Ctx) -> dict[str, Any]:
    assert ctx.member is not None
    wait = ctx.server.message_limit.take(("report", ctx.member["id"]))
    if wait:
        raise Refused(429, "rate_limited", "too many reports; slow down", retry_after=wait)
    body = ctx.body
    report_id = ctx.server.report(ctx.member, body.get("id"), body.get("summary"), body.get("files"),
                                  body.get("patch"), body.get("pr_url"))  # fmt: skip
    return {"report": report_id, "order": ctx.server.order_info(ctx.server.order(body.get("id")))}


async def api_order_decline(ctx: Ctx) -> dict[str, Any]:
    row = ctx.server.decline(ctx.member, ctx.body.get("id"), ctx.body.get("reason"))
    return {"order": ctx.server.order_info(row)}


async def api_order_edit(ctx: Ctx) -> dict[str, Any]:
    server, body = ctx.server, ctx.body
    row = server.order(body.get("id"))
    if any(body.get(k) is not None for k in ("title", "instructions", "criteria", "priority", "deadline_hours",
                                              "claim_minutes")):  # fmt: skip
        row = server.edit(ctx.actor, body.get("id"), body)
    if "to" in body:
        row = server.reassign(ctx.actor, body.get("id"), body.get("to"))
    return {"order": server.order_info(row)}


async def api_order_cancel(ctx: Ctx) -> dict[str, Any]:
    return {"order": ctx.server.order_info(ctx.server.cancel(ctx.actor, ctx.body.get("id"), ctx.body.get("reason")))}


async def api_order_review(ctx: Ctx) -> dict[str, Any]:
    return ctx.server.review(ctx.actor, ctx.query.get("id"))


def _decide(accept: bool) -> Callable[[Ctx], Any]:
    async def act(ctx: Ctx) -> dict[str, Any]:
        body = ctx.body
        row = ctx.server.decide(ctx.actor, body.get("id"), body.get("report"), accept, body.get("reason") or "",
                                str(body.get("then") or "rework"))  # fmt: skip
        return {"order": ctx.server.order_info(row)}

    return act


async def api_order_approve(ctx: Ctx) -> dict[str, Any]:
    """Dispatch approval from a human's command line: signed for exactly the orders it showed, each with the digest
    of its content ("approve yes 12:<digest>,13:<digest>"; "no ..." refuses them), so an agent holding only the
    access token cannot approve its own orders, and an order changed after it was shown is refused."""
    refs = Fleet.approval_refs(ctx.body.get("orders"))
    yes = ctx.body.get("approve", True) is not False
    _signed_by_member(ctx, "approve", f"{'yes' if yes else 'no'} " + ",".join(f"{i}:{d}" for i, d in refs))
    rows = ctx.server.approve(ctx.actor, refs, yes, ctx.body.get("reason") or "")
    return {"orders": [ctx.server.order_info(r) for r in rows]}


async def api_questions(ctx: Ctx) -> dict[str, Any]:
    open_only = ctx.query.get("all") not in ("1", "true")
    rows = ctx.server.db.all("SELECT q.*, m.name, o.title FROM questions q LEFT JOIN members m ON m.id = q.member LEFT"
                             " JOIN orders o ON o.id = q.order_id" + (" WHERE q.answer IS NULL" if open_only else "")
                             + " ORDER BY q.id DESC LIMIT ?", (_limit(ctx),))  # fmt: skip
    return {"questions": [{"id": r["id"], "order": r["order_id"], "title": r["title"], "member": r["name"],
                           "at": r["at"], "text": r["text"], "answer": r["answer"], "answered_by": r["answered_by"]}
                          for r in rows]}  # fmt: skip


async def api_answer(ctx: Ctx) -> dict[str, Any]:
    row = ctx.server.answer(ctx.actor, ctx.body.get("id"), ctx.body.get("text"))
    return {"question": row["id"], "order": row["order_id"]}


async def api_trust(ctx: Ctx) -> dict[str, Any]:
    """Mark a worker as a trusted partner: signed by the director for that member's fingerprint ("trust <fp>"),
    after its human compared the fingerprint or the word phrase with the member's human."""
    server = ctx.server
    target = server.find(ctx.body.get("member"), ("worker",))
    _signed_by_member(ctx, "trust", target["fingerprint"])
    return {"member": server.member_info(server.trust(ctx.actor, target["id"]), full=True)}


async def api_untrust(ctx: Ctx) -> dict[str, Any]:
    server = ctx.server
    return {"member": server.member_info(server.trust(ctx.actor, ctx.body.get("member"), revoke=True), full=True)}


async def api_handover(ctx: Ctx) -> dict[str, Any]:
    """The director offers its role to a trusted partner: signed for that partner's fingerprint."""
    server = ctx.server
    target = server.find(ctx.body.get("member"))
    record = _signed_by_member(ctx, "handover", target["fingerprint"])
    row = server.handover_offer(ctx.actor, target["id"], record)
    return {"handover": row["id"], "expires": row["expires"], "to": server.member_info(target)}


async def api_handover_accept(ctx: Ctx) -> dict[str, Any]:
    """The partner accepts: signed for that handover and the offering director's fingerprint."""
    server = ctx.server
    h = server.handover_row(ctx.body.get("id"))
    record = _signed_by_member(ctx, "accept-handover", f"{h['id']} {h['from_fp']}")
    row = server.handover_accept(ctx.actor, h["id"], record)
    director = server.director()
    return {"handover": row["id"], "status": row["status"],
            "director": server.member_info(director) if director is not None else None}  # fmt: skip


async def api_handover_cancel(ctx: Ctx) -> dict[str, Any]:
    row = ctx.server.handover_cancel(ctx.actor, ctx.body.get("id"))
    return {"handover": row["id"], "status": row["status"]}


async def api_handovers(ctx: Ctx) -> dict[str, Any]:
    return {"handovers": ctx.server.handovers(open_only=ctx.query.get("all") not in ("1", "true"))}


# Every endpoint and the roles that may call it: authorization happens here, for the identity of the token.
API: dict[tuple[str, str], tuple[Callable[[Ctx], Any], tuple[str, ...] | None]] = {
    ("GET", "info"): (api_info, PUBLIC),
    ("POST", "auth/challenge"): (api_challenge, PUBLIC),
    ("POST", "auth/login"): (api_login, PUBLIC),  # signed
    ("POST", "auth/proof"): (api_proof, PUBLIC),  # the server proves its key (after a lost connection)
    ("POST", "panel-link"): (api_panel_link, PUBLIC),  # signed; leads only
    ("POST", "auth/logout"): (api_logout, ANY),
    ("GET", "events"): (api_events, ANY),
    ("POST", "heartbeat"): (api_heartbeat, ANY),
    ("GET", "status"): (api_status, ANY),
    ("POST", "leave"): (api_leave, ANY),
    ("POST", "messages"): (api_message, ADMITTED),
    ("GET", "members"): (api_members, LEADS),
    ("GET", "lobby"): (api_lobby, LEADS),
    ("POST", "members/admit"): (_member_action("admit"), LEADS),
    ("POST", "members/reject"): (_member_action("reject"), LEADS),
    ("POST", "members/kick"): (_member_action("kick"), LEADS),
    ("POST", "members/ban"): (_member_action("ban"), LEADS),
    ("POST", "members/unban"): (_member_action("unban"), ("director",)),
    ("GET", "invites"): (api_invites, LEADS),
    ("POST", "invites"): (api_invite_create, LEADS),
    ("POST", "invites/revoke"): (api_invite_revoke, LEADS),
    ("GET", "settings"): (api_settings, LEADS),
    ("POST", "settings"): (api_settings_set, LEADS),
    ("GET", "audit"): (api_audit, LEADS),
    ("GET", "orders"): (api_orders, ADMITTED),
    ("POST", "orders"): (api_order_create, LEADS),
    ("GET", "orders/get"): (api_order_get, ADMITTED),
    ("GET", "orders/files"): (api_order_files, ADMITTED),
    ("POST", "orders/claim"): (api_order_claim, WORKERS),
    ("POST", "orders/progress"): (api_order_progress, WORKERS),
    ("POST", "orders/ask"): (api_order_ask, WORKERS),
    ("POST", "orders/report"): (api_order_report, WORKERS),
    ("POST", "orders/decline"): (api_order_decline, WORKERS),
    ("POST", "orders/edit"): (api_order_edit, LEADS),
    ("POST", "orders/cancel"): (api_order_cancel, LEADS),
    ("GET", "orders/review"): (api_order_review, LEADS),
    ("POST", "orders/accept"): (_decide(True), LEADS),
    ("POST", "orders/reject"): (_decide(False), LEADS),
    ("POST", "orders/approve"): (api_order_approve, LEADS),  # signed
    ("GET", "questions"): (api_questions, LEADS),
    ("POST", "questions/answer"): (api_answer, LEADS),
    ("POST", "members/trust"): (api_trust, ("director",)),  # signed
    ("POST", "members/untrust"): (api_untrust, ("director",)),
    ("GET", "handover"): (api_handovers, LEADS),
    ("POST", "handover"): (api_handover, ("director",)),  # signed
    ("POST", "handover/accept"): (api_handover_accept, ("partner",)),  # signed
    ("POST", "handover/cancel"): (api_handover_cancel, LEADS),
}
UPLOADS = {"orders", "orders/report"}  # endpoints that carry files: up to MAX_UPLOAD bytes
CHALLENGED = {"auth/login", "panel-link"}  # public endpoints charged at their challenge (when they present it)


def json_response(data: Any, status: int = 200, headers: dict[str, str] | None = None) -> Any:
    from starlette.responses import Response

    body = json.dumps(data, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return Response(body, status, {**JSON_HEADERS, **(headers or {})}, media_type="application/json")


def refused_response(exc: Refused) -> Any:
    headers = {"Retry-After": str(exc.retry_after)} if exc.retry_after else {}
    if exc.status == 401:
        headers["WWW-Authenticate"] = 'Bearer realm="agent-hub-fleet"'
    return json_response({"error": exc.code, "message": exc.message}, exc.status, headers)


def host_ok(server: Server, request: Any) -> bool:
    hosts = request.headers.getlist("host")
    return len(hosts) == 1 and hosts[0].lower() in server.hosts


async def api_endpoint(request: Any) -> Any:
    server: Server = request.app.state.server
    path = request.path_params.get("path", "")
    try:
        if not host_ok(server, request):
            raise Refused(400, "wrong_host", "wrong Host header for this server")
        entry = API.get((request.method, path))
        if entry is None:
            known = {p for _m, p in API}
            raise Refused(405 if path in known else 404, "not_found", "no such endpoint")
        handler, roles = entry
        address = client_address(server, request)
        ctx = Ctx(server, request, address)
        try:
            ctx.query = {k: v[0] for k, v in parse_qs(request.url.query, max_num_fields=20).items()}
        except ValueError:
            raise Refused(400, "invalid", "too many query fields") from None
        if roles is PUBLIC:
            if path not in CHALLENGED:  # those: below, once their body is read
                bucket = {"info": server.page_limit, "auth/proof": server.proof_limit}.get(path, server.auth_limit)
                wait = bucket.take(address)
                if wait:
                    raise Refused(429, "rate_limited", "too many requests from this address; slow down",
                                  retry_after=wait)  # fmt: skip
        else:
            auth = request.headers.get("authorization", "")
            token = auth[7:].strip() if auth[:7].lower() == "bearer " else ""
            if not token or len(token) > 200:
                raise Refused(401, "no_token", "log in first (Authorization: Bearer <token>)")
            ctx.token_hash = sha(token)
            row = server.db.one("SELECT t.expires, t.revoked, m.* FROM tokens t JOIN members m ON m.id = t.member"
                                " WHERE t.hash = ?", (ctx.token_hash,))  # fmt: skip
            if row is None or row["revoked"] or row["expires"] < time.time():
                if row is not None and row["role"] not in ACTIVE:
                    refuse_member(row)
                raise Refused(401, "bad_token", "unknown, expired or revoked access token; log in again")
            if row["role"] not in ACTIVE:
                refuse_member(row)
            wait = server.api_limit.take(row["id"])
            if wait:
                raise Refused(429, "rate_limited", "too many requests; slow down", retry_after=wait)
            if row["role"] not in roles:
                raise Refused(403, "forbidden", f"a {ROLE_NAMES[row['role']]} may not use this endpoint")
            if request.method == "POST" and set(roles) <= set(LEADS) and server.settings()["takeover"]:
                raise Refused(423, "taken_over", "the director's human took over: the fleet's lead actions are done by"
                              " humans in the panel until they hand back (wait: fleet_wait tells you)")  # fmt: skip
            ctx.member = server.member(row["id"])
        if request.method == "POST":
            kind = request.headers.get("content-type", "").split(";")[0].strip().lower()
            if kind != "application/json":
                raise Refused(415, "invalid", "send JSON (Content-Type: application/json)")
            data = await read_body(request, MAX_UPLOAD if path in UPLOADS else MAX_BODY)
            try:
                body = json.loads(data.decode("utf-8")) if data else {}
            except (UnicodeDecodeError, ValueError):
                raise Refused(400, "invalid", "the body is not JSON") from None
            if not isinstance(body, dict):
                raise Refused(400, "invalid", "the body must be a JSON object")
            ctx.body = body
        if roles is PUBLIC and path in CHALLENGED:
            # A login (or a panel link) was charged at its challenge: one this address was given, still valid, is
            # not charged again. Anything else is.
            cid = ctx.body.get("challenge")
            challenge = server.challenges.get(cid) if isinstance(cid, str) else None
            if challenge is None or challenge.expires < time.time() or challenge.address != address:
                wait = server.auth_limit.take(address)
                if wait:
                    raise Refused(429, "rate_limited", "too many requests from this address; slow down",
                                  retry_after=wait)  # fmt: skip
        return json_response(await handler(ctx))
    except Refused as exc:
        return refused_response(exc)
    except sqlite3.Error as exc:
        log(f"database error on {request.method} /api/v1/{path[:100]}: {exc}")
        return json_response({"error": "unavailable", "message": "database busy; try again"}, 503, {"Retry-After": "2"})


# --------------------------------------------------------------------------
# HTTP: HTML pages (the landing page, invite links and the director's panel)
# --------------------------------------------------------------------------


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


CSS = """
:root { color-scheme: light dark; --fg: #1d1d1f; --bg: #f7f7f5; --card: #ffffff; --muted: #6a6a6a; --line: #e2e2de;
  --accent: #2f5bd3; --ok: #1e7a3c; --warn: #9a5b00; --bad: #b3261e; }
@media (prefers-color-scheme: dark) { :root { --fg: #e8e8e6; --bg: #141414; --card: #1c1c1c; --muted: #a2a2a2;
  --line: #2e2e2e; --accent: #8fb0ff; --ok: #7bd88f; --warn: #e0a84a; --bad: #ff8a80; } }
* { box-sizing: border-box; }
body { margin: 0; font: 15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; color: var(--fg);
  background: var(--bg); }
header { background: var(--card); border-bottom: 1px solid var(--line); }
nav { max-width: 1180px; margin: 0 auto; padding: .55rem 1rem; display: flex; gap: 1.1rem; align-items: center;
  flex-wrap: wrap; }
nav a { color: var(--fg); text-decoration: none; border-bottom: 2px solid transparent; padding: .15rem 0; }
nav a.here { border-color: var(--accent); font-weight: 600; }
nav .brand { font-weight: 700; }
main { max-width: 1180px; margin: 0 auto; padding: 1rem; }
h1 { font-size: 1.35rem; margin: .3rem 0 .8rem; }
h2 { font-size: 1.05rem; margin: 1.5rem 0 .5rem; }
p { margin: .4rem 0; }
a { color: var(--accent); }
code { font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace; font-size: .9em; overflow-wrap: anywhere; }
.card { background: var(--card); border: 1px solid var(--line); border-radius: 8px; padding: .75rem 1rem;
  margin: .6rem 0; }
.meta { color: var(--muted); font-size: .87rem; }
.text { white-space: pre-wrap; overflow-wrap: anywhere; max-height: 24em; overflow: auto; margin: .35rem 0;
  font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace; font-size: .88rem; }
table { border-collapse: collapse; width: 100%; background: var(--card); }
th, td { text-align: left; vertical-align: top; padding: .45rem .55rem; border-bottom: 1px solid var(--line); }
th { color: var(--muted); font-size: .84rem; font-weight: 600; }
td { overflow-wrap: anywhere; }
.flash { border: 1px solid var(--line); border-left: 4px solid var(--ok); background: var(--card); border-radius: 6px;
  padding: .55rem .9rem; margin: .5rem 0; white-space: pre-wrap; overflow-wrap: anywhere; }
.flash.error { border-left-color: var(--bad); }
.online { color: var(--ok); font-weight: 600; }
.working { color: var(--accent); font-weight: 600; }
.idle { color: var(--warn); }
.offline { color: var(--muted); }
.badge { display: inline-block; border: 1px solid var(--line); border-radius: 999px; padding: 0 .5rem;
  font-size: .8rem; color: var(--muted); }
form.inline { display: inline-flex; gap: .4rem; align-items: center; margin: .15rem .4rem .15rem 0; flex-wrap: wrap; }
nav form.inline { margin: 0 0 0 auto; }
button { font: inherit; font-size: .9rem; padding: .2rem .75rem; border: 1px solid var(--line); border-radius: 6px;
  background: var(--bg); color: var(--fg); cursor: pointer; }
button.primary { border-color: var(--accent); color: var(--accent); }
button.danger { border-color: var(--bad); color: var(--bad); }
input, select, textarea { font: inherit; font-size: .9rem; padding: .18rem .45rem; border: 1px solid var(--line);
  border-radius: 6px; background: var(--bg); color: var(--fg); }
textarea { width: 100%; min-height: 6rem; }
"""
CSS_HASH = base64.b64encode(hashlib.sha256(CSS.encode()).digest()).decode()
CSP = (f"default-src 'none'; style-src 'sha256-{CSS_HASH}'; form-action 'self'; frame-ancestors 'none';"
       " base-uri 'none'")  # fmt: skip
HTML_HEADERS = {
    "Content-Security-Policy": CSP,
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "same-origin",  # no-referrer would make browsers send "Origin: null" on the panel's forms
    "Cross-Origin-Opener-Policy": "same-origin",
    "Cross-Origin-Resource-Policy": "same-origin",
    "Cache-Control": "no-store",
}
PAGE = ('<!DOCTYPE html>\n<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,'
        ' initial-scale=1"><meta name="referrer" content="same-origin"><title>{title} · Agent Hub fleet</title>'
        '<style>{css}</style></head><body>{nav}<main>{flash}{body}</main></body></html>\n')  # fmt: skip
NAV = (("/panel", "Members"), ("/panel/lobby", "Lobby"), ("/panel/orders", "Orders"), ("/panel/approvals", "Approvals"),
       ("/panel/results", "Results"), ("/panel/questions", "Questions"), ("/panel/partners", "Partners"),
       ("/panel/invites", "Invites"), ("/panel/settings", "Settings"), ("/panel/messages", "Messages"),
       ("/panel/audit", "Audit log"))  # fmt: skip


def ago(ts: Any) -> str:
    if not ts:
        return "never"
    seconds = max(0, int(time.time() - float(ts)))
    for unit, size in (("d", 86400), ("h", 3600), ("m", 60)):
        if seconds >= size:
            return f"{seconds // size}{unit} ago"
    return f"{seconds}s ago"


def until(ts: Any) -> str:
    seconds = int(float(ts) - time.time())
    if seconds <= 0:
        return "expired"
    for unit, size in (("d", 86400), ("h", 3600), ("m", 60)):
        if seconds >= size:
            return f"in {seconds // size}{unit}"
    return f"in {seconds}s"


def html_response(body: Safe, status: int = 200, headers: dict[str, str] | None = None) -> Any:
    from starlette.responses import Response

    extra = dict(headers or {})
    return Response(str(body).encode("utf-8"), status, {**HTML_HEADERS, **extra}, media_type="text/html; charset=utf-8")


def layout(title: str, body: Safe, session: PanelSession | None = None, path: str = "",
           flashes: list[tuple[str, str]] | None = None) -> Safe:  # fmt: skip
    nav = flash = Safe("")
    if session is not None:
        links = cat(fmt('<a href="{}"{}>{}</a>', href, Safe(' class="here" aria-current="page"') if href == path
                        else "", label) for href, label in NAV)  # fmt: skip
        who = fmt('<span class="meta">{}</span>', session.who)
        out = fmt('<form method="post" action="/panel/do/logout" class="inline"><input type="hidden" name="csrf"'
                  ' value="{}"><button type="submit">Sign out</button></form>', session.csrf)  # fmt: skip
        nav = fmt('<header><nav><span class="brand">Agent Hub fleet</span>{}{}{}</nav></header>', links, who, out)
    flash = cat(fmt('<div class="flash{}" role="status">{}</div>', " error" if kind == "error" else "", text)
                for kind, text in flashes or [])  # fmt: skip
    return fmt(PAGE, title=title, css=Safe(CSS), nav=nav, flash=flash, body=body)


def message_page(title: str, text: str, status: int, headers: dict[str, str] | None = None) -> Any:
    return html_response(layout(title, fmt("<h1>{}</h1><p>{}</p>", title, text)), status, headers)


async def page_home(request: Any) -> Any:
    server: Server = request.app.state.server
    if not host_ok(server, request):
        return message_page("Wrong address", "Wrong Host header for this server.", 400)
    if server.page_limit.take(client_address(server, request)):
        return message_page("Slow down", "Too many requests from this address.", 429, {"Retry-After": "30"})
    settings = server.settings()
    body = fmt("<h1>{}</h1><div class=\"card\"><p>This is an Agent Hub fleet server (fleet mode, Agent Hub {}).</p>"
               "<p>To join, turn fleet mode on in your agent's Agent Hub (AGENT_HUB_FLEET=1) and have the agent call"
               " <code>fleet_join(server=\"{}\", name=...)</code>{}.</p><p class=\"meta\">Server id <code>{}</code>"
               " · server key fingerprint <code>{}</code> (compare it with what the owner or the director gave you)"
               "</p></div>", server.conf.get("name") or "Agent Hub fleet", __version__, server.options.public_url,
               " with an invite link from the director" if settings["join_mode"] == "invite" else
               " (it takes no new members right now)" if settings["join_mode"] == "closed" else "", server.server_id,
               server.key_fp)  # fmt: skip
    return html_response(layout("Fleet", body))


async def page_join(request: Any) -> Any:
    server: Server = request.app.state.server
    if not host_ok(server, request):
        return message_page("Wrong address", "Wrong Host header for this server.", 400)
    if server.auth_limit.take(client_address(server, request)):
        return message_page("Slow down", "Too many requests from this address.", 429, {"Retry-After": "30"})
    token = request.path_params.get("token", "")
    invite = server.db.one("SELECT * FROM invites WHERE hash = ?", (sha(token),)) if len(token) < 200 else None
    problem = server.invite_problem(invite, time.time())
    if problem:
        return message_page("Invite not valid", f"This invite {problem}. Ask the director for a new one.", 404)
    link = invite_link(server.options.public_url, token, server.key_fp)
    body = fmt("<h1>Invite to {}</h1><div class=\"card\"><p>Give this to your AI agent (Agent Hub with fleet mode on,"
               " AGENT_HUB_FLEET=1):</p><p><code>fleet_join(server=\"{}\", name=\"your-name\", description=\"...\")"
               "</code></p><p class=\"meta\">Valid {} · {} use(s) left. Orders from the fleet are requests from its"
               " director: your agent follows your instructions first.</p></div>", server.conf.get("name") or "a fleet",
               link, until(invite["expires"]), invite["max_uses"] - invite["uses"])  # fmt: skip
    return html_response(layout("Invite", body))


# -- the panel ---------------------------------------------------------------


@dataclass
class PanelSession:
    hash: str
    member: sqlite3.Row | None  # None: the owner
    csrf: str

    @property
    def actor(self) -> Actor:
        return Actor(self.member)

    @property
    def role(self) -> str:
        return "owner" if self.member is None else self.member["role"]

    @property
    def who(self) -> str:
        if self.member is None:
            return "signed in as the owner"
        return f"signed in as {self.member['name']} ({ROLE_NAMES[self.member['role']]})"


def cookie_name(server: Server) -> str:
    # __Host-: only over HTTPS, for this host only, Path=/. Plain-HTTP development on 127.0.0.1 cannot use it.
    return "__Host-fleet_panel" if server.secure else f"fleet_panel_{server.options.port}"


def session_cookie(server: Server, value: str, max_age: int) -> str:
    secure = "; Secure" if server.secure else ""
    return f"{cookie_name(server)}={value}; Path=/; HttpOnly; SameSite=Strict; Max-Age={max_age}{secure}"


def panel_session(server: Server, request: Any) -> PanelSession | None:
    value = request.cookies.get(cookie_name(server), "")
    if not value or len(value) > 200:
        return None
    row = server.db.one("SELECT * FROM panel_sessions WHERE hash = ?", (sha(value),))
    if row is None or row["expires"] < time.time():
        return None
    member = None
    if row["member"] is not None:
        member = server.member(row["member"])
        if member is None or member["role"] not in LEADS:
            return None
    return PanelSession(row["hash"], member, row["csrf"])


def origin_ok(server: Server, request: Any) -> bool:
    origins = request.headers.getlist("origin")
    host = request.headers.get("host", "").lower()
    scheme = "https" if server.secure else "http"
    return origins == [f"{scheme}://{host}"] and host in server.hosts


def flash(server: Server, session: PanelSession, kind: str, text: str) -> None:
    items = server.flashes.setdefault(session.hash, [])
    items.append((kind, text))
    del items[:-5]


def csrf_field(session: PanelSession) -> Safe:
    return fmt('<input type="hidden" name="csrf" value="{}">', session.csrf)


def button(session: PanelSession, action: str, fields: dict[str, Any], label: str, cls: str = "",
           extra: Safe = Safe("")) -> Safe:  # fmt: skip
    hidden = cat(fmt('<input type="hidden" name="{}" value="{}">', k, v) for k, v in fields.items())
    return fmt('<form method="post" action="/panel/do/{}" class="inline">{}{}{}<button type="submit"{}>{}</button>'
               "</form>", action, csrf_field(session), hidden, extra, Safe(f' class="{cls}"') if cls else "", label)  # fmt: skip


def presence_cell(server: Server, row: Any) -> Safe:
    presence, seen = server.presence(row)
    return fmt('<span class="{}">{}</span><br><span class="meta">{}</span>', presence, presence,
               "now" if presence != "offline" else f"last seen {ago(seen)}")  # fmt: skip


def identity_cell(row: Any) -> Safe:
    raw = base64.b64decode(row["key"])
    return fmt("<code>{}</code><br><span class=\"meta\">{}</span>", row["fingerprint"], phrase(raw))


def page_members(server: Server, session: PanelSession, query: dict[str, str]) -> Safe:
    rows = server.db.all("SELECT * FROM members WHERE role IN ('director', 'partner', 'worker', 'banned') ORDER BY"
                         " CASE role WHEN 'director' THEN 0 WHEN 'partner' THEN 1 WHEN 'worker' THEN 2 ELSE 3 END,"
                         " name")  # fmt: skip
    rank = RANK[session.role]
    working: dict[int, list[Any]] = {}
    for o in server.db.all("SELECT id, title, claimer FROM orders WHERE status IN ('claimed', 'in_progress', 'reported')"
                           " AND claimer IS NOT NULL ORDER BY id"):  # fmt: skip
        working.setdefault(o["claimer"], []).append(o)
    lines = []
    for r in rows:
        actions = Safe("")
        if r["role"] == "banned":
            if rank >= RANK["director"]:
                actions = button(session, "unban", {"member": r["id"]}, "Lift ban")
        elif r["role"] != "director" and rank > RANK[r["role"]] and (session.member is None
                                                                     or session.member["id"] != r["id"]):  # fmt: skip
            reason = Safe('<input name="reason" maxlength="500" placeholder="reason (optional)">')
            actions = (button(session, "kick", {"member": r["id"]}, "Remove", extra=reason)
                       + button(session, "ban", {"member": r["id"]}, "Ban key", "danger", extra=reason))  # fmt: skip
        task = cat(fmt('<a href="/panel/order?id={}">#{} {}</a><br>', o["id"], o["id"], one_line(o["title"], 60))
                   for o in working.get(r["id"], []))  # fmt: skip
        lines.append(fmt("<tr><td>#{}</td><td><b>{}</b><br><span class=\"badge\">{}</span></td><td>{}</td><td>{}</td>"
                         "<td>{}</td><td>{}</td><td><div class=\"text\">{}</div></td><td>{}</td><td>{}</td></tr>",
                         r["id"], r["name"], ROLE_NAMES[r["role"]], presence_cell(server, r) if r["role"] != "banned"
                         else r["removed_reason"], task or "", identity_cell(r), r["agent"], r["description"],
                         r["contact"], actions))  # fmt: skip
    settings = server.settings()
    head = fmt("<h1>Members</h1><p class=\"meta\">{} · join mode <b>{}</b>, approval <b>{}</b>{}{} · server id <code>{}"
               "</code></p>", server.conf.get("name", ""), settings["join_mode"], "on" if settings["approval"] else
               "off", Safe(" · <b>the fleet is paused</b>") if settings["paused"] else "",
               Safe(" · <b>taken over by the director's human</b>") if settings["takeover"] else "", server.server_id)  # fmt: skip
    if not lines:
        return head + fmt('<div class="card"><p>No members yet.</p></div>')
    return head + fmt("<p class=\"meta\">Names, descriptions, agents and contacts are what members wrote about"
                      " themselves. Compare fingerprints or word phrases over another channel (a call) before you"
                      " trust anyone.</p><table><tr><th>#</th><th>name, role</th><th>presence</th><th>current task</th>"
                      "<th>fingerprint, phrase</th><th>agent</th><th>description</th><th>contact</th><th></th></tr>{}"
                      "</table>",
                      cat(lines))  # fmt: skip


def page_lobby(server: Server, session: PanelSession, query: dict[str, str]) -> Safe:
    rows = server.db.all("SELECT * FROM members WHERE role = 'lobby' ORDER BY joined_at")
    lines = []
    for r in rows:
        reason = Safe('<input name="reason" maxlength="500" placeholder="reason (optional)">')
        lines.append(fmt("<tr><td>#{}</td><td><b>{}</b><br><span class=\"meta\">waiting {}</span></td><td>{}</td>"
                         "<td>{}</td><td><div class=\"text\">{}</div></td><td>{}</td><td>{}{}{}</td></tr>", r["id"],
                         r["name"], ago(r["joined_at"]).replace(" ago", ""), identity_cell(r), r["agent"],
                         r["description"], r["contact"], button(session, "admit", {"member": r["id"]}, "Admit",
                                                                 "primary"),
                         button(session, "reject", {"member": r["id"]}, "Reject", extra=reason),
                         button(session, "ban", {"member": r["id"]}, "Ban key", "danger", extra=reason)))  # fmt: skip
    head = fmt("<h1>Lobby</h1><p class=\"meta\">New identities wait here while approval is on. What they say about"
               " themselves is their own words.</p>")  # fmt: skip
    if not lines:
        return head + fmt('<div class="card"><p>Nobody is waiting.</p></div>')
    return head + fmt("<table><tr><th>#</th><th>name</th><th>fingerprint, phrase</th><th>agent</th><th>description"
                      "</th><th>contact</th><th></th></tr>{}</table>", cat(lines))  # fmt: skip


def page_invites(server: Server, session: PanelSession, query: dict[str, str], new_link: str = "") -> Safe:
    rows = server.invites()
    shown = Safe("")
    if new_link:
        shown = fmt('<div class="card"><p><b>New invite link</b> (shown once; copy it now):</p><p><code>{}</code></p>'
                    "</div>", new_link)  # fmt: skip
    form = fmt('<div class="card"><form method="post" action="/panel/do/invite" class="inline">{}<label>uses <input'
               ' type="number" name="uses" value="1" min="1" max="{}"></label><label>valid for (hours) <input'
               ' type="number" name="hours" value="24" min="0.05" max="{}" step="any"></label><input name="note"'
               ' maxlength="200" placeholder="note (who it is for)"><button type="submit" class="primary">Create'
               " invite</button></form></div>", csrf_field(session), INVITE_MAX_USES, INVITE_MAX_HOURS)  # fmt: skip
    lines = cat(fmt("<tr><td>#{}</td><td>{}</td><td>{} / {}</td><td>{}</td><td>{}<br><span class=\"meta\">{}</span>"
                    "</td><td>{}</td><td>{}</td></tr>", r["id"], r["state"], r["uses"], r["max_uses"],
                    until(r["expires"]), r["created_by"], ago(r["created"]), r["note"],
                    button(session, "revoke_invite", {"id": r["id"]}, "Revoke", "danger") if r["state"] == "valid"
                    else "") for r in rows)  # fmt: skip
    table = fmt("<table><tr><th>#</th><th>state</th><th>uses</th><th>expires</th><th>made by</th><th>note</th><th>"
                "</th></tr>{}</table>", lines) if rows else fmt("<p>No invites yet.</p>")  # fmt: skip
    return fmt("<h1>Invites</h1>{}{}{}", shown, form, table)


def page_settings(server: Server, session: PanelSession, query: dict[str, str]) -> Safe:
    s = server.settings()

    def select(name: str, choices: Iterable[str], current: str) -> Safe:
        return fmt('<select name="{}">{}</select>', name, cat(fmt('<option value="{}"{}>{}</option>', c,
                   Safe(" selected") if c == current else "", c) for c in choices))  # fmt: skip

    form = fmt('<div class="card"><form method="post" action="/panel/do/settings">{}<p><label>Join mode {}</label>'
               ' <span class="meta">open: anyone with the address · invite: an invite link is needed · closed: no new'
               ' members</span></p><p><label>Approval {}</label> <span class="meta">on: new identities wait in the'
               ' lobby until admitted</span></p><p><label>Fleet {}</label> <span class="meta">paused: members are'
               ' told to wait</span></p><p><button type="submit" class="primary">Save</button></p></form></div>',
               csrf_field(session), select("join_mode", JOIN_MODES, s["join_mode"]),
               select("approval", ("on", "off"), "on" if s["approval"] else "off"),
               select("paused", ("running", "paused"), "paused" if s["paused"] else "running"))  # fmt: skip
    human = fmt('<div class="card"><form method="post" action="/panel/do/settings">{}<p><label>Dispatch approval {}'
                '</label> <span class="meta">on: every new order waits until a human approves it (Approvals, or'
                ' <code>agent_hub.py fleet approve</code> on a lead\'s computer); agents cannot switch this</span></p>'
                '<p><label>Claims without news are re-queued after {} minute(s)</label> <input type="number"'
                ' name="claim_minutes" value="{}" min="1" max="10080"></p><p><label>A claim whose claimer is offline is'
                ' re-queued after (seconds)</label> <input type="number" name="requeue_offline_seconds" value="{}"'
                ' min="5" max="86400"></p><p><button type="submit" class="primary">Save</button></p></form></div>',
                csrf_field(session), select("dispatch_approval", ("on", "off"), "on" if s["dispatch_approval"] else "off"),
                s["claim_minutes"], s["claim_minutes"], s["requeue_offline_seconds"])  # fmt: skip
    take = Safe("")
    if session.role in ("owner", "director"):
        take = fmt('<div class="card"><p><b>Take over</b>: while on, the leads\' agents cannot give, change or decide'
                   ' orders, admit or remove members or change settings with their access tokens; you act here. Now'
                   ' <b>{}</b>.</p>{}</div>', "on" if s["takeover"] else "off",
                   button(session, "takeover", {"on": "off" if s["takeover"] else "on"}, "Hand back to the agent" if
                          s["takeover"] else "Take over", "" if s["takeover"] else "danger"))  # fmt: skip
    limits = fmt('<p class="meta">Limits (the owner changes them on the server with hub_server.py owner settings):'
                 " {} new identities per address and hour in open mode, lobby up to {}, at most {} members; {} bytes"
                 " of files per order and per report, {} MB in all, {} unfinished orders; a handover {} the owner's"
                 " confirmation.</p>", s["new_keys_per_hour"], s["lobby_max"], s["members_max"], s["max_order_bytes"],
                 s["storage_mb"], s["orders_open_max"], "needs" if s["handover_needs_owner"] else "does not need")  # fmt: skip
    return fmt("<h1>Settings</h1><p class=\"meta\">Changes apply at once, to the next join; members keep their"
               " role.</p>{}{}{}{}", form, human, take, limits)  # fmt: skip


def page_messages(server: Server, session: PanelSession, query: dict[str, str]) -> Safe:
    if session.member is None:
        rows = server.db.all("SELECT e.*, s.name AS s_name, t.name AS t_name FROM events e LEFT JOIN members s ON s.id ="
                             " e.sender LEFT JOIN members t ON t.id = e.member WHERE e.kind IN ('message', 'broadcast')"
                             " ORDER BY e.id DESC LIMIT 200")  # fmt: skip
    else:
        me = session.member["id"]
        rows = server.db.all("SELECT e.*, s.name AS s_name, t.name AS t_name FROM events e LEFT JOIN members s ON s.id ="
                             " e.sender LEFT JOIN members t ON t.id = e.member WHERE (e.kind = 'message' AND"
                             " (e.sender = ? OR e.member = ?)) OR e.kind = 'broadcast' ORDER BY e.id DESC LIMIT 200",
                             (me, me))  # fmt: skip
    members = server.db.all("SELECT id, name, role FROM members WHERE role IN ('director', 'partner', 'worker',"
                            " 'lobby') ORDER BY name")  # fmt: skip
    me_id = session.member["id"] if session.member is not None else None
    choices = [("all", "everyone")] if session.role in ("owner", "director") else []
    choices += [(f"#{m['id']}", f"{m['name']} ({ROLE_NAMES[m['role']]})") for m in members if m["id"] != me_id]
    form = fmt('<div class="card"><form method="post" action="/panel/do/send">{}<p><label>To <select name="to">{}'
               '</select></label></p><p><textarea name="text" maxlength="{}" required></textarea></p><p><button'
               ' type="submit" class="primary">Send</button></p></form></div>', csrf_field(session),
               cat(fmt('<option value="{}">{}</option>', v, label) for v, label in choices), MAX_TEXT)  # fmt: skip
    items = []
    for r in rows:
        sender = "the owner" if r["sender"] == 0 else r["s_name"] or "?"
        to = "everyone" if r["kind"] == "broadcast" else r["t_name"] or "?"
        items.append(fmt('<div class="card"><p class="meta">#{} · {} → {} · {}</p><div class="text">{}</div></div>',
                         r["id"], sender, to, ago(r["at"]), r["text"]))  # fmt: skip
    return fmt("<h1>Messages</h1>{}{}", form, cat(items) if items else fmt("<p>No messages yet.</p>"))


def page_audit(server: Server, session: PanelSession, query: dict[str, str]) -> Safe:
    rows = server.db.all("SELECT * FROM audit ORDER BY id DESC LIMIT 300")
    lines = cat(fmt("<tr><td>#{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>", r["id"],
                    time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(r["at"])), r["actor"], r["action"], r["detail"],
                    r["address"] if session.member is None else "") for r in rows)  # fmt: skip
    return fmt("<h1>Audit log</h1><p class=\"meta\">Who did what, newest first (the last 300 entries; the owner sees"
               " addresses).</p><table><tr><th>#</th><th>when</th><th>who</th><th>what</th><th>details</th><th>address"
               "</th></tr>{}</table>", lines)  # fmt: skip


def status_badge(status: str) -> Safe:
    return fmt('<span class="badge">{}</span>', STATUS_NAMES.get(status, status))


def order_line(r: Any, server: Server) -> Safe:
    info = server.order_info(r)
    deadline = until(r["deadline"]) if r["deadline"] else "none"
    return fmt('<tr><td><a href="/panel/order?id={}">#{}</a></td><td><a href="/panel/order?id={}">{}</a></td><td>{}'
               "</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}<br><span class=\"meta\">{}</span></td><td>{}"
               "</td></tr>", r["id"], r["id"], r["id"], r["title"], r["kind"], PRIORITY_NAMES.get(r["priority"], "?"),
               status_badge(r["status"]), ", ".join(info["to"]) or "anyone", info["claimer"] or "", r["creator"],
               ago(r["created"]), deadline)  # fmt: skip


ORDER_HEAD = Safe("<tr><th>#</th><th>title</th><th>kind</th><th>priority</th><th>status</th><th>for</th><th>claimed by"
                  "</th><th>made by</th><th>deadline</th></tr>")  # fmt: skip


def page_orders(server: Server, session: PanelSession, query: dict[str, str]) -> Safe:
    status = query.get("status", "open")
    if status not in (*ORDER_STATUSES, "open", "final", "all"):
        status = "open"
    counts = {r["status"]: r["n"] for r in server.db.all("SELECT status, COUNT(*) AS n FROM orders GROUP BY status")}
    board = cat(fmt('<a href="/panel/orders?status={}" class="badge">{}: {}</a> ', st, STATUS_NAMES[st],
                    counts.get(st, 0)) for st in ORDER_STATUSES)  # fmt: skip
    wanted = (ORDER_OPEN if status == "open" else ORDER_FINAL if status == "final" else ORDER_STATUSES
              if status == "all" else (status,))  # fmt: skip
    rows = server.db.all(f"SELECT * FROM orders WHERE status IN ({','.join('?' * len(wanted))}) ORDER BY CASE WHEN"
                         " status IN ('approval', 'queued', 'offered') THEN 0 ELSE 1 END, priority DESC, id DESC LIMIT"
                         " 300", wanted)  # fmt: skip
    filters = cat(fmt('<a href="/panel/orders?status={}">{}</a> · ', f, f) for f in ("open", "final", "all"))
    table = fmt("<table>{}{}</table>", ORDER_HEAD, cat(order_line(r, server) for r in rows)) if rows else \
        fmt('<div class="card"><p>No orders here.</p></div>')  # fmt: skip
    return fmt("<h1>Orders</h1><p class=\"meta\">The board: {}</p><p class=\"meta\">Show: {}Titles and texts are what"
               " leads and members wrote.</p>{}", board, filters, table)  # fmt: skip


def page_order(server: Server, session: PanelSession, query: dict[str, str]) -> Safe:
    try:
        row = server.order(query.get("id"))
    except Refused as exc:
        return fmt("<h1>Order</h1><p>{}</p>", exc.message)
    actor = session.actor
    d = server.order_detail(actor, row["id"])
    facts = [("status", STATUS_NAMES.get(row["status"], row["status"]) + (f" ({row['closed_reason']})" if
                                                                          row["closed_reason"] else "")),
             ("kind", row["kind"] + (f": {row['repo_url']} {row['repo_ref']}".rstrip() if row["kind"] == "repo" else
                                     f", {row['files']} file(s), {row['bytes']:,} bytes")),
             ("priority", PRIORITY_NAMES.get(row["priority"], "?")),
             ("deadline", until(row["deadline"]) if row["deadline"] else "none"),
             ("for", ", ".join(d["to"]) or "anyone (the first to claim it)"),
             ("claimed by", (d["claimer"] or "nobody") + (f", last news {ago(row['news_at'])}" if row["news_at"] else "")),
             ("made by", f"{row['creator']}, {ago(row['created'])}" + (f"; approved by {row['approved_by']}" if
                                                                     row["approved_by"] else "")),
             ("claim time", f"{row['claim_minutes']} minute(s) without news")]  # fmt: skip
    table = cat(fmt("<tr><th>{}</th><td>{}</td></tr>", k, v) for k, v in facts)
    parts = [fmt('<h1>Order #{}: {}</h1><table>{}</table><h2>Instructions</h2><div class="card text">{}</div>',
                 row["id"], row["title"], table, row["instructions"])]  # fmt: skip
    if row["criteria"]:
        parts.append(fmt('<h2>Acceptance criteria</h2><div class="card text">{}</div>', row["criteria"]))
    if d["paths"]:
        parts.append(fmt('<h2>Files (as the workers see them)</h2><div class="card text">{}</div>',
                         "\n".join(d["paths"])))  # fmt: skip
    if d["manage"] and row["status"] not in ORDER_FINAL:
        reason = Safe('<input name="reason" maxlength="500" placeholder="reason (optional)">')
        acts = [button(session, "cancel", {"id": row["id"]}, "Cancel order", "danger", extra=reason)]
        if row["status"] == "approval" and server.may_decide(actor, row):
            ref = f"{row['id']}:{d['digest']}"  # exactly this content: changed since, it is refused
            acts = [button(session, "approve", {"orders": ref}, "Approve dispatch", "primary"),
                    button(session, "refuse", {"orders": ref}, "Refuse", extra=reason)] + acts  # fmt: skip
        members = server.db.all("SELECT id, name FROM members WHERE role IN ('worker', 'partner') ORDER BY name")
        choices = cat(fmt('<option value="#{}">{}</option>', m["id"], m["name"]) for m in members)
        if row["status"] != "reported":
            acts.append(fmt('<form method="post" action="/panel/do/reassign" class="inline">{}<input type="hidden"'
                            ' name="id" value="{}"><select name="to"><option value="any">anyone</option>{}</select>'
                            '<button type="submit">Reassign</button></form>', csrf_field(session), row["id"], choices))  # fmt: skip
        edit = fmt('<form method="post" action="/panel/do/edit">{}<input type="hidden" name="id" value="{}"><p><input'
                   ' name="title" maxlength="{}" value="{}" size="60"> <select name="priority">{}</select> deadline in'
                   ' <input type="number" name="deadline_hours" min="0.01" step="any" placeholder="hours"></p><p>'
                   '<textarea name="instructions" maxlength="{}">{}</textarea></p><p><button type="submit">Save changes'
                   '</button> <span class="meta">its claimer is told</span></p></form>', csrf_field(session), row["id"],
                   MAX_TITLE, row["title"], cat(fmt('<option value="{}"{}>{}</option>', k, Safe(" selected") if v ==
                                                    row["priority"] else "", k) for k, v in PRIORITIES.items()),
                   MAX_INSTRUCTIONS, row["instructions"])  # fmt: skip
        parts.append(fmt('<h2>Act</h2><div class="card">{}</div><details class="card"><summary>Edit</summary>{}'
                         "</details>", cat(acts), edit))  # fmt: skip
    if d["questions"]:
        parts.append(fmt("<h2>Questions</h2>{}", cat(fmt('<div class="card"><p class="meta">#{} by {} · {}</p><div'
                                                         ' class="text">{}</div>{}</div>', q["id"], q["member"],
                                                         ago(q["at"]), q["text"],
                                                         fmt('<p class="meta">answer by {}:</p><div class="text">{}'
                                                             "</div>", q["answered_by"], q["answer"])
                                                         if q["answer"] is not None else
                                                         fmt('<p><a href="/panel/questions">answer it</a></p>'))
                                                     for q in d["questions"])))  # fmt: skip
    if d["reports"]:
        parts.append(fmt("<h2>Reports</h2>{}", cat(fmt('<div class="card"><p class="meta">report #{} by {} · {} · {}{}'
                                                       '</p><div class="text">{}</div></div>', r["id"], r["member"],
                                                       ago(r["at"]), r["status"], f" ({r['reason']})" if r["reason"]
                                                       else "", r["summary"]) for r in d["reports"])))  # fmt: skip
        if row["status"] == "reported":
            parts.append(fmt('<p><a href="/panel/results?id={}">Review the result</a></p>', row["id"]))
    parts.append(fmt("<h2>History</h2><table>{}</table>", cat(fmt("<tr><td>{}</td><td>{}</td><td>{}</td><td><div"
                                                                  ' class="text">{}</div></td></tr>',
                                                                  time.strftime("%Y-%m-%d %H:%M:%S",
                                                                                time.localtime(h["at"])), h["who"],
                                                                  h["kind"], h["text"]) for h in d["history"])))  # fmt: skip
    if d["declines"]:
        parts.append(fmt("<h2>Declined by</h2>{}", cat(fmt("<p>{}: {}</p>", x["member"], x["reason"] or "(no reason)")
                                                       for x in d["declines"])))  # fmt: skip
    return cat(parts)


def page_approvals(server: Server, session: PanelSession, query: dict[str, str]) -> Safe:
    on = server.settings()["dispatch_approval"]
    rows = server.db.all("SELECT * FROM orders WHERE status = 'approval' ORDER BY priority DESC, id")
    head = fmt("<h1>Dispatch approvals</h1><p class=\"meta\">Dispatch approval is <b>{}</b> (Settings). While it is on,"
               " every new order waits here until a human approves it; the leads' agents cannot approve their own"
               " orders.</p>", "on" if on else "off")  # fmt: skip
    if not rows:
        return head + fmt('<div class="card"><p>No order waits for approval.</p></div>')
    items, mine = [], []
    for r in rows:
        reason = Safe('<input name="reason" maxlength="500" placeholder="reason (optional)">')
        ref = f"{r['id']}:{server.digest(r)}"  # what is shown here, exactly: changed since, it is refused
        acts = Safe("")
        if server.may_decide(session.actor, r):
            mine.append(ref)
            acts = (button(session, "approve", {"orders": ref}, "Approve", "primary")
                    + button(session, "refuse", {"orders": ref}, "Refuse", extra=reason))  # fmt: skip
        else:
            acts = fmt('<p class="meta">Made by {}: the director\'s human or the owner decides it.</p>', r["creator"])
        files = [x["path"] for x in server.db.all("SELECT path FROM order_files WHERE order_id = ? ORDER BY path",
                                                  (r["id"],))]  # fmt: skip
        extra = fmt('<p class="meta">Acceptance criteria:</p><div class="text">{}</div>', r["criteria"]) if \
            r["criteria"] else Safe("")  # fmt: skip
        what = (f"repository {r['repo_url']} {r['repo_ref']}".rstrip() if r["kind"] == "repo" else
                f"{len(files)} file(s): " + ", ".join(files[:30]) + (" ..." if len(files) > 30 else ""))  # fmt: skip
        items.append(fmt('<div class="card"><p><a href="/panel/order?id={}"><b>#{} {}</b></a> <span class="meta">{} ·'
                         " for {} · by {} · {}</span></p><div class=\"text\">{}</div>{}<p class=\"meta\">{}</p>{}"
                         "</div>", r["id"], r["id"], r["title"], r["kind"],
                         ", ".join(server.order_info(r)["to"]) or "anyone", r["creator"], ago(r["created"]),
                         r["instructions"], extra, what, acts))  # fmt: skip
    every = button(session, "approve", {"orders": ",".join(mine)}, f"Approve these {len(mine)}", "primary") if \
        len(mine) > 1 else Safe("")  # exactly the orders on this page, as shown  # fmt: skip
    return head + fmt("<p>{}</p>{}", every, cat(items))


def unified(old: str, new: str, a: str, b: str) -> str:
    return "".join(difflib.unified_diff(old.splitlines(keepends=True), new.splitlines(keepends=True), a, b, n=3))


def result_card(server: Server, session: PanelSession, row: Any) -> Safe:
    review = server.review(session.actor, row["id"])
    rep = review["report"]
    if rep is None:
        return Safe("")
    shown, budget = [], 400_000
    if row["kind"] == "workspace":
        base = {f["path"]: f["content"] for f in review["base"]}
        for f in rep["files"]:
            old = base.get(f["path"], "")
            text = unified(old if f["kind"] != "new" else "", f["content"] if f["kind"] != "deleted" else "",
                           "/dev/null" if f["kind"] == "new" else f"a/{f['path']}",
                           "/dev/null" if f["kind"] == "deleted" else f"b/{f['path']}")  # fmt: skip
            shown.append(f"===== {f['kind']}: {f['path']} =====\n{text}")
        body = "\n".join(shown) or "(no files changed)"
        note = ("Neutral paths and hidden names, as the worker saw them. The files are applied on the computer that"
                " made the order, with the real paths and names back: fleet_accept there (or agent_hub.py fleet apply"
                f" {row['id']}).")  # fmt: skip
    else:
        body = rep["patch"] or "(no patch)"
        note = ("A patch for the repository (apply it yourself, e.g. git apply --check first)" if rep["patch"] else
                "") + (f" Pull request: {rep['pr_url']}" if rep["pr_url"] else "")  # fmt: skip
    if len(body) > budget:
        body = body[:budget] + f"\n[... cut at {budget:,} characters]"
    acts = Safe("")
    if review["manage"] and rep["status"] == "pending":
        then = Safe('<select name="then"><option value="rework">back to the same worker</option><option value="queue">'
                    'back to the queue</option><option value="close">close the order</option></select>')  # fmt: skip
        reason = Safe('<input name="reason" maxlength="8000" placeholder="why (the worker is told)" required>')
        acts = (button(session, "accept_result", {"id": row["id"], "report": rep["id"]}, "Accept", "primary")
                + button(session, "reject_result", {"id": row["id"], "report": rep["id"]}, "Reject", "danger",
                         extra=reason + then))  # fmt: skip
    return fmt('<div class="card"><p><a href="/panel/order?id={}"><b>#{} {}</b></a> <span class="meta">report #{} by {}'
               " (fingerprint {}) · {} · {}</span></p><p class=\"meta\">The worker's summary (its own words):</p><div"
               ' class="text">{}</div><p class="meta">{}</p><div class="text">{}</div>{}</div>', row["id"], row["id"],
               row["title"], rep["id"], rep["member"], short_fp(rep["fingerprint"] or ""), ago(rep["at"]),
               rep["status"], rep["summary"], note, body, acts)  # fmt: skip


def page_results(server: Server, session: PanelSession, query: dict[str, str]) -> Safe:
    if query.get("id"):
        try:
            rows = [server.order(query["id"])]
        except Refused as exc:
            return fmt("<h1>Results</h1><p>{}</p>", exc.message)
    else:
        rows = server.db.all("SELECT * FROM orders WHERE status = 'reported' ORDER BY priority DESC, updated")
    head = fmt("<h1>Results to review</h1><p class=\"meta\">Results are untrusted until you accept them: read the"
               " changes, not only the summary.</p>")  # fmt: skip
    cards = [c for c in (result_card(server, session, r) for r in rows) if c]
    return head + (cat(cards) if cards else fmt('<div class="card"><p>Nothing to review.</p></div>'))


def page_questions(server: Server, session: PanelSession, query: dict[str, str]) -> Safe:
    rows = server.db.all("SELECT q.*, m.name, o.title FROM questions q LEFT JOIN members m ON m.id = q.member LEFT JOIN"
                         " orders o ON o.id = q.order_id ORDER BY q.answer IS NOT NULL, q.id DESC LIMIT 200")  # fmt: skip
    items = []
    for q in rows:
        if q["answer"] is None:
            form = fmt('<form method="post" action="/panel/do/answer">{}<input type="hidden" name="id" value="{}"><p>'
                       '<textarea name="text" maxlength="{}" required></textarea></p><p><button type="submit"'
                       ' class="primary">Answer</button></p></form>', csrf_field(session), q["id"], MAX_SUMMARY)  # fmt: skip
        else:
            form = fmt('<p class="meta">answered by {}:</p><div class="text">{}</div>', q["answered_by"], q["answer"])
        items.append(fmt('<div class="card"><p class="meta">question #{} · <a href="/panel/order?id={}">order #{} {}</a>'
                         " · by {} · {}</p><div class=\"text\">{}</div>{}</div>", q["id"], q["order_id"],
                         q["order_id"], q["title"], q["name"], ago(q["at"]), q["text"], form))  # fmt: skip
    return fmt("<h1>Questions</h1><p class=\"meta\">Workers' questions about their orders (their own words).</p>{}",
               cat(items) if items else fmt("<p>No questions.</p>"))  # fmt: skip


def page_partners(server: Server, session: PanelSession, query: dict[str, str]) -> Safe:
    s = server.settings()
    rows = server.db.all("SELECT * FROM members WHERE role IN ('director', 'partner', 'worker') ORDER BY CASE role WHEN"
                         " 'director' THEN 0 WHEN 'partner' THEN 1 ELSE 2 END, name")  # fmt: skip
    owner = session.role == "owner"
    lines = []
    for r in rows:
        acts = Safe("")
        if r["role"] == "partner" and session.role in ("owner", "director"):
            acts = button(session, "untrust", {"member": r["id"]}, "Take the trust back")
        elif r["role"] == "worker" and owner:
            acts = button(session, "trust", {"member": r["id"]}, "Mark as trusted partner")
        if owner and r["role"] != "director":
            acts += button(session, "set_director", {"member": r["id"], "old": "partner"}, "Make director", "danger")
        lines.append(fmt("<tr><td>#{}</td><td><b>{}</b><br><span class=\"badge\">{}</span></td><td>{}</td><td>{}</td>"
                         "</tr>", r["id"], r["name"], ROLE_NAMES[r["role"]], identity_cell(r), acts))  # fmt: skip
    handovers = server.handovers()
    hitems = []
    for h in handovers:
        acts = Safe("")
        if h["status"] == "accepted" and owner:
            acts += button(session, "handover_confirm", {"id": h["id"]}, "Confirm handover", "primary")
        if session.role in ("owner", "director") or (session.member is not None and session.member["name"] == h["to"]):
            acts += button(session, "handover_cancel", {"id": h["id"]}, "Cancel handover", "danger")
        hitems.append(fmt('<div class="card"><p>Handover #{}: <b>{}</b> (fingerprint <code>{}</code>) → <b>{}</b>'
                          " (fingerprint <code>{}</code>) · {} · expires {}</p>{}</div>", h["id"], h["from"],
                          h["from_fp"], h["to"], h["to_fp"], "accepted, waits for the owner" if h["status"] ==
                          "accepted" else "offered, waits for the partner", until(h["expires"]), acts))  # fmt: skip
    how = fmt('<p class="meta">The director marks a trusted partner with its key (the fleet_trust tool, after its human'
              " compared the member's fingerprint or word phrase with the member's human over another channel, e.g. a"
              " call) and hands the director role over to a trusted partner the same way (fleet_handover; the partner"
              " accepts with fleet_accept_handover). A handover {} the owner's confirmation. The owner can always pick"
              " the director here or with <code>hub_server.py owner director</code>.</p>",
              "needs" if s["handover_needs_owner"] else "does not need")  # fmt: skip
    return fmt("<h1>Trusted partners and handover</h1>{}<table><tr><th>#</th><th>name, role</th><th>fingerprint, phrase"
               "</th><th></th></tr>{}</table><h2>Handovers</h2>{}", how, cat(lines),
               cat(hitems) if hitems else fmt("<p>No handover is open.</p>"))  # fmt: skip


PANEL_PAGES: dict[str, Callable[[Server, PanelSession, dict[str, str]], Safe]] = {
    "": page_members, "lobby": page_lobby, "invites": page_invites, "settings": page_settings,
    "messages": page_messages, "audit": page_audit, "orders": page_orders, "order": page_order,
    "approvals": page_approvals, "results": page_results, "questions": page_questions, "partners": page_partners,
}  # fmt: skip
PANEL_TITLES = {"": "Members", "lobby": "Lobby", "invites": "Invites", "settings": "Settings", "messages": "Messages",
                "audit": "Audit log", "orders": "Orders", "order": "Order", "approvals": "Approvals",
                "results": "Results", "questions": "Questions", "partners": "Partners"}  # fmt: skip
# Actions that change something for good: the panel asks first, on a page of their own (form field confirmed=yes).
CONFIRM = {"kick", "ban", "reject", "cancel", "refuse", "accept_result", "reject_result", "untrust", "trust",
           "handover_confirm", "handover_cancel", "set_director"}  # fmt: skip
PANEL_BACK = {"admit": "/panel/lobby", "reject": "/panel/lobby", "invite": "/panel/invites",
              "revoke_invite": "/panel/invites", "settings": "/panel/settings", "takeover": "/panel/settings",
              "send": "/panel/messages", "approve": "/panel/approvals", "refuse": "/panel/approvals",
              "accept_result": "/panel/results", "reject_result": "/panel/results", "answer": "/panel/questions",
              "untrust": "/panel/partners", "trust": "/panel/partners", "handover_confirm": "/panel/partners",
              "handover_cancel": "/panel/partners", "set_director": "/panel/partners"}  # fmt: skip


def confirm_page(server: Server, session: PanelSession, action: str, form: dict[str, str]) -> Safe:
    """What a destructive action will do, and a form that does it (the same fields, confirmed)."""
    def member() -> Any:
        return server.find(form.get("member"))

    def ident(row: Any) -> Safe:
        return fmt("{} (#{}, {}) · fingerprint <code>{}</code> · phrase <b>{}</b>", row["name"], row["id"],
                   ROLE_NAMES[row["role"]], row["fingerprint"], phrase(base64.b64decode(row["key"])))  # fmt: skip

    reason = form.get("reason", "")
    why = fmt("<p>Reason: <b>{}</b></p>", reason) if reason else Safe("")
    if action in ("kick", "ban", "reject"):
        row = member()
        text = {"kick": fmt("Remove {} from the fleet? It may ask to join again; its claimed orders go back up.",
                            ident(row)),
                "ban": fmt("Ban the key of {}? It is removed and can never join again with this key (a ban can be"
                           " lifted later).", ident(row)),
                "reject": fmt("Reject {}, waiting in the lobby?", ident(row))}[action]  # fmt: skip
        label = {"kick": "Remove", "ban": "Ban key", "reject": "Reject"}[action]
    elif action == "refuse":
        rows = [server.order(oid) for oid, _digest in server.approval_refs(form.get("orders"))]
        text = fmt("Refuse to dispatch {}? They are cancelled.", cat(fmt('{}order #{} "{}"', ", " if i else "", r["id"],
                                                                         r["title"]) for i, r in enumerate(rows)))  # fmt: skip
        label = "Refuse"
    elif action == "cancel":
        row = server.order(form.get("id"))
        info = server.order_info(row)
        text = fmt("Cancel order #{} \"{}\" ({})?{}", row["id"], row["title"], STATUS_NAMES[row["status"]],
                   f" Its claimer, {info['claimer']}, is told to stop." if info["claimer"] else "")  # fmt: skip
        label = "Cancel order"
    elif action in ("accept_result", "reject_result"):
        row = server.order(form.get("id"))
        rep = server.db.one("SELECT r.*, m.name FROM reports r LEFT JOIN members m ON m.id = r.member WHERE r.id = ?",
                            (str(form.get("report") or "0"),))  # fmt: skip
        who = rep["name"] if rep is not None else "?"
        if action == "accept_result":
            text = fmt("Accept report #{} by {} for order #{} \"{}\"? The order is done.{}", form.get("report"), who,
                       row["id"], row["title"], " Its files are applied on the computer that made the order"
                       " (fleet_accept there)." if row["kind"] == "workspace" else "")  # fmt: skip
            label = "Accept"
        else:
            then = {"rework": "it goes back to the same worker to fix", "queue": "it goes back to the queue for anyone",
                    "close": "the order is closed as rejected"}.get(form.get("then", "rework"), "?")  # fmt: skip
            text = fmt("Reject report #{} by {} for order #{} \"{}\"? Then {}.", form.get("report"), who, row["id"],
                       row["title"], then)  # fmt: skip
            label = "Reject"
    elif action in ("untrust", "trust"):
        row = member()
        text = fmt("Take the trust back from {}? It becomes a worker.", ident(row)) if action == "untrust" else fmt(
            "Mark {} as a trusted partner? Compare the fingerprint or the phrase with the member's human over another"
            " channel first.", ident(row))  # fmt: skip
        label = "Take the trust back" if action == "untrust" else "Mark as trusted partner"
    elif action in ("handover_confirm", "handover_cancel"):
        h = server.handover_row(form.get("id"))
        a, b = server.member(h["from_member"]), server.member(h["to_member"])
        if action == "handover_confirm":
            text = fmt("Confirm handover #{}: {} becomes the director, and {} a trusted partner? Everyone is told.",
                       h["id"], ident(b) if b is not None else "?", ident(a) if a is not None else "?")  # fmt: skip
            label = "Confirm handover"
        else:
            text = fmt("Cancel handover #{} to {}?", h["id"], ident(b) if b is not None else "?")
            label = "Cancel handover"
    elif action == "set_director":
        row = member()
        old = form.get("old", "partner")
        text = fmt("Make {} the director? Compare the fingerprint or the phrase with that person over another channel"
                   " first. The current director becomes {}. Everyone is told.", ident(row),
                   ROLE_NAMES.get(old, old))  # fmt: skip
        label = "Make director"
    else:
        raise Refused(404, "not_found", "no such action")
    hidden = cat(fmt('<input type="hidden" name="{}" value="{}">', k, v) for k, v in form.items() if k not in
                 ("csrf", "confirmed"))  # fmt: skip
    return fmt('<h1>Are you sure?</h1><div class="card"><p>{}</p>{}<form method="post" action="/panel/do/{}"'
               ' class="inline">{}{}<input type="hidden" name="confirmed" value="yes"><button type="submit"'
               ' class="danger">{}</button></form> <a href="{}">No, go back</a></div>', text, why, action,
               csrf_field(session), hidden, label, PANEL_BACK.get(action, "/panel"))  # fmt: skip


async def panel_endpoint(request: Any) -> Any:
    from starlette.responses import Response

    server: Server = request.app.state.server
    path = request.path_params.get("path", "").strip("/")
    method = request.method
    if not host_ok(server, request):
        return message_page("Wrong address", "Wrong Host header for this server.", 400)
    address = client_address(server, request)
    if server.page_limit.take(address):
        return message_page("Slow down", "Too many requests from this address.", 429, {"Retry-After": "30"})
    site = (request.headers.get("sec-fetch-site") or "").strip().lower()
    if site and site not in (("same-origin",) if method == "POST" else ("same-origin", "none")):
        return message_page("Refused", "Requests that another site starts are refused.", 403)
    form: dict[str, str] = {}
    if method == "POST":
        if not origin_ok(server, request):
            return message_page("Refused", "A change must come from this panel's own page.", 403)
        kind = request.headers.get("content-type", "").split(";")[0].strip().lower()
        if kind != "application/x-www-form-urlencoded":
            return message_page("Refused", "Changes are sent as the panel's forms.", 415)
        try:
            data = await read_body(request)
            form = {k: v[0] for k, v in parse_qs(data.decode("utf-8", "replace"), keep_blank_values=True,
                                                 max_num_fields=30).items()}  # fmt: skip
        except Refused as exc:
            return message_page("Refused", exc.message, exc.status)
        except ValueError:
            return message_page("Bad request", "The form could not be read.", 400)
    try:
        if path == "login":
            return await panel_login(server, request, method, form, address)
        session = panel_session(server, request)
        if session is None:
            return message_page("Not signed in", "Sign in with a one-time link: `agent_hub.py fleet panel-login` on the"
                                " director's computer, or `hub_server.py owner panel-login` on the server.", 403)  # fmt: skip
        if method == "GET":
            if path not in PANEL_PAGES:
                return message_page("Not found", "There is no such page here.", 404)
            flashes = server.flashes.pop(session.hash, [])
            query = {k: v for k, v in request.query_params.items() if len(k) < 40 and len(v) < 200}
            here = "/panel" + ("/" + path if path else "")
            return html_response(layout(PANEL_TITLES[path], PANEL_PAGES[path](server, session, query), session,
                                        "/panel/orders" if path == "order" else here, flashes))  # fmt: skip
        if not same(form.get("csrf", ""), session.csrf):
            return message_page("Refused", "This form is out of date or not from this panel: reload the page and try"
                                " again.", 403)  # fmt: skip
        if not path.startswith("do/"):
            return message_page("Not found", "There is no such action.", 404)
        return await panel_action(server, session, path[3:], form, address)
    except sqlite3.Error as exc:
        log(f"database error on {method} /panel/{path[:100]}: {exc}")
        return Response("Database busy; reload in a moment.", 503, {**HTML_HEADERS, "Retry-After": "2"})


async def panel_login(server: Server, request: Any, method: str, form: dict[str, str], address: str) -> Any:
    if server.auth_limit.take(address):
        return message_page("Slow down", "Too many sign-in attempts from this address.", 429, {"Retry-After": "60"})
    token = (request.query_params.get("t") if method == "GET" else form.get("t")) or ""
    row = server.db.one("SELECT * FROM panel_links WHERE hash = ?", (sha(token),)) if 0 < len(token) < 200 else None
    if row is None or row["expires"] < time.time():
        return message_page("Link not valid", "This sign-in link is not valid, expired or was used already. Make a new"
                            " one.", 403)  # fmt: skip
    member = server.member(row["member"]) if row["member"] is not None else None
    if row["member"] is not None and (member is None or member["role"] not in LEADS):
        return message_page("Not allowed", "The panel is for the director and trusted partners.", 403)
    who = "the owner" if member is None else f"{member['name']} ({ROLE_NAMES[member['role']]})"
    if method == "GET":  # GET changes nothing: link previews must not use the link up
        body = fmt('<h1>Sign in</h1><div class="card"><p>Sign in to {} as <b>{}</b>?</p><form method="post"'
                   ' action="/panel/login"><input type="hidden" name="t" value="{}"><button type="submit"'
                   ' class="primary">Sign in</button></form></div>', server.conf.get("name") or "the fleet", who,
                   token)  # fmt: skip
        return html_response(layout("Sign in", body))
    sid, csrf = secrets.token_urlsafe(32), secrets.token_urlsafe(32)
    with server.db.tx() as con:
        gone = con.execute("DELETE FROM panel_links WHERE hash = ?", (row["hash"],)).rowcount
        if not gone:
            return message_page("Link not valid", "This sign-in link was used already.", 403)
        con.execute("INSERT INTO panel_sessions VALUES (?, ?, ?, ?, ?, ?)", (sha(sid), row["member"], csrf, time.time(),
                                                                            time.time() + PANEL_SESSION_SECONDS,
                                                                            address))  # fmt: skip
        server.audit(Actor(member, address), "panel_login", row["member"], f"{who} signed in to the panel")
    from starlette.responses import Response

    return Response(b"", 303, {**HTML_HEADERS, "Location": "/panel",
                               "Set-Cookie": session_cookie(server, sid, PANEL_SESSION_SECONDS)})  # fmt: skip


async def panel_action(server: Server, session: PanelSession, action: str, form: dict[str, str], address: str) -> Any:
    from starlette.responses import Response

    actor = Actor(session.member, address)
    back = PANEL_BACK.get(action, "/panel")
    if action in ("cancel", "edit", "reassign") and form.get("id", "").isdigit():
        back = f"/panel/order?id={form['id']}"
    try:
        if action == "logout":
            with server.db.tx() as con:
                con.execute("DELETE FROM panel_sessions WHERE hash = ?", (session.hash,))
            return message_page("Signed out", "This browser is signed out of the panel.", 200,
                                {"Set-Cookie": session_cookie(server, "", 0)})  # fmt: skip
        if action == "trust" and not actor.owner:
            raise Refused(403, "forbidden", "the director marks a trusted partner with its own key (fleet_trust, after"
                          " comparing the fingerprint or phrase); in the panel only the owner can")  # fmt: skip
        if action in CONFIRM and form.get("confirmed") != "yes":
            return html_response(layout("Are you sure?", confirm_page(server, session, action, form), session,
                                        back.split("?")[0]))  # fmt: skip
        if action == "admit":
            row = server.admit(actor, form.get("member"))
            flash(server, session, "ok", f"{row['name']} is admitted as a worker.")
        elif action in ("reject", "kick", "ban"):
            raw = form.get("member", "")
            target = server.member(int(raw)) if raw.isdigit() and len(raw) < 10 else None
            row = server.remove(actor, form.get("member"), action, form.get("reason", ""))
            flash(server, session, "ok", f"{row['name']}: " + {"reject": "rejected", "kick": "removed from the fleet",
                                                               "ban": "key banned"}[action] + ".")  # fmt: skip
            if action == "ban" and target is not None and target["role"] == "lobby":
                back = "/panel/lobby"
        elif action == "unban":
            row = server.unban(actor, form.get("member"))
            flash(server, session, "ok", f"{row['name']}: the ban is lifted (it may join again).")
        elif action == "invite":
            token, _row = server.create_invite(actor, form.get("uses") or 1, form.get("hours") or 24,
                                               form.get("note", ""))  # fmt: skip
            body = page_invites(server, session, {}, invite_link(server.options.public_url, token, server.key_fp))
            return html_response(layout("Invites", body, session, "/panel/invites"))
        elif action == "revoke_invite":
            server.revoke_invite(actor, form.get("id"))
            flash(server, session, "ok", f"Invite #{form.get('id')} is revoked.")
        elif action == "settings":
            changes: dict[str, Any] = {}
            for key in ("join_mode", "approval", "dispatch_approval", "claim_minutes", "requeue_offline_seconds"):
                if form.get(key):
                    changes[key] = form[key]
            if form.get("paused"):
                changes["paused"] = {"paused": True, "running": False}.get(form["paused"], form["paused"])
            done = server.change_settings(actor, changes, human=True)
            flash(server, session, "ok", ("Changed: " + ", ".join(done) + ".") if done else "Nothing changed.")
        elif action == "takeover":
            done = server.change_settings(actor, {"takeover": form.get("on", "")}, human=True)
            on = server.settings()["takeover"]
            flash(server, session, "ok", ("You took over: the leads' agents cannot act as leads until you hand back."
                                          if on else "Handed back: the leads' agents act again.") if done else
                  "Nothing changed.")  # fmt: skip
        elif action == "send":
            _event, to = server.send(actor, form.get("to"), form.get("text"))
            flash(server, session, "ok", f"Sent to {to}.")
        elif action in ("approve", "refuse"):
            rows = server.approve(actor, form.get("orders"), action == "approve", form.get("reason", ""))
            flash(server, session, "ok", f"{'Approved' if action == 'approve' else 'Refused'}: "
                  + (", ".join(f"#{r['id']}" for r in rows) or "nothing waited") + ".")  # fmt: skip
        elif action == "cancel":
            row = server.cancel(actor, form.get("id"), form.get("reason", ""))
            flash(server, session, "ok", f"Order #{row['id']} is cancelled.")
        elif action == "edit":
            spec = {k: form[k] for k in ("title", "instructions", "priority", "deadline_hours") if form.get(k)}
            row = server.edit(actor, form.get("id"), spec)
            flash(server, session, "ok", f"Order #{row['id']} is changed.")
        elif action == "reassign":
            row = server.reassign(actor, form.get("id"), form.get("to") or "any")
            flash(server, session, "ok", f"Order #{row['id']} is now for "
                  + (", ".join(server.order_info(row)["to"]) or "anyone") + ".")  # fmt: skip
        elif action in ("accept_result", "reject_result"):
            row = server.decide(actor, form.get("id"), form.get("report"), action == "accept_result",
                                form.get("reason", ""), form.get("then") or "rework")  # fmt: skip
            flash(server, session, "ok", f"Order #{row['id']}: the result is "
                  + ("accepted." if action == "accept_result" else f"rejected; the order is {STATUS_NAMES[row['status']]}."))  # fmt: skip
        elif action == "answer":
            q = server.answer(actor, form.get("id"), form.get("text"))
            flash(server, session, "ok", f"Answered question #{q['id']}; the worker is told.")
        elif action in ("trust", "untrust"):
            row = server.trust(actor, form.get("member"), revoke=action == "untrust")
            flash(server, session, "ok", f"{row['name']} is {ROLE_NAMES[row['role']]} now.")
        elif action == "handover_confirm":
            h = server.handover_confirm(actor, form.get("id"))
            flash(server, session, "ok", f"Handover #{h['id']} is done.")
        elif action == "handover_cancel":
            h = server.handover_cancel(actor, form.get("id"))
            flash(server, session, "ok", f"Handover #{h['id']} is cancelled.")
        elif action == "set_director":
            old_role = form.get("old") or "partner"
            new, old = server.set_director(actor, form.get("member"), old_role)
            flash(server, session, "ok", f"{new['name']} is the director" + (f"; {old['name']} is now"
                  f" {ROLE_NAMES[old_role]}." if old is not None else "."))  # fmt: skip
        else:
            return message_page("Not found", "There is no such action.", 404)
    except Refused as exc:
        flash(server, session, "error", exc.message)
    return Response(b"", 303, {**HTML_HEADERS, "Location": back})


# --------------------------------------------------------------------------
# Serving
# --------------------------------------------------------------------------


def build_app(server: Server) -> Any:
    from starlette.applications import Starlette
    from starlette.routing import Route

    async def not_found(request: Any) -> Any:
        return message_page("Not found", "There is no such page here.", 404)

    @contextlib.asynccontextmanager
    async def lifespan(_app: Any):
        task = asyncio.get_running_loop().create_task(server.background())
        try:
            yield
        finally:
            task.cancel()
            with contextlib.suppress(sqlite3.Error):
                server.flush_presence()

    routes = [
        Route("/", page_home, methods=["GET"]),
        Route("/join/{token}", page_join, methods=["GET"]),
        Route("/api/v1/{path:path}", api_endpoint, methods=["GET", "POST"]),
        Route("/panel", panel_endpoint, methods=["GET", "POST"]),
        Route("/panel/{path:path}", panel_endpoint, methods=["GET", "POST"]),
        Route("/{path:path}", not_found, methods=["GET", "POST"]),
    ]
    app = Starlette(routes=routes, lifespan=lifespan)
    app.state.server = server
    return app


def server_home(raw: str | None) -> Path:
    return Path(raw or os.environ.get("AGENT_HUB_SERVER_HOME") or DEFAULT_HOME).expanduser()


def load_conf(home: Path) -> dict[str, Any]:
    path = home / "server.json"
    try:
        conf = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise SystemExit(f"hub_server.py: no fleet server in {home}: run `hub_server.py owner init` first"
                         " (or pass --home)") from None  # fmt: skip
    if not isinstance(conf, dict) or not SERVER_ID_RE.match(str(conf.get("server_id", ""))):
        raise SystemExit(f"hub_server.py: {path} is damaged (no valid server_id)")
    return conf


def check_public_url(url: str) -> str:
    url = url.strip().rstrip("/")
    parts = urlsplit(url)
    if parts.scheme not in ("https", "http") or not parts.hostname or parts.path or parts.query or parts.fragment:
        raise SystemExit(f"hub_server.py: --public-url must be like https://hub.example.com (got {url!r})")
    if parts.scheme == "http" and not is_loopback(parts.hostname):
        raise SystemExit("hub_server.py: the public URL must use https (plain http only for 127.0.0.1 / localhost)")
    return url


def serve(args: argparse.Namespace) -> int:
    home = server_home(args.home)
    conf = load_conf(home)
    tls = bool(args.tls_cert or args.tls_key)
    if tls and not (args.tls_cert and args.tls_key):
        raise SystemExit("hub_server.py: give both --tls-cert and --tls-key")
    loopback = is_loopback(args.host)
    if not loopback and not tls:
        if not args.insecure_plain_http:
            raise SystemExit(f"hub_server.py: refusing to listen on {args.host} without TLS. Either keep the default"
                             " 127.0.0.1 behind a TLS reverse proxy on this machine (--behind-proxy; see the README:"
                             " Caddy), or serve TLS here (--tls-cert/--tls-key). Only if a TLS proxy on another"
                             " machine forwards to this address over a private network, add --behind-proxy"
                             " --insecure-plain-http.")  # fmt: skip
        if not args.behind_proxy:
            raise SystemExit("hub_server.py: --insecure-plain-http only goes with --behind-proxy")
        log("WARNING " + "!" * 60)
        log(f"WARNING: plain HTTP on {args.host}:{args.port} (--insecure-plain-http). Everything, tokens included,")
        log("WARNING: crosses the network unencrypted between the proxy and this server. Firewall this port so only")
        log("WARNING: the TLS proxy can reach it.")
        log("WARNING " + "!" * 60)
    public = args.public_url or conf.get("public_url") or ""
    if not public:
        if args.behind_proxy:
            raise SystemExit("hub_server.py: --behind-proxy needs the public https URL (--public-url or owner init"
                             " --public-url)")  # fmt: skip
        if not loopback:  # members sign for this server's public origin: it must be the real one
            raise SystemExit("hub_server.py: on a public interface the server needs its public https URL (--public-url"
                             " or owner init --public-url)")  # fmt: skip
        public = f"https://{args.host}:{args.port}" if tls else f"http://127.0.0.1:{args.port}"
    public = check_public_url(public)
    if (tls or args.behind_proxy) and not public.startswith("https://"):
        raise SystemExit("hub_server.py: with TLS or behind a proxy the public URL must use https")
    if not tls and not args.behind_proxy and public.startswith("https://"):
        raise SystemExit("hub_server.py: the public URL is https but this server serves plain HTTP: add"
                         " --behind-proxy (a TLS proxy in front) or --tls-cert/--tls-key")  # fmt: skip
    trusted = tuple(args.trusted_proxy or ("127.0.0.1", "::1"))
    options = Options(host=args.host, port=args.port, public_url=public, tls_cert=args.tls_cert or "",
                      tls_key=args.tls_key or "", behind_proxy=args.behind_proxy, trusted_proxies=trusted,
                      token_seconds=args.token_minutes * 60, offline_after=args.offline_after,
                      auth_per_minute=args.auth_per_minute, api_per_minute=args.api_per_minute,
                      messages_per_minute=args.messages_per_minute, start_grace=args.start_grace)  # fmt: skip
    import uvicorn

    if tls:  # this server terminates TLS: browsers keep to https for it (behind a proxy, the proxy sends it)
        for headers in (JSON_HEADERS, HTML_HEADERS):
            headers["Strict-Transport-Security"] = "max-age=31536000"
    server = Server(home, conf, options)
    app = build_app(server)
    config = uvicorn.Config(app, host=args.host, port=args.port, ssl_certfile=args.tls_cert or None,
                            ssl_keyfile=args.tls_key or None, log_level="warning", access_log=False,
                            proxy_headers=False, server_header=False, http="h11", lifespan="on",
                            limit_concurrency=args.max_connections, timeout_keep_alive=KEEP_ALIVE,
                            timeout_graceful_shutdown=3,
                            h11_max_incomplete_event_size=16 * 1024)  # fmt: skip
    print(f"Agent Hub fleet server {__version__}: {conf.get('name') or 'fleet'} ({conf['server_id']})", flush=True)
    print(f"  listening on {args.host}:{args.port}{' with TLS' if tls else ''}; public URL {public}", flush=True)
    print(f"  server key fingerprint {server.key_fp} (phrase \"{phrase(server.key_raw)}\"): members pin it when they"
          " join", flush=True)
    print(f"  data in {home}", flush=True)
    uvicorn.Server(config).run()
    return 0


# --------------------------------------------------------------------------
# The owner's command line (on the VPS; it writes the database directly, the running server picks it up at once)
# --------------------------------------------------------------------------


def owner_fleet(args: argparse.Namespace) -> Fleet:
    home = server_home(args.home)
    return Fleet(home, load_conf(home))


def owner_init(args: argparse.Namespace) -> int:
    home = server_home(args.home)
    home.mkdir(parents=True, exist_ok=True)
    with contextlib.suppress(OSError):
        os.chmod(home, 0o700)
    path = home / "server.json"
    conf: dict[str, Any] = {}
    if path.exists():
        conf = load_conf(home)
    created = not conf
    conf.setdefault("server_id", "fs-" + secrets.token_hex(6))
    conf.setdefault("created", time.time())
    if args.name is not None:
        conf["name"] = one_line(args.name, 80)
    conf.setdefault("name", "Agent Hub fleet")
    if args.public_url is not None:
        conf["public_url"] = check_public_url(args.public_url) if args.public_url else ""
    private_file(path)
    path.write_text(json.dumps(conf, indent=2) + "\n", encoding="utf-8")
    fleet = Fleet(home, conf)
    with fleet.db.tx():
        fleet.audit("the owner", "init" if created else "config", None, f"server {conf['server_id']} name"
                    f" {conf['name']!r} public URL {conf.get('public_url') or '(not set)'}")  # fmt: skip
    print(f"{'Created' if created else 'Updated'} fleet server {conf['server_id']} ({conf['name']}) in {home}")
    print(f"Server key fingerprint {fleet.key_fp}\n           phrase      {phrase(fleet.key_raw)}\nMembers pin this"
          " key when they join (invite links carry it); keep server.key with the database (hub_server.py owner key"
          " shows it again).")  # fmt: skip
    if not conf.get("public_url"):
        print("Set the public URL (https://...) with --public-url before people join over the internet.")
    s = fleet.settings()
    print(f"Join mode {s['join_mode']}, approval {'on' if s['approval'] else 'off'}. Next: start it (hub_server.py"
          " serve), let the future director join (an invite: hub_server.py owner invite), then make it the director"
          " (hub_server.py owner director <name>) after comparing its fingerprint.")  # fmt: skip
    return 0


def owner_key(args: argparse.Namespace) -> int:
    """This server's key: give its fingerprint to members (over another channel) so they can pin or re-pin it."""
    fleet = owner_fleet(args)
    print(f"Server {fleet.server_id} ({fleet.conf.get('name') or 'fleet'}), public URL"
          f" {fleet.conf.get('public_url') or '(not set)'}")  # fmt: skip
    print(f"  server key fingerprint {fleet.key_fp}\n  phrase                 {phrase(fleet.key_raw)}")
    print("Members whose agent_hub.py joined before servers had keys re-pin it with fleet_join(server=<url>,"
          " server_key=<fingerprint>) or agent_hub.py fleet pin.")  # fmt: skip
    return 0


def owner_members(args: argparse.Namespace) -> int:
    fleet = owner_fleet(args)
    roles = ROLES if args.all else ACTIVE + ("banned",)
    marks = ",".join("?" * len(roles))
    rows = fleet.db.all(f"SELECT * FROM members WHERE role IN ({marks}) ORDER BY CASE role WHEN 'director' THEN 0 WHEN"
                        " 'partner' THEN 1 WHEN 'worker' THEN 2 WHEN 'lobby' THEN 3 ELSE 4 END, name", roles)  # fmt: skip
    if not rows:
        print("No members yet.")
    for r in rows:
        presence, seen = fleet.presence(r)
        print(f"#{r['id']:<5} {one_line(r['name'], 32):<20} {ROLE_NAMES[r['role']]:<16} {presence:<8}"
              f" {r['fingerprint']}  {phrase(base64.b64decode(r['key']))}")  # fmt: skip
        extra = [f"agent: {one_line(r['agent'], 80)}" if r["agent"] else "", f"from {r['address']}" if r["address"]
                 else "", f"last seen {ago(seen)}", r["removed_reason"]]  # fmt: skip
        print("       " + " · ".join(e for e in extra if e))
    return 0


def owner_director(args: argparse.Namespace) -> int:
    fleet = owner_fleet(args)
    target = fleet.find(args.member, ("partner", "worker", "lobby", "director"))
    print(f"Make {target['name']} (#{target['id']}) the director?\n  fingerprint {target['fingerprint']}\n  phrase"
          f"      {phrase(base64.b64decode(target['key']))}")  # fmt: skip
    if not args.yes and not confirm("Compare these with the person over another channel. Continue?"):
        print("Nothing changed.")
        return 1
    new, old = fleet.set_director(Actor(None, "owner CLI"), target["id"], args.old)
    print(f"Done: {new['name']} is the director" + (f"; {old['name']} is now {ROLE_NAMES[args.old]}." if old is not None
                                                    else "."))  # fmt: skip
    return 0


def owner_remove(args: argparse.Namespace, how: str) -> int:
    fleet = owner_fleet(args)
    actor = Actor(None, "owner CLI")
    if how == "unban":
        row = fleet.unban(actor, args.member)
        print(f"Done: the ban of {row['name']} (#{row['id']}) is lifted.")
        return 0
    if how == "revoke":
        row = fleet.find(args.member, ACTIVE)
        with fleet.db.tx():
            n = fleet.revoke_tokens(row["id"])
            fleet.audit(actor, "revoke", row["id"], f"{member_label(row)}: {n} access token(s) and panel sessions"
                        " revoked")  # fmt: skip
        print(f"Done: {n} access token(s) of {row['name']} revoked; it has to log in again (with its key).")
        return 0
    target = fleet.find(args.member, ("director", "partner", "worker", "lobby", "removed"))
    if target["role"] == "director":
        print("That is the director: pick another director first (hub_server.py owner director <member>).")
        return 1
    row = fleet.remove(actor, target["id"], how, args.reason or "")
    print(f"Done: {row['name']} (#{row['id']}) " + {"kick": "removed (it may join again)", "ban": "banned (its key"
          " cannot join again)", "reject": "rejected"}[how] + ".")  # fmt: skip
    return 0


def owner_settings(args: argparse.Namespace) -> int:
    fleet = owner_fleet(args)
    changes: dict[str, Any] = {}
    for key in ("join_mode", "approval", "dispatch_approval", "takeover", "handover_needs_owner", "claim_minutes",
                "requeue_offline_seconds", "new_keys_per_hour", "lobby_max", "members_max", "max_order_bytes",
                "max_report_bytes", "storage_mb", "orders_open_max"):  # fmt: skip
        value = getattr(args, key, None)
        if value is not None:
            changes[key] = value
    if args.pause:
        changes["paused"] = True
    if args.resume:
        changes["paused"] = False
    if changes:
        done = fleet.change_settings(Actor(None, "owner CLI"), changes)
        print("Changed: " + ", ".join(done) + "." if done else "Nothing changed.")
    for key, value in fleet.settings().items():
        print(f"  {key:<24} {('on' if value else 'off') if isinstance(value, bool) else value}")
    return 0


def owner_invite(args: argparse.Namespace) -> int:
    fleet = owner_fleet(args)
    actor = Actor(None, "owner CLI")
    if args.revoke is not None:
        fleet.revoke_invite(actor, args.revoke)
        print(f"Done: invite #{args.revoke} is revoked.")
        return 0
    if args.list:
        for r in fleet.invites():
            print(f"#{r['id']:<4} {r['state']:<12} {r['uses']}/{r['max_uses']} uses, expires {until(r['expires'])},"
                  f" by {one_line(r['created_by'], 40)} {one_line(r['note'], 60)}")  # fmt: skip
        return 0
    token, row = fleet.create_invite(actor, args.uses, args.hours, args.note or "")
    base = fleet.conf.get("public_url") or "https://<this server's public URL>"
    print(f"Invite #{row['id']} ({row['max_uses']} use(s), valid {until(row['expires'])}). The link (shown once):")
    print(f"\n    {invite_link(base, token, fleet.key_fp)}\n")
    return 0


def owner_trust(args: argparse.Namespace) -> int:
    fleet = owner_fleet(args)
    target = fleet.find(args.member, ("worker", "partner"))
    print(f"{'Take the trust back from' if args.revoke else 'Mark as a trusted partner'}: {target['name']}"
          f" (#{target['id']})\n  fingerprint {target['fingerprint']}\n  phrase      "
          f"{phrase(base64.b64decode(target['key']))}")  # fmt: skip
    if not args.yes and not confirm("Compare these with the person over another channel. Continue?"):
        print("Nothing changed.")
        return 1
    row = fleet.trust(Actor(None, "owner CLI"), target["id"], revoke=args.revoke)
    print(f"Done: {row['name']} is {ROLE_NAMES[row['role']]} now.")
    return 0


def owner_handovers(args: argparse.Namespace) -> int:
    fleet = owner_fleet(args)
    actor = Actor(None, "owner CLI")
    if args.confirm is not None or args.cancel is not None:
        hid = args.confirm if args.confirm is not None else args.cancel
        h = fleet.handover_row(hid)
        a, b = fleet.member(h["from_member"]), fleet.member(h["to_member"])
        print(f"Handover #{h['id']} ({h['status']}): {a['name'] if a else '?'} (fingerprint {h['from_fp']}) ->"
              f" {b['name'] if b else '?'} (fingerprint {h['to_fp']})")  # fmt: skip
        if not args.yes and not confirm("Confirm this handover?" if args.confirm is not None else "Cancel it?"):
            print("Nothing changed.")
            return 1
        if args.confirm is not None:
            fleet.handover_confirm(actor, h["id"])
            print(f"Done: {b['name'] if b else '?'} is the director now; everyone is told.")
        else:
            fleet.handover_cancel(actor, h["id"])
            print(f"Done: handover #{h['id']} is cancelled.")
        return 0
    rows = fleet.handovers(open_only=not args.all)
    if not rows:
        print("No handover is open." if not args.all else "No handovers yet.")
    for h in rows:
        print(f"#{h['id']:<4} {h['status']:<10} {one_line(h['from'], 32)} -> {one_line(h['to'], 32)}"
              f" (fingerprint {h['to_fp']}), expires {until(h['expires'])}")  # fmt: skip
    return 0


def owner_orders(args: argparse.Namespace) -> int:
    fleet = owner_fleet(args)
    for o in fleet.orders(Actor(None, "owner CLI"), args.status or "open", limit=max(1, min(args.last, 1000))):
        print(f"#{o['id']:<5} {STATUS_NAMES.get(o['status'], o['status']):<30} {o['kind']:<9} {one_line(o['title'], 60)}"
              f"  for {', '.join(o['to']) or 'anyone'}" + (f", claimed by {o['claimer']}" if o["claimer"] else ""))  # fmt: skip
    return 0


def owner_panel_login(args: argparse.Namespace) -> int:
    fleet = owner_fleet(args)
    token = fleet.panel_link(None)
    with fleet.db.tx():
        fleet.audit("the owner", "panel_link", None, "a one-time panel link for the owner (command line)")
    base = args.url or fleet.conf.get("public_url") or f"http://127.0.0.1:{DEFAULT_PORT}"
    print(f"Open this link within {PANEL_LINK_SECONDS // 60} minutes; it works once and signs that browser in as the"
          " owner:")  # fmt: skip
    print(f"\n    {base.rstrip('/')}/panel/login?t={token}\n")
    return 0


def owner_panel_logout(args: argparse.Namespace) -> int:
    fleet = owner_fleet(args)
    with fleet.db.tx() as con:
        n = con.execute("DELETE FROM panel_sessions").rowcount
        con.execute("DELETE FROM panel_links")
        fleet.audit("the owner", "panel_logout", None, f"every browser signed out of the panel ({n} session(s))")
    print(f"Done: {n} panel session(s) ended; unused sign-in links are void too.")
    return 0


def owner_audit(args: argparse.Namespace) -> int:
    fleet = owner_fleet(args)
    rows = fleet.db.all("SELECT * FROM audit ORDER BY id DESC LIMIT ?", (max(1, min(args.last, 10_000)),))
    for r in reversed(rows):
        when = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(r["at"]))
        print(f"#{r['id']:<5} {when} {one_line(r['actor'], 60)} {r['action']}: {one_line(r['detail'], 300)}"
              f"{' [' + r['address'] + ']' if r['address'] else ''}")  # fmt: skip
    return 0


def confirm(question: str) -> bool:
    if not sys.stdin.isatty():
        print(f"{question} (not a terminal: pass --yes to confirm)")
        return False
    return input(f"{question} [y/N] ").strip().lower() in ("y", "yes")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="hub_server.py", description="Agent Hub fleet server (fleet mode).")
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="command", required=True)

    def home(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--home", help=f"the server's data folder (default $AGENT_HUB_SERVER_HOME or {DEFAULT_HOME})")

    s = sub.add_parser("serve", help="run the server")
    home(s)
    s.add_argument("--host", default="127.0.0.1", help="address to listen on (default 127.0.0.1)")
    s.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"port (default {DEFAULT_PORT})")
    s.add_argument("--public-url", help="the https URL people reach this server at (default: from owner init)")
    s.add_argument("--tls-cert", help="serve TLS with this certificate file (PEM)")
    s.add_argument("--tls-key", help="... and this private key file (PEM)")
    s.add_argument("--behind-proxy", action="store_true", help="a TLS reverse proxy forwards to this server: trust its"
                   " X-Forwarded-For (from --trusted-proxy addresses only); secure cookies")  # fmt: skip
    s.add_argument("--trusted-proxy", action="append", help="the proxy's address (default 127.0.0.1 and ::1)")
    s.add_argument("--insecure-plain-http", action="store_true", help="allow plain HTTP on a non-loopback address"
                   " (only behind a TLS proxy on a private network; loudly warned)")  # fmt: skip
    s.add_argument("--token-minutes", type=int, default=15, choices=range(1, 1441), metavar="1-1440",
                   help="access token lifetime (default 15)")  # fmt: skip
    s.add_argument("--offline-after", type=int, default=20, choices=range(3, 3601), metavar="3-3600",
                   help="seconds without contact before a member shows offline (default 20)")  # fmt: skip
    s.add_argument("--auth-per-minute", type=int, default=20, help="logins and challenges per address and minute")
    s.add_argument("--api-per-minute", type=int, default=600, help="API requests per member and minute")
    s.add_argument("--messages-per-minute", type=int, default=30, help="messages per member and minute")
    s.add_argument("--max-connections", type=int, default=2000, help="open connections at most")
    s.add_argument("--start-grace", type=int, default=START_GRACE, choices=range(0, 3601), metavar="0-3600",
                   help="seconds after a start in which no claim goes back up for silence: members reconnect first"
                   f" (default {START_GRACE}; the time the server was down never counts)")  # fmt: skip

    o = sub.add_parser("owner", help="the owner's commands (on this machine)")
    osub = o.add_subparsers(dest="action", required=True)
    i = osub.add_parser("init", help="create (or update) this fleet server")
    home(i)
    i.add_argument("--name")
    i.add_argument("--public-url", help="the https URL people reach this server at")
    k = osub.add_parser("key", help="show this server's key (fingerprint and phrase) for members to pin")
    home(k)
    m = osub.add_parser("members", help="list members")
    home(m)
    m.add_argument("--all", action="store_true", help="also the removed ones")
    d = osub.add_parser("director", help="make a member the director (replaces the current one)")
    home(d)
    d.add_argument("member", help="#id, name or fingerprint")
    d.add_argument("--old", choices=("worker", "partner", "removed"), default="worker",
                   help="what the old director becomes (default worker)")  # fmt: skip
    d.add_argument("--yes", action="store_true", help="do not ask")
    for name, text in (("kick", "remove a member (it may join again)"), ("ban", "ban a member's key"),
                       ("reject", "reject a member in the lobby"), ("unban", "lift a ban"),
                       ("revoke", "revoke a member's access tokens and panel sessions")):  # fmt: skip
        r = osub.add_parser(name, help=text)
        home(r)
        r.add_argument("member", help="#id, name or fingerprint")
        if name in ("kick", "ban", "reject"):
            r.add_argument("--reason")
    st = osub.add_parser("settings", help="show or change settings")
    home(st)
    st.add_argument("--join", dest="join_mode", choices=JOIN_MODES)
    st.add_argument("--approval", choices=("on", "off"))
    st.add_argument("--pause", action="store_true")
    st.add_argument("--resume", action="store_true")
    for flag in ("--dispatch-approval", "--takeover", "--handover-needs-owner"):
        st.add_argument(flag, choices=("on", "off"))
    for flag in ("--claim-minutes", "--requeue-offline-seconds", "--new-keys-per-hour", "--lobby-max", "--members-max",
                 "--max-order-bytes", "--max-report-bytes", "--storage-mb", "--orders-open-max"):  # fmt: skip
        st.add_argument(flag, type=int)
    tr = osub.add_parser("trust", help="mark a worker as a trusted partner (or --revoke)")
    home(tr)
    tr.add_argument("member", help="#id, name or fingerprint")
    tr.add_argument("--revoke", action="store_true")
    tr.add_argument("--yes", action="store_true", help="do not ask")
    ho = osub.add_parser("handovers", help="list open handovers, confirm or cancel one")
    home(ho)
    ho.add_argument("--confirm", type=int, metavar="ID")
    ho.add_argument("--cancel", type=int, metavar="ID")
    ho.add_argument("--all", action="store_true", help="also the closed ones")
    ho.add_argument("--yes", action="store_true", help="do not ask")
    od = osub.add_parser("orders", help="list orders")
    home(od)
    od.add_argument("--status", help="open (default), final, all or one status")
    od.add_argument("--last", type=int, default=100)
    iv = osub.add_parser("invite", help="create, list or revoke invites")
    home(iv)
    iv.add_argument("--uses", type=int, default=1)
    iv.add_argument("--hours", type=float, default=24)
    iv.add_argument("--note")
    iv.add_argument("--list", action="store_true")
    iv.add_argument("--revoke", type=int, metavar="ID")
    pl = osub.add_parser("panel-login", help="a one-time link that signs a browser in to the panel as the owner")
    home(pl)
    pl.add_argument("--url", help="the base URL to print (default: the public URL)")
    po = osub.add_parser("panel-logout", help="sign every browser out of the panel (the owner's too)")
    home(po)
    a = osub.add_parser("audit", help="show the audit log")
    home(a)
    a.add_argument("--last", type=int, default=50)
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "serve":
            return serve(args)
        action = args.action
        if action == "init":
            return owner_init(args)
        handlers: dict[str, Callable[[argparse.Namespace], int]] = {
            "members": owner_members, "key": owner_key, "director": owner_director, "settings": owner_settings,
            "invite": owner_invite, "panel-login": owner_panel_login, "panel-logout": owner_panel_logout,
            "audit": owner_audit, "trust": owner_trust, "handovers": owner_handovers, "orders": owner_orders,
            "kick": lambda a: owner_remove(a, "kick"), "ban": lambda a: owner_remove(a, "ban"),
            "reject": lambda a: owner_remove(a, "reject"), "unban": lambda a: owner_remove(a, "unban"),
            "revoke": lambda a: owner_remove(a, "revoke"),
        }  # fmt: skip
        return handlers[action](args)
    except Refused as exc:
        print(f"hub_server.py: {exc.message}", file=sys.stderr)
        return 1
    except sqlite3.Error as exc:
        print(f"hub_server.py: database error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
