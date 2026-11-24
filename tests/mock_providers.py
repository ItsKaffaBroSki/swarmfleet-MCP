"""Local stand-ins for the API shapes the hub speaks.

Routes (one prefix per provider so the test can tell them apart):
    POST /anthropic/v1/messages        Anthropic Messages API, server-sent events
    POST /openai/responses             OpenAI Responses API
    POST /<name>/chat/completions      OpenAI-compatible chat completions (deepseek, zai, openrouter, ...)

Every request is recorded; the test reads them back from `MockProviders.requests`.
The model id selects special behaviour:
    slow-N      sleep N seconds first          fail-401   always HTTP 401
    flaky*      HTTP 529/503 once, then fine   empty      answer with no text
    long        a 150,000 character answer     cutoff     answer truncated at the limit
    bodyfail*   (chat) HTTP 200 with DeepSeek's documented failure body ({"error", "usage"}) once, then fine
    refusal     (anthropic) partial text, then stop_reason "refusal"
    oddstop     partial answer with an unusual stop reason (sensitive / context window / content_filter)
    pause*      (anthropic) stop_reason "pause_turn" first; the continuation finishes the answer. pause2*: pauses
                twice. pausetool*: the continuation calls the tool the prompt names (TOOL:...), then answers.
    cite        (anthropic) an answer whose text block carries citations (citations_delta)
    cuttool     (with tools) a write_file call cut off by the output limit, every time (Claude: max_tokens with the
                tool_use's JSON cut; Responses: incomplete, max_output_tokens; chat: finish_reason length)
    cutonce     the same once; after the hub's note that the call was cut, an answer
    cutafter-<stop>  (with tools) a complete read_file call first; after its result, a write_file call cut off with
                that stop reason (Claude's stop_reason, the Responses API's incomplete reason, chat's finish_reason);
                after the hub's note, an answer. cutagain-<stop>: cut again after the note. A stop written
                <stop>~<native> also gives a chat answer native_finish_reason <native> (OpenRouter's raw reason).
    nostatus*   (Responses) every response object without its `status` field (optional in the API reference)

The Anthropic route streams like the real API: every block opens with content_block_start; answers carry a
thinking block (summarized for plain answers; omitted, i.e. empty text plus a signature, and a redacted_thinking
block before a tool call). Like the real Messages API for current Claude models it rejects a conversation that
ends with an assistant message (no prefill), unless that message continues a paused turn: one the mock answered
with pause_turn, sent back as it came, or one with a server tool's blocks (server_tool_use, web_search_tool_result).

Tool use: a prompt containing TOOL:<name>:<json arguments> makes the model call that hub tool (when the request
offers tools); TOOLLOOP makes it call read_file forever. With TOOLSEQ in the prompt it calls every TOOL: marker of
the prompt in order, one per round, and then answers with all their results. On the next request each API checks that the tool turn
came back the way the real one requires (Claude: the thinking blocks unmodified; OpenAI: the reasoning item with
its encrypted_content, which this mock returns only when the request has include=["reasoning.encrypted_content"],
the legacy way to ask for it; chat completions: every tool_call_id answered, reasoning_content, and OpenRouter's
reasoning_details unmodified) and answers "<model> tool results: ...". Every tool round gets ids of its own
(toolu_N, call_N), as the real APIs give unique ids. DeepSeek also rejects any earlier assistant turn without
reasoning_content when tools are offered, as its docs say. OpenRouter accepts reasoning_details only on the tool-call
turns of the answer being made (after the last user message: the whole sequence must come back), and rejects them
on any earlier answer: in a later turn the hub rebuilds the history, and reasoning blocks cannot move into a changed
history (Claude's prefix check). The model "notools" answers 400 whenever tools are offered.

Also: hosted web search (Claude's web_search_20250305 tool, OpenAI's web_search tool, DeepSeek's
deepseek_web_search, which fails once with its HTTP 200 failure body for a query with FLAKY, Z.ai's web_search
tool), Tavily (POST /tavily/search; a query with FLAKY fails once with 503),
Brave (GET /brave/res/v1/web/search), images in a request (answered with VISION-OK, refused with 400 when they break
the provider's documented limits), and GET /page, a small web page for screenshot tests.
"""

from __future__ import annotations

import base64
import json
import struct
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

KEYS = {"anthropic": "sk-ant-test", "openai": "sk-openai-test", "deepseek": "sk-deepseek-test", "zai": "sk-zai-test",
        "openrouter": "sk-or-test"}  # fmt: skip
HUB_CUT_NOTE = "[hub] Your last response "  # the user message the hub adds after a response that cut a tool call


def _status(model: str, response: dict[str, Any]) -> dict[str, Any]:
    """A Responses API object; for a model named with "nostatus", without its `status` (optional in the API
    reference)."""
    if "nostatus" in model:
        response.pop("status", None)
    return response


def _is_cut_note(m: dict[str, Any]) -> bool:
    content = m.get("content")
    text = content if isinstance(content, str) else " ".join(
        str(p.get("text") or "") for p in content or [] if isinstance(p, dict)) if isinstance(content, list) else ""
    return m.get("role") == "user" and text.startswith(HUB_CUT_NOTE)
# OpenRouter's reasoning_details: the hub must send them back exactly as received.
OR_DETAILS = [{"type": "reasoning.text", "text": "plan the call", "signature": "sig-or-1", "format": "anthropic-claude-v1",
               "index": 0}, {"type": "reasoning.encrypted", "data": "enc-or-2", "format": "anthropic-claude-v1",
               "index": 1}]  # fmt: skip
OMITTED_THINKING = {"type": "thinking", "thinking": "", "signature": "sig-123"}
# DeepSeek, https://api-docs.deepseek.com/guides/web_tools, "Generation Failures": a failure after the answer started
# comes with HTTP 200; the non-streaming chat completions body is {"error": {...}, "usage": {...}}.
DS_FAILURE = {"error": {"type": "api_error", "message": "The model failed to generate a complete response. Please"
                        " retry."}, "usage": {"prompt_tokens": 31, "completion_tokens": 2}}  # fmt: skip
REDACTED = {"type": "redacted_thinking", "data": "redacted-abc"}
# Per provider, what one image may be (from each provider's documentation): formats, longest side, base64 size,
# and the 32-px patches (OpenAI).
IMAGE_LIMITS = {
    "anthropic": ({"image/png", "image/jpeg", "image/gif", "image/webp"}, 8000, 10_000_000, None),
    "openai": ({"image/png", "image/jpeg", "image/gif", "image/webp"}, 65_535, 50_000_000, 30_000),
    "deepseek": ({"image/png", "image/jpeg", "image/gif", "image/webp"}, 8192, 32 * 1024 * 1024, None),
    "zai": ({"image/png", "image/jpeg"}, 6000, 5_000_000, None),
}


def _flatten(messages: list[dict[str, Any]]) -> str:
    return "\n".join(f"{m.get('role') or m.get('type')}: {m.get('content') or m.get('output') or ''}"
                     for m in messages)  # fmt: skip


def _tool_markers(text: str) -> list[tuple[str, dict[str, Any]]]:
    """Every TOOL:<name>:<json> in the text, in order (the JSON may hold braces inside its strings)."""
    found = []
    start = text.find("TOOL:")
    while start >= 0:
        name, _, rest = text[start + 5:].partition(":")
        try:
            args, _end = json.JSONDecoder().raw_decode(rest)
        except ValueError:
            args = {}
        found.append((name, args if isinstance(args, dict) else {}))
        start = text.find("TOOL:", start + 5)
    return found


def _tool_request(text: str) -> tuple[str, dict[str, Any]] | None:
    if "TOOLLOOP" in text:
        return "read_file", {"path": "app.py"}
    markers = _tool_markers(text)
    if not markers:
        return None
    return markers[0] if "TOOLSEQ" in text else markers[-1]


def _prompt_text(messages: list[dict[str, Any]]) -> str:
    """The last user message given as text (the prompt that started a tool loop)."""
    return next((m["content"] for m in reversed(messages) if m.get("role") == "user" and isinstance(m.get("content"),
                                                                                                    str)), "")


def _tool_rounds(api: str, messages: list[dict[str, Any]]) -> list[str]:
    """The results of every tool round so far, in order."""
    if api == "anthropic":
        return [" | ".join(("ERROR " if b.get("is_error") else "") + str(b.get("content")) for b in m["content"])
                for m in messages if m.get("role") == "user" and isinstance(m.get("content"), list)
                and m["content"] and m["content"][0].get("type") == "tool_result"]
    if api == "responses":
        return [str(i.get("output")) for i in messages if i.get("type") == "function_call_output"]
    return [str(m.get("content")) for m in messages if m.get("role") == "tool"]


def _images(messages: list[dict[str, Any]]) -> list[tuple[str, str]]:
    """(media type, base64 data) of every image in the request."""
    found = []
    for m in messages:
        for part in m.get("content") or [] if isinstance(m.get("content"), list) else []:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "image" and (part.get("source") or {}).get("type") == "base64":
                found.append((part["source"].get("media_type"), part["source"].get("data", "")))
            url = ""
            if part.get("type") == "input_image":
                url = str(part.get("image_url", ""))
            elif part.get("type") == "image_url":
                url = str((part.get("image_url") or {}).get("url", ""))
            if url.startswith("data:image/"):
                head, _, data = url.partition(",")
                found.append((head[5:].split(";")[0], data))
    return found


def _image_size(raw: bytes) -> tuple[int, int] | None:
    try:
        if raw[:8] == b"\x89PNG\r\n\x1a\n":
            return struct.unpack(">II", raw[16:24])
        if raw[:4] == b"GIF8":
            return struct.unpack("<HH", raw[6:10])
    except struct.error:
        return None
    try:
        import io

        from PIL import Image

        with Image.open(io.BytesIO(raw)) as img:
            return img.size
    except Exception:
        return None


def _image_problem(provider: str, mime: str, data: str) -> str | None:
    """Why the provider would refuse this image (its documented limits), or None."""
    formats, side, size, patches = IMAGE_LIMITS.get(provider, IMAGE_LIMITS["zai"])
    if mime not in formats:
        return f"image format {mime} is not supported"
    if len(data) > size:
        return f"image exceeds {size} bytes (base64)"
    dims = _image_size(base64.b64decode(data))
    if dims is None:
        return "could not process image"
    if max(dims) > side:
        return f"image dimensions exceed max allowed size: {side} pixels"
    if patches and -(-dims[0] // 32) * -(-dims[1] // 32) > patches:
        return f"image needs more than {patches} patches"
    return None


class MockProviders:
    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.searches: list[str] = []
        self.flaky_seen: set[tuple[str, str]] = set()
        self.paused: list[Any] = []  # the content of every pause_turn response: a continuation sends it back
        self.lock = threading.Lock()
        outer = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *_args: Any) -> None:
                pass

            def _json(self, status: int, payload: dict[str, Any], headers: dict[str, str] | None = None) -> None:
                raw = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(raw)))
                for k, v in (headers or {}).items():
                    self.send_header(k, v)
                self.end_headers()
                self.wfile.write(raw)

            def _sse(self, events: list[dict[str, Any]]) -> None:
                raw = "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events).encode()
                self.send_response(200)
                self.send_header("content-type", "text/event-stream")
                self.send_header("content-length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            @staticmethod
            def _stream(blocks: list[dict[str, Any]], stop: str, details: Any = None) -> list[dict[str, Any]]:
                """A Messages API stream for these finished blocks, built the way the API streams them."""
                events: list[dict[str, Any]] = [{"type": "message_start", "message": {"usage": {"input_tokens": 11}}},
                                                {"type": "ping"}]  # fmt: skip
                for i, block in enumerate(blocks):
                    kind = block["type"]
                    if kind == "thinking":
                        events.append({"type": "content_block_start", "index": i,
                                       "content_block": {"type": "thinking", "thinking": "", "signature": ""}})  # fmt: skip
                        events.append({"type": "content_block_delta", "index": i,
                                       "delta": {"type": "thinking_delta", "thinking": block["thinking"]}})  # fmt: skip
                        events.append({"type": "content_block_delta", "index": i,
                                       "delta": {"type": "signature_delta", "signature": block["signature"]}})  # fmt: skip
                    elif kind == "text":
                        events.append({"type": "content_block_start", "index": i,
                                       "content_block": {"type": "text", "text": ""}})  # fmt: skip
                        text = block["text"]
                        for piece in (text[: len(text) // 2], text[len(text) // 2:]):
                            if piece:
                                events.append({"type": "content_block_delta", "index": i,
                                               "delta": {"type": "text_delta", "text": piece}})  # fmt: skip
                        for citation in block.get("citations") or []:
                            events.append({"type": "content_block_delta", "index": i,
                                           "delta": {"type": "citations_delta", "citation": citation}})  # fmt: skip
                    elif kind in ("tool_use", "server_tool_use"):
                        events.append({"type": "content_block_start", "index": i,
                                       "content_block": {**block, "input": {}}})  # fmt: skip
                        raw = json.dumps(block["input"])
                        for piece in ("", raw[:5], raw[5:]):  # the first partial_json is empty, as in the docs
                            events.append({"type": "content_block_delta", "index": i,
                                           "delta": {"type": "input_json_delta", "partial_json": piece}})  # fmt: skip
                    else:  # redacted_thinking, web_search_tool_result: whole in their start event
                        events.append({"type": "content_block_start", "index": i, "content_block": block})
                    events.append({"type": "content_block_stop", "index": i})
                events.append({"type": "message_delta", "delta": {"stop_reason": stop, "stop_details": details},
                               "usage": {"output_tokens": 7}})  # fmt: skip
                events.append({"type": "message_stop"})
                return events

            def _answer(self, api: str, model: str, text: str, stop: str) -> None:
                if api == "anthropic":
                    return self._sse(self._stream([{"type": "thinking", "thinking": "hmm", "signature": "sig-a"},
                                                   {"type": "text", "text": text}], stop))  # fmt: skip
                if api == "responses":
                    return self._json(200, _status(model, {"status": "completed", "output": [
                        {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": text}]}],
                        "usage": {"input_tokens": 21, "output_tokens": 9}}))  # fmt: skip
                message = {"role": "assistant", "content": text, "reasoning_content": "reasoning-final"}
                if self.path.startswith("/openrouter/"):
                    message = {"role": "assistant", "content": text, "reasoning": "reasoning-final",
                               "reasoning_details": OR_DETAILS}  # fmt: skip
                return self._json(200, {"choices": [{"index": 0, "finish_reason": "stop", "message": message}],
                                        "usage": {"prompt_tokens": 31, "completion_tokens": 5}})  # fmt: skip

            def _call_tool(self, api: str, model: str, name: str, args: dict[str, Any], body: dict[str, Any],
                           provider: str, messages: list[dict[str, Any]]) -> None:  # fmt: skip
                n = len(_tool_rounds(api, messages)) + 1  # unique ids, as the real APIs give: one round's per call
                if api == "anthropic":
                    blocks = [OMITTED_THINKING, REDACTED, {"type": "tool_use", "id": f"toolu_{n}", "name": name,
                                                           "input": args}]  # fmt: skip
                    return self._sse(self._stream(blocks, "tool_use"))
                if api == "responses":
                    # Stateless: the encrypted reasoning comes back only when the request asks for it.
                    reasoning = {"type": "reasoning", "id": f"rs_{n}", "summary": []}
                    if "reasoning.encrypted_content" in (body.get("include") or []):
                        reasoning["encrypted_content"] = "enc-123"
                    return self._json(200, _status(model, {"status": "completed", "output": [
                        reasoning, {"type": "function_call", "id": f"fc_{n}", "call_id": f"call_{n}", "name": name,
                                    "arguments": json.dumps(args)}],
                        "usage": {"input_tokens": 21, "output_tokens": 9}}))  # fmt: skip
                message = {"role": "assistant", "content": None, "tool_calls": [
                    {"id": f"call_{n}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]}
                if provider == "openrouter":
                    message.update(reasoning="plan the call", reasoning_details=OR_DETAILS, refusal=None)
                else:
                    message["reasoning_content"] = "reasoning-123"
                return self._json(200, {"choices": [{"index": 0, "finish_reason": "tool_calls", "message": message}],
                                        "usage": {"prompt_tokens": 31, "completion_tokens": 5}})  # fmt: skip

            def _cut_tool(self, api: str, stop: str | None = None) -> None:
                """A write_file call cut off in the middle of its arguments, as each API sends it: Claude: the
                tool_use block's partial JSON, then stop_reason max_tokens; Responses: status incomplete
                (max_output_tokens) with the function_call item; chat: finish_reason length. `stop`: another stop
                reason (e.g. model_context_window_exceeded, content_filter, insufficient_system_resource);
                "<stop>~<native>" on a chat route also gives native_finish_reason <native> (OpenRouter's raw one)."""
                stop, _, native = (stop or "").partition("~")
                cut = json.dumps({"path": "cut.py", "content": "x = 1\n" * 40})[:60]
                if api == "anthropic":
                    events = [{"type": "message_start", "message": {"usage": {"input_tokens": 11}}},
                              {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking",
                                                                                            "thinking": "", "signature": ""}},
                              {"type": "content_block_delta", "index": 0, "delta": {"type": "signature_delta",
                                                                                    "signature": "sig-cut"}},
                              {"type": "content_block_stop", "index": 0},
                              {"type": "content_block_start", "index": 1, "content_block": {
                                  "type": "tool_use", "id": "toolu_cut", "name": "write_file", "input": {}}},
                              {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta",
                                                                                    "partial_json": cut}},
                              {"type": "content_block_stop", "index": 1},
                              {"type": "message_delta", "delta": {"stop_reason": stop or "max_tokens"},
                               "usage": {"output_tokens": 64000}},
                              {"type": "message_stop"}]  # fmt: skip
                    return self._sse(events)
                if api == "responses":
                    return self._json(200, {"status": "incomplete", "incomplete_details": {"reason": stop or
                                                                                          "max_output_tokens"},
                                            "output": [{"type": "reasoning", "id": "rs_cut", "summary": [],
                                                        "encrypted_content": "enc-123"},
                                                       {"type": "function_call", "id": "fc_cut", "call_id": "call_cut",
                                                        "name": "write_file", "arguments": cut, "status": "incomplete"}],
                                            "usage": {"input_tokens": 21, "output_tokens": 32000}})  # fmt: skip
                message = {"role": "assistant", "content": "", "reasoning_content": "write it", "tool_calls": [
                    {"id": "call_cut", "type": "function", "function": {"name": "write_file", "arguments": cut}}]}
                if self.path.startswith("/openrouter/"):
                    message = {"role": "assistant", "content": "", "reasoning": "write it",
                               "reasoning_details": OR_DETAILS, "tool_calls": message["tool_calls"]}  # fmt: skip
                choice = {"index": 0, "finish_reason": stop or "length", "message": message}
                if native:
                    choice["native_finish_reason"] = native
                return self._json(200, {"choices": [choice], "usage": {"prompt_tokens": 31, "completion_tokens": 65536}})

            def _tool_results(self, api: str, messages: list[dict[str, Any]], provider: str) -> Any:
                """The tool results text if this request continues a tool turn; (False, why) if it came back
                wrong; None for an ordinary request."""
                if not messages:
                    return None
                last = messages[-1]
                if api == "anthropic":
                    content = last.get("content")
                    if not (isinstance(content, list) and content and content[0].get("type") == "tool_result"):
                        return None
                    turn = messages[-2].get("content") if len(messages) > 1 else None
                    if not (isinstance(turn, list) and turn[:2] == [OMITTED_THINKING, REDACTED]):
                        return False, "messages: the thinking blocks must be passed back unmodified"
                    if not any(b.get("type") == "tool_use" and b.get("id") == content[0].get("tool_use_id")
                               for b in turn):
                        return False, "tool_result does not match a tool_use block"
                    return " | ".join(("ERROR " if b.get("is_error") else "") + str(b.get("content")) for b in content)
                if api == "responses":
                    if last.get("type") != "function_call_output":
                        return None
                    reasoning = [i for i in messages if i.get("type") == "reasoning"]
                    if any(not i.get("encrypted_content") for i in reasoning):
                        return False, (f"Item with id '{reasoning[0].get('id')}' not found. Items are not persisted"
                                       " when `store` is set to false.")  # fmt: skip
                    if not any(i.get("encrypted_content") == "enc-123" for i in reasoning):
                        return False, "the reasoning item must be passed back"
                    if not any(i.get("type") == "function_call" and i.get("call_id") == last.get("call_id")
                               for i in messages):
                        return False, "No tool call found for function call output"
                    return " | ".join(str(i.get("output")) for i in messages if i.get("type") == "function_call_output")
                if last.get("role") != "tool":
                    return None
                # Every tool call of the last assistant turn needs exactly one result after it, by tool_call_id.
                index = max(i for i, m in enumerate(messages) if m.get("role") == "assistant" and m.get("tool_calls"))
                turn = messages[index]
                wanted = sorted(str(c.get("id")) for c in turn["tool_calls"])
                answered = sorted(str(m.get("tool_call_id")) for m in messages[index + 1:] if m.get("role") == "tool")
                if wanted != answered:
                    return False, f"tool_call_id mismatch: calls {wanted}, results {answered}"
                if provider == "openrouter":
                    if turn.get("reasoning_details") != OR_DETAILS:
                        return False, "reasoning_details must be passed back unmodified"
                elif turn.get("reasoning_content") != "reasoning-123":
                    return False, "reasoning_content must be passed back with the tool calls"
                return " | ".join(str(m.get("content")) for m in messages if m.get("role") == "tool")

            def _hosted_search(self, api: str, model: str, query: str, provider: str, continuing: bool) -> None:
                text = f"{model} searched the web for {query}: answer found."
                if api == "anthropic":
                    if "pause" in model and not continuing:  # a long search pauses; the continuation finishes it
                        paused = [{"type": "text", "text": "Searching. "}, {"type": "server_tool_use", "id": "srvtoolu_p",
                                                                            "name": "web_search", "input": {"query": query}}]
                        with outer.lock:
                            outer.paused.append(paused)
                        return self._sse(self._stream(paused, "pause_turn"))
                    return self._sse(self._stream([
                        {"type": "server_tool_use", "id": "srvtoolu_1", "name": "web_search", "input": {"query": query}},
                        {"type": "web_search_tool_result", "tool_use_id": "srvtoolu_1", "content": [
                            {"type": "web_search_result", "url": "https://docs.example/page", "title": "Docs page",
                             "encrypted_content": "x", "page_age": None}]},
                        {"type": "text", "text": text}], "end_turn"))  # fmt: skip
                if api == "responses":
                    return self._json(200, {"status": "completed", "output": [
                        {"type": "web_search_call", "id": "ws_1", "status": "completed",
                         "action": {"type": "search", "query": query}},
                        {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": text,
                         "annotations": [{"type": "url_citation", "url": "https://openai.example/doc", "title": "OA doc",
                                          "start_index": 0, "end_index": 3}]}]}],
                        "usage": {"input_tokens": 21, "output_tokens": 9}})  # fmt: skip
                if provider == "deepseek":  # https://api-docs.deepseek.com/guides/web_tools, Chat Completions API
                    return self._json(200, {"choices": [{"index": 0, "finish_reason": "stop", "message": {
                        "role": "assistant", "content": text, "annotations": [{"type": "url_citation", "url_citation": {
                            "url": "https://deepseek.example/doc", "title": "DS doc", "start_index": 0,
                            "end_index": 5}}],
                        "server_tool_calls": [{"index": 0, "calls": [{"index": 0, "id": "srvtoolu_s_00_1",
                            "type": "deepseek_web_search", "kind": "search",
                            "action": {"type": "search", "queries": [query]}, "status": "completed",
                            "sources": [{"type": "url", "url": "https://deepseek.example/doc", "title": "DS doc",
                                         "snippet": "...", "site_name": "Example"}]}]}]}}],
                        "usage": {"prompt_tokens": 31, "completion_tokens": 5}})  # fmt: skip
                # Z.ai: https://docs.z.ai/api-reference/llm/chat-completion, web_search tool; results at the top level
                return self._json(200, {"choices": [{"index": 0, "finish_reason": "stop", "message": {
                    "role": "assistant", "content": text}}],
                    "web_search": [{"title": "Zai result", "link": "https://zai.example/r", "content": "summary",
                                    "media": "Example", "refer": "1"}],
                    "usage": {"prompt_tokens": 31, "completion_tokens": 5}})  # fmt: skip

            def do_GET(self) -> None:  # noqa: N802
                if self.path == "/page":
                    raw = b"<html><body style='background:#fff'><h1>Mock page OK</h1><p>login form</p></body></html>"
                    self.send_response(200)
                    self.send_header("content-type", "text/html")
                    self.send_header("content-length", str(len(raw)))
                    self.end_headers()
                    self.wfile.write(raw)
                    return
                if self.path.startswith("/brave/res/v1/web/search"):
                    from urllib.parse import parse_qs, urlparse

                    if self.headers.get("x-subscription-token") != "brave-test":
                        return self._json(401, {"error": {"message": "bad brave token"}})
                    query = parse_qs(urlparse(self.path).query).get("q", [""])[0]
                    with outer.lock:
                        outer.searches.append(f"brave:{query}")
                    return self._json(200, {"web": {"results": [{"title": "Brave result", "url": "https://brave.example/r",
                                                                 "description": f"<b>about</b> {query}"}]}})  # fmt: skip
                self._json(404, {"error": {"message": "not found"}})

            def do_POST(self) -> None:  # noqa: N802
                body = json.loads(self.rfile.read(int(self.headers.get("content-length", "0"))) or b"{}")
                provider = self.path.strip("/").split("/")[0]
                if provider == "tavily":
                    if self.headers.get("authorization") != "Bearer tvly-test":
                        return self._json(401, {"error": {"message": "bad tavily key"}})
                    query = str(body.get("query"))
                    with outer.lock:
                        first = "FLAKY" in query and ("tavily", query) not in outer.flaky_seen
                        outer.flaky_seen.add(("tavily", query))
                        outer.searches.append(f"tavily:{query}")
                    if first:
                        return self._json(503, {"error": {"message": "busy"}}, {"retry-after": "0.2"})
                    return self._json(200, {"answer": f"Tavily answer about {query}",
                                            "results": [{"title": "Tavily result", "url": "https://tavily.example/r",
                                                         "content": "result text"}]})  # fmt: skip
                record = {
                    "path": self.path,
                    "provider": provider,
                    "headers": {k.lower(): v for k, v in self.headers.items()},
                    "body": body,
                }
                with outer.lock:
                    outer.requests.append(record)

                # --- auth -------------------------------------------------
                expected = KEYS.get(provider)
                if provider == "anthropic":
                    ok = self.headers.get("x-api-key") == expected and self.headers.get("anthropic-version")
                elif expected:
                    ok = self.headers.get("authorization") == f"Bearer {expected}"
                else:
                    ok = "authorization" not in {k.lower() for k in self.headers}
                if not ok:
                    return self._json(401, {"error": {"message": f"bad credentials for {provider}"}})

                model = str(body.get("model", ""))
                if model == "fail-401":
                    return self._json(401, {"error": {"message": "invalid api key sk-ant-test rejected"}})
                if model.startswith("flaky"):
                    with outer.lock:
                        first = (provider, model) not in outer.flaky_seen
                        outer.flaky_seen.add((provider, model))
                    if first:
                        status = 529 if provider == "anthropic" else 503
                        return self._json(status, {"error": {"message": "overloaded"}}, {"retry-after": "0.2"})
                if model.startswith("bodyfail") and not self.path.endswith(("/v1/messages", "/responses")):
                    with outer.lock:
                        first = (provider, model) not in outer.flaky_seen
                        outer.flaky_seen.add((provider, model))
                    if first:
                        return self._json(200, DS_FAILURE)
                if model.startswith("slow-"):
                    time.sleep(float(model.split("-", 1)[1]))

                # --- what the model "says" --------------------------------
                if self.path.endswith("/v1/messages"):
                    messages, system = body.get("messages", []), body.get("system", "")
                    effort = (body.get("output_config") or {}).get("effort")
                elif self.path.endswith("/responses"):
                    messages, system = body.get("input", []), body.get("instructions", "")
                    effort = (body.get("reasoning") or {}).get("effort")
                else:
                    messages = [m for m in body.get("messages", []) if m.get("role") != "system"]
                    system = next((m["content"] for m in body.get("messages", []) if m.get("role") == "system"), "")
                    effort = body.get("reasoning_effort")
                record["system"] = system
                record["seen"] = _flatten(messages)
                api = "anthropic" if self.path.endswith("/v1/messages") else (
                    "responses" if self.path.endswith("/responses") else "chat")
                continuing = False
                if api == "anthropic" and messages and messages[-1].get("role") == "assistant":
                    content = messages[-1].get("content")
                    # A paused turn goes back as it came: one this mock paused, or one with a server tool's blocks.
                    with outer.lock:
                        continuing = content in outer.paused or (isinstance(content, list) and any(
                            str(b.get("type", "")).startswith("server_tool_use") or
                            str(b.get("type", "")).endswith("_tool_result") for b in content if isinstance(b, dict)))  # fmt: skip
                    if not continuing:
                        return self._json(400, {"type": "error", "error": {"type": "invalid_request_error", "message":
                            "This model does not support assistant message prefill. The conversation must end with"
                            " a user message."}})  # fmt: skip
                record["continued"] = continuing
                tools = body.get("tools") or []
                if model == "notools" and tools:
                    return self._json(400, {"error": {"message": "this model does not support tools"}})
                if provider == "deepseek" and tools:  # DeepSeek: earlier assistant turns need their reasoning back
                    for m in messages:
                        if m.get("role") == "assistant" and not m.get("reasoning_content"):
                            return self._json(400, {"error": {"message": "The reasoning_content in the thinking"
                                                              " mode must be passed back to the API."}})  # fmt: skip
                if provider == "openrouter":  # reasoning_details only with the tool calls of the answer being made
                    # Every tool-call turn after the last user message is part of the current answer's tool loop and
                    # keeps its details (the API needs the whole sequence); an earlier answer's may not come back.
                    # The hub's note after a cut tool call belongs to the answer being made: nothing before it changed.
                    last_user = max((i for i, m in enumerate(messages) if m.get("role") == "user"
                                     and not _is_cut_note(m)), default=-1)  # fmt: skip
                    for i, m in enumerate(messages):
                        if m.get("role") == "assistant" and "reasoning_details" in m and not (
                                i > last_user and m.get("tool_calls")
                                and all(x.get("role") in ("assistant", "tool") or _is_cut_note(x)
                                        for x in messages[i + 1:])):  # fmt: skip
                            return self._json(400, {"error": {"message": "reasoning_details must follow the exact"
                                                              " history they were produced after"}})  # fmt: skip
                if ("cuttool" in model or "cutonce" in model) and tools:
                    # The output limit cuts its write_file call; cutonce writes in smaller steps once told so.
                    if "cuttool" in model or HUB_CUT_NOTE not in record["seen"]:
                        record["cut"] = True
                        return self._cut_tool(api)
                    return self._answer(api, model, f"{model} wrote it in smaller steps after the note", "end_turn")
                if model.startswith(("cutafter-", "cutagain-")) and tools:
                    # A complete read_file round first, then a write_file call cut off with the stop reason named.
                    stop = model.split("-", 1)[1]
                    told = bool(messages) and _is_cut_note(messages[-1])
                    earlier = messages[:-1] if told else messages
                    done = _tool_rounds(api, earlier)
                    if not done:
                        return self._call_tool(api, model, "read_file", {"path": "app.py"}, body, provider, messages)
                    checked = self._tool_results(api, earlier, provider)  # the read round came back as it must
                    if isinstance(checked, tuple) or checked is None:
                        return self._json(400, {"error": {"message": f"the tool round came back wrong: {checked}"}})
                    if told and model.startswith("cutafter-"):
                        return self._answer(api, model, f"{model} answered after the note; read: {done[0][:200]}",
                                            "end_turn")  # fmt: skip
                    record["cut"] = True
                    return self._cut_tool(api, stop)
                hosted = [t for t in tools if t.get("type") in ("web_search_20250305", "deepseek_web_search")
                          or (t.get("type") == "web_search" and (api == "responses" or "web_search" in t))]
                if hosted:
                    if api == "chat" and provider == "zai" and not (hosted[0].get("web_search") or {}).get("search_engine"):
                        return self._json(400, {"error": {"message": "web_search.search_engine is required"}})
                    first_user = next((m for m in messages if m.get("role") == "user"), {})
                    query = str(first_user.get("content", "")).rpartition("Query: ")[2]
                    with outer.lock:
                        outer.searches.append(f"{provider}:{query}")
                        failed = provider == "deepseek" and "FLAKY" in query and (provider, query) not in outer.flaky_seen
                        outer.flaky_seen.add((provider, query))
                    record["hosted_search"] = query
                    if failed:  # a search that broke off after it started: HTTP 200 with the failure body
                        return self._json(200, DS_FAILURE)
                    return self._hosted_search(api, model, query, provider, continuing)
                images = _images(messages)
                if images:
                    mime, data = images[0]
                    problem = _image_problem(provider if provider in IMAGE_LIMITS else "zai", mime, data)
                    record["image"] = mime
                    record["image_size"] = _image_size(base64.b64decode(data))
                    if problem:
                        return self._json(400, {"error": {"message": problem}})
                    return self._answer(api, model, f"VISION-OK {model} saw a {mime} image", "end_turn")
                if api == "anthropic" and "pause" in model:
                    with outer.lock:
                        turns = [m.get("content") in outer.paused for m in messages if m.get("role") == "assistant"]
                    if all(turns):  # nothing but paused turns so far (a tool round comes after them)
                        if len(turns) < (2 if "pause2" in model else 1):
                            # e.g. a long server-side step: the turn pauses, and goes back as it is
                            paused = [{"type": "text", "text": "PAUSED-PART "}]
                            with outer.lock:
                                outer.paused.append(paused)
                            return self._sse(self._stream(paused, "pause_turn"))
                        if "pausetool" not in model:
                            return self._answer(api, model, f"{model} continued after a pause", "end_turn")
                    # pausetool: the continuation calls the tool the prompt names, then answers with its result
                continuation = self._tool_results(api, messages, provider)
                if isinstance(continuation, tuple):  # the tool turn came back wrong
                    return self._json(400, {"error": {"message": continuation[1]}})
                if continuation is not None:
                    record["tool_results"] = continuation
                    if "TOOLLOOP" in record["seen"]:
                        return self._call_tool(api, model, "read_file", {"path": "app.py"}, body, provider, messages)
                    prompt = _prompt_text(messages)
                    if "TOOLSEQ" in prompt:
                        markers, done = _tool_markers(prompt), _tool_rounds(api, messages)
                        if len(done) < len(markers):
                            return self._call_tool(api, model, *markers[len(done)], body, provider, messages)
                        record["tool_results"] = " || ".join(done)
                        return self._answer(api, model, f"{model} tool results: {' || '.join(done)[:8000]}", "end_turn")
                    return self._answer(api, model, f"{model} tool results: {continuation[:1500]}", "end_turn")
                first_user = next((m for m in reversed(messages) if m.get("role") == "user"), {})
                wanted = _tool_request(str(first_user.get("content", ""))) if tools else None
                if wanted:
                    return self._call_tool(api, model, *wanted, body, provider, messages)
                last = str(messages[-1].get("content", "")) if messages else ""
                text = f"{model} says: turns={len(messages)} effort={effort} last={last[-60:]!r}"
                if provider == "openrouter":
                    sent = [m.get("reasoning_details") for m in messages if m.get("role") == "assistant"]
                    text += f" details_back={sum(1 for d in sent if d == OR_DETAILS)}"
                if model == "long":
                    text = "x" * 150_000
                if model == "empty":
                    text = ""
                if model == "refusal":
                    text = "PARTIAL-BEFORE-REFUSAL "

                # --- reply in the provider's own shape --------------------
                if self.path.endswith("/v1/messages"):
                    if body.get("stream") is not True:
                        return self._json(400, {"error": {"message": "test expects stream=true"}})
                    stop = "max_tokens" if model in ("cutoff", "empty") else "end_turn"
                    details = None
                    if model == "refusal":
                        stop = "refusal"
                        why = "This request was declined because it could enable cyber harm."
                        details = {"type": "refusal", "category": "cyber", "explanation": why}
                    if model == "oddstop":
                        stop = "model_context_window_exceeded"
                    blocks: list[dict[str, Any]] = [{"type": "thinking", "thinking": "hmm", "signature": "sig-a"}]
                    if text:
                        block: dict[str, Any] = {"type": "text", "text": text}
                        if model == "cite":
                            block["citations"] = [{"type": "web_search_result_location", "url": "https://c.example",
                                                   "title": "C", "encrypted_index": "e", "cited_text": "x"}]
                        blocks.append(block)
                    self._sse(self._stream(blocks, stop, details))
                elif self.path.endswith("/responses"):
                    incomplete = model in ("cutoff", "empty", "oddstop")
                    reason = "content_filter" if model == "oddstop" else "max_output_tokens"
                    self._json(200, {
                        "id": "resp_1",
                        "status": "incomplete" if incomplete else "completed",
                        "incomplete_details": {"reason": reason} if incomplete else None,
                        "output": [
                            {"type": "reasoning", "summary": []},
                            {"type": "message", "role": "assistant",
                             "content": [{"type": "output_text", "text": text}] if text else []},
                        ],
                        "usage": {"input_tokens": 21, "output_tokens": 9},
                    })  # fmt: skip
                else:
                    message = {"role": "assistant", "content": text, "reasoning_content": "thinking..."}
                    if provider == "openrouter":
                        message = {"role": "assistant", "content": text, "reasoning": "thinking...",
                                   "reasoning_details": OR_DETAILS}  # fmt: skip
                    self._json(200, {
                        "choices": [{
                            "index": 0,
                            "message": message,
                            "finish_reason": "length" if model in ("cutoff", "empty")
                                             else "sensitive" if model == "oddstop" else "stop",
                        }],
                        "usage": {"prompt_tokens": 31, "completion_tokens": 5},
                    })  # fmt: skip

        class Server(ThreadingHTTPServer):
            def handle_error(self, request: Any, client_address: Any) -> None:
                # A test killed (or timed out) a hub mid-request: writing the answer then fails. Not a test failure.
                if isinstance(sys.exc_info()[1], (BrokenPipeError, ConnectionResetError)):
                    return
                super().handle_error(request, client_address)

        self.server = Server(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def last(self, provider: str | None = None) -> dict[str, Any]:
        with self.lock:
            for record in reversed(self.requests):
                if provider is None or record["provider"] == provider:
                    return record
        raise AssertionError(f"no request recorded for {provider}")

    def count(self) -> int:
        with self.lock:
            return len(self.requests)

    def close(self) -> None:
        self.server.shutdown()
