"""Interactive chat in the launch terminal through this server's local API.

It uses the same OpenAI-compatible endpoint and owner key as the browser chat
page, so admission, queuing and errors behave identically. Blocking reads and
requests run in daemon threads: Ctrl-C stops the server even while the prompt
is waiting for input.
"""

import asyncio
import json
import sys
import threading
import time
import uuid
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

HELP = """Type a message and press Enter. Commands:
  /reset   start a new conversation
  /net     show the links between workers (WebRTC or relay)
  /quit    stop the server (same as Ctrl-C)
  /help    show this list
  Ctrl-D   close only this prompt; the server keeps running"""
ROUTES = {
    "relay": "relay",
    "direct": "WebRTC",
    "private": "WebRTC, TURN only",
}


def local_base(host, port, *, secure=False):
    """Loopback URL for this server, even when it listens on all addresses."""
    if host in {"", "0.0.0.0", "::"}:
        host = "127.0.0.1"
    if ":" in host:
        host = f"[{host}]"
    return f"{'https' if secure else 'http'}://{host}:{port}"


def own_server_opener():
    """urlopen for this process's own loopback HTTPS server (--lan).

    Its certificate is self-signed; the request never leaves this machine.
    """
    import ssl

    context = ssl.create_default_context()
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE

    def opener(request, timeout):
        return urlopen(request, timeout=timeout, context=context)

    return opener


def stream_reply(
    base, key, messages, conversation, on_text, opener=urlopen, clock=None
):
    """Send one chat request, pass streamed text to on_text, time it.

    Returns the answer and stats: time to first token, completion tokens
    (the server's usage count when present), total seconds and the decode
    rate after the first token.
    """
    clock = clock or time.perf_counter
    request = Request(
        base + "/v1/chat/completions",
        data=json.dumps(
            {
                "messages": messages,
                "conversation_id": conversation,
                "stream": True,
            }
        ).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
        },
    )
    answer, finished, chunks, usage = [], False, 0, None
    started = clock()
    first = None
    with opener(request, timeout=600) as response:
        for raw in response:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[6:])
            if chunk.get("error"):
                raise ValueError(
                    chunk["error"].get("message") or "Request failed"
                )
            usage = chunk.get("usage") or usage
            choice = (chunk.get("choices") or [{}])[0]
            text = (choice.get("delta") or {}).get("content") or ""
            if text:
                if first is None:
                    first = clock()
                chunks += 1
                answer.append(text)
                on_text(text)
            if choice.get("finish_reason"):
                finished = True
    if not finished:
        raise ValueError("Response interrupted")
    ended = clock()
    tokens = (usage or {}).get("completion_tokens") or chunks
    return "".join(answer), {
        "ttft": None if first is None else first - started,
        "tokens": tokens,
        "prompt_tokens": (usage or {}).get("prompt_tokens"),
        "seconds": ended - started,
        # Decode rate after the first token, so TTFT does not dilute it.
        "rate": (tokens - 1) / (ended - first)
        if first is not None and tokens > 1 and ended > first
        else None,
    }


def in_daemon_thread(function, *args):
    """Run a blocking call without holding up interpreter shutdown."""
    loop = asyncio.get_running_loop()
    future = loop.create_future()

    def settle(result, error):
        if future.done():
            return
        if error is not None:
            future.set_exception(error)
        else:
            future.set_result(result)

    def target():
        try:
            result = function(*args)
        except BaseException as error:  # delivered to the awaiting task
            loop.call_soon_threadsafe(settle, None, error)
        else:
            loop.call_soon_threadsafe(settle, result, None)

    threading.Thread(target=target, daemon=True).start()
    return future


def format_stats(stats):
    parts = []
    if stats.get("ttft") is not None:
        parts.append(f"TTFT {stats['ttft']:.2f} s")
    count = stats["tokens"]
    parts.append(f"{count} token" + ("" if count == 1 else "s"))
    if stats.get("rate"):
        parts.append(f"{stats['rate']:.1f} tok/s")
    return " · ".join(parts)


def stage_links(status):
    """One entry per worker link (A→B, …), deduplicated across both ends."""
    stages = (status or {}).get("stages") or []
    names = [s.get("stage", str(i)) for i, s in enumerate(stages)]
    seen, links = set(), []
    for index, stage in enumerate(stages):
        for link in stage.get("links") or []:
            peer = link.get("peer")
            if not isinstance(peer, int) or not 0 <= peer < len(names):
                continue
            pair = tuple(sorted((index, peer)))
            if pair in seen:
                continue
            seen.add(pair)
            links.append((f"{names[pair[0]]}→{names[pair[1]]}", link))
    return links


def link_text(name, link, detail=False):
    text = (
        f"{name} {link.get('state') or ('open' if link.get('open') else '?')}"
    )
    if link.get("local") or link.get("remote"):
        text += f" {link.get('local', '?')}/{link.get('remote', '?')}"
    if link.get("protocol"):
        text += f" {link['protocol']}"
    if link.get("rttMs") is not None:
        text += f" rtt {link['rttMs']:g} ms"
    if detail:
        for key, label in (
            ("relayProtocol", "TURN via"),
            ("openedMs", "opened in ms"),
            ("relinks", "relinks"),
            ("bytesSent", "bytes sent"),
            ("bytesReceived", "bytes received"),
            ("dropped", "dropped"),
        ):
            if link.get(key) is not None:
                text += f", {label} {link[key]}"
    return text


def network_summary(status, detail=False):
    """Route and WebRTC link state from the session status, or None."""
    if not status:
        return None
    if (status.get("roster") or {}).get("required") == 1:
        # One worker runs every layer: nothing travels between workers.
        return "Route: one worker, no links to report" if detail else None
    route = ROUTES.get(status.get("network"), status.get("network") or "?")
    links = stage_links(status)
    if not links:
        if status.get("network") not in (None, "relay"):
            route += ": no links reported yet" if detail else ", links pending"
        return f"Route: {route}" if detail else route
    texts = [link_text(name, link, detail) for name, link in links]
    if detail:
        return "\n".join([f"Route: {route}", *(f"  {t}" for t in texts)])
    return f"{route}: " + "; ".join(texts)


def error_text(error):
    if isinstance(error, HTTPError):
        try:
            detail = json.loads(error.read() or b"{}").get("detail")
        except ValueError:
            detail = None
        return f"HTTP {error.code}: {detail or error.reason}"
    if isinstance(error, URLError):
        return f"Server not reachable: {error.reason}"
    return str(error)


async def terminal_chat(
    base, key, *, read=input, write=print, opener=urlopen, status=None
):
    """Prompt, send each line, stream the answer; keep one conversation.

    Returns "quit" after /quit (the caller stops the server) and None after
    Ctrl-D.

    After each answer a stats line shows time to first token, tokens and
    decode rate, and the route with any WebRTC links (status: a callable
    returning the session status, or None).
    """
    dim = write is print and sys.stdout.isatty()
    messages, conversation = [], str(uuid.uuid4())
    while True:
        try:
            line = await in_daemon_thread(read, "> ")
        except EOFError:
            write("\nTerminal chat closed; the server keeps running.")
            return
        text = line.strip()
        if not text:
            continue
        if text in ("/help", "/?"):
            write(HELP)
            continue
        if text == "/reset":
            messages, conversation = [], str(uuid.uuid4())
            write("New conversation.")
            continue
        if text in ("/quit", "/exit"):
            write("Stopping the server…")
            return "quit"
        if text == "/net":
            write(
                network_summary(status() if status else None, detail=True)
                or "Network status unavailable."
            )
            continue
        pending = [*messages, {"role": "user", "content": text}]
        try:
            answer, stats = await in_daemon_thread(
                stream_reply,
                base,
                key,
                pending,
                conversation,
                lambda part: write(part, end="", flush=True),
                opener,
            )
        except (HTTPError, URLError, ValueError, OSError) as error:
            write(f"\n[error] {error_text(error)}")
            continue
        line = format_stats(stats)
        network = network_summary(status() if status else None)
        if network:
            line += f" · {network}"
        write(f"\n\033[2m[{line}]\033[0m" if dim else f"\n[{line}]")
        messages = [*pending, {"role": "assistant", "content": answer}]
