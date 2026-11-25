"""Workers as the session page and status bars show them.

One entry per required worker: whether someone holds it and is ready,
loading or checking the GPU, whether its heartbeats reach the coordinator,
and its WebRTC links to the neighbouring workers. Contributor identities and
network addresses are never included: links carry candidate types (direct or
TURN), protocol, state and round trip only.
"""

import string

STAGES = string.ascii_uppercase
# Contributors send a heartbeat every 15 s; two missed ones count as late.
LATE_AFTER_S = 35
DIRECT = {"host", "srflx", "prflx"}


def _letter(index):
    return STAGES[index] if 0 <= index < len(STAGES) else str(index + 1)


def _state(record):
    if record is None:
        return "waiting", None
    if record.get("state") == "active":
        return "ready", None
    progress = record.get("progress") or {}
    loaded, total = progress.get("loaded"), progress.get("total")
    if progress.get("phase") == "self-test" or (
        isinstance(loaded, (int, float)) and total and loaded >= total
    ):
        return "checking", None
    if isinstance(loaded, (int, float)) and total:
        return "downloading", round(100 * max(0, loaded) / total, 1)
    return "joining", None


def _links(stage):
    out = []
    for link in stage.get("links") or []:
        if not isinstance(link, dict):
            continue
        peer = link.get("peer")
        kinds = {link.get("local"), link.get("remote")} - {None}
        entry = dict(
            peer=_letter(peer) if isinstance(peer, int) else None,
            state=link.get("state")
            or ("open" if link.get("open") else "connecting"),
        )
        if kinds:
            entry["path"] = "TURN" if "relay" in kinds else "direct"
        if link.get("protocol"):
            entry["protocol"] = link["protocol"]
        if isinstance(link.get("rttMs"), (int, float)):
            entry["rtt_ms"] = link["rttMs"]
        out.append(entry)
    return out


def workers_view(status, now=None):
    """status: the engine's status(); now: the roster clock's current time."""
    if not status:
        return []
    roster = status.get("roster") or {}
    stages = status.get("stages") or []
    required = roster.get("required") or len(stages)
    holders = {}
    for record in roster.get("contributors") or []:
        if record.get("state") not in {"preparing", "active"}:
            continue
        stage = (record.get("assignment") or {}).get("stage")
        if stage:
            holders[stage] = record
    relay = status.get("network") in (None, "relay")
    workers = []
    for index in range(required):
        letter = _letter(index)
        record = holders.get(letter)
        state, percent = _state(record)
        entry = dict(stage=letter, state=state)
        if percent is not None:
            entry["percent"] = percent
        if record is not None:
            seen = record.get("last_seen")
            if isinstance(seen, (int, float)) and now is not None:
                age = max(0.0, now - seen)
                entry["heartbeat_age_s"] = round(age, 1)
                entry["coordinator"] = (
                    "connected" if age <= LATE_AFTER_S else "late"
                )
            else:
                entry["coordinator"] = "connected"
            stage = stages[index] if index < len(stages) else {}
            links = _links(stage)
            if required > 1:
                entry["links"] = links
                if relay:
                    entry["route"] = "relay"
                elif stage.get("links") is None:
                    # This worker reported no WebRTC links (a runtime without
                    # WebRTC, or not yet started): its traffic uses the relay.
                    entry["route"] = "relay (no WebRTC links reported)"
                elif links and all(
                    link["state"] in {"connected", "open"} for link in links
                ):
                    entry["route"] = "WebRTC"
                else:
                    entry["route"] = "relay (WebRTC pending)"
        workers.append(entry)
    return workers


def workers_text(workers):
    """Compact form for status bars: 'A ready, B 45%'."""
    parts = []
    for worker in workers or []:
        state = worker.get("state")
        if state == "ready":
            text = "ready"
        elif state == "downloading":
            text = f"{worker.get('percent', 0):.0f}%"
        else:
            text = state or "?"
        if worker.get("coordinator") == "late":
            text += " (late)"
        parts.append(f"{worker['stage']} {text}")
    return ", ".join(parts)
