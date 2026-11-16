"""What `ygaraty launch` shows: the link to share, then live workers.

In a terminal the worker rows are redrawn in place; elsewhere (logs, pipes)
changes are printed as plain lines, at most every few seconds. Colors follow
NO_COLOR and only appear on a terminal.
"""

import os
import string
import sys
import time

STAGES = string.ascii_uppercase


class Style:
    def __init__(self, out):
        self.out = out
        try:
            self.tty = out.isatty()
        except (AttributeError, ValueError):
            self.tty = False
        self.color = (
            self.tty
            and not os.environ.get("NO_COLOR")
            and os.environ.get("TERM") != "dumb"
        )

    def __call__(self, text, code):
        return f"\033[{code}m{text}\033[0m" if self.color else text

    def bold(self, text):
        return self(text, "1")

    def dim(self, text):
        return self(text, "2")

    def link(self, text):
        return self(text, "1;36")

    def good(self, text):
        return self(text, "32")

    def warn(self, text):
        return self(text, "33")


def gigabytes(value):
    return (
        f"{value / 1e9:.1f} GB" if value >= 10e9 else f"{value / 1e9:.2f} GB"
    )


def clock(seconds):
    seconds = int(seconds)
    return f"{seconds // 60}:{seconds % 60:02d}"


def sharing_line(args, style):
    if getattr(args, "tls", None):
        return (
            "Shared on your local network (HTTPS with a self-signed "
            "certificate: accept the warning once per device)."
        )
    if getattr(args, "share", False):
        return "Shared on the internet through a Cloudflare quick tunnel."
    if args.public_url:
        return f"Shared at {args.public_url}."
    return "Shared on this computer only. " + style.dim(
        "Other computers: --lan (your network) or --public (internet)."
    )


def header(args, links, out=None):
    """The link to share and what a contributor needs, before anything else."""
    out = out or sys.stdout
    style = Style(out)
    reports = getattr(args, "memory_reports", None) or []
    workers = len(reports) or getattr(args, "workers", None) or 1
    context = (reports[0].get("context") if reports else None) or getattr(
        args, "context", None
    )
    slots = getattr(args, "slots", 1) or 1
    facts = [
        args.model_id,
        f"{workers} worker" + ("s" if workers != 1 else ""),
    ]
    if context:
        facts.append(f"{context:,}-token context")
    facts.append(f"{slots} chat" + ("s" if slots != 1 else "") + " at a time")
    need = max((row["recommended"] for row in reports), default=None)
    gpu = f" with {gigabytes(need)} of free GPU memory" if need else ""
    who = (
        "Each person who opens it runs one worker; you can open it on this "
        "computer too."
        if workers > 1
        else "One person (or this computer) opening it is enough."
    )
    lines = [
        "",
        f"  {style.bold('Ygaraty')}  " + " · ".join(facts),
        "",
        f"  Open this link in Chrome or Edge on a computer{gpu}:",
        "",
        f"    {style.link(links['contribute'])}",
        "",
        f"  {who}",
        f"  {sharing_line(args, style)}",
    ]
    route = getattr(args, "route_label", None)
    if route:
        lines.append("  " + style.dim(f"Network between workers: {route}."))
    lines.append("")
    print("\n".join(lines), file=out, flush=True)


def ready_lines(args, links, elapsed, terminal_chat, style):
    model = args.model_id
    key = args.keys / "owner.key"
    first = style.good(f"✓ Ready in {clock(elapsed)}.")
    if terminal_chat:
        first += " Type a message below; /help lists commands."
    else:
        first += " Open the web chat link to talk to the model."
    return [
        "",
        "  " + first,
        "  " + style.dim("Web chat (private): ") + links["chat"],
        "  "
        + style.dim("API: ")
        + f"{links['api']} · model {model} · key in {key}",
        "",
    ]


class Workers:
    """One row per worker; redrawn in place on a terminal."""

    def __init__(self, required, out=None, interval=10.0, now=time.monotonic):
        self.out = out or sys.stdout
        self.style = Style(self.out)
        self.required = max(1, int(required or 1))
        self.now, self.interval = now, interval
        self.started = now()
        self.drawn = 0  # rows on screen that the next draw replaces
        self.last_text, self.last_at = None, None
        self.loaded, self.loaded_at = {}, None

    def rows(self, status):
        roster = (status or {}).get("roster", {}) if status else {}
        required = roster.get("required") or self.required
        records = {}
        for record in roster.get("contributors", []):
            stage = (record.get("assignment") or {}).get("stage")
            if stage:
                records[stage] = record
        now = self.now()
        rates, loaded_now = {}, {}
        for stage, record in records.items():
            loaded = (record.get("progress") or {}).get("loaded")
            if isinstance(loaded, (int, float)):
                loaded_now[stage] = loaded
                before = self.loaded.get(stage)
                if (
                    before is not None
                    and self.loaded_at
                    and now > self.loaded_at
                ):
                    rates[stage] = (loaded - before) / (now - self.loaded_at)
        self.loaded, self.loaded_at = loaded_now, now
        joined = len(records)
        title = f"Workers  {joined} of {required} joined  " + self.style.dim(
            clock(now - self.started)
        )
        rows = ["  " + self.style.bold(title)]
        for index in range(required):
            stage = STAGES[index] if index < len(STAGES) else str(index + 1)
            rows.append(
                f"    {stage}  "
                + self.describe(records.get(stage), rates.get(stage))
            )
        return rows

    def describe(self, record, rate):
        style = self.style
        if record is None:
            return style.dim("waiting for someone to open the link")
        if record.get("state") == "active":
            return style.good("✓ ready")
        progress = record.get("progress") or {}
        loaded, total = progress.get("loaded"), progress.get("total")
        if isinstance(loaded, (int, float)) and total:
            share = max(0.0, min(1.0, loaded / total))
            if share >= 1 or progress.get("phase") == "self-test":
                return "checking the GPU (self-tests)"
            filled = int(round(share * 20))
            text = (
                "downloading "
                + "█" * filled
                + "░" * (20 - filled)
                + f" {100 * share:3.0f}%  "
                + f"{gigabytes(loaded)} of {gigabytes(total)}"
            )
            if rate and rate > 0:
                text += style.dim(f"  {rate / 1e6:.1f} MB/s")
            return text
        phase = progress.get("phase")
        if phase == "self-test":
            return "checking the GPU (self-tests)"
        return "joined, preparing"

    def draw(self, status, force=False):
        rows = self.rows(status)
        text = "\n".join(rows)
        plain = "\n".join(rows[1:])  # without the ticking clock
        if self.style.tty:
            if self.drawn:
                self.out.write(f"\033[{self.drawn}F")
            self.out.write("".join(row + "\033[K\n" for row in rows))
            self.out.flush()
            self.drawn = len(rows)
            return
        now = self.now()
        if not force and (
            plain == self.last_text
            or (
                self.last_at is not None and now - self.last_at < self.interval
            )
        ):
            return
        self.last_text, self.last_at = plain, now
        print(text, file=self.out, flush=True)

    def release(self):
        """Leave the rows on screen; later output goes below them."""
        self.drawn = 0
