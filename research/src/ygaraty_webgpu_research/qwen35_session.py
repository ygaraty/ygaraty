"""Reusable session preparation and browser benchmark helpers.

Session, range_hashes, read_range, Delayer and browser helpers use public
implementations. Hosted serve/submit/local commands have been retired.
Use ygaraty launch and ygaraty contribute for standalone serving.
"""

import argparse
import hashlib
import heapq
import json
import os
import secrets
import signal
import ssl
import subprocess
import threading
import time
from pathlib import Path

from ygaraty.backends.webgpu.checkpoints.preparation import (
    read_range as read_range,
)
from urllib.parse import parse_qs, urlparse


from ygaraty_webgpu_research.browser_binary import (
    GPU_FLAGS,
    HEADLESS,
    ROOT_FLAGS,
    find_chrome,
    memory_sample,
    summarize_memory,
)

w = None  # qwen35_webgpu, imported by the coordinator only (the browser launcher needs no model packages)
from ygaraty.backends.webgpu._legacy.qwen_protocol import (
    SessionProtocol,
    kernel_defaults,
)

# DO-38: conversation c's user message (the first is the default --prompt's)
MESSAGES = (
    "Write a Python function that returns the n-th Fibonacci number.",
    "Explain in three sentences why the sky is blue.",
    "List five uses of a hash table, one line each.",
    "Write a haiku about a GPU waiting for data.",
    "What is the difference between a process and a thread?",
    "Translate 'good morning, how are you?' into Portuguese and Spanish.",
    "Write a SQL query that returns the ten most recent orders per customer.",
    "Give three tips for writing a clear bug report.",
)


def chat_prompt(message):
    return f"<|im_start|>user\n{message}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


class Delayer:
    """Delivers (queue, item) pairs after a fixed delay, in order."""

    def __init__(self, seconds):
        self.seconds, self.heap, self.seq = seconds, [], 0
        self.cv = threading.Condition()
        threading.Thread(target=self._run, daemon=True).start()

    def put(self, box, item):
        with self.cv:
            heapq.heappush(
                self.heap,
                (time.monotonic() + self.seconds, self.seq, box, item),
            )
            self.seq += 1
            self.cv.notify()

    def _run(self):
        while True:
            with self.cv:
                while not self.heap or self.heap[0][0] > time.monotonic():
                    self.cv.wait(
                        None
                        if not self.heap
                        else self.heap[0][0] - time.monotonic()
                    )
                _, _, box, item = heapq.heappop(self.heap)
            box.put(item)


def range_hashes(gguf, ranges):
    """SHA-256 of each (offset, length) range of the file, cached beside it."""
    stat = os.stat(gguf)
    cache_path = Path(str(gguf) + ".ranges.json")
    cache = {}
    if cache_path.exists():
        saved = json.loads(cache_path.read_text())
        if (
            saved.get("size") == stat.st_size
            and saved.get("mtime") == stat.st_mtime
        ):
            cache = saved["ranges"]
    out, fresh = {}, 0
    with open(gguf, "rb") as handle:
        for offset, length in ranges:
            if offset < 0 or length < 0 or offset + length > stat.st_size:
                raise ValueError("Weight range is outside the file")
            key = f"{offset}:{length}"
            if key not in cache:
                handle.seek(offset)
                digest, left = hashlib.sha256(), length
                while left:
                    block = handle.read(min(1 << 24, left))
                    if not block:
                        raise ValueError(
                            "Truncated weight range while hashing"
                        )
                    digest.update(block)
                    left -= len(block)
                cache[key] = digest.hexdigest()
                fresh += 1
            out[(offset, length)] = cache[key]
    if fresh:
        cache_path.write_text(
            json.dumps(
                dict(size=stat.st_size, mtime=stat.st_mtime, ranges=cache)
            )
        )
    return out


class Session(SessionProtocol):
    def __init__(self, args, *, preparation_only=False):
        from tokenizers import Tokenizer

        global w
        from ygaraty_webgpu_research import qwen35_webgpu as w

        self.args = args
        self.out = Path(args.out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.key = args.session_key or secrets.token_urlsafe(18)
        from ygaraty.backends.webgpu.checkpoints.families import load_model

        model = load_model(
            args.gguf, args.config, tensor_only=preparation_only
        )
        self.family = model.family
        # Experiments 121/125/127 kernels for Qwen3.5 and Kolibri-1 (both reference-checked); dense keeps the old ones.
        self.kernels = (
            kernel_defaults(self.family)
            if self.family in ("qwen35", "kolibri1", "qwen3moe")
            else {}
        )
        from ygaraty.backends.webgpu._legacy.qwen_protocol import chat_format

        self.chat_style = (
            chat_format(json.loads(Path(args.chat_format).read_text()))
            if getattr(args, "chat_format", None)
            else None
        )
        config = json.loads(Path(args.config).read_text())
        eos = config.get("eos_token_id") or config.get("text_config", {}).get(
            "eos_token_id"
        )
        if preparation_only and not args.chat:
            raise ValueError("Bundle preparation requires chat metadata mode")
        # Export copies the tokenizer file; it never encodes prompts or runs a protocol.
        self.tokenizer = (
            None
            if preparation_only
            else Tokenizer.from_file(str(args.tokenizer))
        )
        self.conversations = 1 if args.chat else args.conversations
        convs = range(
            args.first_conversation,
            args.first_conversation + self.conversations,
        )
        if args.chat:
            self.prompts = [[]]
            self.prompt_source = dict(chat=True)
        elif args.text:
            text = Path(args.text).read_bytes()
            ids, n = (
                self.tokenizer.encode(text.decode()).ids,
                args.prompt_tokens,
            )
            if len(ids) < (convs[-1] + 1) * n:
                raise SystemExit(
                    f"{args.text}: {len(ids)} tokens, conversation {convs[-1]} needs {(convs[-1] + 1) * n}"
                )
            self.prompts = [
                ids[c * n : (c + 1) * n] for c in convs
            ]  # conversation c: the c-th slice of the text
            self.prompt_source = dict(
                text=str(args.text),
                sha256=hashlib.sha256(text).hexdigest(),
                conversations=list(convs),
            )
        elif self.conversations > 1 or args.first_conversation:
            self.prompts = [
                self.tokenizer.encode(
                    chat_prompt(MESSAGES[c % len(MESSAGES)])
                ).ids
                for c in convs
            ]
            self.prompt_source = dict(
                messages=[MESSAGES[c % len(MESSAGES)] for c in convs],
                conversations=list(convs),
            )
        else:
            prompt = args.prompt
            if getattr(
                args, "prompt_file", None
            ):  # as "$(cat FILE)": trailing newlines dropped
                prompt = Path(args.prompt_file).read_text().rstrip("\n")
            self.prompts = [self.tokenizer.encode(prompt).ids]
            self.prompt_source = dict(prompt=prompt)
        self.prompt = self.prompts[0]
        self.eos = [eos] if isinstance(eos, int) else list(eos or [])
        self.splits = [int(v) for v in args.splits.split(",")]
        chunks = [
            min(args.chunk, len(self.prompt) - i)
            for i in range(0, len(self.prompt), args.chunk)
        ]
        self.capacity = (
            args.capacity
            if args.chat
            else max(map(len, self.prompts)) + args.generate
        )
        self.files, self.stages = {}, []
        draft = getattr(args, "draft", None)
        if draft and (
            args.chat
            or args.batch
            or draft["k"] + 1 > min(args.chunk, self.capacity)
        ):
            raise SystemExit(
                "--draft is for fixed prompts without --batch, with --chunk above k"
            )
        if args.route == "webrtc" and args.chat:
            raise SystemExit(
                "--route webrtc is for fixed prompts (benchmarks), not --chat"
            )
        ice = (
            Path(args.ice[1:]).read_text()
            if args.ice.startswith("@")
            else args.ice
        )  # @FILE: kept off command lines
        self.ice = json.loads(ice) if ice else []
        if not isinstance(self.ice, list):
            raise SystemExit(
                "--ice must be a JSON list of RTCIceServer objects"
            )
        for k, (first, last) in enumerate(zip(self.splits, self.splits[1:])):
            manifest, files = w.stage_manifest(model, first, last)
            self.files.update(files)
            fixture = dict(
                chunks=chunks,
                prefill_calls=len(chunks),
                max_tokens=min(
                    args.chunk, self.capacity
                ),  # a stage's chunk fits its context
                max_positions=self.capacity,
                kv=args.kv,
                wire="f16",
                prompts=self.prompts if first == 0 and not args.chat else None,
                conversations=self.conversations,
                batch=bool(args.batch),
                # DO-40: hidden states and tokens straight between neighbouring stages over WebRTC
                # data channels (the coordinator only signals); the last stage's token goes to the
                # first stage directly, which stops a conversation on its own by the same rule as here
                route=args.route,
                ice=self.ice,
                ice_policy=args.ice_policy,
                stages=len(self.splits) - 1,
                generate=args.generate,
                stop_ids=sorted(self.eos),
                draft=draft,  # DO-25: n-gram drafts {n, k}, checked exactly (None: plain decoding)
                digests=bool(
                    getattr(args, "logit_digests", False)
                ),  # DO-25: emitted logits' SHA-256
                matvec=getattr(
                    args, "matvec", None
                ),  # WS-16: matrix-vector rows per thread
                **self.kernels,
            )  # experiments 121, 125, 127 (YGARATY_KERNELS=off: without)
            self.stages.append(
                dict(
                    stage=k,
                    layers=[first, last],
                    manifest=manifest,
                    fixture=fixture,
                    names=sorted(files),
                )
            )
        from ygaraty.backends.webgpu.distribution import compile_distribution

        for stage in self.stages:
            stage["manifest"].pop("plan", None)
        self.distribution_plan = compile_distribution(self.stages)
        print(
            f"hashing {len(self.files)} weight ranges (cached beside the GGUF)...",
            flush=True,
        )
        hashes = range_hashes(
            args.gguf, [(o, n) for _, o, n in self.files.values()]
        )
        self.hashes = {
            name: hashes[(o, n)] for name, (_, o, n) in self.files.items()
        }
        self._selftest_fixture(model)
        if getattr(
            args, "matvec", None
        ):  # admission checks the kernel the stages will use (WS-16)
            self.selftest = dict(self.selftest, matvec=args.matvec)
        self.selftest = dict(self.selftest, **self.kernels)
        if preparation_only:
            return
        self.initialize_protocol(
            args, Delayer(args.relay_delay / 1e3) if args.relay_delay else None
        )
        for stage in self.stages:  # a chat also stops at <|im_end|>: the first stage applies the same rule
            stage["fixture"]["stop_ids"] = sorted(self.stop_ids)
            stage["fixture"]["local_loop"] = (
                getattr(args, "loop", "local") == "local"
            )  # DO-61, one stage only
            stage["fixture"]["resident"] = (
                os.environ.get("YGARATY_RESIDENT") != "off"
            )  # experiments 120/129
            stage["fixture"]["report_batch"] = (
                os.environ.get("YGARATY_REPORTS") != "serial"
            )  # DO-85
            slots = int(
                os.environ.get("YGARATY_CHAT_SLOTS", "0")
            )  # DO-87: production's slots (chat on conv 0)
            if args.chat and slots > 1:
                stage["fixture"].update(conversations=slots, batch=True)

    def _selftest_fixture(self, model):
        from ygaraty.backends.webgpu.checkpoints.preparation import (
            selftest_fixture,
        )

        selftest_fixture(self, model)

    def write_report(self, report):
        w.write(self.args, report)


def chrome_command(url, profile, extra=()):
    parsed = urlparse(url)
    origin = f"{parsed.scheme}://{parsed.netloc}"
    flags = []
    if parsed.scheme == "http" and parsed.hostname not in (
        "127.0.0.1",
        "localhost",
    ):
        flags.append(f"--unsafely-treat-insecure-origin-as-secure={origin}")
    return [
        find_chrome(),
        *HEADLESS,
        "--enable-unsafe-webgpu",
        "--no-first-run",
        "--no-default-browser-check",
        "--disable-background-timer-throttling",
        "--disable-renderer-backgrounding",
        "--disable-backgrounding-occluded-windows",
        *GPU_FLAGS,
        *ROOT_FLAGS,
        *flags,
        *extra,
        f"--user-data-dir={profile}",
        url,
    ]


def autostart_url(url):
    """The join page waits for a visitor to accept before it reserves a stage, downloads weights or uses the
    GPU; an experiment's own headless browser accepts on its own behalf with autostart=1 in the fragment."""
    fragment = urlparse(url).fragment
    if "autostart" in parse_qs(fragment):
        return url
    return url + ("&autostart=1" if fragment else "#autostart=1")


def run_browser(
    url,
    out,
    name,
    stop,
    interval=1.0,
    keep_profile=False,
    extra=(),
    profile=None,
):
    """Headless Chromium on the join URL until `stop` is set or the session
    ends; samples its memory and posts the summary to the coordinator."""
    import urllib.request

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if profile is None:
        profile = out / "browser-profile"
    else:
        keep_profile = True
    parsed = urlparse(url)
    key = parse_qs(parsed.fragment).get("key", [""])[0]
    # the session's base: the origin alone, or /sessions/<job> on the general coordinator
    base = f"{parsed.scheme}://{parsed.netloc}" + parsed.path.rstrip("/")
    insecure = (
        ssl._create_unverified_context() if parsed.scheme == "https" else None
    )

    def post(path, body):
        request = urllib.request.Request(
            base + path,
            data=json.dumps(body).encode(),
            headers={"X-Key": key},
            method="POST",
        )
        with urllib.request.urlopen(
            request, timeout=10, context=insecure
        ) as response:
            return response.read()

    def status():
        request = urllib.request.Request(
            base + "/api/status", headers={"X-Key": key}
        )
        with urllib.request.urlopen(
            request, timeout=10, context=insecure
        ) as response:
            return json.loads(response.read())

    url = autostart_url(url)
    (out / "browser-exit.json").unlink(missing_ok=True)
    with (out / "chromium.log").open("w") as log:
        process = subprocess.Popen(
            chrome_command(url, profile, extra),
            stdout=log,
            stderr=log,
            start_new_session=True,
        )
    samples, known, baseline = [], {}, None
    from ygaraty_webgpu_research.browser_binary import device_memory

    baseline = device_memory()
    last_post = 0.0
    exit_reason = "browser_exit"
    try:
        with (out / "memory.jsonl").open("w") as mem:
            while not stop.is_set() and process.poll() is None:
                sample = memory_sample(process.pid, known)
                samples.append(sample)
                mem.write(json.dumps(sample) + "\n")
                if time.monotonic() - last_post > 5:
                    last_post = time.monotonic()
                    try:
                        post(
                            f"/api/host/{name}",
                            summarize_memory(samples, [], baseline),
                        )
                        if status()["state"] in (
                            "finished",
                            "failed",
                            "timeout",
                            "interrupted",
                        ):
                            exit_reason = "coordinator_terminal"
                            post(
                                f"/api/host/{name}",
                                summarize_memory(samples, [], baseline),
                            )
                            break
                    except OSError:
                        pass
                stop.wait(interval)
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)
        (out / "browser-exit.json").write_text(
            json.dumps(
                {
                    "at": int(time.time()),
                    "reason": "requested_stop"
                    if stop.is_set()
                    else exit_reason,
                    "exit_code": process.returncode,
                }
            )
        )
        if not keep_profile:
            import shutil

            shutil.rmtree(profile, ignore_errors=True)
    return summarize_memory(samples, [], baseline) if samples else None


def matvec_spec(text):
    """`SINGLE,ROWS,TOKENS` as {"single": .., "rows": .., "tokens": ..}; `off` as None (one row per thread)."""
    if text == "off":
        return None
    try:
        single, rows, tokens = (int(v) for v in text.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"invalid matvec {text!r}: SINGLE,ROWS,TOKENS"
        )
    if (
        single not in (1, 2, 4)
        or rows not in (1, 2, 4)
        or tokens not in (4, 8)
    ):
        raise argparse.ArgumentTypeError(
            f"invalid matvec {text!r}: rows 1, 2 or 4; tokens 4 or 8"
        )
    return dict(single=single, rows=rows, tokens=tokens)


def draft_spec(text):
    """`ngram:n=N,k=K[,min=M][,adapt=1][,exact=0]` as {"n": N, "k": K, ...}; `none` as None."""
    if text == "none":
        return None
    kind, _, rest = text.partition(":")
    try:
        values = (
            dict(item.split("=", 1) for item in rest.split(","))
            if rest
            else {}
        )
        spec = {"n": int(values.pop("n", 3)), "k": int(values.pop("k", 4))}
        if (
            "exact" in values
        ):  # exact=0: checking passes on the prefill kernels (faster, prefill's rounding)
            spec["exact"] = values.pop("exact") not in ("0", "false")
        if "min" in values:  # the shortest match that may propose drafts
            spec["min"] = int(values.pop("min"))
        if (
            "adapt" in values
        ):  # adapt=1: the number of drafts follows their acceptance; adapt=cost: expected
            value = values.pop(
                "adapt"
            )  # tokens per measured ms (speculative.mjs Drafter)
            spec["adapt"] = (
                "cost" if value == "cost" else value not in ("0", "false")
            )
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid draft {text!r}")
    if (
        kind != "ngram"
        or values
        or not 1 <= spec["n"] <= 8
        or not 1 <= spec["k"] <= 16
        or not 1 <= spec.get("min", 1) <= spec["n"]
    ):
        raise argparse.ArgumentTypeError(
            f"invalid draft {text!r}: ngram:n=1..8,k=1..16[,min=1..n][,adapt=1|cost][,exact=0]"
        )
    return spec
