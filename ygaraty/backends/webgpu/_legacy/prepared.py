"""Load precomputed WebGPU manifests. Never load/execute GGUF tensors on the server."""

import copy
import queue
import json
import os
import re
import secrets
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import quote
import numpy as np
from .qwen_protocol import (
    SessionProtocol,
    SELFTEST_CASES,
    MATVEC_DEFAULT,
    STAGE_NAMES,
    kernel_defaults,
    SignalRejected,
)


from .model_artifacts import local_file


def validate_distribution_plan(stored, compiled):
    """Accept only the current plan or its exact pre-contract v2 projection."""
    from ..operations import legacy_stage

    if stored == compiled:
        return
    legacy = dict(
        compiled,
        version=2,
        stages=[legacy_stage(stage) for stage in compiled["stages"]],
    )
    if stored != legacy:
        raise ValueError("Stored distribution plan differs from stages")


class PreparedSession(SessionProtocol):
    def __init__(
        self,
        bundle_path,
        out,
        assignments,
        memory_gb,
        *,
        storage=None,
        route="relay",
        turn=None,
        context_window=None,
        conversation_slots=8,
        metadata=None,
    ):
        from ygaraty.inference_support.kyse_tokenizer import load_tokenizer

        bundle_path = Path(bundle_path)
        if bundle_path.stat().st_size > 16 * 1024 * 1024:
            raise ValueError("Bundle manifest exceeds 16 MiB")
        data = (
            copy.deepcopy(metadata)
            if metadata is not None
            else json.loads(bundle_path.read_text())
        )
        if data.get("format") != "ygaraty-webgpu-v1":
            raise ValueError("Unsupported WebGPU bundle format")
        self.stages = copy.deepcopy(data["stages"])
        if not 1 <= len(self.stages) <= len(STAGE_NAMES) or [
            s["layers"] for s in self.stages
        ] != [a["layers"] for a in assignments]:
            raise ValueError(
                "Prepared stages do not match the selected distribution plan"
            )
        for i, stage in enumerate(self.stages):
            if (
                stage["stage"] != i
                or stage["fixture"].get("route") != "relay"
                or stage["fixture"].get("conversations") != 1
            ):
                raise ValueError(
                    "User chat bundles require ordered stages, one conversation and relay routing"
                )
            if stage["fixture"].get(
                "ice"
            ):  # a stored artifact never carries TURN servers or credentials
                raise ValueError(
                    "Bundles must not contain ICE servers; the coordinator issues them per contributor"
                )
            stage["fixture"]["memory_budget_bytes"] = memory_gb * 1024**3
        from ..distribution import compile_distribution

        self.distribution_plan = compile_distribution(self.stages)
        if "distribution_plan" in data:
            try:
                validate_distribution_plan(
                    data["distribution_plan"], self.distribution_plan
                )
            except ValueError:
                # Prepared before the DeltaNet gate rows were padded; the
                # fresh (padded) plan is what runs.
                validate_distribution_plan(
                    data["distribution_plan"],
                    compile_distribution(
                        copy.deepcopy(self.stages), aligned_gates=False
                    ),
                )
        from ..bundle_validation import remote_base, remote_piece_path

        # A public HTTPS base: contributors download pieces directly.
        self.storage = remote_base(data)
        self.remote_paths = {}
        self.files, self.hashes = {}, {}
        for name, file in data["files"].items():
            path = (
                None
                if self.storage
                else local_file(bundle_path.parent, file["path"])
            )
            offset, length = file["offset"], file["length"]
            if (
                not re.fullmatch(r"[A-Za-z0-9_.-]+", name)
                or type(offset) is not int
                or type(length) is not int
                or offset < 0
                or length <= 0
                or (path is not None and offset + length > path.stat().st_size)
                or not re.fullmatch(r"[a-f0-9]{64}", file["sha256"])
            ):
                raise ValueError("Invalid prepared weight range")
            if self.storage:
                if offset != 0:
                    raise ValueError(
                        "Remote weights must be complete piece objects"
                    )
                self.remote_paths[name] = remote_piece_path(file["path"])
            self.files[name] = (path, offset, length)
            self.hashes[name] = file["sha256"]
        for stage in self.stages:
            if (
                sum(self.files[n][2] for n in stage["names"])
                > memory_gb * 1024**3
            ):
                raise ValueError(
                    "Prepared stage weights exceed the approved contribution budget"
                )
        self.test_files = {}
        for name, path in data["test_files"].items():
            source = local_file(bundle_path.parent, path)
            if source.stat().st_size > 16 * 1024 * 1024:
                raise ValueError("Self-test artifact exceeds 16 MiB")
            self.test_files[name] = source.read_bytes()
        self.selftest = data["selftest"]
        self.expected = []
        for name in data["expected"]:
            path = local_file(bundle_path.parent, name)
            if path.stat().st_size > 16 * 1024 * 1024:
                raise ValueError("Self-test reference exceeds 16 MiB")
            self.expected.append(np.load(path, allow_pickle=False))
        if not self.expected or len(self.expected) != len(
            self.selftest["cases"]
        ):
            raise ValueError("Self-test references are incomplete")
        self.f16_x = np.frombuffer(
            self.test_files[self.selftest["f16"]], dtype=np.float32
        )
        with np.errstate(over="ignore"):
            self.f16_want = self.f16_x.astype(np.float16)
        options = dict(data["options"])
        from .model_artifacts import bundle_capacity

        validated_capacity = bundle_capacity(data)
        if context_window is not None:
            if (
                type(context_window) is not int
                or not 1 <= context_window <= validated_capacity
            ):
                raise ValueError(
                    "Requested context exceeds the validated bundle capacity"
                )
            options["capacity"] = context_window
            options["chunk"] = min(options["chunk"], context_window)
        for stage in self.stages:
            fixture = stage["fixture"]
            fixture["max_positions"] = options["capacity"]
            fixture["max_tokens"] = min(
                fixture.get("max_tokens", options["chunk"]),
                options["capacity"],
            )

        for key in ("capacity", "generate", "chunk"):
            if type(options[key]) is not int or options[key] <= 0:
                raise ValueError("Invalid prepared chat settings")
        self.out = Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        # The network path is the session's (chosen by its owner at creation), never the bundle's. A direct
        # session needs the deployment's TURN server: it fails here rather than quietly using the relay.
        if route not in ("relay", "webrtc", "private"):
            raise ValueError("Unknown session route")
        if route in ("webrtc", "private"):
            # Direct links on one local network use host candidates; TURN-only
            # links cannot work without the server.
            if turn is None and route == "private":
                raise ValueError(
                    "TURN-only sessions need a TURN server; ask an administrator"
                )
            # One stage (DO-51): no links between stages, only the chat browser's link to that contributor.
        self.turn_config = turn
        self.args = SimpleNamespace(
            out=self.out,
            chat=True,
            kv=options["kv"],
            capacity=options["capacity"],
            generate=options["capacity"],
            chunk=options["chunk"],
            turn_timeout=120,
            route="relay" if route == "relay" else "webrtc",
            ice="",
            ice_policy="relay" if route == "private" else "all",
            relay_delay=0,
            batch=False,
            contributor_db=None,
            production_chat=True,
        )
        self.key = secrets.token_urlsafe(32)
        self.tokenizer = load_tokenizer(local_file(bundle_path.parent, data["tokenizer"]))
        self.prompts, self.prompt, self.prompt_source = (
            [[]],
            [],
            {"chat": True},
        )
        self.eos = data["eos"]
        chat_slots = conversation_slots
        if not 1 <= chat_slots <= 16:
            raise ValueError("YGARATY_CHAT_SLOTS must be between 1 and 16")
        self.conversations, self.capacity = chat_slots, options["capacity"]
        self.args.batch = chat_slots > 1
        from .qwen_protocol import chat_format

        self.chat_style = (
            chat_format(options["chat"])
            if options.get("chat") is not None
            else None
        )  # checked; None: Qwen's
        self.splits = [self.stages[0]["layers"][0]] + [
            s["layers"][1] for s in self.stages
        ]
        self.ice = []  # contributors fetch their own short-lived credentials from /api/ice/<id>
        self.initialize_protocol(self.args)
        from .conversations import Conversations
        from .throughput import Throughput

        self.throughput = Throughput()
        self.concurrent_chat = Conversations(self, chat_slots)
        # The kernel is the deployment's choice, not the bundle's; admission checks the one the stages will use.
        # Measured on Qwen3.5 only (its manifests name no family): dense (Phi-4) and kolibri1 bundles keep one
        # row per thread.
        qwen = all(
            s["manifest"].get("config", {}).get("family", "qwen35") == "qwen35"
            for s in self.stages
        )
        matvec = (
            None
            if not qwen or os.environ.get("YGARATY_MATVEC") == "off"
            else MATVEC_DEFAULT
        )
        if matvec:
            self.selftest = dict(self.selftest, matvec=matvec)
        # Experiments 121/125/127 kernels for Qwen3.5 and Kolibri-1 (both reference-checked); dense keeps the old ones.
        families = {
            s["manifest"].get("config", {}).get("family", "qwen35")
            for s in self.stages
        }
        # DO-80: Qwen3-MoE stages run the same kernels as Kolibri-1 (checked whole-model on an A100 with them).
        kernels = (
            kernel_defaults("qwen35" if families == {"qwen35"} else "kolibri1")
            if families <= {"qwen35", "kolibri1", "qwen3moe"}
            else {}
        )
        if kernels:
            self.selftest = dict(self.selftest, **kernels)
        for stage in self.stages:
            fixture = stage["fixture"]
            fixture.pop("ice", None)
            fixture["matvec"] = matvec
            fixture.update(kernels)
            # DO-61: a one-stage plan keeps its token loop on the contributor (the first stage's stop rule needs
            # `stop_ids` and `generate`); YGARATY_LOCAL_LOOP=off sends every token through the coordinator again.
            # DO-87: batched slots, plus a resident burst (experiment 129) while one chat is the only one decoding;
            # YGARATY_RESIDENT=off keeps every step in the batched loop.
            fixture.update(
                conversations=chat_slots,
                resident=os.environ.get("YGARATY_RESIDENT") != "off",
                batch=self.args.batch,
                concurrent_chat=False,
            )
            fixture.update(
                local_loop=os.environ.get("YGARATY_LOCAL_LOOP") != "off",
                generate=options["capacity"],
                stop_ids=sorted(self.stop_ids),
            )
            if (
                route != "relay"
            ):  # the first stage stops a turn by the coordinator's rule (chat_token)
                fixture.update(
                    chat_direct=True,
                    route="webrtc",
                    ice_policy=self.args.ice_policy,
                    stages=len(self.stages),
                    generate=options["capacity"],
                    stop_ids=sorted(self.stop_ids),
                )

    def ice_servers(self, session, slot):
        """Short-lived TURN credentials for one slot of a direct session (user_webrtc.md, section 3)."""
        if self.args.route != "webrtc":
            raise PermissionError("This session does not use direct links")
        if self.turn_config is None:
            return {"ice": []}  # local network: host candidates only
        return self.turn_config.credentials(session, slot)

    def ice_urls(self):
        return self.turn_config.urls() if self.turn_config else []

    def download(self, name):
        """Public URL descriptor for one piece (spaces_download.mjs)."""
        if not self.storage or name not in self.files:
            raise ValueError("No Spaces piece with that name")
        return {
            "url": self.storage + quote(self.remote_paths[name], safe="/"),
            "bytes": self.files[name][2],
            "sha256": self.hashes[name],
        }

    def close(self, *, restarting=False):
        self.state = "cancelled"
        self.wake_signal_waiters()
        self.throughput.close()
        self.concurrent_chat.abort("Session stopped")
        if self.turn is not None:
            self.turn["events"].put({"error": "Session stopped"})
        for box in self.inbox:
            for _ in range(box.qsize()):
                try:
                    box.get_nowait()
                except queue.Empty:
                    break
            try:
                box.put(
                    ("stop", {"restarting": True} if restarting else {}, b"")
                )
            except SignalRejected:
                pass  # Shutdown must still release logs and retire reservations.
        for record in self.contributor_roster().snapshot()["contributors"]:
            self.contributor_roster().retire(
                record["id"], "coordinator stopped"
            )
        with self.log_lock:
            self.log.close()
