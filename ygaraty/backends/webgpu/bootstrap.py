"""Prepare and launch a local example through the existing public workflow."""

import json
import shlex
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace


@dataclass(frozen=True, slots=True)
class ModelProfile:
    model_id: str
    family: str
    layers: int
    splits: tuple[int, ...]
    memory_gb: int
    evidence: str
    validation: str


PROFILES = {
    "s5-nano": ModelProfile(
        "s5-nano",
        "qwen35",
        32,
        (0, 16, 32),
        3,
        "research/103-fast-sessions-rollout-rerun.md",
        "Two-stage execution measured; 3 GiB device acceptance pending",
    ),
    "qwen3.5-0.8b": ModelProfile(
        "qwen3.5-0.8b",
        "qwen35",
        24,
        (0, 12, 24),
        1,
        "Greedy tokens match the transformers reference (2026-11-15)",
        "1/2/4-worker WebGPU runs with headless contributors; "
        "browser and device variety pending",
    ),
    "qwen3.5-9b": ModelProfile(
        "qwen3.5-9b",
        "qwen35",
        32,
        (0, 8, 16, 24, 32),
        4,
        "research/092-qwen35-webgpu-four-contributors.md",
        "Measured four-contributor WebGPU run; extracted host still pending",
    ),
    "qwen3.8-27b": ModelProfile(
        "qwen3.8-27b",
        "qwen35",
        64,
        (0, 33, 64),
        10,
        "Greedy answers match llama.cpp on the same Q4_K_M weights until "
        "near-tied tokens (2026-11-16)",
        "1/2/4-worker WebGPU runs on one RTX 3090 with headless Chrome; "
        "multi-machine networks pending",
    ),
    "qwen3-coder-30b": ModelProfile(
        "qwen3-coder-30b",
        "qwen3moe",
        48,
        (0, 48),
        32,
        "research/DO-80.md",
        "Partial: whole-model hidden-state gate; "
        "multi-worker/tool gates pending",
    ),
}


@dataclass(frozen=True, slots=True)
class Bootstrap:
    profile: ModelProfile
    checkpoint: Path
    workspace: Path
    gguf_name: str = "model.gguf"
    context: int = 4096
    generate: int = 128
    host: str = "127.0.0.1"
    port: int = 8765

    def __post_init__(self):
        if Path(self.gguf_name).name != self.gguf_name:
            raise ValueError("--gguf must be a filename inside --checkpoint")
        if not 1 <= self.generate < self.context:
            raise ValueError("Require 0 < --generate < --context")
        if not 1 <= self.port <= 65535:
            raise ValueError("Port must be between 1 and 65535")

    def recipe(self):
        return {
            "version": 1,
            "model": self.profile.model_id,
            "checkpoint": str(self.checkpoint.resolve()),
            "gguf": self.gguf_name,
            "context": self.context,
            "generate": self.generate,
            "splits": list(self.profile.splits),
            "memory_gb": self.profile.memory_gb,
        }

    def serve_arguments(self):
        return SimpleNamespace(
            bundle=self.workspace / "model/bundle.json",
            model_id=self.profile.model_id,
            keys=self.workspace / "keys",
            state_dir=self.workspace / "state",
            context=self.context,
            memory_gb=self.profile.memory_gb,
            slots=1,
            host=self.host,
            port=self.port,
        )

    def describe(self):
        args = self.serve_arguments()
        command = ["ygaraty", "serve"]
        for name in (
            "bundle",
            "model_id",
            "keys",
            "state_dir",
            "context",
            "memory_gb",
            "slots",
            "host",
            "port",
        ):
            command.extend(
                ["--" + name.replace("_", "-"), str(getattr(args, name))]
            )
        return {
            **self.recipe(),
            "workspace": str(self.workspace.resolve()),
            "contributors": len(self.profile.splits) - 1,
            "evidence": self.profile.evidence,
            "validation": self.profile.validation,
            "resume_command": shlex.join(command),
            "actions": [
                "inspect local checkpoint",
                "prepare pinned bundle",
                "create owner/contributor keys",
                "validate bundle",
                "serve chat and contributor pages",
            ],
        }

    def prepare(self):
        """Reuse a completed matching workspace, never overwrite other data."""
        from .bundles import (
            BundleBuilder,
            Checkpoint,
            inspect_checkpoint,
            validate_bundle,
        )
        from .cli import initialize

        args = self.serve_arguments()
        receipt = self.workspace / "bootstrap.json"
        if self.workspace.exists():
            if not receipt.is_file() or receipt.stat().st_size > 16384:
                raise ValueError(
                    "Workspace is incomplete or unrelated; "
                    "choose a new --workspace"
                )
            saved = json.loads(receipt.read_text())
            if (
                not isinstance(saved, dict)
                or saved.get("recipe") != self.recipe()
            ):
                raise ValueError(
                    "Workspace belongs to a different bootstrap recipe"
                )
            for name in ("owner.key", "contributor.key"):
                key = args.keys / name
                if not key.is_file() or not key.read_text().strip():
                    raise ValueError(
                        "Workspace credentials missing; "
                        "refusing to replace keys"
                    )
            report = validate_bundle(args.bundle)
            if report.sha256 != saved.get("bundle_sha256"):
                raise ValueError(
                    "Workspace bundle differs from bootstrap receipt"
                )
            return args
        gguf = self.checkpoint / self.gguf_name
        config = self.checkpoint / "config.json"
        report = inspect_checkpoint(gguf, config)
        if (
            not report["compatible"]
            or report["family"] != self.profile.family
            or report["geometry"]["layers"] != self.profile.layers
        ):
            raise ValueError(
                "Checkpoint family/layer count does not match "
                "selected model profile"
            )
        chat = self.checkpoint / "chat.json"
        checkpoint = Checkpoint(
            gguf,
            config,
            self.checkpoint / "tokenizer.json",
            chat if chat.is_file() else None,
        )
        checkpoint.validate()
        self.workspace.mkdir(parents=True)
        # Existing builders own numerical references and atomic bundle export.
        # If preparation fails, keep the workspace for inspection; never serve.
        BundleBuilder(
            checkpoint,
            splits=self.profile.splits,
            context_tokens=self.context,
            max_output_tokens=self.generate,
        ).prepare(self.workspace / "model")
        initialize(args.keys)
        report = validate_bundle(args.bundle)
        pending = receipt.with_suffix(".tmp")
        pending.write_text(
            json.dumps(
                {"recipe": self.recipe(), "bundle_sha256": report.sha256},
                indent=2,
            )
            + "\n"
        )
        pending.replace(receipt)
        return args


def add_parser(commands):
    # Deprecated: launch does the same (--checkpoint, --prepare-only).
    parser = commands.add_parser("bootstrap")
    parser.add_argument("--model", choices=PROFILES, default="qwen3.5-9b")
    parser.add_argument(
        "--num-workers",
        "--workers",
        "--contributors",
        dest="contributors",
        type=int,
        required=True,
        help="Required number of GPU workers",
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--checkpoint", type=Path)
    source.add_argument(
        "--bundle-url", help="Public prepared bundle; preset used by default"
    )
    parser.add_argument("--workspace", type=Path)
    parser.add_argument(
        "--gguf",
        default="model.gguf",
        help="GGUF filename (first shard for split files)",
    )
    parser.add_argument("--context", type=int)
    parser.add_argument("--generate", type=int)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print recipe; no model reads, writes, inference or server",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Prepare numerical references without launching the host",
    )


def run(args):
    import sys

    print(
        "Note: 'ygaraty bootstrap' is deprecated and will be removed. Use "
        "'ygaraty launch --checkpoint DIR --model NAME --num-workers N' "
        "(add --prepare-only to prepare without starting).",
        file=sys.stderr,
        flush=True,
    )
    if args.workspace is None:
        args.workspace = (
            Path("runs") / args.model / f"{args.contributors}-workers"
        )
    if not 1 <= args.contributors <= 26:
        raise ValueError("--num-workers must be between 1 and 26")
    if args.checkpoint is None:
        from .downloads import PUBLIC_BUNDLES, public_url
        from .launch import run as launch

        url = args.bundle_url or PUBLIC_BUNDLES.get(
            (args.model, args.contributors)
        )
        if not url:
            raise ValueError(
                "No public bundle URL for this model; supply --bundle-url"
            )
        public_url(url)
        if args.dry_run:
            print(
                json.dumps(
                    {
                        "model": args.model,
                        "bundle_url": url,
                        "contributors": args.contributors,
                        "memory": "calculated from bundle at launch",
                        "workspace": str(args.workspace.resolve()),
                        "actions": [
                            "download and verify prepared bundle",
                            "initialize keys",
                            "serve",
                        ],
                    },
                    indent=2,
                )
            )
            return 0
        remote = SimpleNamespace(
            **vars(args),
            bundle=None,
            splits=None,
            memory_gb=None,
            slots=1,
            public_url=None,
            open_chat=False,
            local=True,
        )
        return launch(remote)
    profile = PROFILES[args.model]
    if args.contributors > profile.layers:
        raise ValueError("Worker count cannot exceed model layer count")
    if len(profile.splits) - 1 != args.contributors:
        profile = replace(
            profile,
            splits=tuple(
                i * profile.layers // args.contributors
                for i in range(args.contributors + 1)
            ),
        )
    setup = Bootstrap(
        replace(profile, memory_gb=64)
        if profile.family == "qwen35"
        else profile,
        args.checkpoint,
        args.workspace,
        args.gguf,
        args.context or 4096,
        args.generate if args.generate is not None else 128,
        args.host,
        args.port,
    )
    print(json.dumps(setup.describe(), indent=2))
    if args.dry_run:
        return 0
    from .cli import serve
    from .launch import configure_prepared_memory

    try:
        launch = setup.prepare()
    except ImportError as error:
        raise ValueError(
            "Install ygaraty[server,prepare] and its public packages"
        ) from error
    if profile.family == "qwen35":
        configure_prepared_memory(launch, args.contributors, setup.context, 1)
    if not args.prepare_only:
        print(f"Contributor page: http://{args.host}:{args.port}/")
        print(f"Chat page: http://{args.host}:{args.port}/chat")
        print(f"Keys: {launch.keys} (key values are never printed)")
        serve(launch)
    return 0
