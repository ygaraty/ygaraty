"""Standalone local checkpoint inspection and immutable bundle preparation."""

import os
import shutil
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

from ._legacy.chat_contract import chat_format
from .bundle_export import export_bundle
from .bundle_io import local_input, read_json
from .bundle_validation import BundleReport as BundleReport
from .bundle_validation import validate_bundle as validate_bundle
from .configuration import integer


@dataclass(frozen=True, slots=True)
class Checkpoint:
    gguf: Path
    config: Path
    tokenizer: Path
    chat_format: Path | None = None

    def validate(self):
        local_input(self.gguf)
        read_json(self.config)
        tokenizer = local_input(self.tokenizer, limit=64 << 20)
        from tokenizers import Tokenizer

        try:
            Tokenizer.from_file(str(tokenizer))
        except Exception as error:
            raise ValueError("Invalid tokenizer file") from error
        if self.chat_format is not None:
            chat_format(read_json(self.chat_format, limit=1 << 20))


def inspect_checkpoint(gguf, config) -> dict:
    """Inspect local headers/configuration without dequantizing any weights."""
    from .checkpoints import families, gguf_remote

    path = local_input(gguf)
    config = read_json(config)
    locations = [local_input(p) for p in gguf_remote.split_names(str(path))]
    header = gguf_remote.read_split_headers(locations)
    family = families.detect(config, header)
    findings = family.check(config, header)
    # Single-file parsing also needs the range check applied to split files.
    if len(locations) == 1:
        spans = []
        for tensor in header.tensors.values():
            if (
                tensor.offset < header.data_start
                or tensor.nbytes <= 0
                or tensor.offset + tensor.nbytes > header.size
            ):
                raise ValueError(f"Invalid tensor range: {tensor.name}")
            spans.append((tensor.offset, tensor.offset + tensor.nbytes))
        spans.sort()
        if any(a[1] > b[0] for a, b in zip(spans, spans[1:])):
            raise ValueError("Overlapping GGUF tensor ranges")
    return dict(
        family=family.name,
        architecture=header.architecture(),
        geometry=family.geometry(config, header),
        files=[str(p) for p in locations],
        findings=findings,
        compatible=not any(f["level"] == "error" for f in findings),
        chat_template=header.metadata.get("tokenizer.chat_template"),
    )


class BundleBuilder:
    """Prepare existing local files; no downloads, services or cloud adapters.

    Reference self-tests perform numerical computation and therefore require an
    authorized preparation host. Offline tests mock this computation.
    """

    def __init__(
        self,
        checkpoint: Checkpoint,
        *,
        splits: tuple[int, ...] | None = None,
        context_tokens: int = 4096,
        max_output_tokens: int = 256,
    ):
        self.checkpoint = checkpoint
        self.splits = splits
        integer(context_tokens, "context_tokens")
        self.context_tokens = context_tokens
        integer(max_output_tokens, "max_output_tokens")
        self.max_output_tokens = max_output_tokens
        if max_output_tokens > context_tokens:
            raise ValueError("Output token limit exceeds context")

    def _chat(self, report):
        if self.checkpoint.chat_format is not None:
            return chat_format(
                read_json(self.checkpoint.chat_format, limit=1 << 20)
            )
        template = report.get("chat_template")
        if not isinstance(template, str):
            source = Path(self.checkpoint.tokenizer).with_name(
                "tokenizer_config.json"
            )
            if source.is_file():
                template = read_json(source).get("chat_template")
        if not isinstance(template, str) or len(template) > 1 << 20:
            raise ValueError("Supply --chat-format or a local ChatML template")
        from .checkpoints.chat_template import chat_format as convert

        return chat_format(convert(template))

    def prepare(self, output) -> BundleReport:
        """Copy pieces and publish bundle.json last; never overwrite."""
        from .checkpoints.preparation import prepare_session

        output = Path(output).absolute()
        if output.exists():
            raise FileExistsError("Use a new bundle output directory")
        self.checkpoint.validate()
        report = inspect_checkpoint(
            self.checkpoint.gguf, self.checkpoint.config
        )
        if not report["compatible"]:
            raise ValueError(f"Unsupported checkpoint: {report['findings']}")
        layers = report["geometry"]["layers"]
        bounds = self.splits if self.splits is not None else (0, layers)
        if (
            not isinstance(bounds, tuple)
            or len(bounds) < 2
            or any(type(n) is not int for n in bounds)
            or bounds[0] != 0
            or bounds[-1] != layers
            or any(a >= b for a, b in zip(bounds, bounds[1:]))
        ):
            raise ValueError(
                "Splits must cover all layers in increasing order"
            )
        style = self._chat(report)
        # Keep the audited preparation path and its optimized manifest layout.
        args = SimpleNamespace(
            gguf=local_input(self.checkpoint.gguf),
            config=local_input(self.checkpoint.config),
            splits=",".join(map(str, bounds)),
            capacity=self.context_tokens,
            generate=self.max_output_tokens,
            chat_format=None,
        )
        session = prepare_session(args)
        session.chat_style = style
        output.mkdir(parents=True, exist_ok=False)
        try:
            export_bundle(session, output, self.checkpoint.tokenizer)
            pending = output / ".bundle.pending.json"
            report = validate_bundle(pending)
            final = output / "bundle.json"
            os.link(
                pending, final
            )  # Atomic publication, refusing replacement.
            pending.unlink()
            return replace(report, path=final.resolve())
        except BaseException:
            shutil.rmtree(output)
            raise
