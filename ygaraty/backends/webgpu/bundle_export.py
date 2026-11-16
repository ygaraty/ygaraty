"""Serialize detached local bundle pieces and self-test artifacts."""

import json
import shutil
from pathlib import Path

from ._legacy.chat_contract import chat_format
from .bundle_io import file_digest, range_digest


def export_bundle(session, output, tokenizer):
    """Serialize prepared data; byte copies and NumPy serialization only."""
    import numpy as np

    output = Path(output)
    (output / "pieces").mkdir()
    (output / "selftest").mkdir()
    files = {}
    for name, (source, offset, length) in session.files.items():
        digest = session.hashes[name]
        if len(digest) != 64 or any(
            c not in "0123456789abcdef" for c in digest
        ):
            raise ValueError("Invalid prepared weight hash")
        relative = f"pieces/{digest}.bin"
        target = output / relative
        if target.exists():
            actual = range_digest(source, offset, length)
            if target.stat().st_size != length:
                raise ValueError("Conflicting content-addressed piece length")
        else:
            with target.open("xb") as destination:
                actual = range_digest(
                    source, offset, length, output=destination
                )
        if actual != digest:
            raise ValueError("Weight source changed during preparation")
        files[name] = dict(
            path=relative, offset=0, length=length, sha256=digest
        )
    pins, test_files, expected = {}, {}, []
    for index, (name, content) in enumerate(session.test_files.items()):
        relative = f"selftest/input-{index}.bin"
        (output / relative).write_bytes(content)
        test_files[name] = relative
        pins[relative] = file_digest(output / relative)
    for index, reference in enumerate(session.expected):
        relative = f"selftest/expected-{index}.npy"
        np.save(output / relative, reference, allow_pickle=False)
        expected.append(relative)
        pins[relative] = file_digest(output / relative)
    shutil.copyfile(tokenizer, output / "tokenizer.json")
    pins["tokenizer.json"] = file_digest(output / "tokenizer.json")
    stages = json.loads(json.dumps(session.stages))
    for stage in stages:
        if stage["fixture"].get("ice"):
            raise ValueError("Stored bundles must not carry ICE credentials")
        stage["fixture"].pop("ice", None)
        stage["fixture"]["route"] = "relay"
    data = dict(
        format="ygaraty-webgpu-v1",
        stages=stages,
        files=files,
        test_files=test_files,
        selftest=session.selftest,
        expected=expected,
        tokenizer="tokenizer.json",
        eos=session.eos,
        options=dict(
            capacity=session.args.capacity,
            generate=session.args.generate,
            chunk=session.args.chunk,
            kv=session.args.kv,
            chat=chat_format(session.chat_style),
        ),
        distribution_plan=session.distribution_plan,
        artifact_digests=pins,
    )
    manifest = output / ".bundle.pending.json"
    payload = json.dumps(data, indent=2)
    if len(payload.encode()) > 16 << 20:
        raise ValueError("Bundle manifest exceeds 16 MiB")
    with manifest.open("x") as handle:
        handle.write(payload + "\n")
