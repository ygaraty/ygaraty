"""Synthetic source manifests; no model bytes, GPU calls or private imports."""

import json
from copy import deepcopy
from pathlib import Path

FIXTURES = Path(__file__).with_name("browser")


def cases():
    dense = json.loads((FIXTURES / "inference-plans.json").read_text())
    kolibri = json.loads((FIXTURES / "kolibri-plans.json").read_text())
    qwen3 = dict(
        family="qwen3moe",
        boundaries=[0, 2, 4],
        config=dict(
            family="qwen3moe",
            layers=4,
            hidden=2048,
            heads=32,
            kv_heads=4,
            head_dim=128,
            vocab=151936,
            experts=128,
            top_k=8,
            expert_ffn=768,
            norm_topk=True,
            rope_theta=1e7,
            eps=1e-6,
            attention_scale=128**-0.5,
            tied_embeddings=False,
            intermediate=8 * 2 * 768,
        ),
    )
    return [*dense, kolibri, qwen3]


def stages(case):
    result = []
    config = case["config"]
    boundaries = case["boundaries"]
    family = case["family"]
    for owner, (first, last) in enumerate(zip(boundaries, boundaries[1:])):
        layers = []
        for index in range(first, last):
            if family == "dense":
                kind = "dense"
            elif family == "kolibri1":
                kind = "sliding" if config["sliding"][index] else "full"
            elif family == "qwen3moe":
                kind = "full"
            else:
                kind = "full" if index % 4 == 3 else "linear"
            layers.append(dict(index=index, type=kind))
        result.append(
            dict(
                stage=owner,
                layers=[first, last],
                names=["synthetic-weight"],
                fixture=dict(
                    wire="f16",
                    route="relay",
                    conversations=1,
                    max_positions=1024,
                    max_tokens=128,
                ),
                manifest=dict(
                    config=deepcopy(config),
                    layers=layers,
                    embed={"chunks": []} if first == 0 else None,
                    head={"chunks": []} if last == boundaries[-1] else None,
                ),
            )
        )
    return result


def bundle(case=None):
    return dict(
        format="ygaraty-webgpu-v1",
        options=dict(capacity=1024, chunk=128),
        stages=stages(case or cases()[0]),
        files={"synthetic-weight": {"length": 16}},
    )
