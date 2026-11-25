"""Publish public bundles with a larger context and output limit.

The prepared weights do not depend on the context or the output limit: only
the manifest's ``options`` and each stage's ``fixture`` carry them. This copies
every object of ``MODEL/prepared/SOURCE/N-workers`` to ``.../TARGET/...``
inside the bucket (server-side, no download) and writes a manifest with the
new limits. The source prefix is never changed, so existing caches and links
keep working.

Dry run by default; ``--apply`` copies and uploads. Needs boto3 and
YGARATY_SPACES_ACCESS_KEY_ID / YGARATY_SPACES_SECRET_ACCESS_KEY (for example
``set -a; . ~/.config/ygaraty/spaces.env``).

    python scripts/publish_bundle_limits.py --context 16384 --generate 4096
    python scripts/publish_bundle_limits.py --model s5-nano --apply
"""

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from ygaraty.backends.webgpu.downloads import inventory, open_public  # noqa: E402

BUCKET = "ygaraty-models"
ENDPOINT = "https://fra1.digitaloceanspaces.com"
HOST = f"https://{BUCKET}.fra1.digitaloceanspaces.com"
MODELS = ("qwen3.5-0.8b", "s5-nano", "qwen3.5-9b", "qwen3.8-27b")


def relimit(data, context, generate, source_prefix, target_prefix):
    """The manifest with new limits and object keys; weights unchanged."""
    if not 1 <= generate < context:
        raise ValueError("Require 0 < generate < context")
    out = json.loads(json.dumps(data))
    out["options"]["capacity"] = context
    out["options"]["generate"] = generate
    for stage in out["stages"]:
        stage["fixture"]["max_positions"] = context
        stage["fixture"]["generate"] = generate
        stage["fixture"]["max_tokens"] = min(
            stage["fixture"].get("max_tokens", 64), context
        )
    storage = out.get("storage") or {}
    if storage.get("prefix") != source_prefix:
        raise ValueError(f"Unexpected storage prefix {storage.get('prefix')}")
    storage["prefix"] = target_prefix
    for record in out["files"].values():
        key = record.get("object_key")
        if key is not None:
            if not key.startswith(source_prefix + "/"):
                raise ValueError(f"Object outside the bundle prefix: {key}")
            record["object_key"] = target_prefix + key[len(source_prefix) :]
    return out


def client():
    import boto3

    return boto3.client(
        "s3",
        region_name="fra1",
        endpoint_url=ENDPOINT,
        aws_access_key_id=os.environ["YGARATY_SPACES_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["YGARATY_SPACES_SECRET_ACCESS_KEY"],
    )


def publish(model, workers, args, s3):
    source = f"{model}/prepared/{args.source}/{workers}-workers"
    target = f"{model}/prepared/{args.target}/{workers}-workers"
    with open_public(f"{HOST}/{source}/bundle.json") as response:
        data = json.loads(response.read())
    names = sorted(inventory(data))
    out = relimit(data, args.context, args.generate, source, target)
    size = sum(
        f["length"]
        for f in {f["path"]: f for f in data["files"].values()}.values()
    )
    print(
        f"{source} -> {target}: {len(names)} objects, {size / 1e9:.2f} GB, "
        f"context {args.context}, output {args.generate}",
        flush=True,
    )
    if not args.apply:
        return

    def copy(name):
        s3.copy_object(
            Bucket=BUCKET,
            Key=f"{target}/{name}",
            CopySource={"Bucket": BUCKET, "Key": f"{source}/{name}"},
            ACL="public-read",
            MetadataDirective="COPY",
        )

    with ThreadPoolExecutor(16) as pool:
        list(pool.map(copy, names))
    # The manifest last: a reader never sees it before its objects exist.
    s3.put_object(
        Bucket=BUCKET,
        Key=f"{target}/bundle.json",
        Body=(json.dumps(out, indent=2) + "\n").encode(),
        ContentType="application/json",
        ACL="public-read",
        CacheControl="no-cache",
    )
    print(f"  published {HOST}/{target}/bundle.json", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", action="append", choices=MODELS)
    parser.add_argument(
        "--workers", type=int, action="append", choices=range(1, 5)
    )
    parser.add_argument("--source", default="v1")
    parser.add_argument("--target", default="v2")
    parser.add_argument("--context", type=int, default=16384)
    parser.add_argument("--generate", type=int, default=4096)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)
    s3 = client() if args.apply else None
    for model in args.model or MODELS:
        for workers in args.workers or range(1, 5):
            publish(model, workers, args, s3)


if __name__ == "__main__":
    main()
