"""Anonymous HTTPS prepared-bundle downloads with a verified local cache."""

import hashlib
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from .bundle_io import file_digest
from .bundle_validation import remote_base, validate_bundle
from .configuration import json_object


@dataclass(frozen=True, slots=True)
class PublicPreset:
    url: str
    contributors: int
    context: int
    output: int


def _presets():
    from .model_registry import load

    registry = load()
    return {
        (name, workers): PublicPreset(
            registry.bundle_url(name, workers),
            workers,
            record.public.context,
            record.public.output,
        )
        for name, record in registry.models.items()
        if record.public is not None
        for workers in record.public.workers
    }


# Read from the committed models.toml (see model_registry).
PUBLIC_PRESETS = _presets()
PUBLIC_BUNDLES = {key: preset.url for key, preset in PUBLIC_PRESETS.items()}
MANIFEST_LIMIT = 16 << 20
METADATA_LIMIT = 64 << 20
TOTAL_LIMIT = 64 << 30


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError(
            "Bundle downloads require direct HTTPS URLs, not redirects"
        )


def public_url(url):
    parsed = urlsplit(url)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(
            "Use a public HTTPS URL without credentials, query or fragment"
        )
    return url


def open_public(url):
    request = Request(public_url(url), headers={"Accept-Encoding": "identity"})
    try:
        return build_opener(NoRedirect).open(request, timeout=60)
    except HTTPError as error:
        raise ValueError(
            f"Public bundle request returned HTTP {error.code}: {url}. "
            "The manifest and all referenced objects must allow "
            "anonymous reads; "
            "no Spaces credentials are used."
        ) from error
    except URLError as error:
        raise ValueError(
            f"Cannot read public bundle URL: {url}: {error.reason}"
        ) from error


def artifact_path(value):
    if not isinstance(value, str) or not re.fullmatch(
        r"[A-Za-z0-9_./-]+", value
    ):
        raise ValueError("Unsafe public artifact path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(
        part in {"", ".", ".."} for part in value.split("/")
    ):
        raise ValueError("Unsafe public artifact path")
    if value.lower() in {
        "bundle.json",
        ".source.json",
        ".remote-bundle.json",
        ".pending.json",
    }:
        raise ValueError("Artifact path conflicts with cache metadata")
    return value


def digest_string(value):
    if not isinstance(value, str) or not re.fullmatch("[a-f0-9]{64}", value):
        raise ValueError("Invalid artifact SHA-256")
    return value


def inventory(data):
    if data.get("format") != "ygaraty-webgpu-v1":
        raise ValueError("Expected a prepared ygaraty-webgpu-v1 bundle")
    if data.get("storage", {}).get("backend", "local") not in {
        "local",
        "spaces",
    }:
        raise ValueError("Unsupported published storage backend")
    files = data.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("Bundle must contain weight files")
    if (
        not isinstance(data.get("tokenizer"), str)
        or not isinstance(data.get("test_files"), dict)
        or not isinstance(data.get("expected"), list)
        or not all(isinstance(p, str) for p in data["expected"])
        or not all(isinstance(p, str) for p in data["test_files"].values())
    ):
        raise ValueError("Invalid bundle metadata paths")
    records = {}
    for item in files.values():
        if not isinstance(item, dict):
            raise ValueError("Invalid weight record")
        path = artifact_path(item["path"])
        size = item["length"]
        if (
            type(item["offset"]) is not int
            or item["offset"] != 0
            or type(size) is not int
            or size <= 0
        ):
            raise ValueError(
                "Published weights must be complete pieces with offset zero"
            )
        record = (size, digest_string(item["sha256"]))
        if path in records and records[path] != record:
            raise ValueError("Conflicting piece records")
        records[path] = record
    metadata = {
        data["tokenizer"],
        *data["test_files"].values(),
        *data["expected"],
    }
    pins = data.get("artifact_digests")
    if pins is not None and (
        not isinstance(pins, dict) or pins.keys() != metadata
    ):
        raise ValueError("Incomplete artifact digest inventory")
    for path in metadata:
        path = artifact_path(path)
        if path in records:
            raise ValueError("Metadata overlaps a weight piece")
        records[path] = (
            None,
            digest_string(pins[path]) if pins is not None else None,
        )
    paths = set(records)
    if any(
        str(parent) in paths
        for name in paths
        for parent in PurePosixPath(name).parents
    ):
        raise ValueError("Conflicting artifact file/directory paths")
    total = sum(
        size if size is not None else METADATA_LIMIT
        for size, _ in records.values()
    )
    if not records or total > TOTAL_LIMIT:
        raise ValueError("Bundle exceeds the 64 GiB download budget")
    return records


def download_file(url, target, size, expected, opener):
    limit = size if size is not None else METADATA_LIMIT
    total, digest = 0, hashlib.sha256()
    target.parent.mkdir(parents=True, exist_ok=True)
    with opener(url) as response, target.open("xb") as output:
        while block := response.read(min(1 << 20, limit - total + 1)):
            total += len(block)
            if total > limit:
                raise ValueError(
                    "Downloaded artifact exceeds declared size or limit"
                )
            digest.update(block)
            output.write(block)
    if size is not None and total != size:
        raise ValueError("Truncated prepared artifact")
    actual = digest.hexdigest()
    if expected is not None and actual != expected:
        raise ValueError("Downloaded artifact SHA-256 mismatch")
    return actual


def check_contribution_budget(data, contributors=None, memory_gb=None):
    """Check declared stage weights, not total browser/GPU peak memory."""
    if contributors is None and memory_gb is None:
        return
    stages = data.get("stages")
    if not isinstance(stages, list) or not stages:
        raise ValueError("Prepared bundle has no stage assignments")
    if contributors is not None and len(stages) != contributors:
        raise ValueError(
            f"Expected {contributors} contributors, but bundle has "
            f"{len(stages)} stages; choose a matching prepared bundle"
        )
    if memory_gb is not None:
        if type(memory_gb) is not int or memory_gb < 1:
            raise ValueError("Contribution weight budget must be positive")
        for stage in stages:
            try:
                lengths = [
                    data["files"][name]["length"] for name in stage["names"]
                ]
            except (KeyError, TypeError) as error:
                raise ValueError("Invalid stage weight inventory") from error
            if not lengths or any(
                type(n) is not int or n <= 0 for n in lengths
            ):
                raise ValueError("Invalid stage weight lengths")
            if sum(lengths) > memory_gb * 1024**3:
                raise ValueError(
                    f"Stage weights exceed the {memory_gb} GiB contribution "
                    "budget; choose a smaller prepared stage"
                )


def cached_bundle(url, output):
    receipt = output / ".source.json"
    if not receipt.is_file() or receipt.stat().st_size > MANIFEST_LIMIT:
        raise ValueError(
            "Incomplete or unrelated download cache; choose a new workspace"
        )
    source = json_object(receipt.read_bytes())
    if source.get("url") != url:
        raise ValueError("Download cache belongs to a different public URL")
    if file_digest(output / ".remote-bundle.json") != source["remote_sha256"]:
        raise ValueError("Cached remote manifest changed")
    report = validate_bundle(output / "bundle.json")
    if report.sha256 != source["local_sha256"]:
        raise ValueError("Cached bundle changed")
    return report.path


def fetch_bundle(
    url,
    output,
    *,
    opener=None,
    contributors=None,
    memory_gb=None,
    on_manifest=None,
    weights=True,
):
    """Copy public pieces locally; reuse only verified caches.

    HTTPS authenticates first-use metadata. Weight hashes and publisher
    pins are checked; legacy metadata is pinned locally on first download.
    This performs no tensor computations or checkpoint preparation.

    With ``weights=False`` only the tokenizer and self-test artifacts are
    copied. The local bundle then keeps a ``url`` storage backend whose
    ``base_url`` contributors download weight pieces from directly; their
    browsers verify each piece against the pinned SHA-256.
    """
    public_url(url)
    output = Path(output)
    if output.exists():
        path = cached_bundle(url, output)
        if (remote_base(json_object(path.read_bytes())) is None) != weights:
            raise ValueError(
                "Download cache was made "
                + ("without" if weights else "with")
                + " local weights; choose a new workspace"
            )
        check_contribution_budget(
            json_object(path.read_bytes()), contributors, memory_gb
        )
        if on_manifest:
            on_manifest(json_object(path.read_bytes()))
        return path
    opener = opener or open_public
    with opener(url) as response:
        raw = response.read(MANIFEST_LIMIT + 1)
    if len(raw) > MANIFEST_LIMIT:
        raise ValueError("Bundle manifest exceeds 16 MiB")
    data = json_object(raw)
    records = inventory(data)
    check_contribution_budget(data, contributors, memory_gb)
    if on_manifest:
        on_manifest(data)
    base = url.rsplit("/", 1)[0] + "/"
    output.mkdir(parents=True)
    try:
        pins = {}
        for name, (size, digest) in records.items():
            if size is not None and not weights:
                continue
            actual = download_file(
                base + quote(name, safe="/"),
                output / name,
                size,
                digest,
                opener,
            )
            if size is None:
                pins[name] = actual
        # Only storage delivery changes; plans and kernel options stay intact.
        data.pop("storage", None)
        if not weights:
            data["storage"] = {"backend": "url", "base_url": base}
        for record in data["files"].values():
            record.pop("object_key", None)
        data["artifact_digests"] = pins
        (output / ".remote-bundle.json").write_bytes(raw)
        pending = output / ".pending.json"
        pending.write_text(json.dumps(data, indent=2) + "\n")
        report = validate_bundle(pending)
        receipt = {
            "url": url,
            "remote_sha256": hashlib.sha256(raw).hexdigest(),
            "local_sha256": report.sha256,
        }
        (output / ".source.json").write_text(
            json.dumps(receipt, indent=2) + "\n"
        )
        pending.replace(output / "bundle.json")
    except BaseException:
        shutil.rmtree(output)
        raise
    return output / "bundle.json"
