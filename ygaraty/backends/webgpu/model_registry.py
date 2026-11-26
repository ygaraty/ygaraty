"""Ygaraty's public model registry, read from the committed models.toml.

This is the open-source project's own list. Hosts such as Kaa keep separate
registries and never import it.
"""

import tomllib
from dataclasses import dataclass
from functools import cache
from importlib.resources import files

SCHEMA = 1
_MODEL_KEYS = {"family", "layers", "splits", "memory_gb", "evidence", "validation", "public"}
_PUBLIC_KEYS = {"version", "workers", "context", "output"}


@dataclass(frozen=True, slots=True)
class PublicBundles:
    version: str
    workers: tuple[int, ...]
    context: int
    output: int


@dataclass(frozen=True, slots=True)
class ModelRecord:
    model_id: str
    family: str
    layers: int
    splits: tuple[int, ...]
    memory_gb: int
    evidence: str
    validation: str
    public: PublicBundles | None


@dataclass(frozen=True, slots=True)
class Registry:
    bundle_base: str
    models: dict[str, ModelRecord]

    def bundle_url(self, model, workers):
        record = self.models[model]
        if record.public is None or workers not in record.public.workers:
            raise KeyError((model, workers))
        return (
            f"{self.bundle_base}/{model}/prepared/"
            f"{record.public.version}/{workers}-workers/bundle.json"
        )


def _int(value, name, low=1):
    if type(value) is not int or value < low:
        raise ValueError(f"models.toml: {name} must be an integer >= {low}")
    return value


def _text(value, name):
    if not isinstance(value, str) or not value:
        raise ValueError(f"models.toml: {name} must be a non-empty string")
    return value


def _public(name, raw):
    if not isinstance(raw, dict) or set(raw) != _PUBLIC_KEYS:
        raise ValueError(f"models.toml: {name}.public needs exactly {sorted(_PUBLIC_KEYS)}")
    workers = raw["workers"]
    if not isinstance(workers, list) or not workers or workers != sorted(set(workers)):
        raise ValueError(f"models.toml: {name}.public.workers must be ascending and unique")
    context = _int(raw["context"], f"{name}.public.context", 512)
    output = _int(raw["output"], f"{name}.public.output")
    if output >= context:
        raise ValueError(f"models.toml: {name}.public.output must be below context")
    return PublicBundles(
        _text(raw["version"], f"{name}.public.version"),
        tuple(_int(w, f"{name}.public.workers") for w in workers),
        context,
        output,
    )


def parse(data):
    """Validate a parsed models.toml; unknown or missing keys are errors."""
    if data.get("schema") != SCHEMA or set(data) != {"schema", "bundle_base", "models"}:
        raise ValueError(f"models.toml: expected schema = {SCHEMA}, bundle_base and models")
    base = _text(data["bundle_base"], "bundle_base").rstrip("/")
    if not base.startswith("https://"):
        raise ValueError("models.toml: bundle_base must be an HTTPS URL")
    models = {}
    for name, raw in data["models"].items():
        if not isinstance(raw, dict) or not set(raw) <= _MODEL_KEYS or not (_MODEL_KEYS - {"public"}) <= set(raw):
            raise ValueError(f"models.toml: {name} needs exactly {sorted(_MODEL_KEYS - {'public'})} (+ optional public)")
        layers = _int(raw["layers"], f"{name}.layers")
        splits = raw["splits"]
        if (
            not isinstance(splits, list)
            or len(splits) < 2
            or splits[0] != 0
            or splits[-1] != layers
            or splits != sorted(set(splits))
        ):
            raise ValueError(f"models.toml: {name}.splits must rise from 0 to layers")
        models[name] = ModelRecord(
            name,
            _text(raw["family"], f"{name}.family"),
            layers,
            tuple(_int(s, f"{name}.splits", 0) for s in splits),
            _int(raw["memory_gb"], f"{name}.memory_gb"),
            _text(raw["evidence"], f"{name}.evidence"),
            _text(raw["validation"], f"{name}.validation"),
            _public(name, raw["public"]) if "public" in raw else None,
        )
    return Registry(base, models)


@cache
def load():
    text = files(__package__).joinpath("models.toml").read_text(encoding="utf-8")
    return parse(tomllib.loads(text))
