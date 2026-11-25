"""`ygaraty models list`: the models this release supports, from one place.

Supported means a published prepared bundle (launch downloads it with
--model NAME --num-workers N) plus a model profile with its validation
evidence. Experimental profiles run only from a local checkpoint.
"""

import json
import sys

from .bootstrap import PROFILES
from .downloads import PUBLIC_PRESETS


def catalog(include_experimental=False):
    rows = []
    for name, profile in PROFILES.items():
        workers = sorted(w for (model, w) in PUBLIC_PRESETS if model == name)
        if not workers and not include_experimental:
            continue
        rows.append(
            dict(
                model=name,
                family=profile.family,
                layers=profile.layers,
                support="supported" if workers else "experimental",
                public_bundle_workers=workers,
                default_splits=list(profile.splits),
                context_tokens=PUBLIC_PRESETS[(name, workers[0])].context
                if workers
                else None,
                validation=profile.validation,
                evidence=profile.evidence,
            )
        )
    return rows


def _workers(values):
    if not values:
        return "local checkpoint only"
    if values == list(range(values[0], values[-1] + 1)) and len(values) > 1:
        return f"{values[0]}–{values[-1]} workers"
    return ", ".join(map(str, values)) + " workers"


def show(include_experimental=False, as_json=False, out=None):
    out = out or sys.stdout
    rows = catalog(include_experimental)
    if as_json:
        print(json.dumps(rows, indent=2), file=out)
        return 0
    header = ("MODEL", "FAMILY", "LAYERS", "PUBLIC BUNDLE", "STATUS")
    table = [
        (
            r["model"],
            r["family"],
            str(r["layers"]),
            _workers(r["public_bundle_workers"]),
            r["support"],
        )
        for r in rows
    ]
    widths = [max(len(row[i]) for row in [header, *table]) for i in range(5)]

    def line(cells):
        return (
            "  "
            + "  ".join(c.ljust(w) for c, w in zip(cells, widths)).rstrip()
        )

    print("Supported models", file=out)
    print("", file=out)
    print(line(header), file=out)
    for row, data in zip(table, rows):
        print(line(row), file=out)
        print("    " + data["validation"], file=out)
    print("", file=out)
    print(
        "Start one: ygaraty launch --model MODEL --num-workers N",
        file=out,
    )
    if not include_experimental:
        print(
            "Experimental models (local checkpoints only): "
            "ygaraty models list --all",
            file=out,
        )
    return 0


def model_info(name, context=None, slots=1, opener=None):
    """Per worker count: GPU memory and download per worker, from the public
    manifests (metadata only; no weights are downloaded)."""
    from .configuration import json_object
    from .downloads import MANIFEST_LIMIT, open_public
    from .memory import estimate_memory

    profile = PROFILES.get(name)
    workers = sorted(w for (model, w) in PUBLIC_PRESETS if model == name)
    if profile is None or not workers:
        names = ", ".join(row["model"] for row in catalog())
        raise ValueError(
            f"No public bundle for {name!r}. Supported models: {names}"
        )
    opener = opener or open_public
    rows = []
    for count in workers:
        preset = PUBLIC_PRESETS[(name, count)]
        with opener(preset.url) as response:
            raw = response.read(MANIFEST_LIMIT + 1)
        if len(raw) > MANIFEST_LIMIT:
            raise ValueError("Bundle manifest exceeds 16 MiB")
        data = json_object(raw)
        reports = estimate_memory(data, context or preset.context, slots)
        pieces = {}
        for tensor, item in data["files"].items():
            # A piece shared by several tensors counts once.
            pieces[item.get("path", tensor)] = item["length"]
        rows.append(
            dict(
                workers=count,
                context_tokens=reports[0]["context"],
                slots=slots,
                max_output_tokens=(data.get("options") or {}).get("generate"),
                download_bytes=sum(pieces.values()),
                per_worker=[
                    dict(
                        worker=r["worker"],
                        layers=r["layers"],
                        weights_bytes=r["weights"],
                        recommended_gpu_bytes=r["recommended"],
                    )
                    for r in reports
                ],
            )
        )
    return dict(
        model=name,
        family=profile.family,
        layers=profile.layers,
        validation=profile.validation,
        evidence=profile.evidence,
        configurations=rows,
    )


def show_info(args, out=None, opener=None):
    out = out or sys.stdout
    info = model_info(args.name, args.context, args.slots, opener)
    if args.json:
        print(json.dumps(info, indent=2), file=out)
        return 0

    def gb(value):
        return f"{value / 1e9:.2f} GB"

    print(
        f"{info['model']}  ({info['family']}, {info['layers']} layers)",
        file=out,
    )
    print(f"  {info['validation']}", file=out)
    for row in info["configurations"]:
        print("", file=out)
        print(
            f"  {row['workers']} worker"
            + ("s" if row["workers"] != 1 else "")
            + f" · {row['context_tokens']:,}-token context · "
            f"{row['slots']} chat"
            + ("s" if row["slots"] != 1 else "")
            + f" · up to {row['max_output_tokens']} output tokens · "
            f"{gb(row['download_bytes'])} of weights in all",
            file=out,
        )
        for worker in row["per_worker"]:
            first, last = worker["layers"]
            print(
                f"    worker {worker['worker']}: layers {first}–{last - 1}, "
                f"{gb(worker['weights_bytes'])} download, "
                f"{gb(worker['recommended_gpu_bytes'])} free GPU memory",
                file=out,
            )
    print("", file=out)
    print(
        f"Start: ygaraty launch --model {info['model']} --num-workers N",
        file=out,
    )
    return 0


def add_parser(commands):
    models = commands.add_parser(
        "models", help="List supported models; show one model's needs"
    )
    operations = models.add_subparsers(dest="operation")
    listing = operations.add_parser("list", help="List supported models")
    info = operations.add_parser(
        "info",
        help="GPU memory and download per worker for 1–4 workers "
        "(reads the public manifests)",
    )
    info.add_argument("name", help="A model from 'ygaraty models list'")
    info.add_argument(
        "--context", type=int, help="Context tokens (default: the preset's)"
    )
    info.add_argument(
        "--slots", type=int, default=1, help="Chats at a time, 1–16"
    )
    info.add_argument(
        "--json", action="store_true", help="Machine-readable output"
    )
    for parser in (models, listing):
        parser.add_argument(
            "--all",
            action="store_true",
            help="Include experimental models (local checkpoints only)",
        )
        parser.add_argument(
            "--json", action="store_true", help="Machine-readable output"
        )


def run(args):
    if getattr(args, "operation", None) == "info":
        return show_info(args)
    return show(args.all, args.json)
