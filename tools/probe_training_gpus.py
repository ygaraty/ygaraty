"""Verify per-process Vulkan GPU isolation before a same-host scaling run.

Requires NVIDIA management tools, Vulkan tools and Mesa's device-select layer.
The output contains hardware identifiers: keep it private; publish only counts
and verification outcomes. Never infer distinct GPUs from worker count alone.
"""

import argparse
import csv
import io
import json
import os
import re
import subprocess
from pathlib import Path


def probe(expected):
    raw = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,pci.bus_id,name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    rows = list(csv.reader(io.StringIO(raw), skipinitialspace=True))
    if len(rows) != expected:
        raise ValueError(
            f"Expected {expected} visible allocated GPUs, found {len(rows)}"
        )
    observed = []
    for index, uuid, bus, name, memory, driver in rows:
        domain, bus_id, device = bus.split(":")
        device, function = device.split(".")
        selector = (
            f"pci-{int(domain, 16):04x}_{bus_id.lower()}_"
            f"{device.lower()}_{function}!"
        )
        env = dict(
            os.environ,
            DRI_PRIME=selector,
            VK_LOADER_LAYERS_ENABLE="VK_LAYER_MESA_device_select",
        )
        result = subprocess.run(
            ["vulkaninfo", "--summary"],
            env=env,
            text=True,
            capture_output=True,
            timeout=45,
            check=True,
        )
        count = len(re.findall(r"^GPU\d+:", result.stdout, re.M))
        ids = re.findall(r"deviceUUID\s*=\s*(\S+)", result.stdout)
        if (
            count != 1
            or len(ids) != 1
            or "PHYSICAL_DEVICE_TYPE_DISCRETE_GPU" not in result.stdout
        ):
            raise RuntimeError(
                "Vulkan did not expose exactly one discrete GPU for selector "
                + selector
            )
        if (
            ids[0].replace("-", "").lower()
            != uuid.removeprefix("GPU-").replace("-", "").lower()
        ):
            raise RuntimeError(
                "Vulkan device UUID differs from the selected NVIDIA GPU"
            )
        observed.append(
            dict(
                index=int(index),
                uuid=uuid,
                pci_bus=bus,
                name=name,
                memory_mib=int(memory),
                driver=driver,
                selector=selector,
                vulkan_uuid=ids[0],
            )
        )
    if len({x["vulkan_uuid"] for x in observed}) != expected:
        raise RuntimeError("GPU selectors alias the same device")
    return observed


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--expected", type=int, default=4)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.expected < 1:
        p.error("expected must be positive")
    devices = probe(args.expected)
    args.output.mkdir(parents=True, exist_ok=True, mode=0o700)
    (args.output / "gpu-isolation.json").write_text(
        json.dumps(devices, indent=2) + "\n"
    )
    for count in [1, args.expected]:
        (args.output / f"selectors-{count}.json").write_text(
            json.dumps([x["selector"] for x in devices[:count]]) + "\n"
        )
    print(
        json.dumps(
            dict(
                distinct_physical_gpus=len(devices),
                vulkan_single_device_filters_verified=True,
                models=[x["name"] for x in devices],
            )
        )
    )


if __name__ == "__main__":
    main()
