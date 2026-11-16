"""User-supplied PyTorch models over the DiLoCo round contract."""

import math
from itertools import islice

from ygaraty.training.coordinator import RoundResult, copy_tensors


class TorchBackend:
    """Own a persistent replica; validation errors precede model mutation.

    Native parameter ranks/devices are preserved. Wire tensors default to
    [1, numel]; explicit wire_shapes support existing model layouts. Buffers
    are rejected unless buffer_policy='local' explicitly keeps them private.
    A failure after rebasing is terminal, not a promise of rollback/recovery.
    """

    def __init__(
        self,
        worker,
        model,
        optimizer,
        loss_fn,
        *,
        prepare_batch=None,
        parameters=None,
        wire_shapes=None,
        buffer_policy="reject",
        capture_state=None,
        restore_state=None,
    ):
        import torch

        if (capture_state is None) != (restore_state is None):
            raise ValueError("Supply both application state hooks")
        self.capture_state, self.restore_state = capture_state, restore_state

        if buffer_policy not in ("reject", "local"):
            raise ValueError("Unknown buffer policy")
        if (
            hasattr(model, "named_buffers")
            and list(model.named_buffers())
            and buffer_policy == "reject"
        ):
            raise ValueError("Model buffers require explicit local policy")
        self.torch, self.worker = torch, worker
        self.model, self.optimizer, self.loss_fn = model, optimizer, loss_fn
        self.prepare_batch = prepare_batch or (lambda *arrays: arrays)
        self.parameters = dict(
            model.named_parameters() if parameters is None else parameters
        )
        if not self.parameters or any(
            not isinstance(n, str) or not n for n in self.parameters
        ):
            raise ValueError("Expected named model parameters")
        ids = {id(p) for p in self.parameters.values()}
        if len(ids) != len(self.parameters):
            raise ValueError("Tied parameters must have one canonical name")
        if hasattr(model, "named_parameters") and ids != {
            id(p) for _, p in model.named_parameters()
        }:
            raise ValueError("Parameter mapping must cover the entire model")
        updated = [
            p for group in optimizer.param_groups for p in group["params"]
        ]
        trainable = {
            id(p) for p in self.parameters.values() if p.requires_grad
        }
        if (
            len({id(p) for p in updated}) != len(updated)
            or {id(p) for p in updated} != trainable
        ):
            raise ValueError(
                "Optimizer must own exactly the trainable parameters"
            )
        self.native_shapes = {
            n: tuple(p.shape) for n, p in self.parameters.items()
        }
        if any(
            not p.is_floating_point() or p.numel() < 1
            for p in self.parameters.values()
        ):
            raise ValueError("Parameters must be nonempty floating tensors")
        self.shapes = (
            {n: (1, p.numel()) for n, p in self.parameters.items()}
            if wire_shapes is None
            else {n: tuple(s) for n, s in wire_shapes.items()}
        )
        if self.shapes.keys() != self.parameters.keys() or any(
            len(s) != 2
            or any(type(v) is not int or v < 1 for v in s)
            or math.prod(s) != self.parameters[n].numel()
            for n, s in self.shapes.items()
        ):
            raise ValueError("Invalid wire parameter layout")
        self.version, self.seen, self.failed = -1, set(), False

    def validate_weights(self, weights):
        weights = copy_tensors(weights)
        if weights.keys() != self.shapes.keys() or any(
            w.shape != self.shapes[n] for n, w in weights.items()
        ):
            raise ValueError("Parameter layout mismatch")
        return weights

    def export_weights(self):
        """Detached named arrays for initial coordination or a round result."""
        return copy_tensors(
            {
                n: p.detach()
                .to(device="cpu", dtype=self.torch.float64)
                .numpy()
                .reshape(self.shapes[n])
                for n, p in self.parameters.items()
            }
        )

    def run_round(self, assignment, batches):
        if (
            self.failed
            or assignment.worker != self.worker
            or not isinstance(assignment.round_id, str)
            or not assignment.round_id
            or assignment.round_id in self.seen
            or type(assignment.version) is not int
            or assignment.version <= self.version
            or type(assignment.steps) is not int
            or assignment.steps < 1
        ):
            raise ValueError("Invalid or already executed round")
        batches = list(islice(iter(batches), assignment.steps + 1))
        if len(batches) != assignment.steps:
            raise ValueError("Expected exactly the assigned number of batches")
        base = self.validate_weights(assignment.weights)
        if any(
            tuple(p.shape) != self.native_shapes[n]
            for n, p in self.parameters.items()
        ):
            raise ValueError("Model parameter shape changed")
        torch = self.torch
        try:
            with torch.no_grad():
                for name, value in base.items():
                    p = self.parameters[name]
                    incoming = torch.from_numpy(
                        value.reshape(self.native_shapes[name])
                    ).to(device=p.device, dtype=p.dtype)
                    if not bool(torch.isfinite(incoming).all()):
                        raise ValueError("Weights overflow parameter dtype")
                    p.copy_(incoming)
            if hasattr(self.model, "train"):
                self.model.train()
            losses = []
            for batch in batches:
                self.optimizer.zero_grad(set_to_none=True)
                loss = self.loss_fn(self.model, batch)
                if loss.ndim != 0:
                    raise ValueError("loss_fn must return a scalar tensor")
                value = float(loss.detach())
                if not math.isfinite(value):
                    raise ValueError("Nonfinite loss")
                loss.backward()
                self.optimizer.step()
                losses.append(value)
            after = self.export_weights()
            delta = copy_tensors({n: base[n] - after[n] for n in base})
        except Exception:
            self.failed = True
            raise
        self.version = assignment.version
        self.seen.add(assignment.round_id)
        return RoundResult(
            assignment.round_id,
            assignment.version,
            self.worker,
            assignment.steps,
            delta,
            tuple(losses),
        )

    def checkpoint(self):
        """Capture model, optimizer, RNG and application state."""
        import random

        import numpy as np

        from ygaraty.training.recovery import bounded, pack

        if self.failed:
            raise ValueError("Cannot checkpoint a failed backend")

        def encode(value):
            if self.torch.is_tensor(value):
                return {
                    "type": "torch_tensor",
                    "dtype": str(value.dtype).split(".")[-1],
                    "shape": list(value.shape),
                    "data": value.detach().cpu().reshape(-1).tolist(),
                }
            if isinstance(value, dict):
                return {
                    "type": "dict",
                    "items": [
                        [encode(k), encode(v)] for k, v in value.items()
                    ],
                }
            if isinstance(value, (tuple, list)):
                return {
                    "type": "tuple" if isinstance(value, tuple) else "list",
                    "items": [encode(v) for v in value],
                }
            return pack(value)

        names = {id(p): n for n, p in self.parameters.items()}
        model_state = self.model.state_dict()
        state = dict(
            format="ygaraty.torch-state/v1",
            worker=self.worker,
            optimizer=f"{type(self.optimizer).__module__}.{type(self.optimizer).__qualname__}",
            groups=[
                [names[id(p)] for p in g["params"]]
                for g in self.optimizer.param_groups
            ],
            model=model_state,
            model_metadata=getattr(model_state, "_metadata", None),
            private_buffers={
                n: v
                for n, v in self.model.named_buffers()
                if n not in model_state
            }
            if hasattr(self.model, "named_buffers")
            else {},
            optimizer_state=self.optimizer.state_dict(),
            version=self.version,
            seen=sorted(self.seen),
            python_rng=random.getstate(),
            numpy_rng=np.random.get_state(),
            torch_rng=self.torch.get_rng_state(),
            cuda_rng=self.torch.cuda.get_rng_state_all()
            if self.torch.cuda.is_initialized()
            else None,
            extra=self.capture_state() if self.capture_state else None,
        )
        result = encode(state)
        bounded(result)
        return result

    def restore(self, checkpoint):
        import random

        import numpy as np

        from ygaraty.training.recovery import unpack

        def decode(value):
            if isinstance(value, dict):
                if value.get("type") == "torch_tensor":
                    dtype = getattr(self.torch, value["dtype"], None)
                    if value["dtype"] not in (
                        "float16",
                        "bfloat16",
                        "float32",
                        "float64",
                        "int8",
                        "uint8",
                        "int16",
                        "int32",
                        "int64",
                        "bool",
                    ):
                        raise ValueError("Unsupported tensor dtype")
                    return self.torch.tensor(
                        value["data"], dtype=dtype
                    ).reshape(value["shape"])
                if value["type"] == "dict":
                    return {decode(k): decode(v) for k, v in value["items"]}
                if value["type"] in ("tuple", "list"):
                    items = [decode(v) for v in value["items"]]
                    return tuple(items) if value["type"] == "tuple" else items
            return unpack(value)

        state = decode(checkpoint)
        names = {id(p): n for n, p in self.parameters.items()}
        groups = [
            [names[id(p)] for p in g["params"]]
            for g in self.optimizer.param_groups
        ]
        optimizer_type = type(self.optimizer)
        optimizer_name = (
            f"{optimizer_type.__module__}.{optimizer_type.__qualname__}"
        )
        if (
            state["format"] != "ygaraty.torch-state/v1"
            or state["worker"] != self.worker
            or state["optimizer"] != optimizer_name
            or state["groups"] != groups
        ):
            raise ValueError("Checkpoint backend identity mismatch")
        try:
            from collections import OrderedDict

            model_state = OrderedDict(state["model"])
            if state["model_metadata"] is not None:
                model_state._metadata = state["model_metadata"]
            self.model.load_state_dict(model_state, strict=True)
            buffers = (
                dict(self.model.named_buffers())
                if hasattr(self.model, "named_buffers")
                else {}
            )
            private = {
                n: v for n, v in buffers.items() if n not in model_state
            }
            if private.keys() != state["private_buffers"].keys():
                raise ValueError("Private buffer layout changed")
            if private:
                with self.torch.no_grad():
                    for name, value in state["private_buffers"].items():
                        target = private[name]
                        if tuple(target.shape) != tuple(value.shape):
                            raise ValueError("Private buffer shape changed")
                        target.copy_(
                            value.to(device=target.device, dtype=target.dtype)
                        )
            self.optimizer.load_state_dict(state["optimizer_state"])
            if self.restore_state:
                self.restore_state(state["extra"])
            elif state["extra"] is not None:
                raise ValueError(
                    "Checkpoint requires an application restore hook"
                )
            random.setstate(state["python_rng"])
            np.random.set_state(state["numpy_rng"])
            self.torch.set_rng_state(state["torch_rng"])
            if state["cuda_rng"] is not None:
                self.torch.cuda.set_rng_state_all(state["cuda_rng"])
            self.version, self.seen = state["version"], set(state["seen"])
            self.failed = False
        except Exception:
            self.failed = True
            raise
