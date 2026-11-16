"""Pure M-1 B control logic; no workload kernels or services."""

import math
from dataclasses import dataclass


@dataclass
class GrowthGate:
    entry_loss: float
    minimum_gain: float = 0.01
    plateau_gain: float = 0.001
    patience: int = 3
    max_rounds: int = 15
    prior_loss: float | None = None

    def __post_init__(self):
        for loss in (self.entry_loss, self.prior_loss):
            if loss is not None and (not math.isfinite(loss) or loss <= 0):
                raise ValueError(
                    "Reference losses must be finite and positive"
                )
        if any(
            not math.isfinite(v) or not 0 < v < 1
            for v in (self.minimum_gain, self.plateau_gain)
        ):
            raise ValueError("Relative gains must be between zero and one")
        if any(
            type(v) is not int or v < 1
            for v in (self.patience, self.max_rounds)
        ):
            raise ValueError("Round limits must be positive integers")
        self.best = self.anchor = self.entry_loss
        self.rounds = self.quiet = 0

    def observe(self, loss):
        if not math.isfinite(loss) or loss <= 0:
            raise ValueError("Observed loss must be finite and positive")
        self.rounds += 1
        self.best = min(self.best, loss)
        if loss < self.anchor * (1 - self.plateau_gain):
            self.anchor, self.quiet = loss, 0
        else:
            self.quiet += 1
        learned = self.best <= self.entry_loss * (1 - self.minimum_gain)
        if self.prior_loss is not None:
            learned = learned and self.best <= self.prior_loss * (
                1 - self.minimum_gain
            )
        if learned and self.quiet >= self.patience:
            return "grow"
        if self.rounds >= self.max_rounds:
            return "stalled"
        return "continue"
