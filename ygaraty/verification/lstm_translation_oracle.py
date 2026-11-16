"""Independent PyTorch float64 oracle; never imported by distributed workers.

The pilot always casts attention scores to float32 for AMP stability. This
subclass preserves float64 for strict arithmetic parity; architecture,
parameter
names and packed nn.LSTM forward/backward remain those of the pilot.
"""

import torch

from ygaraty.reference.translator import Translator


class Float64Translator(Translator):
    def attend(self, decoded, enc, keys, mask):
        if decoded.dtype != torch.float64 or keys.dtype != torch.float64:
            raise ValueError("This oracle requires float64")
        scores = (
            torch.bmm(decoded, keys.transpose(1, 2))
            / self.config["hidden"] ** 0.5
        )
        weights = scores.masked_fill(mask[:, None, :], float("-inf")).softmax(
            -1
        )
        context = torch.bmm(weights, enc)
        features = self.dropout(
            torch.tanh(self.combine(torch.cat((decoded, context), -1)))
        )
        return torch.nn.functional.linear(
            features, self.embedding.weight, self.output_bias
        ), weights
