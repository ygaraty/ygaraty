"""A small, explicit encoder-decoder LSTM with Luong-style source attention.

The PyTorch reference for ygaraty.models.lstm_translation, copied from
tinyad's
scripts/translation/model.py so this repository needs no sibling checkout. The
parity
tests (tests/test_lstm_translation.py, the parity `lstm` cases) compare
Ygaraty's
LSTM against it; change both together.
"""

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

PAD, UNK, BOS, EOS = 0, 1, 2, 3


class Translator(nn.Module):
    def __init__(
        self, vocab=16000, embedding=256, hidden=512, layers=2, dropout=0.2
    ):
        super().__init__()
        self.config = dict(
            vocab=vocab,
            embedding=embedding,
            hidden=hidden,
            layers=layers,
            dropout=dropout,
        )
        self.embedding = nn.Embedding(vocab, embedding, padding_idx=PAD)
        self.encoder = nn.LSTM(
            embedding,
            hidden,
            layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if layers > 1 else 0,
        )
        self.decoder = nn.LSTM(
            embedding,
            hidden,
            layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0,
        )
        self.bridge_h = nn.Linear(2 * hidden, hidden)
        self.bridge_c = nn.Linear(2 * hidden, hidden)
        self.keys = nn.Linear(2 * hidden, hidden, bias=False)
        self.combine = nn.Linear(3 * hidden, embedding)
        self.output_bias = nn.Parameter(torch.zeros(vocab))
        self.dropout = nn.Dropout(dropout)

    def encode(self, source, lengths):
        packed = pack_padded_sequence(
            self.dropout(self.embedding(source)),
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        enc, (h, c) = self.encoder(packed)
        enc, _ = pad_packed_sequence(
            enc, batch_first=True, total_length=source.size(1)
        )

        def bridge(x, layer):
            # PyTorch order: layer0-forward, layer0-backward, layer1-forward,
            # ...
            return torch.tanh(layer(torch.cat((x[0::2], x[1::2]), dim=-1)))

        return (
            enc,
            self.keys(enc),
            source.eq(PAD),
            (bridge(h, self.bridge_h), bridge(c, self.bridge_c)),
        )

    def attend(self, decoded, enc, keys, mask):
        # Compute softmax in fp32, and never attend to source padding.
        scores = (
            torch.bmm(decoded.float(), keys.float().transpose(1, 2))
            / self.config["hidden"] ** 0.5
        )
        weights = scores.masked_fill(mask[:, None, :], float("-inf")).softmax(
            -1
        )
        context = torch.bmm(weights.to(enc.dtype), enc)
        features = self.dropout(
            torch.tanh(self.combine(torch.cat((decoded, context), -1)))
        )
        return nn.functional.linear(
            features, self.embedding.weight, self.output_bias
        ), weights

    def forward(self, source, lengths, target_input):
        enc, keys, mask, state = self.encode(source, lengths)
        decoded, _ = self.decoder(
            self.dropout(self.embedding(target_input)), state
        )
        return self.attend(decoded, enc, keys, mask)[0]

    @torch.no_grad()
    def generate(self, source, lengths, max_new=96):
        """Batched greedy decoding for reproducible held-out evaluation."""
        enc, keys, mask, state = self.encode(source, lengths)
        token = torch.full(
            (source.size(0), 1), BOS, device=source.device, dtype=torch.long
        )
        done = torch.zeros(
            source.size(0), device=source.device, dtype=torch.bool
        )
        result = []
        for _ in range(max_new):
            decoded, state = self.decoder(self.embedding(token), state)
            logits, _ = self.attend(decoded, enc, keys, mask)
            logits[:, :, PAD] = float("-inf")
            logits[:, :, BOS] = float("-inf")
            token = logits[:, -1].argmax(-1).masked_fill(done, EOS)[:, None]
            result.append(token)
            done |= token[:, 0].eq(EOS)
            if done.all():
                break
        return torch.cat(result, 1)

    @torch.no_grad()
    def beam(self, source, lengths, width=4, max_new=96, alpha=0.6):
        """Single-sentence beam search; length penalty includes generated
        EOS."""
        assert source.size(0) == 1
        enc, keys, mask, initial = self.encode(source, lengths)
        beams = [([BOS], 0.0, initial)]

        def rank(item):
            return item[1] / (((5 + len(item[0]) - 1) / 6) ** alpha)

        for _ in range(max_new):
            candidates = []
            for tokens, score, state in beams:
                if tokens[-1] == EOS:
                    candidates.append((tokens, score, state))
                    continue
                token = torch.tensor([[tokens[-1]]], device=source.device)
                decoded, next_state = self.decoder(
                    self.embedding(token), state
                )
                logits, _ = self.attend(decoded, enc, keys, mask)
                logits[0, 0, PAD] = float("-inf")
                logits[0, 0, BOS] = float("-inf")
                values, indices = (
                    logits[0, 0].float().log_softmax(-1).topk(width)
                )
                for value, index in zip(values.tolist(), indices.tolist()):
                    candidates.append(
                        (tokens + [index], score + value, next_state)
                    )
            beams = sorted(candidates, key=rank, reverse=True)[:width]
            if all(x[0][-1] == EOS for x in beams):
                break
        return max(beams, key=rank)[0][1:]
