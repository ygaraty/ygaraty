"""PyTorch reference for ygaraty.models.transformer_translation:
nn.Transformer
layers, pre-norm.

Parameter names are nn.TransformerEncoderLayer/DecoderLayer's
(self_attn.in_proj_weight,
out_proj, linear1, linear2, norm1..3) under `encoder.layers.N` and
`decoder.layers.N`, plus
embedding.weight (tied output projection), position.weight and output_bias.
"""

import math

import torch
from torch import nn

PAD, UNK, BOS, EOS = 0, 1, 2, 3


class TransformerTranslator(nn.Module):
    def __init__(
        self,
        vocab=16000,
        embedding=512,
        hidden=2048,
        layers=3,
        heads=8,
        dropout=0.0,
        max_length=64,
    ):
        super().__init__()
        self.config = dict(
            vocab=vocab,
            embedding=embedding,
            hidden=hidden,
            layers=layers,
            heads=heads,
            dropout=dropout,
            max_length=max_length,
        )
        self.embedding = nn.Embedding(vocab, embedding, padding_idx=PAD)
        self.position = nn.Embedding(max_length, embedding)
        encoder_layer = nn.TransformerEncoderLayer(
            embedding,
            heads,
            hidden,
            dropout,
            activation="relu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            layers,
            norm=nn.LayerNorm(embedding),
            enable_nested_tensor=False,
        )
        decoder_layer = nn.TransformerDecoderLayer(
            embedding,
            heads,
            hidden,
            dropout,
            activation="relu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer, layers, norm=nn.LayerNorm(embedding)
        )
        self.output_bias = nn.Parameter(torch.zeros(vocab))

    def embed(self, ids):
        positions = torch.arange(ids.size(1), device=ids.device)
        return (
            self.embedding(ids) * math.sqrt(self.config["embedding"])
            + self.position(positions)[None]
        )

    def forward(self, source, *args):
        """forward(source, target_input), or forward(source, lengths,
        target_input) as the older
        callers pass it; padding is read from the zeros, so lengths are not
        needed."""
        target_input = args[-1]
        torch.backends.mha.set_fastpath_enabled(False)
        src_pad, tgt_pad = source.eq(PAD), target_input.eq(PAD)
        causal = torch.triu(
            torch.ones(
                target_input.size(1),
                target_input.size(1),
                dtype=torch.bool,
                device=source.device,
            ),
            1,
        )
        memory = self.encoder(self.embed(source), src_key_padding_mask=src_pad)
        out = self.decoder(
            self.embed(target_input),
            memory,
            tgt_mask=causal,
            tgt_key_padding_mask=tgt_pad,
            memory_key_padding_mask=src_pad,
        )
        return nn.functional.linear(
            out, self.embedding.weight, self.output_bias
        )
