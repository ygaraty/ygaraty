"""Registered parity cases: model, data, seed and optimizer, fixed by name.

A case owns both sides of the comparison: it builds the PyTorch reference and
lowers the same state and batch into an Ygaraty tape. Add a case here rather
than
writing another ad-hoc comparison loop.
"""

import hashlib
import json
from dataclasses import dataclass, field
from importlib.resources import files

import numpy as np

CORE_ROOT = files("ygaraty")

REFERENCE_SOURCES = dict(
    lstm=[
        CORE_ROOT / "reference/translator.py",
        CORE_ROOT / "verification/lstm_translation_oracle.py",
    ],
    transformer=[CORE_ROOT / "reference/transformer.py"],
)

TINY_PAIRS = [
    ("good morning my friend", "bom dia meu amigo"),
    ("where is the bus", "onde está o ônibus"),
    ("i like water", "eu gosto de água"),
    ("i like coffee", "eu gosto de café"),
    ("please open the door", "por favor abra a porta"),
    ("the cat is sleeping", "o gato está dormindo"),
    ("thank you my friend", "obrigado meu amigo"),
    ("see you tomorrow", "até amanhã"),
]


def pad(rows):
    return np.array(
        [r + [0] * (max(map(len, rows)) - len(r)) for r in rows],
        dtype=np.int64,
    )


def corpus_batches(pairs, batch, seed):
    """The tiny lifecycle harness's deterministic epoch-shuffled batches."""
    words = sorted(
        {w for pair in pairs for text in pair for w in text.split()}
    )
    vocab = {w: i + 4 for i, w in enumerate(words)}
    per_epoch = len(pairs) // batch

    def make(cursor):
        epoch, slot = divmod(cursor, per_epoch)
        ids = (
            np.random.default_rng(seed + epoch)
            .permutation(len(pairs))[slot * batch : (slot + 1) * batch]
            .tolist()
        )
        src = [[vocab[w] for w in pairs[i][0].split()] + [3] for i in ids]
        tgt = [[vocab[w] for w in pairs[i][1].split()] for i in ids]
        return dict(
            ids=ids,
            source=pad(src),
            target_input=pad([[2] + r for r in tgt]),
            targets=pad([r + [3] for r in tgt]),
        )

    return make, len(vocab) + 4


def random_batches(vocab, batch, max_source, max_target, seed):
    """Unsorted lengths, repeated IDs, source and target padding in every
    batch."""

    def make(cursor):
        rng = np.random.default_rng([seed, cursor])
        src = [
            rng.integers(4, vocab, rng.integers(1, max_source)).tolist() + [3]
            for _ in range(batch)
        ]
        tgt = [
            rng.integers(4, vocab, rng.integers(1, max_target)).tolist()
            for _ in range(batch)
        ]
        return dict(
            ids=None,
            source=pad(src),
            target_input=pad([[2] + r for r in tgt]),
            targets=pad([r + [3] for r in tgt]),
        )

    return make


@dataclass
class Case:
    name: str
    description: str
    architecture: str  # 'lstm' or 'transformer'
    model: dict
    batches: object  # cursor -> dict(ids, source, target_input, targets)
    seed: int
    learning_rate: float
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    smoothing: float = 0.0
    tile_size: int = 4
    steps: int = 3
    tags: list = field(default_factory=list)

    @property
    def optimizer(self):
        return dict(
            name="adam",
            learning_rate=self.learning_rate,
            beta1=self.beta1,
            beta2=self.beta2,
            epsilon=self.epsilon,
            clip_norm=0.0,
        )

    def describe(self):
        return dict(
            name=self.name,
            description=self.description,
            architecture=self.architecture,
            model=self.model,
            seed=self.seed,
            optimizer=self.optimizer,
            label_smoothing=self.smoothing,
            tags=self.tags,
        )

    def data_fingerprint(self, steps):
        h = hashlib.sha256()
        for cursor in range(steps):
            b = self.batches(cursor)
            for key in ("source", "target_input", "targets"):
                h.update(key.encode())
                h.update(np.ascontiguousarray(b[key]).tobytes())
        return h.hexdigest()

    # Reference side ---------------------------------------------------------
    def reference(self):
        import torch

        torch.set_num_threads(1)
        torch.manual_seed(self.seed)
        if self.architecture == "lstm":
            from ygaraty.verification.lstm_translation_oracle import (
                Float64Translator,
            )

            model = Float64Translator(**self.model).double().eval()
        else:
            from ygaraty.reference.transformer import (
                TransformerTranslator,
            )

            torch.backends.mha.set_fastpath_enabled(False)
            model = TransformerTranslator(**self.model).double().eval()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.learning_rate,
            betas=(self.beta1, self.beta2),
            eps=self.epsilon,
            foreach=False,
        )
        return model, optimizer

    def reference_step(self, model, optimizer, batch):
        """One PyTorch update; returns every expected tensor keyed by parameter
        name."""
        import torch

        src, inp, labels = (
            torch.tensor(batch[k])
            for k in ("source", "target_input", "targets")
        )
        lengths = torch.tensor(np.count_nonzero(batch["source"], axis=1))
        optimizer.zero_grad(set_to_none=True)
        logits = model(src, lengths, inp)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, self.model["vocab"]),
            labels.reshape(-1),
            ignore_index=0,
            label_smoothing=self.smoothing,
        )
        loss.backward()
        diagnostics = {}
        if self.architecture == "lstm":
            with torch.no_grad():
                enc, keys, mask, initial = model.encode(src, lengths)
                decoded, _ = model.decoder(model.embedding(inp), initial)
                _, attention = model.attend(decoded, enc, keys, mask)
            diagnostics = dict(
                encoder=enc.numpy(),
                decoder=decoded.numpy(),
                attention=attention.numpy(),
            )
        before = {
            n: p.detach().numpy().copy() for n, p in model.named_parameters()
        }
        gradients = {
            n: p.grad.numpy().copy() for n, p in model.named_parameters()
        }
        optimizer.step()
        after = {
            n: p.detach().numpy().copy() for n, p in model.named_parameters()
        }
        moments = {
            kind: {
                n: optimizer.state[p][key].numpy().copy()
                for n, p in model.named_parameters()
            }
            for kind, key in (("m", "exp_avg"), ("v", "exp_avg_sq"))
        }
        return dict(
            logits=logits.detach().numpy(),
            loss=loss.item(),
            diagnostics=diagnostics,
            before=before,
            gradients=gradients,
            after=after,
            moments=moments,
        )

    # Ygaraty side
    # -------------------------------------------------------------
    def build_tape(self, weights, batch):
        args = (
            self.model,
            weights,
            batch["source"],
            batch["target_input"],
            batch["targets"],
            self.smoothing,
        )
        if self.architecture == "lstm":
            from ygaraty.models.lstm_translation import (
                build_lstm_translation,
            )

            tape, logits, params = build_lstm_translation(*args)
            diagnostics = getattr(tape, "translation_diagnostics", {})
        else:
            from ygaraty.models.transformer import build_transformer

            tape, logits, params = build_transformer(*args)
            diagnostics = {}
        return tape, logits, params, diagnostics


def _registry():
    tiny, tiny_vocab = corpus_batches(TINY_PAIRS, 2, 2026)
    cases = [
        Case(
            "lstm-tiny-corpus",
            "The 100-update harness recipe: 8-pair EN-PT corpus, "
            "2-layer attention LSTM",
            "lstm",
            dict(vocab=64, embedding=8, hidden=16, layers=2, dropout=0.0),
            tiny,
            2026,
            0.01,
            tile_size=32,
            steps=5,
            tags=["smoke", "lstm"],
        ),
        Case(
            "lstm-edge-2layer",
            "Random padded batches with unsorted lengths; tiles cross "
            "gate boundaries",
            "lstm",
            dict(vocab=11, embedding=4, hidden=5, layers=2, dropout=0.0),
            random_batches(11, 3, 5, 4, 31),
            31,
            0.001,
            tile_size=3,
            steps=3,
            tags=["smoke", "lstm"],
        ),
        Case(
            "lstm-edge-1layer",
            "Single-layer variant of the edge case",
            "lstm",
            dict(vocab=11, embedding=4, hidden=5, layers=1, dropout=0.0),
            random_batches(11, 3, 5, 4, 7),
            7,
            0.001,
            tile_size=3,
            steps=3,
            tags=["lstm"],
        ),
        Case(
            "transformer-tiny",
            "Pre-norm encoder-decoder Transformer, label smoothing "
            "0.1, beta2 0.98",
            "transformer",
            dict(
                vocab=13,
                embedding=8,
                hidden=12,
                layers=2,
                heads=2,
                dropout=0.0,
                max_length=8,
            ),
            random_batches(13, 2, 5, 4, 17),
            17,
            1e-3,
            beta2=0.98,
            smoothing=0.1,
            tile_size=5,
            steps=3,
            tags=["smoke", "transformer"],
        ),
    ]
    assert tiny_vocab <= 64
    return {c.name: c for c in cases}


CASES = _registry()


def select(names):
    """Resolve case names, `all`, or `tag:<tag>`."""
    chosen = []
    for name in names:
        if name == "all":
            chosen += CASES.values()
        elif name.startswith("tag:"):
            chosen += [c for c in CASES.values() if name[4:] in c.tags]
        elif name in CASES:
            chosen.append(CASES[name])
        else:
            raise ValueError(
                f"Unknown case {name!r}; choose from {sorted(CASES)}"
            )
    return list({c.name: c for c in chosen}.values())


def reference_fingerprint(architecture):
    files = REFERENCE_SOURCES[architecture]
    return {
        str(f.relative_to(CORE_ROOT)): hashlib.sha256(
            f.read_bytes()
        ).hexdigest()
        if f.exists()
        else None
        for f in files
    }


def stable_hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True).encode()
    ).hexdigest()
