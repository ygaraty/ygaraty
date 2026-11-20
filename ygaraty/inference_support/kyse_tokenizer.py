"""Kyse's byte/rank BPE, preserving its pruned token IDs and ASCII splitting.

Format and semantics from Kyse (MIT, Copyright 2026 Felipe Cruz). Non-ASCII
bytes count as letters, matching its C tokenizer rather than Unicode BPE rules.
"""

import base64
import heapq
import json
import re
from types import SimpleNamespace

_WORD = re.compile(
    rb"(?i:'s|'t|'re|'ve|'m|'ll|'d)"
    rb"|[^\r\nA-Za-z0-9\x80-\xff]?[A-Za-z\x80-\xff]+"
    rb"|[0-9]| ?[^\sA-Za-z0-9\x80-\xff]+[\r\n]*"
    rb"|\s*[\r\n]+|\s+(?!\S)|\s+"
)


class KyseTokenizer:
    def __init__(self, document):
        if document.get("version") != 1:
            raise ValueError("Unsupported Kyse tokenizer version")
        self.tokens = [
            base64.b64decode(t, validate=True) for t in document["tokens"]
        ]
        self.ranks = document["ranks"]
        self.special = document["special_ids"]
        count = len(self.tokens)
        if (
            not 256 <= count <= 1_000_000
            or len(self.ranks) != count
            or any(type(r) is not int or r < -1 for r in self.ranks)
            or any(
                type(i) is not int or not 0 <= i < count for i in self.special
            )
            or len(set(self.special)) != len(self.special)
            or not self.special
            or any(not self.tokens[i] for i in self.special)
        ):
            raise ValueError("Invalid Kyse tokenizer")
        self.special_set = set(self.special)
        self.ids = {
            t: i
            for i, t in enumerate(self.tokens)
            if t and i not in self.special_set
        }
        if len(self.ids) != sum(
            bool(t)
            for i, t in enumerate(self.tokens)
            if i not in self.special_set
        ):
            raise ValueError("Duplicate Kyse token bytes")
        if any(bytes([i]) not in self.ids for i in range(256)):
            raise ValueError("Kyse tokenizer must retain all byte tokens")
        self.special_ids = {self.tokens[i]: i for i in self.special}
        if len(self.special_ids) != len(self.special):
            raise ValueError("Duplicate Kyse special token bytes")
        self.pattern = re.compile(
            b"|".join(re.escape(t) for t in self.special_ids)
        )

    def _bpe(self, word):
        pieces = {i: bytes([b]) for i, b in enumerate(word)}
        following = {i: i + 1 for i in range(len(word) - 1)}
        previous = {i: i - 1 for i in range(1, len(word))}
        heap = []

        def offer(left):
            right = following.get(left)
            if right is None:
                return
            joined = pieces[left] + pieces[right]
            token = self.ids.get(joined)
            if token is not None and self.ranks[token] >= 0:
                heapq.heappush(heap, (self.ranks[token], left, right, joined))

        for i in following:
            offer(i)
        while heap:
            _, left, right, joined = heapq.heappop(heap)
            if (
                left not in pieces
                or following.get(left) != right
                or pieces[left] + pieces[right] != joined
            ):
                continue
            pieces[left] = joined
            del pieces[right]
            after = following.pop(right, None)
            if after is None:
                following.pop(left, None)
            else:
                following[left] = after
                previous[after] = left
            previous.pop(right, None)
            offer(left)
            if left in previous:
                offer(previous[left])
        return [self.ids[pieces[i]] for i in sorted(pieces)]

    def encode(self, text, **_kwargs):
        raw = text.encode("utf-8")
        if b"\0" in raw:
            raise ValueError("Kyse text cannot contain NUL")
        ids, position = [], 0

        def plain(segment):
            for match in _WORD.finditer(segment):
                ids.extend(self._bpe(match.group()))

        for match in self.pattern.finditer(raw):
            plain(raw[position : match.start()])
            ids.append(self.special_ids[match.group()])
            position = match.end()
        plain(raw[position:])
        return SimpleNamespace(ids=ids)

    def encode_batch(self, texts, **kwargs):
        return [self.encode(text, **kwargs) for text in texts]

    def decode(self, ids, skip_special_tokens=True):
        if any(
            type(i) is not int or not 0 <= i < len(self.tokens) for i in ids
        ):
            raise ValueError("Invalid Kyse token ID")
        return b"".join(
            self.tokens[i]
            for i in ids
            if not skip_special_tokens or i not in self.special_set
        ).decode("utf-8", errors="replace")

    def token_to_id(self, text):
        raw = text.encode("utf-8")
        return self.special_ids.get(raw, self.ids.get(raw))


def load_tokenizer(path):
    """Load a standard tokenizer JSON or the explicit Kyse byte/rank schema."""
    document = json.loads(path.read_text())
    if document.get("type") == "ygaraty-kyse-bpe":
        return KyseTokenizer(document)
    from tokenizers import Tokenizer

    return Tokenizer.from_file(str(path))
