"""Words the model never saw whole must not be dropped.

A chunk is 350 words, but the tokenizer truncates at 512 tokens. URLs, code and
citation markers take several tokens per word, so the tail of a dense chunk
never reached the model and every word in it was dropped, even when the model
would have kept it. The cut can also land inside a word, leaving the model a
prefix to judge.
"""

from __future__ import annotations

import numpy as np
import pytest

from headroom.transforms import kompress_compressor as kc
from headroom.transforms.kompress_compressor import KompressCompressor, KompressConfig

WINDOW = 7  # stands in for 512 tokens


class _Enc(dict):
    def word_ids(self, batch_index=0):
        return self["_word_ids"][batch_index]


class _Tok:
    """Gives each word ``tokens_per_word(word)`` tokens and cuts each row at WINDOW."""

    def __init__(self, tokens_per_word):
        self.tokens_per_word = tokens_per_word

    def __call__(self, chunk_words, **kw):
        batch = chunk_words if chunk_words and isinstance(chunk_words[0], list) else [chunk_words]
        rows = [
            [i for i, w in enumerate(words) for _ in range(self.tokens_per_word(w))][:WINDOW]
            for words in batch
        ]
        width = max(len(row) for row in rows)
        return _Enc(
            input_ids=np.zeros((len(rows), width), dtype=np.int64),
            attention_mask=np.array([[1] * len(row) + [0] * (width - len(row)) for row in rows]),
            _word_ids=[row + [None] * (width - len(row)) for row in rows],
        )


class _KeepAllModel:
    """Keeps every token it is shown."""

    def get_keep_mask(self, input_ids, attention_mask):
        return [[True] * len(row) for row in input_ids]

    def get_scores(self, input_ids, attention_mask):
        return [[1.0] * len(row) for row in input_ids]


class _RejectsLastTokenModel:
    """Rejects the last token of each row and keeps the rest.

    With two tokens per word a whole word survives on its first token. Only a
    word the cut split, present as a single trailing token, is rejected.
    """

    @staticmethod
    def _keep(mask):
        n = int(mask.sum())
        return [i < n - 1 for i in range(len(mask))]

    def get_keep_mask(self, input_ids, attention_mask):
        return [self._keep(mask) for mask in attention_mask]

    def get_scores(self, input_ids, attention_mask):
        return [[1.0 if k else 0.0 for k in self._keep(mask)] for mask in attention_mask]


CASES = {
    "cut between words": (_Tok(lambda w: 1), _KeepAllModel()),
    "cut inside a word": (_Tok(lambda w: 2), _RejectsLastTokenModel()),
    "word longer than the window": (
        _Tok(lambda w: WINDOW + 2 if w == "w5" else 2),
        _RejectsLastTokenModel(),
    ),
}


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("path", ["compress", "compress_batch"])
def test_words_the_model_did_not_see_whole_are_kept(monkeypatch, path, case):
    tokenizer, model = CASES[case]
    monkeypatch.setattr(kc, "_load_kompress", lambda *a, **k: (model, tokenizer, "onnx"))
    # The compressor must agree with the fake on where the window ends.
    monkeypatch.setattr(kc, "_KOMPRESS_MAX_TOKENS", WINDOW, raising=False)
    monkeypatch.setattr(kc, "_model_device_type", lambda *a, **k: "cpu")
    monkeypatch.setenv(kc._KOMPRESS_MUST_KEEP_ENV, "0")
    compressor = KompressCompressor(
        KompressConfig(enable_ccr=False, min_input_words=10, chunk_words=10)
    )
    content = " ".join(f"w{i}" for i in range(25))

    if path == "compress":
        monkeypatch.setattr(compressor, "_should_batch_single_content", lambda *a, **k: False)
        result = compressor.compress(content)
    else:
        monkeypatch.setattr(compressor, "_should_use_sequential_fallback", lambda: False)
        [result] = compressor.compress_batch([content], batch_size=8)

    # The model keeps every word it sees whole, so nothing may go missing.
    assert result.compressed.split() == content.split()
