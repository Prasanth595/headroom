"""Words past the model's 512-token window must not be dropped unscored.

A chunk is 350 words, but the tokenizer truncates at 512 tokens. URLs, code and
citation markers take several tokens per word, so the tail of a dense chunk
never reached the model and every word in it was dropped, even when the model
would have kept it.
"""

from __future__ import annotations

import pytest

from headroom.transforms import kompress_compressor as kc
from headroom.transforms.kompress_compressor import KompressCompressor, KompressConfig

WINDOW = 4  # stands in for 512 tokens: the fake tokenizer sees 4 words per call


class _Enc(dict):
    def word_ids(self, batch_index=0):
        return self["_word_ids"][batch_index]


class _TruncatingTok:
    """Keeps the first WINDOW words of each chunk, like truncation=True at max_length."""

    def __call__(self, chunk_words, **kw):
        batch = chunk_words if chunk_words and isinstance(chunk_words[0], list) else [chunk_words]
        seen = [min(len(words), WINDOW) for words in batch]
        return _Enc(
            input_ids=[[0] * n for n in seen],
            attention_mask=[[1] * n for n in seen],
            _word_ids=[list(range(n)) for n in seen],
        )


class _KeepAllModel:
    """Keeps every word it is shown."""

    def get_keep_mask(self, input_ids, attention_mask):
        return [[True] * len(row) for row in input_ids]

    def get_scores(self, input_ids, attention_mask):
        return [[1.0] * len(row) for row in input_ids]


@pytest.mark.parametrize("path", ["compress", "compress_batch"])
def test_words_past_the_token_window_are_kept(monkeypatch, path):
    monkeypatch.setattr(
        kc, "_load_kompress", lambda *a, **k: (_KeepAllModel(), _TruncatingTok(), "onnx")
    )
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

    # The model keeps everything it sees, so nothing may go missing.
    assert result.compressed.split() == content.split()
