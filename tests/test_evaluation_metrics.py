"""Tests for evaluation metrics using a stub tokenizer."""

from __future__ import annotations

from src.utils.evaluation_metrics import (
    compute_all_metrics,
    fertility,
    pct_continued_words,
    vocabulary_coverage,
)


class StubTokenizer:
    """Character-level tokenizer: each char is a token, whitespace preserved."""

    def __init__(self, vocab: set[str] | None = None):
        self._vocab = vocab if vocab is not None else set()

    def encode(self, text: str) -> list[int]:
        return list(range(len(text)))

    def get_vocab(self) -> dict:
        return {tok: i for i, tok in enumerate(self._vocab)}


class WordTokenizer:
    """Whitespace-splitting tokenizer with a controlled vocab."""

    def __init__(self, vocab: set[str]):
        self._vocab = vocab

    def encode(self, text: str) -> list[int]:
        return list(range(len(text.split())))

    def get_vocab(self) -> dict:
        return {tok: i for i, tok in enumerate(self._vocab)}


# ---------- fertility -------------------------------------------------------


def test_fertility_with_char_tokenizer():
    # "hello world" = 11 chars / 2 words = 5.5
    tok = StubTokenizer()
    assert fertility(tok, ["hello world"]) == 11 / 2


def test_fertility_with_word_tokenizer():
    # 1 token per word => fertility == 1
    tok = WordTokenizer(vocab=set())
    assert fertility(tok, ["hello world", "foo bar baz"]) == 1.0


def test_fertility_empty_input_is_zero():
    tok = StubTokenizer()
    assert fertility(tok, []) == 0.0
    assert fertility(tok, [""]) == 0.0


# ---------- vocabulary_coverage --------------------------------------------


def test_vocabulary_coverage_all_in_vocab():
    tok = WordTokenizer(vocab={"hello", "world"})
    assert vocabulary_coverage(tok, ["hello world"]) == 1.0


def test_vocabulary_coverage_none_in_vocab():
    tok = WordTokenizer(vocab=set())
    assert vocabulary_coverage(tok, ["hello world"]) == 0.0


def test_vocabulary_coverage_partial():
    tok = WordTokenizer(vocab={"hello"})
    # 1 of 2 word types covered
    assert vocabulary_coverage(tok, ["hello world"]) == 0.5


def test_vocabulary_coverage_no_words_is_one():
    tok = WordTokenizer(vocab=set())
    assert vocabulary_coverage(tok, []) == 1.0


# ---------- pct_continued_words --------------------------------------------


def test_pct_continued_words_all_split():
    # char tokenizer: every word >1 char is split
    tok = StubTokenizer()
    # "a bb ccc" -> words ["a", "bb", "ccc"], split = 2, total = 3
    assert pct_continued_words(tok, ["a bb ccc"]) == 2 / 3


def test_pct_continued_words_none_split():
    tok = WordTokenizer(vocab=set())
    assert pct_continued_words(tok, ["hello world foo"]) == 0.0


def test_pct_continued_words_empty_is_zero():
    tok = StubTokenizer()
    assert pct_continued_words(tok, []) == 0.0


# ---------- compute_all_metrics --------------------------------------------


def test_compute_all_metrics_returns_all_keys():
    tok = WordTokenizer(vocab={"hello"})
    result = compute_all_metrics(tok, ["hello world"])
    assert set(result.keys()) == {"fertility", "vocabulary_coverage", "pct_continued_words"}
    assert all(isinstance(v, float) for v in result.values())
