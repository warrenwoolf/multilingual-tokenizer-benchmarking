"""Low-level evaluation metrics for tokenizer benchmarking.

Key metrics
-----------
fertility           – average tokens per word (lower = more efficient)
vocabulary_coverage – fraction of test words representable without UNK
token_length_dist   – distribution of token lengths (proxy for morphological fit)
pct_continued_words – fraction of words split into >1 token (split rate)
"""

from __future__ import annotations


def compute_all_metrics(tokenizer, sentences: list[str]) -> dict:
    """Return a dict of all metrics for a tokenizer evaluated on sentences."""
    return {
        "fertility": fertility(tokenizer, sentences),
        "vocabulary_coverage": vocabulary_coverage(tokenizer, sentences),
        "pct_continued_words": pct_continued_words(tokenizer, sentences),
    }


def fertility(tokenizer, sentences: list[str]) -> float:
    """Average number of tokens produced per whitespace-delimited word."""
    total_words = 0
    total_tokens = 0
    for sentence in sentences:
        words = sentence.split()
        if not words:
            continue
        total_words += len(words)
        total_tokens += len(tokenizer.encode(sentence))
    if total_words == 0:
        return 0.0
    return total_tokens / total_words


def vocabulary_coverage(tokenizer, sentences: list[str]) -> float:
    """Fraction of word types in sentences that are in the tokenizer vocab as a single token.

    Byte-level vocabularies (tiktoken, ByT5) store tokens in their byte-level
    encoding (e.g. ``Ġhello`` rather than ``hello``), so a raw word-string
    lookup against ``get_vocab()`` reports near-zero coverage. For those
    tokenizers we fall back to checking whether the word encodes to a single
    id, which is the property the metric is trying to capture in the first
    place.
    """
    word_types = {w for s in sentences for w in s.split()}
    if not word_types:
        return 1.0
    if getattr(tokenizer, "is_byte_level", False):
        covered = sum(1 for w in word_types if len(tokenizer.encode(w)) == 1)
    else:
        vocab = set(getattr(tokenizer, "get_vocab", lambda: {})())
        covered = sum(1 for w in word_types if w in vocab)
    return covered / len(word_types)


def pct_continued_words(tokenizer, sentences: list[str]) -> float:
    """Fraction of words that are split into more than one token."""
    total_words = 0
    split_words = 0
    for sentence in sentences:
        words = sentence.split()
        for word in words:
            total_words += 1
            if len(tokenizer.encode(word)) > 1:
                split_words += 1
    if total_words == 0:
        return 0.0
    return split_words / total_words
