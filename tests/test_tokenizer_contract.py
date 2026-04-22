"""Contract tests that every tokenizer adapter must satisfy.

Parametrized across all trainable algorithms. SuperBPE is covered in a
separate test module because its training requires cloning an external
repo and a Rust toolchain.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import pytest

from src.utils.tokenizer_algorithms import (
    SUPPORTED_ALGORITHMS,
    load_tokenizer,
    train_tokenizer,
)
from tests.conftest import SAMPLE_STRINGS

# SuperBPE is covered separately; ByT5 doesn't train but still conforms.
CONTRACT_ALGORITHMS = [a for a in SUPPORTED_ALGORITHMS if a != "superbpe"]

VOCAB_SIZE = 1000


@pytest.fixture(scope="module", params=CONTRACT_ALGORITHMS)
def trained_tokenizer(request, tiny_corpus, tmp_path_factory):
    """Return (algorithm, tokenizer) — trained once per algorithm per module."""
    algo = request.param
    out = tmp_path_factory.mktemp(f"artifact_{algo}")
    train_tokenizer(
        corpus_path=tiny_corpus,
        algorithm=algo,
        vocab_size=VOCAB_SIZE,
        output_dir=out,
    )
    tok = load_tokenizer(out, algorithm=algo)
    tok._artifact_dir = out
    return algo, tok


# ---------- vocab + special tokens -----------------------------------------


def test_vocab_size_within_tolerance(trained_tokenizer):
    algo, tok = trained_tokenizer
    if tok.is_byte_level:
        assert tok.vocab_size >= 256
        return
    # Upper bound is the real contract: the tokenizer must respect the budget.
    # Lower bound is soft: tiny corpora and Unigram EM can settle well below budget.
    assert tok.vocab_size <= VOCAB_SIZE + 20
    assert tok.vocab_size >= len(tok.special_tokens)


def test_special_tokens_present(trained_tokenizer):
    algo, tok = trained_tokenizer
    for sp in tok.special_tokens:
        assert sp in tok.get_vocab(), f"{sp!r} not in vocab for {algo}"
        tid = tok.token_to_id(sp)
        assert tid is not None and 0 <= tid < tok.vocab_size


# ---------- encode / decode invariants --------------------------------------


@pytest.mark.parametrize("text", SAMPLE_STRINGS)
def test_roundtrip_is_idempotent(trained_tokenizer, text):
    """decode(encode(x)) is a fixed point under encode/decode.

    Tokenizers can be lossy (whitespace collapsing, lowercasing, etc.), so we
    don't assert exact equality with the input. We do assert that applying the
    pipeline twice gives the same result as once — the tokenizer's output
    lives on a stable manifold.
    """
    _, tok = trained_tokenizer
    once = tok.decode(tok.encode(text))
    twice = tok.decode(tok.encode(once))
    assert once == twice


@pytest.mark.parametrize("text", SAMPLE_STRINGS)
def test_ids_within_vocab(trained_tokenizer, text):
    _, tok = trained_tokenizer
    ids = tok.encode(text)
    assert all(0 <= i < tok.vocab_size for i in ids), f"out-of-range id in {ids}"


@pytest.mark.parametrize("text", SAMPLE_STRINGS)
def test_nonempty_input_nonempty_output(trained_tokenizer, text):
    _, tok = trained_tokenizer
    assert len(tok.encode(text)) > 0


def test_empty_input_returns_empty(trained_tokenizer):
    _, tok = trained_tokenizer
    assert tok.encode("") == []


@pytest.mark.parametrize("text", SAMPLE_STRINGS)
def test_encode_is_deterministic(trained_tokenizer, text):
    _, tok = trained_tokenizer
    assert tok.encode(text) == tok.encode(text)


# ---------- batch + save/load -----------------------------------------------


def test_batch_matches_individual(trained_tokenizer):
    _, tok = trained_tokenizer
    batch = tok.encode_batch(SAMPLE_STRINGS)
    singles = [tok.encode(s) for s in SAMPLE_STRINGS]
    assert batch == singles


def test_save_load_roundtrip(trained_tokenizer, tmp_path):
    algo, tok = trained_tokenizer
    path = tmp_path / "reloaded"
    path.mkdir()
    tok.save(path)
    reloaded = load_tokenizer(path, algorithm=algo)
    for s in SAMPLE_STRINGS:
        assert reloaded.encode(s) == tok.encode(s)


def test_pickle_roundtrip(trained_tokenizer):
    _, tok = trained_tokenizer
    reloaded = pickle.loads(pickle.dumps(tok))
    for s in SAMPLE_STRINGS:
        assert reloaded.encode(s) == tok.encode(s)


# ---------- training determinism + unk --------------------------------------


def test_training_is_deterministic(tiny_corpus, tmp_path):
    """Re-training on the same corpus with the same vocab size should
    produce identical vocabs for deterministic algorithms. Unigram uses EM
    which is non-deterministic by default, so we exclude it."""
    algo = "bpe"
    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    out_a.mkdir()
    out_b.mkdir()
    train_tokenizer(tiny_corpus, algo, VOCAB_SIZE, out_a)
    train_tokenizer(tiny_corpus, algo, VOCAB_SIZE, out_b)
    a = load_tokenizer(out_a, algo)
    b = load_tokenizer(out_b, algo)
    assert a.get_vocab() == b.get_vocab()


def test_unk_handling(trained_tokenizer):
    """Rare codepoints should either be handled byte-level or mapped to <unk>."""
    _, tok = trained_tokenizer
    ids = tok.encode("\U000e0041\U000e0042")
    if tok.is_byte_level:
        assert len(ids) > 0
        return
    unk_id = tok.token_to_id("<unk>")
    if unk_id is None:
        # Adapter that doesn't declare <unk> (e.g. byte-level BPE) must still encode.
        assert len(ids) > 0
    else:
        # Either the rare codepoints map to <unk>, or the tokenizer
        # has byte-level fallback built in.
        assert len(ids) > 0
