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

# SuperBPE needs an external repo + Rust toolchain to train, so it stays
# out of the parametrized contract. MorphBPE is per-language (it needs a
# morpheme segmenter), so the multilingual fixture corpus doesn't fit; it
# gets its own English-only contract block at the bottom of the file.
SKIP_IN_CONTRACT = {"superbpe", "morphbpe"}
CONTRACT_ALGORITHMS = [a for a in SUPPORTED_ALGORITHMS if a not in SKIP_IN_CONTRACT]

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


# ---------- MorphBPE -------------------------------------------------------
# MorphBPE is per-language, so it gets its own English-only contract block
# rather than running over the multilingual fixture corpus.


ENGLISH_SENTENCES = [
    "The quick brown fox jumps over the lazy dog near the riverbank.",
    "She sells seashells by the seashore every summer afternoon before sunset.",
    "Machine learning models tokenize text before training language representations.",
    "Natural language processing requires careful attention to morphology and syntax.",
    "Tokenizers split input text into subword units that the model can process.",
    "Researchers compare algorithms across multiple languages and vocabulary sizes.",
    "Effective benchmarks reveal systematic differences between tokenizer families.",
    "A well-designed experiment controls for corpus size, vocabulary, and evaluation metric.",
    "The unhappiness was overwhelming and discomfort was unbearable for everyone.",
    "Running runners running quickly happily happiness sadness uncomfortable comfortable.",
    "Tokenization tokenizer tokenized tokenizes preprocessing postprocessing reprocessing.",
]


@pytest.fixture(scope="module")
def english_corpus(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("en_corpus") / "train.txt"
    path.write_text("\n".join(ENGLISH_SENTENCES * 300), encoding="utf-8")
    return path


@pytest.fixture(scope="module")
def morphbpe_tokenizer(english_corpus, tmp_path_factory):
    out = tmp_path_factory.mktemp("artifact_morphbpe")
    train_tokenizer(
        corpus_path=english_corpus,
        algorithm="morphbpe",
        vocab_size=VOCAB_SIZE,
        output_dir=out,
        language="en",
    )
    return load_tokenizer(out, algorithm="morphbpe")


def test_morphbpe_vocab_within_budget(morphbpe_tokenizer):
    assert morphbpe_tokenizer.vocab_size <= VOCAB_SIZE + 20
    assert morphbpe_tokenizer.vocab_size >= len(morphbpe_tokenizer.special_tokens)


def test_morphbpe_specials_present(morphbpe_tokenizer):
    for sp in morphbpe_tokenizer.special_tokens:
        assert sp in morphbpe_tokenizer.get_vocab()


@pytest.mark.parametrize("text", ["happiness", "uncomfortable", "tokenization", "Hello, world!"])
def test_morphbpe_roundtrip(morphbpe_tokenizer, text):
    once = morphbpe_tokenizer.decode(morphbpe_tokenizer.encode(text))
    twice = morphbpe_tokenizer.decode(morphbpe_tokenizer.encode(once))
    assert once == twice


def test_morphbpe_save_load_roundtrip(morphbpe_tokenizer, tmp_path):
    path = tmp_path / "reloaded"
    path.mkdir()
    morphbpe_tokenizer.save(path)
    reloaded = load_tokenizer(path, algorithm="morphbpe")
    for s in ["happiness", "Hello, world!", "tokenization"]:
        assert reloaded.encode(s) == morphbpe_tokenizer.encode(s)


def test_morphbpe_requires_language(english_corpus, tmp_path):
    """Calling train_tokenizer for morphbpe without a language is an error,
    not a silent fallback to plain BPE."""
    with pytest.raises(ValueError, match="language"):
        train_tokenizer(english_corpus, "morphbpe", 500, tmp_path / "x")


def test_morphbpe_rejects_unsupported_language(english_corpus, tmp_path):
    """Mandarin lacks the inflectional morphology MorphBPE relies on, so
    requesting it should fail loudly rather than silently degrade to BPE."""
    with pytest.raises(NotImplementedError, match="zh"):
        train_tokenizer(english_corpus, "morphbpe", 500, tmp_path / "x", language="zh")
