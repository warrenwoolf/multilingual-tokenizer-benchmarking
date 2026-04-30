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
        # No declared <unk>: tokenizer must still produce output (byte-level fallback).
        assert len(ids) > 0
    else:
        # Tokenizer has <unk>: rare codepoints that can't be encoded any other
        # way must actually appear as UNK tokens, not silently vanish.
        assert unk_id in ids, (
            f"rare tag-space codepoints should produce <unk> (id={unk_id}), got ids={ids}"
        )


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


# ---------- BPE decode limitation ------------------------------------------
# BPE is trained without end_of_word_suffix (adding it introduces non-
# determinism in HF tokenizers' tie-breaking, breaking the training
# determinism contract).  As a result, BPEDecoder cannot locate word
# boundaries and concatenates all subwords without spaces.  This does not
# affect the BPB or fertility metrics (both use encode-only), but it means
# decode() is lossy for multi-word inputs.  This test documents that known
# behaviour so we notice if it ever accidentally changes.


def test_bpe_decode_drops_interword_spaces(tiny_corpus, tmp_path):
    """Documents the known decode limitation: inter-word spaces are dropped."""
    out = tmp_path / "bpe_decode_test"
    out.mkdir()
    train_tokenizer(tiny_corpus, "bpe", vocab_size=VOCAB_SIZE, output_dir=out)
    tok = load_tokenizer(out, algorithm="bpe")

    text = "Hello world"
    decoded = tok.decode(tok.encode(text))
    # Spaces are dropped; the decode is concatenative.
    assert " " not in decoded, (
        "BPE decode is unexpectedly preserving spaces — the end_of_word_suffix "
        "limitation may have been fixed; update this test and the docstring in "
        "_train_bpe if so."
    )


# ---------- MorphBPE decode behaviour --------------------------------------
# Documents (and pins) the known limitation: because the segmented training
# corpus does not distinguish morpheme-boundary spaces from word-boundary
# spaces, the trained tokenizer cannot reconstruct inter-word spaces on
# decode().  Single-word inputs round-trip correctly; multi-word inputs lose
# the space.  The BPB / fertility pipelines only call encode(), so this does
# not affect benchmark results.


def test_morphbpe_single_word_decode_is_lossless(morphbpe_tokenizer):
    """Single-word inputs: decode(encode(word)) == word (morphemes concatenate)."""
    for word in ["happiness", "tokenization", "uncomfortable", "running"]:
        decoded = morphbpe_tokenizer.decode(morphbpe_tokenizer.encode(word))
        assert decoded == word, (
            f"MorphBPE single-word decode should be lossless: {word!r} -> {decoded!r}"
        )


def test_morphbpe_multiword_decode_drops_spaces(morphbpe_tokenizer):
    """Multi-word decode is known to lose inter-word spaces (documented limitation)."""
    text = "playing tennis"
    decoded = morphbpe_tokenizer.decode(morphbpe_tokenizer.encode(text))
    # The limitation: spaces are dropped.
    assert " " not in decoded, (
        "If this assertion fails, MorphBPE decode now preserves spaces — "
        "update the limitation note in _train_morphbpe and this test."
    )


# ---------- MorphBPE morpheme-boundary constraint --------------------------
# Verifies that, when Morfessor *does* produce splits, BPE does not create
# tokens that cross morpheme boundaries.
#
# Morfessor's MDL objective only splits words when the vocabulary is large
# and morphological patterns repeat across many word types — typical of
# agglutinative languages.  We generate a synthetic Finnish-style corpus
# (regular root+suffix paradigms) that reliably triggers segmentation.
# The supported language nearest to Finnish in the codebase is Turkish ("tr"),
# which shares the same agglutinative typology.


def _make_agglutinative_corpus(path: Path) -> None:
    """Write a corpus rich enough for Morfessor to discover morpheme splits.

    Morfessor's MDL objective only segments words when re-using morpheme pieces
    across many word types reduces the total encoding cost.  We simulate a
    Finnish-style agglutinative paradigm: many roots × many case/number/tense
    suffixes (including vowel-harmony variants), resulting in 500+ word types
    with clearly shared sub-strings.

    Vowel-harmony pairs (e.g. "lle"/"llä", "lta"/"ltä") are essential:
    Morfessor finds the shared root by observing that "auto", "talo", etc.
    each appear with both variants.  Without this variety the MDL cost of
    splitting exceeds its benefit.
    """
    roots = [
        "auto", "talo", "kirja", "koira", "kissa", "mies", "nainen", "lapsi",
        "maa", "puu", "katu", "koulu", "kauppa", "pankki", "ravintola",
        "museo", "teatteri", "kirjasto",
    ]
    # Singular case suffixes (back-vowel / front-vowel harmony pairs)
    case_suffixes = [
        "",                                   # nominative
        "n",                                  # genitive
        "a", "ä",                             # partitive
        "lle", "llä",                         # allative
        "lta", "ltä",                         # ablative
        "lla", "llä",                         # adessive
        "sta", "stä",                         # elative
        "han", "hen", "hin", "hon", "hun",    # illative variants
        "ksi",                                # translative
        "ssa", "ssä",                         # inessive
        "ja", "jä",                           # comitative
    ]
    # Plural case suffixes (with "i" infix)
    plural_suffixes = [
        "t",           # nominative plural
        "jen",         # genitive plural
        "ja", "jä",    # partitive plural
        "ille",        # allative plural
        "ilta", "iltä",# ablative plural
        "illa", "illä",# adessive plural
        "ista", "istä",# elative plural
    ]
    import random as _random
    rng = _random.Random(0)
    lines = []
    for root in roots:
        for suffix in case_suffixes:
            count = rng.randint(100, 3000)
            lines.extend([root + suffix] * count)
        for suffix in plural_suffixes:
            count = rng.randint(50, 500)
            lines.extend([root + "i" + suffix] * count)
    rng.shuffle(lines)
    path.write_text("\n".join(lines), encoding="utf-8")


@pytest.fixture(scope="module")
def agglutinative_corpus(tmp_path_factory) -> Path:
    p = tmp_path_factory.mktemp("aggl_corpus") / "train.txt"
    _make_agglutinative_corpus(p)
    return p


def test_morfessor_segments_agglutinative_corpus(agglutinative_corpus):
    """Sanity-check: Morfessor must learn to split morphemes on this corpus.

    Morfessor may keep *training* words unitary (it stores an explicit analysis
    per seen compound), but it will split *unseen* derived forms by applying
    the morpheme inventory it learned.  We therefore probe a small set of
    held-out words (root + plural-infix + case-suffix combinations that were
    NOT emitted by the corpus generator) to verify the model learned a useful
    morpheme decomposition.
    """
    from src.utils.morpheme_segmentation import (
        _collect_word_counts,
        _train_morfessor,
    )
    counts = _collect_word_counts(agglutinative_corpus)
    model = _train_morfessor(counts, max_types=200_000)

    # Held-out forms: root + "i" + adessive-suffix "lla" (not emitted by the
    # corpus generator whose plural suffixes end in -lle/-ilta etc., not -illa).
    held_out = [
        "autoilla",    # auto + i + lla
        "taloilla",    # talo + i + lla
        "kouluilla",   # koulu + i + lla
        "kirjoilla",   # kirja → kirjo + i + lla (stem change)
        "kissoilla",   # kissa → kissoi + lla
    ]
    split_found = any(
        len(model.viterbi_segment(w)[0]) > 1
        for w in held_out
        if w not in counts   # must be truly unseen
    )
    assert split_found, (
        "Morfessor produced zero splits for held-out Finnish plural forms; "
        "the model may not have learned root/suffix decomposition.  "
        "MorphBPE would be identical to BPE for this corpus."
    )


def test_morphbpe_no_cross_morpheme_tokens(agglutinative_corpus, tmp_path_factory):
    """Core MorphBPE invariant: no token in the vocabulary may span a morpheme boundary.

    We train MorphBPE on the agglutinative corpus (where Morfessor splits reliably),
    recover the Morfessor segmentation for every word type, and assert that every
    learned BPE merge exists *within* a morpheme — never crossing one.
    """
    from src.utils.morpheme_segmentation import (
        _collect_word_counts,
        _train_morfessor,
        _segment_word,
    )

    out = tmp_path_factory.mktemp("morphbpe_aggl")
    train_tokenizer(
        corpus_path=agglutinative_corpus,
        algorithm="morphbpe",
        vocab_size=500,
        output_dir=out,
        language="tr",
    )
    tok = load_tokenizer(out, algorithm="morphbpe")

    counts = _collect_word_counts(agglutinative_corpus)
    morph_model = _train_morfessor(counts, max_types=200_000)
    cache: dict = {}

    cross_boundary_tokens: list[str] = []
    for word in counts:
        morphemes = _segment_word(morph_model, word, cache)
        if len(morphemes) <= 1:
            continue  # unsplit word — no boundary to check
        ids = tok.encode(word)
        tokens = [tok.tokenizer.id_to_token(i) for i in ids]
        # Reconstruct the character positions of each morpheme boundary.
        pos = 0
        boundaries: set[int] = set()
        for m in morphemes[:-1]:
            pos += len(m)
            boundaries.add(pos)
        # Check whether any token straddles a boundary.
        char_pos = 0
        for token in tokens:
            raw = token.replace("</w>", "")  # strip end-of-word marker if present
            token_start = char_pos
            token_end = char_pos + len(raw)
            for b in boundaries:
                if token_start < b < token_end:
                    cross_boundary_tokens.append(
                        f"{word!r}: token {token!r} crosses boundary at pos {b} "
                        f"(morphemes={morphemes})"
                    )
            char_pos = token_end

    assert not cross_boundary_tokens, (
        "MorphBPE produced tokens that cross morpheme boundaries:\n"
        + "\n".join(cross_boundary_tokens[:10])
    )
