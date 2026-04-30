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
def morphynet_cache_dir(tmp_path_factory) -> Path:
    """Minimal MorphyNet-format TSVs for English and Hungarian.

    Written to a tmp dir so tests never hit the network. Format mirrors the
    real files: lemma TAB inflected TAB features TAB pipe-separated morphemes.
    """
    cache = tmp_path_factory.mktemp("morphynet")

    en_rows = [
        "run\trunning\tV|V.PTCP;PRS\trun|n|ing",
        "run\trunner\tN\trun|n|er",
        "run\trunners\tN|PL\trun|n|er|s",
        "run\truns\tV|PRS;3;SG\trun|s",
        "happy\thappiness\tN\thappi|ness",
        "happy\thappily\tR\thappi|ly",
        "happy\thappier\tJ\thappi|er",
        "happy\tunhappiness\tN\tun|happi|ness",
        "sad\tsadness\tN\tsad|ness",
        "token\ttokenization\tN\ttoken|iz|ation",
        "token\ttokenizer\tN\ttoken|iz|er",
        "token\ttokenizers\tN|PL\ttoken|iz|er|s",
        "token\ttokenized\tV|PST\ttoken|iz|ed",
        "token\ttokenizes\tV|PRS;3;SG\ttoken|iz|es",
        "tokenize\ttokenize\tV\ttoken|ize",
        "process\tpreprocessing\tV|V.PTCP;PRS\tpre|process|ing",
        "process\tpostprocessing\tV|V.PTCP;PRS\tpost|process|ing",
        "process\treprocessing\tV|V.PTCP;PRS\tre|process|ing",
        "process\tprocessing\tV|V.PTCP;PRS\tprocess|ing",
        "comfort\tdiscomfort\tN\tdis|comfort",
        "comfort\tcomfortable\tJ\tcomfort|able",
        "comfort\tuncomfortable\tJ\tun|comfort|able",
        "bear\tunbearable\tJ\tun|bear|able",
        "whelm\toverwhelming\tV|V.PTCP;PRS\tover|whelm|ing",
        "quick\tquickly\tR\tquick|ly",
        "sell\tsells\tV|PRS;3;SG\tsell|s",
        "learn\tlearning\tV|V.PTCP;PRS\tlearn|ing",
        "train\ttraining\tV|V.PTCP;PRS\ttrain|ing",
        "represent\trepresentations\tN|PL\tre|present|ation|s",
        "require\trequires\tV|PRS;3;SG\trequire|s",
        "compare\tcompares\tV|PRS;3;SG\tcompare|s",
        "reveal\treveals\tV|PRS;3;SG\treveal|s",
        "control\tcontrols\tV|PRS;3;SG\tcontrol|s",
        "split\tsplits\tV|PRS;3;SG\tsplit|s",
        "jump\tjumps\tV|PRS;3;SG\tjump|s",
    ]
    (cache / "eng.inflectional.v1.tsv").write_text("\n".join(en_rows), encoding="utf-8")

    # Stub Hungarian file — a handful of real inflectional forms so the
    # MorphyNet loader path is exercised without a live download.
    hu_rows = [
        "ház\tházak\tN|PL\tház|ak",
        "ház\tházban\tN|IN+ESS\tház|ban",
        "ház\tháztól\tN|ELA\tház|tól",
        "ember\tembernek\tN|DAT\tember|nek",
        "ember\temberek\tN|PL\tember|ek",
        "tanul\ttanulnak\tV|PRS;3;PL\ttanul|nak",
        "tanul\ttanulás\tN\ttanul|ás",
        "szép\tszépen\tR\tszép|en",
    ]
    (cache / "hu.inflectional.segmentation.v1.tsv").write_text(
        "\n".join(hu_rows), encoding="utf-8"
    )

    return cache


@pytest.fixture(scope="module")
def morphbpe_tokenizer(english_corpus, morphynet_cache_dir, tmp_path_factory):
    out = tmp_path_factory.mktemp("artifact_morphbpe")
    train_tokenizer(
        corpus_path=english_corpus,
        algorithm="morphbpe",
        vocab_size=VOCAB_SIZE,
        output_dir=out,
        language="en",
        morphynet_cache_dir=morphynet_cache_dir,
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


# ---------- BPE decode ------------------------------------------
# BPE uses a ByteLevel pre-tokenizer (Ġ prefix for word-initial subwords) and
# a matching ByteLevel decoder, so inter-word spaces are correctly preserved
# on decode.  This test pins that behaviour so we notice if it regresses.


def test_bpe_decode_drops_interword_spaces(tiny_corpus, tmp_path):
    """BPE decode preserves inter-word spaces via ByteLevel pre-tokenizer."""
    out = tmp_path / "bpe_decode_test"
    out.mkdir()
    train_tokenizer(tiny_corpus, "bpe", vocab_size=VOCAB_SIZE, output_dir=out)
    tok = load_tokenizer(out, algorithm="bpe")

    text = "Hello world"
    decoded = tok.decode(tok.encode(text))
    assert " " in decoded, (
        "BPE decode should preserve inter-word spaces via ByteLevel pre-tokenizer."
    )


# ---------- MorphBPE decode behaviour --------------------------------------
# MorphBPEAdapter segments each word into morphemes at encode time, so the
# inference distribution matches training.  decode() returns the morpheme-
# split text (spaces at morpheme boundaries) — faithful round-trip to the
# original word is intentionally not preserved, because BPB measurement uses
# the original text byte count, not the decoded output.


def test_morphbpe_single_word_decode_shows_morpheme_splits(morphbpe_tokenizer):
    """decode(encode(word)) returns the morpheme-segmented form, not the original word."""
    cases = {
        "running": "run n ing",
        "happiness": "happi ness",
        "tokenization": "token iz ation",
        "uncomfortable": "un comfort able",
    }
    for word, expected in cases.items():
        decoded = morphbpe_tokenizer.decode(morphbpe_tokenizer.encode(word))
        assert decoded == expected, (
            f"MorphBPE decode should produce morpheme-split text: "
            f"{word!r} -> {decoded!r}, expected {expected!r}"
        )


def test_morphbpe_multiword_decode_preserves_word_boundaries(morphbpe_tokenizer):
    """Words not in the MorphyNet lookup pass through unsplit; spaces are preserved."""
    text = "playing tennis"
    decoded = morphbpe_tokenizer.decode(morphbpe_tokenizer.encode(text))
    assert " " in decoded, (
        "MorphBPE decode should preserve spaces between words."
    )


# ---------- MorphBPE morpheme-boundary constraint --------------------------
# Verifies that BPE does not create tokens that cross morpheme boundaries.
#
# The production path uses MorphyNet gold segmentations for both English and
# Hungarian, so the invariant test uses the English corpus + morphynet_cache_dir
# fixture (no network access required).
#
# The agglutinative corpus and Morfessor tests below are kept as unit tests
# for the internal Morfessor utilities (_train_morfessor, _segment_word),
# which are still present in the module for research use.


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


def test_morphbpe_segment_corpus_splits_at_morpheme_boundaries(
    morphynet_cache_dir, tmp_path_factory
):
    """The preprocessing approach inserts spaces at morpheme boundaries.

    BPE with a ByteLevel pre-tokenizer splits on whitespace, so it cannot
    count or learn merges across spaces.  Replacing 'running' with 'run n ing'
    in the training corpus guarantees no cross-morpheme merge is ever learned.

    MorphBPEAdapter enforces the same constraint at inference time by
    segmenting each word before encoding, so the inference distribution
    matches training and cross-morpheme tokens cannot be produced.
    """
    from src.utils.morpheme_segmentation import segment_corpus

    corpus = tmp_path_factory.mktemp("seg_check") / "corpus.txt"
    corpus.write_text(
        "running happiness tokenization comfortable\n", encoding="utf-8"
    )
    output = tmp_path_factory.mktemp("seg_out") / "out.txt"
    segment_corpus(
        corpus, output, language="en", morphynet_cache_dir=morphynet_cache_dir
    )

    result = output.read_text(encoding="utf-8").strip()
    assert "run n ing" in result        # running   → run|n|ing
    assert "happi ness" in result       # happiness → happi|ness
    assert "token iz ation" in result   # tokenization → token|iz|ation
    assert "comfort able" in result     # comfortable  → comfort|able
    # Original unsplit forms must not appear
    for w in ("running", "happiness", "tokenization", "comfortable"):
        assert w not in result, f"Expected {w!r} to be split but found it whole"
