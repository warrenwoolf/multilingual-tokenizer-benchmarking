"""SuperBPE-specific tests verifying the two-stage merge hierarchy.

These tests check the algorithmic invariants that distinguish SuperBPE from
plain BPE:

  1. The first ~90 % of merges are word-boundary-only: no Ġ past position 0
     in any merged token (Ġ = U+0120, the ByteLevel space encoding).
  2. The last ~10 % of merges include at least some cross-word tokens:
     merged tokens with Ġ appearing past position 0.
  3. Basic contract: encode/decode round-trip, vocab size, special tokens.

The Ġ (U+0120) character is the ByteLevel encoding for ASCII space (0x20).
A token with Ġ past position 0 spans what would be a word boundary in
stage-1, making it a "cross-word" token.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.utils.tokenizer_algorithms import load_tokenizer, train_tokenizer

# Ġ = U+0120 = ByteLevel encoding of ASCII space
_SPACE = "Ġ"

VOCAB_SIZE = 1_000
STAGE1_SIZE = int(VOCAB_SIZE * 0.9)   # 900
STAGE2_BUDGET = VOCAB_SIZE - STAGE1_SIZE  # 100

# Corpus with many repeated cross-word bigrams so stage-2 has clear signals.
# "in the", "of the", "to the", "natural language", etc. repeat across sentences.
_SENTENCES = [
    "in the beginning was the word and the word was with god",
    "the quick brown fox jumps over the lazy dog near the river",
    "to be or not to be that is the question of the day",
    "all the world is a stage and all the men and women merely players",
    "in the middle of the night in the darkness of the forest",
    "the end of the world is not the end of time",
    "of the people by the people for the people of this land",
    "in the name of the father and of the son in the beginning",
    "the best of times the worst of times it was the age of wisdom",
    "in the heat of the moment all the things we say to one another",
    "natural language processing requires tokenizers that work across languages",
    "machine learning models learn from the data that we provide to them",
]


@pytest.fixture(scope="module")
def superbpe_corpus(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("superbpe_corpus") / "train.txt"
    # 800 repeats → ~9 600 lines; common bigrams get frequency ~800
    path.write_text("\n".join(_SENTENCES * 800), encoding="utf-8")
    return path


@pytest.fixture(scope="module")
def superbpe_artifact(superbpe_corpus, tmp_path_factory) -> tuple:
    """Returns (tokenizer_adapter, artifact_dir) trained once per module."""
    out = tmp_path_factory.mktemp("artifact_superbpe")
    train_tokenizer(
        corpus_path=superbpe_corpus,
        algorithm="superbpe",
        vocab_size=VOCAB_SIZE,
        output_dir=out,
    )
    tok = load_tokenizer(out, algorithm="superbpe")
    return tok, out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_merges(artifact_dir: Path) -> list[tuple[str, str]]:
    data = json.loads((artifact_dir / "tokenizer.json").read_text(encoding="utf-8"))
    return [tuple(m) for m in data["model"]["merges"]]


def _load_config(artifact_dir: Path) -> dict:
    return json.loads((artifact_dir / "config.json").read_text(encoding="utf-8"))


def _is_crossword(a: str, b: str) -> bool:
    """True if merging a and b produces a token with Ġ past position 0."""
    merged = a + b
    return _SPACE in merged[1:]


# ---------------------------------------------------------------------------
# Basic contract
# ---------------------------------------------------------------------------


def test_superbpe_trains(superbpe_artifact):
    tok, _ = superbpe_artifact
    # Byte-level tokenizer must cover at least 256 byte tokens.
    assert tok.vocab_size >= 256
    assert tok.is_byte_level


def test_superbpe_vocab_within_budget(superbpe_artifact):
    tok, _ = superbpe_artifact
    # Allow a small overshoot; vocab cannot wildly exceed the budget.
    assert tok.vocab_size <= VOCAB_SIZE + 20


def test_superbpe_special_tokens_present(superbpe_artifact):
    tok, _ = superbpe_artifact
    for sp in tok.special_tokens:
        assert sp in tok.get_vocab(), f"{sp!r} missing from superbpe vocab"


def test_superbpe_encode_nonempty(superbpe_artifact):
    tok, _ = superbpe_artifact
    for text in ["Hello world", "the quick brown fox", "in the beginning"]:
        assert len(tok.encode(text)) > 0


def test_superbpe_empty_input(superbpe_artifact):
    tok, _ = superbpe_artifact
    assert tok.encode("") == []


def test_superbpe_roundtrip_idempotent(superbpe_artifact):
    tok, _ = superbpe_artifact
    for text in ["Hello world", "the quick brown fox", "in the beginning"]:
        once = tok.decode(tok.encode(text))
        twice = tok.decode(tok.encode(once))
        assert once == twice, f"Roundtrip not idempotent for {text!r}"


def test_superbpe_ids_within_vocab(superbpe_artifact):
    tok, _ = superbpe_artifact
    for text in ["Hello world", "in the end", "natural language"]:
        ids = tok.encode(text)
        assert all(0 <= i < tok.vocab_size for i in ids)


def test_superbpe_save_load_roundtrip(superbpe_artifact, tmp_path):
    tok, _ = superbpe_artifact
    save_path = tmp_path / "reloaded"
    save_path.mkdir()
    tok.save(save_path)
    reloaded = load_tokenizer(save_path, algorithm="superbpe")
    for text in ["Hello world", "the quick brown fox"]:
        assert reloaded.encode(text) == tok.encode(text)


# ---------------------------------------------------------------------------
# Two-stage merge hierarchy invariants
# ---------------------------------------------------------------------------


def test_stage1_merges_are_word_boundary_only(superbpe_artifact):
    """Stage-1 merges must not produce cross-word tokens.

    Stage 1 trains with a GPT-2-style regex that splits on word boundaries,
    so no BPE merge in stage 1 can ever join tokens from different segments
    (i.e. no Ġ may appear past position 0 in any stage-1 merged token).

    The exact stage boundary is read from config.json (stage1_merge_count).
    """
    _, out = superbpe_artifact
    merges = _load_merges(out)
    cfg = _load_config(out)
    s1_count = cfg["stage1_merge_count"]
    stage1_merges = merges[:s1_count]
    crossword = [(a, b) for a, b in stage1_merges if _is_crossword(a, b)]
    assert crossword == [], (
        f"Found {len(crossword)} cross-word merge(s) in {s1_count} "
        f"stage-1 merges (expected 0). Examples: {crossword[:3]}"
    )


def test_stage2_contains_crossword_merges(superbpe_artifact):
    """Stage-2 merges must contain at least one cross-word token.

    Stage 2 uses a permissive pre-tokenizer that does not split on spaces
    before letters/digits, so BPE can learn merges spanning word boundaries.
    At least one such merge must appear in the stage-2 portion of the list.
    """
    _, out = superbpe_artifact
    merges = _load_merges(out)
    cfg = _load_config(out)
    s1_count = cfg["stage1_merge_count"]
    stage2_merges = merges[s1_count:]
    assert len(stage2_merges) > 0, "No stage-2 merges found in tokenizer"
    crossword = [(a, b) for a, b in stage2_merges if _is_crossword(a, b)]
    assert len(crossword) > 0, (
        f"No cross-word merges found in {len(stage2_merges)} stage-2 merges. "
        f"Stage-2 sample: {stage2_merges[:5]}"
    )


def test_merge_list_is_ordered_stage1_then_stage2(superbpe_artifact):
    """Stage-1 merges (index < stage1_merge_count) precede stage-2 merges.

    Verify that the stored stage1_merge_count is accurate: all merges in
    the stage-1 slice are word-boundary-only, and the first cross-word merge
    (if any) appears at or after index stage1_merge_count.
    """
    _, out = superbpe_artifact
    merges = _load_merges(out)
    cfg = _load_config(out)
    s1_count = cfg["stage1_merge_count"]

    # Find the first cross-word merge in the entire list.
    first_crossword_idx = next(
        (i for i, (a, b) in enumerate(merges) if _is_crossword(a, b)),
        None,
    )
    if first_crossword_idx is None:
        pytest.skip("No cross-word merges present — cannot verify ordering.")

    assert first_crossword_idx >= s1_count, (
        f"First cross-word merge is at index {first_crossword_idx}, "
        f"which is before the stage boundary at {s1_count}. "
        f"Merge: {merges[first_crossword_idx]}"
    )


def test_stage2_tokens_reference_existing_vocab(superbpe_artifact):
    """Both tokens in each stage-2 merge exist in the final vocabulary.

    Stage-2 extension builds merges from token sequences encoded by stage-1
    BPE, so every token referenced by a stage-2 merge must be known to the
    tokenizer (either a stage-1 token or an earlier stage-2 merged token).
    """
    tok, out = superbpe_artifact
    merges = _load_merges(out)
    cfg = _load_config(out)
    s1_count = cfg["stage1_merge_count"]
    vocab = tok.get_vocab()
    stage2_merges = merges[s1_count:]

    missing = [
        (a, b)
        for a, b in stage2_merges
        if a not in vocab or b not in vocab
    ]
    assert missing == [], (
        f"Stage-2 merge tokens not in final vocab: {missing[:5]}"
    )
