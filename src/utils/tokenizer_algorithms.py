"""Tokenizer training and loading for all supported algorithms.

Supported algorithms
--------------------
bpe        – standard Byte-Pair Encoding (Sennrich et al. 2016) via HF tokenizers
superbpe   – SuperBPE two-stage byte-level BPE (Nut et al. 2025): stage 1 with a
             GPT-2-style word-boundary regex, stage 2 with a permissive regex that
             allows merges across spaces (cross-word tokens)
tiktoken   – tiktoken/GPT-style byte-level BPE (ByteLevel pre-tokenizer + BPE)
morphbpe   – MorphBPE (Asgari et al. 2025): morpheme-segment the corpus first,
             then train standard BPE — see src/utils/morpheme_segmentation.py
wordpiece  – BERT-style WordPiece via HF tokenizers
unigram    – Unigram LM / SentencePiece-style via HF tokenizers
byt5       – Pure byte-level baseline via transformers.ByT5Tokenizer (no training)

Each algorithm's train function writes an artifact to output_dir, and the
corresponding load function returns an adapter with a common interface:

    encode(text: str) -> list[int]
    decode(ids: list[int]) -> str
    encode_batch(texts: list[str]) -> list[list[int]]
    get_vocab() -> dict[str, int]
    vocab_size -> int
    token_to_id(token: str) -> int | None
    save(path: Path) -> None
    special_tokens: list[str]
    is_byte_level: bool
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SUPPORTED_ALGORITHMS = (
    "bpe",
    "superbpe",
    "tiktoken",
    "morphbpe",
    "wordpiece",
    "unigram",
    "byt5",
)

DEFAULT_SPECIAL_TOKENS = ["<pad>", "<unk>", "<s>", "</s>"]

_TOKENIZER_FILENAME = "tokenizer.json"
_CONFIG_FILENAME = "config.json"


# ---------------------------------------------------------------------------
# Adapter classes
# ---------------------------------------------------------------------------


@dataclass
class HFAdapter:
    """Wraps a HuggingFace tokenizers.Tokenizer with the common interface."""

    tokenizer: Any  # tokenizers.Tokenizer
    algorithm: str
    special_tokens: list[str] = field(default_factory=lambda: list(DEFAULT_SPECIAL_TOKENS))
    is_byte_level: bool = False

    def encode(self, text: str) -> list[int]:
        if text == "":
            return []
        return list(self.tokenizer.encode(text).ids)

    def decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(list(ids), skip_special_tokens=False)

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        return [self.encode(t) for t in texts]

    def get_vocab(self) -> dict[str, int]:
        return self.tokenizer.get_vocab()

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()

    def token_to_id(self, token: str) -> int | None:
        return self.tokenizer.token_to_id(token)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.tokenizer.save(str(path / _TOKENIZER_FILENAME))
        (path / _CONFIG_FILENAME).write_text(
            json.dumps(
                {
                    "algorithm": self.algorithm,
                    "special_tokens": self.special_tokens,
                    "is_byte_level": self.is_byte_level,
                }
            ),
            encoding="utf-8",
        )

    def __getstate__(self) -> dict:
        return {
            "tokenizer_json": self.tokenizer.to_str(),
            "algorithm": self.algorithm,
            "special_tokens": self.special_tokens,
            "is_byte_level": self.is_byte_level,
        }

    def __setstate__(self, state: dict) -> None:
        from tokenizers import Tokenizer

        self.tokenizer = Tokenizer.from_str(state["tokenizer_json"])
        self.algorithm = state["algorithm"]
        self.special_tokens = state["special_tokens"]
        self.is_byte_level = state["is_byte_level"]


@dataclass
class ByT5Adapter:
    """Byte-level baseline: each UTF-8 byte is a token (+ special tokens)."""

    algorithm: str = "byt5"
    is_byte_level: bool = True
    special_tokens: list[str] = field(default_factory=lambda: ["<pad>", "</s>", "<unk>"])

    def __post_init__(self) -> None:
        from transformers import ByT5Tokenizer

        self._tok = ByT5Tokenizer()

    def encode(self, text: str) -> list[int]:
        if text == "":
            return []
        # add_special_tokens=False keeps the output as pure byte ids.
        return self._tok.encode(text, add_special_tokens=False)

    def decode(self, ids: list[int]) -> str:
        return self._tok.decode(list(ids), skip_special_tokens=False)

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        return [self.encode(t) for t in texts]

    def get_vocab(self) -> dict[str, int]:
        return self._tok.get_vocab()

    @property
    def vocab_size(self) -> int:
        return self._tok.vocab_size

    def token_to_id(self, token: str) -> int | None:
        tid = self._tok.convert_tokens_to_ids(token)
        # transformers returns the unk id for unknown tokens; we want None instead.
        if tid == self._tok.unk_token_id and token != self._tok.unk_token:
            return None
        return tid

    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        (path / _CONFIG_FILENAME).write_text(
            json.dumps({"algorithm": self.algorithm, "is_byte_level": True}),
            encoding="utf-8",
        )

    def __getstate__(self) -> dict:
        return {"algorithm": self.algorithm}

    def __setstate__(self, state: dict) -> None:
        self.algorithm = state["algorithm"]
        self.is_byte_level = True
        self.special_tokens = ["<pad>", "</s>", "<unk>"]
        from transformers import ByT5Tokenizer

        self._tok = ByT5Tokenizer()


# ---------------------------------------------------------------------------
# Training dispatch
# ---------------------------------------------------------------------------


def train_tokenizer(
    corpus_path: Path,
    algorithm: str,
    vocab_size: int,
    output_dir: Path,
    language: str | None = None,
) -> None:
    """Train a tokenizer and persist it to output_dir.

    ``language`` is required for algorithms that need a per-language asset
    (currently only MorphBPE, which needs a morpheme segmenter); other
    algorithms ignore it.
    """
    if algorithm not in SUPPORTED_ALGORITHMS:
        raise ValueError(f"Unknown algorithm {algorithm!r}. Choose from {SUPPORTED_ALGORITHMS}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if algorithm == "morphbpe":
        if language is None:
            raise ValueError(
                "MorphBPE training requires a `language` argument so the "
                "right morpheme segmenter can be selected."
            )
        _train_morphbpe(Path(corpus_path), vocab_size, output_dir, language)
        return
    dispatch = {
        "bpe": _train_bpe,
        "superbpe": _train_superbpe,
        "tiktoken": _train_tiktoken,
        "wordpiece": _train_wordpiece,
        "unigram": _train_unigram,
        "byt5": _train_byt5,
    }
    dispatch[algorithm](Path(corpus_path), vocab_size, output_dir)


def _train_bpe(corpus_path: Path, vocab_size: int, output_dir: Path) -> None:
    from tokenizers import Tokenizer, decoders
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.trainers import BpeTrainer

    tok = Tokenizer(BPE(unk_token="<unk>"))
    tok.pre_tokenizer = Whitespace()
    tok.decoder = decoders.BPEDecoder()
    trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=DEFAULT_SPECIAL_TOKENS)
    tok.train(files=[str(corpus_path)], trainer=trainer)
    HFAdapter(tok, algorithm="bpe").save(output_dir)


def _train_wordpiece(corpus_path: Path, vocab_size: int, output_dir: Path) -> None:
    from tokenizers import Tokenizer, decoders
    from tokenizers.models import WordPiece
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.trainers import WordPieceTrainer

    tok = Tokenizer(WordPiece(unk_token="<unk>"))
    tok.pre_tokenizer = Whitespace()
    tok.decoder = decoders.WordPiece(prefix="##")
    trainer = WordPieceTrainer(vocab_size=vocab_size, special_tokens=DEFAULT_SPECIAL_TOKENS)
    tok.train(files=[str(corpus_path)], trainer=trainer)
    HFAdapter(tok, algorithm="wordpiece").save(output_dir)


def _train_unigram(corpus_path: Path, vocab_size: int, output_dir: Path) -> None:
    from tokenizers import Tokenizer
    from tokenizers.models import Unigram
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.trainers import UnigramTrainer

    tok = Tokenizer(Unigram())
    tok.pre_tokenizer = Whitespace()
    trainer = UnigramTrainer(
        vocab_size=vocab_size,
        special_tokens=DEFAULT_SPECIAL_TOKENS,
        unk_token="<unk>",
    )
    tok.train(files=[str(corpus_path)], trainer=trainer)
    HFAdapter(tok, algorithm="unigram").save(output_dir)


def _train_byt5(corpus_path: Path, vocab_size: int, output_dir: Path) -> None:
    # No training — ByT5 is stateless. Persist a config marker so load works.
    ByT5Adapter().save(output_dir)


def _train_tiktoken(corpus_path: Path, vocab_size: int, output_dir: Path) -> None:
    """Train tiktoken-style byte-level BPE (GPT-2 / GPT-4 style).

    `tiktoken` itself doesn't support production training, so we train
    byte-level BPE via HuggingFace tokenizers (ByteLevel pre-tokenizer + BPE
    model + ByteLevel decoder), which is the algorithm tiktoken implements.
    The resulting tokenizer.json is a drop-in for HF and the merges can be
    re-exported into tiktoken's mergeable_ranks format if needed.
    """
    from tokenizers import Tokenizer, decoders
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import ByteLevel as ByteLevelPre
    from tokenizers.processors import ByteLevel as ByteLevelPost
    from tokenizers.trainers import BpeTrainer

    tok = Tokenizer(BPE())
    tok.pre_tokenizer = ByteLevelPre(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.post_processor = ByteLevelPost(trim_offsets=True)
    # Byte-level BPE doesn't need <unk> (any byte sequence is encodable),
    # but we still register the GPT-2-style end-of-text marker.
    specials = ["<|endoftext|>", "<pad>", "<s>", "</s>"]
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=specials,
        initial_alphabet=ByteLevelPre.alphabet(),
    )
    tok.train(files=[str(corpus_path)], trainer=trainer)
    HFAdapter(tok, algorithm="tiktoken", special_tokens=specials, is_byte_level=True).save(output_dir)


def _train_morphbpe(
    corpus_path: Path, vocab_size: int, output_dir: Path, language: str
) -> None:
    """MorphBPE (Asgari et al. 2025).

    The paper's Algorithm 1 is: initialize the vocabulary with characters,
    morpheme-segment the corpus, then run BPE while skipping any candidate
    pair that would cross a morpheme boundary. We implement that constraint
    by morpheme-segmenting the corpus first (each word becomes its
    whitespace-joined morphemes) and then training standard HF BPE on the
    rewritten corpus. The Whitespace pre-tokenizer guarantees BPE never
    counts or merges a pair across a morpheme boundary, so it's equivalent.

    See src/utils/morpheme_segmentation.py for the per-language segmenter.
    The official llm-lab-org/MorphBPE repo was an empty placeholder at the
    time of writing, so this is a from-scratch implementation of the
    paper's algorithm rather than a wrapper around official code.
    """
    from tokenizers import Tokenizer, decoders
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.trainers import BpeTrainer

    from src.utils.morpheme_segmentation import segment_corpus

    with tempfile.TemporaryDirectory(prefix="morphbpe_") as td:
        segmented = Path(td) / "segmented.txt"
        segment_corpus(corpus_path, segmented, language=language)

        tok = Tokenizer(BPE(unk_token="<unk>"))
        tok.pre_tokenizer = Whitespace()
        tok.decoder = decoders.BPEDecoder()
        trainer = BpeTrainer(
            vocab_size=vocab_size, special_tokens=DEFAULT_SPECIAL_TOKENS
        )
        tok.train(files=[str(segmented)], trainer=trainer)
        HFAdapter(tok, algorithm="morphbpe").save(output_dir)


# Stage-1 regex: GPT-2 / tiktoken multilingual word-boundary split.
# Matches (in order): lowercase-led words, uppercase-led words, 1-3 digits,
# optional-space + symbols, newline sequences, trailing whitespace, other whitespace.
# Taken verbatim from PythonNut/superbpe scripts/train_tokenizer.sh.
_SUPERBPE_STAGE1_REGEX = (
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+"
    r"|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*"
    r"|\p{N}{1,3}"
    r"| ?[^\s\p{L}\p{N}]+[\r\n/]*"
    r"|\s*[\r\n]+"
    r"|\s+(?!\S)"
    r"|\s+"
)

# Stage-2 regex: permissive split that does NOT split on spaces before letters or
# digits, so "hello world" stays as one BPE unit and merges can cross word
# boundaries.  Only splits on: 1-3 digits, 2+ special chars, trailing spaces.
# Taken verbatim from PythonNut/superbpe scripts/extend_tokenizer.sh.
_SUPERBPE_STAGE2_REGEX = (
    r"\p{N}{1,3}"
    r"| ?[^\s\p{L}\p{N}]{2,}[\r\n/]*"
    r"| +(?!\S)"
)


def _superbpe_extend_stage2(
    tok1,
    corpus_path: Path,
    s1_vocab: dict,
    s1_merges: list,
    stage2_budget: int,
    max_lines: int = 20_000,
) -> tuple[dict, list]:
    """Extend stage-1 BPE with cross-word merges via token-level BPE extension.

    BpeTrainer ignores pre-loaded merges and retrains from scratch, so we
    implement stage-2 extension manually:
    1. Encode the corpus using stage-1 BPE with the stage-2 (permissive)
       pre-tokenizer already set on tok1.  Each document segment is now a
       sequence of stage-1 tokens that can span what used to be word
       boundaries.
    2. Run iterative BPE on those token-string sequences, counting adjacent
       pair frequencies and merging the most frequent pair each step.
    3. Return the updated vocab and the new stage-2 merges.

    The resulting merge list is [s1_merges] + [new_merges], preserving the
    hierarchical structure: word-boundary merges come first.
    """
    from collections import defaultdict

    sequences: list[list[str]] = []
    with open(corpus_path, encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if i >= max_lines:
                break
            line = line.rstrip("\n")
            if not line:
                continue
            try:
                enc = tok1.encode(line)
                tokens = list(enc.tokens)
            except Exception:
                continue
            if len(tokens) >= 2:
                sequences.append(tokens)

    if not sequences:
        return dict(s1_vocab), []

    # Count all adjacent pair frequencies across the corpus.
    pair_counts: dict[tuple[str, str], int] = defaultdict(int)
    for seq in sequences:
        for j in range(len(seq) - 1):
            pair_counts[(seq[j], seq[j + 1])] += 1

    vocab = dict(s1_vocab)
    next_id = max(vocab.values()) + 1 if vocab else 256
    new_merges: list[tuple[str, str]] = []

    for _ in range(stage2_budget):
        if not pair_counts:
            break
        best_pair = max(pair_counts, key=pair_counts.__getitem__)
        if pair_counts[best_pair] < 2:
            break
        a, b = best_pair
        merged = a + b
        if merged not in vocab:
            vocab[merged] = next_id
            next_id += 1
        new_merges.append(best_pair)
        del pair_counts[best_pair]

        # Incremental update: scan sequences, merge occurrences, adjust counts
        # for context pairs that are created or destroyed by each merge.
        for seq in sequences:
            i = 0
            while i < len(seq) - 1:
                if seq[i] == a and seq[i + 1] == b:
                    if i > 0:
                        gone = (seq[i - 1], a)
                        pair_counts[gone] -= 1
                        if pair_counts[gone] <= 0:
                            del pair_counts[gone]
                        pair_counts[(seq[i - 1], merged)] += 1
                    if i + 2 < len(seq):
                        gone = (b, seq[i + 2])
                        pair_counts[gone] -= 1
                        if pair_counts[gone] <= 0:
                            del pair_counts[gone]
                        pair_counts[(merged, seq[i + 2])] += 1
                    seq[i] = merged
                    del seq[i + 1]
                    # Stay at i: the new token may form a new pair with the next.
                else:
                    i += 1

    return vocab, new_merges


def _train_superbpe(corpus_path: Path, vocab_size: int, output_dir: Path) -> None:
    """Byte-level two-stage SuperBPE (Nut et al. 2025).

    Both stages use Sequence([Split(regex), ByteLevel()]) — the same
    byte-level BPE architecture as tiktoken/GPT-2.

    Stage 1 — GPT-2-style word-boundary regex; trains 90 % of the vocab.
    Stage 2 — permissive regex that does NOT split on spaces before
               letters/digits; extends stage-1 merges with cross-word
               tokens for the remaining 10 %, using token-level BPE
               extension (not BpeTrainer, which ignores pre-loaded merges).

    Final tokenizer: BPE(merges=[stage-1 merges] + [stage-2 merges]) with
    the stage-2 permissive pre-tokenizer so encoding uses cross-word context.
    """
    import json as _json
    from tokenizers import Regex, Tokenizer
    from tokenizers import decoders as _decoders
    from tokenizers import pre_tokenizers as _pre_tokenizers
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import ByteLevel, Split
    from tokenizers.trainers import BpeTrainer

    def _pretok(regex: str):
        return _pre_tokenizers.Sequence([
            Split(pattern=Regex(regex), behavior="isolated", invert=False),
            ByteLevel(add_prefix_space=False, trim_offsets=True, use_regex=False),
        ])

    stage1_size = int(vocab_size * 0.9)
    stage2_budget = vocab_size - stage1_size

    # --- Stage 1: word-boundary byte-level BPE ---
    tok1 = Tokenizer(BPE())
    tok1.pre_tokenizer = _pretok(_SUPERBPE_STAGE1_REGEX)
    tok1.decoder = _decoders.ByteLevel()
    trainer1 = BpeTrainer(vocab_size=stage1_size, special_tokens=DEFAULT_SPECIAL_TOKENS)
    tok1.train(files=[str(corpus_path)], trainer=trainer1)

    s1 = _json.loads(tok1.to_str())
    s1_vocab = s1["model"]["vocab"]
    s1_merges = [tuple(m) for m in s1["model"]["merges"]]

    # --- Stage 2: token-level BPE extension with permissive pre-tokenizer ---
    # Switch tok1 to stage-2 pre-tokenizer so it encodes with cross-word
    # segmentation while still applying stage-1 merges within each segment.
    tok1.pre_tokenizer = _pretok(_SUPERBPE_STAGE2_REGEX)
    final_vocab, s2_merges = _superbpe_extend_stage2(
        tok1, corpus_path, s1_vocab, s1_merges, stage2_budget
    )

    # Build the final tokenizer with the hierarchical merge list.
    final_merges = s1_merges + s2_merges
    tok_final = Tokenizer(BPE(vocab=final_vocab, merges=final_merges))
    tok_final.pre_tokenizer = _pretok(_SUPERBPE_STAGE2_REGEX)
    tok_final.decoder = _decoders.ByteLevel()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tok_final.save(str(output_dir / _TOKENIZER_FILENAME))
    # Store stage1_merge_count so tests/diagnostics can identify the boundary.
    (output_dir / _CONFIG_FILENAME).write_text(
        json.dumps({
            "algorithm": "superbpe",
            "special_tokens": DEFAULT_SPECIAL_TOKENS,
            "is_byte_level": True,
            "stage1_merge_count": len(s1_merges),
        }),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Loading dispatch
# ---------------------------------------------------------------------------


def load_tokenizer(artifact_dir: Path, algorithm: str):
    """Return an adapter for a previously saved artifact directory."""
    if algorithm not in SUPPORTED_ALGORITHMS:
        raise ValueError(f"Unknown algorithm {algorithm!r}. Choose from {SUPPORTED_ALGORITHMS}")
    dispatch = {
        "bpe": _load_bpe,
        "superbpe": _load_superbpe,
        "tiktoken": _load_tiktoken,
        "morphbpe": _load_morphbpe,
        "wordpiece": _load_wordpiece,
        "unigram": _load_unigram,
        "byt5": _load_byt5,
    }
    return dispatch[algorithm](Path(artifact_dir))


def _load_hf(artifact_dir: Path, algorithm: str) -> HFAdapter:
    from tokenizers import Tokenizer

    tok_path = artifact_dir / _TOKENIZER_FILENAME
    if not tok_path.exists():
        raise FileNotFoundError(f"No {_TOKENIZER_FILENAME} in {artifact_dir}")
    tok = Tokenizer.from_file(str(tok_path))
    return HFAdapter(tok, algorithm=algorithm)


def _load_bpe(artifact_dir: Path) -> HFAdapter:
    return _load_hf(artifact_dir, "bpe")


def _load_wordpiece(artifact_dir: Path) -> HFAdapter:
    return _load_hf(artifact_dir, "wordpiece")


def _load_unigram(artifact_dir: Path) -> HFAdapter:
    return _load_hf(artifact_dir, "unigram")


def _load_superbpe(artifact_dir: Path) -> HFAdapter:
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(artifact_dir / _TOKENIZER_FILENAME))
    return HFAdapter(tok, algorithm="superbpe", is_byte_level=True)


def _load_tiktoken(artifact_dir: Path) -> HFAdapter:
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(artifact_dir / _TOKENIZER_FILENAME))
    return HFAdapter(
        tok,
        algorithm="tiktoken",
        special_tokens=["<|endoftext|>", "<pad>", "<s>", "</s>"],
        is_byte_level=True,
    )


def _load_morphbpe(artifact_dir: Path) -> HFAdapter:
    return _load_hf(artifact_dir, "morphbpe")


def _load_byt5(artifact_dir: Path) -> ByT5Adapter:
    return ByT5Adapter()
