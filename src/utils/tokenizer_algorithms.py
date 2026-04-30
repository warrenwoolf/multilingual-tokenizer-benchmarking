"""Tokenizer training and loading for all supported algorithms.

Supported algorithms
--------------------
bpe        – standard Byte-Pair Encoding (Sennrich et al. 2016) via HF tokenizers
superbpe   – SuperBPE two-stage training (Nut et al. 2025) via official scripts
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
_MORPHEME_LOOKUP_FILENAME = "morpheme_lookup.json"


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


@dataclass
class MorphBPEAdapter:
    """Wraps a ByteLevel BPE tokenizer with inference-time morpheme segmentation.

    encode() segments each word into morphemes before tokenizing, so the
    inference distribution matches the morpheme-segmented training corpus and
    no cross-morpheme tokens are produced at inference time.

    decode() returns morpheme-split text (spaces at morpheme boundaries).
    This is intentional: faithful round-trip is not required for BPB
    measurement, and the byte denominator in BPB is always taken from the
    original (unsegmented) text, not from the decoded output.
    """

    tokenizer: Any  # tokenizers.Tokenizer
    language: str
    lookup: dict = field(repr=False)  # word → [morpheme, ...]
    algorithm: str = "morphbpe"
    special_tokens: list[str] = field(default_factory=lambda: list(DEFAULT_SPECIAL_TOKENS))
    is_byte_level: bool = False
    _seg_cache: dict = field(default_factory=dict, repr=False)

    def _segmenter(self, word: str) -> list[str]:
        return self.lookup.get(word, [word])

    def encode(self, text: str) -> list[int]:
        if text == "":
            return []
        from src.utils.morpheme_segmentation import _segment_word

        words = text.split()
        pieces: list[str] = []
        for w in words:
            pieces.extend(_segment_word(self._segmenter, w, self._seg_cache))
        return list(self.tokenizer.encode(" ".join(pieces)).ids)

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
                    "language": self.language,
                }
            ),
            encoding="utf-8",
        )
        (path / _MORPHEME_LOOKUP_FILENAME).write_text(
            json.dumps(self.lookup),
            encoding="utf-8",
        )

    def __getstate__(self) -> dict:
        return {
            "tokenizer_json": self.tokenizer.to_str(),
            "language": self.language,
            "lookup": self.lookup,
            "algorithm": self.algorithm,
            "special_tokens": self.special_tokens,
            "is_byte_level": self.is_byte_level,
        }

    def __setstate__(self, state: dict) -> None:
        from tokenizers import Tokenizer

        self.tokenizer = Tokenizer.from_str(state["tokenizer_json"])
        self.language = state["language"]
        self.lookup = state["lookup"]
        self.algorithm = state["algorithm"]
        self.special_tokens = state["special_tokens"]
        self.is_byte_level = state["is_byte_level"]
        self._seg_cache = {}


# ---------------------------------------------------------------------------
# Training dispatch
# ---------------------------------------------------------------------------


def train_tokenizer(
    corpus_path: Path,
    algorithm: str,
    vocab_size: int,
    output_dir: Path,
    language: str | None = None,
    morphynet_cache_dir: Path | None = None,
) -> None:
    """Train a tokenizer and persist it to output_dir.

    ``language`` is required for algorithms that need a per-language asset
    (currently only MorphBPE, which needs a morpheme segmenter); other
    algorithms ignore it.

    ``morphynet_cache_dir`` controls where MorphyNet data is cached for
    English MorphBPE. Defaults to ``~/.cache/morphynet``; pass a tmp
    directory in tests to avoid network access.
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
        _train_morphbpe(Path(corpus_path), vocab_size, output_dir, language, morphynet_cache_dir=morphynet_cache_dir)
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
    from tokenizers.pre_tokenizers import ByteLevel as ByteLevelPre
    from tokenizers.trainers import BpeTrainer

    tok = Tokenizer(BPE(unk_token="<unk>"))
    tok.pre_tokenizer = ByteLevelPre(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
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
    corpus_path: Path,
    vocab_size: int,
    output_dir: Path,
    language: str,
    morphynet_cache_dir: Path | None = None,
) -> None:
    """MorphBPE (Asgari et al. 2025).

    The paper's Algorithm 1 trains BPE on the *original* corpus while
    filtering out any candidate merge whose pair spans a morpheme boundary.
    We approximate that by morpheme-segmenting the corpus first (each word
    becomes its whitespace-joined morphemes) and then training standard HF
    BPE on the rewritten corpus.  The ByteLevel pre-tokenizer splits on
    those inserted spaces, guaranteeing BPE never counts or merges a pair
    across a morpheme boundary during training.

    Training guarantee: no cross-morpheme merge is ever *learned* because
    the ByteLevel pre-tokenizer splits on whitespace, so it cannot count
    pairs across the inserted morpheme-boundary spaces.

    Inference: encode() in MorphBPEAdapter segments each word at inference
    time using the same MorphyNet lookup, so the inference distribution
    matches training and no cross-morpheme tokens are produced.  The lookup
    is persisted in the artifact (morpheme_lookup.json) so load_tokenizer
    can reconstruct the adapter without network access.

    Approximation note: Algorithm 1 trains on the original (unsegmented)
    corpus and filters cross-morpheme merge candidates inline.  Our approach
    trains on the segmented corpus, which places Ġ on every post-morpheme-
    boundary token rather than only on word-initial tokens.  The two
    approaches produce different merge tables and different BPB values, but
    both enforce morpheme boundaries and both give valid, comparable BPB.

    Per-language segmenters (src/utils/morpheme_segmentation.py):
      en — MorphyNet gold inflectional lookup (~650k entries, downloaded once).
      hu — MorphyNet gold inflectional lookup (~1M entries, downloaded once).

    The official llm-lab-org/MorphBPE repo was an empty placeholder at the
    time of writing, so this is a from-scratch implementation of the
    paper's algorithm rather than a wrapper around official code.
    """
    from src.utils.morpheme_segmentation import (
        MORPHYNET_DEFAULT_CACHE,
        _load_morphynet,
        segment_corpus,
    )
    from tokenizers import Tokenizer, decoders
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import ByteLevel as ByteLevelPre
    from tokenizers.trainers import BpeTrainer

    cache_dir = Path(morphynet_cache_dir) if morphynet_cache_dir is not None else MORPHYNET_DEFAULT_CACHE

    with tempfile.TemporaryDirectory(prefix="morphbpe_") as td:
        segmented = Path(td) / "segmented.txt"
        # segment_corpus validates the language and raises NotImplementedError for
        # unsupported ones (e.g. zh) before we attempt to load the MorphyNet lookup.
        segment_corpus(corpus_path, segmented, language=language, morphynet_cache_dir=morphynet_cache_dir)
        lookup = _load_morphynet(language, cache_dir)

        tok = Tokenizer(BPE(unk_token="<unk>"))
        tok.pre_tokenizer = ByteLevelPre(add_prefix_space=False)
        tok.decoder = decoders.ByteLevel()
        trainer = BpeTrainer(
            vocab_size=vocab_size, special_tokens=DEFAULT_SPECIAL_TOKENS
        )
        tok.train(files=[str(segmented)], trainer=trainer)
        MorphBPEAdapter(tokenizer=tok, language=language, lookup=lookup).save(output_dir)


def _train_superbpe(corpus_path: Path, vocab_size: int, output_dir: Path) -> None:
    """Train SuperBPE by delegating to the official PythonNut/superbpe repo.

    The previous in-process implementation has been removed because it was a
    manual approximation. If that legacy code path is ever reintroduced, it
    should fail loudly rather than silently diverge from the official repo.
    """
    from src.tools.superbpe_runner import (
        SuperBPESetupError,
        train_superbpe as _train_superbpe_official,
    )

    try:
        _train_superbpe_official(corpus_path=corpus_path, vocab_size=vocab_size, output_dir=output_dir)
    except SuperBPESetupError as exc:
        raise RuntimeError(
            "SuperBPE requires the official PythonNut/superbpe checkout with its Rust-backed tokenizers fork. "
            "Run `make install-superbpe` or execute `scripts/install_superbpe.sh` first."
        ) from exc


def _train_superbpe_legacy_manual(*_: object, **__: object) -> None:
    raise RuntimeError(
        "The old manual SuperBPE implementation has been removed. Use the official PythonNut/superbpe repo instead."
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
    from tokenizers import Tokenizer, decoders

    tok = Tokenizer.from_file(str(artifact_dir / _TOKENIZER_FILENAME))
    tok.decoder = decoders.ByteLevel()
    return HFAdapter(tok, algorithm="superbpe")


def _load_tiktoken(artifact_dir: Path) -> HFAdapter:
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(artifact_dir / _TOKENIZER_FILENAME))
    return HFAdapter(
        tok,
        algorithm="tiktoken",
        special_tokens=["<|endoftext|>", "<pad>", "<s>", "</s>"],
        is_byte_level=True,
    )


def _load_morphbpe(artifact_dir: Path) -> MorphBPEAdapter:
    from tokenizers import Tokenizer

    tok_path = artifact_dir / _TOKENIZER_FILENAME
    if not tok_path.exists():
        raise FileNotFoundError(f"No {_TOKENIZER_FILENAME} in {artifact_dir}")
    tok = Tokenizer.from_file(str(tok_path))

    config = json.loads((artifact_dir / _CONFIG_FILENAME).read_text(encoding="utf-8"))

    lookup_path = artifact_dir / _MORPHEME_LOOKUP_FILENAME
    if not lookup_path.exists():
        raise FileNotFoundError(f"No {_MORPHEME_LOOKUP_FILENAME} in {artifact_dir}")
    lookup = json.loads(lookup_path.read_text(encoding="utf-8"))

    return MorphBPEAdapter(tokenizer=tok, language=config["language"], lookup=lookup)


def _load_byt5(artifact_dir: Path) -> ByT5Adapter:
    return ByT5Adapter()
