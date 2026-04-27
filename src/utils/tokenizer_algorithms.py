"""Tokenizer training and loading for all supported algorithms.

Supported algorithms
--------------------
bpe        – standard Byte-Pair Encoding (Sennrich et al. 2016) via HF tokenizers
superbpe   – SuperBPE two-stage training (Nut et al. 2025) via official scripts
tiktoken   – tiktoken/GPT-style byte-level BPE (ByteLevel pre-tokenizer + BPE)
morphbpe   – MorphBPE (Asgari et al. 2025); stub pending llm-lab-org integration
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
) -> None:
    """Train a tokenizer and persist it to output_dir."""
    if algorithm not in SUPPORTED_ALGORITHMS:
        raise ValueError(f"Unknown algorithm {algorithm!r}. Choose from {SUPPORTED_ALGORITHMS}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dispatch = {
        "bpe": _train_bpe,
        "superbpe": _train_superbpe,
        "tiktoken": _train_tiktoken,
        "morphbpe": _train_morphbpe,
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


def _train_morphbpe(corpus_path: Path, vocab_size: int, output_dir: Path) -> None:
    """MorphBPE (Asgari et al. 2025) — STUB.

    The official implementation lives at https://github.com/llm-lab-org/MorphBPE.
    Integration is non-trivial because MorphBPE blocks BPE merges that cross
    morpheme boundaries, which requires a per-language morpheme segmenter
    (Morfessor for English / Turkish; Mandarin is super-analytic and lacks
    the morphology MorphBPE was designed to exploit, so it should probably
    be skipped for zh).

    To wire this up, clone the repo (`make install-morphbpe` once added) and
    set MORPHBPE_REPO; this stub will then shell out to its training entry
    point and copy the resulting tokenizer.json into output_dir, like the
    SuperBPE adapter.
    """
    raise NotImplementedError(
        "MorphBPE adapter is a stub. To use it: clone "
        "https://github.com/llm-lab-org/MorphBPE, configure a per-language "
        "morpheme segmenter, and replace this body with a subprocess call "
        "into the official training entry point. See REFERENCES.md for "
        "details."
    )


def _train_superbpe(corpus_path: Path, vocab_size: int, output_dir: Path) -> None:
    """Shell out to the official PythonNut/superbpe scripts.

    Requires SUPERBPE_REPO env var pointing to a cloned superbpe repo, Python
    3.12, and the patched alisawuffles/tokenizers-superbpe fork installed
    (handled by `make install-superbpe`).

    The superbpe workflow is two-stage:
      1. train a vanilla BPE with whitespace pre-tokenizer
      2. extend by training a second stage without whitespace pre-tokenization
    We allocate 90% of the vocab budget to stage 1 and 10% to the stage-2
    transition, matching the paper's reported 180k/200k split ratio.
    """
    repo = os.environ.get("SUPERBPE_REPO")
    if not repo:
        raise RuntimeError(
            "SuperBPE training requires SUPERBPE_REPO env var pointing to the "
            "cloned PythonNut/superbpe repo. Run `make install-superbpe` first."
        )
    repo_path = Path(repo).resolve()
    if not (repo_path / "scripts" / "train_tokenizer.sh").exists():
        raise RuntimeError(
            f"SUPERBPE_REPO={repo_path} does not look like a superbpe clone "
            "(missing scripts/train_tokenizer.sh)."
        )

    stage1_size = int(vocab_size * 0.9)
    work_dir = output_dir / "_superbpe_work"
    work_dir.mkdir(parents=True, exist_ok=True)

    stage1 = subprocess.run(
        [
            "bash",
            str(repo_path / "scripts" / "train_tokenizer.sh"),
            str(corpus_path),
            str(stage1_size),
            str(work_dir / "stage1"),
        ],
        check=True,
    )
    subprocess.run(
        [
            "bash",
            str(repo_path / "scripts" / "extend_tokenizer.sh"),
            str(work_dir / "stage1"),
            str(corpus_path),
            str(vocab_size),
            str(work_dir / "stage2"),
        ],
        check=True,
    )

    # The extended tokenizer.json is the artifact.
    final = work_dir / "stage2" / "tokenizer.json"
    if not final.exists():
        raise RuntimeError(f"SuperBPE finished but no tokenizer.json at {final}")
    (output_dir / _TOKENIZER_FILENAME).write_bytes(final.read_bytes())
    (output_dir / _CONFIG_FILENAME).write_text(
        json.dumps(
            {"algorithm": "superbpe", "special_tokens": DEFAULT_SPECIAL_TOKENS, "is_byte_level": False}
        ),
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
    return _load_hf(artifact_dir, "superbpe")


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
