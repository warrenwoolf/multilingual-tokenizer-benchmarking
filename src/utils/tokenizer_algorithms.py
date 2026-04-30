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


def _train_superbpe(corpus_path: Path, vocab_size: int, output_dir: Path) -> None:
    """Call train_tokenizer.py from the PythonNut/superbpe repo directly.

    Requires SUPERBPE_REPO env var pointing to a cloned superbpe repo, Python
    3.12, and the alisawuffles/tokenizers-superbpe fork installed
    (handled by `make install-superbpe`).

    The two-stage workflow mirrors extend_tokenizer.sh:
      Stage 1 — train standard BPE on 90% of the vocab budget with a corpus dir.
      Stage 2 — extend with cross-word merges (remaining 10%) by seeding the
                stage-2 dir with stage-1 merges+metadata and re-running without
                --corpus_dir so train_tokenizer reads the paths from meta.json.
    """
    import shutil
    import sys as _sys

    repo = os.environ.get("SUPERBPE_REPO")
    if not repo:
        raise RuntimeError(
            "SuperBPE training requires SUPERBPE_REPO env var pointing to the "
            "cloned PythonNut/superbpe repo. Run `make install-superbpe` first."
        )
    repo_path = Path(repo).resolve()
    if not (repo_path / "train_tokenizer.py").exists():
        raise RuntimeError(
            f"SUPERBPE_REPO={repo_path} does not look like a superbpe clone "
            "(missing train_tokenizer.py)."
        )

    # Resolve all paths up front so they stay valid when cwd=repo_path.
    corpus_path = corpus_path.resolve()
    stage1_size = int(vocab_size * 0.9)
    work_dir = output_dir.resolve() / "_superbpe_work"
    stage1_dir = work_dir / "stage1"
    stage2_dir = work_dir / "stage2"

    # train_tokenizer.py expects --corpus_dir (a directory), not a single file.
    # Expose the corpus file inside a small directory via a symlink.
    corpus_dir = work_dir / "corpus"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    corpus_link = corpus_dir / corpus_path.name
    if not corpus_link.exists():
        corpus_link.symlink_to(corpus_path)

    # Stage 1: word-boundary-respecting BPE.
    stage1_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            _sys.executable, "-m", "train_tokenizer",
            "--output_dir", str(stage1_dir),
            "--corpus_dir", str(corpus_dir),
            "--vocab_size", str(stage1_size),
        ],
        check=True,
        cwd=str(repo_path),
    )

    # Stage 2: cross-word extension (the SuperBPE step).
    # Seed stage2 with stage1's merges and meta.json so train_tokenizer extends
    # rather than retrains. No --corpus_dir needed; it reads paths from meta.json.
    stage2_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(stage1_dir / "merges.txt", stage2_dir / "merges.txt")
    shutil.copy(stage1_dir / "meta.json", stage2_dir / "meta.json")
    subprocess.run(
        [
            _sys.executable, "-m", "train_tokenizer",
            "--output_dir", str(stage2_dir),
            "--vocab_size", str(vocab_size),
        ],
        check=True,
        cwd=str(repo_path),
    )

    final = stage2_dir / "tokenizer.json"
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
