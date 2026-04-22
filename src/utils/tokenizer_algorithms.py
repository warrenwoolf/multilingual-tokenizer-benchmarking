"""Low-level dispatch layer for tokenizer training and loading.

Supported algorithms
--------------------
bpe       – standard Byte-Pair Encoding (via tokenizers / sentencepiece)
superbpe  – SuperBPE (handles whitespace-spanning clusters)
magnet    – MAGNET morphology-aware tokenizer
"""

from pathlib import Path

SUPPORTED_ALGORITHMS = ("bpe", "superbpe", "magnet")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_tokenizer(
    corpus_path: Path,
    algorithm: str,
    vocab_size: int,
    output_dir: Path,
) -> None:
    """Train a tokenizer and persist it to output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    dispatch = {
        "bpe": _train_bpe,
        "superbpe": _train_superbpe,
        "magnet": _train_magnet,
    }
    dispatch[algorithm](corpus_path, vocab_size, output_dir)


def _train_bpe(corpus_path: Path, vocab_size: int, output_dir: Path) -> None:
    raise NotImplementedError("BPE training not yet implemented")


def _train_superbpe(corpus_path: Path, vocab_size: int, output_dir: Path) -> None:
    raise NotImplementedError("SuperBPE training not yet implemented")


def _train_magnet(corpus_path: Path, vocab_size: int, output_dir: Path) -> None:
    raise NotImplementedError("MAGNET training not yet implemented")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_tokenizer(artifact_dir: Path, algorithm: str):
    """Return a tokenizer object from a previously saved artifact directory."""
    dispatch = {
        "bpe": _load_bpe,
        "superbpe": _load_superbpe,
        "magnet": _load_magnet,
    }
    return dispatch[algorithm](artifact_dir)


def _load_bpe(artifact_dir: Path):
    raise NotImplementedError("BPE loading not yet implemented")


def _load_superbpe(artifact_dir: Path):
    raise NotImplementedError("SuperBPE loading not yet implemented")


def _load_magnet(artifact_dir: Path):
    raise NotImplementedError("MAGNET loading not yet implemented")
