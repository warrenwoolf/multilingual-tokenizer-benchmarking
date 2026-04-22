"""High-level operation: train and save one tokenizer artifact."""

from pathlib import Path

from src.utils.tokenizer_algorithms import train_tokenizer
from src.utils.tokenizer_algorithms import SUPPORTED_ALGORITHMS


def create_tokenizer_artifact(
    corpus_path: str | Path,
    algorithm: str,
    vocab_size: int,
    output_dir: str | Path,
    language: str,
) -> Path:
    """Train a tokenizer on a corpus and write the artifact to disk.

    Args:
        corpus_path: Path to a plain-text corpus file (one sentence per line).
        algorithm: One of SUPPORTED_ALGORITHMS (e.g. "bpe", "superbpe", "wordpiece", "unigram", "byt5").
        vocab_size: Target vocabulary size.
        output_dir: Directory where the artifact will be saved.
        language: ISO 639-1 language code (used to name the artifact).

    Returns:
        Path to the saved tokenizer artifact directory.
    """
    corpus_path = Path(corpus_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if algorithm not in SUPPORTED_ALGORITHMS:
        raise ValueError(
            f"Unknown algorithm '{algorithm}'. Choose from: {SUPPORTED_ALGORITHMS}"
        )

    artifact_dir = output_dir / f"{language}_{algorithm}_v{vocab_size}"
    train_tokenizer(
        corpus_path=corpus_path,
        algorithm=algorithm,
        vocab_size=vocab_size,
        output_dir=artifact_dir,
    )
    return artifact_dir
