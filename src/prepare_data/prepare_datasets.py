"""Transform raw downloads into clean, deduplicated training / eval splits."""

from pathlib import Path


def prepare_corpus(
    raw_path: str | Path,
    output_dir: str | Path,
    train_fraction: float = 0.95,
    max_sentences: int | None = None,
) -> dict[str, Path]:
    """Clean, shuffle, and split a raw corpus file into train / eval sets.

    Args:
        raw_path: Path to the raw corpus produced by download_datasets.
        output_dir: Directory where train.txt and eval.txt will be written.
        train_fraction: Fraction of sentences to use for training.
        max_sentences: Cap total sentences (useful for fast iteration).

    Returns:
        Dict with keys "train" and "eval" pointing to the split files.
    """
    raw_path = Path(raw_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raise NotImplementedError("Corpus preparation not yet implemented")
