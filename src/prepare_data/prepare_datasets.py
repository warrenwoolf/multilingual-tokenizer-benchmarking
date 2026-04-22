"""Transform raw downloads into clean, deduplicated train / eval splits."""

from __future__ import annotations

import hashlib
import random
from pathlib import Path

SEED = 42


def prepare_corpus(
    raw_path: str | Path,
    output_dir: str | Path,
    train_fraction: float = 0.95,
    max_sentences: int | None = None,
) -> dict[str, Path]:
    """Clean, dedupe, shuffle, and split a raw corpus into train / eval.

    Steps:
      1. Read all lines from raw_path.
      2. Strip whitespace; drop empty lines and lines shorter than 16 chars.
      3. Deduplicate by SHA1 hash of the line.
      4. Shuffle with a fixed seed (reproducible).
      5. Cap to max_sentences if provided.
      6. Split at train_fraction; write train.txt and eval.txt.

    Args:
        raw_path: Path to the raw corpus produced by download_datasets.
        output_dir: Directory where train.txt / eval.txt will be written.
        train_fraction: Fraction of sentences to use for training (0 < f < 1).
        max_sentences: Cap total sentences retained (None = no cap).

    Returns:
        ``{"train": Path, "eval": Path}``.
    """
    if not 0.0 < train_fraction < 1.0:
        raise ValueError(f"train_fraction must be in (0, 1), got {train_fraction}")

    raw_path = Path(raw_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seen: set[str] = set()
    kept: list[str] = []
    with raw_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if len(line) < 16:
                continue
            h = hashlib.sha1(line.encode("utf-8")).hexdigest()
            if h in seen:
                continue
            seen.add(h)
            kept.append(line)

    rng = random.Random(SEED)
    rng.shuffle(kept)

    if max_sentences is not None:
        kept = kept[:max_sentences]

    if len(kept) < 2:
        raise ValueError(
            f"Corpus at {raw_path} has only {len(kept)} usable sentences; "
            "need at least 2 to split train/eval."
        )

    split_at = max(1, int(len(kept) * train_fraction))
    train_lines = kept[:split_at]
    eval_lines = kept[split_at:]
    if not eval_lines:
        # train_fraction rounded to everything; peel off one for eval.
        eval_lines = [train_lines.pop()]

    train_path = output_dir / "train.txt"
    eval_path = output_dir / "eval.txt"
    train_path.write_text("\n".join(train_lines) + "\n", encoding="utf-8")
    eval_path.write_text("\n".join(eval_lines) + "\n", encoding="utf-8")

    return {"train": train_path, "eval": eval_path}
