"""Evaluate every trained tokenizer artifact on its language's eval corpus.

Run after `generate_tokenizers.py`. Writes one CSV row per artifact.

Output:
    results.csv with columns
      language, algorithm, vocab_size, fertility, vocabulary_coverage, pct_continued_words
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

from src.tools.evaluate_tokenizer import evaluate_tokenizer_artifact
from src.utils.tokenizer_algorithms import SUPPORTED_ALGORITHMS

# --- config ---------------------------------------------------------------
DATA_DIR = Path("data") / "prepared"
ARTIFACT_DIR = Path("artifacts")
RESULTS_PATH = Path("results.csv")

ARTIFACT_NAME_RE = re.compile(r"^(?P<lang>[a-z]{2,3})_(?P<algo>[a-z0-9]+)_v(?P<vs>\d+)$")


def _parse_artifact_name(name: str):
    """Return (language, algorithm, vocab_size) or None if unrecognized."""
    m = ARTIFACT_NAME_RE.match(name)
    if not m:
        return None
    return m["lang"], m["algo"], int(m["vs"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--artifact-dir", type=Path, default=ARTIFACT_DIR)
    parser.add_argument("--results", type=Path, default=RESULTS_PATH)
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()

    if not args.artifact_dir.exists():
        print(f"No artifact dir at {args.artifact_dir}; run generate_tokenizers.py first")
        sys.exit(1)

    rows: list[dict] = []
    for artifact in sorted(args.artifact_dir.iterdir()):
        if not artifact.is_dir():
            continue
        parsed = _parse_artifact_name(artifact.name)
        if parsed is None:
            print(f"[skip] unrecognized artifact name: {artifact.name}")
            continue
        lang, algo, vs = parsed
        if algo not in SUPPORTED_ALGORITHMS:
            print(f"[skip] unsupported algorithm in {artifact.name}")
            continue
        eval_corpus = args.data_dir / lang / "eval.txt"
        if not eval_corpus.exists():
            print(f"[{artifact.name}] SKIP: no eval corpus at {eval_corpus}")
            continue

        print(f"[{artifact.name}] evaluating ...")
        try:
            metrics = evaluate_tokenizer_artifact(
                artifact_dir=artifact,
                eval_corpus_path=eval_corpus,
                algorithm=algo,
            )
        except Exception as exc:
            print(f"[{artifact.name}] FAILED: {exc}")
            if not args.continue_on_error:
                raise
            continue

        rows.append({
            "language": lang,
            "algorithm": algo,
            "vocab_size": vs,
            **{k: f"{v:.6f}" for k, v in metrics.items()},
        })
        print(f"[{artifact.name}] {metrics}")

    if not rows:
        print("No artifacts evaluated.")
        sys.exit(1)

    fieldnames = ["language", "algorithm", "vocab_size",
                  "fertility", "vocabulary_coverage", "pct_continued_words"]
    with args.results.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {args.results}")


if __name__ == "__main__":
    main()
