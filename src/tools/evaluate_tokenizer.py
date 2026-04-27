"""Evaluate tokenizer artifacts — single-shot and full-sweep entry points."""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

from src.utils.evaluation_metrics import compute_all_metrics
from src.utils.tokenizer_algorithms import SUPPORTED_ALGORITHMS, load_tokenizer

ARTIFACT_NAME_RE = re.compile(r"^(?P<lang>[a-z]{2,3})_(?P<algo>[a-z0-9]+)_v(?P<vs>\d+)$")

CSV_FIELDNAMES = [
    "language",
    "algorithm",
    "vocab_size",
    "fertility",
    "vocabulary_coverage",
    "pct_continued_words",
]


def evaluate_tokenizer_artifact(
    artifact_dir: str | Path,
    eval_corpus_path: str | Path,
    algorithm: str,
) -> dict:
    """Load one tokenizer artifact and compute metrics on an eval corpus."""
    tokenizer = load_tokenizer(Path(artifact_dir), algorithm=algorithm)
    corpus = Path(eval_corpus_path).read_text(encoding="utf-8")
    sentences = [s for s in corpus.splitlines() if s.strip()]
    return compute_all_metrics(tokenizer=tokenizer, sentences=sentences)


def parse_artifact_name(name: str) -> tuple[str, str, int] | None:
    """Return (language, algorithm, vocab_size) or None if unrecognized."""
    m = ARTIFACT_NAME_RE.match(name)
    if not m:
        return None
    return m["lang"], m["algo"], int(m["vs"])


def evaluate_all_tokenizers(
    data_dir: Path,
    artifact_dir: Path,
    results_path: Path,
    continue_on_error: bool = False,
) -> Path:
    """Iterate over every artifact in ``artifact_dir``, evaluate, write CSV.

    Each artifact's directory name is parsed as ``{lang}_{algo}_v{vocab_size}``;
    its eval corpus is read from ``{data_dir}/{lang}/eval.txt``. Skips
    artifacts whose eval corpus is missing or whose name is unparseable.

    Returns:
        Path to the CSV.
    """
    data_dir = Path(data_dir)
    artifact_dir = Path(artifact_dir)
    results_path = Path(results_path)

    if not artifact_dir.exists():
        print(f"No artifact dir at {artifact_dir}; run generate_tokenizers.py first")
        sys.exit(1)

    rows: list[dict] = []
    for artifact in sorted(artifact_dir.iterdir()):
        if not artifact.is_dir():
            continue
        parsed = parse_artifact_name(artifact.name)
        if parsed is None:
            print(f"[skip] unrecognized artifact name: {artifact.name}")
            continue
        lang, algo, vs = parsed
        if algo not in SUPPORTED_ALGORITHMS:
            print(f"[skip] unsupported algorithm in {artifact.name}")
            continue
        eval_corpus = data_dir / lang / "eval.txt"
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
            if not continue_on_error:
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

    with results_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {results_path}")
    return results_path
