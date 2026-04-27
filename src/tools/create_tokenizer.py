"""Train tokenizer artifacts — both single-shot and full-sweep entry points."""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

from src.utils.tokenizer_algorithms import SUPPORTED_ALGORITHMS, train_tokenizer

# ByT5 has a fixed ~259-id vocab (training is a no-op), so we only emit one
# artifact per language regardless of the requested vocab sweep.
SINGLE_SIZE_ALGORITHMS = {"byt5"}


def create_tokenizer_artifact(
    corpus_path: str | Path,
    algorithm: str,
    vocab_size: int,
    output_dir: str | Path,
    language: str,
) -> Path:
    """Train one tokenizer and write its artifact directory.

    Returns the path to the created artifact directory
    (``{output_dir}/{language}_{algorithm}_v{vocab_size}``).
    """
    if algorithm not in SUPPORTED_ALGORITHMS:
        raise ValueError(
            f"Unknown algorithm {algorithm!r}. Choose from: {SUPPORTED_ALGORITHMS}"
        )
    corpus_path = Path(corpus_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir = output_dir / f"{language}_{algorithm}_v{vocab_size}"
    train_tokenizer(
        corpus_path=corpus_path,
        algorithm=algorithm,
        vocab_size=vocab_size,
        output_dir=artifact_dir,
    )
    return artifact_dir


def iter_jobs(
    languages: list[str],
    algorithms: list[str],
    vocab_sizes: list[int],
):
    """Yield (language, algorithm, vocab_size) triples to train.

    Algorithms in SINGLE_SIZE_ALGORITHMS get a single vocab_size of 0 since
    their vocab is fixed.
    """
    for lang in languages:
        for algo in algorithms:
            sizes = [0] if algo in SINGLE_SIZE_ALGORITHMS else vocab_sizes
            for vs in sizes:
                yield lang, algo, vs


def generate_all_tokenizers(
    languages: list[str],
    algorithms: list[str],
    vocab_sizes: list[int],
    data_dir: Path,
    artifact_dir: Path,
    continue_on_error: bool = False,
) -> list[Path]:
    """Train every (language × algorithm × vocab_size) combination.

    Reads each language's training corpus from
    ``{data_dir}/{language}/train.txt`` (produced by download_datasets).
    Skips combinations whose corpus is missing. Prints progress to stdout.
    Exits the process with a non-zero status if any combination failed and
    ``continue_on_error`` is False.

    Returns:
        List of artifact directories that were successfully created.
    """
    unknown = set(algorithms) - set(SUPPORTED_ALGORITHMS)
    if unknown:
        raise ValueError(
            f"Unknown algorithms: {unknown}. Supported: {SUPPORTED_ALGORITHMS}"
        )

    data_dir = Path(data_dir)
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    successes: list[Path] = []
    failures: list[tuple[str, str, int, str]] = []

    for lang, algo, vs in iter_jobs(languages, algorithms, vocab_sizes):
        label = f"{lang}/{algo}/v{vs if vs else 'fixed'}"
        corpus = data_dir / lang / "train.txt"
        if not corpus.exists():
            print(
                f"[{label}] SKIP: {corpus} does not exist "
                "(run download_data.py first)"
            )
            continue
        print(f"[{label}] training ...")
        try:
            artifact = create_tokenizer_artifact(
                corpus_path=corpus,
                algorithm=algo,
                vocab_size=vs,
                output_dir=artifact_dir,
                language=lang,
            )
        except Exception as exc:
            failures.append((lang, algo, vs, repr(exc)))
            print(f"[{label}] FAILED: {exc}")
            if not continue_on_error:
                traceback.print_exc()
                sys.exit(1)
            continue
        successes.append(artifact)
        print(f"[{label}] wrote {artifact}")

    if failures:
        print(f"\n{len(failures)} failures:")
        for lang, algo, vs, msg in failures:
            print(f"  {lang}/{algo}/v{vs}: {msg}")
        if not continue_on_error:
            sys.exit(1)

    return successes
