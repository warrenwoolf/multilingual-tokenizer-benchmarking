"""Train tokenizer artifacts — both single-shot and full-sweep entry points."""

from __future__ import annotations

import itertools
import sys
import traceback
from pathlib import Path
from typing import Any

from src.utils.morpheme_segmentation import SUPPORTED_LANGUAGES as MORPHBPE_LANGUAGES
from src.utils.tokenizer_algorithms import (
    SUPPORTED_ALGORITHMS,
    train_tokenizer,
    load_tokenizer,
)
from src.tools.wandb_artifacts import push_tokenizer_artifact
from src.tools.evaluate_tokenizer import evaluate_tokenizer_artifact

# ByT5 has a fixed 256-byte vocab (training is a no-op), so we only emit one
# artifact per language regardless of the requested vocab sweep.
SINGLE_SIZE_ALGORITHMS = {"byt5"}
# Actual vocab sizes for fixed-size algorithms (used in artifact naming so the
# CSV correctly shows 256 rather than 0).
_FIXED_VOCAB_SIZES: dict[str, int] = {"byt5": 256}

# Per-algorithm language allow-lists. Anything not listed here is treated
# as language-agnostic. MorphBPE needs a per-language morpheme segmenter,
# so we skip languages we don't have one for (notably Mandarin).
ALGORITHM_LANGUAGE_ALLOWLIST: dict[str, frozenset[str]] = {
    "morphbpe": frozenset(MORPHBPE_LANGUAGES),
}


def _algorithm_supports_language(algorithm: str, language: str) -> bool:
    allowed = ALGORITHM_LANGUAGE_ALLOWLIST.get(algorithm)
    return allowed is None or language in allowed


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
        language=language,
    )
    return artifact_dir


def _first_nonempty_line(path: Path) -> str:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.rstrip("\n\r")
            if text.strip():
                return text
    raise ValueError(f"No non-empty training samples found in {path}")


def _token_texts_for_ids(tokenizer: Any, token_ids: list[int]) -> list[str]:
    if hasattr(tokenizer, "tokenizer"):
        return [tokenizer.tokenizer.id_to_token(i) or f"<id:{i}>" for i in token_ids]
    if hasattr(tokenizer, "_tok"):
        return [str(t) for t in tokenizer._tok.convert_ids_to_tokens(token_ids)]
    return [str(i) for i in token_ids]


def _truncate_display(text: str, limit: int = 100) -> str:
    """Return a possibly-truncated representation suitable for printing."""
    if text is None:
        return ""
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."


def _print_sample_roundtrip(artifact_dir: Path, algorithm: str, corpus_path: Path, label: str) -> None:
    tokenizer = load_tokenizer(artifact_dir, algorithm)
    source_text = _first_nonempty_line(corpus_path)
    token_ids = tokenizer.encode(source_text)
    token_texts = _token_texts_for_ids(tokenizer, token_ids)
    detokenized = tokenizer.decode(token_ids)

    # Truncate long samples for compact logging in quick runs / Colab
    truncated_source = _truncate_display(source_text, limit=100)
    joined_tokens = " ".join(token_texts)
    truncated_tokens = _truncate_display(joined_tokens, limit=100)
    truncated_detok = _truncate_display(detokenized, limit=100)
    print(f"[{label}] sample original: {truncated_source!r}")
    print(f"[{label}] sample tokens: {truncated_tokens!r}")
    print(f"[{label}] sample detokenized: {truncated_detok!r}")

    # Re-encode to verify round-trip consistency (IDs should match).
    # Note: space normalization is acceptable — some tokenizers normalize whitespace.
    re_encoded_ids = tokenizer.encode(detokenized)
    if re_encoded_ids != token_ids:
        raise ValueError(
            f"Detokenization consistency error: "
            f"encode → decode → encode gave different IDs. "
            f"Original IDs: {token_ids}, re-encoded: {re_encoded_ids}"
        )
    print(f"[{label}] sample round-trip check: OK")


def iter_jobs(
    languages: list[str],
    algorithms: list[str],
    vocab_sizes: list[int],
):
    """Yield (language, algorithm, vocab_size) triples to train.

    Algorithms in SINGLE_SIZE_ALGORITHMS get their actual fixed vocab size
    instead of the sweep values, so artifact names reflect the real vocab.
    """
    for lang, algo in itertools.product(languages, algorithms):
        sizes = [_FIXED_VOCAB_SIZES[algo]] if algo in SINGLE_SIZE_ALGORITHMS else vocab_sizes
        for vs in sizes:
            yield lang, algo, vs


def generate_all_tokenizers(
    languages: list[str],
    algorithms: list[str],
    vocab_sizes: list[int],
    data_dir: Path,
    artifact_dir: Path,
    continue_on_error: bool = False,
    *,
    skip_existing: bool = True,
    evaluate_each: bool = False,
    upload_each: bool = False,
    wandb_project: str | None = None,
    wandb_entity: str | None = None,
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
        if not _algorithm_supports_language(algo, lang):
            print(f"[{label}] SKIP: {algo} has no segmenter for {lang}")
            continue
        corpus = data_dir / lang / "train.txt"
        if not corpus.exists():
            print(
                f"[{label}] SKIP: {corpus} does not exist "
                "(run download_data.py first)"
            )
            continue
        # Skip if artifact already exists (useful when resuming or when
        # artifacts were downloaded from W&B). The presence of tokenizer.json
        # indicates a completed artifact.
        artifact = artifact_dir / f"{lang}_{algo}_v{vs}"
        if skip_existing and artifact.exists() and (artifact / "tokenizer.json").exists():
            print(f"[{label}] SKIP: artifact already present at {artifact}")
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
            _print_sample_roundtrip(
                artifact_dir=artifact,
                algorithm=algo,
                corpus_path=corpus,
                label=label,
            )
        except Exception as exc:
            failures.append((lang, algo, vs, repr(exc)))
            print(f"[{label}] FAILED: {exc}")
            if not continue_on_error:
                traceback.print_exc()
                sys.exit(1)
            continue
        # Immediately evaluate the freshly-trained artifact if requested.
        if evaluate_each:
            eval_corpus = data_dir / lang / "eval.txt"
            if eval_corpus.exists():
                try:
                    metrics = evaluate_tokenizer_artifact(artifact, eval_corpus, algorithm=algo)
                    print(f"[{label}] immediate metrics: {metrics}")
                except Exception as exc:
                    print(f"[{label}] immediate evaluation failed: {exc}")
            else:
                print(f"[{label}] immediate evaluation skipped: no eval corpus at {eval_corpus}")

        # Immediately upload the artifact to W&B if requested.
        if upload_each:
            if wandb_project:
                try:
                    push_tokenizer_artifact(artifact, project=wandb_project, entity=wandb_entity)
                except Exception as exc:
                    print(f"[{label}] W&B upload failed (continuing): {exc}")
            else:
                print(f"[{label}] W&B upload requested but no project specified; skipping upload")
        successes.append(artifact)
        print(f"[{label}] wrote {artifact}")

    if failures:
        print(f"\n{len(failures)} failures:")
        for lang, algo, vs, msg in failures:
            print(f"  {lang}/{algo}/v{vs}: {msg}")
        if not continue_on_error:
            sys.exit(1)

    return successes
