"""Sweep over tokenizer artifacts: train a small LM with each, log perplexity.

The loop matches ``evaluate_tokenizer.py``'s artifact discovery — it parses
``{lang}_{algo}_v{vs}`` directory names from the artifact dir and pairs each
with ``{data_dir}/{lang}/{train,eval}.txt``.
"""

from __future__ import annotations

import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path

from src.tools.evaluate_tokenizer import parse_artifact_name
from src.utils.llm_training import LLMConfig, train_and_evaluate
from src.utils.tokenizer_algorithms import SUPPORTED_ALGORITHMS, load_tokenizer


CSV_FIELDNAMES = [
    "language",
    "algorithm",
    "vocab_size",
    "param_count",
    "train_tokens",
    "train_seconds",
    "perplexity",
    "bits_per_byte",
    "mean_nll_per_token",
    "eval_tokens_scored",
    "eval_bytes_scored",
]


def train_and_evaluate_artifact(
    artifact_dir: Path,
    train_corpus_path: Path,
    eval_corpus_path: Path,
    algorithm: str,
    config: LLMConfig,
    log_fn=print,
) -> dict:
    tokenizer = load_tokenizer(Path(artifact_dir), algorithm=algorithm)
    return train_and_evaluate(
        tokenizer=tokenizer,
        train_corpus_path=Path(train_corpus_path),
        eval_corpus_path=Path(eval_corpus_path),
        cfg=config,
        log_fn=log_fn,
    )


def train_all_llms(
    data_dir: Path,
    artifact_dir: Path,
    results_path: Path,
    config: LLMConfig,
    continue_on_error: bool = False,
    only_languages: list[str] | None = None,
    only_algorithms: list[str] | None = None,
    only_vocab_sizes: list[int] | None = None,
) -> Path:
    """Iterate every artifact, train a small LM, and write results to CSV.

    The CSV is written incrementally (one row per finished combo) so a long
    sweep can be safely interrupted — partial results are preserved.
    """
    data_dir = Path(data_dir)
    artifact_dir = Path(artifact_dir)
    results_path = Path(results_path)

    if not artifact_dir.exists():
        print(f"No artifact dir at {artifact_dir}; run generate_tokenizers.py first")
        sys.exit(1)

    artifacts = []
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
        if only_languages and lang not in only_languages:
            continue
        if only_algorithms and algo not in only_algorithms:
            continue
        if only_vocab_sizes and vs not in only_vocab_sizes:
            continue
        artifacts.append((artifact, lang, algo, vs))

    if not artifacts:
        print("No matching artifacts to train.")
        sys.exit(1)

    print(f"Will train {len(artifacts)} LM(s). Config:")
    print(json.dumps(asdict(config), indent=2))

    results_path.parent.mkdir(parents=True, exist_ok=True)
    fh = results_path.open("w", encoding="utf-8", newline="")
    writer = csv.DictWriter(fh, fieldnames=CSV_FIELDNAMES)
    writer.writeheader()
    fh.flush()

    successes = 0
    failures: list[tuple[str, str]] = []
    try:
        for artifact, lang, algo, vs in artifacts:
            train_corpus = data_dir / lang / "train.txt"
            eval_corpus = data_dir / lang / "eval.txt"
            if not train_corpus.exists() or not eval_corpus.exists():
                print(f"[{artifact.name}] SKIP: missing train/eval corpus under {data_dir / lang}")
                continue

            print(f"\n[{artifact.name}] training small LM ...")
            try:
                result = train_and_evaluate_artifact(
                    artifact_dir=artifact,
                    train_corpus_path=train_corpus,
                    eval_corpus_path=eval_corpus,
                    algorithm=algo,
                    config=config,
                )
            except Exception as exc:
                print(f"[{artifact.name}] FAILED: {exc}")
                failures.append((artifact.name, repr(exc)))
                if not continue_on_error:
                    raise
                continue

            row = {
                "language": lang,
                "algorithm": algo,
                "vocab_size": vs,
                "param_count": result["param_count"],
                "train_tokens": config.train_tokens,
                "train_seconds": f"{result['train_seconds']:.1f}",
                "perplexity": f"{result['perplexity']:.4f}",
                "bits_per_byte": f"{result['bits_per_byte']:.4f}",
                "mean_nll_per_token": f"{result['mean_nll_per_token']:.4f}",
                "eval_tokens_scored": result["eval_tokens_scored"],
                "eval_bytes_scored": f"{result['eval_bytes_scored']:.0f}",
            }
            writer.writerow(row)
            fh.flush()
            successes += 1
            print(
                f"[{artifact.name}] ppl={result['perplexity']:.2f} "
                f"bpb={result['bits_per_byte']:.4f} "
                f"params={result['param_count']:,}"
            )
    finally:
        fh.close()

    print(f"\nWrote {successes} rows to {results_path}")
    if failures:
        print(f"{len(failures)} failures:")
        for name, msg in failures:
            print(f"  {name}: {msg}")
        if not continue_on_error:
            sys.exit(1)
    return results_path
