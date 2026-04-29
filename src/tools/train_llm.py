"""Sweep over tokenizer artifacts: train a small LM with each, log perplexity.

The loop matches ``evaluate_tokenizer.py``'s artifact discovery — it parses
``{lang}_{algo}_v{vs}`` directory names from the artifact dir and pairs each
with ``{data_dir}/{lang}/{train,eval}.txt``. Each LM is monolingual.

Two eval sets per LM:
- ``test_*``: held-out FineWeb test set (in-domain)
- ``flores_*``: FLORES-200 devtest (out-of-distribution generalisation)

If ``config.wandb_project`` is set, every (lang, algo, vocab) run opens its
own W&B run with the artifact name as the run name and ``[lang, algo]`` tags.
"""

from __future__ import annotations

import csv
import json
import sys
from dataclasses import asdict, replace
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
    # In-domain (FineWeb test split)
    "test_perplexity",
    "test_bits_per_byte",
    "test_mean_nll_per_token",
    "test_eval_tokens_scored",
    "test_eval_bytes_scored",
    # Out-of-distribution (FLORES-200 devtest)
    "flores_perplexity",
    "flores_bits_per_byte",
    "flores_mean_nll_per_token",
    "flores_eval_tokens_scored",
    "flores_eval_bytes_scored",
]


def train_and_evaluate_artifact(
    artifact_dir: Path,
    train_corpus_path: Path,
    eval_corpus_path: Path,
    algorithm: str,
    language: str,
    config: LLMConfig,
    log_fn=print,
    eval_flores: bool = True,
) -> dict:
    tokenizer = load_tokenizer(Path(artifact_dir), algorithm=algorithm)
    return train_and_evaluate(
        tokenizer=tokenizer,
        train_corpus_path=Path(train_corpus_path),
        eval_corpus_path=Path(eval_corpus_path),
        cfg=config,
        log_fn=log_fn,
        language=language,
        eval_flores=eval_flores,
        wandb_extra_config={
            "language": language,
            "algorithm": algorithm,
            "artifact": Path(artifact_dir).name,
        },
    )


def _format_row(lang: str, algo: str, vs: int, train_tokens: int, result: dict) -> dict:
    def _fmt(key: str, fmt: str = ".4f") -> str:
        v = result.get(key)
        return format(v, fmt) if isinstance(v, (int, float)) else ""

    return {
        "language": lang,
        "algorithm": algo,
        "vocab_size": vs,
        "param_count": result.get("param_count", ""),
        "train_tokens": train_tokens,
        "train_seconds": _fmt("train_seconds", ".1f"),
        "test_perplexity": _fmt("test_perplexity"),
        "test_bits_per_byte": _fmt("test_bits_per_byte"),
        "test_mean_nll_per_token": _fmt("test_mean_nll_per_token"),
        "test_eval_tokens_scored": result.get("test_eval_tokens_scored", ""),
        "test_eval_bytes_scored": _fmt("test_eval_bytes_scored", ".0f"),
        "flores_perplexity": _fmt("flores_perplexity"),
        "flores_bits_per_byte": _fmt("flores_bits_per_byte"),
        "flores_mean_nll_per_token": _fmt("flores_mean_nll_per_token"),
        "flores_eval_tokens_scored": result.get("flores_eval_tokens_scored", ""),
        "flores_eval_bytes_scored": _fmt("flores_eval_bytes_scored", ".0f"),
    }


def train_all_llms(
    data_dir: Path,
    artifact_dir: Path,
    results_path: Path,
    config: LLMConfig,
    continue_on_error: bool = False,
    only_languages: list[str] | None = None,
    only_algorithms: list[str] | None = None,
    only_vocab_sizes: list[int] | None = None,
    eval_flores: bool = True,
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
            run_cfg = replace(
                config,
                wandb_run_name=artifact.name,
                wandb_tags=list(config.wandb_tags) + [f"lang:{lang}", f"algo:{algo}"],
            )
            try:
                result = train_and_evaluate_artifact(
                    artifact_dir=artifact,
                    train_corpus_path=train_corpus,
                    eval_corpus_path=eval_corpus,
                    algorithm=algo,
                    language=lang,
                    config=run_cfg,
                    eval_flores=eval_flores,
                )
            except Exception as exc:
                print(f"[{artifact.name}] FAILED: {exc}")
                failures.append((artifact.name, repr(exc)))
                if not continue_on_error:
                    raise
                continue

            row = _format_row(lang, algo, vs, config.train_tokens, result)
            writer.writerow(row)
            fh.flush()
            successes += 1
            test_bpb = result.get("test_bits_per_byte", float("nan"))
            flores_bpb = result.get("flores_bits_per_byte", float("nan"))
            print(
                f"[{artifact.name}] test_bpb={test_bpb:.4f} "
                f"flores_bpb={flores_bpb:.4f} "
                f"params={result.get('param_count', 0):,}"
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
