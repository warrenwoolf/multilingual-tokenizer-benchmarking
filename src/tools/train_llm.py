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
import gc
import json
import math
import sys
from dataclasses import asdict, replace
from pathlib import Path

from src.tools.evaluate_tokenizer import parse_artifact_name
from src.utils.llm_training import LLMConfig, train_and_evaluate
from src.utils.tokenizer_algorithms import SUPPORTED_ALGORITHMS, load_tokenizer


def _scale_config_for_vocab(config: LLMConfig, vocab_size: int) -> LLMConfig:
    """Reduce batch_size (and LR proportionally) for large-vocab runs.

    The LM head produces logits of shape [batch, seq_len, vocab_size], so VRAM
    from that tensor (plus its gradient) scales linearly with vocab_size.  We
    scale batch_size inversely with vocab_size relative to an 8 k reference,
    snapping to a power of two, and apply the sqrt rule to keep the effective
    learning rate calibrated.
    """
    ref_vocab = 8_000
    if vocab_size <= ref_vocab:
        return config
    raw_bs = config.batch_size * ref_vocab / vocab_size
    new_bs = max(8, 2 ** int(math.log2(max(1.0, raw_bs))))
    if new_bs == config.batch_size:
        return config
    lr_scale = math.sqrt(new_bs / config.batch_size)
    return replace(
        config,
        batch_size=new_bs,
        learning_rate=config.learning_rate * lr_scale,
        min_lr=config.min_lr * lr_scale,
    )


CSV_FIELDNAMES = [
    "language",
    "algorithm",
    "vocab_size",
    "param_count",
    "train_tokens_budget",
    "train_tokens_actual",
    "train_rows",
    "train_bytes_per_row",
    "train_tokens_per_row",
    "train_seconds",
    # In-domain (FineWeb test split)
    "test_perplexity",
    "test_bits_per_byte",
    "test_mean_nll_per_token",
    "test_eval_tokens_scored",
    "test_eval_rows_scored",
    "test_eval_bytes_scored",
    "test_eval_bytes_per_row",
    # Out-of-distribution (FLORES-200 devtest)
    "flores_perplexity",
    "flores_bits_per_byte",
    "flores_mean_nll_per_token",
    "flores_eval_tokens_scored",
    "flores_eval_rows_scored",
    "flores_eval_bytes_scored",
    "flores_eval_bytes_per_row",
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
        tokenizer_artifact_dir=Path(artifact_dir),
        artifact_name=Path(artifact_dir).name,
    )


def _format_row(lang: str, algo: str, vs: int, train_tokens_budget: int, result: dict) -> dict:
    def _fmt(key: str, fmt: str = ".4f") -> str:
        v = result.get(key)
        return format(v, fmt) if isinstance(v, (int, float)) else ""

    def _int(key: str) -> str:
        v = result.get(key)
        return str(int(v)) if isinstance(v, (int, float)) else ""

    return {
        "language": lang,
        "algorithm": algo,
        "vocab_size": vs,
        "param_count": result.get("param_count", ""),
        "train_tokens_budget": train_tokens_budget,
        "train_tokens_actual": _int("train_tokens_actual"),
        "train_rows": _int("train_rows"),
        "train_bytes_per_row": _fmt("train_bytes_per_row", ".2f"),
        "train_tokens_per_row": _fmt("train_tokens_per_row", ".3f"),
        "train_seconds": _fmt("train_seconds", ".1f"),
        "test_perplexity": _fmt("test_perplexity"),
        "test_bits_per_byte": _fmt("test_bits_per_byte"),
        "test_mean_nll_per_token": _fmt("test_mean_nll_per_token"),
        "test_eval_tokens_scored": _int("test_eval_tokens_scored"),
        "test_eval_rows_scored": _fmt("test_eval_rows_scored", ".1f"),
        "test_eval_bytes_scored": _fmt("test_eval_bytes_scored", ".0f"),
        "test_eval_bytes_per_row": _fmt("test_eval_bytes_per_row", ".2f"),
        "flores_perplexity": _fmt("flores_perplexity"),
        "flores_bits_per_byte": _fmt("flores_bits_per_byte"),
        "flores_mean_nll_per_token": _fmt("flores_mean_nll_per_token"),
        "flores_eval_tokens_scored": _int("flores_eval_tokens_scored"),
        "flores_eval_rows_scored": _fmt("flores_eval_rows_scored", ".1f"),
        "flores_eval_bytes_scored": _fmt("flores_eval_bytes_scored", ".0f"),
        "flores_eval_bytes_per_row": _fmt("flores_eval_bytes_per_row", ".2f"),
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
    skip_runs: list[str] | None = None,
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
        if skip_runs and artifact.name in skip_runs:
            print(f"[skip] {artifact.name} listed in skip_runs")
            continue
        artifacts.append((artifact, lang, algo, vs))

    if not artifacts:
        print("No matching artifacts to train.")
        sys.exit(1)

    print(f"Will train {len(artifacts)} LM(s). Config:")
    print(json.dumps(asdict(config), indent=2))

    results_path.parent.mkdir(parents=True, exist_ok=True)
    completed: set[tuple[str, str, int]] = set()
    append_mode = results_path.exists() and results_path.stat().st_size > 0
    if append_mode:
        with results_path.open("r", encoding="utf-8", newline="") as rf:
            reader = csv.DictReader(rf)
            for row in reader:
                try:
                    completed.add((row["language"], row["algorithm"], int(row["vocab_size"])))
                except (KeyError, ValueError, TypeError):
                    continue

    pending = [(a, l, g, v) for (a, l, g, v) in artifacts if (l, g, v) not in completed]
    skipped_completed = len(artifacts) - len(pending)
    if skipped_completed:
        print(f"Skipping {skipped_completed} already completed artifact(s) in existing CSV")

    fh = results_path.open("a" if append_mode else "w", encoding="utf-8", newline="")
    writer = csv.DictWriter(fh, fieldnames=CSV_FIELDNAMES)
    if not append_mode:
        writer.writeheader()
        fh.flush()

    successes = 0
    failures: list[tuple[str, str]] = []
    try:
        for artifact, lang, algo, vs in pending:
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
            run_cfg = _scale_config_for_vocab(run_cfg, vs)
            if run_cfg.batch_size != config.batch_size:
                print(
                    f"  vocab={vs:,}: batch_size scaled {config.batch_size}→{run_cfg.batch_size}, "
                    f"lr={run_cfg.learning_rate:.2e} (was {config.learning_rate:.2e})"
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
            finally:
                # Flush the CUDA allocator cache between runs regardless of
                # success/failure so the next artifact starts with clean memory.
                gc.collect()
                try:
                    import torch as _torch
                    _torch.cuda.empty_cache()
                except Exception:
                    pass

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
