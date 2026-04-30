"""Smoke tests for the small-LM downstream tokenizer evaluation.

These tests train a tiny LM (a few thousand parameters, a handful of steps)
on the multilingual fixture corpus, just to assert the train + perplexity
pipeline runs end-to-end and produces sane numbers.

Skipped when torch is not installed — the LLM eval is an optional extra.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")  # whole module is gated on torch availability

from src.utils.llm_training import (
    LLMConfig,
    count_parameters,
    evaluate_perplexity,
    tokenize_corpus,
    train_and_evaluate,
    train_lm,
)
from src.utils.tokenizer_algorithms import load_tokenizer, train_tokenizer


TINY_VOCAB = 512
TINY_CONFIG = LLMConfig(
    d_model=64,
    n_layers=2,
    n_heads=4,
    d_ff=128,
    ctx_len=32,
    train_tokens=20_000,   # tiny: a few dozen steps
    batch_size=8,
    warmup_steps=5,
    learning_rate=1e-3,
    seed=0,
    device="cpu",
    dtype="fp32",
)


@pytest.fixture(scope="module")
def trained_bpe(tiny_corpus, tmp_path_factory):
    out = tmp_path_factory.mktemp("bpe_artifact")
    train_tokenizer(
        corpus_path=tiny_corpus,
        algorithm="bpe",
        vocab_size=TINY_VOCAB,
        output_dir=out,
    )
    return load_tokenizer(out, algorithm="bpe")


def test_tokenize_corpus_respects_max_tokens(tiny_corpus, trained_bpe):
    corpus = tokenize_corpus(trained_bpe, tiny_corpus, max_tokens=1000)
    assert corpus.ids.dtype.name == "int32"
    assert 0 < corpus.n_tokens <= 1000
    assert corpus.rows > 0
    assert corpus.source_bytes > 0


def test_tokenize_corpus_full_pass(tiny_corpus, trained_bpe):
    corpus = tokenize_corpus(trained_bpe, tiny_corpus, max_tokens=None)
    # Tiny corpus is ~26 lines × 200 repeats; should yield way more than ctx_len.
    assert corpus.n_tokens > TINY_CONFIG.ctx_len * 10
    assert corpus.rows > 100  # 26 lines * 200 repeats = 5200 — plenty of margin


def test_tokenize_corpus_bytes_per_row_is_sane(tiny_corpus, trained_bpe):
    """Bytes-per-row indicator must be positive and roughly the line length."""
    corpus = tokenize_corpus(trained_bpe, tiny_corpus, max_tokens=None)
    # The fixture mixes en/ru/hi/tr, ~50-100 chars/line; UTF-8 bumps non-Latin
    # rows. A range of 30-300 covers anything sane while catching off-by-orders.
    assert 30 < corpus.bytes_per_row < 300
    assert corpus.tokens_per_row > 0


def test_train_lm_runs_and_reduces_loss(tiny_corpus, trained_bpe, capsys):
    """Train a tiny model for a handful of steps; assert no crash and finite loss."""
    losses: list[float] = []

    def capture(msg):
        # Capture per-step lines like "    step 1/N  loss=4.12  lr=...".
        if "loss=" in msg:
            try:
                losses.append(float(msg.split("loss=")[1].split()[0]))
            except (IndexError, ValueError):
                pass

    model, device, amp_dtype, train_seconds, train_corpus = train_lm(
        trained_bpe, tiny_corpus, TINY_CONFIG, log_fn=capture
    )
    assert train_seconds >= 0
    assert losses, "expected at least one logged loss"
    assert all(math.isfinite(l) for l in losses)
    # Sanity: parameter count is in the right ballpark for the tiny config.
    n = count_parameters(model)
    assert 10_000 < n < 5_000_000
    # train_corpus stats are surfaced for the bytes-per-row indicator.
    assert train_corpus.rows > 0
    assert train_corpus.bytes_per_row > 0


def test_train_and_evaluate_returns_finite_metrics(tiny_corpus, trained_bpe):
    metrics = train_and_evaluate(
        tokenizer=trained_bpe,
        train_corpus_path=tiny_corpus,
        eval_corpus_path=tiny_corpus,  # same fixture for the smoke test
        cfg=TINY_CONFIG,
        log_fn=lambda *a, **k: None,
        # language=None disables FLORES; we don't want network in unit tests.
        eval_flores=False,
    )
    assert math.isfinite(metrics["test_perplexity"])
    assert metrics["test_perplexity"] > 1.0  # PPL is bounded below by 1
    assert math.isfinite(metrics["test_bits_per_byte"])
    assert metrics["test_bits_per_byte"] > 0
    assert metrics["param_count"] > 0
    assert metrics["test_eval_tokens_scored"] > 0
    # New row-based stats should be present and consistent.
    assert metrics["train_rows"] > 0
    assert metrics["train_bytes_per_row"] > 0
    assert metrics["test_eval_rows_scored"] > 0
    assert metrics["test_eval_bytes_per_row"] > 0
    # FLORES eval was disabled, so flores_* keys must not be present.
    assert not any(k.startswith("flores_") for k in metrics)


def test_flores_config_map_is_complete():
    """Every language we configure for tokenizer training has a FLORES code."""
    from src.prepare_data.download_datasets import LANGUAGE_CONFIGS
    from src.utils.llm_training import FLORES_CONFIGS

    missing = set(LANGUAGE_CONFIGS) - set(FLORES_CONFIGS)
    assert not missing, f"FLORES_CONFIGS missing entries for {missing}"


def test_load_flores_rejects_unknown_language():
    from src.utils.llm_training import load_flores_devtest

    with pytest.raises(ValueError, match="No FLORES config"):
        load_flores_devtest("xx")


def test_wandb_disabled_when_project_unset(tiny_corpus, trained_bpe):
    """No W&B run is created when ``wandb_project`` is None — tests stay offline."""
    cfg = LLMConfig(**{**vars(TINY_CONFIG), "wandb_project": None})
    metrics = train_and_evaluate(
        tokenizer=trained_bpe,
        train_corpus_path=tiny_corpus,
        eval_corpus_path=tiny_corpus,
        cfg=cfg,
        eval_flores=False,
        log_fn=lambda *a, **k: None,
    )
    assert "test_perplexity" in metrics


def test_perplexity_matches_exp_of_mean_nll(tiny_corpus, trained_bpe):
    """exp(mean_nll_per_token) must equal perplexity (sanity check for BPB)."""
    model, device, amp_dtype, _, _ = train_lm(
        trained_bpe, tiny_corpus, TINY_CONFIG, log_fn=lambda *a, **k: None
    )
    metrics = evaluate_perplexity(
        model=model,
        tokenizer=trained_bpe,
        eval_corpus_path=tiny_corpus,
        cfg=TINY_CONFIG,
        device=device,
        amp_dtype=amp_dtype,
        log_fn=lambda *a, **k: None,
    )
    assert math.isclose(
        metrics["perplexity"],
        math.exp(metrics["mean_nll_per_token"]),
        rel_tol=1e-6,
    )


def test_tiny_corpus_too_small_raises(tmp_path, trained_bpe):
    """Token streams shorter than ctx_len+2 should error clearly."""
    short = tmp_path / "short.txt"
    short.write_text("hi", encoding="utf-8")
    cfg = LLMConfig(**{**vars(TINY_CONFIG), "ctx_len": 64})
    with pytest.raises(RuntimeError, match="Train corpus only produced"):
        train_lm(trained_bpe, short, cfg, log_fn=lambda *a, **k: None)
