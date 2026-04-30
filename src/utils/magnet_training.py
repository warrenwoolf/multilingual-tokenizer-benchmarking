"""Train a MAGNET model on a byte corpus and evaluate bits-per-byte.

MAGNET (Ahia et al. 2024) trains tokenisation end-to-end with the language
model. The model operates on raw UTF-8 bytes (via the ByT5 tokenizer's
byte vocabulary) and learns where to place segment boundaries through a
per-position boundary predictor trained with a Binomial prior.

Because every prediction target is a byte, the cross-entropy loss is
*already* in nats/byte. Converting to bits just requires dividing by ln(2):

    bpb = mean_cross_entropy_nats / ln(2)

This makes MAGNET's BPB directly comparable to the BPB reported by the
subword LMs in llm_training.py (where the normalisation happens explicitly
by dividing total-nats by total-raw-bytes).

Training
--------
Total loss = LM cross-entropy + boundary_lambda * boundary_loss.

- LM loss: standard next-byte prediction (cross-entropy over the byte vocab).
- Boundary loss: Binomial regulariser from BoundaryPredictor.boundary_loss()
  that encourages the total number of boundaries per sequence to match
  Binomial(L, prior).  This is the only signal that trains the boundary
  predictor; the LM loss does not backpropagate through boundary positions.

Evaluation
----------
In eval mode the model uses hard boundaries (sigmoid threshold) and the
boundary loss is omitted. We report BPB on both an in-domain held-out set
and optionally on FLORES-200 devtest sentences.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterator

import numpy as np


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class MagnetConfig:
    """Hyperparameters for MAGNET training.

    Defaults are deliberately small so the model trains in reasonable time
    on a single GPU. The byte-level vocabulary (~384 tokens) means sequences
    are 3-6× longer than their subword equivalents, so we use a shorter
    context length and fewer parameters than the GPT baseline.

    Boundary prior controls compression: prior=0.25 → ~4 bytes/segment on
    average, which is comparable to a subword tokenizer with fertility ≈ 4.
    """

    # Architecture
    d_model: int = 256
    n_heads: int = 4
    d_ff: int = 1024
    dropout: float = 0.0
    pre_layers: int = 4
    shortened_layers: int = 4
    post_layers: int = 0
    ctx_len: int = 512  # context length in bytes

    # Boundary predictor
    boundary_prior: float = 0.25   # target fraction of bytes that are boundaries
    boundary_temp: float = 1.0     # Gumbel temperature (lower = sharper boundaries)
    boundary_threshold: float = 0.5
    boundary_lambda: float = 1.0   # weight for the boundary regularisation loss

    # Training
    train_tokens: int = 100_000_000  # bytes to train on (fewer than subword LMs
                                      # because byte sequences are much longer)
    batch_size: int = 16
    learning_rate: float = 3e-4
    min_lr: float = 3e-5
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    warmup_steps: int = 100

    # Misc
    seed: int = 0
    device: str = "auto"
    dtype: str = "auto"
    log_every: int = 50

    # Weights & Biases (optional)
    wandb_project: str | None = None
    wandb_entity: str | None = None
    wandb_run_name: str | None = None
    wandb_tags: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# ByT5 byte vocabulary
# ---------------------------------------------------------------------------


def _load_byt5_tokenizer():
    """Return the ByT5 tokenizer used as the byte vocabulary."""
    from transformers import ByT5Tokenizer

    return ByT5Tokenizer()


def _encode_line(tokenizer, line: str) -> list[int]:
    return tokenizer.encode(line, add_special_tokens=False)


# ---------------------------------------------------------------------------
# Corpus helpers
# ---------------------------------------------------------------------------


def _iter_lines(path: Path) -> Iterator[str]:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield line


def tokenize_byte_corpus(
    tokenizer,
    corpus_path: Path,
    max_tokens: int | None = None,
) -> tuple[np.ndarray, int, int]:
    """Tokenize a text corpus into a flat array of byte ids.

    Returns (ids_array, n_rows, source_bytes).
    ``source_bytes`` is the raw UTF-8 byte count of the consumed text,
    which is the correct denominator for BPB when using a byte-level model.
    """
    chunks: list[np.ndarray] = []
    total_tokens = 0
    rows = 0
    src_bytes = 0

    for line in _iter_lines(corpus_path):
        ids = _encode_line(tokenizer, line)
        if not ids:
            continue
        if max_tokens is not None and total_tokens + len(ids) > max_tokens:
            ids = ids[: max_tokens - total_tokens]
            if ids:
                chunks.append(np.asarray(ids, dtype=np.int32))
                total_tokens += len(ids)
                rows += 1
                src_bytes += len(line.encode("utf-8"))
            break
        chunks.append(np.asarray(ids, dtype=np.int32))
        total_tokens += len(ids)
        rows += 1
        src_bytes += len(line.encode("utf-8"))

    arr = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.int32)
    return arr, rows, src_bytes


# ---------------------------------------------------------------------------
# Device / dtype helpers (mirrors llm_training.py)
# ---------------------------------------------------------------------------


def _resolve_device(spec: str):
    import torch

    if spec != "auto":
        return torch.device(spec)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _resolve_amp_dtype(spec: str, device):
    import torch

    if spec == "auto":
        if device.type == "cuda" and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return None
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": None}[spec]


# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------


def _lr_at_step(step: int, total_steps: int, cfg: MagnetConfig) -> float:
    if step < cfg.warmup_steps:
        return cfg.learning_rate * (step + 1) / max(1, cfg.warmup_steps)
    if step >= total_steps:
        return cfg.min_lr
    progress = (step - cfg.warmup_steps) / max(1, total_steps - cfg.warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return cfg.min_lr + (cfg.learning_rate - cfg.min_lr) * coeff


# ---------------------------------------------------------------------------
# Batch sampling
# ---------------------------------------------------------------------------


def _sample_batch(
    token_array: np.ndarray,
    batch_size: int,
    ctx_len: int,
    generator,
):
    import torch

    n = token_array.shape[0]
    high = n - ctx_len - 1
    if high <= 0:
        raise RuntimeError(
            f"Token stream too short ({n} bytes) for ctx_len={ctx_len}; "
            "increase the training corpus or reduce ctx_len."
        )
    starts = torch.randint(0, high, (batch_size,), generator=generator).numpy()
    inputs = np.stack([token_array[s : s + ctx_len] for s in starts])
    targets = np.stack([token_array[s + 1 : s + 1 + ctx_len] for s in starts])
    x = torch.from_numpy(inputs).long()
    y = torch.from_numpy(targets).long()
    return x, y


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_magnet(
    train_corpus_path: Path,
    cfg: MagnetConfig,
    log_fn=print,
    wandb_run=None,
):
    """Train a MAGNET model for cfg.train_tokens bytes.

    Returns (model, tokenizer, device, amp_dtype, train_seconds, corpus_stats).
    ``corpus_stats`` is a dict with n_tokens, rows, source_bytes.
    """
    import torch
    from src.utils.magnet_model import MagnetLM

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = _resolve_device(cfg.device)
    amp_dtype = _resolve_amp_dtype(cfg.dtype, device)

    log_fn(f"  device={device}  amp_dtype={amp_dtype}")

    tokenizer = _load_byt5_tokenizer()
    vocab_size = tokenizer.vocab_size

    log_fn(f"  tokenizing train corpus (cap = {cfg.train_tokens:,} bytes) ...")
    t0 = time.time()
    train_ids, rows, src_bytes = tokenize_byte_corpus(
        tokenizer, train_corpus_path, max_tokens=cfg.train_tokens
    )
    n_tokens = train_ids.shape[0]
    log_fn(
        f"  tokenized {n_tokens:,} bytes from {rows:,} rows "
        f"({src_bytes / rows:.1f} bytes/row) in {time.time() - t0:.1f}s"
    )

    if n_tokens < cfg.ctx_len + 2:
        raise RuntimeError(
            f"Train corpus only produced {n_tokens} bytes; "
            f"need at least ctx_len+2={cfg.ctx_len + 2}."
        )

    model = MagnetLM(
        vocab_size=vocab_size,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        d_ff=cfg.d_ff,
        dropout=cfg.dropout,
        pre_layers=cfg.pre_layers,
        shortened_layers=cfg.shortened_layers,
        post_layers=cfg.post_layers,
        ctx_len=cfg.ctx_len,
        boundary_prior=cfg.boundary_prior,
        boundary_temp=cfg.boundary_temp,
        boundary_threshold=cfg.boundary_threshold,
    ).to(device)

    n_params = model.count_parameters()
    log_fn(f"  model: vocab={vocab_size}  params={n_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        betas=(cfg.beta1, cfg.beta2),
        weight_decay=cfg.weight_decay,
    )

    tokens_per_step = cfg.batch_size * cfg.ctx_len
    total_steps = max(1, cfg.train_tokens // tokens_per_step)
    log_fn(f"  training: {total_steps} steps × {tokens_per_step:,} bytes/step")

    gen = torch.Generator().manual_seed(cfg.seed)
    model.train()
    t_start = time.time()
    stdout_every = max(1, total_steps // 20)

    for step in range(total_steps):
        lr = _lr_at_step(step, total_steps, cfg)
        for g in optimizer.param_groups:
            g["lr"] = lr

        x, y = _sample_batch(train_ids, cfg.batch_size, cfg.ctx_len, gen)
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        if amp_dtype is not None:
            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype):
                _, lm_loss, boundary_loss = model(x, y)
        else:
            _, lm_loss, boundary_loss = model(x, y)

        loss = lm_loss + cfg.boundary_lambda * boundary_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        if step == 0 or (step + 1) % stdout_every == 0 or step == total_steps - 1:
            bpb = lm_loss.item() / math.log(2)
            log_fn(
                f"    step {step + 1:>5}/{total_steps}"
                f"  lm_loss={lm_loss.item():.4f}"
                f"  bpb={bpb:.4f}"
                f"  boundary_loss={boundary_loss.item():.4f}"
                f"  lr={lr:.2e}"
            )
        if wandb_run is not None and (step % cfg.log_every == 0 or step == total_steps - 1):
            wandb_run.log(
                {
                    "train/lm_loss": lm_loss.item(),
                    "train/bpb": lm_loss.item() / math.log(2),
                    "train/boundary_loss": boundary_loss.item(),
                    "train/lr": lr,
                    "train/bytes_seen": (step + 1) * tokens_per_step,
                },
                step=step,
            )

    train_seconds = time.time() - t_start
    log_fn(f"  trained in {train_seconds:.1f}s")

    corpus_stats = {
        "n_tokens": int(n_tokens),
        "rows": int(rows),
        "source_bytes": int(src_bytes),
    }
    return model, tokenizer, device, amp_dtype, train_seconds, corpus_stats


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate_bpb(
    model,
    tokenizer,
    corpus_path: Path,
    cfg: MagnetConfig,
    device,
    amp_dtype,
    log_fn=print,
    label: str = "eval",
) -> dict:
    """Compute BPB (and auxiliary stats) on a byte corpus.

    BPB = (sum of per-byte NLL in nats) / (number of bytes) / ln(2).

    For a byte-level model this simplifies to mean_CE / ln(2) because
    each prediction target is one byte.
    """
    import torch

    ids, rows, src_bytes = tokenize_byte_corpus(tokenizer, corpus_path)
    n = ids.shape[0]
    log_fn(
        f"  eval ({label}): {n:,} bytes from {rows:,} rows "
        f"({src_bytes / rows:.1f} bytes/row)"
    )

    if n < 2:
        raise RuntimeError(f"Eval corpus '{label}' too short ({n} bytes).")

    ctx = cfg.ctx_len
    model.eval()
    total_nll = 0.0
    total_targets = 0

    with torch.no_grad():
        i = 0
        while i + ctx + 1 <= n:
            window = torch.from_numpy(ids[i : i + ctx + 1]).long().unsqueeze(0).to(device)
            x_w, y_w = window[:, :-1], window[:, 1:]
            if amp_dtype is not None:
                with torch.amp.autocast(device_type=device.type, dtype=amp_dtype):
                    _, lm_loss, _ = model(x_w, y_w)
            else:
                _, lm_loss, _ = model(x_w, y_w)
            n_tgt = y_w.numel()
            total_nll += lm_loss.item() * n_tgt
            total_targets += n_tgt
            i += ctx

    if total_targets == 0:
        raise RuntimeError(
            f"Eval data too short for even one window of ctx_len={ctx}."
        )

    mean_nll = total_nll / total_targets
    bpb = mean_nll / math.log(2)
    fraction_scored = total_targets / max(1, n - 1)
    scored_bytes = src_bytes * fraction_scored
    scored_rows = rows * fraction_scored

    return {
        "bits_per_byte": bpb,
        "mean_nll_per_byte": mean_nll,
        "eval_bytes_scored": scored_bytes,
        "eval_rows_scored": scored_rows,
        "eval_bytes_per_row": (scored_bytes / scored_rows) if scored_rows else 0.0,
    }


def evaluate_bpb_on_sentences(
    model,
    tokenizer,
    sentences: list[str],
    cfg: MagnetConfig,
    device,
    amp_dtype,
    label: str = "sentences",
    log_fn=print,
) -> dict:
    """Score a list of sentences (e.g. FLORES devtest) under a MAGNET model."""
    cleaned = [s.strip() for s in sentences if s and s.strip()]
    if not cleaned:
        raise RuntimeError(f"No non-empty sentences in '{label}'.")

    joined = " ".join(cleaned)
    ids = np.asarray(_encode_line(tokenizer, joined), dtype=np.int32)
    if ids.shape[0] == 0:
        raise RuntimeError(f"Tokenizer produced 0 ids for '{label}'.")

    log_fn(
        f"  eval ({label}): {ids.shape[0]:,} bytes, {len(cleaned)} sentences"
    )

    ctx = cfg.ctx_len
    model.eval()
    import torch

    total_nll = 0.0
    total_targets = 0
    n = ids.shape[0]

    with torch.no_grad():
        i = 0
        while i + ctx + 1 <= n:
            window = torch.from_numpy(ids[i : i + ctx + 1]).long().unsqueeze(0).to(device)
            x_w, y_w = window[:, :-1], window[:, 1:]
            if amp_dtype is not None:
                with torch.amp.autocast(device_type=device.type, dtype=amp_dtype):
                    _, lm_loss, _ = model(x_w, y_w)
            else:
                _, lm_loss, _ = model(x_w, y_w)
            n_tgt = y_w.numel()
            total_nll += lm_loss.item() * n_tgt
            total_targets += n_tgt
            i += ctx

    if total_targets == 0:
        raise RuntimeError(
            f"Eval data too short for even one window of ctx_len={ctx}."
        )

    mean_nll = total_nll / total_targets
    bpb = mean_nll / math.log(2)
    src_bytes = sum(len(s.encode("utf-8")) for s in cleaned)
    fraction_scored = total_targets / max(1, n - 1)

    return {
        "bits_per_byte": bpb,
        "mean_nll_per_byte": mean_nll,
        "eval_bytes_scored": src_bytes * fraction_scored,
        "eval_rows_scored": len(cleaned) * fraction_scored,
    }


# ---------------------------------------------------------------------------
# Top-level: train + eval one MAGNET run
# ---------------------------------------------------------------------------

# Maps our language codes to FLORES-200 configs (same as llm_training.py).
FLORES_CONFIGS: dict[str, str] = {
    "en": "eng_Latn",
    "zh": "zho_Hans",
    "tr": "tur_Latn",
    "ru": "rus_Cyrl",
    "hi": "hin_Deva",
}


def _load_flores_devtest(language: str) -> list[str]:
    if language not in FLORES_CONFIGS:
        raise ValueError(
            f"No FLORES config for {language!r}. Known: {sorted(FLORES_CONFIGS)}"
        )
    from datasets import load_dataset

    ds = load_dataset("facebook/flores", FLORES_CONFIGS[language], split="devtest")
    return [row["sentence"] for row in ds]


def train_and_evaluate_magnet(
    train_corpus_path: Path,
    eval_corpus_path: Path,
    cfg: MagnetConfig,
    language: str | None = None,
    eval_flores: bool = True,
    log_fn=print,
    wandb_extra_config: dict | None = None,
) -> dict:
    """Train a MAGNET model then score it on the test set and (optionally) FLORES.

    Returns a dict with ``test_*`` and ``flores_*`` keys.
    """
    wandb_run = None
    if cfg.wandb_project:
        import wandb

        init_cfg = asdict(cfg)
        if wandb_extra_config:
            init_cfg.update(wandb_extra_config)
        wandb_run = wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            name=cfg.wandb_run_name,
            tags=cfg.wandb_tags or None,
            config=init_cfg,
            reinit=True,
        )

    try:
        model, tokenizer, device, amp_dtype, train_seconds, corpus_stats = train_magnet(
            train_corpus_path, cfg, log_fn=log_fn, wandb_run=wandb_run
        )

        out: dict = {}

        test_metrics = evaluate_bpb(
            model, tokenizer, eval_corpus_path, cfg, device, amp_dtype,
            log_fn=log_fn, label="test",
        )
        for k, v in test_metrics.items():
            out[f"test_{k}"] = v

        if eval_flores and language is not None:
            try:
                sents = _load_flores_devtest(language)
            except Exception as exc:
                log_fn(f"  FLORES eval skipped: {exc}")
            else:
                flores_metrics = evaluate_bpb_on_sentences(
                    model, tokenizer, sents, cfg, device, amp_dtype,
                    label=f"flores/{FLORES_CONFIGS[language]}", log_fn=log_fn,
                )
                for k, v in flores_metrics.items():
                    out[f"flores_{k}"] = v

        out.update(
            {
                "train_bytes": corpus_stats["n_tokens"],
                "train_rows": corpus_stats["rows"],
                "train_source_bytes": corpus_stats["source_bytes"],
                "param_count": model.count_parameters(),
                "train_seconds": train_seconds,
                "config": asdict(cfg),
            }
        )

        if wandb_run is not None:
            for k, v in out.items():
                if isinstance(v, (int, float)):
                    wandb_run.summary[k] = v

        return out

    finally:
        if wandb_run is not None:
            wandb_run.finish()
