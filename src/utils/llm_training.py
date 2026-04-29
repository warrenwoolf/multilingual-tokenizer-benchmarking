"""Train a small (~50M-param) GPT-style LM per tokenizer and evaluate perplexity.

This is the *downstream* tokenizer evaluation: train a fixed-architecture LM
with each candidate tokenizer, then measure how well it predicts held-out text.
The intent is to predict (cheaply) which tokenizer would win in a much larger
training run.

Cross-tokenizer comparison
--------------------------
Per-token perplexity is *not* comparable across different tokenizers, because
each tokenizer defines a different distribution support and segments the same
text into a different number of tokens. We therefore also report
**bits-per-byte (BPB)**: total cross-entropy on the eval set, normalized by
the raw UTF-8 byte count of that eval set. BPB is invariant to the tokenizer
choice and is the standard fair-comparison metric (Gao et al., MEGA, etc.).

    bpb = (sum_ce_nats / total_eval_bytes) / ln(2)

Compute budget
--------------
We follow the user spec and fix the **training-token budget** (e.g. 50M tokens)
across all tokenizers. This is intentionally imperfect: a tokenizer with high
fertility sees less *content* for the same token count, and a tokenizer with
larger vocab has a bigger embedding table (so more parameters and FLOPs per
token). We document this caveat rather than try to fix it. In a fairer setup
you'd fix wall-clock or training-FLOPs; here we trade rigor for reproducibility.

Architecture
------------
Plain pre-LN decoder-only transformer (GPT-2 style):
    - token + learned positional embeddings
    - causal self-attention via F.scaled_dot_product_attention
    - GELU MLP, weight-tied LM head
    - LayerNorm (not RMSNorm) for portability across torch versions

Default size: d_model=512, n_layers=8, n_heads=8, d_ff=2048, ctx_len=512.
Total params depend on vocab size (the embedding table dominates):
    vocab=8k  -> ~30M    vocab=32k -> ~42M    vocab=64k -> ~58M
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterator


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class LLMConfig:
    """Architecture + training hyperparameters.

    Defaults target ~50M params at 32k vocab and run in well under an hour
    on a single modern GPU. Override for smoke tests or bigger sweeps.
    """

    # Architecture
    d_model: int = 512
    n_layers: int = 8
    n_heads: int = 8
    d_ff: int = 2048
    ctx_len: int = 512
    dropout: float = 0.0

    # Training
    train_tokens: int = 50_000_000  # fixed token budget (per user spec)
    batch_size: int = 32
    learning_rate: float = 3e-4
    min_lr: float = 3e-5
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    warmup_steps: int = 100
    eval_every: int = 0  # 0 = only at end

    # Misc
    seed: int = 0
    device: str = "auto"  # "auto" -> cuda if available else cpu
    dtype: str = "auto"  # "auto" -> bf16 on CUDA else fp32


def resolve_device(spec: str):
    import torch

    if spec != "auto":
        return torch.device(spec)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_amp_dtype(spec: str, device) -> "torch.dtype | None":
    """Return autocast dtype, or None to disable autocast."""
    import torch

    if spec == "auto":
        if device.type == "cuda" and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return None
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": None}[spec]


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


def _build_model(vocab_size: int, cfg: LLMConfig):
    """Construct the GPT model. Imported lazily so torch is only required when used."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class CausalSelfAttention(nn.Module):
        def __init__(self):
            super().__init__()
            assert cfg.d_model % cfg.n_heads == 0
            self.n_heads = cfg.n_heads
            self.d_head = cfg.d_model // cfg.n_heads
            self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
            self.proj = nn.Linear(cfg.d_model, cfg.d_model)
            self.dropout = cfg.dropout

        def forward(self, x):
            B, T, C = x.shape
            qkv = self.qkv(x).view(B, T, 3, self.n_heads, self.d_head)
            q, k, v = qkv.unbind(dim=2)
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            out = F.scaled_dot_product_attention(
                q, k, v,
                is_causal=True,
                dropout_p=self.dropout if self.training else 0.0,
            )
            out = out.transpose(1, 2).contiguous().view(B, T, C)
            return self.proj(out)

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.ln1 = nn.LayerNorm(cfg.d_model)
            self.attn = CausalSelfAttention()
            self.ln2 = nn.LayerNorm(cfg.d_model)
            self.mlp = nn.Sequential(
                nn.Linear(cfg.d_model, cfg.d_ff),
                nn.GELU(),
                nn.Linear(cfg.d_ff, cfg.d_model),
            )

        def forward(self, x):
            x = x + self.attn(self.ln1(x))
            x = x + self.mlp(self.ln2(x))
            return x

    class GPT(nn.Module):
        def __init__(self):
            super().__init__()
            self.tok_emb = nn.Embedding(vocab_size, cfg.d_model)
            self.pos_emb = nn.Embedding(cfg.ctx_len, cfg.d_model)
            self.drop = nn.Dropout(cfg.dropout)
            self.blocks = nn.ModuleList([Block() for _ in range(cfg.n_layers)])
            self.ln_f = nn.LayerNorm(cfg.d_model)
            self.lm_head = nn.Linear(cfg.d_model, vocab_size, bias=False)
            # Weight tying: lm_head shares the embedding matrix.
            self.lm_head.weight = self.tok_emb.weight
            self.apply(self._init_weights)

        @staticmethod
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)

        def forward(self, idx, targets=None):
            B, T = idx.shape
            pos = torch.arange(T, device=idx.device)
            x = self.drop(self.tok_emb(idx) + self.pos_emb(pos)[None])
            for blk in self.blocks:
                x = blk(x)
            x = self.ln_f(x)
            logits = self.lm_head(x)
            if targets is None:
                return logits, None
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                reduction="mean",
            )
            return logits, loss

    return GPT()


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters())


# ---------------------------------------------------------------------------
# Tokenization helpers
# ---------------------------------------------------------------------------


def _iter_lines(path: Path) -> Iterator[str]:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield line


def tokenize_corpus(
    tokenizer,
    corpus_path: Path,
    max_tokens: int | None = None,
    eos_id: int | None = None,
) -> "numpy.ndarray":
    """Tokenize a text corpus into a 1-D int32 array of token ids.

    Reads line-by-line so the source text never lives fully in memory; stops
    early once ``max_tokens`` ids have been collected. ``eos_id`` is appended
    after each document if provided, separating documents in the packed stream.
    """
    import numpy as np

    chunks: list[np.ndarray] = []
    total = 0
    for line in _iter_lines(corpus_path):
        ids = tokenizer.encode(line)
        if eos_id is not None:
            ids = ids + [eos_id]
        if not ids:
            continue
        if max_tokens is not None and total + len(ids) > max_tokens:
            ids = ids[: max_tokens - total]
            if ids:
                chunks.append(np.asarray(ids, dtype=np.int32))
                total += len(ids)
            break
        chunks.append(np.asarray(ids, dtype=np.int32))
        total += len(ids)
    if not chunks:
        return np.zeros(0, dtype=np.int32)
    return np.concatenate(chunks)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def _lr_at_step(step: int, total_steps: int, cfg: LLMConfig) -> float:
    if step < cfg.warmup_steps:
        return cfg.learning_rate * (step + 1) / max(1, cfg.warmup_steps)
    if step >= total_steps:
        return cfg.min_lr
    progress = (step - cfg.warmup_steps) / max(1, total_steps - cfg.warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return cfg.min_lr + (cfg.learning_rate - cfg.min_lr) * coeff


def _sample_batch(token_array, batch_size: int, ctx_len: int, generator):
    """Randomly sample ``batch_size`` (input, target) sequences of length ctx_len."""
    import numpy as np
    import torch

    n = token_array.shape[0]
    # Each sample needs ctx_len + 1 contiguous tokens (input + shifted target).
    high = n - ctx_len - 1
    if high <= 0:
        raise RuntimeError(
            f"Token stream too short ({n} tokens) for ctx_len={ctx_len}; "
            "increase the training corpus or shrink ctx_len."
        )
    starts = torch.randint(0, high, (batch_size,), generator=generator).numpy()
    inputs = np.stack([token_array[s : s + ctx_len] for s in starts])
    targets = np.stack([token_array[s + 1 : s + 1 + ctx_len] for s in starts])
    x = torch.from_numpy(inputs).long()
    y = torch.from_numpy(targets).long()
    return x, y


def train_lm(
    tokenizer,
    train_corpus_path: Path,
    cfg: LLMConfig,
    log_fn=print,
):
    """Train a GPT on tokenized ``train_corpus_path`` for cfg.train_tokens tokens.

    Returns the trained model + the device + the autocast dtype.
    """
    import numpy as np
    import torch

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = resolve_device(cfg.device)
    amp_dtype = resolve_amp_dtype(cfg.dtype, device)

    log_fn(f"  device={device} amp_dtype={amp_dtype}")
    log_fn(f"  tokenizing train corpus (cap = {cfg.train_tokens:,} tokens) ...")
    t0 = time.time()
    train_ids = tokenize_corpus(
        tokenizer,
        train_corpus_path,
        max_tokens=cfg.train_tokens,
    )
    log_fn(f"  tokenized {train_ids.shape[0]:,} train tokens in {time.time() - t0:.1f}s")

    if train_ids.shape[0] < cfg.ctx_len + 2:
        raise RuntimeError(
            f"Train corpus only produced {train_ids.shape[0]} tokens; "
            f"need at least ctx_len+2={cfg.ctx_len + 2}."
        )

    vocab_size = tokenizer.vocab_size
    model = _build_model(vocab_size, cfg).to(device)
    log_fn(f"  model: vocab={vocab_size:,} params={count_parameters(model):,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        betas=(cfg.beta1, cfg.beta2),
        weight_decay=cfg.weight_decay,
    )

    tokens_per_step = cfg.batch_size * cfg.ctx_len
    total_steps = max(1, cfg.train_tokens // tokens_per_step)
    log_fn(f"  training: {total_steps} steps × {tokens_per_step:,} tokens/step")

    gen = torch.Generator().manual_seed(cfg.seed)
    model.train()
    t_start = time.time()
    for step in range(total_steps):
        lr = _lr_at_step(step, total_steps, cfg)
        for g in optimizer.param_groups:
            g["lr"] = lr
        x, y = _sample_batch(train_ids, cfg.batch_size, cfg.ctx_len, gen)
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        if amp_dtype is not None:
            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype):
                _, loss = model(x, y)
        else:
            _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        if step == 0 or (step + 1) % max(1, total_steps // 10) == 0 or step == total_steps - 1:
            log_fn(f"    step {step + 1:>5}/{total_steps}  loss={loss.item():.4f}  lr={lr:.2e}")

    train_seconds = time.time() - t_start
    log_fn(f"  trained in {train_seconds:.1f}s")
    return model, device, amp_dtype, train_seconds


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate_perplexity(
    model,
    tokenizer,
    eval_corpus_path: Path,
    cfg: LLMConfig,
    device,
    amp_dtype,
    log_fn=print,
) -> dict:
    """Compute per-token perplexity and bits-per-byte on ``eval_corpus_path``.

    Bits-per-byte (BPB) is the cross-tokenizer-comparable metric. It uses the
    raw UTF-8 byte count of the eval text in the denominator — independent of
    how the tokenizer chose to segment it.
    """
    import torch

    eval_ids = tokenize_corpus(tokenizer, eval_corpus_path, max_tokens=None)
    eval_bytes = eval_corpus_path.stat().st_size  # raw UTF-8 byte count
    log_fn(f"  eval: {eval_ids.shape[0]:,} tokens, {eval_bytes:,} bytes")

    if eval_ids.shape[0] < 2:
        raise RuntimeError("Eval corpus has fewer than 2 tokens; cannot compute perplexity.")

    model.eval()
    ctx = cfg.ctx_len
    total_loss_sum = 0.0
    total_targets = 0
    with torch.no_grad():
        # Non-overlapping windows of length ctx+1 (last token shifts to target).
        # Simple, fast, and within ~5% of sliding-window PPL for ctx=512.
        i = 0
        while i + ctx + 1 <= eval_ids.shape[0]:
            window = torch.from_numpy(eval_ids[i : i + ctx + 1]).long().unsqueeze(0).to(device)
            x = window[:, :-1]
            y = window[:, 1:]
            if amp_dtype is not None:
                with torch.amp.autocast(device_type=device.type, dtype=amp_dtype):
                    _, loss = model(x, y)
            else:
                _, loss = model(x, y)
            n = y.numel()
            total_loss_sum += loss.item() * n
            total_targets += n
            i += ctx

    if total_targets == 0:
        raise RuntimeError(
            f"Eval corpus too short for one full window of ctx_len={ctx}."
        )

    mean_nll_per_token = total_loss_sum / total_targets
    perplexity = math.exp(mean_nll_per_token)

    # BPB: total CE in nats / total eval bytes / ln(2).
    # We score `total_targets` tokens, but the eval file may contain more bytes
    # than those tokens cover (we drop a tail < ctx). Scale eval_bytes to the
    # fraction of tokens we actually scored so BPB is per-byte-of-scored-text.
    fraction_scored = total_targets / max(1, eval_ids.shape[0] - 1)
    scored_bytes = eval_bytes * fraction_scored
    bits_per_byte = (total_loss_sum / scored_bytes) / math.log(2) if scored_bytes > 0 else float("nan")

    return {
        "perplexity": perplexity,
        "mean_nll_per_token": mean_nll_per_token,
        "bits_per_byte": bits_per_byte,
        "eval_tokens_scored": total_targets,
        "eval_bytes_scored": scored_bytes,
    }


# ---------------------------------------------------------------------------
# Top-level: train + eval one tokenizer
# ---------------------------------------------------------------------------


def train_and_evaluate(
    tokenizer,
    train_corpus_path: Path,
    eval_corpus_path: Path,
    cfg: LLMConfig,
    log_fn=print,
) -> dict:
    """Train an LM with `tokenizer` then return perplexity + BPB on the eval set."""
    model, device, amp_dtype, train_seconds = train_lm(
        tokenizer, train_corpus_path, cfg, log_fn=log_fn
    )
    metrics = evaluate_perplexity(
        model, tokenizer, eval_corpus_path, cfg, device, amp_dtype, log_fn=log_fn
    )
    metrics["param_count"] = count_parameters(model)
    metrics["train_seconds"] = train_seconds
    metrics["config"] = asdict(cfg)
    return metrics
