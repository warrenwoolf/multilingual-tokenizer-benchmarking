"""Train a small (~50M-param) GPT-style LM per tokenizer and evaluate perplexity.

This is the *downstream* tokenizer evaluation: train a fixed-architecture LM
with each candidate tokenizer, then measure how well it predicts held-out text.
The intent is to predict (cheaply) which tokenizer would win in a much larger
training run.

Each LM is **monolingual**: it's trained on a single language's corpus with
a tokenizer that was itself trained on that same language. The artifact-naming
scheme (``{lang}_{algo}_v{vocab}``) enforces this — there's no cross-language
mixing in the training pipeline.

Cross-tokenizer comparison
--------------------------
Per-token perplexity is *not* comparable across different tokenizers, because
each tokenizer defines a different distribution support and segments the same
text into a different number of tokens. We report
**bits-per-byte (BPB)** instead: total cross-entropy on the eval set,
normalized by the raw UTF-8 byte count of that eval set. BPB is invariant to
the tokenizer choice and is the standard fair-comparison metric.

    bpb = (sum_ce_nats / total_eval_bytes) / ln(2)

Two eval sets
-------------
1. **In-domain held-out** — ``data/{lang}/eval.txt`` from the same FineWeb
   distribution as training.
2. **FLORES-200 devtest** — out-of-distribution generalization benchmark
   (clean, professionally translated). Loaded on demand; see ``FLORES_CONFIGS``.

Compute budget
--------------
We follow the user spec and fix the **training-token budget** across all
tokenizers (default 1B, ~Chinchilla-optimal for 50M params). This is
intentionally imperfect: a tokenizer with high fertility sees less *content*
for the same token count, and a tokenizer with larger vocab has a bigger
embedding table (so more parameters and FLOPs per token). We document this
caveat rather than try to fix it.

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

Wall-clock estimate: A100 40GB at ~250K tok/s bf16 -> ~65 min per 1B-token run.
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Iterator


# ---------------------------------------------------------------------------
# FLORES-200 language code map
# ---------------------------------------------------------------------------
# Maps our internal codes to FLORES-200 ``{iso639-3}_{script}`` configs.
# Mandarin: FineWeb 2 uses ``cmn_Hani`` (script-agnostic); FLORES-200 distinguishes
# Simplified (zho_Hans) and Traditional (zho_Hant). zho_Hans is the closer match.

FLORES_CONFIGS: dict[str, str] = {
    "en": "eng_Latn",
    "zh": "zho_Hans",
    "hu": "hun_Latn",
    "ru": "rus_Cyrl",
    "hi": "hin_Deva",
}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class LLMConfig:
    """Architecture + training hyperparameters.

    Defaults target ~50M params at 32k vocab. The default 1B-token budget
    is Chinchilla-optimal for that size and runs in roughly an hour on a
    single A100 40GB. Override for smoke tests or bigger sweeps.
    """

    # Architecture
    d_model: int = 512
    n_layers: int = 8
    n_heads: int = 8
    d_ff: int = 2048
    ctx_len: int = 512
    dropout: float = 0.0

    # Training
    train_tokens: int = 1_000_000_000  # ~Chinchilla-optimal for 50M params
    batch_size: int = 128  # A100 40GB with FlashAttention (F.sdpa); ~64K tok/step
    learning_rate: float = 5e-4  # sqrt-scaled from 3e-4 @ bs=32: 3e-4 * sqrt(128/32) = 6e-4
    min_lr: float = 5e-5
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    warmup_steps: int = 100
    # Number of evenly-spaced evals to run *during* training (not counting the
    # pre-training random-init eval or the final post-training eval).  The
    # actual step interval is computed as max(1, total_steps // n_mid_evals) so
    # it scales automatically with batch size and vocab-size adjustments.
    # Set to 0 to disable mid-training eval entirely.
    n_mid_evals: int = 5

    # Misc
    seed: int = 0
    device: str = "auto"  # "auto" -> cuda if available else cpu
    dtype: str = "auto"  # "auto" -> bf16 on CUDA else fp32
    log_every: int = 50   # step interval for stdout + wandb training-loss logging

    # Weights & Biases (optional). When ``wandb_project`` is None, no W&B
    # calls are made and wandb is not imported.
    wandb_project: str | None = None
    wandb_entity: str | None = None
    wandb_run_name: str | None = None  # filled in by orchestrator per artifact
    wandb_tags: list[str] = field(default_factory=list)
    # If True (and wandb is enabled), upload the tokenizer artifact dir at run
    # start and the trained model state_dict at run end. Each model is ~200MB
    # at 50M params; turn this off for large sweeps if storage is a concern.
    wandb_log_tokenizer_artifact: bool = True
    wandb_log_model_artifact: bool = True


# Where to look for a W&B API key on disk if WANDB_API_KEY is not set.
WANDB_TOKEN_PATH = Path("tokens") / "wandb.token"


def _ensure_wandb_login() -> None:
    """If WANDB_API_KEY is unset, populate it from ``tokens/wandb.token``.

    Falls through silently if neither is available — wandb.init will then
    prompt or fail, depending on its mode setting.
    """
    if os.environ.get("WANDB_API_KEY"):
        return
    if WANDB_TOKEN_PATH.exists():
        key = WANDB_TOKEN_PATH.read_text(encoding="utf-8").strip()
        if key:
            os.environ["WANDB_API_KEY"] = key


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


def resolve_eos_id(tokenizer) -> int | None:
    """Best-effort EOS id lookup across adapter types.

    Returns None when no EOS-like token is available.
    """
    # Prefer canonical EOS spellings used across our adapters.
    for tok in ("</s>", "<|endoftext|>", "<eos>"):
        try:
            tid = tokenizer.token_to_id(tok)
        except Exception:
            tid = None
        if tid is not None:
            return int(tid)

    # Fallback for transformer-backed adapters exposing eos_token_id.
    tok_obj = getattr(tokenizer, "_tok", None)
    eos_tid = getattr(tok_obj, "eos_token_id", None)
    if eos_tid is not None:
        return int(eos_tid)
    return None


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


def _build_hf_model(vocab_size: int, cfg: LLMConfig):
    """Build a HuggingFace GPT2-style model matching our `LLMConfig`.

    Imported lazily so `transformers` is optional.
    """
    from transformers import GPT2Config, GPT2LMHeadModel

    hf_cfg = GPT2Config(
        vocab_size=vocab_size,
        n_embd=cfg.d_model,
        n_layer=cfg.n_layers,
        n_head=cfg.n_heads,
        n_positions=cfg.ctx_len,
        resid_pdrop=cfg.dropout,
        embd_pdrop=cfg.dropout,
        attn_pdrop=cfg.dropout,
    )
    return GPT2LMHeadModel(hf_cfg)


class HFWrapper:
    """Lightweight wrapper around a HF model exposing the small API used
    elsewhere in this module: `.to()`, `.eval()`, `.train()`, `.state_dict()`
    and `forward(idx, targets=None)` returning `(logits, loss)`.
    This avoids importing `torch` at module import time.
    """
    def __init__(self, hf_model):
        self.hf = hf_model

    def to(self, *args, **kwargs):
        self.hf.to(*args, **kwargs)
        return self

    def eval(self):
        self.hf.eval()

    def train(self, mode=True):
        self.hf.train(mode)

    def state_dict(self):
        return self.hf.state_dict()

    def parameters(self):
        return self.hf.parameters()

    def __call__(self, idx, targets=None):
        return self.forward(idx, targets)

    def forward(self, idx, targets=None):
        if targets is None:
            outputs = self.hf(input_ids=idx)
            return outputs.logits, None
        outputs = self.hf(input_ids=idx, labels=targets)
        return outputs.logits, outputs.loss


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters())


def _sample_batch(*args, **kwargs):
    """Removed: manual batch sampling was replaced by HF Trainer's DataLoader."""
    raise RuntimeError("Manual sampling is removed")


# ---------------------------------------------------------------------------
# Tokenization helpers
# ---------------------------------------------------------------------------


def _iter_lines(path: Path) -> Iterator[str]:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield line


@dataclass
class TokenizedCorpus:
    """A tokenized corpus along with the row + byte counts it was built from.

    The byte count (``source_bytes``) is the raw UTF-8 byte length of the
    *consumed* portion of the source text — i.e. it accounts for early stops
    on ``max_tokens``. This is what BPB normalises by, and dividing by
    ``rows`` gives the bytes-per-row sanity-check indicator.
    """

    ids: object  # numpy.ndarray of int32, kept opaque to avoid eager import
    rows: int
    source_bytes: int

    @property
    def n_tokens(self) -> int:
        return int(self.ids.shape[0])

    @property
    def bytes_per_row(self) -> float:
        return self.source_bytes / self.rows if self.rows else 0.0

    @property
    def tokens_per_row(self) -> float:
        return self.n_tokens / self.rows if self.rows else 0.0


def tokenize_corpus(
    tokenizer,
    corpus_path: Path,
    max_tokens: int | None = None,
    eos_id: int | None = None,
) -> TokenizedCorpus:
    """Tokenize a text corpus into ids + row/byte accounting.

    Reads line-by-line so the source text never lives fully in memory; stops
    early once ``max_tokens`` ids have been collected. ``eos_id`` is appended
    after each document if provided, separating documents in the packed stream.

    The byte count is summed from the consumed lines (UTF-8) so callers can
    log a ``bytes_per_row`` indicator that's tied to what was actually used,
    not to the on-disk file size.
    """
    import numpy as np

    chunks: list[np.ndarray] = []
    total_tokens = 0
    rows = 0
    src_bytes = 0
    for line in _iter_lines(corpus_path):
        text_ids = tokenizer.encode(line)
        if not text_ids:
            continue
        ids = text_ids
        if eos_id is not None:
            ids = ids + [eos_id]
        if max_tokens is not None and total_tokens + len(ids) > max_tokens:
            ids = ids[: max_tokens - total_tokens]
            if ids:
                chunks.append(np.asarray(ids, dtype=np.int32))
                total_tokens += len(ids)
                rows += 1
                line_bytes = len(line.encode("utf-8"))
                if eos_id is None:
                    kept_text_tokens = len(ids)
                else:
                    kept_text_tokens = min(len(text_ids), len(ids))
                frac = kept_text_tokens / max(1, len(text_ids))
                src_bytes += int(round(line_bytes * frac))
            break
        chunks.append(np.asarray(ids, dtype=np.int32))
        total_tokens += len(ids)
        rows += 1
        src_bytes += len(line.encode("utf-8"))
    arr = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.int32)
    return TokenizedCorpus(ids=arr, rows=rows, source_bytes=src_bytes)


def _tokenize_sentences_to_corpus(
    tokenizer,
    sentences: list[str],
    eos_id: int | None,  # unused; kept for call-site compatibility
) -> TokenizedCorpus:
    """Tokenize a list of sentences into a single ``TokenizedCorpus``.

    Sentences are joined with a single space and tokenized as one continuous
    stream. This keeps source_bytes exactly equal to the bytes of text that the
    model scores, so BPB's denominator is correct. Per-sentence EOS injection
    is intentionally omitted: the mismatch between EOS tokens in the numerator
    and no corresponding bytes in the denominator would inflate BPB for
    high-fertility tokenizers (which have more total tokens, so EOS predictions
    are a smaller fraction of the loss, skewing intra-language comparisons).
    """
    import numpy as np

    cleaned = [s.strip() for s in sentences if s and s.strip()]
    if not cleaned:
        raise RuntimeError("No non-empty sentences to tokenize.")
    joined = " ".join(cleaned)
    all_ids = tokenizer.encode(joined)
    if not all_ids:
        raise RuntimeError("Tokenizer produced 0 ids for sentence list.")
    return TokenizedCorpus(
        ids=np.asarray(all_ids, dtype=np.int32),
        rows=len(cleaned),
        source_bytes=len(joined.encode("utf-8")),
    )


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------


def _flops(param_count: int, tokens: int) -> int:
    """Approximate training FLOPs via the Chinchilla 6·N·D rule."""
    return 6 * param_count * tokens


def _save_checkpoint_to_path(model, cfg: LLMConfig, vocab_size: int, path: Path) -> None:
    """Save model state dict + config to *path* (created if missing)."""
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "vocab_size": vocab_size,
            "config": asdict(cfg),
        },
        path,
    )


def _upload_checkpoint_artifact(
    wandb_run,
    path: Path,
    artifact_name: str,
    step: int,
) -> None:
    """Upload *path* as a new version of the named W&B model artifact.

    Raises on any failure — callers must not silently swallow this.
    Each call creates a new artifact version; W&B assigns v0, v1, … automatically.
    """
    import wandb

    art = wandb.Artifact(
        name=artifact_name,
        type="model",
        metadata={"step": step},
    )
    art.add_file(str(path), name=path.name)
    wandb_run.log_artifact(art)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


class _WindowDataset:
    """A lightweight Dataset that returns dicts {'input_ids', 'labels'}
    for Trainer. Windows are materialized on demand to keep memory usage low.
    """
    def __init__(self, token_array, starts, ctx_len: int):
        # token_array: numpy.ndarray
        # starts: numpy.ndarray of start indices
        self.token_array = token_array
        self.starts = starts
        self.ctx_len = int(ctx_len)

    def __len__(self):
        return int(self.starts.shape[0])

    def __getitem__(self, idx):
        s = int(self.starts[idx])
        arr = self.token_array
        x = arr[s : s + self.ctx_len]
        y = arr[s + 1 : s + 1 + self.ctx_len]
        import torch

        return {"input_ids": torch.from_numpy(x).long(), "labels": torch.from_numpy(y).long()}


def train_lm_transformers(
    tokenizer,
    train_corpus_path: Path,
    cfg: LLMConfig,
    log_fn=print,
    wandb_run=None,
    *,
    eval_corpus: "TokenizedCorpus | None" = None,
    flores_corpus: "TokenizedCorpus | None" = None,
    checkpoint_dir: "Path | None" = None,
    artifact_name: "str | None" = None,
    run_timestamp: "str | None" = None,
    vocab_size: "int | None" = None,
):
    """Train using HuggingFace ``Trainer`` and return the same result tuple
    as ``train_lm()`` for compatibility.

    When ``eval_corpus`` is provided, evaluates the randomly-initialised model
    at step 0, then fires ``cfg.n_mid_evals`` evenly-spaced evals during
    training, logging rich metrics to W&B and saving checkpoints to
    ``checkpoint_dir``.
    """
    import numpy as np
    import torch
    from transformers import TrainingArguments, Trainer, TrainerCallback

    log_fn("  using HuggingFace Trainer path")

    # ------------------------------------------------------------------
    # Tokenize training corpus
    # ------------------------------------------------------------------
    eos_id = resolve_eos_id(tokenizer)
    train_corpus = tokenize_corpus(tokenizer, train_corpus_path, max_tokens=cfg.train_tokens, eos_id=eos_id)
    train_ids = train_corpus.ids

    if train_corpus.n_tokens < cfg.ctx_len + 2:
        raise RuntimeError(
            f"Train corpus only produced {train_corpus.n_tokens} tokens; "
            f"need at least ctx_len+2={cfg.ctx_len + 2}."
        )

    tokens_per_step = cfg.batch_size * cfg.ctx_len
    total_steps = max(1, cfg.train_tokens // tokens_per_step)
    log_fn(f"  training (HF Trainer): {total_steps} steps × {tokens_per_step:,} tokens/step")

    # Pre-compute per-token scaling factors for progress metrics.
    _bytes_per_token = train_corpus.source_bytes / max(1, train_corpus.n_tokens)
    _rows_per_token = train_corpus.rows / max(1, train_corpus.n_tokens)

    n = train_ids.shape[0]
    high = n - cfg.ctx_len
    if high <= 0:
        raise RuntimeError("Token stream too short for ctx_len")
    total_samples = total_steps * cfg.batch_size
    rng = np.random.RandomState(cfg.seed)
    starts = rng.randint(0, high, size=total_samples, dtype=np.int64)

    ds = _WindowDataset(train_ids, starts, cfg.ctx_len)

    hf_model = _build_hf_model(tokenizer.vocab_size, cfg)
    _param_count = count_parameters(hf_model)
    _vocab_size = vocab_size if vocab_size is not None else tokenizer.vocab_size

    device = resolve_device(cfg.device)
    amp_dtype = resolve_amp_dtype(cfg.dtype, device)

    import tempfile
    outdir = tempfile.mkdtemp(prefix="hf-trainer-")
    fp16 = amp_dtype is not None and amp_dtype.__name__ == "float16" if hasattr(amp_dtype, "__name__") else False
    try:
        import torch as _t
        fp16 = amp_dtype is not None and amp_dtype == _t.float16
    except Exception:
        pass

    # We handle all W&B logging ourselves so the Trainer doesn't double-log.
    training_args = TrainingArguments(
        output_dir=outdir,
        per_device_train_batch_size=cfg.batch_size,
        max_steps=total_steps,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        warmup_steps=cfg.warmup_steps,
        # Log ~100 times per run regardless of length; respect log_every as a cap.
        logging_steps=max(1, min(cfg.log_every, total_steps // 100)),
        remove_unused_columns=False,
        fp16=fp16,
        gradient_accumulation_steps=1,
        optim="adamw_torch",
        report_to=[],  # all W&B logging is handled by our callbacks
    )

    # Eval interval: divide training evenly so exactly n_mid_evals checkpoints
    # land during training regardless of how batch size was scaled for vocab.
    _eval_every = max(1, total_steps // cfg.n_mid_evals) if cfg.n_mid_evals > 0 else 0

    if _eval_every > 0 and eval_corpus is not None:
        log_fn(
            f"  eval schedule: step 0 (init) + every {_eval_every} steps "
            f"× {cfg.n_mid_evals} = {1 + cfg.n_mid_evals} callback evals total"
        )

    t_start = time.time()

    # ------------------------------------------------------------------
    # Shared eval + checkpoint logic (called at step 0 and in the callback)
    # ------------------------------------------------------------------

    def _run_eval_at_step(hf_model_ref, step: int, dev, amp) -> None:
        """Evaluate model, save checkpoint, log to W&B.  Raises on hard errors."""
        if eval_corpus is None:
            return

        wrapped = HFWrapper(hf_model_ref)
        label = "init" if step == 0 else f"{step}/{total_steps}"

        t_eval = time.time()
        try:
            test_m = _score_corpus(wrapped, eval_corpus, cfg, dev, amp)
        except Exception as exc:
            log_fn(f"  [eval {label}] test scoring failed: {exc}")
            raise

        flores_m: dict = {}
        if flores_corpus is not None:
            try:
                flores_m = _score_corpus(wrapped, flores_corpus, cfg, dev, amp)
            except Exception as exc:
                log_fn(f"  [eval {label}] flores scoring failed: {exc}")

        eval_time = time.time() - t_eval
        tokens_seen = step * tokens_per_step
        wall_time = time.time() - t_start
        flops = _flops(_param_count, tokens_seen)

        test_bpb = test_m.get("bits_per_byte", float("nan"))
        flores_bpb = flores_m.get("bits_per_byte", float("nan"))
        log_fn(
            f"  [eval {label}]"
            f"  test_bpb={test_bpb:.4f}"
            + (f"  flores_bpb={flores_bpb:.4f}" if flores_m else "")
            + f"  eval_time={eval_time:.1f}s"
        )

        if checkpoint_dir is not None:
            ckpt_path = Path(checkpoint_dir) / f"step_{step:07d}.pt"
            _save_checkpoint_to_path(wrapped, cfg, _vocab_size, ckpt_path)
            log_fn(f"  [eval {label}] checkpoint saved → {ckpt_path}")
            if wandb_run is not None and artifact_name is not None:
                _upload_checkpoint_artifact(wandb_run, ckpt_path, artifact_name, step)
                log_fn(f"  [eval {label}] checkpoint uploaded to W&B artifact '{artifact_name}'")

        if wandb_run is not None:
            log_dict: dict = {
                "train/tokens_seen": tokens_seen,
                "train/bytes_seen": int(tokens_seen * _bytes_per_token),
                "train/rows_seen": tokens_seen * _rows_per_token,
                "train/wall_time_s": wall_time,
                "train/flops": flops,
                "eval/eval_time_s": eval_time,
            }
            for k, v in test_m.items():
                log_dict[f"eval/test_{k}"] = v
            for k, v in flores_m.items():
                log_dict[f"eval/flores_{k}"] = v
            wandb_run.log(log_dict, step=step)

    # ------------------------------------------------------------------
    # Step-0 eval: random-init baseline before any training
    # ------------------------------------------------------------------
    if _eval_every > 0 and eval_corpus is not None:
        # Model is on CPU at this point; the Trainer will move it to the target
        # device when .train() is called.  Evaluate on CPU with no autocast so
        # the baseline measurement is independent of device availability.
        _run_eval_at_step(hf_model, step=0, dev=torch.device("cpu"), amp=None)

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    class _TrainingMetricsCallback(TrainerCallback):
        """Logs training loss + rich progress metrics to stdout and W&B."""

        def on_log(self, args, state, control, logs=None, **kwargs):
            if not logs:
                return
            loss = logs.get("loss")
            lr = logs.get("learning_rate")
            if loss is None:
                return
            step = int(getattr(state, "global_step", 0))
            tokens_seen = step * tokens_per_step
            wall_time = time.time() - t_start
            flops = _flops(_param_count, tokens_seen)
            throughput = tokens_seen / wall_time if wall_time > 0 else 0.0

            parts = [f"    step {step:>5}/{total_steps}  loss={float(loss):.4f}"]
            if lr is not None:
                parts.append(f"lr={float(lr):.2e}")
            parts.append(f"tok={tokens_seen / 1e6:.1f}M")
            parts.append(f"flops={flops:.2e}")
            log_fn("  ".join(parts))

            if wandb_run is not None:
                log_dict: dict = {
                    "train/loss": float(loss),
                    "train/tokens_seen": tokens_seen,
                    "train/bytes_seen": int(tokens_seen * _bytes_per_token),
                    "train/rows_seen": tokens_seen * _rows_per_token,
                    "train/flops": flops,
                    "train/wall_time_s": wall_time,
                    "train/throughput_tok_s": throughput,
                }
                if lr is not None:
                    log_dict["train/lr"] = float(lr)
                wandb_run.log(log_dict, step=step)

    class _EvalAndCheckpointCallback(TrainerCallback):
        """Fires eval at evenly-spaced steps computed from n_mid_evals."""

        def on_step_end(self, args, state, control, model=None, **kwargs):
            if _eval_every <= 0 or eval_corpus is None:
                return control
            step = int(getattr(state, "global_step", 0))
            if step == 0 or step % _eval_every != 0:
                return control
            if model is None:
                log_fn(f"  [eval {step}/{total_steps}] model not in callback, skipping")
                return control

            _dev = next(model.parameters()).device
            _amp = resolve_amp_dtype(cfg.dtype, _dev)
            model.eval()
            try:
                _run_eval_at_step(model, step=step, dev=_dev, amp=_amp)
            finally:
                model.train()
            return control

    def collate_fn(batch):
        import torch

        input_ids = torch.stack([b["input_ids"] for b in batch])
        labels = torch.stack([b["labels"] for b in batch])
        return {"input_ids": input_ids, "labels": labels}

    callbacks = [_TrainingMetricsCallback(), _EvalAndCheckpointCallback()]

    trainer = Trainer(
        model=hf_model,
        args=training_args,
        train_dataset=ds,
        data_collator=collate_fn,
        callbacks=callbacks,
    )

    t0 = time.time()
    trainer.train()
    train_seconds = time.time() - t0

    del trainer
    import gc
    gc.collect()
    try:
        import torch as _torch
        _torch.cuda.empty_cache()
    except Exception:
        pass
    import shutil
    shutil.rmtree(outdir, ignore_errors=True)

    wrapped = HFWrapper(hf_model)
    wrapped.to(device)
    return wrapped, device, amp_dtype, train_seconds, train_corpus


def train_lm(
    tokenizer,
    train_corpus_path: Path,
    cfg: LLMConfig,
    log_fn=print,
    wandb_run=None,
    *,
    eval_corpus: "TokenizedCorpus | None" = None,
    flores_corpus: "TokenizedCorpus | None" = None,
    checkpoint_dir: "Path | None" = None,
    artifact_name: "str | None" = None,
    run_timestamp: "str | None" = None,
    vocab_size: "int | None" = None,
):
    """Train a GPT on tokenized ``train_corpus_path`` for cfg.train_tokens tokens.

    Returns the trained model + the device + the autocast dtype + train wall time.
    Logs per-step ``train/loss`` and ``train/lr`` to ``wandb_run`` if provided.
    """
    return train_lm_transformers(
        tokenizer, train_corpus_path, cfg,
        log_fn=log_fn, wandb_run=wandb_run,
        eval_corpus=eval_corpus,
        flores_corpus=flores_corpus,
        checkpoint_dir=checkpoint_dir,
        artifact_name=artifact_name,
        run_timestamp=run_timestamp,
        vocab_size=vocab_size,
    )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def _score_corpus(model, corpus: TokenizedCorpus, cfg: LLMConfig, device, amp_dtype) -> dict:
    """Score a TokenizedCorpus under ``model`` and return PPL/BPB metrics.

    BPB normalises by the source byte count from the corpus. If we drop a
    tail of < ctx_len ids that don't fill a final window, we scale the byte
    count and the row count proportionally so each metric remains
    per-(byte/row)-of-scored-text.
    """
    import torch

    eval_ids = corpus.ids
    total_bytes = corpus.source_bytes
    total_rows = corpus.rows
    if eval_ids.shape[0] < 2:
        raise RuntimeError("Need at least 2 tokens to compute perplexity.")

    ctx = cfg.ctx_len
    model.eval()
    total_loss_sum = 0.0
    total_targets = 0
    with torch.no_grad():
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
            f"Eval data too short for even one window of ctx_len={ctx}."
        )

    mean_nll_per_token = total_loss_sum / total_targets
    perplexity = math.exp(mean_nll_per_token)

    fraction_scored = total_targets / max(1, eval_ids.shape[0] - 1)
    scored_bytes = total_bytes * fraction_scored
    scored_rows = total_rows * fraction_scored
    bits_per_byte = (total_loss_sum / scored_bytes) / math.log(2) if scored_bytes > 0 else float("nan")

    return {
        "perplexity": perplexity,
        "mean_nll_per_token": mean_nll_per_token,
        "bits_per_byte": bits_per_byte,
        "eval_tokens_scored": total_targets,
        "eval_bytes_scored": scored_bytes,
        "eval_rows_scored": scored_rows,
        "eval_bytes_per_row": (scored_bytes / scored_rows) if scored_rows else 0.0,
    }


def evaluate_perplexity(
    model,
    tokenizer,
    eval_corpus_path: Path,
    cfg: LLMConfig,
    device,
    amp_dtype,
    log_fn=print,
) -> dict:
    """Compute per-token perplexity and bits-per-byte on ``eval_corpus_path``."""
    eos_id = resolve_eos_id(tokenizer)
    corpus = tokenize_corpus(tokenizer, eval_corpus_path, max_tokens=None, eos_id=eos_id)
    log_fn(
        f"  eval (test set): {corpus.n_tokens:,} tokens, "
        f"{corpus.rows:,} rows, "
        f"{corpus.source_bytes:,} bytes "
        f"({corpus.bytes_per_row:.1f} bytes/row, "
        f"{corpus.tokens_per_row:.2f} tokens/row)"
    )
    return _score_corpus(model, corpus, cfg, device, amp_dtype)


def evaluate_perplexity_on_sentences(
    model,
    tokenizer,
    sentences: list[str],
    cfg: LLMConfig,
    device,
    amp_dtype,
    label: str = "sentences",
    log_fn=print,
) -> dict:
    """Score a list of sentences (e.g. FLORES devtest) under ``model``.

    Each sentence counts as one row for the bytes-per-row indicator.
    Sentences are concatenated with a single space separator before tokenizing,
    so the windowed scoring sees a continuous stream — one FLORES sentence is
    too short to fill a 512-token context on its own.
    """
    eos_id = resolve_eos_id(tokenizer)
    corpus = _tokenize_sentences_to_corpus(tokenizer, sentences, eos_id)
    log_fn(
        f"  eval ({label}): {corpus.n_tokens:,} tokens, "
        f"{corpus.rows:,} rows, "
        f"{corpus.source_bytes:,} bytes "
        f"({corpus.bytes_per_row:.1f} bytes/row, "
        f"{corpus.tokens_per_row:.2f} tokens/row)"
    )
    return _score_corpus(model, corpus, cfg, device, amp_dtype)


_FLORES_CDN = "https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz"


def load_flores_devtest(language: str) -> list[str]:
    """Return FLORES-200 devtest sentences for ``language`` (en/zh/tr/ru/hi).

    Downloads the official FLORES-200 archive from Meta's CDN directly,
    bypassing the HuggingFace datasets loading script (which is broken in
    datasets >= 3.0). The archive is cached in the HF datasets cache dir
    (or a system temp dir) so the download only happens once.
    """
    if language not in FLORES_CONFIGS:
        raise ValueError(
            f"No FLORES config registered for language {language!r}. "
            f"Known: {sorted(FLORES_CONFIGS)}"
        )
    config = FLORES_CONFIGS[language]

    import tarfile
    import tempfile
    import urllib.request

    # Resolve cache directory: prefer HF datasets cache so it coexists with
    # other cached datasets; fall back to a persistent temp dir.
    try:
        from datasets import config as _ds_cfg
        cache_root = Path(_ds_cfg.HF_DATASETS_CACHE)
    except Exception:
        cache_root = Path(tempfile.gettempdir())

    flores_root = cache_root / "flores200_dataset"
    data_file = flores_root / "devtest" / f"{config}.devtest"

    if not data_file.exists():
        archive = cache_root / "flores200_dataset.tar.gz"
        if not archive.exists():
            cache_root.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(_FLORES_CDN, archive)
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(cache_root)

    lines = data_file.read_text(encoding="utf-8").splitlines()
    return [s.strip() for s in lines if s.strip()]


# ---------------------------------------------------------------------------
# W&B artifact helpers
# ---------------------------------------------------------------------------


def _log_tokenizer_artifact(wandb_run, tokenizer_artifact_dir: Path, name: str) -> None:
    """Upload the tokenizer artifact directory as a W&B Artifact (type=tokenizer).

    Raises on failure — callers must not silently swallow this.
    """
    import wandb

    art = wandb.Artifact(name=name, type="tokenizer")
    art.add_dir(str(tokenizer_artifact_dir))
    wandb_run.log_artifact(art)


def _log_model_artifact(
    wandb_run,
    model,
    cfg: LLMConfig,
    vocab_size: int,
    name: str,
    path: Path,
) -> None:
    """Save the final model state_dict + config to *path* and upload to W&B.

    *path* must be a persistent location (not a temp dir) — W&B reads the file
    asynchronously after this function returns.  Raises on any failure.
    """
    import wandb

    _save_checkpoint_to_path(model, cfg, vocab_size, path)
    art = wandb.Artifact(
        name=name,
        type="model",
        metadata={"vocab_size": vocab_size, "step": "final", **asdict(cfg)},
    )
    art.add_file(str(path), name="final.pt")
    wandb_run.log_artifact(art)


def _define_wandb_metrics(wandb_run) -> None:
    """Register metric definitions so W&B knows which x-axes to use.

    All train/* and eval/* metrics use the global training step as their
    primary x-axis.  In the W&B UI you can create additional custom panels
    using train/flops or train/wall_time_s as x-axes to get the BPB-vs-compute
    and BPB-vs-time curves.
    """
    import wandb

    wandb.define_metric("train/step")
    wandb.define_metric("train/*", step_metric="train/step")
    wandb.define_metric("eval/*", step_metric="train/step")


# ---------------------------------------------------------------------------
# W&B run init
# ---------------------------------------------------------------------------


def _init_wandb(cfg: LLMConfig, extra_config: dict | None = None):
    """Start a wandb run if cfg.wandb_project is set; otherwise return None."""
    if not cfg.wandb_project:
        return None
    _ensure_wandb_login()
    import wandb

    init_config = asdict(cfg)
    if extra_config:
        init_config.update(extra_config)
    run = wandb.init(
        project=cfg.wandb_project,
        entity=cfg.wandb_entity,
        name=cfg.wandb_run_name,
        tags=cfg.wandb_tags or None,
        config=init_config,
        reinit=True,
    )
    _define_wandb_metrics(run)
    return run


# ---------------------------------------------------------------------------
# Top-level: train + eval one tokenizer
# ---------------------------------------------------------------------------


def train_and_evaluate(
    tokenizer,
    train_corpus_path: Path,
    eval_corpus_path: Path,
    cfg: LLMConfig,
    log_fn=print,
    language: str | None = None,
    eval_flores: bool = True,
    wandb_extra_config: dict | None = None,
    tokenizer_artifact_dir: Path | None = None,
    artifact_name: str | None = None,
) -> dict:
    """Train an LM with ``tokenizer`` then score it on test set + (optionally) FLORES.

    Each LM is monolingual: it sees only ``train_corpus_path`` (single-language
    by construction) and is scored on monolingual eval sets.

    Returns metrics with ``test_*`` and ``flores_*`` keys (the latter only if
    FLORES eval ran). If ``cfg.wandb_project`` is set, a W&B run is opened
    around the train+eval loop and final metrics are logged to its summary.

    Artifact naming
    ---------------
    Tokenizer  → ``{artifact_name}-tok``   (stable; same tokenizer every run)
    Model      → ``{artifact_name}-{run_timestamp}``  (unique per training run)

    Checkpoint uploads raise on failure so errors are never silently lost.
    """
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = artifact_name or "model"
    tokenizer_art_name = f"{base_name}-tok"
    model_art_name = f"{base_name}-{run_timestamp}"

    # Set up a persistent checkpoint directory alongside the tokenizer artifact.
    checkpoint_dir: Path | None = None
    if cfg.n_mid_evals > 0:
        if tokenizer_artifact_dir is not None:
            checkpoint_dir = Path(tokenizer_artifact_dir) / "checkpoints" / run_timestamp
        else:
            checkpoint_dir = Path("artifacts") / "checkpoints" / base_name / run_timestamp
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        log_fn(f"  checkpoints → {checkpoint_dir}")

    # Pre-tokenize eval corpus once so both mid-training callbacks and the
    # final post-training eval reuse the same token stream.
    eos_id = resolve_eos_id(tokenizer)
    eval_tc = tokenize_corpus(tokenizer, eval_corpus_path, max_tokens=None, eos_id=eos_id)
    log_fn(
        f"  eval corpus: {eval_tc.n_tokens:,} tokens, "
        f"{eval_tc.rows:,} rows, "
        f"{eval_tc.source_bytes:,} bytes "
        f"({eval_tc.bytes_per_row:.1f} bytes/row)"
    )

    # Pre-load and pre-tokenize FLORES so the training callback doesn't hit the
    # network mid-run and every eval step reuses the same tokenized object.
    flores_tc: TokenizedCorpus | None = None
    if eval_flores and language is not None and language in FLORES_CONFIGS:
        try:
            flores_sents = load_flores_devtest(language)
            flores_tc = _tokenize_sentences_to_corpus(tokenizer, flores_sents, eos_id)
            log_fn(
                f"  flores corpus ({FLORES_CONFIGS[language]}): "
                f"{flores_tc.n_tokens:,} tokens, {flores_tc.rows:,} rows"
            )
        except Exception as exc:
            import traceback as _tb
            log_fn(f"  FLORES pre-load failed for {language!r} (will skip): {exc}")
            log_fn(_tb.format_exc())

    wandb_run = _init_wandb(cfg, extra_config=wandb_extra_config)
    try:
        # Upload tokenizer artifact up-front so it's available even if training
        # crashes mid-way.  Use a stable "-tok" suffix to avoid name collisions
        # with the model artifact.
        if (
            wandb_run is not None
            and cfg.wandb_log_tokenizer_artifact
            and tokenizer_artifact_dir is not None
            and Path(tokenizer_artifact_dir).is_dir()
        ):
            _log_tokenizer_artifact(
                wandb_run,
                Path(tokenizer_artifact_dir),
                name=tokenizer_art_name,
            )
            log_fn(f"  uploaded tokenizer artifact to W&B as '{tokenizer_art_name}'")

        model, device, amp_dtype, train_seconds, train_corpus = train_lm(
            tokenizer, train_corpus_path, cfg,
            log_fn=log_fn,
            wandb_run=wandb_run,
            eval_corpus=eval_tc,
            flores_corpus=flores_tc,
            checkpoint_dir=checkpoint_dir,
            artifact_name=model_art_name,
            run_timestamp=run_timestamp,
            vocab_size=tokenizer.vocab_size,
        )

        # Final eval — reuse the pre-tokenized corpora.
        out: dict = {}

        log_fn(
            f"  final eval (test set): {eval_tc.n_tokens:,} tokens, "
            f"{eval_tc.rows:,} rows, {eval_tc.source_bytes:,} bytes"
        )
        test_metrics = _score_corpus(model, eval_tc, cfg, device, amp_dtype)
        for k, v in test_metrics.items():
            out[f"test_{k}"] = v

        if flores_tc is not None:
            log_fn(
                f"  final eval (flores/{FLORES_CONFIGS.get(language, '?')}): "
                f"{flores_tc.n_tokens:,} tokens, {flores_tc.rows:,} rows"
            )
            try:
                flores_metrics = _score_corpus(model, flores_tc, cfg, device, amp_dtype)
                for k, v in flores_metrics.items():
                    out[f"flores_{k}"] = v
            except Exception as exc:
                import traceback as _tb
                log_fn(f"  final FLORES eval failed: {exc}")
                log_fn(_tb.format_exc())

        # Training-corpus stats (rows + bytes/row, the sanity-check indicator).
        out["train_tokens_actual"] = train_corpus.n_tokens
        out["train_rows"] = train_corpus.rows
        out["train_source_bytes"] = train_corpus.source_bytes
        out["train_bytes_per_row"] = train_corpus.bytes_per_row
        out["train_tokens_per_row"] = train_corpus.tokens_per_row

        out["param_count"] = count_parameters(model)
        out["train_seconds"] = train_seconds
        out["config"] = asdict(cfg)

        if wandb_run is not None:
            for k, v in out.items():
                if isinstance(v, (int, float)):
                    wandb_run.summary[k] = v

        # Upload final model checkpoint.  Uses the same artifact name as
        # mid-training checkpoints so W&B versions them together (final.pt
        # appears as the last version).  Raises on failure — no silent loss.
        if wandb_run is not None and cfg.wandb_log_model_artifact:
            final_ckpt_path = (
                checkpoint_dir / "final.pt"
                if checkpoint_dir is not None
                else Path("artifacts") / "checkpoints" / base_name / run_timestamp / "final.pt"
            )
            _log_model_artifact(
                wandb_run, model, cfg,
                vocab_size=tokenizer.vocab_size,
                name=model_art_name,
                path=final_ckpt_path,
            )
            log_fn(f"  uploaded final model artifact to W&B as '{model_art_name}'")

        import gc
        del model
        gc.collect()
        try:
            import torch as _torch
            _torch.cuda.empty_cache()
        except Exception:
            pass

        return out
    finally:
        if wandb_run is not None:
            wandb_run.finish()
