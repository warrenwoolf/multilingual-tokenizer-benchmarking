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
    batch_size: int = 512  # fills A100 40GB with FlashAttention (F.sdpa); ~256K tok/step
    learning_rate: float = 1e-3  # sqrt-scaled from 3e-4 @ bs=32: 3e-4 * sqrt(512/32) ≈ 1.2e-3
    min_lr: float = 1e-4
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


@dataclass
class TokenizedCorpus:
    """A tokenized corpus along with the row + byte counts it was built from.

    The byte count (``source_bytes``) is the raw UTF-8 byte length of the
    *consumed* portion of the source text — i.e. it accounts for early stops
    on ``max_tokens``. This is what BPB normalizes by, and dividing by
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
        ids = tokenizer.encode(line)
        if eos_id is not None:
            ids = ids + [eos_id]
        if not ids:
            continue
        if max_tokens is not None and total_tokens + len(ids) > max_tokens:
            ids = ids[: max_tokens - total_tokens]
            if ids:
                chunks.append(np.asarray(ids, dtype=np.int32))
                total_tokens += len(ids)
                rows += 1
                # Approximate byte contribution proportional to the kept
                # token fraction for the truncated final row. Rough but
                # bounded by one row's worth, which is negligible.
                src_bytes += len(line.encode("utf-8"))
            break
        chunks.append(np.asarray(ids, dtype=np.int32))
        total_tokens += len(ids)
        rows += 1
        src_bytes += len(line.encode("utf-8"))
    arr = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.int32)
    return TokenizedCorpus(ids=arr, rows=rows, source_bytes=src_bytes)


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


def _init_wandb(cfg: LLMConfig, extra_config: dict | None = None):
    """Start a wandb run if cfg.wandb_project is set; otherwise return None."""
    if not cfg.wandb_project:
        return None
    _ensure_wandb_login()
    import wandb

    init_config = asdict(cfg)
    if extra_config:
        init_config.update(extra_config)
    return wandb.init(
        project=cfg.wandb_project,
        entity=cfg.wandb_entity,
        name=cfg.wandb_run_name,
        tags=cfg.wandb_tags or None,
        config=init_config,
        reinit=True,
    )


def train_lm(
    tokenizer,
    train_corpus_path: Path,
    cfg: LLMConfig,
    log_fn=print,
    wandb_run=None,
):
    """Train a GPT on tokenized ``train_corpus_path`` for cfg.train_tokens tokens.

    Returns the trained model + the device + the autocast dtype + train wall time.
    Logs per-step ``train/loss`` and ``train/lr`` to ``wandb_run`` if provided.
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
    train_corpus = tokenize_corpus(
        tokenizer,
        train_corpus_path,
        max_tokens=cfg.train_tokens,
    )
    log_fn(
        f"  tokenized {train_corpus.n_tokens:,} train tokens "
        f"from {train_corpus.rows:,} rows "
        f"({train_corpus.bytes_per_row:.1f} bytes/row, "
        f"{train_corpus.tokens_per_row:.2f} tokens/row) "
        f"in {time.time() - t0:.1f}s"
    )

    if train_corpus.n_tokens < cfg.ctx_len + 2:
        raise RuntimeError(
            f"Train corpus only produced {train_corpus.n_tokens} tokens; "
            f"need at least ctx_len+2={cfg.ctx_len + 2}."
        )

    vocab_size = tokenizer.vocab_size
    model = _build_model(vocab_size, cfg).to(device)
    n_params = count_parameters(model)
    log_fn(f"  model: vocab={vocab_size:,} params={n_params:,}")
    if wandb_run is not None:
        wandb_run.summary["param_count"] = n_params
        wandb_run.summary["vocab_size"] = vocab_size
        wandb_run.summary["train_tokens_actual"] = train_corpus.n_tokens
        wandb_run.summary["train_rows"] = train_corpus.rows
        wandb_run.summary["train_bytes_per_row"] = train_corpus.bytes_per_row
        wandb_run.summary["train_tokens_per_row"] = train_corpus.tokens_per_row
    train_ids = train_corpus.ids

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
                _, loss = model(x, y)
        else:
            _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        loss_val = loss.item()
        if wandb_run is not None and (step % cfg.log_every == 0 or step == total_steps - 1):
            wandb_run.log(
                {
                    "train/loss": loss_val,
                    "train/lr": lr,
                    "train/tokens_seen": (step + 1) * tokens_per_step,
                },
                step=step,
            )
        if step == 0 or (step + 1) % stdout_every == 0 or step == total_steps - 1:
            log_fn(f"    step {step + 1:>5}/{total_steps}  loss={loss_val:.4f}  lr={lr:.2e}")

    train_seconds = time.time() - t_start
    log_fn(f"  trained in {train_seconds:.1f}s")
    if wandb_run is not None:
        wandb_run.summary["train_seconds"] = train_seconds
    return model, device, amp_dtype, train_seconds, train_corpus


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
    corpus = tokenize_corpus(tokenizer, eval_corpus_path, max_tokens=None)
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
    import numpy as np

    cleaned = [s.strip() for s in sentences if s and s.strip()]
    if not cleaned:
        raise RuntimeError(f"No non-empty sentences in '{label}' eval set.")
    joined = " ".join(cleaned)
    ids = tokenizer.encode(joined)
    if not ids:
        raise RuntimeError(f"Tokenizer produced 0 ids for '{label}' eval text.")
    corpus = TokenizedCorpus(
        ids=np.asarray(ids, dtype=np.int32),
        rows=len(cleaned),
        source_bytes=sum(len(s.encode("utf-8")) for s in cleaned),
    )
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
# Top-level: train + eval one tokenizer
# ---------------------------------------------------------------------------


def _log_tokenizer_artifact(wandb_run, tokenizer_artifact_dir: Path, name: str) -> None:
    """Upload the tokenizer artifact directory as a W&B Artifact (type=tokenizer)."""
    import wandb

    art = wandb.Artifact(name=name, type="tokenizer")
    art.add_dir(str(tokenizer_artifact_dir))
    wandb_run.log_artifact(art)


def _log_model_artifact(wandb_run, model, cfg: LLMConfig, vocab_size: int, name: str) -> None:
    """Save the trained model state_dict + config to a W&B Artifact (type=model)."""
    import tempfile

    import torch
    import wandb

    art = wandb.Artifact(name=name, type="model", metadata={"vocab_size": vocab_size, **asdict(cfg)})
    with tempfile.TemporaryDirectory() as tmp:
        ckpt_path = Path(tmp) / "model.pt"
        torch.save(
            {
                "state_dict": model.state_dict(),
                "vocab_size": vocab_size,
                "config": asdict(cfg),
            },
            ckpt_path,
        )
        art.add_file(str(ckpt_path), name="model.pt")
        wandb_run.log_artifact(art)


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
    When ``tokenizer_artifact_dir`` is provided and ``cfg.wandb_log_*_artifact``
    is on, the tokenizer dir is uploaded as a W&B artifact at the start of the
    run and the trained model state_dict at the end.
    """
    wandb_run = _init_wandb(cfg, extra_config=wandb_extra_config)
    try:
        # Upload tokenizer artifact (small) up-front so it's available even
        # if training crashes mid-way.
        if (
            wandb_run is not None
            and cfg.wandb_log_tokenizer_artifact
            and tokenizer_artifact_dir is not None
            and Path(tokenizer_artifact_dir).is_dir()
        ):
            try:
                _log_tokenizer_artifact(
                    wandb_run,
                    Path(tokenizer_artifact_dir),
                    name=artifact_name or Path(tokenizer_artifact_dir).name,
                )
                log_fn("  uploaded tokenizer artifact to W&B")
            except Exception as exc:
                log_fn(f"  W&B tokenizer artifact upload failed (continuing): {exc}")

        model, device, amp_dtype, train_seconds, train_corpus = train_lm(
            tokenizer, train_corpus_path, cfg, log_fn=log_fn, wandb_run=wandb_run,
        )

        out: dict = {}
        test_metrics = evaluate_perplexity(
            model, tokenizer, eval_corpus_path, cfg, device, amp_dtype, log_fn=log_fn,
        )
        for k, v in test_metrics.items():
            out[f"test_{k}"] = v

        if eval_flores and language is not None:
            try:
                flores_sents = load_flores_devtest(language)
                flores_metrics = evaluate_perplexity_on_sentences(
                    model, tokenizer, flores_sents, cfg, device, amp_dtype,
                    label=f"flores/{FLORES_CONFIGS[language]}", log_fn=log_fn,
                )
                for k, v in flores_metrics.items():
                    out[f"flores_{k}"] = v
            except Exception as exc:
                import traceback as _tb
                log_fn(f"  FLORES eval skipped for {language!r}: {exc}")
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

        # Upload model checkpoint after eval so its metadata is final.
        if wandb_run is not None and cfg.wandb_log_model_artifact:
            try:
                _log_model_artifact(
                    wandb_run, model, cfg,
                    vocab_size=tokenizer.vocab_size,
                    name=artifact_name or "model",
                )
                log_fn("  uploaded model artifact to W&B")
            except Exception as exc:
                log_fn(f"  W&B model artifact upload failed (continuing): {exc}")

        return out
    finally:
        if wandb_run is not None:
            wandb_run.finish()
