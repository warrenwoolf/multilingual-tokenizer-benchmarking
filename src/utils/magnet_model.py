"""MAGNET: byte-level LM with learned segment boundaries.

Reference: Ahia et al. (2024) "MAGNET: Improving the Multilingual Fairness of
Language Models with Adaptive Gradient-Based Tokenization."
https://arxiv.org/abs/2407.08818

Architecture (per the paper):

    byte embeddings
        → pre_blocks  (full-length transformer layers on bytes)
        → BoundaryPredictor  →  hard_boundaries [B, L]
        → _downsample  (mean-pool bytes within each segment, prepend null group)
        → LayerNorm
        → shortened_blocks  (transformer layers on compressed sequence)
        → _upsample  (distribute each segment repr back to its byte positions)
        → residual add
        → post_blocks  (optional extra transformer layers on bytes)
        → LM head → next-byte logits

Because every prediction target is a raw byte, cross-entropy is already in
nats/byte. BPB = mean_CE / ln(2) with no further normalisation.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Shared transformer block (pre-LN causal)
# ---------------------------------------------------------------------------


class _Block(nn.Module):
    """Pre-LN causal self-attention block."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.ln1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        h = self.ln1(x)
        qkv = self.qkv(h).view(B, T, 3, self.n_heads, self.d_head)
        q, k, v = qkv.unbind(2)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        attn = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=True,
            dropout_p=self.drop.p if self.training else 0.0,
        )
        attn = attn.transpose(1, 2).contiguous().view(B, T, C)
        x = x + self.drop(self.proj(attn))
        x = x + self.drop(self.mlp(self.ln2(x)))
        return x


# ---------------------------------------------------------------------------
# Boundary predictor
# ---------------------------------------------------------------------------


class BoundaryPredictor(nn.Module):
    """Score each byte position as a potential segment-boundary start.

    Uses Gumbel-Bernoulli (RelaxedBernoulli) sampling for differentiable
    boundaries during training, and a hard threshold at inference.

    Boundary loss: Binomial prior that penalises deviations from the target
    compression rate encoded in ``prior`` (= expected fraction of bytes that
    start a new segment).
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        prior: float,
        temp: float,
        threshold: float,
    ):
        super().__init__()
        self.prior = prior
        self.temp = temp
        self.threshold = threshold
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, 1),
        )

    def forward(
        self, hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (soft_boundaries, hard_boundaries), both [B, L].

        During training uses Gumbel-Bernoulli; during eval uses raw sigmoid.
        hard_boundaries is computed with the straight-through estimator so
        gradients flow to the boundary predictor via the boundary loss.
        """
        logits = self.net(hidden).squeeze(-1)  # [B, L]
        probs = torch.sigmoid(logits)

        if self.training:
            dist = torch.distributions.relaxed_bernoulli.RelaxedBernoulli(
                temperature=self.temp, probs=probs
            )
            soft = dist.rsample()
        else:
            soft = probs

        hard = (soft > self.threshold).float()
        # Straight-through estimator: forward = hard, backward = soft.
        hard = hard - soft.detach() + soft
        return soft, hard

    def boundary_loss(self, soft: torch.Tensor) -> torch.Tensor:
        """Binomial regulariser: -log P(#boundaries | Binomial(L, prior)).

        Normalised by sequence length so the scale is independent of L.

        ``soft`` is the Gumbel-relaxed boundary probability (values in (0,1)),
        so its sum is a soft approximation of the boundary count. We use
        ``validate_args=False`` because the non-integer sum is intentional —
        this gives a differentiable proxy for the integer Binomial log-prob
        that trains the boundary predictor via the soft count.
        """
        L = soft.size(-1)
        binom = torch.distributions.Binomial(
            L,
            probs=torch.tensor(self.prior, device=soft.device, dtype=soft.dtype),
            validate_args=False,
        )
        return (-binom.log_prob(soft.sum(dim=-1)).mean()) / L


# ---------------------------------------------------------------------------
# Downsample / upsample
# ---------------------------------------------------------------------------


def _downsample(
    boundaries: torch.Tensor,
    hidden: torch.Tensor,
    null_group: torch.Tensor,
) -> torch.Tensor:
    """Group bytes into segments by mean-pooling, then prepend a null group.

    boundaries : [B, L]  –  1 = start of a new segment
    hidden     : [B, L, D]
    null_group : [1, 1, D]  –  learnable null-segment embedding
    returns    : [B, S+1, D]  where S = max #boundaries across the batch

    The null group at index 0 represents "context before any processed
    segment", enabling the first real byte to attend to something meaningful
    after upsampling.  The last group of bytes (those with exclusive-cumsum
    equal to n_segs) is discarded — it has no following context to predict
    into — so the compressed length is always S (not S+1).

    Segment membership uses an *exclusive* cumsum: a boundary at position i
    marks the last byte of the current segment (not the first of the next),
    so positions 0…i all belong to the same segment.
    """
    B, L, D = hidden.shape
    n_segs = int(boundaries.sum(dim=-1).max().item())

    if n_segs == 0:
        return null_group.expand(B, 1, D)

    # Exclusive cumsum → segment index for each position (0 = before 1st boundary).
    seg_ids = (boundaries.cumsum(dim=1) - boundaries).long()  # [B, L]

    # Build a soft assignment matrix: weight[b, l, s] = 1/count if position l
    # belongs to segment s (in {0, …, n_segs-1}), else 0.
    seg_range = torch.arange(n_segs, device=hidden.device).view(1, 1, n_segs)
    mask = (seg_ids.unsqueeze(-1) == seg_range).float()  # [B, L, S]
    counts = mask.sum(dim=1, keepdim=True).clamp(min=1e-9)  # [B, 1, S]
    weights = mask / counts  # columns sum to 1 across L dim

    shortened = torch.einsum("bls,bld->bsd", weights, hidden)  # [B, S, D]
    null = null_group.expand(B, 1, D)
    return torch.cat([null, shortened], dim=1)  # [B, S+1, D]


def _upsample(
    boundaries: torch.Tensor,
    shortened: torch.Tensor,
) -> torch.Tensor:
    """Distribute each segment representation back to its byte positions.

    Each byte position receives the representation of the *preceding* segment
    (inclusive cumsum → shift-by-one), preserving causality.

    boundaries : [B, L]
    shortened  : [B, S+1, D]
    returns    : [B, L, D]
    """
    B, L = boundaries.shape
    n_total = shortened.shape[1]  # S+1

    # Inclusive cumsum: position l gets shortened[cumsum[l]].
    cumsum = boundaries.cumsum(dim=1).long()  # [B, L], values in [0, S]

    seg_range = torch.arange(n_total, device=shortened.device).view(1, 1, n_total)
    weights = (cumsum.unsqueeze(-1) == seg_range).float()  # [B, L, S+1]
    # Each row should have exactly one 1; clamp guards against rounding.
    weights = weights / weights.sum(dim=2, keepdim=True).clamp(min=1e-9)

    return torch.einsum("bls,bsd->bld", weights, shortened)  # [B, L, D]


# ---------------------------------------------------------------------------
# Full MAGNET LM
# ---------------------------------------------------------------------------


class MagnetLM(nn.Module):
    """Byte-level language model with learned boundary-based compression.

    Hyperparameters
    ---------------
    vocab_size        : byte vocabulary size (ByT5 default ≈ 384)
    d_model           : hidden dimension
    n_heads           : attention heads (must divide d_model)
    d_ff              : feed-forward inner dimension
    dropout           : dropout rate
    pre_layers        : number of transformer blocks before boundary prediction
    shortened_layers  : number of transformer blocks on compressed sequence
    post_layers       : number of transformer blocks after upsampling (can be 0)
    ctx_len           : context length in bytes (for positional embeddings)
    boundary_prior    : target fraction of bytes that start a segment
                        (controls compression rate; 0.25 → ~4× compression)
    boundary_temp     : Gumbel-Sigmoid temperature (lower → harder boundaries)
    boundary_threshold: hard-boundary cutoff (default 0.5)
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
        pre_layers: int,
        shortened_layers: int,
        post_layers: int,
        ctx_len: int,
        boundary_prior: float,
        boundary_temp: float,
        boundary_threshold: float = 0.5,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.ctx_len = ctx_len

        def _make_blocks(n: int) -> nn.ModuleList:
            return nn.ModuleList([_Block(d_model, n_heads, d_ff, dropout) for _ in range(n)])

        # Byte-level input representation
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(ctx_len, d_model)
        self.drop = nn.Dropout(dropout)

        # Pre-compression layers (run on full byte sequence)
        self.pre_blocks = _make_blocks(pre_layers)

        # Boundary prediction + downsampling machinery
        self.boundary_predictor = BoundaryPredictor(
            d_model=d_model,
            d_ff=d_ff,
            prior=boundary_prior,
            temp=boundary_temp,
            threshold=boundary_threshold,
        )
        self.null_group = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.null_group, std=0.02)
        self.down_ln = nn.LayerNorm(d_model)

        # Positional embeddings for the shortened (segment-level) sequence.
        # ctx_len is a safe upper bound: at most one boundary per byte.
        self.seg_pos_emb = nn.Embedding(ctx_len, d_model)

        # Shortened-sequence transformer layers
        self.shortened_blocks = _make_blocks(shortened_layers)

        # Optional post-upsampling layers
        self.post_blocks = _make_blocks(post_layers)

        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # weight tying

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Forward pass.

        Parameters
        ----------
        idx     : [B, T] integer byte ids
        targets : [B, T] shifted byte ids (optional; required for losses)

        Returns
        -------
        logits         : [B, T, vocab_size]
        lm_loss        : scalar mean cross-entropy in nats/byte, or None
        boundary_loss  : scalar Binomial regulariser, or None
        """
        B, T = idx.shape
        assert T <= self.ctx_len, f"Sequence length {T} > ctx_len {self.ctx_len}"

        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))  # [B, T, D]

        for blk in self.pre_blocks:
            x = blk(x)

        soft_bounds, hard_bounds = self.boundary_predictor(x)

        # Downsample: use detached hard boundaries so the CE gradient flows
        # through hidden values (x), not through boundary positions.
        shortened = _downsample(hard_bounds.detach(), x, self.null_group)  # [B, S+1, D]
        shortened = self.down_ln(shortened)

        # Add segment-level positional embeddings.
        S = shortened.shape[1]
        seg_pos = torch.arange(S, device=idx.device)
        shortened = shortened + self.seg_pos_emb(seg_pos)

        for blk in self.shortened_blocks:
            shortened = blk(shortened)

        up = _upsample(hard_bounds.detach(), shortened)  # [B, T, D]
        x = x + up  # residual

        for blk in self.post_blocks:
            x = blk(x)

        logits = self.lm_head(self.ln_f(x))  # [B, T, V]

        lm_loss = None
        boundary_loss = None
        if targets is not None:
            lm_loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1)
            )
            boundary_loss = self.boundary_predictor.boundary_loss(soft_bounds)

        return logits, lm_loss, boundary_loss
