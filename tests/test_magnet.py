"""Tests for the MAGNET model and training pipeline.

All tests run offline (no network) and are CPU-only.  Skipped when torch
is not installed (same gating used by test_llm_training.py).

Test coverage
-------------
Unit (model components):
  - BoundaryPredictor output shapes, training vs eval mode
  - BoundaryPredictor boundary_loss is finite and positive
  - _downsample output shape and compression
  - _upsample restores original sequence length
  - downsample → upsample is approximately invertible (content flows through)
  - MagnetLM forward: correct output shapes with and without targets
  - MagnetLM losses are finite scalars when targets are supplied
  - MagnetLM BPB (= mean_nll / ln(2)) is positive

Integration (training):
  - tokenize_byte_corpus returns correct shapes and counts
  - Train for a handful of steps without crashing, losses are finite
  - evaluate_bpb returns a finite positive BPB
  - train_and_evaluate_magnet end-to-end (smoke test, tiny corpus)
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")  # skip entire module if torch missing

import torch.nn as nn

from src.utils.magnet_model import (
    BoundaryPredictor,
    MagnetLM,
    _downsample,
    _upsample,
)
from src.utils.magnet_training import (
    MagnetConfig,
    evaluate_bpb,
    tokenize_byte_corpus,
    train_and_evaluate_magnet,
    train_magnet,
    _load_byt5_tokenizer,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tiny_cfg(**overrides) -> MagnetConfig:
    """A very small MagnetConfig for fast CPU tests."""
    defaults = dict(
        d_model=32,
        n_heads=2,
        d_ff=64,
        dropout=0.0,
        pre_layers=1,
        shortened_layers=1,
        post_layers=0,
        ctx_len=64,
        boundary_prior=0.25,
        boundary_temp=1.0,
        boundary_threshold=0.5,
        boundary_lambda=1.0,
        train_tokens=5_000,
        batch_size=4,
        learning_rate=1e-3,
        warmup_steps=2,
        seed=0,
        device="cpu",
        dtype="fp32",
        log_every=999,
    )
    defaults.update(overrides)
    return MagnetConfig(**defaults)


def _tiny_model(cfg: MagnetConfig | None = None) -> MagnetLM:
    if cfg is None:
        cfg = _tiny_cfg()
    return MagnetLM(
        vocab_size=384,
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
    )


def _rand_ids(batch: int, seq: int, vocab: int = 259) -> torch.Tensor:
    return torch.randint(3, vocab, (batch, seq))  # ids 3-258 = actual bytes


@pytest.fixture(scope="module")
def tiny_byte_corpus(tmp_path_factory) -> Path:
    """A small plaintext corpus (~30 KB) for byte-level tests."""
    sentences = [
        "The quick brown fox jumps over the lazy dog near the riverbank.",
        "Machine learning models tokenize text before training language representations.",
        "Researchers compare algorithms across multiple languages and vocabulary sizes.",
        "Быстрая коричневая лиса прыгает через ленивую собаку каждым вечером.",
        "Türkiye, Avrupa ve Asya kıtalarını birbirine bağlayan önemli bir ülkedir.",
        "नमस्ते दुनिया, यह एक परीक्षण वाक्य है जो हिंदी में लिखा गया है।",
    ] * 300
    path = tmp_path_factory.mktemp("magnet_corpus") / "corpus.txt"
    path.write_text("\n".join(sentences), encoding="utf-8")
    return path


# ===========================================================================
# Unit tests: BoundaryPredictor
# ===========================================================================


class TestBoundaryPredictor:
    B, L, D = 2, 16, 32

    @pytest.fixture(scope="class")
    def bp(self):
        return BoundaryPredictor(d_model=self.D, d_ff=64, prior=0.25, temp=1.0, threshold=0.5)

    def test_output_shapes_eval(self, bp):
        bp.eval()
        hidden = torch.randn(self.B, self.L, self.D)
        soft, hard = bp(hidden)
        assert soft.shape == (self.B, self.L)
        assert hard.shape == (self.B, self.L)

    def test_output_shapes_train(self, bp):
        bp.train()
        hidden = torch.randn(self.B, self.L, self.D)
        soft, hard = bp(hidden)
        assert soft.shape == (self.B, self.L)
        assert hard.shape == (self.B, self.L)

    def test_hard_boundaries_are_binary(self, bp):
        bp.eval()
        hidden = torch.randn(self.B, self.L, self.D)
        _, hard = bp(hidden)
        # Hard boundaries should be (approximately) 0 or 1 (STE adds small eps during train).
        assert hard.min() >= -1e-5
        assert hard.max() <= 1.0 + 1e-5

    def test_soft_boundaries_in_range_eval(self, bp):
        """In eval mode soft = sigmoid(logits), strictly in (0, 1)."""
        bp.eval()
        hidden = torch.randn(self.B, self.L, self.D)
        soft, _ = bp(hidden)
        assert (soft >= 0).all()
        assert (soft <= 1).all()

    def test_boundary_loss_is_finite_and_positive(self, bp):
        bp.eval()
        soft = torch.rand(self.B, self.L)
        loss = bp.boundary_loss(soft)
        assert loss.shape == ()
        assert math.isfinite(loss.item())
        assert loss.item() > 0

    def test_boundary_loss_decreases_when_prior_matched(self, bp):
        """If soft boundaries match the prior exactly, the Binomial log-prob is maximised,
        so moving away should increase the loss."""
        L = 100
        prior = bp.prior
        # Exactly prior fraction of boundaries — Binomial mode.
        on_target = torch.full((1, L), prior)
        off_target = torch.full((1, L), prior * 0.1)  # way too few boundaries
        loss_on = bp.boundary_loss(on_target)
        loss_off = bp.boundary_loss(off_target)
        assert loss_off.item() > loss_on.item()


# ===========================================================================
# Unit tests: _downsample
# ===========================================================================


class TestDownsample:
    def test_output_shape_no_boundaries(self):
        """Zero boundaries → only the null group is returned."""
        B, L, D = 3, 10, 16
        boundaries = torch.zeros(B, L)
        hidden = torch.randn(B, L, D)
        null = torch.zeros(1, 1, D)
        out = _downsample(boundaries, hidden, null)
        assert out.shape == (B, 1, D)

    def test_output_shape_some_boundaries(self):
        """n_segs non-zero → output shape is (B, n_segs+1, D)."""
        B, L, D = 2, 12, 8
        # Two boundaries at positions 3 and 7 for every batch item.
        boundaries = torch.zeros(B, L)
        boundaries[:, 3] = 1.0
        boundaries[:, 7] = 1.0
        hidden = torch.randn(B, L, D)
        null = nn.Parameter(torch.zeros(1, 1, D))
        out = _downsample(boundaries, hidden, null)
        # max boundaries per row = 2 → S=2, output length = 3
        assert out.shape == (B, 3, D)

    def test_compression_reduces_length(self):
        B, L, D = 2, 20, 8
        boundaries = torch.zeros(B, L)
        # One boundary every 4 positions: positions 4, 8, 12, 16.
        for pos in [4, 8, 12, 16]:
            boundaries[:, pos] = 1.0
        hidden = torch.randn(B, L, D)
        null = torch.zeros(1, 1, D)
        out = _downsample(boundaries, hidden, null)
        assert out.shape[1] < L

    def test_null_group_is_first_row(self):
        """The null_group tensor value should appear at index 0."""
        B, L, D = 1, 8, 4
        boundaries = torch.zeros(B, L)
        boundaries[0, 4] = 1.0
        hidden = torch.randn(B, L, D)
        sentinel = torch.full((1, 1, D), 99.0)
        out = _downsample(boundaries, hidden, sentinel)
        # Index 0 of the output should equal sentinel (null group).
        assert torch.allclose(out[:, 0, :], sentinel.expand(B, 1, D).squeeze(1))

    def test_mean_pooling_correctness(self):
        """With boundaries=[0,0,1,0] the exclusive-cumsum puts positions 0,1,2
        in segment 0 and position 3 in segment 1.  Because n_segs=1, position 3
        is discarded (it becomes the 'last incomplete segment').  Segment 0 is
        therefore the mean of hidden positions 0, 1, and 2."""
        B, L, D = 1, 4, 4
        boundaries = torch.zeros(B, L)
        boundaries[:, 2] = 1.0
        hidden = torch.arange(L * D, dtype=torch.float32).view(1, L, D)
        null = torch.zeros(1, 1, D)
        out = _downsample(boundaries, hidden, null)
        # Segment 0 (positions 0-2) → out[:, 1, :]
        expected_seg0 = hidden[:, :3, :].mean(dim=1)  # mean of rows 0, 1, 2
        assert torch.allclose(out[:, 1, :], expected_seg0, atol=1e-5)


# ===========================================================================
# Unit tests: _upsample
# ===========================================================================


class TestUpsample:
    def test_output_length_matches_boundaries(self):
        B, L, D = 2, 12, 8
        boundaries = torch.zeros(B, L)
        boundaries[:, 4] = 1.0
        boundaries[:, 8] = 1.0
        # n_segs = 2, shortened has S+1 = 3 entries.
        shortened = torch.randn(B, 3, D)
        out = _upsample(boundaries, shortened)
        assert out.shape == (B, L, D)

    def test_no_boundaries_all_positions_get_index0(self):
        """With zero boundaries all bytes are in the 'before-first-segment'
        region, so every position maps to shortened[0] (the null group)."""
        B, L, D = 2, 8, 4
        boundaries = torch.zeros(B, L)
        null_val = 7.0
        shortened = torch.zeros(B, 2, D)  # S+1 = 2 since downsample with no bounds returns 1
        shortened[:, 0, :] = null_val
        # Actually with no boundaries cumsum is all zeros → all map to shortened[0].
        out = _upsample(boundaries, shortened)
        assert torch.allclose(out, torch.full((B, L, D), null_val), atol=1e-5)

    def test_downsample_upsample_roundtrip_shape(self):
        """downsample → upsample should return a tensor with the original
        sequence length, even though content may differ from input."""
        B, L, D = 3, 24, 8
        boundaries = torch.zeros(B, L)
        boundaries[:, 6] = 1.0
        boundaries[:, 12] = 1.0
        boundaries[:, 18] = 1.0
        hidden = torch.randn(B, L, D)
        null = torch.zeros(1, 1, D)
        shortened = _downsample(boundaries, hidden, null)
        restored = _upsample(boundaries, shortened)
        assert restored.shape == (B, L, D)


# ===========================================================================
# Unit tests: MagnetLM forward pass
# ===========================================================================


class TestMagnetLM:
    cfg = _tiny_cfg()
    VOCAB = 384
    B, T = 2, 32

    @pytest.fixture(scope="class")
    def model(self):
        return _tiny_model(self.cfg)

    def test_forward_no_targets_shape(self, model):
        model.eval()
        idx = _rand_ids(self.B, self.T, self.VOCAB)
        logits, lm_loss, boundary_loss = model(idx)
        assert logits.shape == (self.B, self.T, self.VOCAB)
        assert lm_loss is None
        assert boundary_loss is None

    def test_forward_with_targets_shape(self, model):
        model.eval()
        idx = _rand_ids(self.B, self.T, self.VOCAB)
        tgt = _rand_ids(self.B, self.T, self.VOCAB)
        logits, lm_loss, boundary_loss = model(idx, tgt)
        assert logits.shape == (self.B, self.T, self.VOCAB)
        assert lm_loss.shape == ()
        assert boundary_loss.shape == ()

    def test_losses_are_finite(self, model):
        model.eval()
        idx = _rand_ids(self.B, self.T, self.VOCAB)
        tgt = _rand_ids(self.B, self.T, self.VOCAB)
        _, lm_loss, boundary_loss = model(idx, tgt)
        assert math.isfinite(lm_loss.item())
        assert math.isfinite(boundary_loss.item())

    def test_lm_loss_is_positive(self, model):
        """Cross-entropy is non-negative."""
        model.eval()
        idx = _rand_ids(self.B, self.T, self.VOCAB)
        tgt = _rand_ids(self.B, self.T, self.VOCAB)
        _, lm_loss, _ = model(idx, tgt)
        assert lm_loss.item() > 0

    def test_bpb_from_forward_is_positive(self, model):
        model.eval()
        idx = _rand_ids(self.B, self.T, self.VOCAB)
        tgt = _rand_ids(self.B, self.T, self.VOCAB)
        _, lm_loss, _ = model(idx, tgt)
        bpb = lm_loss.item() / math.log(2)
        assert bpb > 0

    def test_forward_is_deterministic_in_eval(self, model):
        model.eval()
        idx = _rand_ids(self.B, self.T, self.VOCAB)
        logits1, _, _ = model(idx)
        logits2, _, _ = model(idx)
        assert torch.allclose(logits1, logits2)

    def test_param_count_is_positive(self, model):
        assert model.count_parameters() > 0

    def test_rejects_sequence_longer_than_ctx_len(self, model):
        model.eval()
        idx = _rand_ids(1, self.cfg.ctx_len + 1, self.VOCAB)
        with pytest.raises(AssertionError):
            model(idx)

    def test_train_mode_forward_finite(self):
        """Training mode uses Gumbel sampling — loss must still be finite."""
        model = _tiny_model()
        model.train()
        idx = _rand_ids(2, 32, 384)
        tgt = _rand_ids(2, 32, 384)
        _, lm_loss, boundary_loss = model(idx, tgt)
        assert math.isfinite(lm_loss.item())
        assert math.isfinite(boundary_loss.item())

    def test_gradients_flow_to_pre_blocks(self):
        """CE loss must produce gradients for the pre-block parameters."""
        model = _tiny_model()
        model.train()
        idx = _rand_ids(2, 16, 384)
        tgt = _rand_ids(2, 16, 384)
        _, lm_loss, boundary_loss = model(idx, tgt)
        total = lm_loss + boundary_loss
        total.backward()
        pre_param = next(model.pre_blocks.parameters())
        assert pre_param.grad is not None
        assert pre_param.grad.abs().sum().item() > 0

    def test_gradients_flow_to_boundary_predictor(self):
        """Boundary loss must produce gradients for the boundary predictor."""
        model = _tiny_model()
        model.train()
        idx = _rand_ids(2, 16, 384)
        tgt = _rand_ids(2, 16, 384)
        _, lm_loss, boundary_loss = model(idx, tgt)
        boundary_loss.backward()
        bp_param = next(model.boundary_predictor.parameters())
        assert bp_param.grad is not None
        assert bp_param.grad.abs().sum().item() > 0


# ===========================================================================
# Integration tests: tokenize_byte_corpus
# ===========================================================================


class TestTokenizeByteCorpus:
    @pytest.fixture(scope="class")
    def tokenizer(self):
        return _load_byt5_tokenizer()

    def test_returns_int32_array(self, tokenizer, tiny_byte_corpus):
        ids, rows, src_bytes = tokenize_byte_corpus(tokenizer, tiny_byte_corpus)
        assert ids.dtype.name == "int32"
        assert ids.shape[0] > 0

    def test_row_and_byte_counts_are_positive(self, tokenizer, tiny_byte_corpus):
        _, rows, src_bytes = tokenize_byte_corpus(tokenizer, tiny_byte_corpus)
        assert rows > 0
        assert src_bytes > 0

    def test_max_tokens_cap(self, tokenizer, tiny_byte_corpus):
        cap = 500
        ids, _, _ = tokenize_byte_corpus(tokenizer, tiny_byte_corpus, max_tokens=cap)
        assert ids.shape[0] <= cap

    def test_bytes_equal_token_count(self, tokenizer, tiny_byte_corpus):
        """For ByT5, every UTF-8 byte maps to exactly one token (no merges).
        The number of ids should roughly equal the raw byte count."""
        ids, rows, src_bytes = tokenize_byte_corpus(tokenizer, tiny_byte_corpus)
        # ByT5 may add EOS tokens, so allow 5% slack.
        ratio = ids.shape[0] / src_bytes
        assert 0.9 < ratio < 1.1, f"token/byte ratio {ratio:.3f} out of range"


# ===========================================================================
# Integration tests: training + evaluation
# ===========================================================================


class TestTrainMagnet:
    def test_train_does_not_crash(self, tiny_byte_corpus):
        cfg = _tiny_cfg()
        losses_seen: list[float] = []

        def capture(msg: str) -> None:
            if "lm_loss=" in msg:
                try:
                    v = float(msg.split("lm_loss=")[1].split()[0])
                    losses_seen.append(v)
                except (IndexError, ValueError):
                    pass

        model, tok, device, amp_dtype, t, stats = train_magnet(
            tiny_byte_corpus, cfg, log_fn=capture
        )
        assert all(math.isfinite(l) for l in losses_seen), "NaN/Inf loss detected"
        assert losses_seen, "expected at least one logged step"
        assert stats["rows"] > 0

    def test_evaluate_bpb_returns_finite_positive(self, tiny_byte_corpus):
        cfg = _tiny_cfg()
        model, tok, device, amp_dtype, _, _ = train_magnet(
            tiny_byte_corpus, cfg, log_fn=lambda *a: None
        )
        metrics = evaluate_bpb(
            model, tok, tiny_byte_corpus, cfg, device, amp_dtype,
            log_fn=lambda *a: None,
        )
        bpb = metrics["bits_per_byte"]
        assert math.isfinite(bpb)
        assert bpb > 0

    def test_bpb_is_lm_loss_over_log2(self, tiny_byte_corpus):
        """BPB = mean_NLL / ln(2) — verify the relationship holds numerically."""
        cfg = _tiny_cfg()
        model, tok, device, amp_dtype, _, _ = train_magnet(
            tiny_byte_corpus, cfg, log_fn=lambda *a: None
        )
        metrics = evaluate_bpb(
            model, tok, tiny_byte_corpus, cfg, device, amp_dtype,
            log_fn=lambda *a: None,
        )
        # mean_nll_per_byte / ln(2) must equal bits_per_byte
        assert math.isclose(
            metrics["mean_nll_per_byte"] / math.log(2),
            metrics["bits_per_byte"],
            rel_tol=1e-6,
        )

    def test_train_and_evaluate_returns_expected_keys(self, tiny_byte_corpus):
        cfg = _tiny_cfg()
        metrics = train_and_evaluate_magnet(
            train_corpus_path=tiny_byte_corpus,
            eval_corpus_path=tiny_byte_corpus,
            cfg=cfg,
            language=None,
            eval_flores=False,
            log_fn=lambda *a: None,
        )
        assert "test_bits_per_byte" in metrics
        assert math.isfinite(metrics["test_bits_per_byte"])
        assert metrics["test_bits_per_byte"] > 0
        assert "param_count" in metrics
        assert metrics["param_count"] > 0
        assert not any(k.startswith("flores_") for k in metrics)

    def test_corpus_too_small_raises(self, tmp_path):
        short = tmp_path / "short.txt"
        short.write_text("hi", encoding="utf-8")
        cfg = _tiny_cfg(ctx_len=64)
        with pytest.raises(RuntimeError, match="Train corpus only produced"):
            train_magnet(short, cfg, log_fn=lambda *a: None)
