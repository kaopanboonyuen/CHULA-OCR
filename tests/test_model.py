"""
CHULA-OCR Test Suite

Author: Teerapong Panboonyuen (Kao), Ph.D.

Usage:
    pytest tests/ -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import pytest
from src.model import CHULA_OCR, CHULAOCRLoss, UncertaintyGating
from src.dataset import ThaiOCRTokenizer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="module")
def small_model(device):
    """Lightweight model for fast testing."""
    return CHULA_OCR(
        vocab_size=100,
        img_size=64,
        patch_size=16,
        embed_dim=128,
        num_decoder_layers=2,
        num_heads=4,
        ff_dim=256,
        max_seq_len=32,
    ).to(device)


@pytest.fixture(scope="module")
def tokenizer():
    return ThaiOCRTokenizer()


# ---------------------------------------------------------------------------
# Tokenizer Tests
# ---------------------------------------------------------------------------

class TestTokenizer:
    def test_vocab_size(self, tokenizer):
        assert len(tokenizer) > 0

    def test_special_tokens(self, tokenizer):
        assert tokenizer.pad_idx == 0
        assert tokenizer.bos_idx == 1
        assert tokenizer.eos_idx == 2

    def test_encode_decode_roundtrip(self, tokenizer):
        texts = ["กข", "12345", "โฉนด"]
        for text in texts:
            enc = tokenizer.encode(text)
            dec = tokenizer.decode(enc)
            assert dec == text, f"Roundtrip failed: '{text}' -> '{dec}'"

    def test_batch_encode(self, tokenizer):
        texts = ["กข", "คง", "จฉ"]
        tokens, mask = tokenizer.batch_encode(texts, max_len=16)
        assert tokens.shape == (3, 16)
        assert mask.shape == (3, 16)
        assert tokens.dtype == torch.long

    def test_unknown_char(self, tokenizer):
        enc = tokenizer.encode("🎉")
        assert tokenizer.unk_idx in enc


# ---------------------------------------------------------------------------
# Model Architecture Tests
# ---------------------------------------------------------------------------

class TestModel:
    def test_forward_training(self, small_model, device):
        B = 2
        images = torch.randn(B, 3, 64, 64, device=device)
        tokens = torch.randint(3, 100, (B, 16), device=device)
        tokens[:, 0] = 1   # BOS
        tokens[:, -1] = 2  # EOS

        out = small_model(images, tokens)

        assert "loss" in out
        assert "logits" in out
        assert out["loss"].item() > 0
        assert out["logits"].shape == (B, 15, 100)  # T-1 decoder steps

    def test_inference(self, small_model, device):
        B = 2
        images = torch.randn(B, 3, 64, 64, device=device)
        preds, unc = small_model.predict(images, return_uncertainty=True)

        assert preds.shape[0] == B
        assert unc is not None
        assert unc.shape[0] == B

    def test_uncertainty_range(self, small_model, device):
        images = torch.randn(2, 3, 64, 64, device=device)
        _, uncertainty, _ = small_model.encode(images)
        assert (uncertainty >= 0).all()
        assert (uncertainty <= 1).all()

    def test_loss_components(self, small_model, device):
        B = 2
        images = torch.randn(B, 3, 64, 64, device=device)
        tokens = torch.randint(3, 100, (B, 16), device=device)
        tokens[:, 0] = 1
        tokens[:, -1] = 2

        out = small_model(images, tokens)
        for key in ["loss", "loss_ce", "loss_uncertainty", "loss_heuristic"]:
            assert key in out
            assert not torch.isnan(out[key])

    def test_gradient_flow(self, small_model, device):
        images = torch.randn(2, 3, 64, 64, device=device)
        tokens = torch.randint(3, 100, (2, 16), device=device)
        tokens[:, 0] = 1

        out = small_model(images, tokens)
        out["loss"].backward()

        for name, p in small_model.named_parameters():
            if p.grad is not None:
                assert not torch.isnan(p.grad).any(), f"NaN grad in {name}"


# ---------------------------------------------------------------------------
# Loss Tests
# ---------------------------------------------------------------------------

class TestLoss:
    def test_chula_loss_basic(self, device):
        loss_fn = CHULAOCRLoss()
        B, T, V, N, D = 2, 10, 100, 16, 128

        logits = torch.randn(B, T, V, device=device)
        targets = torch.randint(1, V, (B, T), device=device)
        uncertainty = torch.sigmoid(torch.randn(B, N, 1, device=device))
        features = torch.randn(B, N, D, device=device)

        losses = loss_fn(logits, targets, uncertainty, features)

        assert losses["loss"].item() > 0
        assert not torch.isnan(losses["loss"])

    def test_entropy_weights(self, device):
        loss_fn = CHULAOCRLoss()
        # High entropy → low weight
        uniform_logits = torch.zeros(1, 5, 100, device=device)
        w_high = loss_fn.compute_entropy_weights(uniform_logits)

        # Low entropy → high weight
        peaked = torch.zeros(1, 5, 100, device=device)
        peaked[0, :, 42] = 10.0
        w_low = loss_fn.compute_entropy_weights(peaked)

        assert w_low.mean() > w_high.mean()


# ---------------------------------------------------------------------------
# Uncertainty Gating Tests
# ---------------------------------------------------------------------------

class TestUncertaintyGating:
    def test_gating_suppresses_uncertain(self, device):
        gate = UncertaintyGating(feature_dim=64).to(device)
        B, N, D = 2, 16, 64
        features = torch.ones(B, N, D, device=device)

        gated, uncertainty = gate(features)

        # Gated features should be <= original
        assert (gated <= features + 1e-5).all()
        assert gated.shape == features.shape
        assert uncertainty.shape == (B, N, 1)

    def test_uncertainty_bounds(self, device):
        gate = UncertaintyGating(feature_dim=64).to(device)
        features = torch.randn(2, 16, 64, device=device)
        _, uncertainty = gate(features)
        assert (uncertainty >= 0).all()
        assert (uncertainty <= 1).all()


# ---------------------------------------------------------------------------
# Run tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
