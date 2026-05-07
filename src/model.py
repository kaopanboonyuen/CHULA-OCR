"""
CHULA-OCR: Uncertainty-Aware Robust Text Recognition
for National-Scale Land Title Deed Digitization

Author: Teerapong Panboonyuen (Kao), Ph.D.
        C2F Postdoctoral Fellow, Chulalongkorn University
Contact: teerapong.panboonyuen@gmail.com
Project: https://kaopanboonyuen.github.io/CHULA-OCR/

This research is supported by the Second Century Fund (C2F)
Postdoctoral Fellowship, Chulalongkorn University.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Uncertainty Estimation Head
# ---------------------------------------------------------------------------

class UncertaintyHead(nn.Module):
    """
    Lightweight MLP that predicts per-feature uncertainty scores.

    Given visual features x_i, computes:
        u_i = σ(g_φ(x_i))

    High u_i → high uncertainty → feature is suppressed in gating.
    """

    def __init__(self, feature_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, features: Tensor) -> Tensor:
        """
        Args:
            features: (B, N, D) visual feature tokens

        Returns:
            uncertainty: (B, N, 1) uncertainty scores in [0, 1]
        """
        logits = self.mlp(features)           # (B, N, 1)
        return torch.sigmoid(logits)          # u_i ∈ (0, 1)


# ---------------------------------------------------------------------------
# Uncertainty-Guided Feature Gating
# ---------------------------------------------------------------------------

class UncertaintyGating(nn.Module):
    """
    Suppresses unreliable visual activations using predicted uncertainty:

        F̃ = (1 - u) ⊙ F

    Regions with high uncertainty (blur, stamps, faded ink) contribute
    less to the downstream decoder.
    """

    def __init__(self, feature_dim: int):
        super().__init__()
        self.uncertainty_head = UncertaintyHead(feature_dim)

    def forward(self, features: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            features: (B, N, D) raw visual feature tokens

        Returns:
            gated_features: (B, N, D) uncertainty-modulated features
            uncertainty:    (B, N, 1) predicted uncertainty map
        """
        uncertainty = self.uncertainty_head(features)   # (B, N, 1)
        gated = (1.0 - uncertainty) * features          # suppression gate
        return gated, uncertainty


# ---------------------------------------------------------------------------
# Multi-Scale Visual Encoder
# ---------------------------------------------------------------------------

class VisualEncoder(nn.Module):
    """
    Multi-scale feature extractor for degraded document images.

    Combines a CNN backbone with a lightweight feature pyramid to capture
    both fine-grained stroke details (low-level) and global layout structure
    (high-level) — both critical for Thai cadastral document recognition.
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
        num_scales: int = 3,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_scales = num_scales

        # Stem: patch embedding (shared across scales conceptually)
        self.patch_embed = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )
        self.norm = nn.LayerNorm(embed_dim)

        # Multi-scale pooling branches
        self.scale_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim, kernel_size=3,
                          padding=2 ** i, dilation=2 ** i, groups=embed_dim),
                nn.Conv2d(embed_dim, embed_dim, kernel_size=1),
                nn.GELU(),
            )
            for i in range(num_scales)
        ])

        # Fusion
        self.fusion = nn.Linear(embed_dim * num_scales, embed_dim)

        # Positional encoding
        num_patches = (img_size // patch_size) ** 2
        self.pos_embedding = nn.Parameter(
            torch.randn(1, num_patches, embed_dim) * 0.02
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, 3, H, W) input document image

        Returns:
            features: (B, N, D) multi-scale visual tokens
        """
        # Patch embedding
        feats = self.patch_embed(x)                     # (B, D, H', W')
        B, D, H, W = feats.shape

        # Multi-scale feature extraction
        scale_outputs = [conv(feats) for conv in self.scale_convs]

        # Flatten and concatenate scales
        flat = [
            s.flatten(2).transpose(1, 2)                # (B, N, D)
            for s in scale_outputs
        ]
        multi = torch.cat(flat, dim=-1)                 # (B, N, D*S)
        fused = self.fusion(multi)                      # (B, N, D)

        # Add positional encoding
        fused = fused + self.pos_embedding
        return self.norm(fused)


# ---------------------------------------------------------------------------
# Transformer Decoder
# ---------------------------------------------------------------------------

class TransformerDecoder(nn.Module):
    """
    Auto-regressive transformer decoder for sequence generation.

    Cross-attends over uncertainty-gated visual features to produce
    character-level predictions.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 768,
        num_heads: int = 8,
        num_layers: int = 6,
        ff_dim: int = 2048,
        max_len: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_len = max_len

        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Embedding(max_len, embed_dim)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,        # Pre-LN for training stability
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers)
        self.output_proj = nn.Linear(embed_dim, vocab_size)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.pos_embedding.weight, std=0.02)
        nn.init.zeros_(self.output_proj.bias)

    def forward(
        self,
        tgt_tokens: Tensor,
        memory: Tensor,
        tgt_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            tgt_tokens:           (B, T) target token indices
            memory:               (B, N, D) encoder output (uncertainty-gated)
            tgt_key_padding_mask: (B, T) True where padding

        Returns:
            logits: (B, T, V) per-position character logits
        """
        B, T = tgt_tokens.shape
        positions = torch.arange(T, device=tgt_tokens.device).unsqueeze(0)

        tgt_emb = self.token_embedding(tgt_tokens) + self.pos_embedding(positions)

        # Causal mask
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            T, device=tgt_tokens.device
        )

        out = self.transformer(
            tgt=tgt_emb,
            memory=memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )
        return self.output_proj(out)                    # (B, T, V)


# ---------------------------------------------------------------------------
# CHULA-OCR Loss
# ---------------------------------------------------------------------------

class CHULAOCRLoss(nn.Module):
    """
    Custom Heuristic Uncertainty-Guided Loss for Accurate Land-title Recognition.

    L_CHULA = L_CE  +  λ_u * L_u  +  λ_h * L_h

    Components:
      L_CE  — standard token-level cross-entropy
      L_u   — uncertainty regularization (penalizes large gradients in noisy regions)
      L_h   — heuristic consistency (cadastral format constraints)
    """

    def __init__(
        self,
        lambda_u: float = 0.5,
        lambda_h: float = 0.3,
        alpha_entropy: float = 2.0,
        ignore_index: int = 0,
    ):
        super().__init__()
        self.lambda_u = lambda_u
        self.lambda_h = lambda_h
        self.alpha_entropy = alpha_entropy
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction="none")

    def compute_entropy_weights(self, logits: Tensor) -> Tensor:
        """
        Entropy-guided weighting:
            w_t = exp(-α * H_t)

        High-entropy (uncertain) positions receive lower weights.
        """
        probs = torch.softmax(logits, dim=-1)           # (B, T, V)
        H = -(probs * (probs + 1e-9).log()).sum(-1)     # (B, T)  entropy
        weights = torch.exp(-self.alpha_entropy * H)    # (B, T)
        return weights

    def uncertainty_regularization(
        self, uncertainty: Tensor, features: Tensor
    ) -> Tensor:
        """
        L_u = Σ_i u_i * ||∇_{x_i} log P(Y|I)||²

        Approximated as: uncertainty-weighted feature L2 norm.
        Encourages smooth gradients in ambiguous regions.
        """
        u = uncertainty.squeeze(-1)                     # (B, N)
        feat_norm = (features ** 2).sum(-1)             # (B, N)
        return (u * feat_norm).mean()

    def heuristic_consistency(self, logits: Tensor, targets: Tensor) -> Tensor:
        """
        L_h: penalize predictions that violate cadastral format rules.

        A simplified differentiable proxy: penalize sequence-level
        prediction confidence mismatch relative to ground-truth structure.
        """
        # Sequence-level confidence
        pred_conf = torch.softmax(logits, dim=-1).max(-1).values  # (B, T)
        tgt_mask = (targets != 0).float()                          # non-padding
        consistency = 1.0 - (pred_conf * tgt_mask).sum(-1) / (tgt_mask.sum(-1) + 1e-6)
        return consistency.mean()

    def forward(
        self,
        logits: Tensor,
        targets: Tensor,
        uncertainty: Tensor,
        raw_features: Tensor,
    ) -> Dict[str, Tensor]:
        """
        Args:
            logits:       (B, T, V) decoder output logits
            targets:      (B, T)   ground-truth token indices
            uncertainty:  (B, N, 1) per-feature uncertainty scores
            raw_features: (B, N, D) pre-gating features

        Returns:
            dict with individual loss components and total loss
        """
        B, T, V = logits.shape

        # ── L_CE with entropy weighting ──────────────────────────────────
        ce_per_token = self.ce(
            logits.reshape(B * T, V),
            targets.reshape(B * T)
        ).reshape(B, T)

        entropy_weights = self.compute_entropy_weights(logits)
        L_ce = (entropy_weights * ce_per_token).mean()

        # ── L_u : uncertainty regularization ─────────────────────────────
        L_u = self.uncertainty_regularization(uncertainty, raw_features)

        # ── L_h : heuristic consistency ───────────────────────────────────
        L_h = self.heuristic_consistency(logits, targets)

        # ── Total ─────────────────────────────────────────────────────────
        total = L_ce + self.lambda_u * L_u + self.lambda_h * L_h

        return {
            "loss": total,
            "loss_ce": L_ce,
            "loss_uncertainty": L_u,
            "loss_heuristic": L_h,
        }


# ---------------------------------------------------------------------------
# CHULA-OCR — Full Model
# ---------------------------------------------------------------------------

class CHULA_OCR(nn.Module):
    """
    CHULA-OCR: Uncertainty-Aware OCR for Thai Land Title Deed Digitization.

    Architecture (4 blocks):
      Block 1: Multi-scale Visual Encoder
      Block 2: Uncertainty Estimation Head
      Block 3: Uncertainty-Guided Feature Gating
      Block 4: Transformer Sequence Decoder

    Paper: https://kaopanboonyuen.github.io/CHULA-OCR/
    Author: Teerapong Panboonyuen (Kao), Chulalongkorn University
    """

    def __init__(
        self,
        vocab_size: int = 200,
        img_size: int = 224,
        patch_size: int = 16,
        embed_dim: int = 768,
        num_encoder_scales: int = 3,
        num_decoder_layers: int = 6,
        num_heads: int = 8,
        ff_dim: int = 2048,
        max_seq_len: int = 256,
        dropout: float = 0.1,
        # Loss hyperparameters
        lambda_u: float = 0.5,
        lambda_h: float = 0.3,
        alpha_entropy: float = 2.0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len

        # ── Block 1: Visual Encoder ───────────────────────────────────────
        self.encoder = VisualEncoder(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            num_scales=num_encoder_scales,
        )

        # ── Blocks 2 & 3: Uncertainty Estimation + Gating ─────────────────
        self.uncertainty_gating = UncertaintyGating(embed_dim)

        # ── Block 4: Transformer Decoder ──────────────────────────────────
        self.decoder = TransformerDecoder(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_decoder_layers,
            ff_dim=ff_dim,
            max_len=max_seq_len,
            dropout=dropout,
        )

        # ── Loss ──────────────────────────────────────────────────────────
        self.criterion = CHULAOCRLoss(
            lambda_u=lambda_u,
            lambda_h=lambda_h,
            alpha_entropy=alpha_entropy,
        )

        self._init_weights()
        self._log_params()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _log_params(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[CHULA-OCR] Total params    : {total:,}")
        print(f"[CHULA-OCR] Trainable params: {trainable:,}")

    def encode(self, images: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Forward pass through encoder + uncertainty gating.

        Returns:
            gated_features: (B, N, D)
            uncertainty:    (B, N, 1)
            raw_features:   (B, N, D)
        """
        raw_features = self.encoder(images)
        gated_features, uncertainty = self.uncertainty_gating(raw_features)
        return gated_features, uncertainty, raw_features

    def forward(
        self,
        images: Tensor,
        tgt_tokens: Tensor,
        tgt_padding_mask: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """
        Full forward pass for training.

        Args:
            images:           (B, 3, H, W)
            tgt_tokens:       (B, T) teacher-forced target tokens
            tgt_padding_mask: (B, T) True where padding

        Returns:
            dict: logits, loss components
        """
        # Encode
        gated, uncertainty, raw = self.encode(images)

        # Decode (teacher forcing: shift targets right)
        decoder_input = tgt_tokens[:, :-1]              # drop last token
        targets = tgt_tokens[:, 1:]                     # drop first token (BOS)

        logits = self.decoder(
            tgt_tokens=decoder_input,
            memory=gated,
            tgt_key_padding_mask=(
                tgt_padding_mask[:, :-1] if tgt_padding_mask is not None else None
            ),
        )

        # Compute loss
        losses = self.criterion(
            logits=logits,
            targets=targets,
            uncertainty=uncertainty,
            raw_features=raw,
        )

        return {"logits": logits, **losses}

    @torch.no_grad()
    def predict(
        self,
        images: Tensor,
        bos_token: int = 1,
        eos_token: int = 2,
        max_len: Optional[int] = None,
        return_uncertainty: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Greedy auto-regressive inference.

        Args:
            images:             (B, 3, H, W)
            bos_token:          BOS token index
            eos_token:          EOS token index
            max_len:            Maximum sequence length
            return_uncertainty: Whether to return uncertainty maps

        Returns:
            predictions: (B, T) predicted token sequences
            uncertainty: (B, N, 1) uncertainty map (if requested)
        """
        self.eval()
        max_len = max_len or self.max_seq_len
        B = images.size(0)
        device = images.device

        # Encode once
        gated, uncertainty, _ = self.encode(images)

        # Start with BOS
        tokens = torch.full((B, 1), bos_token, dtype=torch.long, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        for _ in range(max_len - 1):
            logits = self.decoder(tokens, gated)        # (B, t, V)
            next_token = logits[:, -1, :].argmax(-1, keepdim=True)  # (B, 1)
            tokens = torch.cat([tokens, next_token], dim=1)

            finished |= (next_token.squeeze(-1) == eos_token)
            if finished.all():
                break

        if return_uncertainty:
            return tokens, uncertainty
        return tokens, None

    @classmethod
    def from_pretrained(cls, checkpoint_path: str, **kwargs) -> "CHULA_OCR":
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        config = checkpoint.get("config", {})
        config.update(kwargs)
        model = cls(**config)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"[CHULA-OCR] Loaded checkpoint from: {checkpoint_path}")
        return model

    def save_checkpoint(self, path: str, config: dict, epoch: int, metrics: dict):
        """Save model checkpoint with metadata."""
        torch.save({
            "model_state_dict": self.state_dict(),
            "config": config,
            "epoch": epoch,
            "metrics": metrics,
        }, path)
        print(f"[CHULA-OCR] Checkpoint saved to: {path}")


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  CHULA-OCR — Sanity Check")
    print("  Author: Teerapong Panboonyuen (Kao), Ph.D.")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    model = CHULA_OCR(
        vocab_size=200,
        img_size=224,
        patch_size=16,
        embed_dim=512,
        num_decoder_layers=4,
        num_heads=8,
    ).to(device)

    # Dummy batch
    B = 2
    images = torch.randn(B, 3, 224, 224, device=device)
    tokens = torch.randint(3, 200, (B, 32), device=device)
    tokens[:, 0] = 1   # BOS
    tokens[:, -1] = 2  # EOS

    # Training forward
    out = model(images, tokens)
    print(f"\n✅ Training forward pass OK")
    print(f"   Logits shape : {out['logits'].shape}")
    print(f"   Total loss   : {out['loss'].item():.4f}")
    print(f"   CE loss      : {out['loss_ce'].item():.4f}")
    print(f"   Unc. loss    : {out['loss_uncertainty'].item():.4f}")
    print(f"   Heuristic    : {out['loss_heuristic'].item():.4f}")

    # Inference
    preds, unc = model.predict(images, return_uncertainty=True)
    print(f"\n✅ Inference forward pass OK")
    print(f"   Predictions shape  : {preds.shape}")
    print(f"   Uncertainty shape  : {unc.shape}")
    print(f"   Mean uncertainty   : {unc.mean().item():.4f}")

    print("\n✅ All checks passed. CHULA-OCR is ready.\n")
