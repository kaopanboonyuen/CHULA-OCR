"""
CHULA-OCR Dataset — Thai Land Title Deed Loader

Author: Teerapong Panboonyuen (Kao), Ph.D.
        C2F Postdoctoral Fellow, Chulalongkorn University

⚠️  PDPA NOTICE: This dataset is governed by Thailand's PDPA B.E. 2562.
    All documents are fully anonymized. Data is the exclusive property of
    the Thai Department of Lands and cannot be redistributed.
    See PDPA_COMPLIANCE.md for full declaration.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T


# ---------------------------------------------------------------------------
# Thai Character Tokenizer
# ---------------------------------------------------------------------------

# Thai consonants (44), vowel forms (subset), digits, special tokens
THAI_CONSONANTS = (
    "กขฃคฅฆงจฉชซฌญฎฏฐฑฒณดตถทธนบปผฝพฟภมยรลวศษสหฬอฮ"
)
THAI_VOWELS = "าิีึืุูเแโใไ็่้๊๋ัํๆๅ"
THAI_DIGITS = "๐๑๒๓๔๕๖๗๘๙"
ARABIC_DIGITS = "0123456789"
SPECIAL_CHARS = " /-.()"
PUNCTUATION = ".,;:!?\"'"

FULL_VOCAB = (
    ["<PAD>", "<BOS>", "<EOS>", "<UNK>"]
    + list(THAI_CONSONANTS)
    + list(THAI_VOWELS)
    + list(THAI_DIGITS)
    + list(ARABIC_DIGITS)
    + list(SPECIAL_CHARS)
    + list(PUNCTUATION)
    + list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    + list("abcdefghijklmnopqrstuvwxyz")
)

PAD_IDX = 0
BOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3


class ThaiOCRTokenizer:
    """
    Character-level tokenizer for Thai land title deed OCR.

    Supports: Thai consonants, vowel forms, tonal markers,
              Thai & Arabic digits, Latin alphabet, and special symbols.
    """

    def __init__(self, vocab: Optional[List[str]] = None):
        self.vocab = vocab or FULL_VOCAB
        self.char2idx = {c: i for i, c in enumerate(self.vocab)}
        self.idx2char = {i: c for i, c in enumerate(self.vocab)}
        self.pad_idx = PAD_IDX
        self.bos_idx = BOS_IDX
        self.eos_idx = EOS_IDX
        self.unk_idx = UNK_IDX

    def __len__(self) -> int:
        return len(self.vocab)

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        tokens = [self.char2idx.get(c, self.unk_idx) for c in text]
        if add_special_tokens:
            tokens = [self.bos_idx] + tokens + [self.eos_idx]
        return tokens

    def decode(self, indices: List[int], skip_special: bool = True) -> str:
        special = {self.pad_idx, self.bos_idx, self.eos_idx}
        chars = []
        for i in indices:
            if skip_special and i in special:
                continue
            if i == self.eos_idx:
                break
            chars.append(self.idx2char.get(i, "<UNK>"))
        return "".join(chars)

    def batch_encode(
        self,
        texts: List[str],
        max_len: int = 256,
        pad: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode a batch of strings.

        Returns:
            tokens:       (B, max_len) padded token indices
            padding_mask: (B, max_len) True where padding
        """
        encoded = [self.encode(t) for t in texts]
        lengths = [min(len(e), max_len) for e in encoded]

        if pad:
            tokens = torch.full((len(texts), max_len), self.pad_idx, dtype=torch.long)
            mask = torch.ones((len(texts), max_len), dtype=torch.bool)
            for i, (enc, L) in enumerate(zip(encoded, lengths)):
                tokens[i, :L] = torch.tensor(enc[:L], dtype=torch.long)
                mask[i, :L] = False
        else:
            tokens = [torch.tensor(e[:max_len]) for e in encoded]
            mask = None

        return tokens, mask

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "ThaiOCRTokenizer":
        with open(path, encoding="utf-8") as f:
            vocab = json.load(f)
        return cls(vocab)


# ---------------------------------------------------------------------------
# Document-Specific Augmentations
# ---------------------------------------------------------------------------

class CadastralAugmentation:
    """
    Augmentation pipeline tailored for degraded land title deed documents.

    Simulates: scanning artifacts, ink fading, handwriting variability,
    stamp overlays, and physical document aging.
    """

    def __init__(
        self,
        rotation_deg: float = 5.0,
        blur_sigma_range: Tuple[float, float] = (0.1, 2.0),
        noise_std: float = 0.03,
        elastic_alpha: float = 34.0,
        elastic_sigma: float = 4.0,
        cutout_ratio: float = 0.1,
        p: float = 0.5,
    ):
        self.rotation_deg = rotation_deg
        self.blur_sigma_range = blur_sigma_range
        self.noise_std = noise_std
        self.elastic_alpha = elastic_alpha
        self.elastic_sigma = elastic_sigma
        self.cutout_ratio = cutout_ratio
        self.p = p

    def __call__(self, image: np.ndarray) -> np.ndarray:
        if random.random() > self.p:
            return image

        ops = [
            self._random_rotation,
            self._perspective_distortion,
            self._gaussian_blur,
            self._gaussian_noise,
            self._brightness_contrast_jitter,
            self._elastic_deformation,
            self._cutout,
            self._ink_fade,
        ]
        random.shuffle(ops)
        n_ops = random.randint(1, 4)
        for op in ops[:n_ops]:
            image = op(image)
        return image.clip(0, 255).astype(np.uint8)

    def _random_rotation(self, img: np.ndarray) -> np.ndarray:
        angle = random.uniform(-self.rotation_deg, self.rotation_deg)
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

    def _perspective_distortion(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        margin = int(min(h, w) * 0.05)
        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dst = np.float32([
            [random.randint(0, margin), random.randint(0, margin)],
            [w - random.randint(0, margin), random.randint(0, margin)],
            [w - random.randint(0, margin), h - random.randint(0, margin)],
            [random.randint(0, margin), h - random.randint(0, margin)],
        ])
        M = cv2.getPerspectiveTransform(src, dst)
        return cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

    def _gaussian_blur(self, img: np.ndarray) -> np.ndarray:
        sigma = random.uniform(*self.blur_sigma_range)
        k = int(sigma * 4) | 1  # odd kernel
        return cv2.GaussianBlur(img, (k, k), sigma)

    def _gaussian_noise(self, img: np.ndarray) -> np.ndarray:
        noise = np.random.normal(0, self.noise_std * 255, img.shape)
        return (img.astype(np.float32) + noise)

    def _brightness_contrast_jitter(self, img: np.ndarray) -> np.ndarray:
        alpha = random.uniform(0.7, 1.3)   # contrast
        beta = random.uniform(-30, 30)     # brightness
        return alpha * img.astype(np.float32) + beta

    def _elastic_deformation(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        dx = cv2.GaussianBlur(
            np.random.randn(h, w).astype(np.float32),
            (0, 0), self.elastic_sigma
        ) * self.elastic_alpha
        dy = cv2.GaussianBlur(
            np.random.randn(h, w).astype(np.float32),
            (0, 0), self.elastic_sigma
        ) * self.elastic_alpha
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        map_x = (x + dx).astype(np.float32)
        map_y = (y + dy).astype(np.float32)
        return cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REPLICATE)

    def _cutout(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        ch, cw = int(h * self.cutout_ratio), int(w * self.cutout_ratio)
        y0 = random.randint(0, h - ch)
        x0 = random.randint(0, w - cw)
        img = img.copy()
        img[y0:y0 + ch, x0:x0 + cw] = random.randint(200, 255)
        return img

    def _ink_fade(self, img: np.ndarray) -> np.ndarray:
        """Simulate aged ink fading."""
        fade = random.uniform(0.6, 1.0)
        return img.astype(np.float32) * fade + (1 - fade) * 240


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ThaiLandTitleDeedDataset(Dataset):
    """
    Thai Land Title Deed OCR Dataset.

    ⚠️  DATA GOVERNANCE NOTICE:
        This dataset is exclusively owned by the Thai Department of Lands
        and is governed under Thailand PDPA B.E. 2562.
        All PII has been removed. Data cannot be redistributed.

    Dataset structure:
        data_dir/
            images/
                sample_00001.jpg
                sample_00002.jpg
                ...
            annotations.json   # {"filename": "label text", ...}

    Statistics (full dataset):
        - ~20,000 documents
        - ~1,100,000 cropped text instances
        - ~10,000,000 annotated characters
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",                  # train / val / test
        tokenizer: Optional[ThaiOCRTokenizer] = None,
        img_size: int = 224,
        max_seq_len: int = 256,
        augment: bool = True,
        augment_prob: float = 0.5,
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.tokenizer = tokenizer or ThaiOCRTokenizer()
        self.img_size = img_size
        self.max_seq_len = max_seq_len

        # Load annotations
        ann_path = self.data_dir / f"{split}_annotations.json"
        with open(ann_path, encoding="utf-8") as f:
            raw = json.load(f)

        self.samples = [
            {"image_path": self.data_dir / "images" / k, "label": v}
            for k, v in raw.items()
        ]

        # Augmentation
        self.augmentor = CadastralAugmentation(p=augment_prob) if augment else None

        # Base transforms (always applied)
        self.base_transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

        print(f"[Dataset] Split: {split} | Samples: {len(self.samples):,}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        # Load image
        img = cv2.imread(str(sample["image_path"]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Augment
        if self.augmentor is not None and self.split == "train":
            img = self.augmentor(img)

        # To PIL → transform
        pil_img = Image.fromarray(img.astype(np.uint8))
        tensor_img = self.base_transform(pil_img)

        # Tokenize label
        label_text = sample["label"]
        encoded = self.tokenizer.encode(label_text, add_special_tokens=True)
        encoded = encoded[: self.max_seq_len]

        # Pad
        padded = torch.full((self.max_seq_len,), PAD_IDX, dtype=torch.long)
        padded[: len(encoded)] = torch.tensor(encoded, dtype=torch.long)
        padding_mask = padded == PAD_IDX

        return {
            "image": tensor_img,           # (3, H, W)
            "tokens": padded,              # (max_seq_len,)
            "padding_mask": padding_mask,  # (max_seq_len,)
            "label": label_text,
            "image_path": str(sample["image_path"]),
        }

    @staticmethod
    def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
        return {
            "image": torch.stack([b["image"] for b in batch]),
            "tokens": torch.stack([b["tokens"] for b in batch]),
            "padding_mask": torch.stack([b["padding_mask"] for b in batch]),
            "label": [b["label"] for b in batch],
            "image_path": [b["image_path"] for b in batch],
        }


# ---------------------------------------------------------------------------
# Demo tokenizer stats
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tok = ThaiOCRTokenizer()
    print(f"Vocabulary size : {len(tok)}")

    sample = "โฉนดที่ดิน เลขที่ 12345 แขวงลาดยาว"
    enc = tok.encode(sample)
    dec = tok.decode(enc)
    print(f"Original  : {sample}")
    print(f"Encoded   : {enc}")
    print(f"Decoded   : {dec}")
    print(f"✅ Roundtrip OK: {sample == dec}")
