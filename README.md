# CHULA-OCR: Uncertainty-Aware Robust OCR for National-Scale Thai Land Title Deed Digitization

> **CHULA-OCR** is a robustness-first OCR framework designed for real-world, highly degraded legal documents—specifically Thai land title deeds—where traditional OCR systems fail due to noise, handwriting, stamps, and character ambiguity.

---

## 🔥 Why CHULA-OCR Matters

Modern OCR systems are surprisingly fragile in the real world.

Even state-of-the-art models like Transformer-based OCR or OCR-free document models struggle when confronted with:

- Faded ink in historical land records  
- Handwritten annotations over printed text  
- Stamps and seals overlapping text regions  
- Visually similar Thai glyphs (e.g., ฎ vs ฏ)  
- Severe scanning degradation in government archives  

In high-stakes legal settings such as **land ownership verification**, even small OCR errors propagate into **critical administrative failures**.

👉 CHULA-OCR is designed to solve exactly this.

---

## 🇹🇭 National-Scale Motivation

This project is motivated by real deployment needs from Thailand’s land administration system.

The goal is:

> 🏛️ **Enable reliable, automated digitization of Thai land title deeds at national scale**

This includes:

- Land ownership record digitization  
- Automated cadastral database construction  
- Reducing manual verification workload  
- Supporting digital government transformation  

CHULA-OCR is not just a model—it is a **national infrastructure component for document intelligence**.

---

## 🧠 Key Idea

> Instead of treating OCR as deterministic prediction, CHULA-OCR models **uncertainty explicitly**.

We ask:

> *“Which characters should the model trust when the document itself is unreliable?”*

---

### ✨ Core Innovation: Uncertainty-Aware OCR

We introduce:

- 🔹 Token-level uncertainty estimation  
- 🔹 Entropy-guided learning  
- 🔹 Feature gating for noisy regions  
- 🔹 Structure-aware decoding for legal documents  

### Objective

\[
\mathcal{L} = \mathcal{L}_{CE} + \lambda_u \mathcal{L}_{uncertainty} + \lambda_h \mathcal{L}_{structure}
\]

This allows CHULA-OCR to:

- Focus on reliable strokes  
- Suppress noisy stamps/background artifacts  
- Maintain structural consistency in cadastral codes  

---

## 🏗️ Architecture Overview

### Pipeline:

1. **Visual Encoder**
   - Extracts multi-scale document features
   - Handles stamps, blur, handwriting, noise

2. **Uncertainty Estimation Head**
   - Predicts unreliable regions in the image
   - Outputs confidence-aware feature masks

3. **Uncertainty-Gated Features**
   - Suppresses ambiguous visual tokens
   - Enhances stable text regions

4. **Transformer Decoder**
   - Generates structured text sequences
   - Optimized for cadastral formatting consistency

---

## 📊 Key Results

### 🧾 Overall Performance

| Model | Accuracy | Seq Acc | NED ↓ |
|------|--------|--------|------|
| CRNN | 35.2 | 21.4 | 3.92 |
| TrOCR | 49.1 | 35.6 | 2.88 |
| Donut | 52.3 | 38.2 | 2.61 |
| **CHULA-OCR** | **64.2** | **55.8** | **1.64** |

---

### 🧩 Robustness Under Degradation

| Condition | CHULA-OCR |
|----------|----------|
| Clean | **73.1%** |
| Moderate Noise | **65.7%** |
| Severe Degradation | **52.0%** |

👉 Lowest performance drop among all methods

---

### 🧾 Hard Cadastral Identifier Parsing

| Model | Exact Match |
|------|------------|
| TrOCR | 37.5% |
| Donut | 40.3% |
| **CHULA-OCR** | **58.2%** |

✔ Critical improvement for legal-grade document correctness

---

## 🔍 What Makes CHULA-OCR Different?

Unlike prior OCR systems:

### ❌ Traditional OCR
- Treats all pixels equally  
- Assumes clean input  
- Fails under real-world noise  

### ❌ OCR-free models (e.g., Donut-style)
- Skip explicit text modeling  
- Lose character-level precision  
- Not suitable for legal systems  

### ❌ Post-OCR correction methods
- Fix errors after prediction  
- Cannot recover lost visual information  

---

### ✅ CHULA-OCR (Ours)

- Models **uncertainty during perception**
- Filters unreliable visual evidence
- Preserves **legal structure integrity**
- Designed for **real government deployment**

---

## 🧪 Real-World Insight

From qualitative analysis:

- Baselines confuse Thai glyph pairs under blur  
- Handwritten overlays break sequence models  
- Stamp interference causes catastrophic decoding errors  

CHULA-OCR:

✔ Maintains stable predictions  
✔ Reduces character-level confusion  
✔ Preserves cadastral identifier structure  

---

## 🌍 Impact

CHULA-OCR enables:

- 🏛️ National land registry digitization  
- 📄 Automated legal document processing  
- ⚡ Reduced manual verification cost  
- 📊 Scalable government AI infrastructure  

---

## 📌 Conclusion

CHULA-OCR demonstrates that:

> **Robust OCR for legal documents requires uncertainty-aware perception, not just stronger backbones.**

It bridges the gap between:

- Academic OCR benchmarks  
- Real-world government document systems  

and provides a **deployable solution for Thai land title deed digitization at national scale**.

---

## 📎 Citation

If you use this work, please cite:

```bibtex
@article{chula_ocr_2026,
  title={CHULA-OCR: Uncertainty-Aware Robust Text Recognition for National-Scale Land Title Deed Digitization},
  author={Panboonyuen, Teerapong},
  year={2026}
}