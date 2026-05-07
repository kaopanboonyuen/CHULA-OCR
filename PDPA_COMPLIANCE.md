# 🔒 Data Privacy & PDPA Compliance Declaration

## CHULA-OCR — ประกาศการคุ้มครองข้อมูลส่วนบุคคล

---

## English Declaration

### Overview

The CHULA-OCR project processes Thai land title deed documents in collaboration with the **Thai Department of Lands** (กรมที่ดิน). This document formally declares the data governance, privacy compliance framework, and legal basis under which all data in this project is collected, processed, and stored.

---

### 1. Legal Framework

This project fully complies with:

- 🇹🇭 **Thailand's Personal Data Protection Act B.E. 2562 (PDPA)** — พระราชบัญญัติคุ้มครองข้อมูลส่วนบุคคล พ.ศ. 2562
- **Government Data Governance Protocols** of the Thai Department of Lands
- **Chulalongkorn University Research Ethics Standards**

---

### 2. Data Ownership & Custodianship

> ⚠️ **The dataset used in CHULA-OCR is exclusively owned and governed by the Thai Department of Lands (กรมที่ดิน), Ministry of Interior, Kingdom of Thailand.**

- All raw documents, scanned images, and original land title deeds **remain the legal property of the Royal Thai Government**.
- Chulalongkorn University holds a **research-use license only**, with no rights to redistribute, sell, or sublicense any data.
- This dataset is **not publicly available** and **cannot be released, shared, or distributed** under any circumstance without explicit written authorization from the Thai Department of Lands.

---

### 3. Personally Identifiable Information (PII) Removal

All documents underwent a rigorous **de-identification and anonymization process** prior to any research use:

| PII Category | Treatment |
|---|---|
| 👤 Owner Full Names | Fully masked / removed |
| 🪪 National ID Numbers | Fully masked / removed |
| 📍 Home Addresses | Fully masked / removed |
| 📞 Contact Information | Fully masked / removed |
| 🖊️ Handwritten Signatures | Redacted |
| 📅 Sensitive Date Fields | Generalized or removed |
| 🗺️ Parcel GPS Coordinates | Generalized to district level only |

De-identification was performed by the **Thai Department of Lands** prior to dataset delivery. The research team at Chulalongkorn University **never received, processed, or stored** any non-anonymized documents.

---

### 4. Data Access Controls

- 🔐 Access is restricted to **approved researchers** on the project team only
- 🔐 Data is stored on **secured, air-gapped** servers at Chulalongkorn University
- 🔐 All access events are **logged and auditable**
- 🔐 Data is encrypted at rest (AES-256) and in transit (TLS 1.3)
- 🔐 No data is transferred to cloud services, third parties, or external collaborators without explicit authorization

---

### 5. Research Purpose Limitation

This dataset is used **solely** for the following approved research purpose:

> Development and evaluation of the CHULA-OCR uncertainty-aware OCR system for the purpose of improving national-scale digitization of Thai land title deeds within the Thai Department of Lands' governmental digitization program.

Any use outside this stated purpose is **prohibited** and would constitute a violation of the data use agreement and Thai PDPA.

---

### 6. Data Retention & Destruction

- Research copies of the anonymized dataset will be **retained only for the duration** of the approved research project
- Upon project completion, all dataset copies will be **securely deleted** according to DoL data destruction protocols
- Model weights trained on this data may be retained for ongoing research under continued authorization

---

### 7. Subject Rights

Although the dataset is fully anonymized and no individual can be re-identified from the data:

- The Thai Department of Lands remains the **data controller** for all original documents
- Any data subject rights requests (e.g., erasure, correction) should be directed to the **Thai Department of Lands** directly

**Contact (Data Controller):**
> กรมที่ดิน (Department of Lands)
> กระทรวงมหาดไทย (Ministry of Interior)
> Website: https://www.dol.go.th

---

### 8. Model Outputs & Inference

- The **trained model weights** (CHULA-OCR) do not memorize or reconstruct any original document content
- Inference outputs are **character-level text predictions** — they contain no PII
- Public model releases (if any) will be reviewed by the Thai Department of Lands prior to publication

---

### 9. Contact for Privacy Concerns

For any privacy-related questions regarding this research:

**Research Team Contact:**
> Teerapong Panboonyuen (Kao), Ph.D.
> C2F Postdoctoral Fellow, Chulalongkorn University
> teerapong.panboonyuen@gmail.com

---

---

## ประกาศภาษาไทย (Thai Language Declaration)

### ภาพรวม

โครงการ CHULA-OCR ดำเนินการประมวลผลเอกสารโฉนดที่ดินของไทยโดยร่วมมือกับ **กรมที่ดิน กระทรวงมหาดไทย** ประกาศนี้ระบุกรอบการกำกับดูแลข้อมูล ความเป็นส่วนตัว และฐานทางกฎหมายที่ใช้ในการรวบรวม ประมวลผล และจัดเก็บข้อมูลทั้งหมดในโครงการนี้

---

### 1. กรอบทางกฎหมาย

โครงการนี้ปฏิบัติตามกฎหมายต่อไปนี้อย่างครบถ้วน:

- **พระราชบัญญัติคุ้มครองข้อมูลส่วนบุคคล พ.ศ. 2562 (PDPA)**
- **ระเบียบการกำกับดูแลข้อมูลของกรมที่ดิน**
- **มาตรฐานจริยธรรมการวิจัยของจุฬาลงกรณ์มหาวิทยาลัย**

---

### 2. ความเป็นเจ้าของข้อมูล

> ⚠️ **ข้อมูลทั้งหมดที่ใช้ในโครงการ CHULA-OCR เป็นกรรมสิทธิ์และอยู่ภายใต้การกำกับดูแลของกรมที่ดิน กระทรวงมหาดไทย แห่งราชอาณาจักรไทยเท่านั้น**

- เอกสารต้นฉบับ ภาพสแกน และโฉนดที่ดินทั้งหมด **ยังคงเป็นทรัพย์สินทางกฎหมายของรัฐบาลไทย**
- จุฬาลงกรณ์มหาวิทยาลัยมีสิทธิ์ **ใช้เพื่อการวิจัยเท่านั้น** โดยไม่มีสิทธิ์เผยแพร่ ขาย หรืออนุญาตช่วงข้อมูลใดๆ
- **ข้อมูลนี้ไม่สามารถเผยแพร่สู่สาธารณะได้** และ **ไม่สามารถแจกจ่ายได้** ไม่ว่ากรณีใดก็ตาม หากไม่ได้รับอนุญาตเป็นลายลักษณ์อักษรจากกรมที่ดิน

---

### 3. การลบข้อมูลส่วนบุคคล (PII)

เอกสารทั้งหมดผ่านกระบวนการ **ลบและปกปิดข้อมูลส่วนบุคคลอย่างเข้มงวด** ก่อนนำมาใช้ในการวิจัย:

| ประเภทข้อมูลส่วนบุคคล | วิธีการจัดการ |
|---|---|
| 👤 ชื่อ-นามสกุลเจ้าของ | ปิดข้อมูล / ลบออกทั้งหมด |
| 🪪 เลขประจำตัวประชาชน | ปิดข้อมูล / ลบออกทั้งหมด |
| 📍 ที่อยู่อาศัย | ปิดข้อมูล / ลบออกทั้งหมด |
| 📞 ข้อมูลติดต่อ | ปิดข้อมูล / ลบออกทั้งหมด |
| 🖊️ ลายเซ็นลายมือ | ลบออก |
| 📅 วันที่ที่ละเอียดอ่อน | ทำให้คลุมเครือหรือลบออก |
| 🗺️ พิกัด GPS แปลงที่ดิน | ทำให้คลุมเครือในระดับอำเภอเท่านั้น |

การลบข้อมูลส่วนบุคคลดำเนินการโดย **กรมที่ดิน** ก่อนส่งมอบข้อมูล ทีมวิจัยของจุฬาลงกรณ์มหาวิทยาลัย **ไม่เคยได้รับ ประมวลผล หรือจัดเก็บ** เอกสารที่ไม่ผ่านการปกปิดข้อมูล

---

### 4. วัตถุประสงค์การวิจัย

ข้อมูลนี้ใช้ **เฉพาะ** เพื่อวัตถุประสงค์การวิจัยที่ได้รับอนุมัติดังนี้:

> การพัฒนาและประเมินระบบ CHULA-OCR ที่คำนึงถึงความไม่แน่นอนในการรู้จำตัวอักษร เพื่อปรับปรุงการแปลงโฉนดที่ดินไทยเป็นดิจิทัลในระดับประเทศ ภายใต้โครงการดิจิทัลภาครัฐของกรมที่ดิน

---

### 5. ติดต่อสอบถาม

**ผู้ควบคุมข้อมูล:**
> กรมที่ดิน กระทรวงมหาดไทย
> https://www.dol.go.th

---

<div align="center">

*This declaration is maintained in compliance with PDPA B.E. 2562 and updated as required.*

**Last reviewed: 2025**

</div>
