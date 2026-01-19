# EchoDynamicsNet (EDN)
This repository provides a PyTorch implementation of **EchoDynamicsNet (EDN)** for **fine-grained Chinese simile sentiment classification**.

EDN aims to jointly model:
- **Global semantic consistency** via an **Echo Enhancement Network (EEN)**, and  
- **Local emotion fluctuation patterns** via a **Fluctuation-Aware Convolution (FAC)** branch guided by multi-granularity fluctuation representations.

---
## Framework Overview

<img src="01EDN.PDF" width="700">

The overall architecture of the proposed EDN model.



## Requirements

- Python **3.10**
- PyTorch **>= 1.12**
- transformers **>= 4.30**
- pandas **>= 1.5**
- openpyxl **>= 3.0**
- scikit-learn **>= 1.0**
- tqdm **>= 4.0**
- matplotlib **>= 3.0**

---

## Dataset (NOT Open-Sourced)

The dataset is provided by the **CCAC (Chinese Conference on Affective Computing)**.

⚠️ **Important:** The dataset is **NOT open-sourced** and is **NOT included** in this repository.  
Readers must **contact CCAC officially** to request access:

https://ccac2025.xhu.edu.cn/

### Data Format

Each split is stored in an Excel file and should contain the following columns:

- `句子序号`
- `句子`
- `情感`

Default paths used in the code:

- `DATA/train_data.xlsx`
- `DATA/valid_split_valid.xlsx`
- `DATA/valid_split_test.xlsx`

---

## Pretrained Encoder

We use **Chinese RoBERTa with Whole Word Masking**:

- `hfl/chinese-roberta-wwm-ext`  
  https://huggingface.co/hfl/chinese-roberta-wwm-ext

You can either:
- load it directly from HuggingFace by setting `text_encoder_name = "hfl/chinese-roberta-wwm-ext"`, or
- download it and set `text_encoder_name` to a local directory (e.g., `pre_model/chinese-roberta-wwm-ext`).

---

## Repository Structure

---

## Training & Evaluation

### Step 1: Configure paths and switches

Edit `main.py` to set:
- `text_encoder_name`
- dataset paths under `DATA/`
- EDN switches:
  - `use_een`: enable/disable EEN
  - `use_fac`: enable/disable FAC

### Step 2: Run training

```bash
python main.py
