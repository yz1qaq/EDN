# EchoDynamicsNet (EDN)
This repository provides a PyTorch implementation of **EchoDynamicsNet (EDN)** for **fine-grained Chinese simile sentiment classification**.

EDN aims to jointly model:
- **Global semantic consistency** via an **Echo Enhancement Network (EEN)**, and  
- **Local emotion fluctuation patterns** via a **Fluctuation-Aware Convolution (FAC)** branch guided by multi-granularity fluctuation representations.



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
 
@article{YUAN2026116401,
title = {EchoDynamicsNet: A Fluctuation-Aware Echo-Enhanced Network for Chinese simile sentiment classification},
journal = {Engineering Applications of Artificial Intelligence},
volume = {184},
pages = {116401},
year = {2026},
issn = {0952-1976},
doi = {https://doi.org/10.1016/j.engappai.2026.116401},
url = {https://www.sciencedirect.com/science/article/pii/S0952197626026850},
author = {Zhaoyi Yuan and Jia Liu and Xipeng Yuan and Wei Huang and Pengfei Zhang and Tianrui Li},
keywords = {Chinese simile sentiment classification, Sentiment analysis, Echo enhancement, Dynamic convolution, Deep learning}
