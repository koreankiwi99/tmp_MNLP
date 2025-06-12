# DPO – CS552 Modern NLP (Spring 2025, EPFL)

This repository contains my personal work on **Direct Preference Optimization (DPO)** for the final project in **Modern Natural Language Processing (CS552)** at **EPFL**, Spring 2025.
> 🏫 Developed as part of [EPFL CS552 – Modern Natural Language Processing (Spring 2025)](https://github.com/cs-552)

📄 **[FINALREPORT.pdf](FINALREPORT.pdf)** – Revised version of the final report (uploaded here for portfolio reference only; the original was submitted through the team repo before the deadline).  
📝 **[proposal.pdf](proposal.pdf)** – Initial project proposal outlining the goals, plan, and methodology.  
📈 **[progress_report.pdf](progress_report.pdf)** – Mid-project progress report describing partial results and individual contributions.

---

## 🔍 Overview

The goal of this project was to implement and evaluate DPO based on **Qwen3-0.6B-Base**, using diverse pairwise preference data.

This repo includes:
- DPO, pre-DPO, and SFT training scripts
- Dataset analysis and preprocessing utilities
- Evaluation modules (LightEval-compatible)
- Configs and helper scripts

---

## 🤗 Hugging Face Collections

- **Milestone 3 – Trained DPO Models**  
  👉 [View Collection on Hugging Face](https://huggingface.co/collections/koreankiwi99/2025-mnlp-m3-dpo-6847624db5964ddef0fb0b4b)

- **Evaluation Dataset (LightEval-compatible)**  
  👉 [View Dataset Collection](https://huggingface.co/collections/koreankiwi99/epfl-lighteval-dpo-datasets-68474bc94137e41ade8e560f)  
  ↪ Compatible with [**LightEval**](https://github.com/eric11eca/lighteval-epfl-mnlp)

- **Milestone 2 – Baseline Models**  
  👉 [View Baseline Models](https://huggingface.co/collections/koreankiwi99/2025-mnlp-m2-dpo-68357712650d51732ca5e7c5)

---

## 🚀 Quick Start (Training Scripts)

Install dependencies:

```bash
pip install transformers peft trl accelerate
````
### 🔧 Run SFT Training

```bash
python train_sft.py \
  --config_path <path_to_config_json>         # e.g., MNLP/config/sft_base.json \
  --sft_dataset_name <sft_dataset_name>       # e.g., koreankiwi99/mnlp_stem_curriculum
```

### 🔧 Run DPO Training

```bash
python train_dpo.py \
  --config_path <path_to_config_json>         # e.g., MNLP/config/lower_beta.json \
  --hf_dataset_name <dataset_name>            # e.g., koreankiwi99/mnlp_aggregate \
  --model_name <base_or_sft_model>            # e.g., koreankiwi99/sft_model_sft_base_mnlp_stem_balanced_plus
```

### 🔧 Run Pre-DPO Preference Model Training

```bash
python train_predpo.py \
  --config_path <path_to_config_json>         # e.g., MNLP/config/predpo_lower_beta.json \
  --hf_dataset_name <dataset_name>            # e.g., koreankiwi99/mnlp_aggregate \
  --model_name <base_or_sft_model>            # e.g., koreankiwi99/sft_model_sft_base_mnlp_stem_balanced \
  --ref_model_name <dpo_or_simpo_model>       # e.g., koreankiwi99/sft_model_sft_base_mnlp_stem_balanced_lower_beta_mnlp_aggregate
```

---

## 📁 Repo Structure

```
.
├── config/              # Training configs
├── data/                # Subsets of EPFL Dataset
├── evaluation/          # Convert public benchmark to LightEval format
├── legacy/              # Deprecated or testing code
├── preprocessing/       # Combining SFT datasets
├── check_overlap.py     # Utility for overlap analysis
├── dataset_stats.py     # Dataset statistics and diagnostics
├── train_dpo.py         # Main DPO training script
├── train_predpo.py      # Pre-DPO preference modeling
├── train_sft.py         # Supervised fine-tuning script
├── FINALREPORT.pdf
├── proposal.pdf
├── progress_report.pdf
└── README.md
```

---

## 👩🏻‍💻 About

**Kyuhee Kim**
Master’s in Data Science @ EPFL
🌐 [GitHub](https://github.com/koreankiwi99) | 🤗 [Hugging Face](https://huggingface.co/koreankiwi99)
