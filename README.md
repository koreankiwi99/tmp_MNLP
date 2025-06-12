# DPO – CS552 Modern NLP (Spring 2025, EPFL)

This repository contains my personal work on **Direct Preference Optimization (DPO)** for the final project in **Modern Natural Language Processing (CS552)** at **EPFL**, Spring 2025.

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
  🔗 https://huggingface.co/collections/koreankiwi99/2025-mnlp-m3-dpo-6847624db5964ddef0fb0b4b

- **Evaluation Dataset (LightEval-compatible)**  
  🔗 https://huggingface.co/collections/koreankiwi99/epfl-lighteval-dpo-datasets-68474bc94137e41ade8e560f

- **Milestone 2 – Baseline Models**  
  🔗 https://huggingface.co/collections/koreankiwi99/2025-mnlp-m2-dpo-68357712650d51732ca5e7c5

---

## 📁 Repo Structure

```

.
├── MNLP/
│   ├── config/
│   ├── data/
│   ├── evaluation/
│   ├── legacy/
│   ├── preprocessing/
│   ├── check\_overlap.py
│   └── dataset\_stats.py
├── train\_dpo.py
├── train\_predpo.py
├── train\_sft.py
├── FINALREPORT.pdf
├── proposal.pdf
├── progress\_report.pdf
└── README.md

```

---

## 👩🏻‍💻 About

**Kyuhee Kim**  
Master’s in Data Science @ EPFL  
---
