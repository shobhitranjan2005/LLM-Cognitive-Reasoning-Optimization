<div align="center">

# 🧠 LLM Cognitive Reasoning Optimization

**Fine-tuning DeepSeek-R1-Distill-Qwen-7B with LoRA to improve structured reasoning on physics problems — while fighting memorisation on tiny datasets.**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![HuggingFace](https://img.shields.io/badge/🤗_Transformers-4.44+-FFD21E?style=for-the-badge)](https://huggingface.co/docs/transformers)
[![PEFT](https://img.shields.io/badge/PEFT-LoRA-blueviolet?style=for-the-badge)](https://github.com/huggingface/peft)
[![Kaggle](https://img.shields.io/badge/Runs_on-Kaggle_T4-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://kaggle.com)

</div>

---

## 📌 The Problem

Large Language Models can reason, but they often **memorise** training examples instead of **learning to reason**. When you fine-tune on a small, curated dataset (e.g., 52–72 physics problems), standard SFT quickly overfits — the model regurgitates training traces verbatim rather than generalising the reasoning process to unseen problems.

## 💡 The Approach — LOOM v2

**LOOM** (LoRA Optimisation for Overfit Mitigation) is a systematic sweep framework that trains **8 different LoRA adapter configurations** on the same dataset and **automatically selects the best one** based on held-out evaluation — not human judgement.

Each "arm" varies a different anti-memorisation lever:

| Arm | Strategy | What it tests |
|---|---|---|
| `A_v1_control` | Baseline v1 recipe, no early stopping | Shows the overfit curve |
| `B_half` | Half rank, half LR | Capacity reduction |
| `C_attn_only` | Attention-only LoRA (no MLP) | Architectural constraint |
| `D_lowrank` | Rank 4 — too small to store 52 traces | Extreme compression |
| `E_gentle` | Full capacity, quarter learning rate | Slower convergence |
| `F_neftune` | NEFTune embedding noise | Regularisation via noise |
| `G_rslora` | Rank-stabilised LoRA scaling | Stable gradient flow |
| `H_regularised` | Weight decay + label smoothing | Classical regularisation |

### Key Design Decisions

- 🔄 **Base model reloaded fresh** for every arm — no cross-contamination
- 🧪 **Stratified val/pristine split** (10/10) — the winner is chosen on data it has *never* seen
- 🛡️ **Damage gate** — any arm that breaks format compliance or general ability is automatically disqualified
- ⏸️ **Resumable** — if Kaggle's session dies, re-run the cells and finished arms are skipped

---

## 🏗️ Project Structure

```
.
├── notebooks/
│   ├── loom_v2_train.ipynb          # Main training notebook (run on Kaggle)
│   └── deep_reason_finetune.ipynb   # v1 Colab notebook (archived)
├── src/
│   ├── loom_v2_train.py             # Training script (editable version of the notebook)
│   ├── loom_worker.py               # Per-arm subprocess worker (manages VRAM)
│   ├── orchestrator.py              # Multi-GPU sweep orchestrator
│   ├── physics_worker.py            # Physics evaluation worker
│   ├── physics_orchestrator.py      # Physics eval orchestrator
│   ├── test_grading_logic.py        # Offline grading tests (no GPU needed)
│   ├── test_gsm8k_grading.py        # Math grading unit tests
│   └── test_physics_grading.py      # Physics grading unit tests
├── baseline_heldout.json            # Baseline evaluation results
├── .gitignore
└── README.md
```

---

## ⚙️ Technical Details

| Component | Detail |
|---|---|
| **Base Model** | [DeepSeek-R1-Distill-Qwen-7B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B) |
| **Fine-tuning Method** | QLoRA (4-bit quantised LoRA) via PEFT |
| **LoRA Rank** | 4–16 (varies per arm) |
| **LoRA Alpha** | 8–32 (varies per arm) |
| **Target Modules** | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| **Max Sequence Length** | 768 tokens |
| **Training Data** | 52–72 curated physics reasoning traces |
| **Hardware** | Kaggle T4 GPU (free tier) |
| **Training Time** | ~2–3.5 hours for the full 8-arm sweep |

---

## 🚀 Quick Start

### Run on Kaggle (Recommended)

1. Go to [kaggle.com](https://kaggle.com) → **Create → New Notebook** → **Import** `notebooks/loom_v2_train.ipynb`
2. **Settings** → Accelerator: **GPU T4**, Internet: **On**
3. Upload `train.jsonl` as a dataset (name it `loom-train`)
4. Click **Run All** → come back in ~2.5 hours

### Run Locally (requires a GPU)

```bash
# Clone the repo
git clone https://github.com/shobhitranjan2005/LLM-Cognitive-Reasoning-Optimization.git
cd LLM-Cognitive-Reasoning-Optimization

# Install dependencies
pip install -U "transformers>=4.44" accelerate peft bitsandbytes datasets torch

# Run the training script
python src/loom_v2_train.py
```

---

## 🏆 How the Winner is Selected

Selection is **fully automated** — no human cherry-picking:

1. **Damage Gate** — format compliance ≥ 0.9, general ability within 10 pts of base, perplexity ≤ 1.25× base. Failed arms are marked `DAMAGED` and disqualified.
2. **Accuracy** — highest score on 10 pristine (never-seen) physics problems.
3. **Tie-breaking** — lower regurgitation score (shortest verbatim copy from training), then fewer thinking tokens.

---

## 📊 Outputs

After a successful run, you get:

| File | Description |
|---|---|
| `loom_v2_best.zip` | The winning LoRA adapter + full provenance documentation |
| `v2_report.json` | Complete run report: loss curves for all 8 arms, every generated answer, all evaluation metrics |

---

## 🛡️ Safety & Reproducibility

- Base model is **reloaded fresh** for every arm — one bad arm cannot contaminate the next
- All training data is **read-only** — nothing is modified in-place
- v1 adapter is preserved with **SHA-256 checksums** for full audit trail
- Seed is fixed (`20260819`) for reproducibility

---

## 📜 License

This project is for research and educational purposes.

---

<div align="center">

**Built with ❤️ by [Shobhit Ranjan](https://github.com/shobhitranjan2005)**

*If this project helped you, consider giving it a ⭐!*

</div>

