# Reasoning Distillation for SLM (DeepSeek-R1 Style)

## 📌 Overview
This project explores methods to distill the **Reasoning Capabilities (Chain-of-Thought)** of large reasoning models (like DeepSeek-R1) into small language models (SLMs, e.g., **Qwen2.5-1.5B**).

For a GPU-session execution guide, see [RUNBOOK.md](RUNBOOK.md).

We focus on two key approaches:
1.  **Standard SFT:** Fine-tuning on high-quality reasoning traces containing `<think>` tags.
2.  **Step-Aware Loss:** A custom loss function that assigns higher weight (2.0x) to tokens within the reasoning block (`<think>...</think>`) to enforce logic learning over rote memorization.

## 📊 Current Results (GSM8K)

| Model | Setup | Accuracy (N=1319) | Improvement |
|-------|-------|-------------------|-------------|
| **Baseline** | Qwen2.5-1.5B-Instruct (Zero-shot) | 30.93% | - |
| **Distilled (SFT)** | 1 Epoch, Standard Loss | **32.07%** | **+1.14%** |
| **Distilled (Weighted)** | 3 Epochs, 2x Think Weight | **35.78%** | **+4.85%** |

> **Finding:** Standard SFT improved accuracy slightly, but **Step-Aware Loss** (weighting reasoning tokens) provided a significant boost, proving that the model needs to prioritize the *process* over the answer.

## 🛠️ Environment Setup

```bash
# Create virtual environment
pip install uv
uv venv .venv
source .venv/bin/activate

# Install dependencies (Unsloth, TRL, etc.)
uv pip install -r requirements.txt
```

## 🚀 Usage

### 1. Training (Standard SFT)
Fine-tune the model using standard Supervised Fine-Tuning.
```bash
python train_sft.py
```

### 2. Training (Step-Aware Loss)
Fine-tune with a custom trainer that weights reasoning tokens higher.
```bash
python train_weighted.py
```

### 3. Evaluation (GSM8K)
Evaluate the model's accuracy on the GSM8K benchmark.
```bash
# Evaluate Baseline
python evaluate.py --model_name "Qwen/Qwen2.5-1.5B-Instruct"

# Evaluate Fine-tuned Model
python evaluate.py --model_name "./lora_model"
```

## 📂 File Structure
- `train_sft.py`: Standard SFT training script (Unsloth).
- `train_weighted.py`: Custom SFT script with `WeightedLossTrainer` (Step-aware loss).
- `evaluate.py`: Evaluation script for GSM8K (supports `<think>` and `\boxed{}` parsing).
- `requirements.txt`: Python dependencies.

## 📝 Dataset
- **Reasoning Source**: `bespokelabs/Bespoke-Stratos-17k`
- **Evaluation**: `GSM8K`

## 👨‍💻 Author
Experiments on distilling reasoning to on-device models.
