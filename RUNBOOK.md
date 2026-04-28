# Runbook

This project requires a GPU-visible environment for `unsloth` training and inference.

## 1. Enter The Project

```bash
cd /storage/geonju511/math_reasoning_distillation
source .venv/bin/activate
export HF_HOME=/storage/geonju511/math_reasoning_distillation/.hf_cache
```

## 2. Confirm GPU Access

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
```

Expected result:
- `nvidia-smi` works
- `torch.cuda.is_available()` is `True`
- `torch.cuda.device_count()` is at least `1`

If not, stop here and switch to a GPU-enabled shell/session.

## 3. Inspect The Training Dataset

```bash
python inspect_data.py
```

Expected result:
- dataset loads successfully
- columns include `system` and `conversations`

## 4. Run A Small Baseline Evaluation

```bash
python evaluate.py --model_name "Qwen/Qwen2.5-1.5B-Instruct" --limit 50
```

Use a small `--limit` first to confirm model loading and generation.

## 5. Train Standard SFT

```bash
python train_sft.py
```

Outputs:
- `outputs/`
- `lora_model/`

## 6. Evaluate The SFT Model

```bash
python evaluate.py --model_name "./lora_model" --limit 50
```

After the smoke test passes, you can remove `--limit` for a full GSM8K run.

## 7. Train The Weighted Loss Variant

```bash
python train_weighted.py
```

Outputs:
- `outputs_weighted/`
- `lora_model_weighted/`

## 8. Evaluate The Weighted Model

```bash
python evaluate.py --model_name "./lora_model_weighted" --limit 50
```

## 9. Run A Single Inference Demo

```bash
python demo_inference.py
```

## Notes

- Start with small evaluation limits before full runs.
- The first Hugging Face download can take a while.
- If `unsloth` says it cannot find a GPU, re-check the session with step 2.
- Model and dataset downloads require network access.
