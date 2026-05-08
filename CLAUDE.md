# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MiniMind is an educational LLM training project that trains a ~64M parameter language model entirely from scratch using native PyTorch. It covers the full pipeline: pretraining → SFT → LoRA → DPO → RLAIF (PPO/GRPO/CISPO) → Agentic RL → Distillation. The model architecture (Dense + MoE) is aligned with the Qwen3/Qwen3-MoE ecosystem.

## Common Commands

### Install dependencies
```bash
pip install -r requirements.txt
# PyTorch must be installed separately (commented out in requirements.txt):
# pip install torch==2.6.0 torchvision==0.21.0
```

### Training (run from repo root, scripts are in `trainer/`)

**Single GPU:**
```bash
# Pretraining
python trainer/train_pretrain.py --data_path ../dataset/pretrain_t2t_mini.jsonl

# Supervised Fine-Tuning (SFT)
python trainer/train_full_sft.py

# LoRA fine-tuning
python trainer/train_lora.py

# DPO (RLHF)
python trainer/train_dpo.py

# GRPO (RLAIF)
python trainer/train_grpo.py

# PPO (RLAIF)
python trainer/train_ppo.py

# Agentic RL
python trainer/train_agent.py

# Knowledge Distillation
python trainer/train_distillation.py
```

**Multi-GPU (DDP):**
```bash
torchrun --nproc_per_node=N trainer/train_pretrain.py [args]
```

Common training args: `--hidden_size 768 --num_hidden_layers 8 --use_moe 0 --batch_size 32 --learning_rate 5e-4 --device cuda:0 --dtype bfloat16`

### Inference
```bash
# Interactive CLI chat
python eval_llm.py --weight full_sft --hidden_size 768

# Load from HuggingFace/ModelScope format
python eval_llm.py --load_from jingyaogong/minimind-3

# With LoRA weights
python eval_llm.py --weight full_sft --lora_weight lora_identity

# Enable adaptive thinking
python eval_llm.py --open_thinking 1
```

### Serving
```bash
# OpenAI-compatible API server (FastAPI + uvicorn)
python scripts/serve_openai_api.py

# Streamlit web demo
streamlit run scripts/web_demo.py

# Chat API client example
python scripts/chat_api.py
```

### Evaluation
```bash
python eval_llm.py  # benchmark via third-party suites (C-Eval, C-MMLU, OpenBookQA)
python scripts/eval_toolcall.py  # tool call evaluation
```

### Model conversion
```bash
# Merge LoRA weights into base model
python scripts/convert_model.py
```

### Tokenizer training
```bash
python trainer/train_tokenizer.py
```

## Architecture

### Model (`model/`)
- `model_minimind.py` — Full model definition. `MiniMindConfig` (extends `PretrainedConfig`) controls all hyperparameters. `MiniMindForCausalLM` (extends `PreTrainedModel + GenerationMixin`) supports both Dense and MoE variants. Key sub-modules: `RMSNorm`, `MiniMindAttention` (GQA + RoPE + optional Flash Attention), `MiniMindMLP` / `MiniMindMoE` (sparse expert routing), `MiniMindDecoderLayer`.
- `model_lora.py` — LoRA implemented from scratch. `apply_lora()` patches `nn.Linear` layers in-place; `save_lora()` / `load_lora()` handle weight I/O.
- `tokenizer.json` / `tokenizer_config.json` — Custom BPE tokenizer (vocab size 6400) with special tokens: `<|im_start|>`, `<|im_end|>`, `<think>`, `<tool_call>`, `<tool_response>`, plus reserved buffer tokens.

### Training pipeline (`trainer/`)
- `trainer_utils.py` — Shared utilities: `init_model()`, `init_distributed_mode()` (NCCL DDP), `lm_checkpoint()` (save/resume), cosine LR schedule (`get_lr()`), `LMForRewardModel`.
- `rollout_engine.py` — Decoupled rollout backend for RLAIF; used by `train_grpo.py`, `train_ppo.py`, `train_agent.py`.
- Each `train_*.py` script is self-contained with its own `argparse` block and `if __name__ == "__main__"` entrypoint.

### Data (`dataset/`)
- `lm_dataset.py` — Dataset classes: `PretrainDataset` (plain text JSONL), `SFTDataset` (chat template JSONL), `RLAIFDataset`. Chat preprocessing applies random system prompts and handles `<think>` tag stripping.
- Training data format: JSONL files. Pretrain uses `{"text": "..."}`. SFT uses conversation arrays with `role`/`content` fields (and optional `tools` for tool-call data).

### Scripts (`scripts/`)
- `serve_openai_api.py` — FastAPI server implementing OpenAI `/v1/chat/completions` with streaming, `reasoning_content`, `tool_calls`, and `open_thinking` support.
- `web_demo.py` — Streamlit UI supporting thinking display, tool selection, multi-turn tool calls.
- `convert_model.py` — LoRA merge + HuggingFace export pipeline.

## Key Conventions

**Model loading:** Two modes controlled by `--load_from`:
- `"model"` (default) — loads raw PyTorch `.pth` weights from `out/` directory, named `{weight}_{hidden_size}[_moe].pth`
- Any other path — loads via `AutoModelForCausalLM.from_pretrained()` (HuggingFace/ModelScope format)

**Checkpoint naming:** `out/pretrain_768.pth`, `out/full_sft_768.pth`, `out/grpo_768.pth`, etc. MoE adds `_moe` suffix.

**Mixed precision:** All trainers use `torch.amp.autocast` + `GradScaler`. Default dtype is `bfloat16`.

**DDP detection:** `init_distributed_mode()` checks `RANK` env var; returns `local_rank=0` for single-GPU. Use `torchrun` to launch multi-GPU runs.

**Logging:** `wandb` and `swanlab` are both supported; pass `--use_wandb` to enable.
