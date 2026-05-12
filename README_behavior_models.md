# MiniMind-BehaviorModels

A single small LLM (~64M parameters) built on [MiniMind](https://github.com/jingyaogong/minimind) that implements a simplified version of the **User Behavior Theory** for LLM products.

**Pipeline**:
```
User Query → [Behavior Model] → intent tags + system prompt → [Any LLM API]
```

The model outputs two parts separated by a blank line:
```
type:explanation | style:detailed | ref:yes

Provide a comprehensive, structured answer. Conclude with a References section.
```

The calling code splits on the first blank line: `tags, system_prompt = output.split('\n\n', 1)`.

Neither model needs domain knowledge — it only learns structural patterns — which is why 64M parameters is sufficient.

---

## Dataset Sources

### Combined dataset (`dataset/sft_behavior_combined.jsonl`)

| Source | Dataset | Records | Transform |
|--------|---------|---------|-----------|
| EN | [databricks/databricks-dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k) | ~15k | Map 8 categories → tag + rule-generated prompt |
| ZH | [BelleGroup/train_0.5M_CN](https://huggingface.co/datasets/BelleGroup/train_0.5M_CN) | ~50k (sampled) | Keyword heuristic tag + rule-generated prompt |
| ZH | `dataset/sft_medical.jsonl` (local) | ~2k (sampled) | Fixed `type:explanation \| style:detailed \| ref:yes` + ref prompt |
| EN | [Open-Orca/OpenOrca](https://huggingface.co/datasets/Open-Orca/OpenOrca) | ~20k (filtered) | Inferred tag + real Orca system prompt |

Tag format: `type:{X} | style:{Y} | ref:{Z}`

| Dimension | Values |
|-----------|--------|
| `type` | `fact`, `judgment`, `explanation`, `creative`, `plan` |
| `style` | `brief`, `detailed` |
| `ref` | `yes`, `no` |

---

## Step 1 — Prepare Data

```bash
python scripts/prepare_behavior_data.py
```

Key flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--belle_samples` | 50000 | BelleGroup ZH examples |
| `--medical_samples` | 2000 | Medical examples |
| `--orca_samples` | 20000 | OpenOrca examples |
| `--seed` | 42 | Random seed for reproducibility |
| `--preview_n` | 3 | Samples to print per dataset |

Set `HF_ENDPOINT=https://hf-mirror.com` if you need to use a HuggingFace mirror (e.g. in mainland China).

---

## Step 2 — Train

Fine-tunes on query → (tags + system prompt) pairs.

```bash
python trainer/train_full_sft.py \
  --data_path dataset/sft_behavior_combined.jsonl \
  --from_weight pretrain \
  --save_weight behavior_model \
  --epochs 3 \
  --batch_size 32 \
  --learning_rate 2e-5 \
  --max_seq_len 512 \
  --empty_think_ratio 0
```

Output checkpoint: `out/behavior_model_768.pth`

---

## Step 3 — Run the Pipeline

```bash
# Single query
python scripts/run_behavior_pipeline.py --query "今天需不需要看病？"

# Interactive mode
python scripts/run_behavior_pipeline.py
```

Example outputs:

| Query | Expected Tags | Expected System Prompt style |
|-------|--------------|------------------------------|
| 今天需不需要看病？ | `type:judgment \| style:brief \| ref:no` | "Answer in 1-2 sentences…" |
| 信息觅食理论是什么？ | `type:explanation \| style:detailed \| ref:no` | "Provide a comprehensive, structured answer…" |
| 甲减应该如何治疗？ | `type:explanation \| style:detailed \| ref:yes` | "…Conclude with a References section…" |

---

## File Structure

```
minimind/
├── scripts/
│   ├── prepare_behavior_data.py     # Dataset download + transform pipeline
│   └── run_behavior_pipeline.py     # End-to-end demo: query → tags + system prompt
├── dataset/
│   └── sft_behavior_combined.jsonl  # Generated: training data
├── trainer/
│   └── train_full_sft.py            # SFT trainer (used as-is)
├── out/
│   ├── pretrain_768.pth             # Base weights (user provides)
│   └── behavior_model_768.pth       # Trained checkpoint
└── eval_llm.py                      # Interactive inference (MiniMind standard)
```

---

## Notes

- The base MiniMind pretrain weights must exist at `out/pretrain_768.pth` before training. Follow the main [MiniMind training guide](README_en.md) to obtain them.
- The model output is always `tags\n\nsystem_prompt`. Split on the first blank line to get both parts.
- The Orca source contributes real (non-rule-generated) system prompts which improve output diversity.

### RunPod tips

Upload pretrain weights via the RunPod file manager or:
```bash
scp out/pretrain_768.pth root@<pod-ip>:/workspace/minimind/out/
```

Multi-GPU training (replace `N` with number of GPUs):
```bash
torchrun --nproc_per_node=N trainer/train_full_sft.py [same args]
```

Download trained weights after training:
```bash
zip behavior_models.zip out/behavior_model_768.pth
```
