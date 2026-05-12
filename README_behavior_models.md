# MiniMind-BehaviorModels

Two small LLMs (~64M parameters each) built on [MiniMind](https://github.com/jingyaogong/minimind) that implement a simplified version of the **User Behavior Theory** for LLM products.

**Pipeline**:
```
User Query → [Model A: Intent Tagger] → intent tags → [Model B: Prompt Augmenter] → crafted system prompt → [Any LLM API]
```

- **Model A** classifies a query into compact intent tags (`type`, `style`, `ref`).
- **Model B** takes the query + tags and outputs a system prompt that instructs a downstream LLM to respond with the right style: concise for conclusive questions, detailed for open-ended ones, with references for professional topics.

Neither model needs domain knowledge — they only learn structural patterns — which is why 64M parameters is sufficient.

---

## Dataset Sources

### Intent Tagger (`dataset/sft_intent_tagger.jsonl`)

| Source | Dataset | Records | Transform |
|--------|---------|---------|-----------|
| EN | [databricks/databricks-dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k) | ~15k | Map 8 categories → intent tag string |
| ZH | [BelleGroup/train_0.5M_CN](https://huggingface.co/datasets/BelleGroup/train_0.5M_CN) | ~10k (sampled) | Keyword heuristic tagging |
| ZH | `dataset/sft_medical.jsonl` (local) | ~2k (sampled) | Fixed tag `type:explanation \| style:detailed \| ref:yes` |

Tag format: `type:{X} | style:{Y} | ref:{Z}`

| Dimension | Values |
|-----------|--------|
| `type` | `fact`, `judgment`, `explanation`, `creative`, `plan` |
| `style` | `brief`, `detailed` |
| `ref` | `yes`, `no` |

### Prompt Augmenter (`dataset/sft_prompt_augmenter.jsonl`)

| Source | Dataset | Records | Transform |
|--------|---------|---------|-----------|
| EN/ZH | Derived from intent tagger data | ~27k | Rule-generate style prompt from tag |
| EN | [Open-Orca/OpenOrca](https://huggingface.co/datasets/Open-Orca/OpenOrca) | ~5k (filtered) | Filter system prompts with style directives |
| ZH | `dataset/sft_medical.jsonl` (local) | ~2k (sampled) | Reference-requiring system prompt |

---

## Step 1 — Prepare Data

```bash
python scripts/prepare_behavior_data.py
```

Key flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--belle_samples` | 10000 | BelleGroup ZH examples for the tagger |
| `--medical_samples` | 2000 | Medical examples per dataset |
| `--orca_samples` | 5000 | OpenOrca examples for the augmenter |
| `--seed` | 42 | Random seed for reproducibility |
| `--preview_n` | 3 | Samples to print per dataset |

Set `HF_ENDPOINT=https://hf-mirror.com` if you need to use a HuggingFace mirror (e.g. in mainland China).

---

## Step 2 — Train Model A (Intent Tagger)

Fine-tunes on query → compact tag pairs. Short `max_seq_len` (256) because outputs are always ≤20 tokens.

```bash
python trainer/train_full_sft.py \
  --data_path dataset/sft_intent_tagger.jsonl \
  --from_weight pretrain \
  --save_weight intent_tagger \
  --epochs 5 \
  --batch_size 32 \
  --learning_rate 2e-5 \
  --max_seq_len 256 \
  --empty_think_ratio 0
```

Output checkpoint: `out/intent_tagger_768.pth`

---

## Step 3 — Train Model B (Prompt Augmenter)

Fine-tunes on (query + tags) → system prompt pairs. Longer `max_seq_len` (512) for richer outputs.

```bash
python trainer/train_full_sft.py \
  --data_path dataset/sft_prompt_augmenter.jsonl \
  --from_weight pretrain \
  --save_weight prompt_augmenter \
  --epochs 3 \
  --batch_size 16 \
  --learning_rate 1e-5 \
  --max_seq_len 512 \
  --empty_think_ratio 0
```

Output checkpoint: `out/prompt_augmenter_768.pth`

---

## Step 4 — Run the Pipeline

```bash
# Single query
python scripts/run_behavior_pipeline.py --query "今天需不需要看病？"

# Interactive mode
python scripts/run_behavior_pipeline.py
```

Example outputs to verify each model:

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
│   └── run_behavior_pipeline.py     # End-to-end demo: Model A → Model B
├── dataset/
│   ├── sft_intent_tagger.jsonl      # Generated: Model A training data
│   └── sft_prompt_augmenter.jsonl   # Generated: Model B training data
├── trainer/
│   └── train_full_sft.py            # SFT trainer (used as-is)
├── out/
│   ├── pretrain_768.pth             # Base weights (user provides)
│   ├── intent_tagger_768.pth        # Model A checkpoint
│   └── prompt_augmenter_768.pth     # Model B checkpoint
└── eval_llm.py                      # Interactive inference (MiniMind standard)
```

---

## Notes

- Model A uses a higher learning rate (`2e-5`) and more epochs (5) because its output vocabulary is constrained — it only needs to learn one fixed template pattern.
- Model B uses a lower learning rate (`1e-5`) since it generates longer, more varied outputs.
- The base MiniMind pretrain weights must exist at `out/pretrain_768.pth` before training. Follow the main [MiniMind training guide](README_en.md) to obtain them.

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
zip behavior_models.zip out/intent_tagger_768.pth out/prompt_augmenter_768.pth
```
