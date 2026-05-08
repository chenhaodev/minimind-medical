# MiniMind-Medical

A bilingual (Chinese + English) medical domain LLM built on [MiniMind](https://github.com/jingyaogong/minimind) (~64M parameters). The training strategy is **continued pretraining** on a medical corpus followed by **full SFT** on bilingual medical Q&A — injecting domain knowledge without training from scratch.

---

## Dataset Sources

### Pretrain corpus (`dataset/pretrain_medical.jsonl`)

Format: `{"text": "..."}` — one record per line.

| Language | Dataset | Records |
|----------|---------|---------|
| ZH | [HuatuoGPT-sft-data-v1](https://huggingface.co/datasets/FreedomIntelligence/HuatuoGPT-sft-data-v1) (Q&A → text) | up to 50k |
| EN | [medical_meadow_wikidoc](https://huggingface.co/datasets/medalpaca/medical_meadow_wikidoc) | ~9.8k |
| EN | [MedRAG/textbooks](https://huggingface.co/datasets/MedRAG/textbooks) | ~125k |
| EN | [ccdv/pubmed-summarization](https://huggingface.co/datasets/ccdv/pubmed-summarization) (abstracts) | up to 50k |

### SFT corpus (`dataset/sft_medical.jsonl`)

Format: `{"conversations": [{"role": "user"/"assistant"/"system", "content": "..."}]}` — one record per line.

| Language | Dataset | Records |
|----------|---------|---------|
| ZH | [HuatuoGPT-sft-data-v1](https://huggingface.co/datasets/FreedomIntelligence/HuatuoGPT-sft-data-v1) | ~226k |
| ZH | [CMtMedQA](https://huggingface.co/datasets/Suprit/CMtMedQA) | ~68k |
| ZH | [DISC-Med-SFT](https://huggingface.co/datasets/Flmc/DISC-Med-SFT) | up to 20k |
| ZH | [ChatMed_Consult_Dataset](https://huggingface.co/datasets/michaelwzhu/ChatMed_Consult_Dataset) | up to 20k |
| EN | [ChatDoctor-HealthCareMagic-100k](https://huggingface.co/datasets/lavita/ChatDoctor-HealthCareMagic-100k) | ~112k |
| EN | [medical_meadow_mediqa](https://huggingface.co/datasets/medalpaca/medical_meadow_mediqa) | ~2.2k |
| EN | [medical_meadow_health_advice](https://huggingface.co/datasets/medalpaca/medical_meadow_health_advice) | ~2.2k |
| EN | [PubMedQA](https://huggingface.co/datasets/pubmed_qa) | ~1k |

---

## Step 1 — Prepare Data

```bash
# Mini run (fast, for testing)
python scripts/prepare_medical_data.py --stage all

# Full run (production scale)
python scripts/prepare_medical_data.py --stage all \
  --pubmed_samples 500000 \
  --huatuo_samples 300000 \
  --disc_samples 100000 \
  --chatmed_samples 100000
```

Key flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--stage` | `all` | `pretrain`, `sft`, or `all` |
| `--pubmed_samples` | 50000 | PubMed abstracts to include |
| `--huatuo_samples` | 50000 | HuatuoGPT entries to include |
| `--disc_samples` | 20000 | DISC-Med-SFT entries to include |
| `--chatmed_samples` | 20000 | ChatMed entries to include |
| `--seed` | 42 | Random seed for reproducibility |

The script downloads datasets from HuggingFace Hub, normalizes them to MiniMind format, deduplicates with SimHash, shuffles, and writes the output JSONL files.

Set `HF_ENDPOINT=https://hf-mirror.com` if you need to use a mirror.

---

## Step 2 — Continue Pretrain on Medical Corpus

Loads the existing `pretrain` weights and continues training on the medical text corpus. This injects domain vocabulary and knowledge while preserving general language ability.

```bash
python trainer/train_pretrain.py \
  --data_path dataset/pretrain_medical.jsonl \
  --from_weight pretrain \
  --save_weight pretrain_medical \
  --learning_rate 1e-4 \
  --epochs 1
```

Output checkpoint: `out/pretrain_medical_*.pth`

---

## Step 3 — Full SFT on Medical Q&A

Fine-tunes the medical pretrain checkpoint on bilingual instruction pairs to teach the model to follow medical Q&A format.

```bash
python trainer/train_full_sft.py \
  --data_path dataset/sft_medical.jsonl \
  --from_weight pretrain_medical \
  --save_weight full_sft_medical \
  --learning_rate 5e-6 \
  --epochs 1
```

Output checkpoint: `out/full_sft_medical_*.pth`

---

## Step 4 — Inference

```bash
python eval_llm.py --weight full_sft_medical
```

Example queries to test domain knowledge:
- 糖尿病的主要症状有哪些？
- What are the contraindications for aspirin?
- 高血压患者应该注意哪些饮食禁忌？

---

## File Structure

```
minimind/
├── scripts/
│   └── prepare_medical_data.py   # Dataset download + normalize + dedup pipeline
├── dataset/
│   ├── pretrain_medical.jsonl    # Generated pretrain corpus
│   └── sft_medical.jsonl         # Generated SFT instruction pairs
├── trainer/
│   ├── train_pretrain.py         # Pretrain trainer
│   └── train_full_sft.py         # Full SFT trainer
├── out/
│   ├── pretrain_medical_*.pth    # Medical pretrain checkpoint
│   └── full_sft_medical_*.pth    # Final medical SFT checkpoint
└── eval_llm.py                   # Interactive inference
```

---

## Notes

- The learning rate for continued pretrain (`1e-4`) is lower than the base pretrain default (`5e-4`) to avoid overwriting general language representations.
- The SFT learning rate (`5e-6`) is lower than the base SFT default (`1e-5`) for domain stability.
- ~20% of SFT samples automatically receive a medical system prompt (handled by `_maybe_add_system` in the data pipeline).
- The base MiniMind pretrain weights must exist at `out/pretrain_*.pth` before running Step 2. Follow the main [MiniMind training guide](README_en.md) to obtain them.
