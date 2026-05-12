"""
User Behavior Theory SFT dataset preparation for MiniMind.

Outputs:
  dataset/sft_intent_tagger.jsonl     - Model A training data (query → intent tags)
  dataset/sft_prompt_augmenter.jsonl  - Model B training data (query+tags → system prompt)

Usage:
  python scripts/prepare_behavior_data.py
  python scripts/prepare_behavior_data.py --belle_samples 5000 --orca_samples 3000
"""

import argparse
import json
import os
import random
import re
import sys

__package__ = "scripts"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from datasets import load_dataset

OUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dataset"))

# ---------------------------------------------------------------------------
# Tag + prompt definitions
# ---------------------------------------------------------------------------

# Maps databricks/databricks-dolly-15k categories → (task_type, style, ref)
DOLLY_TAG_MAP = {
    "closed_qa":              ("fact",        "brief",    "no"),
    "open_qa":                ("explanation", "detailed", "no"),
    "brainstorming":          ("creative",    "detailed", "no"),
    "classification":         ("judgment",    "brief",    "no"),
    "generation":             ("creative",    "detailed", "no"),
    "information_extraction": ("fact",        "detailed", "no"),
    "summarization":          ("explanation", "detailed", "no"),
    "free_form":              ("explanation", "detailed", "no"),
}

# Style prompt templates used by Model B output
_STYLE_PROMPTS = {
    "brief":    "Answer in 1-2 sentences. State only the conclusion, no elaboration.",
    "detailed": "Provide a comprehensive, structured answer with clear sections.",
    "creative": "Be creative and thorough. Explore the topic broadly with original ideas.",
    "step":     "Provide a clear step-by-step guide. Use numbered steps.",
    "ref":      "Provide a detailed, evidence-based answer. Conclude with a References section citing authoritative sources.",
}

# Chinese keyword patterns for BelleGroup auto-labelling
_ZH_JUDGMENT_RE = re.compile(r"应该|需不需要|要不要|该不该|是否|能不能|可以吗|对吗|好吗|可行")
_ZH_EXPLANATION_RE = re.compile(r"为什么|是什么|什么是|解释|说明|介绍|如何理解")
_ZH_CREATIVE_RE = re.compile(r"写一|写个|帮我写|创作|生成一|写一篇")
_ZH_PLAN_RE = re.compile(r"如何|怎么|怎样|步骤|方法|流程|怎么做")
_ZH_REF_RE = re.compile(r"医|药|病|症|健康|法律|法规|政策|科学|研究|论文")

# English keyword patterns for OpenOrca auto-labelling
_EN_EXPLANATION_RE = re.compile(r"\bwhy\b|\bexplain\b|what is\b|\bdescribe\b", re.I)
_EN_JUDGMENT_RE = re.compile(r"\bshould\b|is it\b|can i\b|do i need\b", re.I)
_EN_CREATIVE_RE = re.compile(r"\bwrite\b|\bcreate\b|\bgenerate\b|\bcompose\b", re.I)
_EN_PLAN_RE = re.compile(r"how to\b|how do\b|\bsteps\b|\bguide\b", re.I)
_EN_REF_RE = re.compile(r"\bmedical\b|\bresearch\b|\bstudy\b|\bevidence\b|\blegal\b|\bscientific\b", re.I)

# Regex to detect style-directive language in OpenOrca system prompts
_ORCA_STYLE_RE = re.compile(
    r"brief|concise|detailed|step.by.step|reference|cite|evidence|comprehensive|"
    r"简洁|详细|步骤|参考|引用|详尽",
    re.I,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clean(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _write_jsonl(path: str, records: list) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"  Wrote {len(records):,} records → {path}")


def _make_tag(task_type: str, style: str, ref: str) -> str:
    return f"type:{task_type} | style:{style} | ref:{ref}"


def _make_style_prompt(task_type: str, style: str, ref: str) -> str:
    if ref == "yes":
        return _STYLE_PROMPTS["ref"]
    if task_type == "creative":
        return _STYLE_PROMPTS["creative"]
    if task_type == "plan":
        return _STYLE_PROMPTS["step"]
    return _STYLE_PROMPTS.get(style, _STYLE_PROMPTS["detailed"])


def _zh_classify(text: str):
    """Return (task_type, style, ref) for a Chinese instruction via keyword heuristics."""
    if _ZH_JUDGMENT_RE.search(text):
        task_type, style = "judgment", "brief"
    elif _ZH_EXPLANATION_RE.search(text):
        task_type, style = "explanation", "detailed"
    elif _ZH_CREATIVE_RE.search(text):
        task_type, style = "creative", "detailed"
    elif _ZH_PLAN_RE.search(text):
        task_type, style = "plan", "detailed"
    else:
        task_type, style = "fact", "brief"
    ref = "yes" if _ZH_REF_RE.search(text) else "no"
    return task_type, style, ref


def _en_classify(text: str):
    """Return (task_type, style, ref) for an English query via keyword heuristics."""
    if _EN_JUDGMENT_RE.search(text):
        task_type, style = "judgment", "brief"
    elif _EN_EXPLANATION_RE.search(text):
        task_type, style = "explanation", "detailed"
    elif _EN_CREATIVE_RE.search(text):
        task_type, style = "creative", "detailed"
    elif _EN_PLAN_RE.search(text):
        task_type, style = "plan", "detailed"
    else:
        task_type, style = "fact", "brief"
    ref = "yes" if _EN_REF_RE.search(text) else "no"
    return task_type, style, ref


def _preview(name: str, records: list, n: int = 3) -> None:
    print(f"\n--- {name} preview ({len(records):,} total) ---")
    for rec in random.sample(records, min(n, len(records))):
        convs = rec["conversations"]
        print(f"  USER: {convs[0]['content'][:90]!r}")
        print(f"  ASST: {convs[-1]['content'][:120]!r}")
        print()


# ---------------------------------------------------------------------------
# Model A: Intent Tagger dataset
# ---------------------------------------------------------------------------

def build_intent_tagger_data(belle_samples: int, medical_samples: int, seed: int) -> list:
    rng = random.Random(seed)
    records = []

    # Source 1: databricks/databricks-dolly-15k
    print("  Loading databricks/databricks-dolly-15k ...")
    dolly = load_dataset("databricks/databricks-dolly-15k", split="train")
    for row in dolly:
        instruction = _clean(row.get("instruction", "") or "")
        context = _clean(row.get("context", "") or "")
        category = row.get("category", "")
        if not instruction or category not in DOLLY_TAG_MAP:
            continue
        task_type, style, ref = DOLLY_TAG_MAP[category]
        query = f"{instruction}\n\n{context}" if context else instruction
        records.append({
            "conversations": [
                {"role": "user", "content": query},
                {"role": "assistant", "content": _make_tag(task_type, style, ref)},
            ]
        })
    print(f"  Dolly: {len(records):,} examples")

    # Source 2: BelleGroup/train_0.5M_CN (streaming)
    print(f"  Loading BelleGroup/train_0.5M_CN (streaming, target {belle_samples:,}) ...")
    belle_stream = load_dataset("BelleGroup/train_0.5M_CN", split="train", streaming=True)
    pool = []
    for row in belle_stream:
        instr = _clean(row.get("instruction", "") or "")
        if len(instr) < 10:
            continue
        pool.append(instr)
        if len(pool) >= belle_samples * 4:
            break
    rng.shuffle(pool)
    belle_added = 0
    for instr in pool[:belle_samples]:
        task_type, style, ref = _zh_classify(instr)
        records.append({
            "conversations": [
                {"role": "user", "content": instr},
                {"role": "assistant", "content": _make_tag(task_type, style, ref)},
            ]
        })
        belle_added += 1
    print(f"  BelleGroup: {belle_added:,} examples")

    # Source 3: local sft_medical.jsonl (ref:yes bucket)
    medical_path = os.path.join(OUT_DIR, "sft_medical.jsonl")
    if os.path.exists(medical_path):
        print(f"  Loading sft_medical.jsonl (sample {medical_samples:,}) ...")
        queries = []
        with open(medical_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    user_turns = [t for t in obj.get("conversations", []) if t.get("role") == "user"]
                    if user_turns:
                        q = _clean(user_turns[0]["content"])
                        if len(q) >= 5:
                            queries.append(q)
                except (json.JSONDecodeError, KeyError):
                    continue
        rng.shuffle(queries)
        med_added = 0
        for q in queries[:medical_samples]:
            records.append({
                "conversations": [
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": _make_tag("explanation", "detailed", "yes")},
                ]
            })
            med_added += 1
        print(f"  Medical: {med_added:,} examples (ref:yes)")
    else:
        print("  sft_medical.jsonl not found — skipping medical source")

    rng.shuffle(records)
    return records


# ---------------------------------------------------------------------------
# Model B: Prompt Augmenter dataset
# ---------------------------------------------------------------------------

def build_prompt_augmenter_data(
    orca_samples: int,
    medical_samples: int,
    seed: int,
    tagger_records: list,
) -> list:
    rng = random.Random(seed)
    records = []

    # Source 1: derive from tagger records (rule-generated system prompts)
    for rec in tagger_records:
        convs = rec["conversations"]
        query = convs[0]["content"]
        tag = convs[1]["content"]
        # Parse tag string back to (type, style, ref)
        parts = {}
        for segment in tag.split("|"):
            segment = segment.strip()
            if ":" in segment:
                k, v = segment.split(":", 1)
                parts[k.strip()] = v.strip()
        task_type = parts.get("type", "fact")
        style = parts.get("style", "brief")
        ref = parts.get("ref", "no")
        style_prompt = _make_style_prompt(task_type, style, ref)
        records.append({
            "conversations": [
                {"role": "user", "content": f"Query: {query}\nTags: {tag}"},
                {"role": "assistant", "content": f"System: {style_prompt}\nQuery: {query}"},
            ]
        })

    # Source 2: Open-Orca/OpenOrca — filter for style-directive system prompts
    print(f"  Loading Open-Orca/OpenOrca (streaming, target {orca_samples:,}) ...")
    orca_stream = load_dataset("Open-Orca/OpenOrca", split="train", streaming=True)
    orca_added = 0
    for row in orca_stream:
        sys_prompt = row.get("system_prompt", "") or ""
        question = _clean(row.get("question", "") or "")
        if not question or not sys_prompt or not _ORCA_STYLE_RE.search(sys_prompt):
            continue
        sys_prompt = _clean(sys_prompt)
        task_type, style, ref = _en_classify(question)
        tag = _make_tag(task_type, style, ref)
        records.append({
            "conversations": [
                {"role": "user", "content": f"Query: {question}\nTags: {tag}"},
                {"role": "assistant", "content": f"System: {sys_prompt}\nQuery: {question}"},
            ]
        })
        orca_added += 1
        if orca_added >= orca_samples:
            break
    print(f"  OpenOrca: {orca_added:,} examples with style prompts")

    # Source 3: medical queries with reference-requiring prompt
    medical_path = os.path.join(OUT_DIR, "sft_medical.jsonl")
    if os.path.exists(medical_path):
        print(f"  Loading sft_medical.jsonl for augmenter (sample {medical_samples:,}) ...")
        queries = []
        with open(medical_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    user_turns = [t for t in obj.get("conversations", []) if t.get("role") == "user"]
                    if user_turns:
                        q = _clean(user_turns[0]["content"])
                        if len(q) >= 5:
                            queries.append(q)
                except (json.JSONDecodeError, KeyError):
                    continue
        rng.shuffle(queries)
        med_added = 0
        tag = _make_tag("explanation", "detailed", "yes")
        ref_prompt = _STYLE_PROMPTS["ref"]
        for q in queries[:medical_samples]:
            records.append({
                "conversations": [
                    {"role": "user", "content": f"Query: {q}\nTags: {tag}"},
                    {"role": "assistant", "content": f"System: {ref_prompt}\nQuery: {q}"},
                ]
            })
            med_added += 1
        print(f"  Medical augmenter: {med_added:,} examples (ref:yes)")

    rng.shuffle(records)
    return records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Prepare User Behavior Theory SFT datasets")
    parser.add_argument("--belle_samples", type=int, default=10000,
                        help="Max BelleGroup ZH examples for intent tagger (default: 10000)")
    parser.add_argument("--medical_samples", type=int, default=2000,
                        help="Max medical examples per dataset (default: 2000)")
    parser.add_argument("--orca_samples", type=int, default=5000,
                        help="Max OpenOrca examples for prompt augmenter (default: 5000)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--preview_n", type=int, default=3,
                        help="Preview samples to print per dataset (default: 3)")
    args = parser.parse_args()

    random.seed(args.seed)

    print("\n=== Building Intent Tagger dataset (Model A) ===")
    tagger_records = build_intent_tagger_data(
        belle_samples=args.belle_samples,
        medical_samples=args.medical_samples,
        seed=args.seed,
    )
    tagger_path = os.path.join(OUT_DIR, "sft_intent_tagger.jsonl")
    _write_jsonl(tagger_path, tagger_records)
    _preview("Intent Tagger", tagger_records, args.preview_n)

    print("\n=== Building Prompt Augmenter dataset (Model B) ===")
    augmenter_records = build_prompt_augmenter_data(
        orca_samples=args.orca_samples,
        medical_samples=args.medical_samples,
        seed=args.seed,
        tagger_records=tagger_records,
    )
    augmenter_path = os.path.join(OUT_DIR, "sft_prompt_augmenter.jsonl")
    _write_jsonl(augmenter_path, augmenter_records)
    _preview("Prompt Augmenter", augmenter_records, args.preview_n)

    print("\nDone. Run training next:")
    print("  python trainer/train_full_sft.py --data_path dataset/sft_intent_tagger.jsonl "
          "--from_weight pretrain --save_weight intent_tagger --epochs 5 --max_seq_len 256")
    print("  python trainer/train_full_sft.py --data_path dataset/sft_prompt_augmenter.jsonl "
          "--from_weight pretrain --save_weight prompt_augmenter --epochs 3 --max_seq_len 512")


if __name__ == "__main__":
    main()
