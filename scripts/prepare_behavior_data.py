"""
User Behavior Theory SFT dataset preparation for MiniMind.

Outputs:
  dataset/sft_behavior_combined.jsonl - Single model training data (query → tags + system prompt)

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

_STYLE_PROMPTS = {
    "brief":    "Answer in 1-2 sentences. State only the conclusion, no elaboration.",
    "detailed": "Provide a comprehensive, structured answer with clear sections.",
    "creative": "Be creative and thorough. Explore the topic broadly with original ideas.",
    "step":     "Provide a clear step-by-step guide. Use numbered steps.",
    "ref":      "Provide a detailed, evidence-based answer. Conclude with a References section citing authoritative sources.",
}

# Ordered (task_type, style, regex) entries — first match wins
_ZH_CLASSIFY_RULES = [
    ("judgment",    "brief",    re.compile(r"应该|需不需要|要不要|该不该|是否|能不能|可以吗|对吗|好吗|可行")),
    ("explanation", "detailed", re.compile(r"为什么|是什么|什么是|解释|说明|介绍|如何理解")),
    ("creative",    "detailed", re.compile(r"写一|写个|帮我写|创作|生成一|写一篇|续写|改编|改写|扩写")),
    ("plan",        "detailed", re.compile(r"如何|怎么|怎样|步骤|方法|流程|怎么做|给定.*将|对以下|根据.*生成|将其")),
]
_ZH_REF_RE = re.compile(r"医|药|病|症|健康|法律|法规|政策|科学|研究|论文")

_EN_CLASSIFY_RULES = [
    ("judgment",    "brief",    re.compile(r"\bshould\b|is it\b|can i\b|do i need\b", re.I)),
    ("explanation", "detailed", re.compile(r"\bwhy\b|\bexplain\b|what is\b|\bdescribe\b", re.I)),
    # match write/writing/written, create/creating, generate/generating, continue writing, etc.
    ("creative",    "detailed", re.compile(r"\bwrit\w*\b|\bcreate?\w*\b|\bgenerate?\w*\b|\bcompose?\w*\b|\bcontinue\b|\brewrite\b|\bsummariz\w*\b", re.I)),
    ("plan",        "detailed", re.compile(r"how to\b|how do\b|\bsteps\b|\bguide\b|\bgiven\b.*\b(convert|replace|transform|modify|find|extract)\b", re.I)),
]
_EN_REF_RE = re.compile(r"\bmedical\b|\bresearch\b|\bstudy\b|\bevidence\b|\blegal\b|\bscientific\b", re.I)

# Matches style-directive Orca prompts; excludes generic "You are an AI assistant" boilerplate
# by requiring the directive word to appear outside of a generic opener
_ORCA_STYLE_RE = re.compile(
    r"\b(brief|concise|step.by.step|comprehensive)\b|"
    r"\b(reference|cite|evidence-based)\b|"
    r"in (\d+ sentences?|bullet points?|numbered steps?)|"
    r"简洁|步骤|参考|引用|详尽",
    re.I,
)
# Generic Orca opener — skip prompts that are only this with no real style directive
_ORCA_GENERIC_RE = re.compile(
    r"^you are an? (ai |helpful )?(assistant|chatbot|language model)",
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
    os.makedirs(os.path.dirname(path), exist_ok=True)
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


def _classify(text: str, rules: list, ref_re) -> tuple:
    """Return (task_type, style, ref) using the first matching rule."""
    for task_type, style, pattern in rules:
        if pattern.search(text):
            break
    else:
        # unmatched instructions are typically task-following → treat as plan/detailed
        task_type, style = "plan", "detailed"
    ref = "yes" if ref_re.search(text) else "no"
    return task_type, style, ref


def _load_medical_queries(path: str, rng: random.Random) -> list:
    """Read all user queries from a MiniMind SFT JSONL file, shuffled."""
    queries = []
    with open(path, encoding="utf-8") as f:
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
    return queries


def _preview(name: str, records: list, n: int = 3) -> None:
    print(f"\n--- {name} preview ({len(records):,} total) ---")
    for rec in random.sample(records, min(n, len(records))):
        convs = rec["conversations"]
        print(f"  USER: {convs[0]['content'][:90]!r}")
        print(f"  ASST: {convs[-1]['content'][:120]!r}")
        print()


# ---------------------------------------------------------------------------
# Build tag metadata: (query, task_type, style, ref) tuples
# ---------------------------------------------------------------------------

def build_tag_meta(
    belle_samples: int,
    medical_queries: list,
    medical_samples: int,
    seed: int,
) -> list:
    rng = random.Random(seed)
    tag_meta = []

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
        tag_meta.append((query, task_type, style, ref))
    print(f"  Dolly: {len(tag_meta):,} examples")

    # Source 2: BelleGroup/train_0.5M_CN (reservoir sampling — O(belle_samples) memory)
    print(f"  Loading BelleGroup/train_0.5M_CN (streaming, sample {belle_samples:,}) ...")
    belle_stream = load_dataset("BelleGroup/train_0.5M_CN", split="train", streaming=True)
    reservoir = []
    valid_count = 0
    for row in belle_stream:
        instr = _clean(row.get("instruction", "") or "")
        if len(instr) < 10:
            continue
        if len(reservoir) < belle_samples:
            reservoir.append(instr)
        else:
            j = rng.randint(0, valid_count)
            if j < belle_samples:
                reservoir[j] = instr
        valid_count += 1
    belle_start = len(tag_meta)
    for instr in reservoir:
        task_type, style, ref = _classify(instr, _ZH_CLASSIFY_RULES, _ZH_REF_RE)
        tag_meta.append((instr, task_type, style, ref))
    print(f"  BelleGroup: {len(tag_meta) - belle_start:,} examples")

    # Source 3: pre-loaded medical queries (ref:yes bucket)
    med_start = len(tag_meta)
    for q in medical_queries[:medical_samples]:
        tag_meta.append((q, "explanation", "detailed", "yes"))
    print(f"  Medical: {len(tag_meta) - med_start:,} examples (ref:yes)")

    rng.shuffle(tag_meta)
    return tag_meta


# ---------------------------------------------------------------------------
# Combined: single model — query → tags + system prompt
# ---------------------------------------------------------------------------

def build_combined_data(
    tag_meta: list,
    orca_samples: int,
    seed: int,
) -> list:
    rng = random.Random(seed)
    records = []

    # Source 1: Dolly + Belle + Medical
    for query, task_type, style, ref in tag_meta:
        tag = _make_tag(task_type, style, ref)
        system_prompt = _make_style_prompt(task_type, style, ref)
        records.append({
            "conversations": [
                {"role": "user", "content": query},
                {"role": "assistant", "content": f"{tag}\n\n{system_prompt}"},
            ]
        })

    # Source 2: Open-Orca/OpenOrca — use real system prompts, infer tags from query
    print(f"  Loading Open-Orca/OpenOrca (streaming, target {orca_samples:,}) ...")
    orca_stream = load_dataset("Open-Orca/OpenOrca", split="train", streaming=True)
    orca_added = 0
    for row in orca_stream:
        raw_sys = row.get("system_prompt", "") or ""
        raw_q = row.get("question", "") or ""
        if not raw_sys or not raw_q:
            continue
        sys_prompt = _clean(raw_sys)
        question = _clean(raw_q)
        if not _ORCA_STYLE_RE.search(sys_prompt) or _ORCA_GENERIC_RE.match(sys_prompt):
            continue
        task_type, style, ref = _classify(question, _EN_CLASSIFY_RULES, _EN_REF_RE)
        tag = _make_tag(task_type, style, ref)
        records.append({
            "conversations": [
                {"role": "user", "content": question},
                {"role": "assistant", "content": f"{tag}\n\n{sys_prompt}"},
            ]
        })
        orca_added += 1
        if orca_added >= orca_samples:
            break
    print(f"  OpenOrca: {orca_added:,} examples")

    rng.shuffle(records)
    return records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Prepare User Behavior Theory SFT datasets")
    parser.add_argument("--belle_samples", type=int, default=50000,
                        help="Max BelleGroup ZH examples for intent tagger (default: 50000)")
    parser.add_argument("--medical_samples", type=int, default=2000,
                        help="Max medical examples per dataset (default: 2000)")
    parser.add_argument("--orca_samples", type=int, default=20000,
                        help="Max OpenOrca examples for prompt augmenter (default: 20000)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--preview_n", type=int, default=3,
                        help="Preview samples to print per dataset (default: 3)")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    medical_queries = []
    medical_path = os.path.join(OUT_DIR, "sft_medical.jsonl")
    if os.path.exists(medical_path):
        print(f"Pre-loading sft_medical.jsonl ...")
        medical_queries = _load_medical_queries(medical_path, rng)
        print(f"  {len(medical_queries):,} medical queries loaded")
    else:
        print("sft_medical.jsonl not found — skipping medical source")

    print("\n=== Building tag_meta (query → tags) ===")
    tag_meta = build_tag_meta(
        belle_samples=args.belle_samples,
        medical_queries=medical_queries,
        medical_samples=args.medical_samples,
        seed=args.seed,
    )

    print("\n=== Building Combined dataset (query → tags + system prompt) ===")
    combined_records = build_combined_data(
        tag_meta=tag_meta,
        orca_samples=args.orca_samples,
        seed=args.seed,
    )
    combined_path = os.path.join(OUT_DIR, "sft_behavior_combined.jsonl")
    _write_jsonl(combined_path, combined_records)
    _preview("Combined", combined_records, args.preview_n)

    print("\nDone. Run training next:")
    print("  python trainer/train_full_sft.py --data_path dataset/sft_behavior_combined.jsonl "
          "--from_weight pretrain --save_weight behavior_model --epochs 3 --batch_size 32 "
          "--learning_rate 2e-5 --max_seq_len 512 --empty_think_ratio 0")


if __name__ == "__main__":
    main()
