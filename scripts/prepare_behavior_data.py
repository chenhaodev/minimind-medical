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
    ("creative",    "detailed", re.compile(r"写一|写个|帮我写|创作|生成一|写一篇")),
    ("plan",        "detailed", re.compile(r"如何|怎么|怎样|步骤|方法|流程|怎么做")),
]
_ZH_REF_RE = re.compile(r"医|药|病|症|健康|法律|法规|政策|科学|研究|论文")

_EN_CLASSIFY_RULES = [
    ("judgment",    "brief",    re.compile(r"\bshould\b|is it\b|can i\b|do i need\b", re.I)),
    ("explanation", "detailed", re.compile(r"\bwhy\b|\bexplain\b|what is\b|\bdescribe\b", re.I)),
    ("creative",    "detailed", re.compile(r"\bwrite\b|\bcreate\b|\bgenerate\b|\bcompose\b", re.I)),
    ("plan",        "detailed", re.compile(r"how to\b|how do\b|\bsteps\b|\bguide\b", re.I)),
]
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
        task_type, style = "fact", "brief"
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
# Model A: Intent Tagger dataset
# ---------------------------------------------------------------------------

def build_intent_tagger_data(
    belle_samples: int,
    medical_queries: list,
    medical_samples: int,
    seed: int,
) -> tuple:
    """Returns (sft_records, tag_meta).

    tag_meta is a list of (query, task_type, style, ref) tuples passed to the
    augmenter builder so it can derive style prompts without re-parsing strings.
    """
    rng = random.Random(seed)
    records = []
    tag_meta = []  # (query, task_type, style, ref)

    def _add(query: str, task_type: str, style: str, ref: str) -> None:
        records.append({
            "conversations": [
                {"role": "user", "content": query},
                {"role": "assistant", "content": _make_tag(task_type, style, ref)},
            ]
        })
        tag_meta.append((query, task_type, style, ref))

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
        _add(query, task_type, style, ref)
    print(f"  Dolly: {len(records):,} examples")

    # Source 2: BelleGroup/train_0.5M_CN (reservoir sampling — O(belle_samples) memory)
    print(f"  Loading BelleGroup/train_0.5M_CN (streaming, sample {belle_samples:,}) ...")
    belle_stream = load_dataset("BelleGroup/train_0.5M_CN", split="train", streaming=True)
    reservoir = []
    for i, row in enumerate(belle_stream):
        instr = _clean(row.get("instruction", "") or "")
        if len(instr) < 10:
            continue
        if len(reservoir) < belle_samples:
            reservoir.append(instr)
        else:
            j = rng.randint(0, i)
            if j < belle_samples:
                reservoir[j] = instr
    belle_start = len(records)
    for instr in reservoir:
        task_type, style, ref = _classify(instr, _ZH_CLASSIFY_RULES, _ZH_REF_RE)
        _add(instr, task_type, style, ref)
    print(f"  BelleGroup: {len(records) - belle_start:,} examples")

    # Source 3: pre-loaded medical queries (ref:yes bucket)
    med_start = len(records)
    for q in medical_queries[:medical_samples]:
        _add(q, "explanation", "detailed", "yes")
    print(f"  Medical: {len(records) - med_start:,} examples (ref:yes)")

    # Shuffle records and tag_meta together
    indices = list(range(len(records)))
    rng.shuffle(indices)
    records = [records[i] for i in indices]
    tag_meta = [tag_meta[i] for i in indices]

    return records, tag_meta


# ---------------------------------------------------------------------------
# Model B: Prompt Augmenter dataset
# ---------------------------------------------------------------------------

def build_prompt_augmenter_data(
    tag_meta: list,
    orca_samples: int,
    medical_queries: list,
    medical_samples: int,
    seed: int,
) -> list:
    rng = random.Random(seed)
    records = []

    # Source 1: derive from tagger tag_meta (structured tuples — no string parsing needed)
    for query, task_type, style, ref in tag_meta:
        tag = _make_tag(task_type, style, ref)
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
        task_type, style, ref = _classify(question, _EN_CLASSIFY_RULES, _EN_REF_RE)
        tag = _make_tag(task_type, style, ref)
        records.append({
            "conversations": [
                {"role": "user", "content": f"Query: {question}\nTags: {tag}"},
                {"role": "assistant", "content": f"System: {_clean(sys_prompt)}\nQuery: {question}"},
            ]
        })
        orca_added += 1
        if orca_added >= orca_samples:
            break
    print(f"  OpenOrca: {orca_added:,} examples with style prompts")

    # Source 3: medical queries with reference-requiring system prompt
    ref_prompt = _STYLE_PROMPTS["ref"]
    med_tag = _make_tag("explanation", "detailed", "yes")
    med_start = len(records)
    for q in medical_queries[:medical_samples]:
        records.append({
            "conversations": [
                {"role": "user", "content": f"Query: {q}\nTags: {med_tag}"},
                {"role": "assistant", "content": f"System: {ref_prompt}\nQuery: {q}"},
            ]
        })
    print(f"  Medical augmenter: {len(records) - med_start:,} examples (ref:yes)")

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
    rng = random.Random(args.seed)

    # Load medical queries once; reused by both builders
    medical_queries = []
    medical_path = os.path.join(OUT_DIR, "sft_medical.jsonl")
    if os.path.exists(medical_path):
        print(f"Pre-loading sft_medical.jsonl ...")
        medical_queries = _load_medical_queries(medical_path, rng)
        print(f"  {len(medical_queries):,} medical queries loaded")
    else:
        print("sft_medical.jsonl not found — skipping medical source")

    print("\n=== Building Intent Tagger dataset (Model A) ===")
    tagger_records, tag_meta = build_intent_tagger_data(
        belle_samples=args.belle_samples,
        medical_queries=medical_queries,
        medical_samples=args.medical_samples,
        seed=args.seed,
    )
    tagger_path = os.path.join(OUT_DIR, "sft_intent_tagger.jsonl")
    _write_jsonl(tagger_path, tagger_records)
    _preview("Intent Tagger", tagger_records, args.preview_n)

    print("\n=== Building Prompt Augmenter dataset (Model B) ===")
    augmenter_records = build_prompt_augmenter_data(
        tag_meta=tag_meta,
        orca_samples=args.orca_samples,
        medical_queries=medical_queries,
        medical_samples=args.medical_samples,
        seed=args.seed,
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
