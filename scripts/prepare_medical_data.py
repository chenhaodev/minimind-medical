"""
Bilingual (ZH + EN) medical dataset preparation for MiniMind.

Outputs:
  dataset/pretrain_medical.jsonl   {"text": "..."}
  dataset/sft_medical.jsonl        {"conversations": [...]}

Usage:
  python scripts/prepare_medical_data.py --stage all
  python scripts/prepare_medical_data.py --stage pretrain
  python scripts/prepare_medical_data.py --stage sft
  python scripts/prepare_medical_data.py --stage all --pubmed_samples 50000
"""

import argparse
import json
import os
import random
import re
import sys
import unicodedata

__package__ = "scripts"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from datasets import load_dataset
from simhash import Simhash, SimhashIndex

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

OUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dataset"))

MEDICAL_SYSTEM_PROMPTS = [
    "你是一位专业的医疗AI助手，请根据医学知识为用户提供准确的健康信息。",
    "你是一位医学顾问，请用专业且通俗易懂的语言回答患者的医疗问题。",
    "你是MiniMind医疗版，请为用户提供医学相关的准确信息，但不能替代实际就医。",
    "You are a knowledgeable medical AI assistant. Provide accurate health information based on medical evidence.",
    "You are a medical assistant. Answer questions clearly and accurately. Always recommend consulting a doctor for personal medical decisions.",
]

# role aliases used by ShareGPT-format datasets
_SHAREGPT_ROLE_MAP = {
    "human": "user",
    "user": "user",
    "gpt": "assistant",
    "assistant": "assistant",
    "system": "system",
    "tool": "tool",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_garbled(text: str) -> bool:
    if not text:
        return True
    # Only count truly non-printable control chars (excluding \n \t \r which are valid)
    bad = sum(
        1 for c in text
        if c not in ("\n", "\t", "\r")
        and (unicodedata.category(c) in ("Cc", "Cs") or c == "�")
    )
    return bad / len(text) > 0.3


def _clean(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _normalize_conversations(conversations: list) -> list:
    """Convert ShareGPT-format (from/value) or mixed dicts to MiniMind role/content format.

    Always passes the role through _SHAREGPT_ROLE_MAP so that ShareGPT-style
    values like "gpt" or "human" found in the `role` field are normalised to
    "assistant" / "user" rather than passed through verbatim.
    """
    out = []
    for turn in conversations:
        if not isinstance(turn, dict):
            continue
        # Prefer `role` field; fall back to ShareGPT `from` field
        raw_role = str(turn.get("role") or turn.get("from") or "")
        # Always map through the alias table; unknown roles produce "" → turn dropped
        role = _SHAREGPT_ROLE_MAP.get(raw_role, "")
        content = turn.get("content") or turn.get("value") or ""
        if role and isinstance(content, str):
            out.append({"role": role, "content": _clean(content)})
    return out


def _simhash_dedup(records: list, text_fn, threshold: int = 3) -> list:
    """Remove near-duplicates using SimhashIndex for O(n log n) lookup."""
    # SimhashIndex requires k = number of differing bits tolerated
    index = SimhashIndex([], k=threshold)
    out = []
    uid = 0
    for rec in records:
        key = text_fn(rec)
        if not key:
            continue  # drop records with no extractable key — likely malformed
        h = Simhash(key)
        if not index.get_near_dups(h):
            index.add(str(uid), h)
            uid += 1
            out.append(rec)
    return out


def _write_jsonl(path: str, records: list) -> None:
    dirname = os.path.dirname(path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"  Wrote {len(records):,} records → {path}")


def _has_assistant_content(conversations: list) -> bool:
    """Return True only if at least one assistant turn has non-empty content."""
    return any(
        t.get("role") == "assistant" and t.get("content", "").strip()
        for t in conversations
    )


def _strip_prefix(s: str, prefix: str) -> str:
    s = _clean(str(s))
    return s[len(prefix):].strip() if s.startswith(prefix) else s


def _maybe_add_system(conversations: list, ratio: float = 0.2) -> list:
    if conversations and conversations[0].get("role") == "system":
        return conversations
    if random.random() < ratio:
        return [{"role": "system", "content": random.choice(MEDICAL_SYSTEM_PROMPTS)}] + conversations
    return conversations

# ---------------------------------------------------------------------------
# Pretrain sources
# ---------------------------------------------------------------------------

def load_pretrain_huatuo_gpt_as_text(max_samples: int = 50_000) -> list:
    """FreedomIntelligence/HuatuoGPT-sft-data-v1 — use as ZH medical pretrain text.

    Schema: each row has a `data` field = [question_str, answer_str] where
    strings are already prefixed with "问：" / "答：".
    """
    print(f"  [ZH] Loading HuatuoGPT-sft-data-v1 as pretrain text (up to {max_samples:,}) ...")
    try:
        ds = load_dataset("FreedomIntelligence/HuatuoGPT-sft-data-v1", split="train")
    except Exception as e:
        print(f"    Skipped: {e}")
        return []
    records = []
    for i, row in enumerate(ds):
        if i >= max_samples:
            break
        data = row.get("data") or []
        if isinstance(data, list) and len(data) >= 2:
            text = _clean("\n".join(str(s) for s in data if s))
            if len(text) >= 30 and not _is_garbled(text):
                records.append({"text": text})
    print(f"    → {len(records):,} entries")
    return records


def load_pretrain_pubmed(max_samples: int = 50_000) -> list:
    """ccdv/pubmed-summarization — PubMed abstracts (English, Parquet-based)."""
    print(f"  [EN] Loading PubMed abstracts via ccdv/pubmed-summarization (up to {max_samples:,}) ...")
    try:
        ds = load_dataset("ccdv/pubmed-summarization", "document", split="train", streaming=True)
    except Exception as e:
        print(f"    Skipped: {e}")
        return []
    records = []
    for i, row in enumerate(ds):
        if i >= max_samples:
            break
        # "abstract" field contains the abstract; "article" is the full paper
        abstract = _clean(str(row.get("abstract") or ""))
        if len(abstract) >= 50 and not _is_garbled(abstract):
            records.append({"text": abstract})
    print(f"    → {len(records):,} entries")
    return records


def load_pretrain_wikidoc() -> list:
    """medalpaca/medical_meadow_wikidoc — medical reference text (English)."""
    print("  [EN] Loading medical_meadow_wikidoc ...")
    try:
        ds = load_dataset("medalpaca/medical_meadow_wikidoc", split="train")
    except Exception as e:
        print(f"    Skipped: {e}")
        return []
    records = []
    for row in ds:
        # "output" contains the answer/explanation; prefer it over "input" which is the question
        text = _clean(str(row.get("output") or row.get("input") or ""))
        if len(text) >= 50 and not _is_garbled(text):
            records.append({"text": text})
    print(f"    → {len(records):,} entries")
    return records


def load_pretrain_medrag_textbooks() -> list:
    """MedRAG/textbooks — chunked medical textbook passages (English)."""
    print("  [EN] Loading MedRAG/textbooks ...")
    try:
        ds = load_dataset("MedRAG/textbooks", split="train")
    except Exception as e:
        print(f"    Skipped: {e}")
        return []
    records = []
    for row in ds:
        text = _clean(str(row.get("content") or row.get("text") or ""))
        if len(text) >= 50 and not _is_garbled(text):
            records.append({"text": text})
    print(f"    → {len(records):,} entries")
    return records


# ---------------------------------------------------------------------------
# SFT sources
# ---------------------------------------------------------------------------

def load_sft_huatuo_gpt() -> list:
    """FreedomIntelligence/HuatuoGPT-sft-data-v1 — 226k high-quality ZH pairs.

    Schema: each row has `data` = [question_str, answer_str] prefixed with 问：/答：.
    """
    print("  [ZH] Loading HuatuoGPT-sft-data-v1 ...")
    try:
        ds = load_dataset("FreedomIntelligence/HuatuoGPT-sft-data-v1", split="train")
    except Exception as e:
        print(f"    Skipped: {e}")
        return []
    records = []
    for row in ds:
        data = row.get("data") or []
        if not (isinstance(data, list) and len(data) >= 2):
            continue
        q = _strip_prefix(data[0], "问：")
        a = _strip_prefix(data[1], "答：")
        if len(q) >= 5 and len(a) >= 20 and not _is_garbled(a):
            conversations = _maybe_add_system([
                {"role": "user", "content": q},
                {"role": "assistant", "content": a},
            ])
            records.append({"conversations": conversations})
    print(f"    → {len(records):,} entries")
    return records


def load_sft_disc_med(max_samples: int = 100_000) -> list:
    """Flmc/DISC-Med-SFT — Chinese multi-turn medical consultations.

    Schema: `conversation` (singular) = list of {"role": "user"/"assistant", "content": "..."}.
    """
    print(f"  [ZH] Loading DISC-Med-SFT (up to {max_samples:,}) ...")
    try:
        ds = load_dataset("Flmc/DISC-Med-SFT", split="train")
    except Exception as e:
        print(f"    Skipped: {e}")
        return []
    records = []
    for i, row in enumerate(ds):
        if i >= max_samples:
            break
        # Field is "conversation" (singular), not "conversations"
        raw_convs = row.get("conversation") or []
        conversations = _normalize_conversations(raw_convs) if raw_convs else []
        if conversations and _has_assistant_content(conversations):
            if (all(t.get("content", "").strip() for t in conversations)
                    and not any(_is_garbled(t.get("content", "")) for t in conversations)):
                conversations = _maybe_add_system(conversations)
                records.append({"conversations": conversations})
    print(f"    → {len(records):,} entries")
    return records


def load_sft_chatmed(max_samples: int = 100_000) -> list:
    """michaelwzhu/ChatMed_Consult_Dataset — real patient-doctor consultations.

    Schema: `query` (patient question), `response` (doctor answer).
    """
    print(f"  [ZH] Loading ChatMed_Consult_Dataset (up to {max_samples:,}) ...")
    try:
        ds = load_dataset("michaelwzhu/ChatMed_Consult_Dataset", split="train")
    except Exception as e:
        print(f"    Skipped: {e}")
        return []
    records = []
    for i, row in enumerate(ds):
        if i >= max_samples:
            break
        q = _clean(str(row.get("query") or ""))
        a = _clean(str(row.get("response") or ""))
        if len(q) >= 5 and len(a) >= 20 and not _is_garbled(a):
            conversations = _maybe_add_system([
                {"role": "user", "content": q},
                {"role": "assistant", "content": a},
            ])
            records.append({"conversations": conversations})
    print(f"    → {len(records):,} entries")
    return records


def load_sft_cmtmedqa() -> list:
    """Suprit/CMtMedQA — high-quality multi-turn Chinese medical QA.

    Schema: `instruction` (current question), `output` (answer),
    `history` = list of [question_str, answer_str] pairs (prior turns).
    """
    print("  [ZH] Loading CMtMedQA ...")
    try:
        ds = load_dataset("Suprit/CMtMedQA", split="train")
    except Exception as e:
        print(f"    Skipped: {e}")
        return []
    records = []
    for row in ds:
        conversations = []
        # Reconstruct multi-turn from history
        history = row.get("history") or []
        if isinstance(history, list):
            for item in history:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    conversations.append({"role": "user",      "content": _clean(str(item[0]))})
                    conversations.append({"role": "assistant", "content": _clean(str(item[1]))})
        instruction = _clean(str(row.get("instruction") or ""))
        output      = _clean(str(row.get("output") or ""))
        if instruction:
            conversations.append({"role": "user",      "content": instruction})
        if output:
            conversations.append({"role": "assistant", "content": output})
        if conversations and _has_assistant_content(conversations):
            if not any(_is_garbled(t.get("content", "")) for t in conversations):
                conversations = _maybe_add_system(conversations)
                records.append({"conversations": conversations})
    print(f"    → {len(records):,} entries")
    return records


def load_sft_healthcaremagic() -> list:
    """lavita/ChatDoctor-HealthCareMagic-100k — 100k EN patient-doctor pairs."""
    print("  [EN] Loading ChatDoctor-HealthCareMagic-100k ...")
    try:
        ds = load_dataset("lavita/ChatDoctor-HealthCareMagic-100k", split="train")
    except Exception as e:
        print(f"    Skipped: {e}")
        return []
    records = []
    for row in ds:
        q = _clean(str(row.get("input") or row.get("question") or ""))
        a = _clean(str(row.get("output") or row.get("answer") or ""))
        if len(q) >= 5 and len(a) >= 20 and not _is_garbled(q) and not _is_garbled(a):
            conversations = _maybe_add_system([
                {"role": "user", "content": q},
                {"role": "assistant", "content": a},
            ])
            records.append({"conversations": conversations})
    print(f"    → {len(records):,} entries")
    return records


def load_sft_medalpaca_mediqa() -> list:
    """medalpaca/medical_meadow_mediqa — clinical NLP Q&A (EN)."""
    print("  [EN] Loading medical_meadow_mediqa ...")
    try:
        ds = load_dataset("medalpaca/medical_meadow_mediqa", split="train")
    except Exception as e:
        print(f"    Skipped: {e}")
        return []
    records = []
    for row in ds:
        q = _clean(str(row.get("input") or ""))
        a = _clean(str(row.get("output") or ""))
        if len(q) >= 5 and len(a) >= 20 and not _is_garbled(a):
            conversations = _maybe_add_system([
                {"role": "user", "content": q},
                {"role": "assistant", "content": a},
            ])
            records.append({"conversations": conversations})
    print(f"    → {len(records):,} entries")
    return records


def load_sft_medalpaca_health_advice() -> list:
    """medalpaca/medical_meadow_health_advice — EN health advice Q&A."""
    print("  [EN] Loading medical_meadow_health_advice ...")
    try:
        ds = load_dataset("medalpaca/medical_meadow_health_advice", split="train")
    except Exception as e:
        print(f"    Skipped: {e}")
        return []
    records = []
    for row in ds:
        q = _clean(str(row.get("input") or ""))
        a = _clean(str(row.get("output") or ""))
        if len(q) >= 5 and len(a) >= 20 and not _is_garbled(a):
            conversations = _maybe_add_system([
                {"role": "user", "content": q},
                {"role": "assistant", "content": a},
            ])
            records.append({"conversations": conversations})
    print(f"    → {len(records):,} entries")
    return records


def load_sft_pubmedqa() -> list:
    """pubmed_qa — PubMed research Q&A (EN)."""
    print("  [EN] Loading PubMedQA ...")
    try:
        ds = load_dataset("pubmed_qa", "pqa_labeled", split="train")
    except Exception as e:
        print(f"    Skipped: {e}")
        return []
    records = []
    for row in ds:
        q = _clean(str(row.get("question") or ""))
        long_ans = _clean(str(row.get("long_answer") or ""))
        yes_no = _clean(str(row.get("final_decision") or ""))
        if q and long_ans and not _is_garbled(long_ans):
            a = long_ans + (f"\n\nIn summary: {yes_no}." if yes_no else "")
            conversations = _maybe_add_system([
                {"role": "user", "content": q},
                {"role": "assistant", "content": a},
            ])
            records.append({"conversations": conversations})
    print(f"    → {len(records):,} entries")
    return records


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def build_pretrain(pubmed_samples: int, huatuo_samples: int) -> None:
    print("\n=== Building pretrain_medical.jsonl ===")
    all_records: list = []
    all_records += load_pretrain_huatuo_gpt_as_text(max_samples=huatuo_samples)
    all_records += load_pretrain_wikidoc()
    all_records += load_pretrain_medrag_textbooks()
    all_records += load_pretrain_pubmed(max_samples=pubmed_samples)

    print(f"\nTotal before dedup: {len(all_records):,}")
    print("  Deduplicating ...")
    all_records = _simhash_dedup(all_records, lambda r: r["text"][:200])
    print(f"  After dedup: {len(all_records):,}")

    random.shuffle(all_records)
    _write_jsonl(os.path.join(OUT_DIR, "pretrain_medical.jsonl"), all_records)


def build_sft(disc_samples: int, chatmed_samples: int) -> None:
    print("\n=== Building sft_medical.jsonl ===")
    all_records: list = []
    all_records += load_sft_huatuo_gpt()
    all_records += load_sft_cmtmedqa()
    all_records += load_sft_disc_med(max_samples=disc_samples)
    all_records += load_sft_chatmed(max_samples=chatmed_samples)
    all_records += load_sft_healthcaremagic()
    all_records += load_sft_medalpaca_mediqa()
    all_records += load_sft_medalpaca_health_advice()
    all_records += load_sft_pubmedqa()

    print(f"\nTotal before dedup: {len(all_records):,}")
    print("  Deduplicating ...")

    def sft_key(r: dict) -> str:
        for turn in r.get("conversations", []):
            if turn.get("role") == "user":
                return turn.get("content", "")[:200]
        return ""

    all_records = _simhash_dedup(all_records, sft_key)
    print(f"  After dedup: {len(all_records):,}")

    random.shuffle(all_records)
    _write_jsonl(os.path.join(OUT_DIR, "sft_medical.jsonl"), all_records)


def validate(path: str) -> None:
    print(f"\nValidating {path} ...")
    errors = 0
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                json.loads(line)
            except json.JSONDecodeError as e:
                print(f"  Line {i + 1} invalid: {e}")
                errors += 1
    print(f"  {'OK' if errors == 0 else f'{errors} errors found'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare bilingual medical datasets for MiniMind")
    parser.add_argument("--stage", choices=["pretrain", "sft", "all"], default="all")
    parser.add_argument("--pubmed_samples", type=int, default=50_000,
                        help="Max PubMed abstracts to include (default: 50000 mini; use 500000 for full)")
    parser.add_argument("--huatuo_samples", type=int, default=50_000,
                        help="Max Huatuo-26M entries to include (default: 50000 mini; use 300000 for full)")
    parser.add_argument("--disc_samples", type=int, default=20_000,
                        help="Max DISC-Med-SFT entries (default: 20000 mini; use 100000 for full)")
    parser.add_argument("--chatmed_samples", type=int, default=20_000,
                        help="Max ChatMed entries (default: 20000 mini; use 100000 for full)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    if args.stage in ("pretrain", "all"):
        build_pretrain(pubmed_samples=args.pubmed_samples, huatuo_samples=args.huatuo_samples)
        pretrain_path = os.path.join(OUT_DIR, "pretrain_medical.jsonl")
        if os.path.exists(pretrain_path):
            validate(pretrain_path)

    if args.stage in ("sft", "all"):
        build_sft(disc_samples=args.disc_samples, chatmed_samples=args.chatmed_samples)
        sft_path = os.path.join(OUT_DIR, "sft_medical.jsonl")
        if os.path.exists(sft_path):
            validate(sft_path)

    print("\nDone. Next steps:")
    print("  1. python trainer/train_pretrain.py --data_path ../dataset/pretrain_medical.jsonl --from_weight pretrain --save_weight pretrain_medical --learning_rate 1e-4 --epochs 1")
    print("  2. python trainer/train_full_sft.py  --data_path ../dataset/sft_medical.jsonl    --from_weight pretrain_medical --save_weight full_sft_medical --learning_rate 5e-6 --epochs 1")
    print("  3. python eval_llm.py --weight full_sft_medical")
