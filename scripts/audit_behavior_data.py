"""
Audit sft_behavior_combined.jsonl for tag distribution, length, and quality issues.

Usage:
  python scripts/audit_behavior_data.py
  python scripts/audit_behavior_data.py --data_path dataset/sft_behavior_combined.jsonl --examples 3
"""

import argparse
import json
import os
import random
import sys
from collections import Counter

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

OUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dataset"))


def _percentile(sorted_vals: list, p: float) -> int:
    if not sorted_vals:
        return 0
    idx = int(len(sorted_vals) * p / 100)
    return sorted_vals[min(idx, len(sorted_vals) - 1)]


def audit(data_path: str, examples_per_bucket: int, seed: int) -> None:
    rng = random.Random(seed)

    records = []
    malformed = 0
    seen_keys: set[str] = set()
    duplicates = 0

    print(f"Reading {data_path} ...")
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue
            convs = obj.get("conversations", [])
            if len(convs) < 2:
                malformed += 1
                continue
            user_content = convs[0].get("content", "")
            asst_content = convs[-1].get("content", "")
            if "\n\n" not in asst_content:
                malformed += 1
                continue
            key = user_content[:100]
            if key in seen_keys:
                duplicates += 1
            else:
                seen_keys.add(key)
            records.append((user_content, asst_content))

    total = len(records)
    print(f"\nTotal records : {total:,}")
    print(f"Malformed     : {malformed:,}")
    print(f"Duplicates    : {duplicates:,}")

    # --- Tag distribution ---
    tag_counts: Counter = Counter()
    bucket_samples: dict[str, list] = {}
    query_lengths = []

    for user_content, asst_content in records:
        tag_line = asst_content.split("\n\n", 1)[0].strip()
        tag_counts[tag_line] += 1
        bucket_samples.setdefault(tag_line, []).append((user_content, asst_content))
        query_lengths.append(len(user_content))

    print("\n--- Tag distribution ---")
    print(f"{'Tag':<45} {'Count':>7}  {'%':>5}")
    print("-" * 60)
    for tag, count in sorted(tag_counts.items(), key=lambda x: -x[1]):
        print(f"{tag:<45} {count:>7,}  {count/total*100:>5.1f}%")

    # --- Length statistics ---
    query_lengths.sort()
    print("\n--- Query length (chars) ---")
    print(f"  p50={_percentile(query_lengths, 50)}  "
          f"p90={_percentile(query_lengths, 90)}  "
          f"p99={_percentile(query_lengths, 99)}  "
          f"max={query_lengths[-1] if query_lengths else 0}")

    # --- Examples per bucket ---
    print(f"\n--- {examples_per_bucket} examples per tag bucket ---")
    for tag in sorted(tag_counts.keys()):
        samples = bucket_samples[tag]
        chosen = rng.sample(samples, min(examples_per_bucket, len(samples)))
        print(f"\n[{tag}]  ({tag_counts[tag]:,} total)")
        for user_c, asst_c in chosen:
            sys_prompt = asst_c.split("\n\n", 1)[1].strip() if "\n\n" in asst_c else asst_c
            print(f"  USER: {user_c[:100]!r}")
            print(f"  SYS : {sys_prompt[:100]!r}")


def main():
    parser = argparse.ArgumentParser(description="Audit sft_behavior_combined.jsonl")
    parser.add_argument("--data_path", default=os.path.join(OUT_DIR, "sft_behavior_combined.jsonl"))
    parser.add_argument("--examples", type=int, default=2,
                        help="Random examples to show per tag bucket (default: 2)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not os.path.exists(args.data_path):
        print(f"File not found: {args.data_path}")
        print("Run python scripts/prepare_behavior_data.py first.")
        sys.exit(1)

    audit(args.data_path, args.examples, args.seed)


if __name__ == "__main__":
    main()
