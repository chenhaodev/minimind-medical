"""
User Behavior Theory — end-to-end inference pipeline.

Chains Model A (Intent Tagger) and Model B (Prompt Augmenter) to transform a
raw user query into a crafted system prompt ready for any downstream LLM API.

Usage:
  python scripts/run_behavior_pipeline.py --query "今天需不需要看病？"
  python scripts/run_behavior_pipeline.py  # interactive mode
"""

import argparse
import os
import sys
import warnings

import torch
from transformers import AutoTokenizer

warnings.filterwarnings("ignore")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.model_minimind import MiniMindConfig, MiniMindForCausalLM

_MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "model"))


def _load_model(weight_name: str, save_dir: str, hidden_size: int, num_layers: int, device: str):
    model = MiniMindForCausalLM(
        MiniMindConfig(hidden_size=hidden_size, num_hidden_layers=num_layers)
    )
    ckp = os.path.join(save_dir, f"{weight_name}_{hidden_size}.pth")
    if not os.path.exists(ckp):
        raise FileNotFoundError(f"Checkpoint not found: {ckp}")
    model.load_state_dict(torch.load(ckp, map_location=device, weights_only=False), strict=True)
    model = model.half()
    model.train(False)
    return model.to(device)


@torch.inference_mode()
def _generate(model, tokenizer, prompt: str, device: str, max_new_tokens: int = 128) -> str:
    conversation = [{"role": "user", "content": prompt}]
    inputs_text = tokenizer.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(inputs_text, return_tensors="pt", truncation=True).to(device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=1.0,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def run_pipeline(query: str, tagger_model, augmenter_model, tokenizer, device: str) -> dict:
    tags = _generate(tagger_model, tokenizer, query, device, max_new_tokens=32)
    system_prompt = _generate(
        augmenter_model, tokenizer,
        f"Query: {query}\nTags: {tags}",
        device, max_new_tokens=128,
    )
    return {"query": query, "tags": tags, "system_prompt": system_prompt}


def main():
    parser = argparse.ArgumentParser(description="User Behavior Theory pipeline demo")
    parser.add_argument("--query", type=str, default=None,
                        help="Input user query (omit for interactive mode)")
    parser.add_argument("--save_dir", default="out",
                        help="Checkpoint directory (default: out)")
    parser.add_argument("--tagger_weight", default="intent_tagger",
                        help="Model A weight prefix (default: intent_tagger)")
    parser.add_argument("--augmenter_weight", default="prompt_augmenter",
                        help="Model B weight prefix (default: prompt_augmenter)")
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--num_hidden_layers", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    save_dir = os.path.abspath(args.save_dir)
    tokenizer = AutoTokenizer.from_pretrained(_MODEL_DIR)

    print("Loading Model A (Intent Tagger) ...")
    tagger_model = _load_model(
        args.tagger_weight, save_dir, args.hidden_size, args.num_hidden_layers, args.device
    )
    print("Loading Model B (Prompt Augmenter) ...")
    augmenter_model = _load_model(
        args.augmenter_weight, save_dir, args.hidden_size, args.num_hidden_layers, args.device
    )
    print("Models ready.\n")

    def _show(result: dict) -> None:
        print(f"Query      : {result['query']}")
        print(f"Tags       : {result['tags']}")
        print(f"SysPrompt  : {result['system_prompt']}")
        print()

    if args.query:
        _show(run_pipeline(args.query, tagger_model, augmenter_model, tokenizer, args.device))
    else:
        print("Interactive mode — enter a query (Ctrl+C to quit)\n")
        try:
            while True:
                query = input("Query: ").strip()
                if not query:
                    continue
                _show(run_pipeline(query, tagger_model, augmenter_model, tokenizer, args.device))
        except (KeyboardInterrupt, EOFError):
            print("\nDone.")


if __name__ == "__main__":
    main()
