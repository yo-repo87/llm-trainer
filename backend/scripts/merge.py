#!/usr/bin/env python3
"""Merge a LoRA adapter into its base model and save the full merged model."""
import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    print(f"[INFO] Loading base model: {args.base_model}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    device_map = "cuda:1" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Loading onto: {device_map} (host RAM is limited on this box, prefer GPU)", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        dtype=torch.float16,
        device_map=device_map,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    print(f"[INFO] Loading LoRA adapter: {args.adapter}", flush=True)
    model = PeftModel.from_pretrained(model, args.adapter)

    print("[INFO] Merging adapter weights into base model...", flush=True)
    model = model.merge_and_unload()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out))
    tokenizer.save_pretrained(str(out))
    print(f"[INFO] Merged model saved to {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
