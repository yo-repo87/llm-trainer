#!/usr/bin/env python3
"""Standalone training script — run by the server as a subprocess."""
import argparse
import json
import os
import sys
from pathlib import Path


def log(msg: str):
    print(msg, flush=True)


def load_dataset_jsonl(path: str, template: str, tokenizer, max_len: int):
    from datasets import Dataset

    samples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            if template == "alpaca":
                if "instruction" in obj:
                    inp = obj.get("input", "")
                    text = (
                        f"### Instruction:\n{obj['instruction']}\n\n"
                        + (f"### Input:\n{inp}\n\n" if inp else "")
                        + f"### Response:\n{obj.get('output', obj.get('response', ''))}"
                    )
                else:
                    text = obj.get("text", str(obj))
            elif template == "chatml":
                if "messages" in obj:
                    parts = []
                    for m in obj["messages"]:
                        parts.append(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>")
                    text = "\n".join(parts)
                else:
                    text = obj.get("text", str(obj))
            else:
                text = obj.get("text", str(obj))

            samples.append({"text": text})

    return Dataset.from_list(samples)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job_id", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--warmup_steps", type=int, default=10)
    parser.add_argument("--gradient_accumulation", type=int, default=4)
    parser.add_argument("--instruction_template", default="alpaca")
    parser.add_argument("--use_4bit", action="store_true")
    args = parser.parse_args()

    log(f"[INFO] Job {args.job_id} — loading dependencies...")

    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
        Trainer,
        DataCollatorForLanguageModeling,
        BitsAndBytesConfig,
    )
    from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"[INFO] Device: {device}")
    if device == "cuda":
        log(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log(f"[INFO] Loading tokenizer: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model, trust_remote_code=True, use_fast=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    log(f"[INFO] Loading base model: {args.base_model}")
    # "auto" lets accelerate consider multi-device/CPU-offload dispatch, which
    # calls model.to(device) internally — unsupported for 4-bit bitsandbytes
    # weights and raises ValueError. Force single-device placement instead.
    if device == "cuda":
        device_map = {"": 0} if args.use_4bit else "auto"
    else:
        device_map = None
    load_kwargs = {
        "trust_remote_code": True,
        "dtype": torch.float16 if device == "cuda" else torch.float32,
        "device_map": device_map,
    }

    if args.use_4bit and device == "cuda":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        load_kwargs["quantization_config"] = bnb_config
        log("[INFO] Using 4-bit quantization (QLoRA)")

    model = AutoModelForCausalLM.from_pretrained(args.base_model, **load_kwargs)

    if args.use_4bit and device == "cuda":
        model = _prepare_kbit_training_light(model)

    log(f"[INFO] Applying LoRA (r={args.lora_r}, alpha={args.lora_alpha})")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=_find_target_modules(model),
    )
    model = get_peft_model(model, lora_config)
    trainable, total = _count_params(model)
    log(f"[INFO] Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    log(f"[INFO] Loading dataset: {args.dataset}")
    raw_ds = load_dataset_jsonl(args.dataset, args.instruction_template, tokenizer, args.max_seq_length)
    log(f"[INFO] Dataset size: {len(raw_ds)} samples")

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=args.max_seq_length,
            padding=False,
        )

    tokenized = raw_ds.map(tokenize, batched=True, remove_columns=["text"])
    log(f"[INFO] Tokenization complete")

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        logging_steps=10,
        save_steps=args.warmup_steps * 5,
        save_total_limit=2,
        fp16=device == "cuda",
        report_to="none",
        dataloader_num_workers=0,
        remove_unused_columns=False,
        # Standard QLoRA memory fix: default AdamW keeps fp32 momentum+variance
        # for every trainable param resident on GPU. Paged 8-bit Adam keeps
        # 8-bit state and pages it to CPU RAM under memory pressure instead —
        # this host's GPUs are right at their VRAM ceiling for this model size.
        optim="paged_adamw_8bit" if (args.use_4bit and device == "cuda") else "adamw_torch",
    )

    import time

    class LoggingTrainer(Trainer):
        def log(self, logs: dict, start_time=None):
            super().log(logs, start_time) if start_time is not None else super().log(logs)
            step = logs.get("step", self.state.global_step)
            loss = logs.get("loss")
            lr = logs.get("learning_rate")
            if loss is not None:
                epoch = logs.get("epoch", "?")
                msg = f"[TRAIN] step={step} epoch={epoch} loss={loss:.4f}"
                if lr:
                    msg += f" lr={lr:.2e}"
                print(msg, flush=True)

                if not hasattr(self, "_train_start"):
                    self._train_start = time.time()
                elapsed = time.time() - self._train_start
                max_steps = self.state.max_steps
                pct = round(step / max_steps * 100, 1) if max_steps > 0 else 0
                eta = int(elapsed / step * (max_steps - step)) if step > 0 else 0
                print(
                    f"[PROGRESS] step={step} total={max_steps} pct={pct}"
                    f" elapsed={int(elapsed)} eta={eta}",
                    flush=True,
                )

    trainer = LoggingTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=collator,
        processing_class=tokenizer,
    )

    log("[INFO] Starting training...")
    trainer.train()

    log(f"[INFO] Saving adapter to {output_dir}")
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    log(f"[SUCCESS] Training complete. Adapter saved to {output_dir}")
    log(f"[INFO] To use with Ollama, merge and convert to GGUF using the Models tab.")
    return 0


def _prepare_kbit_training_light(model, numel_threshold: int = 5_000_000):
    """Same as peft's prepare_model_for_kbit_training (freeze base params,
    upcast non-4bit params to fp32, enable gradient checkpointing), except it
    skips the fp32 upcast for any single parameter tensor larger than
    numel_threshold. Large-vocab models (e.g. Qwen2.5's 151,936-token
    embed_tokens/lm_head, tied so ~544M params) would otherwise get upcast to
    fp32, spiking VRAM by ~1-2GB for a tensor that's frozen anyway (LoRA never
    trains it) and doesn't need fp32 precision to stay numerically stable
    through a forward/backward pass.
    """
    import torch

    for param in model.parameters():
        param.requires_grad = False

    for param in model.parameters():
        if (
            (param.dtype == torch.float16 or param.dtype == torch.bfloat16)
            and param.__class__.__name__ != "Params4bit"
            and param.numel() <= numel_threshold
        ):
            param.data = param.data.to(torch.float32)

    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    else:
        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)

        model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    model.gradient_checkpointing_enable()
    return model


def _find_target_modules(model) -> list[str]:
    modules = set()
    for name, _ in model.named_modules():
        if any(k in name for k in ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]):
            part = name.split(".")[-1]
            modules.add(part)
    if modules:
        return list(modules)
    # Fallback for GPT-style models
    for name, _ in model.named_modules():
        if any(k in name for k in ["c_attn", "c_proj", "c_fc"]):
            part = name.split(".")[-1]
            modules.add(part)
    return list(modules) if modules else ["q_proj", "v_proj"]


def _count_params(model) -> tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


if __name__ == "__main__":
    sys.exit(main())
