#!/usr/bin/env python3
"""
Knowledge distillation training script.

Loss = alpha * CE(student, hard_labels) + (1 - alpha) * T^2 * KL(student || teacher)

The teacher is frozen. The student is wrapped with LoRA.
Both can be loaded in 4-bit (QLoRA) to fit on smaller GPUs.
"""
import argparse
import json
import sys
from pathlib import Path


def log(msg: str):
    print(msg, flush=True)


def load_jsonl_dataset(path: str, template: str, max_samples: int = 0) -> list[dict]:
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
                    parts = [
                        f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>"
                        for m in obj["messages"]
                    ]
                    text = "\n".join(parts)
                else:
                    text = obj.get("text", str(obj))
            else:
                text = obj.get("text", str(obj))

            samples.append({"text": text})
            if max_samples and len(samples) >= max_samples:
                break

    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job_id", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--teacher_model", required=True)
    parser.add_argument("--student_model", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--temperature", type=float, default=4.0)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--gradient_accumulation", type=int, default=4)
    parser.add_argument("--warmup_steps", type=int, default=10)
    parser.add_argument("--instruction_template", default="alpaca")
    parser.add_argument("--use_4bit_student", action="store_true")
    parser.add_argument("--use_4bit_teacher", action="store_true")
    args = parser.parse_args()

    log(f"[INFO] Job: {args.job_id}")
    log(f"[INFO] Teacher: {args.teacher_model}")
    log(f"[INFO] Student: {args.student_model}")
    log(f"[INFO] KD temperature: {args.temperature}  alpha: {args.alpha}")
    log("[INFO] Loading dependencies...")

    import torch
    import torch.nn.functional as F
    from torch.utils.data import Dataset as TorchDataset, DataLoader
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"[INFO] Device: {device}")
    if device == "cuda":
        log(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
        total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        log(f"[INFO] VRAM: {total_vram:.1f} GB")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load teacher ──────────────────────────────────────────────────────────
    log(f"[INFO] Loading teacher tokenizer: {args.teacher_model}")
    teacher_tok = AutoTokenizer.from_pretrained(
        args.teacher_model, trust_remote_code=True, use_fast=True
    )
    if teacher_tok.pad_token is None:
        teacher_tok.pad_token = teacher_tok.eos_token

    teacher_load_kw: dict = {
        "trust_remote_code": True,
        "torch_dtype": torch.float16 if device == "cuda" else torch.float32,
    }
    if args.use_4bit_teacher and device == "cuda":
        teacher_load_kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        teacher_load_kw["device_map"] = "auto"
        log("[INFO] Teacher: 4-bit (QLoRA)")
    elif device == "cuda":
        teacher_load_kw["device_map"] = "auto"

    log(f"[INFO] Loading teacher model: {args.teacher_model}")
    teacher = AutoModelForCausalLM.from_pretrained(args.teacher_model, **teacher_load_kw)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    log("[INFO] Teacher frozen (eval mode)")

    # ── Load student ──────────────────────────────────────────────────────────
    log(f"[INFO] Loading student tokenizer: {args.student_model}")
    student_tok = AutoTokenizer.from_pretrained(
        args.student_model, trust_remote_code=True, use_fast=True
    )
    if student_tok.pad_token is None:
        student_tok.pad_token = student_tok.eos_token

    student_load_kw: dict = {
        "trust_remote_code": True,
        "torch_dtype": torch.float16 if device == "cuda" else torch.float32,
    }
    if args.use_4bit_student and device == "cuda":
        student_load_kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        student_load_kw["device_map"] = "auto"
        log("[INFO] Student: 4-bit (QLoRA)")
    elif device == "cuda":
        student_load_kw["device_map"] = "auto"

    log(f"[INFO] Loading student model: {args.student_model}")
    student = AutoModelForCausalLM.from_pretrained(args.student_model, **student_load_kw)

    if args.use_4bit_student and device == "cuda":
        student = prepare_model_for_kbit_training(student)

    log(f"[INFO] Applying LoRA to student (r={args.lora_r}, alpha={args.lora_alpha})")
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=_find_target_modules(student),
    )
    student = get_peft_model(student, lora_cfg)
    trainable, total = sum(p.numel() for p in student.parameters() if p.requires_grad), \
                       sum(p.numel() for p in student.parameters())
    log(f"[INFO] Student trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    # ── Check vocab compatibility ─────────────────────────────────────────────
    teacher_vocab = teacher.config.vocab_size
    student_vocab = student.config.vocab_size
    same_vocab = teacher_vocab == student_vocab
    log(f"[INFO] Teacher vocab: {teacher_vocab}  Student vocab: {student_vocab}")
    if not same_vocab:
        log("[WARN] Vocab sizes differ — soft-label KD will project teacher logits to student vocab size")

    # ── Dataset ───────────────────────────────────────────────────────────────
    log(f"[INFO] Loading dataset: {args.dataset}")
    raw_samples = load_jsonl_dataset(args.dataset, args.instruction_template)
    log(f"[INFO] Dataset: {len(raw_samples)} samples")

    class TextDataset(TorchDataset):
        def __init__(self, samples, tokenizer, max_len):
            self.encodings = tokenizer(
                [s["text"] for s in samples],
                truncation=True,
                max_length=max_len,
                padding="max_length",
                return_tensors="pt",
            )

        def __len__(self):
            return self.encodings["input_ids"].shape[0]

        def __getitem__(self, idx):
            return {k: v[idx] for k, v in self.encodings.items()}

    # Use student tokenizer for the training data; teacher is queried in fp
    student_dataset = TextDataset(raw_samples, student_tok, args.max_seq_length)
    # Teacher needs its own tokenized version to get logits
    teacher_dataset = TextDataset(raw_samples, teacher_tok, args.max_seq_length)

    student_loader = DataLoader(student_dataset, batch_size=args.batch_size, shuffle=False)
    teacher_loader = DataLoader(teacher_dataset, batch_size=args.batch_size, shuffle=False)

    # ── Optimizer ─────────────────────────────────────────────────────────────
    from transformers import get_linear_schedule_with_warmup

    optimizer = torch.optim.AdamW(
        [p for p in student.parameters() if p.requires_grad],
        lr=args.learning_rate,
    )
    total_steps = (len(student_loader) // args.gradient_accumulation) * args.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=max(total_steps, 1)
    )

    T = args.temperature
    alpha = args.alpha
    log(f"[INFO] Starting distillation: {args.num_epochs} epochs, {len(student_loader)} steps/epoch")
    log(f"[INFO] Loss = {alpha:.2f}*CE + {1-alpha:.2f}*KL(T={T})")

    global_step = 0
    for epoch in range(1, args.num_epochs + 1):
        epoch_ce_loss = 0.0
        epoch_kd_loss = 0.0
        student.train()
        optimizer.zero_grad()

        for step, (s_batch, t_batch) in enumerate(zip(student_loader, teacher_loader), 1):
            # Move to device (skip if device_map=auto handled it)
            if device != "cuda" or not hasattr(student, "hf_device_map"):
                s_batch = {k: v.to(device) for k, v in s_batch.items()}
                t_batch = {k: v.to(device) for k, v in t_batch.items()}

            # ── Teacher forward (no grad) ─────────────────────────────────────
            with torch.no_grad():
                t_out = teacher(
                    input_ids=t_batch["input_ids"],
                    attention_mask=t_batch["attention_mask"],
                )
                t_logits = t_out.logits  # [B, L, V_teacher]

            # ── Student forward ───────────────────────────────────────────────
            labels = s_batch["input_ids"].clone()
            labels[labels == student_tok.pad_token_id] = -100

            s_out = student(
                input_ids=s_batch["input_ids"],
                attention_mask=s_batch["attention_mask"],
                labels=labels,
            )
            ce_loss = s_out.loss
            s_logits = s_out.logits  # [B, L, V_student]

            # ── KD loss ───────────────────────────────────────────────────────
            # Align sequence length (may differ due to different tokenizations)
            min_len = min(s_logits.size(1), t_logits.size(1))
            s_logits_kd = s_logits[:, :min_len, :]
            t_logits_kd = t_logits[:, :min_len, :]

            # Align vocab: if different sizes, truncate teacher to student vocab
            min_vocab = min(s_logits_kd.size(-1), t_logits_kd.size(-1))
            s_logits_kd = s_logits_kd[:, :, :min_vocab]
            t_logits_kd = t_logits_kd[:, :, :min_vocab]

            # Reshape to [B*L, V]
            B, L, V = s_logits_kd.shape
            s_flat = s_logits_kd.reshape(B * L, V).float()
            t_flat = t_logits_kd.reshape(B * L, V).float()

            student_log_probs = F.log_softmax(s_flat / T, dim=-1)
            teacher_probs = F.softmax(t_flat / T, dim=-1)
            kd_loss = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (T ** 2)

            loss = alpha * ce_loss + (1.0 - alpha) * kd_loss
            loss = loss / args.gradient_accumulation
            loss.backward()

            epoch_ce_loss += ce_loss.item()
            epoch_kd_loss += kd_loss.item()

            if step % args.gradient_accumulation == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in student.parameters() if p.requires_grad], 1.0
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % 5 == 0:
                    avg_ce = epoch_ce_loss / step
                    avg_kd = epoch_kd_loss / step
                    log(f"[DISTILL] step={global_step} epoch={epoch} ce={avg_ce:.4f} kd={avg_kd:.4f} total={alpha*avg_ce+(1-alpha)*avg_kd:.4f}")

        log(f"[INFO] Epoch {epoch} complete — avg CE: {epoch_ce_loss/max(len(student_loader),1):.4f}  avg KD: {epoch_kd_loss/max(len(student_loader),1):.4f}")

    log(f"[INFO] Saving distilled student adapter to {output_dir}")
    student.save_pretrained(str(output_dir))
    student_tok.save_pretrained(str(output_dir))

    # Save distillation metadata alongside adapter
    meta = {
        "teacher_model": args.teacher_model,
        "student_model": args.student_model,
        "temperature": T,
        "alpha": alpha,
        "epochs": args.num_epochs,
        "type": "knowledge_distillation",
    }
    (output_dir / "distillation_config.json").write_text(json.dumps(meta, indent=2))

    log(f"[SUCCESS] Distillation complete! Student adapter saved to {output_dir}")
    log("[INFO] Use the Convert tab to merge and export as GGUF for Ollama.")
    return 0


def _find_target_modules(model) -> list[str]:
    found = set()
    for name, _ in model.named_modules():
        if any(k in name for k in ["q_proj", "v_proj", "k_proj", "o_proj",
                                    "gate_proj", "up_proj", "down_proj",
                                    "c_attn", "c_proj", "c_fc"]):
            found.add(name.split(".")[-1])
    return list(found) if found else ["q_proj", "v_proj"]


if __name__ == "__main__":
    sys.exit(main())
