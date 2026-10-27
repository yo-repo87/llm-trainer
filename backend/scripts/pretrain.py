#!/usr/bin/env python3
"""
Pre-training a GPT-2 style LLM from scratch.

Steps:
  1. Load text corpus from dataset file
  2. Optionally train a BPE tokenizer on the corpus
  3. Tokenize corpus into one long token sequence
  4. Create sliding-window chunks (x, y pairs)
  5. Train model with cosine LR + warmup, mixed precision, gradient clipping
  6. Save checkpoints and final model in HuggingFace format
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path


def log(msg: str):
    print(msg, flush=True)


# ── Dataset loading ────────────────────────────────────────────────────────────

def load_corpus(dataset_path: str) -> str:
    """Return all text from the dataset as one big string."""
    path = Path(dataset_path)
    ext = path.suffix.lower()
    chunks = []

    if ext == ".jsonl":
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    # Prefer 'text', fall back to joining string values
                    text = obj.get("text") or obj.get("content") or obj.get("output") or ""
                    if not text and "instruction" in obj:
                        inp = obj.get("input", "")
                        text = obj["instruction"] + ("\n" + inp if inp else "") + "\n" + obj.get("output", "")
                    if text:
                        chunks.append(text.strip())
                except Exception:
                    pass
    elif ext == ".json":
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, list):
            for item in data:
                if isinstance(item, str):
                    chunks.append(item)
                elif isinstance(item, dict):
                    text = item.get("text") or item.get("content") or ""
                    if text:
                        chunks.append(text)
        elif isinstance(data, dict):
            chunks.append(str(data))
    elif ext == ".csv":
        import csv
        with open(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                text = row.get("text") or row.get("content") or " ".join(row.values())
                chunks.append(text)
    else:
        # Plain text — treat each non-empty line as a document
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    chunks.append(line)

    corpus = "\n\n".join(chunks)
    return corpus


# ── Tokenizer ─────────────────────────────────────────────────────────────────

def train_bpe_tokenizer(corpus: str, vocab_size: int, output_dir: Path):
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import ByteLevel
    from tokenizers.decoders import ByteLevel as BLDecoder
    from tokenizers.processors import ByteLevel as BLProcessor
    from transformers import PreTrainedTokenizerFast

    log(f"[INFO] Training BPE tokenizer (vocab_size={vocab_size}) on corpus...")

    tok = Tokenizer(BPE(unk_token="<|unk|>"))
    tok.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tok.decoder = BLDecoder()
    tok.post_processor = BLProcessor(trim_offsets=False)

    special_tokens = ["<|endoftext|>", "<|unk|>", "<|pad|>"]
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=special_tokens,
        show_progress=False,
    )

    # Stream the corpus in chunks to the trainer
    def corpus_iter():
        chunk_size = 10_000
        for i in range(0, len(corpus), chunk_size):
            yield corpus[i:i + chunk_size]

    tok.train_from_iterator(corpus_iter(), trainer=trainer)
    actual_vocab = tok.get_vocab_size()
    log(f"[INFO] Tokenizer trained. Actual vocab size: {actual_vocab}")

    tokenizer_dir = output_dir / "tokenizer"
    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    tok.save(str(tokenizer_dir / "tokenizer.json"))

    fast_tok = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        bos_token="<|endoftext|>",
        eos_token="<|endoftext|>",
        pad_token="<|pad|>",
        unk_token="<|unk|>",
    )
    fast_tok.save_pretrained(str(tokenizer_dir))
    log(f"[INFO] Tokenizer saved to {tokenizer_dir}")
    return fast_tok, actual_vocab


def load_gpt2_tokenizer(output_dir: Path):
    from transformers import GPT2TokenizerFast
    log("[INFO] Loading GPT-2 tokenizer (50257 vocab)")
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    tok_dir = output_dir / "tokenizer"
    tok_dir.mkdir(parents=True, exist_ok=True)
    tok.save_pretrained(str(tok_dir))
    return tok, tok.vocab_size


# ── Model ─────────────────────────────────────────────────────────────────────

def build_model(n_layer, n_head, n_embd, n_positions, vocab_size):
    from transformers import GPT2Config, GPT2LMHeadModel

    config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=n_positions,
        n_ctx=n_positions,
        n_embd=n_embd,
        n_layer=n_layer,
        n_head=n_head,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        bos_token_id=0,
        eos_token_id=0,
    )
    model = GPT2LMHeadModel(config)
    return model, config


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters())


# ── LR schedule ───────────────────────────────────────────────────────────────

def cosine_lr(step: int, warmup: int, total: int, lr_max: float, lr_min: float = 1e-5) -> float:
    if step < warmup:
        return lr_max * step / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * progress))


# ── Sliding-window dataset ────────────────────────────────────────────────────

class TokenDataset:
    def __init__(self, token_ids: list[int], seq_len: int):
        self.ids = token_ids
        self.seq_len = seq_len
        # How many full chunks fit
        self.n = max(0, (len(token_ids) - 1) // seq_len)

    def __len__(self):
        return self.n

    def get_batch(self, indices):
        import torch
        xs, ys = [], []
        for i in indices:
            start = i * self.seq_len
            xs.append(self.ids[start: start + self.seq_len])
            ys.append(self.ids[start + 1: start + self.seq_len + 1])
        return (
            torch.tensor(xs, dtype=torch.long),
            torch.tensor(ys, dtype=torch.long),
        )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job_id",     required=True)
    parser.add_argument("--dataset",    required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_name", default="my-llm")
    parser.add_argument("--n_layer",    type=int, default=6)
    parser.add_argument("--n_head",     type=int, default=6)
    parser.add_argument("--n_embd",     type=int, default=384)
    parser.add_argument("--n_positions", type=int, default=512)
    parser.add_argument("--vocab_size", type=int, default=8000)
    parser.add_argument("--train_tokenizer", action="store_true")
    parser.add_argument("--num_steps",  type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay",  type=float, default=0.1)
    parser.add_argument("--warmup_steps",  type=int, default=200)
    parser.add_argument("--gradient_accumulation", type=int, default=4)
    parser.add_argument("--save_every", type=int, default=500)
    parser.add_argument("--grad_clip",  type=float, default=1.0)
    parser.add_argument("--use_fp16",   action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log(f"[INFO] Job: {args.job_id}")
    log(f"[INFO] Output: {output_dir}")
    log(f"[INFO] Architecture: {args.n_layer}L / {args.n_head}H / {args.n_embd}D / ctx={args.n_positions}")

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"[INFO] Device: {device}")
    if device == "cuda":
        log(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        log(f"[INFO] VRAM: {vram:.1f} GB")
    use_fp16 = args.use_fp16 and device == "cuda"
    if use_fp16:
        log("[INFO] Mixed precision: FP16 enabled")

    # ── Load corpus ────────────────────────────────────────────────────────────
    log(f"[INFO] Loading corpus from {args.dataset}")
    corpus = load_corpus(args.dataset)
    log(f"[INFO] Corpus size: {len(corpus):,} characters")
    if len(corpus) < 1000:
        log("[WARN] Corpus is very small (<1000 chars). Model will overfit quickly.")

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    if args.train_tokenizer:
        tokenizer, actual_vocab = train_bpe_tokenizer(corpus, args.vocab_size, output_dir)
    else:
        tokenizer, actual_vocab = load_gpt2_tokenizer(output_dir)

    # ── Tokenize corpus ───────────────────────────────────────────────────────
    log("[INFO] Tokenizing corpus...")
    eos = tokenizer.eos_token or "<|endoftext|>"
    # Split into documents, tokenize each, join with EOS
    docs = corpus.split("\n\n")
    all_ids: list[int] = []
    eos_id = tokenizer.convert_tokens_to_ids(eos)
    if eos_id is None or eos_id < 0:
        eos_id = 0
    for doc in docs:
        doc = doc.strip()
        if not doc:
            continue
        ids = tokenizer.encode(doc, add_special_tokens=False)
        all_ids.extend(ids)
        all_ids.append(eos_id)

    total_tokens = len(all_ids)
    log(f"[INFO] Total tokens: {total_tokens:,}")
    if total_tokens < args.n_positions * 2:
        log(f"[WARN] Very few tokens ({total_tokens}) for sequence length {args.n_positions}. "
            f"Consider a larger dataset or shorter sequence length.")

    dataset = TokenDataset(all_ids, args.n_positions)
    n_chunks = len(dataset)
    log(f"[INFO] Training chunks: {n_chunks}")
    if n_chunks == 0:
        log("[ERROR] Not enough tokens to create even one training chunk. "
            "Use a larger dataset or reduce --n_positions.")
        return 1

    # ── Build model ───────────────────────────────────────────────────────────
    log("[INFO] Building model from scratch...")
    model, config = build_model(
        args.n_layer, args.n_head, args.n_embd, args.n_positions, actual_vocab
    )
    params = count_params(model)
    log(f"[INFO] Parameters: {params:,} ({params/1e6:.1f}M)")
    log(f"[INFO] Model type: {config.model_type}")

    model = model.to(device)
    if use_fp16:
        model = model.half()

    # ── Optimizer ─────────────────────────────────────────────────────────────
    # Separate weight decay: don't apply to biases and LN params
    decay_params = [p for n, p in model.named_parameters()
                    if p.requires_grad and p.dim() >= 2]
    no_decay_params = [p for n, p in model.named_parameters()
                       if p.requires_grad and p.dim() < 2]
    optimizer = torch.optim.AdamW([
        {"params": decay_params,    "weight_decay": args.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ], lr=args.learning_rate, betas=(0.9, 0.95), eps=1e-8)

    scaler = torch.cuda.amp.GradScaler() if use_fp16 else None

    # ── Training loop ─────────────────────────────────────────────────────────
    log(f"[INFO] Starting pre-training: {args.num_steps} steps, "
        f"batch={args.batch_size}, accum={args.gradient_accumulation}")
    log(f"[INFO] Effective batch = {args.batch_size * args.gradient_accumulation} "
        f"× {args.n_positions} tokens = "
        f"{args.batch_size * args.gradient_accumulation * args.n_positions:,} tokens/step")

    import random
    import numpy as np

    random.seed(42)
    torch.manual_seed(42)

    model.train()
    optimizer.zero_grad()

    loss_accum = 0.0
    t0 = time.time()
    chunk_indices = list(range(n_chunks))

    for step in range(1, args.num_steps + 1):
        # Update LR
        lr = cosine_lr(step, args.warmup_steps, args.num_steps, args.learning_rate)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # Accumulate gradients
        step_loss = 0.0
        for acc_step in range(args.gradient_accumulation):
            # Sample a random batch of chunks
            batch_idx = random.choices(chunk_indices, k=args.batch_size)
            x, y = dataset.get_batch(batch_idx)
            x, y = x.to(device), y.to(device)

            if use_fp16:
                with torch.cuda.amp.autocast():
                    out = model(x, labels=y)
                    loss = out.loss / args.gradient_accumulation
                scaler.scale(loss).backward()
            else:
                out = model(x, labels=y)
                loss = out.loss / args.gradient_accumulation
                loss.backward()

            step_loss += loss.item()

        # Gradient clip + optimizer step
        if use_fp16:
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        if use_fp16:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad()

        loss_accum += step_loss

        # Logging every 10 steps
        if step % 10 == 0:
            avg_loss = loss_accum / 10
            loss_accum = 0.0
            elapsed = time.time() - t0
            steps_per_sec = 10 / elapsed
            tokens_per_sec = int(steps_per_sec * args.batch_size * args.gradient_accumulation * args.n_positions)
            remaining = (args.num_steps - step) / steps_per_sec
            perplexity = math.exp(min(avg_loss, 20))
            log(f"[PRETRAIN] step={step}/{args.num_steps} loss={avg_loss:.4f} "
                f"ppl={perplexity:.1f} lr={lr:.2e} "
                f"tok/s={tokens_per_sec:,} eta={_fmt_time(remaining)}")
            t0 = time.time()

        # Checkpoint
        if step % args.save_every == 0:
            ckpt_dir = output_dir / f"checkpoint-{step}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            _save_model(model, config, tokenizer, ckpt_dir, use_fp16)
            log(f"[INFO] Checkpoint saved: {ckpt_dir}")

    # ── Save final model ───────────────────────────────────────────────────────
    log(f"[INFO] Saving final model to {output_dir}")
    _save_model(model, config, tokenizer, output_dir, use_fp16)

    # Write a README
    readme = (
        f"# {args.model_name}\n\n"
        f"Pre-trained from scratch with LLM Trainer for Ollama.\n\n"
        f"## Architecture\n"
        f"- Layers: {args.n_layer}\n"
        f"- Heads: {args.n_head}\n"
        f"- Hidden size: {args.n_embd}\n"
        f"- Context length: {args.n_positions}\n"
        f"- Vocab size: {actual_vocab}\n"
        f"- Parameters: {params:,} ({params/1e6:.1f}M)\n\n"
        f"## Training\n"
        f"- Steps: {args.num_steps}\n"
        f"- Tokens: {total_tokens:,}\n"
        f"- Learning rate: {args.learning_rate}\n\n"
        f"## Usage\n"
        f"Convert to GGUF using the Convert tab, then create a Modelfile and import into Ollama.\n"
    )
    (output_dir / "README.md").write_text(readme)

    log(f"[SUCCESS] Pre-training complete! {params/1e6:.1f}M parameter model saved to {output_dir}")
    log(f"[INFO] Next steps: Convert tab → convert to GGUF → Modelfile tab → import to Ollama")
    return 0


def _save_model(model, config, tokenizer, save_dir: Path, was_fp16: bool):
    """Save model in float32 HuggingFace format for maximum compatibility."""
    import torch
    save_dir.mkdir(parents=True, exist_ok=True)

    # Convert back to fp32 for saving (llama.cpp expects fp32 or fp16 in the gguf)
    if was_fp16:
        model_to_save = model.float()
    else:
        model_to_save = model

    model_to_save.save_pretrained(str(save_dir))
    config.save_pretrained(str(save_dir))

    # Save tokenizer — check tokenizer subdir first, then root
    tok_src = save_dir.parent / "tokenizer" if (save_dir.parent / "tokenizer").exists() else None
    if tok_src:
        import shutil
        for f in tok_src.iterdir():
            shutil.copy2(str(f), str(save_dir / f.name))
    else:
        try:
            tokenizer.save_pretrained(str(save_dir))
        except Exception:
            pass

    if was_fp16:
        # Put model back to fp16 for continued training
        model.half()


def _fmt_time(seconds: float) -> str:
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds//60}m{seconds%60:02d}s"
    return f"{seconds//3600}h{(seconds%3600)//60:02d}m"


if __name__ == "__main__":
    sys.exit(main())
