"""
Pre-training from scratch API.

Flow:
  1. User selects a dataset and model architecture preset (or custom)
  2. Optionally trains a BPE tokenizer on the corpus
  3. Tokenizes corpus into sliding-window chunks
  4. Trains a GPT-2-style model from random weights
  5. Saves in HuggingFace format → ready for Convert tab → GGUF → Ollama
"""
import asyncio
import json
import sys
import uuid
from pathlib import Path

import aiosqlite
from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel

from database import create_job, get_job, list_jobs, update_job

router = APIRouter()
DB_PATH = Path("data/llm_trainer.db")
_log_buffers: dict[str, list[str]] = {}
_ws_manager = None

PRESETS = {
    "nano": {
        "n_layer": 4, "n_head": 4, "n_embd": 256, "n_positions": 512,
        "description": "~7M params — trains on CPU, minutes per epoch",
    },
    "tiny": {
        "n_layer": 6, "n_head": 6, "n_embd": 384, "n_positions": 512,
        "description": "~17M params — CPU feasible, GPU recommended",
    },
    "small": {
        "n_layer": 12, "n_head": 12, "n_embd": 768, "n_positions": 1024,
        "description": "~125M params — requires GPU (4GB+ VRAM)",
    },
    "medium": {
        "n_layer": 24, "n_head": 16, "n_embd": 1024, "n_positions": 1024,
        "description": "~350M params — requires GPU (12GB+ VRAM)",
    },
}


def set_ws_manager(manager):
    global _ws_manager
    _ws_manager = manager


class PretrainConfig(BaseModel):
    dataset_id: str
    model_name: str = "my-llm"
    preset: str = "tiny"          # nano / tiny / small / medium / custom
    # Architecture (used when preset=="custom", or override any preset)
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    n_positions: int = 512
    vocab_size: int = 8000
    # Tokenizer
    train_tokenizer: bool = True   # False = use GPT-2 tokenizer
    # Training
    num_steps: int = 5000
    batch_size: int = 8
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    warmup_steps: int = 200
    gradient_accumulation: int = 4
    save_every: int = 500
    grad_clip: float = 1.0
    use_fp16: bool = True


async def _log(job_id: str, msg: str):
    _log_buffers.setdefault(job_id, []).append(msg)
    if _ws_manager:
        await _ws_manager.broadcast(json.dumps({"type": "log", "message": msg}), job_id)


async def _set_status(job_id: str, status: str, **extra):
    if _ws_manager:
        await _ws_manager.broadcast(
            json.dumps({"type": "status", "status": status, **extra}), job_id
        )


async def _run_pretrain(job_id: str, config: dict):
    await update_job(job_id, "running")

    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute("SELECT path FROM datasets WHERE id=?", (config["dataset_id"],)) as cur:
            row = await cur.fetchone()

    if not row:
        await update_job(job_id, "failed", error="Dataset not found")
        await _log(job_id, "[ERROR] Dataset not found")
        return

    # Resolve to absolute paths — subprocess CWD is /app/backend, data lives in /app/data
    app_root = Path(__file__).resolve().parent.parent.parent
    dataset_path = str((app_root / row["path"]).resolve())
    output_dir = app_root / "data" / "output" / config["model_name"]

    # Resolve architecture from preset
    if config["preset"] in PRESETS and config["preset"] != "custom":
        p = PRESETS[config["preset"]]
        config["n_layer"]     = p["n_layer"]
        config["n_head"]      = p["n_head"]
        config["n_embd"]      = p["n_embd"]
        config["n_positions"] = p["n_positions"]

    cmd = [
        sys.executable, "scripts/pretrain.py",
        "--job_id",    job_id,
        "--dataset",   dataset_path,
        "--output_dir", str(output_dir),
        "--model_name", config["model_name"],
        "--n_layer",   str(config["n_layer"]),
        "--n_head",    str(config["n_head"]),
        "--n_embd",    str(config["n_embd"]),
        "--n_positions", str(config["n_positions"]),
        "--vocab_size", str(config["vocab_size"]),
        "--num_steps", str(config["num_steps"]),
        "--batch_size", str(config["batch_size"]),
        "--learning_rate", str(config["learning_rate"]),
        "--weight_decay",  str(config["weight_decay"]),
        "--warmup_steps",  str(config["warmup_steps"]),
        "--gradient_accumulation", str(config["gradient_accumulation"]),
        "--save_every", str(config["save_every"]),
        "--grad_clip",  str(config["grad_clip"]),
    ]
    if config.get("train_tokenizer"):
        cmd.append("--train_tokenizer")
    if config.get("use_fp16"):
        cmd.append("--use_fp16")

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=Path(__file__).parent.parent,
        )
        async for line in proc.stdout:
            msg = line.decode().rstrip()
            if msg:
                await _log(job_id, msg)
        await proc.wait()

        if proc.returncode == 0:
            await update_job(job_id, "completed", result={"output_dir": str(output_dir)})
            await _set_status(job_id, "completed", output_dir=str(output_dir))
        else:
            await update_job(job_id, "failed", error=f"Exit code {proc.returncode}")
            await _set_status(job_id, "failed")
    except Exception as e:
        await update_job(job_id, "failed", error=str(e))
        await _log(job_id, f"[ERROR] {e}")
        await _set_status(job_id, "failed")


@router.post("/start")
async def start_pretrain(config: PretrainConfig, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())[:12]
    await create_job(job_id, "pretrain", config.model_dump())
    _log_buffers[job_id] = []
    background_tasks.add_task(_run_pretrain, job_id, config.model_dump())
    return {"job_id": job_id, "status": "pending"}


@router.get("/presets")
async def get_presets():
    return PRESETS


@router.get("/jobs")
async def list_pretrain_jobs():
    return await list_jobs("pretrain")


@router.get("/jobs/{job_id}")
async def get_pretrain_job(job_id: str):
    job = await get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return job


@router.get("/jobs/{job_id}/logs")
async def get_pretrain_logs(job_id: str):
    return {"logs": _log_buffers.get(job_id, [])}


@router.get("/checkpoints")
async def list_checkpoints():
    """Scan data/output for pre-trained model directories."""
    output_dir = Path("data/output")
    checkpoints = []
    if output_dir.exists():
        for d in sorted(output_dir.iterdir()):
            cfg_file = d / "config.json"
            if d.is_dir() and cfg_file.exists():
                try:
                    cfg = json.loads(cfg_file.read_text())
                    arch = cfg.get("model_type", "unknown")
                    n_layer = cfg.get("n_layer", cfg.get("num_hidden_layers", "?"))
                    n_embd  = cfg.get("n_embd",  cfg.get("hidden_size", "?"))
                    vocab   = cfg.get("vocab_size", "?")
                    # Count parameter estimate
                    if isinstance(n_layer, int) and isinstance(n_embd, int) and isinstance(vocab, int):
                        params = int(n_layer * 12 * n_embd**2 + 2 * vocab * n_embd)
                    else:
                        params = None
                    # Check for tokenizer
                    has_tokenizer = (d / "tokenizer.json").exists() or (d / "tokenizer" / "tokenizer.json").exists()
                    # Check for sub-checkpoints
                    ckpts = sorted([c.name for c in d.iterdir() if c.is_dir() and c.name.startswith("checkpoint-")])
                    checkpoints.append({
                        "name": d.name,
                        "path": str(d),
                        "arch": arch,
                        "n_layer": n_layer,
                        "n_embd": n_embd,
                        "vocab_size": vocab,
                        "params": params,
                        "has_tokenizer": has_tokenizer,
                        "checkpoints": ckpts,
                    })
                except Exception:
                    pass
    return checkpoints
