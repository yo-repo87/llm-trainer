"""
Teacher-student knowledge distillation API.

Two modes:
  1. generate-dataset  — teacher (Ollama model) answers prompts → saved as JSONL dataset
  2. start             — true KD: both HuggingFace models loaded, student trained on
                         soft logit targets (KL divergence) + hard CE loss
"""
import asyncio
import json
import sys
import uuid
from pathlib import Path

import aiosqlite
import requests
from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel

from database import create_job, get_job, list_jobs, update_job

router = APIRouter()
DB_PATH = Path("data/llm_trainer.db")
DATASETS_DIR = Path("data/datasets")
_log_buffers: dict[str, list[str]] = {}
_ws_manager = None


def set_ws_manager(manager):
    global _ws_manager
    _ws_manager = manager


# ── Pydantic models ───────────────────────────────────────────────────────────

class GenerateDatasetConfig(BaseModel):
    teacher_ollama_model: str
    prompts_source: str = "dataset"      # "dataset" | "inline"
    dataset_id: str = ""                  # used when prompts_source=="dataset"
    inline_prompts: list[str] = []        # used when prompts_source=="inline"
    system_prompt: str = ""
    output_name: str = "teacher_generated"
    temperature: float = 0.7
    max_tokens: int = 512
    prompt_field: str = "instruction"     # field in JSONL to use as prompt


class DistillConfig(BaseModel):
    teacher_model: str                    # HuggingFace model id
    student_model: str                    # HuggingFace model id
    dataset_id: str
    output_name: str = "distilled-student"
    # KD hyper-params
    temperature: float = 4.0             # softmax temperature for soft targets
    alpha: float = 0.5                   # weight: alpha*CE + (1-alpha)*KD
    # LoRA for student
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    # Training
    num_epochs: int = 3
    batch_size: int = 2
    learning_rate: float = 2e-4
    max_seq_length: int = 512
    gradient_accumulation: int = 4
    warmup_steps: int = 10
    use_4bit_student: bool = True
    use_4bit_teacher: bool = True
    instruction_template: str = "alpaca"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ollama_base() -> str:
    import os
    return os.environ.get("OLLAMA_HOST", "http://host.docker.internal:11434")


async def _log(job_id: str, msg: str):
    _log_buffers.setdefault(job_id, []).append(msg)
    if _ws_manager:
        await _ws_manager.broadcast(json.dumps({"type": "log", "message": msg}), job_id)


async def _set_status(job_id: str, status: str, **extra):
    if _ws_manager:
        await _ws_manager.broadcast(
            json.dumps({"type": "status", "status": status, **extra}), job_id
        )


# ── Mode 1: Generate dataset via Ollama teacher ───────────────────────────────

async def _run_generate_dataset(job_id: str, config: dict):
    await update_job(job_id, "running")
    ollama_url = _ollama_base()
    teacher = config["teacher_ollama_model"]
    await _log(job_id, f"[INFO] Teacher model (Ollama): {teacher}")

    # Collect prompts
    prompts: list[str] = []
    if config["prompts_source"] == "inline":
        prompts = [p for p in config["inline_prompts"] if p.strip()]
        await _log(job_id, f"[INFO] Using {len(prompts)} inline prompts")
    else:
        ds_id = config["dataset_id"]
        async with aiosqlite.connect(DB_PATH) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute("SELECT path, format FROM datasets WHERE id=?", (ds_id,)) as cur:
                row = await cur.fetchone()
        if not row:
            await update_job(job_id, "failed", error="Dataset not found")
            await _log(job_id, "[ERROR] Dataset not found")
            return
        app_root = Path(__file__).resolve().parent.parent.parent
        path = (app_root / row["path"]).resolve()
        field = config.get("prompt_field", "instruction")
        try:
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    if field in obj:
                        prompts.append(obj[field])
                    elif "text" in obj:
                        prompts.append(obj["text"])
                    else:
                        prompts.append(str(list(obj.values())[0]))
        except Exception as e:
            await update_job(job_id, "failed", error=str(e))
            await _log(job_id, f"[ERROR] Reading dataset: {e}")
            return
        await _log(job_id, f"[INFO] Loaded {len(prompts)} prompts from dataset")

    if not prompts:
        await update_job(job_id, "failed", error="No prompts found")
        await _log(job_id, "[ERROR] No prompts to generate from")
        return

    # Query teacher for each prompt
    out_name = config.get("output_name", "teacher_generated")
    _app_root = Path(__file__).resolve().parent.parent.parent
    out_path = (_app_root / "data" / "datasets" / f"{out_name}_{job_id[:6]}.jsonl").resolve()
    system = config.get("system_prompt", "")
    temp = config.get("temperature", 0.7)
    max_tok = config.get("max_tokens", 512)

    records = []
    failed = 0
    for i, prompt in enumerate(prompts):
        await _log(job_id, f"[INFO] Generating {i+1}/{len(prompts)}: {prompt[:60]}...")
        payload = {
            "model": teacher,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temp, "num_predict": max_tok},
        }
        if system:
            payload["system"] = system
        try:
            resp = requests.post(f"{ollama_url}/api/generate", json=payload, timeout=120)
            resp.raise_for_status()
            response_text = resp.json().get("response", "").strip()
            records.append({
                "instruction": prompt,
                "input": "",
                "output": response_text,
                "teacher": teacher,
            })
            await _log(job_id, f"[TEACHER] → {response_text[:80]}...")
        except Exception as e:
            await _log(job_id, f"[WARN] Failed prompt {i+1}: {e}")
            failed += 1

    if not records:
        await update_job(job_id, "failed", error="All generations failed")
        await _log(job_id, "[ERROR] No successful generations")
        return

    with open(out_path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    ds_id = str(uuid.uuid4())[:8]
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT INTO datasets (id, name, path, format, size, rows) VALUES (?,?,?,?,?,?)",
            (ds_id, out_path.name, str(out_path), "jsonl",
             out_path.stat().st_size, len(records))
        )
        await db.commit()

    await update_job(job_id, "completed", result={"dataset_id": ds_id, "rows": len(records), "failed": failed})
    await _log(job_id, f"[SUCCESS] Generated {len(records)} samples ({failed} failed). Dataset ID: {ds_id}")
    await _set_status(job_id, "completed", dataset_id=ds_id, rows=len(records))


@router.post("/generate-dataset")
async def generate_dataset(config: GenerateDatasetConfig, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())[:12]
    await create_job(job_id, "distill_generate", config.model_dump())
    _log_buffers[job_id] = []
    background_tasks.add_task(_run_generate_dataset, job_id, config.model_dump())
    return {"job_id": job_id, "status": "pending"}


# ── Mode 2: True knowledge distillation (HuggingFace models) ─────────────────

async def _run_distillation(job_id: str, config: dict):
    await update_job(job_id, "running")
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute("SELECT path FROM datasets WHERE id=?", (config["dataset_id"],)) as cur:
            row = await cur.fetchone()
    if not row:
        await update_job(job_id, "failed", error="Dataset not found")
        await _log(job_id, "[ERROR] Dataset not found")
        return

    app_root = Path(__file__).resolve().parent.parent.parent
    dataset_path = str((app_root / row["path"]).resolve())
    output_dir = app_root / "data" / "output" / config["output_name"]

    cmd = [
        sys.executable, "scripts/distill.py",
        "--job_id", job_id,
        "--dataset", dataset_path,
        "--teacher_model", config["teacher_model"],
        "--student_model", config["student_model"],
        "--output_dir", str(output_dir),
        "--temperature", str(config["temperature"]),
        "--alpha", str(config["alpha"]),
        "--lora_r", str(config["lora_r"]),
        "--lora_alpha", str(config["lora_alpha"]),
        "--lora_dropout", str(config["lora_dropout"]),
        "--num_epochs", str(config["num_epochs"]),
        "--batch_size", str(config["batch_size"]),
        "--learning_rate", str(config["learning_rate"]),
        "--max_seq_length", str(config["max_seq_length"]),
        "--gradient_accumulation", str(config["gradient_accumulation"]),
        "--warmup_steps", str(config["warmup_steps"]),
        "--instruction_template", config["instruction_template"],
    ]
    if config.get("use_4bit_student"):
        cmd.append("--use_4bit_student")
    if config.get("use_4bit_teacher"):
        cmd.append("--use_4bit_teacher")

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
async def start_distillation(config: DistillConfig, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())[:12]
    await create_job(job_id, "distillation", config.model_dump())
    _log_buffers[job_id] = []
    background_tasks.add_task(_run_distillation, job_id, config.model_dump())
    return {"job_id": job_id, "status": "pending"}


# ── Shared job endpoints ───────────────────────────────────────────────────────

@router.get("/jobs")
async def list_distill_jobs():
    gen = await list_jobs("distill_generate")
    kd = await list_jobs("distillation")
    combined = gen + kd
    combined.sort(key=lambda j: j["created_at"], reverse=True)
    return combined


@router.get("/jobs/{job_id}")
async def get_distill_job(job_id: str):
    job = await get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return job


@router.get("/jobs/{job_id}/logs")
async def get_distill_logs(job_id: str):
    return {"logs": _log_buffers.get(job_id, [])}
