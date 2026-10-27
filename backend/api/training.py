import uuid
import json
import asyncio
from pathlib import Path
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from database import create_job, get_job, list_jobs, update_job

router = APIRouter()

# Global log buffer per job and WebSocket manager reference
_log_buffers: dict[str, list[str]] = {}
_ws_manager = None


def set_ws_manager(manager):
    global _ws_manager
    _ws_manager = manager


class TrainingConfig(BaseModel):
    dataset_id: str
    base_model: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    output_name: str = "my-finetune"
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    num_epochs: int = 3
    batch_size: int = 4
    learning_rate: float = 2e-4
    max_seq_length: int = 512
    warmup_steps: int = 10
    save_steps: int = 50
    gradient_accumulation: int = 4
    use_4bit: bool = True
    instruction_template: str = "alpaca"


async def _stream_log(job_id: str, message: str):
    if job_id not in _log_buffers:
        _log_buffers[job_id] = []
    _log_buffers[job_id].append(message)
    if _ws_manager:
        await _ws_manager.broadcast(json.dumps({"type": "log", "message": message}), job_id)


async def _run_training(job_id: str, config: dict):
    import sys
    import subprocess
    from pathlib import Path

    await update_job(job_id, "running")
    await _stream_log(job_id, f"[INFO] Starting training job {job_id}")
    await _stream_log(job_id, f"[INFO] Base model: {config['base_model']}")
    await _stream_log(job_id, f"[INFO] Dataset ID: {config['dataset_id']}")

    # Locate the dataset
    import aiosqlite
    db_path = Path("data/llm_trainer.db")
    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute("SELECT path FROM datasets WHERE id=?", (config["dataset_id"],)) as cur:
            row = await cur.fetchone()

    if not row:
        await update_job(job_id, "failed", error="Dataset not found")
        await _stream_log(job_id, "[ERROR] Dataset not found")
        return

    app_root = Path(__file__).resolve().parent.parent.parent
    dataset_path = str((app_root / row["path"]).resolve())
    output_dir = app_root / "data" / "output" / config["output_name"]

    script_path = Path("scripts/train.py")
    cmd = [
        sys.executable, str(script_path),
        "--job_id", job_id,
        "--dataset", dataset_path,
        "--base_model", config["base_model"],
        "--output_dir", str(output_dir),
        "--lora_r", str(config["lora_r"]),
        "--lora_alpha", str(config["lora_alpha"]),
        "--lora_dropout", str(config["lora_dropout"]),
        "--num_epochs", str(config["num_epochs"]),
        "--batch_size", str(config["batch_size"]),
        "--learning_rate", str(config["learning_rate"]),
        "--max_seq_length", str(config["max_seq_length"]),
        "--warmup_steps", str(config["warmup_steps"]),
        "--gradient_accumulation", str(config["gradient_accumulation"]),
        "--instruction_template", config["instruction_template"],
    ]
    if config.get("use_4bit"):
        cmd.append("--use_4bit")

    try:
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=Path(__file__).parent.parent
        )

        async for line in process.stdout:
            msg = line.decode().rstrip()
            if msg:
                await _stream_log(job_id, msg)

        await process.wait()

        if process.returncode == 0:
            await update_job(job_id, "completed", result={"output_dir": str(output_dir)})
            await _stream_log(job_id, f"[SUCCESS] Training complete! Model saved to {output_dir}")
            if _ws_manager:
                await _ws_manager.broadcast(
                    json.dumps({"type": "status", "status": "completed", "output_dir": str(output_dir)}),
                    job_id
                )
        else:
            await update_job(job_id, "failed", error=f"Process exited with code {process.returncode}")
            await _stream_log(job_id, f"[ERROR] Training failed with exit code {process.returncode}")
            if _ws_manager:
                await _ws_manager.broadcast(json.dumps({"type": "status", "status": "failed"}), job_id)

    except Exception as e:
        await update_job(job_id, "failed", error=str(e))
        await _stream_log(job_id, f"[ERROR] {str(e)}")
        if _ws_manager:
            await _ws_manager.broadcast(json.dumps({"type": "status", "status": "failed"}), job_id)


@router.post("/start")
async def start_training(config: TrainingConfig, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())[:12]
    await create_job(job_id, "training", config.model_dump())
    _log_buffers[job_id] = []
    background_tasks.add_task(_run_training, job_id, config.model_dump())
    return {"job_id": job_id, "status": "pending"}


@router.get("/jobs")
async def get_training_jobs():
    return await list_jobs("training")


@router.get("/jobs/{job_id}")
async def get_training_job(job_id: str):
    job = await get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return job


@router.get("/jobs/{job_id}/logs")
async def get_job_logs(job_id: str):
    return {"logs": _log_buffers.get(job_id, [])}


@router.delete("/jobs/{job_id}")
async def cancel_job(job_id: str):
    job = await get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    if job["status"] in ("completed", "failed"):
        raise HTTPException(400, "Job already finished")
    await update_job(job_id, "cancelled")
    return {"status": "cancelled"}
