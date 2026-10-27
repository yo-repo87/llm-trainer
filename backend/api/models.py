import uuid
import asyncio
import json
import sys
from pathlib import Path
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
import aiosqlite
from database import create_job, get_job, list_jobs, update_job

router = APIRouter()
DB_PATH = Path("data/llm_trainer.db")
_log_buffers: dict[str, list[str]] = {}
_ws_manager = None


def set_ws_manager(manager):
    global _ws_manager
    _ws_manager = manager


class ConvertConfig(BaseModel):
    source_path: str
    model_name: str
    quantization: str = "q4_k_m"
    output_name: str = ""


class ModelfileConfig(BaseModel):
    name: str
    content: str


async def _stream_log(job_id: str, message: str):
    if job_id not in _log_buffers:
        _log_buffers[job_id] = []
    _log_buffers[job_id].append(message)
    if _ws_manager:
        await _ws_manager.broadcast(json.dumps({"type": "log", "message": message}), job_id)


async def _run_conversion(job_id: str, config: dict):
    await update_job(job_id, "running")
    source = Path(config["source_path"])
    quant = config["quantization"].lower()
    out_name = config.get("output_name") or config["model_name"].replace("/", "_")
    output_dir = Path("data/models")
    output_gguf = output_dir / f"{out_name}_{quant}.gguf"

    await _stream_log(job_id, f"[INFO] Converting {source} to GGUF ({quant})")

    llama_cpp_dir = Path("/app/llama.cpp")
    convert_script = llama_cpp_dir / "convert_hf_to_gguf.py"
    if not convert_script.exists():
        convert_script = llama_cpp_dir / "convert.py"

    if not convert_script.exists():
        await update_job(job_id, "failed", error="llama.cpp not found")
        await _stream_log(job_id, "[ERROR] llama.cpp convert script not found. Ensure it is installed at /app/llama.cpp")
        return

    # If source is a LoRA adapter (no config.json), merge with base model first
    convert_source = source
    merged_dir = None
    is_lora = (source / "adapter_config.json").exists() and not (source / "config.json").exists()
    if is_lora:
        base_model = config.get("model_name", "").strip()
        if not base_model or Path(base_model) == source:
            await update_job(job_id, "failed", error="Base model required for LoRA adapter")
            await _stream_log(job_id, "[ERROR] Source is a LoRA adapter — set the 'Base Model' field to the HuggingFace model name (e.g. TinyLlama/TinyLlama-1.1B-Chat-v1.0)")
            return

        merged_dir = Path(f"/tmp/merged_{job_id}")
        await _stream_log(job_id, f"[INFO] LoRA adapter detected — merging with base model: {base_model}")
        merge_script = Path(__file__).resolve().parent.parent / "scripts" / "merge.py"
        merge_cmd = [
            sys.executable, str(merge_script),
            "--adapter", str(source),
            "--base_model", base_model,
            "--output_dir", str(merged_dir),
        ]
        proc_merge = await asyncio.create_subprocess_exec(
            *merge_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        async for line in proc_merge.stdout:
            msg = line.decode().rstrip()
            if msg:
                await _stream_log(job_id, msg)
        await proc_merge.wait()
        if proc_merge.returncode != 0:
            await update_job(job_id, "failed", error="LoRA merge failed")
            await _stream_log(job_id, "[ERROR] Failed to merge LoRA adapter with base model")
            if _ws_manager:
                await _ws_manager.broadcast(json.dumps({"type": "status", "status": "failed"}), job_id)
            return
        convert_source = merged_dir
        await _stream_log(job_id, "[INFO] Merge complete — starting GGUF conversion...")

    # Step 1: Convert to F16 GGUF
    f16_path = output_dir / f"{out_name}_f16.gguf"
    cmd_convert = [sys.executable, str(convert_script), str(convert_source), "--outfile", str(f16_path), "--outtype", "f16"]

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd_convert,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT
        )
        async for line in proc.stdout:
            msg = line.decode().rstrip()
            if msg:
                await _stream_log(job_id, msg)
        await proc.wait()

        if proc.returncode != 0:
            await update_job(job_id, "failed", error="Conversion to F16 failed")
            await _stream_log(job_id, "[ERROR] F16 conversion failed")
            return

        await _stream_log(job_id, "[INFO] F16 conversion complete, quantizing...")

        # Step 2: Quantize
        quantize_bin = llama_cpp_dir / "build" / "bin" / "llama-quantize"
        if not quantize_bin.exists():
            quantize_bin = llama_cpp_dir / "quantize"

        if quantize_bin.exists():
            import os
            quant_env = os.environ.copy()
            quant_env["LD_LIBRARY_PATH"] = "/lib/x86_64-linux-gnu:" + quant_env.get("LD_LIBRARY_PATH", "")
            cmd_quant = [str(quantize_bin), str(f16_path), str(output_gguf), quant.upper()]
            proc2 = await asyncio.create_subprocess_exec(
                *cmd_quant,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                env=quant_env,
            )
            async for line in proc2.stdout:
                msg = line.decode().rstrip()
                if msg:
                    await _stream_log(job_id, msg)
            await proc2.wait()

            if proc2.returncode == 0:
                if f16_path.exists():
                    f16_path.unlink()
            else:
                await _stream_log(job_id, "[WARN] Quantization failed — keeping F16 GGUF")
                output_gguf = f16_path
        else:
            output_gguf = f16_path
            await _stream_log(job_id, "[WARN] quantize binary not found, keeping F16 GGUF")

        size = output_gguf.stat().st_size if output_gguf.exists() else 0
        model_id = str(uuid.uuid4())[:8]
        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute(
                "INSERT INTO gguf_models (id, name, path, base_model, quantization, size) VALUES (?,?,?,?,?,?)",
                (model_id, out_name, str(output_gguf), config.get("model_name", ""), quant, size)
            )
            await db.commit()

        await update_job(job_id, "completed", result={"model_id": model_id, "path": str(output_gguf)})
        await _stream_log(job_id, f"[SUCCESS] GGUF model saved: {output_gguf}")
        if _ws_manager:
            await _ws_manager.broadcast(json.dumps({"type": "status", "status": "completed"}), job_id)

    except Exception as e:
        await update_job(job_id, "failed", error=str(e))
        await _stream_log(job_id, f"[ERROR] {e}")
        if _ws_manager:
            await _ws_manager.broadcast(json.dumps({"type": "status", "status": "failed"}), job_id)
    finally:
        if merged_dir and merged_dir.exists():
            import shutil
            shutil.rmtree(merged_dir, ignore_errors=True)


@router.post("/convert")
async def convert_model(config: ConvertConfig, background_tasks: BackgroundTasks):
    source = Path(config.source_path)
    if not source.exists():
        raise HTTPException(400, f"Source path does not exist: {config.source_path}")
    job_id = str(uuid.uuid4())[:12]
    await create_job(job_id, "conversion", config.model_dump())
    _log_buffers[job_id] = []
    background_tasks.add_task(_run_conversion, job_id, config.model_dump())
    return {"job_id": job_id, "status": "pending"}


@router.get("/gguf")
async def list_gguf_models():
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute("SELECT * FROM gguf_models ORDER BY created_at DESC") as cur:
            rows = await cur.fetchall()
    return [dict(r) for r in rows]


@router.get("/gguf/{model_id}")
async def get_gguf_model(model_id: str):
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute("SELECT * FROM gguf_models WHERE id=?", (model_id,)) as cur:
            row = await cur.fetchone()
    if not row:
        raise HTTPException(404, "Model not found")
    return dict(row)


@router.delete("/gguf/{model_id}")
async def delete_gguf_model(model_id: str):
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute("SELECT path FROM gguf_models WHERE id=?", (model_id,)) as cur:
            row = await cur.fetchone()
        if not row:
            raise HTTPException(404, "Model not found")
        p = Path(row["path"])
        if p.exists():
            p.unlink()
        await db.execute("DELETE FROM gguf_models WHERE id=?", (model_id,))
        await db.commit()
    return {"status": "deleted"}


@router.get("/conversions")
async def list_conversion_jobs():
    return await list_jobs("conversion")


@router.get("/conversions/{job_id}/logs")
async def get_conversion_logs(job_id: str):
    return {"logs": _log_buffers.get(job_id, [])}


# Modelfile endpoints
@router.get("/modelfiles")
async def list_modelfiles():
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute("SELECT * FROM modelfiles ORDER BY created_at DESC") as cur:
            rows = await cur.fetchall()
    return [dict(r) for r in rows]


@router.post("/modelfiles")
async def save_modelfile(config: ModelfileConfig):
    mf_id = str(uuid.uuid4())[:8]
    mf_path = Path("data/modelfiles") / f"{mf_id}.Modelfile"
    mf_path.write_text(config.content)
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT INTO modelfiles (id, name, content) VALUES (?, ?, ?)",
            (mf_id, config.name, config.content)
        )
        await db.commit()
    return {"id": mf_id, "name": config.name, "path": str(mf_path)}


@router.put("/modelfiles/{mf_id}")
async def update_modelfile(mf_id: str, config: ModelfileConfig):
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute("SELECT id FROM modelfiles WHERE id=?", (mf_id,)) as cur:
            row = await cur.fetchone()
        if not row:
            raise HTTPException(404, "Modelfile not found")
        await db.execute(
            "UPDATE modelfiles SET name=?, content=?, updated_at=CURRENT_TIMESTAMP WHERE id=?",
            (config.name, config.content, mf_id)
        )
        await db.commit()
    mf_path = Path("data/modelfiles") / f"{mf_id}.Modelfile"
    mf_path.write_text(config.content)
    return {"id": mf_id, "status": "updated"}


@router.delete("/modelfiles/{mf_id}")
async def delete_modelfile(mf_id: str):
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute("SELECT id FROM modelfiles WHERE id=?", (mf_id,)) as cur:
            row = await cur.fetchone()
        if not row:
            raise HTTPException(404, "Modelfile not found")
        await db.execute("DELETE FROM modelfiles WHERE id=?", (mf_id,))
        await db.commit()
    mf_path = Path("data/modelfiles") / f"{mf_id}.Modelfile"
    if mf_path.exists():
        mf_path.unlink()
    return {"status": "deleted"}


@router.get("/scan-output")
async def scan_output_directory():
    """Scan data/output for fine-tuned model adapter directories."""
    output_dir = Path("data/output")
    models = []
    if output_dir.exists():
        for d in sorted(output_dir.iterdir()):
            if d.is_dir() and (d / "adapter_config.json").exists():
                models.append({
                    "name": d.name,
                    "path": str(d),
                    "type": "lora_adapter"
                })
            elif d.is_file() and d.suffix == ".gguf":
                models.append({
                    "name": d.stem,
                    "path": str(d),
                    "type": "gguf"
                })
    return models
