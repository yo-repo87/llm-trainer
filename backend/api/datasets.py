import uuid
import json
import csv
import aiosqlite
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import aiofiles

router = APIRouter()
DATASETS_DIR = Path("data/datasets")
DB_PATH = Path("data/llm_trainer.db")


@router.post("/upload")
async def upload_dataset(file: UploadFile = File(...)):
    if not file.filename.endswith((".jsonl", ".csv", ".txt", ".json")):
        raise HTTPException(400, "Supported formats: .jsonl, .csv, .txt, .json")

    ds_id = str(uuid.uuid4())[:8]
    ext = Path(file.filename).suffix
    save_path = DATASETS_DIR / f"{ds_id}{ext}"

    content = await file.read()
    async with aiofiles.open(save_path, "wb") as f:
        await f.write(content)

    rows, fmt = _analyze_file(save_path, ext)

    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT INTO datasets (id, name, path, format, size, rows) VALUES (?, ?, ?, ?, ?, ?)",
            (ds_id, file.filename, str(save_path), fmt, save_path.stat().st_size, rows)
        )
        await db.commit()

    return {"id": ds_id, "name": file.filename, "rows": rows, "format": fmt, "size": save_path.stat().st_size}


def _analyze_file(path: Path, ext: str) -> tuple[int, str]:
    try:
        if ext == ".jsonl":
            with open(path) as f:
                lines = [l for l in f if l.strip()]
            return len(lines), "jsonl"
        elif ext == ".json":
            with open(path) as f:
                data = json.load(f)
            return len(data) if isinstance(data, list) else 1, "json"
        elif ext == ".csv":
            with open(path) as f:
                reader = csv.reader(f)
                rows = sum(1 for _ in reader) - 1
            return max(rows, 0), "csv"
        else:
            with open(path) as f:
                lines = [l for l in f if l.strip()]
            return len(lines), "txt"
    except Exception:
        return 0, ext.lstrip(".")


@router.get("/")
async def list_datasets():
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute("SELECT * FROM datasets ORDER BY created_at DESC") as cur:
            rows = await cur.fetchall()
    return [dict(r) for r in rows]


@router.get("/{ds_id}/preview")
async def preview_dataset(ds_id: str, limit: int = 5):
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute("SELECT * FROM datasets WHERE id=?", (ds_id,)) as cur:
            row = await cur.fetchone()
    if not row:
        raise HTTPException(404, "Dataset not found")

    ds = dict(row)
    path = Path(ds["path"])
    if not path.exists():
        raise HTTPException(404, "File not found on disk")

    samples = []
    try:
        if ds["format"] == "jsonl":
            with open(path) as f:
                for i, line in enumerate(f):
                    if i >= limit:
                        break
                    if line.strip():
                        samples.append(json.loads(line))
        elif ds["format"] == "json":
            with open(path) as f:
                data = json.load(f)
            samples = data[:limit] if isinstance(data, list) else [data]
        elif ds["format"] == "csv":
            with open(path) as f:
                reader = csv.DictReader(f)
                for i, row in enumerate(reader):
                    if i >= limit:
                        break
                    samples.append(dict(row))
        else:
            with open(path) as f:
                lines = [l.rstrip() for l in f if l.strip()]
            samples = [{"text": l} for l in lines[:limit]]
    except Exception as e:
        raise HTTPException(500, f"Error reading file: {e}")

    return {"id": ds_id, "samples": samples, "format": ds["format"]}


@router.delete("/{ds_id}")
async def delete_dataset(ds_id: str):
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute("SELECT path FROM datasets WHERE id=?", (ds_id,)) as cur:
            row = await cur.fetchone()
        if not row:
            raise HTTPException(404, "Dataset not found")
        path = Path(row["path"])
        if path.exists():
            path.unlink()
        await db.execute("DELETE FROM datasets WHERE id=?", (ds_id,))
        await db.commit()
    return {"status": "deleted"}


@router.post("/{ds_id}/convert")
async def convert_dataset(ds_id: str, target_format: str = "jsonl"):
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute("SELECT * FROM datasets WHERE id=?", (ds_id,)) as cur:
            row = await cur.fetchone()
    if not row:
        raise HTTPException(404, "Dataset not found")

    ds = dict(row)
    src_path = Path(ds["path"])
    out_path = DATASETS_DIR / f"{ds_id}_converted.jsonl"

    try:
        samples = []
        if ds["format"] == "csv":
            with open(src_path) as f:
                reader = csv.DictReader(f)
                for r in reader:
                    if "text" in r:
                        samples.append({"text": r["text"]})
                    elif "instruction" in r and "output" in r:
                        samples.append({
                            "instruction": r.get("instruction", ""),
                            "input": r.get("input", ""),
                            "output": r.get("output", "")
                        })
                    else:
                        samples.append({"text": " ".join(r.values())})
        elif ds["format"] == "txt":
            with open(src_path) as f:
                for line in f:
                    if line.strip():
                        samples.append({"text": line.strip()})
        else:
            raise HTTPException(400, "Only CSV and TXT conversion to JSONL is supported")

        with open(out_path, "w") as f:
            for s in samples:
                f.write(json.dumps(s) + "\n")

        new_id = str(uuid.uuid4())[:8]
        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute(
                "INSERT INTO datasets (id, name, path, format, size, rows) VALUES (?, ?, ?, ?, ?, ?)",
                (new_id, f"{ds['name']}_converted.jsonl", str(out_path), "jsonl",
                 out_path.stat().st_size, len(samples))
            )
            await db.commit()

        return {"id": new_id, "rows": len(samples), "format": "jsonl"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))
