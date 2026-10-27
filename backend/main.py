import asyncio
import json
import os
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import psutil

from database import init_db
from api import datasets, training, models, ollama, distill, pretrain

app = FastAPI(title="LLM Trainer for Ollama", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ConnectionManager:
    def __init__(self):
        self.active: dict[str, list[WebSocket]] = {}

    async def connect(self, ws: WebSocket, job_id: str):
        await ws.accept()
        self.active.setdefault(job_id, []).append(ws)

    def disconnect(self, ws: WebSocket, job_id: str):
        if job_id in self.active:
            try:
                self.active[job_id].remove(ws)
            except ValueError:
                pass

    async def broadcast(self, message: str, job_id: str):
        dead = []
        for ws in self.active.get(job_id, []):
            try:
                await ws.send_text(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws, job_id)


manager = ConnectionManager()

# Inject WebSocket manager into API modules
training.set_ws_manager(manager)
models.set_ws_manager(manager)
ollama.set_ws_manager(manager)
distill.set_ws_manager(manager)
pretrain.set_ws_manager(manager)

app.include_router(datasets.router, prefix="/api/datasets", tags=["datasets"])
app.include_router(training.router, prefix="/api/training", tags=["training"])
app.include_router(models.router, prefix="/api/models", tags=["models"])
app.include_router(ollama.router, prefix="/api/ollama", tags=["ollama"])
app.include_router(distill.router, prefix="/api/distill", tags=["distill"])
app.include_router(pretrain.router, prefix="/api/pretrain", tags=["pretrain"])


@app.websocket("/ws/{job_id}")
async def ws_endpoint(websocket: WebSocket, job_id: str):
    await manager.connect(websocket, job_id)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket, job_id)


@app.get("/api/system")
async def system_info():
    import torch
    import subprocess
    gpu_info = None
    try:
        if torch.cuda.is_available():
            util_pct = None
            try:
                out = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                    timeout=2
                ).decode().strip()
                util_pct = int(out.split('\n')[0].strip())
            except Exception:
                pass
            gpu_info = {
                "name": torch.cuda.get_device_name(0),
                "memory_total_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2),
                "memory_used_gb": round(torch.cuda.memory_allocated(0) / 1e9, 2),
                "utilization_pct": util_pct,
            }
    except Exception:
        pass

    mem = psutil.virtual_memory()
    disk = psutil.disk_usage("data")
    return {
        "cpu_percent": psutil.cpu_percent(interval=0.1),
        "cpu_count": psutil.cpu_count(),
        "ram_total_gb": round(mem.total / 1e9, 2),
        "ram_used_gb": round(mem.used / 1e9, 2),
        "ram_percent": mem.percent,
        "disk_total_gb": round(disk.total / 1e9, 2),
        "disk_used_gb": round(disk.used / 1e9, 2),
        "disk_percent": disk.percent,
        "gpu": gpu_info,
        "python_version": __import__("sys").version.split()[0],
    }


@app.on_event("startup")
async def startup():
    await init_db()
    for d in ["data/datasets", "data/models", "data/output", "data/modelfiles"]:
        Path(d).mkdir(parents=True, exist_ok=True)


frontend_dir = Path(__file__).parent.parent / "frontend"
if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")


@app.get("/")
async def serve_index():
    idx = frontend_dir / "index.html"
    if idx.exists():
        return FileResponse(str(idx))
    return JSONResponse({"status": "LLM Trainer API running", "docs": "/docs"})


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port, reload=False)
