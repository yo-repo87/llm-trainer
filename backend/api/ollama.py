import json
import asyncio
from pathlib import Path
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
import requests

router = APIRouter()
_ws_manager = None
_log_buffers: dict[str, list[str]] = {}


def set_ws_manager(manager):
    global _ws_manager
    _ws_manager = manager


def _ollama_base() -> str:
    import os
    return os.environ.get("OLLAMA_HOST", "http://host.docker.internal:11434")


class CreateModelRequest(BaseModel):
    name: str
    modelfile_content: str


class GenerateRequest(BaseModel):
    model: str
    prompt: str
    system: str = ""
    temperature: float = 0.7
    max_tokens: int = 512


class PullModelRequest(BaseModel):
    model_name: str


@router.get("/status")
async def ollama_status():
    try:
        r = requests.get(f"{_ollama_base()}/api/tags", timeout=5)
        if r.status_code == 200:
            return {"connected": True, "url": _ollama_base()}
    except Exception:
        pass
    return {"connected": False, "url": _ollama_base()}


@router.get("/models")
async def list_ollama_models():
    try:
        r = requests.get(f"{_ollama_base()}/api/tags", timeout=10)
        r.raise_for_status()
        data = r.json()
        return {"models": data.get("models", [])}
    except requests.ConnectionError:
        raise HTTPException(503, "Cannot connect to Ollama. Is it running?")
    except Exception as e:
        raise HTTPException(500, str(e))


@router.post("/create")
async def create_ollama_model(req: CreateModelRequest):
    """Create/import a model in Ollama using a Modelfile via the ollama CLI."""
    import subprocess, tempfile, os, shutil
    ollama_bin = shutil.which("ollama")
    if not ollama_bin:
        raise HTTPException(500, "ollama CLI not found on host")
    with tempfile.NamedTemporaryFile(mode="w", suffix=".Modelfile", delete=False) as f:
        f.write(req.modelfile_content)
        mf_path = f.name
    try:
        env = os.environ.copy()
        env["OLLAMA_HOST"] = _ollama_base()
        result = subprocess.run(
            [ollama_bin, "create", req.name, "-f", mf_path],
            capture_output=True, text=True, timeout=300, env=env,
        )
        if result.returncode != 0:
            raise HTTPException(500, result.stderr or result.stdout or "ollama create failed")
        return {"status": "created", "messages": [result.stdout.strip()]}
    except subprocess.TimeoutExpired:
        raise HTTPException(504, "ollama create timed out")
    finally:
        os.unlink(mf_path)


@router.post("/pull")
async def pull_model(req: PullModelRequest):
    """Pull a model from Ollama registry."""
    try:
        payload = {"name": req.model_name}
        r = requests.post(f"{_ollama_base()}/api/pull", json=payload, timeout=600, stream=True)
        r.raise_for_status()
        messages = []
        for line in r.iter_lines():
            if line:
                try:
                    obj = json.loads(line)
                    messages.append(obj.get("status", ""))
                except Exception:
                    pass
        return {"status": "pulled", "messages": messages}
    except requests.ConnectionError:
        raise HTTPException(503, "Cannot connect to Ollama")
    except Exception as e:
        raise HTTPException(500, str(e))


@router.delete("/models/{model_name:path}")
async def delete_ollama_model(model_name: str):
    try:
        r = requests.delete(f"{_ollama_base()}/api/delete", json={"name": model_name}, timeout=30)
        r.raise_for_status()
        return {"status": "deleted"}
    except requests.ConnectionError:
        raise HTTPException(503, "Cannot connect to Ollama")
    except Exception as e:
        raise HTTPException(500, str(e))


@router.post("/generate")
async def generate_text(req: GenerateRequest):
    """Run inference on an Ollama model."""
    try:
        payload = {
            "model": req.model,
            "prompt": req.prompt,
            "stream": False,
            "options": {"temperature": req.temperature, "num_predict": req.max_tokens}
        }
        if req.system:
            payload["system"] = req.system
        r = requests.post(f"{_ollama_base()}/api/generate", json=payload, timeout=120)
        r.raise_for_status()
        data = r.json()
        return {
            "response": data.get("response", ""),
            "eval_count": data.get("eval_count", 0),
            "eval_duration": data.get("eval_duration", 0),
        }
    except requests.ConnectionError:
        raise HTTPException(503, "Cannot connect to Ollama")
    except Exception as e:
        raise HTTPException(500, str(e))


@router.get("/modelfile-template")
async def get_modelfile_template():
    """Return template Modelfiles for common use cases."""
    templates = {
        "basic": 'FROM {model_path}\n\nSYSTEM """\nYou are a helpful AI assistant.\n"""\n\nPARAMETER temperature 0.7\nPARAMETER num_ctx 4096\n',
        "chat": 'FROM {model_path}\n\nSYSTEM """\nYou are a friendly and knowledgeable assistant. Answer questions clearly and concisely.\n"""\n\nPARAMETER temperature 0.8\nPARAMETER top_p 0.9\nPARAMETER num_ctx 4096\nPARAMETER stop "<|im_end|>"\n',
        "code": 'FROM {model_path}\n\nSYSTEM """\nYou are an expert software engineer. Provide clean, efficient, well-commented code. Explain your solutions clearly.\n"""\n\nPARAMETER temperature 0.2\nPARAMETER top_k 40\nPARAMETER num_ctx 8192\n',
        "custom_gguf": 'FROM /path/to/your/model.gguf\n\nSYSTEM """\nYou are a helpful assistant trained on custom data.\n"""\n\nPARAMETER temperature 0.7\nPARAMETER num_ctx 4096\n',
    }
    return templates
