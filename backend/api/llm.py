"""
NovaDev Cloud — LLM Installer API
One-click model download into the user's container via Ollama / HuggingFace Hub.

Categories: Coding · Fast · Deep Reasoning
"""
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
import httpx, asyncio, logging

log = logging.getLogger("llm_installer")
router = APIRouter()

# ── Model registry ────────────────────────────────────────────────────────────
# Each model maps to its Ollama tag (preferred) or HuggingFace repo.

MODEL_REGISTRY = {
    # ── Coding ──────────────────────────────────────────────────────────────
    "deepseek-coder-6.7b": {
        "name":     "DeepSeek Coder 6.7B",
        "category": "coding",
        "desc":     "Best for code generation, debugging, completion",
        "size_gb":  3.8,
        "ollama_tag": "deepseek-coder:6.7b",
        "min_vram_gb": 4.0,
        "tier":     "both",    # free and paid
    },
    "codellama-7b": {
        "name":     "CodeLlama 7B",
        "category": "coding",
        "desc":     "Meta · multilingual code assistant",
        "size_gb":  4.1,
        "ollama_tag": "codellama:7b",
        "min_vram_gb": 4.0,
        "tier":     "both",
    },
    "starcoder2-7b": {
        "name":     "StarCoder2 7B",
        "category": "coding",
        "desc":     "HuggingFace · 600+ programming languages",
        "size_gb":  4.4,
        "ollama_tag": "starcoder2:7b",
        "min_vram_gb": 4.0,
        "tier":     "paid",    # paid only — VRAM requirement > free slot
    },

    # ── Fast ────────────────────────────────────────────────────────────────
    "phi3-mini-3.8b": {
        "name":     "Phi-3 Mini 3.8B",
        "category": "fast",
        "desc":     "Microsoft · ultra-fast, high accuracy for size",
        "size_gb":  2.2,
        "ollama_tag": "phi3:mini",
        "min_vram_gb": 2.5,
        "tier":     "both",
    },
    "gemma-2b": {
        "name":     "Gemma 2B",
        "category": "fast",
        "desc":     "Google · efficient, instruction-tuned",
        "size_gb":  1.4,
        "ollama_tag": "gemma:2b",
        "min_vram_gb": 1.5,
        "tier":     "both",
    },
    "tinyllama-1.1b": {
        "name":     "TinyLlama 1.1B",
        "category": "fast",
        "desc":     "Smallest, fastest — great for quick inference",
        "size_gb":  0.6,
        "ollama_tag": "tinyllama",
        "min_vram_gb": 0.7,
        "tier":     "both",
    },

    # ── Deep reasoning ──────────────────────────────────────────────────────
    "llama3-8b": {
        "name":     "Llama 3 8B",
        "category": "reasoning",
        "desc":     "Meta · best general reasoning in its class",
        "size_gb":  4.9,
        "ollama_tag": "llama3:8b",
        "min_vram_gb": 5.0,
        "tier":     "paid",
    },
    "mistral-7b": {
        "name":     "Mistral 7B v0.3",
        "category": "reasoning",
        "desc":     "Strong instruction following, long context",
        "size_gb":  4.1,
        "ollama_tag": "mistral:7b",
        "min_vram_gb": 4.0,
        "tier":     "paid",
    },
    "qwen2-7b": {
        "name":     "Qwen2 7B",
        "category": "reasoning",
        "desc":     "Alibaba · multilingual, math, reasoning",
        "size_gb":  4.5,
        "ollama_tag": "qwen2:7b",
        "min_vram_gb": 4.5,
        "tier":     "paid",
    },
}


class InstallRequest(BaseModel):
    model_id: str
    container_port: int   # The user's assigned container port
    user_tier: str        # "free" or "paid"


class InstallStatus(BaseModel):
    model_id: str
    status: str           # queued | downloading | ready | error
    progress_pct: float = 0.0
    message: str = ""


# In-memory install tracker (use Redis in production)
_install_jobs: dict[str, InstallStatus] = {}


@router.get("/models")
async def list_models(tier: str = "free"):
    """Return full model catalogue filtered by tier."""
    result = {}
    for model_id, m in MODEL_REGISTRY.items():
        if m["tier"] in ("both", tier):
            result[model_id] = {
                **m,
                "model_id": model_id,
                "available_free": m["tier"] in ("both", "free"),
                "available_paid": m["tier"] in ("both", "paid"),
            }
    # Group by category
    grouped = {"coding": [], "fast": [], "reasoning": []}
    for model_id, m in result.items():
        grouped[m["category"]].append(m)
    return grouped


@router.post("/install")
async def install_model(req: InstallRequest, bg: BackgroundTasks):
    """Trigger one-click model install inside the user's container via Ollama."""
    model = MODEL_REGISTRY.get(req.model_id)
    if not model:
        raise HTTPException(404, "Model not found")

    # Tier gate
    if req.user_tier == "free" and model["tier"] == "paid":
        raise HTTPException(403, "This model requires a paid plan (VRAM too large for free slot)")

    status = InstallStatus(model_id=req.model_id, status="queued")
    _install_jobs[req.model_id] = status

    bg.add_task(_do_install, req, model, status)
    return {"message": "Install started", "model_id": req.model_id}


async def _do_install(req: InstallRequest, model: dict, status: InstallStatus):
    """
    Calls Ollama API inside the user's Docker container to pull the model.
    Ollama runs on port 11434 inside each container, mapped externally.
    """
    ollama_port = req.container_port + 1000   # e.g. container 8800 → ollama 9800
    ollama_url = f"http://localhost:{ollama_port}/api/pull"

    status.status = "downloading"
    log.info("Pulling %s into container port %d", model["ollama_tag"], ollama_port)

    try:
        async with httpx.AsyncClient(timeout=600) as client:
            async with client.stream(
                "POST",
                ollama_url,
                json={"name": model["ollama_tag"], "stream": True},
            ) as resp:
                total = model["size_gb"] * 1024 ** 3
                downloaded = 0
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    import json as _json
                    try:
                        chunk = _json.loads(line)
                        if "completed" in chunk and "total" in chunk:
                            downloaded = chunk["completed"]
                            total = chunk["total"] or total
                            pct = min(round(downloaded / total * 100, 1), 99.9)
                            status.progress_pct = pct
                    except Exception:
                        pass

        status.status = "ready"
        status.progress_pct = 100.0
        log.info("Model %s ready.", model["name"])

    except Exception as e:
        status.status = "error"
        status.message = str(e)
        log.error("Install failed for %s: %s", req.model_id, e)


@router.get("/install/{model_id}/status")
async def install_status(model_id: str):
    job = _install_jobs.get(model_id)
    if not job:
        return {"status": "not_started"}
    return job
