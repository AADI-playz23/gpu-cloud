"""
NovaDev Cloud — Sessions API
Handles session start, stop, status, and Kaggle kernel submission.
"""
from fastapi import APIRouter, Request, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional

from docker.pool import SlotType
from kaggle.bridge import kaggle_bridge

router = APIRouter()


class StartSessionRequest(BaseModel):
    user_id: str
    tier: str = "free"   # "free" | "paid"


class RunCodeRequest(BaseModel):
    user_id: str
    session_id: str
    notebook_json: str              # Full .ipynb as string
    dataset_sources: Optional[list] = None


@router.post("/start")
async def start_session(req: StartSessionRequest, request: Request):
    """Enqueue user for a GPU session."""
    queue_manager = request.app.state.queue_manager
    result = await queue_manager.enqueue(req.user_id, req.tier)
    return result


@router.post("/run")
async def run_code(req: RunCodeRequest, request: Request):
    """Submit notebook to Kaggle GPU kernel and stream output."""
    docker_pool = request.app.state.docker_pool

    # Verify session is active
    slot = None
    for s in docker_pool.slots.values():
        if s.session_id == req.session_id and s.user_id == req.user_id:
            slot = s
            break

    if not slot:
        raise HTTPException(404, "No active session found")

    run = await kaggle_bridge.submit_kernel(
        user_id=req.user_id,
        session_id=req.session_id,
        notebook_source=req.notebook_json,
        dataset_sources=req.dataset_sources,
    )
    return {"run_id": run.run_id, "status": "submitted"}


@router.delete("/{session_id}")
async def end_session(session_id: str, request: Request):
    """Manually terminate a session and release the container slot."""
    docker_pool = request.app.state.docker_pool

    for slot in docker_pool.slots.values():
        if slot.session_id == session_id:
            await docker_pool.release_slot(slot.slot_id)
            # Notify queue manager
            queue_manager = request.app.state.queue_manager
            await queue_manager.on_slot_released(slot.slot_type)
            return {"message": "Session terminated", "slot_id": slot.slot_id}

    raise HTTPException(404, "Session not found")


@router.get("/stats")
async def pool_stats(request: Request):
    """Return live Docker pool + queue stats."""
    docker_pool   = request.app.state.docker_pool
    queue_manager = request.app.state.queue_manager
    pool  = docker_pool.stats()
    queue = await queue_manager.queue_stats()
    return {**pool, **queue}


@router.get("/quota")
async def kaggle_quota():
    """Return remaining Kaggle GPU quota."""
    return await kaggle_bridge.remaining_gpu_quota()
