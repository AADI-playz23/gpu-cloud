"""
NovaDev Cloud — Main API Server
FastAPI + WebSocket backend for GPU notebook platform
"""
import asyncio
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from api.sessions import router as sessions_router
from api.queue import router as queue_router
from api.llm import router as llm_router
from api.admin import router as admin_router
from api.auth import router as auth_router
from core.config import settings
from core.websocket_manager import ws_manager
from queue.manager import QueueManager
from docker.pool import DockerPool

queue_manager: QueueManager = None
docker_pool: DockerPool = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global queue_manager, docker_pool
    print("[startup] Initialising Docker pool...")
    docker_pool = DockerPool(
        free_slots=settings.FREE_CONTAINER_SLOTS,
        paid_slots=settings.PAID_CONTAINER_SLOTS,
    )
    await docker_pool.initialise()

    print("[startup] Starting queue manager...")
    queue_manager = QueueManager(docker_pool=docker_pool)
    asyncio.create_task(queue_manager.run())

    print("[startup] All systems ready.")
    app.state.queue_manager = queue_manager
    app.state.docker_pool = docker_pool
    yield

    print("[shutdown] Draining containers...")
    await docker_pool.shutdown_all()


app = FastAPI(
    title="NovaDev Cloud API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router,     prefix="/api/auth",     tags=["auth"])
app.include_router(sessions_router, prefix="/api/sessions", tags=["sessions"])
app.include_router(queue_router,    prefix="/api/queue",    tags=["queue"])
app.include_router(llm_router,      prefix="/api/llm",      tags=["llm"])
app.include_router(admin_router,    prefix="/api/admin",    tags=["admin"])


@app.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str):
    await ws_manager.connect(websocket, user_id)
    try:
        while True:
            data = await websocket.receive_text()
            await ws_manager.handle_message(user_id, data)
    except WebSocketDisconnect:
        ws_manager.disconnect(user_id)


@app.get("/health")
async def health():
    return {"status": "ok", "version": "1.0.0"}
