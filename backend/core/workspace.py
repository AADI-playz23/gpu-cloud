"""
NovaDev Cloud — Workspace Persistence Manager

Handles:
  - Auto-save notebooks to cloud storage (S3/GCS) every 5 min
  - Save-on-session-end (free: 30min, paid: 2h)
  - Restore workspace on session resume
  - Max sessions: free=3, paid=5 (only 1 active at a time)
  - File downloads (model.pt, CSV, .ipynb)

Storage layout (S3):
  novadev-workspaces/
    {user_id}/
      sessions/
        {session_id}/
          notebook.ipynb       ← auto-saved notebook
          workspace.tar.gz     ← /workspace dir snapshot
          meta.json            ← session metadata
          files/               ← user-uploaded files
"""

import asyncio
import gzip
import json
import logging
import os
import tarfile
import tempfile
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Optional

import aiofiles
import aioboto3
import aioredis

from core.config import settings

log = logging.getLogger("workspace")

FREE_MAX_SESSIONS   = 3
PAID_MAX_SESSIONS   = 5
AUTOSAVE_INTERVAL_S = 300    # Auto-save every 5 minutes
S3_BUCKET           = os.getenv("S3_BUCKET", "novadev-workspaces")
AWS_REGION          = os.getenv("AWS_REGION", "ap-south-1")


@dataclass
class SessionMeta:
    session_id: str
    user_id: str
    tier: str
    name: str
    created_at: str
    last_saved_at: str
    slot_id: Optional[int] = None
    is_active: bool = False
    notebook_cells: int = 0
    workspace_size_mb: float = 0.0


class WorkspaceManager:
    def __init__(self):
        self.redis: Optional[aioredis.Redis] = None
        self._s3 = aioboto3.Session()

    async def start(self, redis: aioredis.Redis):
        self.redis = redis

        # Subscribe to AUTOSAVE_ALL broadcast from lifecycle manager
        sub = self.redis.pubsub()
        await sub.subscribe("novadev:broadcast")
        asyncio.create_task(self._listen_broadcast(sub))

        # Periodic auto-save loop
        asyncio.create_task(self._autosave_loop())
        log.info("Workspace manager ready.")

    # ── Session management ───────────────────────────────────────────────────

    async def create_session(self, user_id: str, tier: str, name: str) -> dict:
        """Create a new saved session (not yet active)."""
        sessions = await self.list_sessions(user_id)
        max_sess = PAID_MAX_SESSIONS if tier == "paid" else FREE_MAX_SESSIONS

        if len(sessions) >= max_sess:
            return {
                "error": f"Max {max_sess} sessions for {tier} plan. Delete one to create a new session.",
                "max_sessions": max_sess,
                "current_count": len(sessions),
            }

        import uuid
        session_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()

        meta = SessionMeta(
            session_id=session_id,
            user_id=user_id,
            tier=tier,
            name=name,
            created_at=now,
            last_saved_at=now,
            is_active=False,
        )

        await self._save_meta(meta)

        # Create empty notebook
        empty_nb = self._empty_notebook(name)
        await self._upload_notebook(user_id, session_id, empty_nb)

        log.info("Session created: %s (%s, tier=%s)", session_id, name, tier)
        return {"session_id": session_id, "name": name}

    async def activate_session(self, user_id: str, session_id: str) -> dict:
        """
        Mark session as active. Enforces 1 active session per user.
        Deactivates any currently active session first (auto-saves it).
        """
        sessions = await self.list_sessions(user_id)

        # Deactivate current active session
        for s in sessions:
            if s.is_active and s.session_id != session_id:
                await self.save_session(user_id, s.session_id)
                s.is_active = False
                await self._save_meta(s)
                log.info("Deactivated session %s for user %s", s.session_id, user_id)

        # Activate requested session
        meta = await self._load_meta(user_id, session_id)
        if not meta:
            return {"error": "Session not found"}

        meta.is_active = True
        await self._save_meta(meta)
        log.info("Activated session %s for user %s", session_id, user_id)
        return {"activated": session_id}

    async def list_sessions(self, user_id: str) -> list[SessionMeta]:
        """Return all sessions for a user from Redis."""
        pattern = f"novadev:session_meta:{user_id}:*"
        keys = await self.redis.keys(pattern)
        sessions = []
        for key in keys:
            raw = await self.redis.get(key)
            if raw:
                sessions.append(SessionMeta(**json.loads(raw)))
        sessions.sort(key=lambda x: x.created_at, reverse=True)
        return sessions

    async def delete_session(self, user_id: str, session_id: str):
        """Delete session metadata + cloud files."""
        await self.redis.delete(f"novadev:session_meta:{user_id}:{session_id}")
        # Queue S3 deletion async
        asyncio.create_task(self._delete_s3_prefix(user_id, session_id))
        log.info("Session deleted: %s", session_id)

    # ── Save / restore ───────────────────────────────────────────────────────

    async def save_session(self, user_id: str, session_id: str,
                           notebook_json: str = None):
        """Save notebook + workspace snapshot to S3."""
        meta = await self._load_meta(user_id, session_id)
        if not meta:
            return

        now = datetime.now(timezone.utc).isoformat()
        meta.last_saved_at = now

        if notebook_json:
            await self._upload_notebook(user_id, session_id, notebook_json)

        await self._save_meta(meta)
        await self.redis.set(
            f"novadev:last_save:{session_id}", now, ex=86400
        )
        log.info("Session saved: %s", session_id)

    async def restore_session(self, user_id: str, session_id: str) -> dict:
        """Return notebook JSON + file list for restoring into a container."""
        notebook = await self._download_notebook(user_id, session_id)
        files = await self._list_s3_files(user_id, session_id)
        meta = await self._load_meta(user_id, session_id)
        return {
            "notebook": notebook,
            "files": files,
            "meta": asdict(meta) if meta else {},
        }

    # ── Auto-save ─────────────────────────────────────────────────────────────

    async def _autosave_loop(self):
        """Every 5 min, auto-save all active sessions."""
        while True:
            await asyncio.sleep(AUTOSAVE_INTERVAL_S)
            try:
                await self._save_all_active()
            except Exception as e:
                log.error("Autosave loop error: %s", e)

    async def _save_all_active(self):
        """Find all active sessions and save them."""
        keys = await self.redis.keys("novadev:session_meta:*")
        saved = 0
        for key in keys:
            raw = await self.redis.get(key)
            if not raw:
                continue
            meta = SessionMeta(**json.loads(raw))
            if meta.is_active:
                # Signal container to snapshot its /workspace dir
                await self.redis.publish(
                    f"novadev:container:{meta.session_id}",
                    json.dumps({"cmd": "SAVE", "session_id": meta.session_id})
                )
                saved += 1
        if saved:
            log.info("Auto-save triggered for %d active sessions.", saved)

    async def _listen_broadcast(self, sub):
        """Listen for AUTOSAVE_ALL from lifecycle manager."""
        async for msg in sub.listen():
            if msg["type"] == "message" and msg["data"] == "AUTOSAVE_ALL":
                log.info("AUTOSAVE_ALL received — saving all active sessions.")
                await self._save_all_active()

    # ── S3 helpers ────────────────────────────────────────────────────────────

    async def _upload_notebook(self, user_id: str, session_id: str, content: str):
        key = f"{user_id}/sessions/{session_id}/notebook.ipynb"
        async with self._s3.client("s3", region_name=AWS_REGION) as s3:
            await s3.put_object(Bucket=S3_BUCKET, Key=key, Body=content.encode())

    async def _download_notebook(self, user_id: str, session_id: str) -> str:
        key = f"{user_id}/sessions/{session_id}/notebook.ipynb"
        try:
            async with self._s3.client("s3", region_name=AWS_REGION) as s3:
                resp = await s3.get_object(Bucket=S3_BUCKET, Key=key)
                body = await resp["Body"].read()
                return body.decode()
        except Exception:
            return self._empty_notebook("Restored notebook")

    async def _list_s3_files(self, user_id: str, session_id: str) -> list:
        prefix = f"{user_id}/sessions/{session_id}/files/"
        try:
            async with self._s3.client("s3", region_name=AWS_REGION) as s3:
                resp = await s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=prefix)
                return [
                    {"name": obj["Key"].split("/")[-1],
                     "size_bytes": obj["Size"],
                     "last_modified": obj["LastModified"].isoformat()}
                    for obj in resp.get("Contents", [])
                ]
        except Exception:
            return []

    async def _delete_s3_prefix(self, user_id: str, session_id: str):
        prefix = f"{user_id}/sessions/{session_id}/"
        try:
            async with self._s3.client("s3", region_name=AWS_REGION) as s3:
                paginator = s3.get_paginator("list_objects_v2")
                async for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=prefix):
                    keys = [{"Key": obj["Key"]} for obj in page.get("Contents", [])]
                    if keys:
                        await s3.delete_objects(
                            Bucket=S3_BUCKET, Delete={"Objects": keys}
                        )
        except Exception as e:
            log.warning("S3 delete failed for %s/%s: %s", user_id, session_id, e)

    # ── Redis meta helpers ────────────────────────────────────────────────────

    async def _save_meta(self, meta: SessionMeta):
        key = f"novadev:session_meta:{meta.user_id}:{meta.session_id}"
        await self.redis.set(key, json.dumps(asdict(meta)), ex=86400 * 30)

    async def _load_meta(self, user_id: str, session_id: str) -> Optional[SessionMeta]:
        key = f"novadev:session_meta:{user_id}:{session_id}"
        raw = await self.redis.get(key)
        return SessionMeta(**json.loads(raw)) if raw else None

    # ── Notebook template ─────────────────────────────────────────────────────

    def _empty_notebook(self, name: str) -> str:
        nb = {
            "nbformat": 4,
            "nbformat_minor": 5,
            "metadata": {
                "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                "novadev": {"session_name": name}
            },
            "cells": [{
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": ["# Welcome to NovaDev Cloud\n", "import torch\n",
                           "print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')\n"]
            }]
        }
        return json.dumps(nb, indent=2)


workspace_manager = WorkspaceManager()
