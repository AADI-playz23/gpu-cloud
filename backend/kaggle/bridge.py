"""
NovaDev Cloud — Kaggle API Bridge
Submits notebook kernels to Kaggle's T4×2 GPU backend and streams output.

Flow:
  1. User submits code from notebook editor
  2. Bridge pushes kernel to Kaggle API (POST /api/v1/kernels)
  3. Polls kernel status until complete
  4. Streams stdout/stderr back via WebSocket

Kaggle gives 30h/week of free T4 GPU time.
T4×2 = 32 GB VRAM — we logically partition this into 40 slots.
"""
import asyncio
import json
import logging
import time
from dataclasses import dataclass
from typing import AsyncGenerator, Optional

import httpx

from core.config import settings
from core.websocket_manager import ws_manager

log = logging.getLogger("kaggle_bridge")

KAGGLE_BASE = "https://www.kaggle.com/api/v1"
HEADERS = {"Content-Type": "application/json"}


@dataclass
class KernelRun:
    run_id: str
    kernel_slug: str
    user_id: str
    session_id: str
    status: str = "pending"   # pending | running | complete | error


class KaggleBridge:
    def __init__(self):
        self.auth = (settings.KAGGLE_USERNAME, settings.KAGGLE_KEY)
        self._active_runs: dict[str, KernelRun] = {}

    # ── Submit ───────────────────────────────────────────────────────────────

    async def submit_kernel(
        self,
        user_id: str,
        session_id: str,
        notebook_source: str,
        dataset_sources: list[str] | None = None,
    ) -> KernelRun:
        """
        Push a notebook to Kaggle and start a GPU kernel.
        notebook_source: the .ipynb JSON as a string
        """
        slug = f"novadev-{session_id[:8]}"
        payload = {
            "id":             f"{settings.KAGGLE_USERNAME}/{slug}",
            "title":          f"NovaDev Run {session_id[:8]}",
            "code_file":      notebook_source,
            "language":       "python",
            "kernel_type":    "notebook",
            "is_private":     True,
            "enable_gpu":     True,
            "enable_tpu":     False,
            "enable_internet": False,
            "dataset_sources": dataset_sources or [],
            "accelerator":    settings.KAGGLE_ACCELERATOR,   # T4×2
        }

        async with httpx.AsyncClient(auth=self.auth, timeout=30) as client:
            resp = await client.post(
                f"{KAGGLE_BASE}/kernels",
                json=payload,
                headers=HEADERS,
            )
            resp.raise_for_status()
            data = resp.json()
            log.info("Kernel submitted: %s", slug)

        run = KernelRun(
            run_id=data.get("ref", slug),
            kernel_slug=slug,
            user_id=user_id,
            session_id=session_id,
            status="running",
        )
        self._active_runs[session_id] = run

        # Start background polling
        asyncio.create_task(self._poll_and_stream(run))
        return run

    # ── Poll & stream ─────────────────────────────────────────────────────────

    async def _poll_and_stream(self, run: KernelRun):
        """Poll Kaggle until kernel completes; stream output to user via WS."""
        slug = f"{settings.KAGGLE_USERNAME}/{run.kernel_slug}"
        seen_lines = 0

        async with httpx.AsyncClient(auth=self.auth, timeout=60) as client:
            while True:
                await asyncio.sleep(settings.KAGGLE_POLL_INTERVAL_SEC)

                # Fetch kernel status
                try:
                    status_resp = await client.get(
                        f"{KAGGLE_BASE}/kernels/{slug}",
                        headers=HEADERS,
                    )
                    status_resp.raise_for_status()
                    status_data = status_resp.json()
                    run.status = status_data.get("status", "running").lower()
                except Exception as e:
                    log.warning("Status poll failed for %s: %s", slug, e)
                    continue

                # Fetch output log
                try:
                    log_resp = await client.get(
                        f"{KAGGLE_BASE}/kernels/{slug}/output",
                        headers=HEADERS,
                    )
                    log_resp.raise_for_status()
                    log_data = log_resp.json()
                    stdout = log_data.get("log", "")
                    lines = stdout.split("\n")

                    # Stream only new lines
                    new_lines = lines[seen_lines:]
                    if new_lines:
                        seen_lines = len(lines)
                        await ws_manager.send(run.user_id, {
                            "event": "kernel_output",
                            "session_id": run.session_id,
                            "lines": new_lines,
                            "status": run.status,
                        })
                except Exception as e:
                    log.warning("Log fetch failed for %s: %s", slug, e)

                # Terminal states
                if run.status in ("complete", "error", "cancelacknowledged"):
                    await ws_manager.send(run.user_id, {
                        "event": "kernel_done",
                        "session_id": run.session_id,
                        "status": run.status,
                    })
                    log.info("Kernel %s finished: %s", slug, run.status)
                    self._active_runs.pop(run.session_id, None)
                    break

    # ── Output download ───────────────────────────────────────────────────────

    async def download_output(self, kernel_slug: str) -> bytes:
        """Download the kernel output ZIP (model files, CSVs, etc.)."""
        url = f"{KAGGLE_BASE}/kernels/{settings.KAGGLE_USERNAME}/{kernel_slug}/output"
        async with httpx.AsyncClient(auth=self.auth, timeout=120) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            return resp.content

    # ── Quota check ───────────────────────────────────────────────────────────

    async def remaining_gpu_quota(self) -> dict:
        """Returns remaining GPU quota hours from Kaggle."""
        async with httpx.AsyncClient(auth=self.auth, timeout=15) as client:
            resp = await client.get(f"{KAGGLE_BASE}/users/{settings.KAGGLE_USERNAME}")
            data = resp.json()
            return {
                "used_gpu_hours": data.get("gpuQuotaUsed", 0),
                "total_gpu_hours": settings.KAGGLE_GPU_QUOTA_HOURS,
                "remaining_hours": settings.KAGGLE_GPU_QUOTA_HOURS - data.get("gpuQuotaUsed", 0),
            }

    # ── Cancel ────────────────────────────────────────────────────────────────

    async def cancel_kernel(self, session_id: str):
        run = self._active_runs.get(session_id)
        if not run:
            return
        slug = f"{settings.KAGGLE_USERNAME}/{run.kernel_slug}"
        async with httpx.AsyncClient(auth=self.auth, timeout=15) as client:
            await client.post(f"{KAGGLE_BASE}/kernels/{slug}/stop", headers=HEADERS)
        log.info("Kernel cancel requested: %s", slug)


kaggle_bridge = KaggleBridge()
