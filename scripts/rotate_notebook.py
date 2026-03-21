#!/usr/bin/env python3
"""
NovaDev Cloud — Notebook Rotation Script
Triggered by GitHub Actions workflow_dispatch at the 11h mark.

Steps:
  1. Generate a new kernel slug (rolling index)
  2. Push new notebook to Kaggle
  3. Wait for it to reach "running" state (max 15 min)
  4. Notify backend → backend migrates all sessions
  5. Stop old notebook
  6. Update relay state with new slug
"""

import asyncio
import hashlib
import hmac
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("rotation")

KAGGLE_USERNAME = os.environ["KAGGLE_USERNAME"]
KAGGLE_KEY      = os.environ["KAGGLE_KEY"]
BACKEND_URL     = os.environ["BACKEND_WEBHOOK_URL"]
BACKEND_SECRET  = os.environ["BACKEND_SECRET"]
OLD_SLUG        = os.environ.get("OLD_SLUG", "")
KAGGLE_BASE     = "https://www.kaggle.com/api/v1"
KAGGLE_AUTH     = (KAGGLE_USERNAME, KAGGLE_KEY)

STATE_FILE      = Path("/tmp/novadev_relay_state.json")


def sign(payload: str) -> str:
    return hmac.new(BACKEND_SECRET.encode(), payload.encode(), hashlib.sha256).hexdigest()


def new_slug() -> str:
    ts = int(time.time())
    return f"novadev-pool-{ts}"


def notebook_source(slug: str) -> str:
    return json.dumps({
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            }
        },
        "cells": [{
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                f"# NovaDev Pool Notebook — {slug}\n",
                "# Auto-rotated by GitHub Actions at 11h mark.\n",
                "\n",
                "import subprocess, time\n",
                "\n",
                "result = subprocess.run(\n",
                "    ['nvidia-smi', '--query-gpu=name,memory.total,memory.free',\n",
                "     '--format=csv,noheader'],\n",
                "    capture_output=True, text=True\n",
                ")\n",
                "print('GPU:', result.stdout.strip())\n",
                "\n",
                "subprocess.run(['pip', 'install', 'jupyter_kernel_gateway', '-q'], check=True)\n",
                "\n",
                "gw = subprocess.Popen([\n",
                "    'jupyter', 'kernelgateway',\n",
                "    '--KernelGatewayApp.ip=0.0.0.0',\n",
                "    '--KernelGatewayApp.port=8888',\n",
                "    '--KernelGatewayApp.allow_origin=*',\n",
                "    '--KernelGatewayApp.auth_token=',\n",
                "])\n",
                "print(f'Kernel gateway PID: {gw.pid}')\n",
                "\n",
                "hb = 0\n",
                "while True:\n",
                "    time.sleep(60)\n",
                "    hb += 1\n",
                f"    print(f'[{slug}] heartbeat {{hb}} | {{time.strftime(\"%H:%M:%S\")}}')\n",
            ],
        }],
    }, indent=2)


async def run():
    old_slug = OLD_SLUG
    new_slug_val = new_slug()

    log.info("Rotation: %s → %s", old_slug or "(unknown)", new_slug_val)

    async with httpx.AsyncClient(timeout=30) as client:

        # ── 1. Push new notebook ───────────────────────────────────────────────
        log.info("Pushing new notebook: %s", new_slug_val)
        try:
            resp = await client.post(
                f"{KAGGLE_BASE}/kernels",
                json={
                    "id":              f"{KAGGLE_USERNAME}/{new_slug_val}",
                    "title":           f"NovaDev Pool {new_slug_val}",
                    "code_file":       notebook_source(new_slug_val),
                    "language":        "python",
                    "kernel_type":     "notebook",
                    "is_private":      True,
                    "enable_gpu":      True,
                    "enable_internet": False,
                    "accelerator":     "nvidiaTeslaT4",
                },
                auth=KAGGLE_AUTH,
            )
            resp.raise_for_status()
            log.info("New notebook pushed: %s", resp.json())
        except Exception as e:
            log.error("Failed to push new notebook: %s", e)
            raise

        # ── 2. Notify backend: rotation starting ──────────────────────────────
        payload = json.dumps({
            "event":    "rotation_starting",
            "old_slug": old_slug,
            "new_slug": new_slug_val,
            "ts":       datetime.now(timezone.utc).isoformat(),
        })
        await client.post(
            f"{BACKEND_URL}/api/internal/relay-webhook",
            content=payload,
            headers={"Content-Type": "application/json", "X-NovaDev-Sig": sign(payload)},
        )

        # ── 3. Wait for new notebook to be "running" ──────────────────────────
        log.info("Waiting for new notebook to start (max 15 min)...")
        started = False
        for i in range(30):   # 30 × 30s = 15 min
            await asyncio.sleep(30)
            try:
                status_resp = await client.get(
                    f"{KAGGLE_BASE}/kernels/{KAGGLE_USERNAME}/{new_slug_val}",
                    auth=KAGGLE_AUTH,
                )
                status = status_resp.json().get("status", "").lower()
                log.info("  [%d] New notebook status: %s", i + 1, status)
                if status == "running":
                    started = True
                    break
            except Exception as e:
                log.warning("  Status check failed: %s", e)

        if not started:
            log.error("New notebook did not reach running state!")
            payload = json.dumps({
                "event": "rotation_failed",
                "new_slug": new_slug_val,
                "reason": "Did not reach running state in 15 min",
            })
            await client.post(
                f"{BACKEND_URL}/api/internal/relay-webhook",
                content=payload,
                headers={"Content-Type": "application/json", "X-NovaDev-Sig": sign(payload)},
            )
            return

        # ── 4. Notify backend: ready to migrate ───────────────────────────────
        log.info("New notebook running! Notifying backend to migrate sessions.")
        payload = json.dumps({
            "event":    "rotation_ready",
            "old_slug": old_slug,
            "new_slug": new_slug_val,
            "ts":       datetime.now(timezone.utc).isoformat(),
        })
        await client.post(
            f"{BACKEND_URL}/api/internal/relay-webhook",
            content=payload,
            headers={"Content-Type": "application/json", "X-NovaDev-Sig": sign(payload)},
        )

        # ── 5. Wait for backend to confirm migration (2 min grace period) ─────
        log.info("Waiting 2 min for session migration to complete...")
        await asyncio.sleep(120)

        # ── 6. Stop old notebook ──────────────────────────────────────────────
        if old_slug:
            log.info("Stopping old notebook: %s", old_slug)
            try:
                stop_resp = await client.post(
                    f"{KAGGLE_BASE}/kernels/{KAGGLE_USERNAME}/{old_slug}/stop",
                    auth=KAGGLE_AUTH,
                )
                log.info("Old notebook stopped: %d", stop_resp.status_code)
            except Exception as e:
                log.warning("Could not stop old notebook: %s", e)

        # ── 7. Final confirmation webhook ─────────────────────────────────────
        payload = json.dumps({
            "event":        "rotation_complete",
            "new_slug":     new_slug_val,
            "old_slug":     old_slug,
            "ts":           datetime.now(timezone.utc).isoformat(),
        })
        await client.post(
            f"{BACKEND_URL}/api/internal/relay-webhook",
            content=payload,
            headers={"Content-Type": "application/json", "X-NovaDev-Sig": sign(payload)},
        )

        # ── 8. Update state file for next relay run ───────────────────────────
        if STATE_FILE.exists():
            state = json.loads(STATE_FILE.read_text())
        else:
            state = {}
        state["notebook_slug"]       = new_slug_val
        state["notebook_started_at"] = datetime.now(timezone.utc).isoformat()
        state["rotation_triggered"]  = False
        STATE_FILE.write_text(json.dumps(state, indent=2))

        log.info("=== Rotation complete: %s is now primary ===", new_slug_val)


if __name__ == "__main__":
    asyncio.run(run())
