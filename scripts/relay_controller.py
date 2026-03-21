#!/usr/bin/env python3
"""
NovaDev Cloud — Relay Controller
Runs inside GitHub Actions for a 6-hour window.
Four of these run per day in sequence to create 24/7 monitoring.

Responsibilities:
  - Check if Kaggle notebook is alive every 30 seconds
  - Start notebook if demand exists and it's offline
  - Stop notebook if quota exhausted or no demand for 10 min
  - Trigger rotation workflow at 11h (before 12h Kaggle limit)
  - Send all state changes to backend via webhook
  - Handle weekly quota tracking
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
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("relay.log"),
    ],
)
log = logging.getLogger("relay")

# ── Config from env ───────────────────────────────────────────────────────────
KAGGLE_USERNAME     = os.environ["KAGGLE_USERNAME"]
KAGGLE_KEY          = os.environ["KAGGLE_KEY"]
BACKEND_URL         = os.environ["BACKEND_WEBHOOK_URL"]
BACKEND_SECRET      = os.environ["BACKEND_SECRET"]
KERNEL_SLUG         = os.environ.get("KAGGLE_KERNEL_SLUG", "novadev-gpu-pool")
POLL_SECS           = int(os.environ.get("POLL_INTERVAL_SECS", 30))
MAX_RUNTIME_H       = float(os.environ.get("MAX_RUNTIME_HOURS", 11))
RELAY_WINDOW_H      = float(os.environ.get("RELAY_WINDOW_HOURS", 6))
FORCE_START         = os.environ.get("FORCE_START", "false").lower() == "true"
FORCE_STOP          = os.environ.get("FORCE_STOP",  "false").lower() == "true"
GITHUB_TOKEN        = os.environ.get("GITHUB_TOKEN", "")
GITHUB_REPO         = os.environ.get("GITHUB_REPOSITORY", "AADI-playz23/gpu-cloud")

KAGGLE_BASE         = "https://www.kaggle.com/api/v1"
KAGGLE_AUTH         = (KAGGLE_USERNAME, KAGGLE_KEY)

# Quota limits
QUOTA_TOTAL_H       = 30.0
FREE_QUOTA_H        = 1.0    # per user per week
PAID_QUOTA_H        = 2.0    # per user per week
IDLE_STOP_SECS      = 600    # stop notebook after 10min no demand
ROTATION_WARN_H     = 11.0   # trigger rotation at this age


# ── State file (persisted across poll loops within one GHA run) ───────────────
STATE_FILE = Path("/tmp/novadev_relay_state.json")

def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {
        "notebook_slug": KERNEL_SLUG,
        "notebook_started_at": None,
        "notebook_status": "unknown",
        "last_demand_at": None,
        "idle_since": None,
        "rotation_triggered": False,
        "quota_used_h": 0.0,
        "quota_week_start": None,
    }

def save_state(state: dict):
    STATE_FILE.write_text(json.dumps(state, indent=2))


# ── Webhook to backend ────────────────────────────────────────────────────────

def sign_payload(payload: str) -> str:
    return hmac.new(
        BACKEND_SECRET.encode(), payload.encode(), hashlib.sha256
    ).hexdigest()

async def webhook(client: httpx.AsyncClient, event: str, data: dict):
    payload = json.dumps({"event": event, "ts": datetime.now(timezone.utc).isoformat(), **data})
    sig = sign_payload(payload)
    try:
        resp = await client.post(
            f"{BACKEND_URL}/api/internal/relay-webhook",
            content=payload,
            headers={"Content-Type": "application/json", "X-NovaDev-Sig": sig},
            timeout=10,
        )
        log.info("Webhook %s → %d", event, resp.status_code)
    except Exception as e:
        log.warning("Webhook failed (%s): %s", event, e)


# ── Kaggle API calls ──────────────────────────────────────────────────────────

async def kaggle_get(client: httpx.AsyncClient, path: str) -> dict:
    resp = await client.get(f"{KAGGLE_BASE}{path}", auth=KAGGLE_AUTH, timeout=20)
    resp.raise_for_status()
    return resp.json()

async def kaggle_post(client: httpx.AsyncClient, path: str, data: dict) -> dict:
    resp = await client.post(
        f"{KAGGLE_BASE}{path}", json=data, auth=KAGGLE_AUTH, timeout=30
    )
    resp.raise_for_status()
    return resp.json()

async def get_notebook_status(client: httpx.AsyncClient, slug: str) -> str:
    """Returns: running | complete | error | cancelAcknowledged | notFound"""
    try:
        data = await kaggle_get(client, f"/kernels/{KAGGLE_USERNAME}/{slug}")
        return data.get("status", "unknown").lower()
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            return "notfound"
        raise

async def start_notebook(client: httpx.AsyncClient, slug: str) -> dict:
    """Push (or re-push) the Kaggle notebook kernel to start it."""
    log.info("Starting Kaggle notebook: %s", slug)

    # Read the notebook source from repo
    nb_path = Path("kaggle_notebooks/novadev_pool.ipynb")
    if nb_path.exists():
        notebook_source = nb_path.read_text()
    else:
        notebook_source = _default_notebook_source()

    payload = {
        "id":              f"{KAGGLE_USERNAME}/{slug}",
        "title":           "NovaDev GPU Pool",
        "code_file":       notebook_source,
        "language":        "python",
        "kernel_type":     "notebook",
        "is_private":      True,
        "enable_gpu":      True,
        "enable_internet": False,
        "accelerator":     "nvidiaTeslaT4",
    }

    result = await kaggle_post(client, "/kernels", payload)
    log.info("Notebook push result: %s", result)
    return result

async def stop_notebook(client: httpx.AsyncClient, slug: str):
    """Cancel a running Kaggle kernel."""
    try:
        await kaggle_post(client, f"/kernels/{KAGGLE_USERNAME}/{slug}/stop", {})
        log.info("Notebook stop requested: %s", slug)
    except Exception as e:
        log.warning("Could not stop notebook %s: %s", slug, e)

async def get_kaggle_quota(client: httpx.AsyncClient) -> dict:
    """Fetch remaining GPU quota from Kaggle user profile."""
    try:
        data = await kaggle_get(client, f"/users/{KAGGLE_USERNAME}")
        used = float(data.get("gpuQuotaUsed", 0))
        total = QUOTA_TOTAL_H
        return {
            "used_h": round(used, 2),
            "remaining_h": round(total - used, 2),
            "total_h": total,
            "pct_used": round(used / total * 100, 1),
        }
    except Exception as e:
        log.warning("Could not fetch Kaggle quota: %s", e)
        return {"used_h": 0, "remaining_h": QUOTA_TOTAL_H, "total_h": QUOTA_TOTAL_H, "pct_used": 0}

async def check_backend_demand(client: httpx.AsyncClient) -> bool:
    """Ask the backend if any users are in the queue or have active sessions."""
    try:
        resp = await client.get(
            f"{BACKEND_URL}/api/internal/demand",
            headers={"X-NovaDev-Secret": BACKEND_SECRET},
            timeout=10,
        )
        data = resp.json()
        return data.get("has_demand", False)
    except Exception as e:
        log.warning("Could not check demand: %s — assuming demand exists", e)
        return True  # Fail open: assume demand if backend unreachable


# ── Rotation trigger ──────────────────────────────────────────────────────────

async def trigger_rotation(old_slug: str):
    """Dispatch the notebook_rotation workflow via GitHub API."""
    if not GITHUB_TOKEN or not GITHUB_REPO:
        log.warning("No GITHUB_TOKEN/REPO set — cannot trigger rotation workflow")
        return
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"https://api.github.com/repos/{GITHUB_REPO}/dispatches",
            json={"event_type": "rotate-notebook",
                  "client_payload": {"old_slug": old_slug}},
            headers={
                "Authorization": f"Bearer {GITHUB_TOKEN}",
                "Accept": "application/vnd.github+json",
            },
            timeout=15,
        )
        if resp.status_code == 204:
            log.info("Rotation workflow dispatched for slug: %s", old_slug)
        else:
            log.error("Rotation dispatch failed: %d %s", resp.status_code, resp.text)


# ── Main relay loop ───────────────────────────────────────────────────────────

async def relay_loop():
    state = load_state()
    start_time = time.time()
    relay_window_secs = RELAY_WINDOW_H * 3600
    idle_since: float | None = None

    log.info("=== Relay controller started. Window: %.0fh. Slug: %s ===",
             RELAY_WINDOW_H, state["notebook_slug"])

    async with httpx.AsyncClient() as client:

        # Handle force stop immediately
        if FORCE_STOP:
            log.info("Force stop requested.")
            await stop_notebook(client, state["notebook_slug"])
            await webhook(client, "notebook_force_stopped", {"slug": state["notebook_slug"]})
            save_state(state)
            return

        iteration = 0
        while time.time() - start_time < relay_window_secs:
            iteration += 1
            now = time.time()
            elapsed_relay = now - start_time

            # ── 1. Get current notebook status ──────────────────────────────
            nb_status = await get_notebook_status(client, state["notebook_slug"])
            nb_alive  = nb_status == "running"
            log.info("[%03d] Notebook: %-20s | Relay elapsed: %.0fm",
                     iteration, nb_status, elapsed_relay / 60)

            # ── 2. Calculate notebook age ────────────────────────────────────
            notebook_age_h = 0.0
            if state.get("notebook_started_at") and nb_alive:
                started = datetime.fromisoformat(state["notebook_started_at"])
                notebook_age_h = (datetime.now(timezone.utc) - started).total_seconds() / 3600

            # ── 3. Quota check ───────────────────────────────────────────────
            quota = await get_kaggle_quota(client)
            log.info("     Quota: %.2fh used / %.2fh remaining (%.1f%%)",
                     quota["used_h"], quota["remaining_h"], quota["pct_used"])

            # Send hourly quota report (every 120 iterations × 30s = 1h)
            if iteration % 120 == 1:
                await webhook(client, "quota_report", quota)

            # ── 4. Quota exhausted → stop everything ─────────────────────────
            if quota["remaining_h"] <= 0:
                log.warning("Quota exhausted! Stopping notebook and blocking sessions.")
                if nb_alive:
                    await stop_notebook(client, state["notebook_slug"])
                await webhook(client, "quota_exhausted", {
                    **quota,
                    "message": "Weekly GPU quota exhausted. All new sessions blocked."
                })
                save_state(state)
                log.info("Quota exhausted. Relay exiting.")
                return

            # ── 5. Rotation check (at 11h notebook age) ───────────────────────
            if nb_alive and notebook_age_h >= ROTATION_WARN_H and not state.get("rotation_triggered"):
                log.info("Notebook age %.1fh ≥ %.1fh — triggering rotation.",
                         notebook_age_h, ROTATION_WARN_H)
                state["rotation_triggered"] = True
                save_state(state)
                await webhook(client, "rotation_starting", {
                    "old_slug": state["notebook_slug"],
                    "notebook_age_h": round(notebook_age_h, 2),
                })
                await trigger_rotation(state["notebook_slug"])

            # ── 6. Check demand ───────────────────────────────────────────────
            has_demand = FORCE_START or await check_backend_demand(client)

            # ── 7. Notebook offline + demand → start it ────────────────────────
            if not nb_alive and has_demand:
                log.info("Notebook offline + demand detected — starting notebook.")
                try:
                    await start_notebook(client, state["notebook_slug"])
                    state["notebook_started_at"] = datetime.now(timezone.utc).isoformat()
                    state["rotation_triggered"]  = False
                    idle_since = None

                    # Wait for notebook to reach running state (up to 10 min)
                    for wait_i in range(20):
                        await asyncio.sleep(30)
                        nb_status = await get_notebook_status(client, state["notebook_slug"])
                        log.info("  Startup check %d: %s", wait_i + 1, nb_status)
                        if nb_status == "running":
                            await webhook(client, "notebook_started", {
                                "slug": state["notebook_slug"],
                                "started_at": state["notebook_started_at"],
                            })
                            break
                    else:
                        log.error("Notebook did not reach running state after 10 min!")
                        await webhook(client, "notebook_start_failed", {
                            "slug": state["notebook_slug"]
                        })
                except Exception as e:
                    log.error("Failed to start notebook: %s", e)
                    await webhook(client, "notebook_start_error", {"error": str(e)})

            # ── 8. Notebook running + no demand → idle timer ──────────────────
            elif nb_alive and not has_demand:
                if idle_since is None:
                    idle_since = now
                    log.info("No demand detected — idle timer started.")
                elif now - idle_since >= IDLE_STOP_SECS:
                    log.info("Idle for %ds — stopping notebook.", int(now - idle_since))
                    await stop_notebook(client, state["notebook_slug"])
                    await webhook(client, "notebook_idle_stopped", {
                        "slug": state["notebook_slug"],
                        "idle_secs": int(now - idle_since),
                    })
                    state["notebook_started_at"] = None
                    idle_since = None

            # ── 9. Demand present → reset idle timer ──────────────────────────
            elif nb_alive and has_demand:
                if idle_since is not None:
                    log.info("Demand resumed — idle timer reset.")
                    idle_since = None

            # ── 10. Gate webhook based on quota level ─────────────────────────
            remaining = quota["remaining_h"]
            if iteration % 30 == 0:   # every 15 min
                gate = (
                    "blocked"   if remaining <= 1 else
                    "paid_only" if remaining <= 3 else
                    "warn"      if remaining <= 6 else
                    "normal"
                )
                await webhook(client, "quota_gate_update", {
                    "gate": gate,
                    "remaining_h": round(remaining, 2),
                    "notebook_alive": nb_alive,
                    "notebook_age_h": round(notebook_age_h, 2),
                })

            save_state(state)
            await asyncio.sleep(POLL_SECS)

    log.info("=== Relay window complete (%.0fh). Handing off to next run. ===", RELAY_WINDOW_H)


# ── Notebook source ───────────────────────────────────────────────────────────

def _default_notebook_source() -> str:
    """
    The Kaggle notebook that runs as the GPU backend.
    It starts a Jupyter server on the Kaggle VM so users can
    connect to it via the kernel gateway.
    """
    return json.dumps({
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}
        },
        "cells": [
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# NovaDev Cloud — GPU Pool Notebook\n",
                    "# This notebook runs as the shared GPU backend.\n",
                    "# It starts a Jupyter kernel gateway that all user containers connect to.\n",
                    "\n",
                    "import subprocess, time, os, json\n",
                    "\n",
                    "# Print GPU info\n",
                    "result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,memory.free',\n",
                    "                         '--format=csv,noheader'], capture_output=True, text=True)\n",
                    "print('GPU info:', result.stdout.strip())\n",
                    "\n",
                    "# Install kernel gateway\n",
                    "subprocess.run(['pip', 'install', 'jupyter_kernel_gateway', '-q'], check=True)\n",
                    "\n",
                    "# Start Jupyter kernel gateway on port 8888\n",
                    "# Users connect to this via ngrok or Cloudflare tunnel\n",
                    "gw = subprocess.Popen([\n",
                    "    'jupyter', 'kernelgateway',\n",
                    "    '--KernelGatewayApp.ip=0.0.0.0',\n",
                    "    '--KernelGatewayApp.port=8888',\n",
                    "    '--KernelGatewayApp.allow_origin=*',\n",
                    "    '--KernelGatewayApp.auth_token=',\n",
                    "])\n",
                    "\n",
                    "print('Kernel gateway started. PID:', gw.pid)\n",
                    "\n",
                    "# Keep-alive heartbeat (prevents Kaggle idle kill after 20min)\n",
                    "heartbeat = 0\n",
                    "while True:\n",
                    "    time.sleep(60)\n",
                    "    heartbeat += 1\n",
                    "    print(f'[heartbeat {heartbeat}] {time.strftime(\"%H:%M:%S\")} | GPU alive')\n",
                ]
            }
        ]
    }, indent=2)


if __name__ == "__main__":
    asyncio.run(relay_loop())
