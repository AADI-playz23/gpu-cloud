"""
NovaDev Cloud — WebSocket Manager
Lives permanently on FastAPI server (absoracode.fanclub.rocks).
GitHub Actions relay never touches this — it only sends HTTP POSTs.

Cloudflare free plan drops WebSocket after 100 seconds of inactivity.
We send a ping every 30s to keep the connection alive.
Client js/api.js auto-reconnects if connection drops.
"""
import asyncio
import json
import logging
from typing import Dict
from fastapi import WebSocket, WebSocketDisconnect

log = logging.getLogger("ws_manager")

PING_INTERVAL = 30   # seconds — keeps alive through Cloudflare 100s timeout


class WebSocketManager:
    def __init__(self):
        # user_id → WebSocket connection
        self._connections: Dict[str, WebSocket] = {}
        self._ping_task: asyncio.Task | None = None

    async def connect(self, websocket: WebSocket, user_id: str):
        await websocket.accept()
        self._connections[user_id] = websocket
        log.info("WS connected: %s (total: %d)", user_id, len(self._connections))

        # Start ping loop if not running
        if self._ping_task is None or self._ping_task.done():
            self._ping_task = asyncio.create_task(self._ping_loop())

    def disconnect(self, user_id: str):
        self._connections.pop(user_id, None)
        log.info("WS disconnected: %s (total: %d)", user_id, len(self._connections))

    async def send(self, user_id: str, data: dict):
        """Send event to one user."""
        ws = self._connections.get(user_id)
        if ws:
            try:
                await ws.send_text(json.dumps(data))
            except Exception:
                self.disconnect(user_id)

    async def broadcast(self, data: dict):
        """Send event to all connected users."""
        if not self._connections:
            return
        msg = json.dumps(data)
        dead = []
        for uid, ws in self._connections.items():
            try:
                await ws.send_text(msg)
            except Exception:
                dead.append(uid)
        for uid in dead:
            self.disconnect(uid)
        if data.get("event") != "ping":
            log.info("Broadcast %s → %d users", data.get("event"), len(self._connections))

    async def handle_message(self, user_id: str, raw: str):
        """Handle messages from client (pong, etc.)"""
        try:
            msg = json.loads(raw)
            if msg.get("type") == "pong":
                pass   # heartbeat acknowledged
        except Exception:
            pass

    async def _ping_loop(self):
        """
        Ping all clients every 30s.
        Cloudflare free plan closes WebSocket after 100s idle.
        30s ping keeps it alive well within that limit.
        """
        while self._connections:
            await asyncio.sleep(PING_INTERVAL)
            await self.broadcast({"event": "ping", "ts": asyncio.get_event_loop().time()})


ws_manager = WebSocketManager()
