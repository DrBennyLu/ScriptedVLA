"""Shared WebSocket client for LIBERO environment server."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

try:
    import websockets
except ImportError as exc:
    raise ImportError("Install websockets: uv add websockets") from exc


class LiberoWSClient:
    def __init__(self, ws_url: str):
        self.ws_url = ws_url
        self._ws = None

    async def connect(self):
        self._ws = await websockets.connect(self.ws_url, max_size=64 * 1024 * 1024)
        return self

    async def close(self):
        if self._ws is not None:
            await self._ws.close()
            self._ws = None

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.close()

    async def send(self, message: Dict[str, Any]) -> Dict[str, Any]:
        if self._ws is None:
            raise RuntimeError("WebSocket not connected")
        await self._ws.send(json.dumps(message))
        raw = await self._ws.recv()
        reply = json.loads(raw)
        if reply.get("type") == "error":
            raise RuntimeError(reply.get("message", "Unknown server error"))
        return reply

    async def ping(self) -> Dict[str, Any]:
        return await self.send({"type": "ping"})

    async def list_tasks(self, suite: str = "libero_object") -> List[Dict[str, Any]]:
        reply = await self.send({"type": "list_tasks", "suite": suite})
        return reply.get("tasks", [])

    async def create_episode(
        self,
        task_id: int,
        init_id: int = 0,
        suite: str = "libero_object",
        max_steps: int = 600,
    ) -> Dict[str, Any]:
        return await self.send(
            {
                "type": "create_episode",
                "task_id": task_id,
                "init_id": init_id,
                "suite": suite,
                "max_steps": max_steps,
            }
        )

    async def get_obs(
        self, episode_id: str, include_images: bool = True
    ) -> Dict[str, Any]:
        return await self.send(
            {
                "type": "get_obs",
                "episode_id": episode_id,
                "include_images": include_images,
            }
        )

    async def step(
        self,
        episode_id: str,
        action: List[float],
        include_images: bool = True,
    ) -> Dict[str, Any]:
        return await self.send(
            {
                "type": "step",
                "episode_id": episode_id,
                "action": action,
                "include_images": include_images,
            }
        )

    async def close_episode(self, episode_id: str) -> Dict[str, Any]:
        return await self.send({"type": "close_episode", "episode_id": episode_id})
