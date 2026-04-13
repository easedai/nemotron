from __future__ import annotations

import httpx
import structlog

from .config import settings

log = structlog.get_logger(__name__)

_COLOURS = {
    "info":    0x5865F2,  # Discord blurple
    "success": 0x57F287,  # Green
    "warning": 0xFEE75C,  # Yellow
    "error":   0xED4245,  # Red
}


class Discord:
    async def send(self, message: str, level: str = "info") -> None:
        # Prefix non-prod replicas so local dev is visually distinct in Discord.
        if settings.orchestrator_id != "prod":
            message = f"[{settings.orchestrator_id}] {message}"
        payload = {
            "embeds": [{
                "description": message,
                "color": _COLOURS.get(level, _COLOURS["info"]),
            }]
        }
        log.debug("discord.send", level=level, message=message)
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                r = await client.post(settings.discord_webhook_url, json=payload)
                r.raise_for_status()
                log.debug("discord.sent", status=r.status_code)
        except Exception as exc:
            # Never let a notification failure crash the orchestrator
            log.warning("discord.send.failed", error=str(exc))
