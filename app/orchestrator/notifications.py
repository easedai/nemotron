from __future__ import annotations

from typing import Optional

import httpx
import structlog

from ..config import settings

log = structlog.get_logger(__name__)

_COLOURS = {
    "info":    0x5865F2,  # Discord blurple
    "success": 0x57F287,  # Green
    "warning": 0xFEE75C,  # Yellow
    "error":   0xED4245,  # Red
}


class Discord:
    async def send(
        self,
        message: str,
        level:   str            = "info",
        title:   Optional[str]  = None,
        fields:  Optional[list] = None,
    ) -> None:
        """
        Post an embed to the configured Discord webhook.

        Parameters
        ----------
        message : str
            Embed description (markdown supported).
        level : str
            Colour key — "info" | "warning" | "error" | "success".
        title : str, optional
            Bold title line rendered above the description.
        fields : list[dict], optional
            Discord embed fields: ``[{"name": "...", "value": "...", "inline": True}, ...]``
        """
        # Prefix non-prod replicas so local dev is visually distinct in Discord.
        prefix = f"[{settings.orchestrator_id}] " if settings.orchestrator_id != "prod" else ""

        embed: dict = {
            "description": f"{prefix}{message}",
            "color":       _COLOURS.get(level, _COLOURS["info"]),
        }
        if title:
            # Put the env prefix on the title only; description already carries it.
            embed["title"] = f"{prefix}{title}" if prefix else title
        if fields:
            embed["fields"] = fields

        payload = {"embeds": [embed]}
        log.debug("discord.send", level=level, title=title, message=message)
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                r = await client.post(settings.discord_webhook_url, json=payload)
                r.raise_for_status()
                log.debug("discord.sent", status=r.status_code)
        except Exception as exc:
            # Never let a notification failure crash the orchestrator
            log.warning("discord.send.failed", error=str(exc))
