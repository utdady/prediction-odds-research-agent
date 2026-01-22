"""Alert manager for high-confidence signals."""
from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import httpx
import structlog

if TYPE_CHECKING:
    from pm_agent.schemas import Signal

log = structlog.get_logger(__name__)


class AlertManager:
    """Manages real-time alerts for signals."""

    def __init__(self, email: str | None = None, slack_webhook: str | None = None):
        self.email = email
        self.slack_webhook = slack_webhook

    async def send_signal_alert(self, signal: Signal) -> None:
        """Send alert when new high-confidence signal is generated."""
        if signal.strength < 0.7:  # Only alert on high confidence
            return

        message = f"""
🚨 High Confidence Signal
Ticker: {signal.entity_id}
Strategy: {signal.strategy}
Strength: {signal.strength:.2%}
Side: {signal.side}
Horizon: {signal.horizon_days} days
Timestamp: {signal.ts}
"""

        # Send via available channels
        tasks = []
        if self.email:
            tasks.append(self._send_email(message, signal))
        if self.slack_webhook:
            tasks.append(self._send_slack(message))

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
            log.info("signal_alert_sent", entity_id=signal.entity_id, strength=signal.strength)

    async def _send_email(self, message: str, signal: Signal) -> None:
        """Send email alert (placeholder - requires SMTP config)."""
        # TODO: Implement actual email sending with smtplib
        log.info("email_alert_skipped", reason="not_configured", entity_id=signal.entity_id)

    async def _send_slack(self, text: str) -> None:
        """Send Slack webhook notification."""
        if not self.slack_webhook:
            return

        try:
            async with httpx.AsyncClient() as client:
                await client.post(
                    self.slack_webhook,
                    json={"text": text},
                    timeout=5.0,
                )
        except Exception as e:
            log.warning("slack_alert_failed", error=str(e))

