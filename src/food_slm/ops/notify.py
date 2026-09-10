"""Email notification for the watchdog. Gmail SMTP over implicit TLS (port 465).

Credentials come from the environment (never committed) — load a local `.env`
first with `load_env()`:

    WATCHDOG_SMTP_USER          your gmail address
    WATCHDOG_SMTP_APP_PASSWORD  16-char Google App Password (Account > Security >
                                2-Step Verification > App passwords)
    WATCHDOG_MAIL_TO            recipient (defaults to WATCHDOG_SMTP_USER)
    WATCHDOG_MAIL_FROM          From: header (defaults to WATCHDOG_SMTP_USER)

No paid API. stdlib only.
"""
from __future__ import annotations

import os
import smtplib
import ssl
from email.message import EmailMessage
from email.utils import formatdate
from pathlib import Path

_SMTP_HOST = "smtp.gmail.com"
_SMTP_PORT = 465


def load_env(path: str | Path = ".env") -> None:
    """Minimal .env loader (KEY=VALUE per line, # comments, optional quotes).
    Does not overwrite variables already set in the environment."""
    p = Path(path)
    if not p.exists():
        return
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val


class MailConfig:
    def __init__(self) -> None:
        self.user = os.environ.get("WATCHDOG_SMTP_USER", "")
        self.password = os.environ.get("WATCHDOG_SMTP_APP_PASSWORD", "")
        self.to = os.environ.get("WATCHDOG_MAIL_TO", self.user)
        self.sender = os.environ.get("WATCHDOG_MAIL_FROM", self.user)

    @property
    def ready(self) -> bool:
        return bool(self.user and self.password and self.to)

    def missing(self) -> list[str]:
        out = []
        if not self.user:
            out.append("WATCHDOG_SMTP_USER")
        if not self.password:
            out.append("WATCHDOG_SMTP_APP_PASSWORD")
        if not self.to:
            out.append("WATCHDOG_MAIL_TO")
        return out


def send_mail(subject: str, body: str, *, cfg: MailConfig | None = None,
              timeout: float = 30.0) -> None:
    """Raises on failure (SMTP error, bad credentials, no network)."""
    cfg = cfg or MailConfig()
    if not cfg.ready:
        raise RuntimeError(f"mail not configured — missing {', '.join(cfg.missing())}")
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = cfg.sender
    msg["To"] = cfg.to
    msg["Date"] = formatdate(localtime=True)
    msg.set_content(body)
    ctx = ssl.create_default_context()
    with smtplib.SMTP_SSL(_SMTP_HOST, _SMTP_PORT, timeout=timeout, context=ctx) as s:
        s.login(cfg.user, cfg.password)
        s.send_message(msg)
