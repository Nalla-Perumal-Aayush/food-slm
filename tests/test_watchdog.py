"""Watchdog check-kinds + .env loader + mail-config gating."""
from __future__ import annotations

import time
from pathlib import Path

import pytest

from food_slm.ops.notify import MailConfig, load_env


def test_load_env_does_not_clobber(tmp_path, monkeypatch):
    monkeypatch.setenv("WATCHDOG_MAIL_TO", "already@set.com")
    env = tmp_path / ".env"
    env.write_text('WATCHDOG_MAIL_TO=other@x.com\nWATCHDOG_SMTP_USER="u@g.com"\n# c\nBAD\n',
                   encoding="utf-8")
    load_env(env)
    import os
    assert os.environ["WATCHDOG_MAIL_TO"] == "already@set.com"   # not overwritten
    assert os.environ["WATCHDOG_SMTP_USER"] == "u@g.com"         # quotes stripped


def test_mail_config_not_ready_without_creds(monkeypatch):
    for k in ("WATCHDOG_SMTP_USER", "WATCHDOG_SMTP_APP_PASSWORD", "WATCHDOG_MAIL_TO"):
        monkeypatch.delenv(k, raising=False)
    cfg = MailConfig()
    assert not cfg.ready
    assert set(cfg.missing()) == {"WATCHDOG_SMTP_USER", "WATCHDOG_SMTP_APP_PASSWORD",
                                  "WATCHDOG_MAIL_TO"}


def test_file_fresh_check(tmp_path):
    from scripts.watchdog import _check_file_fresh  # type: ignore

    f = tmp_path / "metrics.jsonl"
    f.write_text("x", encoding="utf-8")
    ok, _ = _check_file_fresh({"path": str(f), "max_age_seconds": 60})
    assert ok
    old = time.time() - 3600
    import os
    os.utime(f, (old, old))
    ok, detail = _check_file_fresh({"path": str(f), "max_age_seconds": 60})
    assert not ok and "stale" in detail
    ok, detail = _check_file_fresh({"path": str(tmp_path / "nope"), "max_age_seconds": 60})
    assert not ok and "does not exist" in detail


@pytest.fixture(autouse=True)
def _scripts_on_path():
    import sys
    root = Path(__file__).resolve().parents[1]
    for p in (str(root), str(root / "src")):
        if p not in sys.path:
            sys.path.insert(0, p)
