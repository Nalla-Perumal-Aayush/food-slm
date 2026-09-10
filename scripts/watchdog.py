"""Process/health watchdog: email on death, auto-restart the training run.

    python scripts/watchdog.py --test-mail      # verify .env credentials, send one mail
    python scripts/watchdog.py --once           # evaluate all checks once, print, exit
    python scripts/watchdog.py                  # run the monitoring loop (foreground)

Runs independently of the Claude session and the training PowerShell, so it can
report when *those* die. For it to survive logout/reboot, register it as a
Windows Scheduled Task (see the block printed by --print-task).

Config: configs/watchdog.yaml   Credentials: .env (gitignored, see .env.example)
stdlib + PyYAML only. No GPU, no paid API.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from food_slm.ops.notify import MailConfig, load_env, send_mail  # noqa: E402
from food_slm.utils import load_yaml  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
HOST = None


def _now() -> str:
    return datetime.now(UTC).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")


def _log(path: Path, msg: str) -> None:
    line = f"[{_now()}] {msg}"
    print(line, flush=True)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except OSError:
        pass


# --- individual check kinds ------------------------------------------------

def _check_file_fresh(spec: dict) -> tuple[bool, str]:
    p = (REPO / spec["path"]) if not Path(spec["path"]).is_absolute() else Path(spec["path"])
    if not p.exists():
        return False, f"{spec['path']} does not exist"
    age = time.time() - p.stat().st_mtime
    limit = float(spec.get("max_age_seconds", 300))
    if age <= limit:
        return True, f"fresh ({age:.0f}s ago)"
    return False, f"stale - last write {age:.0f}s ago (limit {limit:.0f}s)"


def _check_http_ok(spec: dict) -> tuple[bool, str]:
    try:
        req = urllib.request.Request(spec["url"], headers={"User-Agent": "watchdog"})
        with urllib.request.urlopen(req, timeout=10) as r:  # noqa: S310 - local URL
            return (200 <= r.status < 300), f"HTTP {r.status}"
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"


def _check_process(spec: dict) -> tuple[bool, str]:
    """Match a running process by a substring of its command line (cross-platform
    via `wmic` / `ps`)."""
    needle = spec["match"].lower()
    try:
        if sys.platform == "win32":
            out = subprocess.run(
                ["wmic", "process", "get", "CommandLine"], capture_output=True,
                text=True, timeout=15).stdout
        else:
            out = subprocess.run(["ps", "-eo", "args"], capture_output=True,
                                 text=True, timeout=15).stdout
    except Exception as exc:  # noqa: BLE001
        return False, f"process list failed: {type(exc).__name__}"
    hits = [ln.strip() for ln in out.splitlines() if needle in ln.lower()
            and "watchdog" not in ln.lower()]
    return (len(hits) > 0), (f"{len(hits)} match" if hits else "no matching process")


_KINDS = {"file_fresh": _check_file_fresh, "http_ok": _check_http_ok, "process": _check_process}


# --- restart handling -----------------------------------------------------

def _do_restart(spec: dict, state: dict, logf: Path) -> str:
    r = spec["restart"]
    st = state.setdefault(spec["name"], {})
    now = time.time()
    n = st.get("restart_count", 0)
    if n >= int(r.get("max_restarts", 3)):
        return f"restart limit reached ({n}) - NOT restarting; manual intervention needed"
    if now - st.get("last_restart_ts", 0) < float(r.get("cooldown_seconds", 600)):
        return "within restart cooldown - not restarting this cycle"
    cwd = (REPO / r.get("cwd", ".")).resolve()
    out_log = (REPO / r["log"]) if not Path(r["log"]).is_absolute() else Path(r["log"])
    out_log.parent.mkdir(parents=True, exist_ok=True)
    try:
        fh = out_log.open("a", encoding="utf-8")
        fh.write(f"\n===== watchdog restart {_now()} =====\n")
        fh.flush()
        kw = {}
        if sys.platform == "win32":
            kw["creationflags"] = 0x00000008 | 0x00000200  # DETACHED_PROCESS | NEW_PROCESS_GROUP
        else:
            kw["start_new_session"] = True
        subprocess.Popen(r["cmd"], shell=True, cwd=str(cwd), stdout=fh,
                         stderr=subprocess.STDOUT, **kw)
    except Exception as exc:  # noqa: BLE001
        return f"restart FAILED to launch: {type(exc).__name__}: {exc}"
    st["restart_count"] = n + 1
    st["last_restart_ts"] = now
    _log(logf, f"{spec['name']}: launched restart #{n + 1}: {r['cmd']}")
    return f"auto-restart #{n + 1} launched: `{r['cmd']}` (logging to {r['log']})"


# --- main loop -----------------------------------------------------------

def _evaluate(spec: dict) -> tuple[bool, str]:
    fn = _KINDS.get(spec["kind"])
    if fn is None:
        return False, f"unknown check kind {spec['kind']!r}"
    return fn(spec)


def _mail(subject: str, body: str, cfg: MailConfig, logf: Path) -> None:
    try:
        send_mail(subject, body + f"\n\n-- watchdog @ {HOST} · {_now()}", cfg=cfg)
        _log(logf, f"emailed: {subject}")
    except Exception as exc:  # noqa: BLE001
        _log(logf, f"EMAIL FAILED ({type(exc).__name__}: {exc}) - {subject}")


def run(cfg_path: Path, once: bool) -> int:
    cfg = load_yaml(cfg_path)
    logf = REPO / cfg.get("log_file", "artifacts/watchdog.log")
    statef = REPO / cfg.get("state_file", "artifacts/watchdog_state.json")
    poll = int(cfg.get("poll_seconds", 60))
    fail_n = int(cfg.get("fail_threshold", 3))
    hb_hours = float(cfg.get("heartbeat_hours", 0) or 0)
    mail = MailConfig()

    state: dict = {}
    if statef.exists():
        try:
            state = json.loads(statef.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            state = {}
    state.setdefault("_meta", {}).setdefault("last_heartbeat_ts", time.time())

    _log(logf, f"watchdog up - {len(cfg['checks'])} checks, poll {poll}s, "
               f"alert after {fail_n} fails, mail_to={mail.to or 'UNSET'}")
    if not mail.ready:
        _log(logf, f"WARNING: email not configured (missing {', '.join(mail.missing())}) "
                   f"- alerts will be logged only")

    while True:
        cycle_lines = []
        for spec in cfg["checks"]:
            name = spec["name"]
            st = state.setdefault(name, {"status": "unknown", "fails": 0, "ever_up": False})
            ok, detail = _evaluate(spec)
            cycle_lines.append(f"  {name:16s} {'OK ' if ok else 'BAD'}  {detail}")

            if ok:
                st["ever_up"] = True
                if st.get("status") == "down":
                    st["restart_count"] = 0
                    _mail(f"[watchdog] RECOVERED: {name}",
                          f"{name} is healthy again.\n  {detail}", mail, logf)
                st["status"] = "up"
                st["fails"] = 0
                continue

            # not ok
            st["fails"] = st.get("fails", 0) + 1
            if st.get("status") == "down" or st["fails"] < fail_n:
                if st.get("status") != "down":
                    _log(logf, f"{name}: fail {st['fails']}/{fail_n} - {detail}")
                continue

            # transition healthy -> down
            optional = spec.get("optional", False)
            if optional and not st.get("ever_up", False):
                st["status"] = "down"           # never came up; stay quiet
                continue
            st["status"] = "down"
            _log(logf, f"{name}: DOWN - {detail}")
            body = f"{name} went DOWN.\n  {detail}\n  (no healthy poll in " \
                   f"{fail_n * poll}s)"
            if "restart" in spec:
                body += "\n\n" + _do_restart(spec, state, logf)
            _mail(f"[watchdog] DOWN: {name}", body, mail, logf)

        # periodic heartbeat
        if hb_hours > 0:
            since = time.time() - state["_meta"].get("last_heartbeat_ts", 0)
            if since >= hb_hours * 3600:
                summary = "\n".join(cycle_lines)
                _mail("[watchdog] heartbeat - all monitored jobs",
                      f"Routine check-in.\n\n{summary}", mail, logf)
                state["_meta"]["last_heartbeat_ts"] = time.time()

        try:
            statef.parent.mkdir(parents=True, exist_ok=True)
            statef.write_text(json.dumps(state, indent=2), encoding="utf-8")
        except OSError:
            pass

        if once:
            print("\n".join(cycle_lines))
            return 0
        time.sleep(poll)


def _print_task() -> None:
    py = sys.executable
    script = REPO / "scripts" / "watchdog.py"
    print(f"""# Register the watchdog as a Scheduled Task that starts at logon and
# auto-restarts if it ever exits (survives the Claude session / PowerShell closing):

schtasks /Create /TN FoodSLM-Watchdog /SC ONLOGON /RL LIMITED /F ^
  /TR "'{py}' '{script}'"

# start it now without waiting for a logon:
schtasks /Run /TN FoodSLM-Watchdog

# check / stop / remove:
schtasks /Query /TN FoodSLM-Watchdog /V /FO LIST
schtasks /End  /TN FoodSLM-Watchdog
schtasks /Delete /TN FoodSLM-Watchdog /F
""")


def main() -> int:
    global HOST
    import socket
    HOST = socket.gethostname()

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=REPO / "configs" / "watchdog.yaml")
    ap.add_argument("--test-mail", action="store_true", help="send one test email and exit")
    ap.add_argument("--once", action="store_true", help="run every check once, print, exit")
    ap.add_argument("--print-task", action="store_true", help="print the schtasks command")
    args = ap.parse_args()

    load_env(REPO / ".env")

    if args.print_task:
        _print_task()
        return 0
    if args.test_mail:
        cfg = MailConfig()
        if not cfg.ready:
            print(f"email not configured - set {', '.join(cfg.missing())} in .env")
            return 1
        send_mail("[watchdog] test email",
                  f"If you are reading this, watchdog email works.\n"
                  f"host={HOST}  to={cfg.to}  {_now()}", cfg=cfg)
        print(f"sent to {cfg.to}")
        return 0

    return run(args.config, args.once)


if __name__ == "__main__":
    raise SystemExit(main())
