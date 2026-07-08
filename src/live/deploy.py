"""Simple live deployment: daily post-close signal generation + email.

A deployment wraps a *strategy* task. Each trading day after the A-share
close (default 16:30 Asia/Shanghai), the scheduler:

    1. fetches bars up to today for the task's symbols,
    2. replays the strategy's signal function over the full history to
       derive the current model position (deterministic, stateless),
    3. emits next-day signals (BUY / SELL / HOLD / STAY FLAT) per symbol,
    4. emails them (config/email.yaml) and archives them under
       workspace/signals/.

Deliberately NOT order execution: no broker, no money moves. This is the
maximum automation the risk gate allows — a human reads the email and
decides. Registry lives in workspace/deployments.json.
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import secrets
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

import yaml

from .. import data as data_mod
from ..config import config_dir, workspace_dir
from ..dsl import DSLError, TaskSpec

logger = logging.getLogger(__name__)

TZ = ZoneInfo("Asia/Shanghai")


class DeployError(RuntimeError):
    pass


@dataclass
class Deployment:
    id: str
    name: str
    task_yaml: str
    email_to: str = ""  # empty -> file-only delivery
    run_at: str = "16:30"  # Asia/Shanghai, weekdays
    enabled: bool = True
    created_at: str = ""
    last_run_date: str = ""  # "YYYY-MM-DD" of last successful run
    last_signals: List[Dict[str, Any]] = field(default_factory=list)
    last_status: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _registry_file() -> Path:
    return workspace_dir() / "deployments.json"


def _signals_dir() -> Path:
    path = workspace_dir() / "signals"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _load_all() -> List[Deployment]:
    path = _registry_file()
    if not path.exists():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    out = []
    for item in raw:
        known = {f for f in Deployment.__dataclass_fields__}
        out.append(Deployment(**{k: v for k, v in item.items() if k in known}))
    return out


def _save_all(deployments: List[Deployment]) -> None:
    _registry_file().write_text(
        json.dumps([d.to_dict() for d in deployments], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# ------------------------------------------------------------------ CRUD
def create_deployment(
    task_yaml: str, email_to: str = "", run_at: str = "16:30", name: str = ""
) -> Deployment:
    try:
        spec = TaskSpec.from_yaml(task_yaml)
    except DSLError as exc:
        raise DeployError(f"invalid task: {exc}") from exc
    if spec.kind != "strategy":
        raise DeployError("only strategy tasks can be deployed for daily signals")
    if spec.data.source not in ("etf", "akshare"):
        raise DeployError(
            "deployment needs a real data source (data.source: etf); "
            f"got {spec.data.source!r}"
        )
    from ..strategy_source import StrategySourceError, validate_strategy_params

    try:
        validate_strategy_params(spec.strategy.params)
    except StrategySourceError as exc:
        raise DeployError(f"invalid strategy: {exc}") from exc
    if not _valid_time(run_at):
        raise DeployError("run_at must be HH:MM (24h)")

    dep = Deployment(
        id=f"dep-{_dt.datetime.now(TZ).strftime('%Y%m%d')}-{secrets.token_hex(3)}",
        name=name or spec.name,
        task_yaml=task_yaml,
        email_to=email_to.strip(),
        run_at=run_at,
        created_at=_dt.datetime.now(TZ).isoformat(timespec="seconds"),
    )
    deployments = _load_all()
    deployments.append(dep)
    _save_all(deployments)
    return dep


def list_deployments() -> List[Deployment]:
    return _load_all()


def delete_deployment(dep_id: str) -> bool:
    deployments = _load_all()
    kept = [d for d in deployments if d.id != dep_id]
    if len(kept) == len(deployments):
        return False
    _save_all(kept)
    return True


def set_enabled(dep_id: str, enabled: bool) -> bool:
    deployments = _load_all()
    for dep in deployments:
        if dep.id == dep_id:
            dep.enabled = enabled
            _save_all(deployments)
            return True
    return False


def _valid_time(value: str) -> bool:
    try:
        _dt.time.fromisoformat(value)
        return True
    except ValueError:
        return False


# ------------------------------------------------------------ signal run
def compute_signals(spec: TaskSpec) -> List[Dict[str, Any]]:
    """Run the real backtest through today; report the current stance per
    symbol from the resulting position history (last bar vs. the one
    before it), so deployments see exactly what a live run would hold --
    no separate replay logic to keep in sync with the adapter."""
    from ..adapters.akquant_engine import run_backtest

    data_spec = spec.data
    data_spec.end = _dt.date.today().isoformat()
    frames = data_mod.load(data_spec)

    output = run_backtest(spec, frames)
    positions = output.raw.positions
    last = positions.iloc[-1].to_dict() if len(positions) else {}
    prev = positions.iloc[-2].to_dict() if len(positions) > 1 else {}

    signals = []
    for symbol, df in frames.items():
        pos_now = float(last.get(symbol, 0.0) or 0.0)
        pos_prev = float(prev.get(symbol, 0.0) or 0.0)
        if pos_now > 0 and pos_prev <= 0:
            action = "BUY"
        elif pos_now <= 0 and pos_prev > 0:
            action = "SELL"
        elif pos_now > 0:
            action = "HOLD"
        else:
            action = "STAY_FLAT"
        signals.append(
            {
                "symbol": symbol,
                "action": action,
                "position": pos_now,
                "last_close": round(float(df["close"].iloc[-1]), 4),
                "last_bar_date": str(pd_date(df)),
                "bars": len(df),
            }
        )
    return signals


def pd_date(df) -> object:
    return df["date"].iloc[-1].date()


def _format_report(dep: Deployment, spec: TaskSpec, signals: List[Dict]) -> str:
    now = _dt.datetime.now(TZ).strftime("%Y-%m-%d %H:%M %Z")
    lines = [
        f"VibeQuant daily signals / 每日信号 — {dep.name}",
        f"generated: {now}",
        f"strategy: {spec.strategy.name} {spec.strategy.params}",
        "",
        f"{'symbol':<12} {'action':<10} {'close':>10}  last bar",
    ]
    for s in signals:
        lines.append(
            f"{s['symbol']:<12} {s['action']:<10} {s['last_close']:>10}  "
            f"{s['last_bar_date']}"
        )
    lines += [
        "",
        "Actions apply to the NEXT trading session. Research signals only — "
        "not investment advice; no orders are placed automatically.",
        "信号针对下一交易日，仅供研究参考，系统不会自动下单。",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------- email
def load_email_config() -> Optional[Dict[str, Any]]:
    path = config_dir() / "email.yaml"
    if not path.exists():
        return None
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError:
        return None
    required = ("smtp_host", "username", "password", "from_addr")
    if not all(raw.get(k) for k in required):
        return None
    raw.setdefault("smtp_port", 465)
    raw.setdefault("use_ssl", True)
    return raw


def send_email(subject: str, body: str, to_addr: str) -> None:
    conf = load_email_config()
    if conf is None:
        raise DeployError(
            "email not configured — create config/email.yaml "
            "(see config/email.example.yaml)"
        )
    import smtplib
    from email.header import Header
    from email.mime.text import MIMEText

    msg = MIMEText(body, "plain", "utf-8")
    msg["Subject"] = Header(subject, "utf-8")
    msg["From"] = conf["from_addr"]
    msg["To"] = to_addr

    port = int(conf["smtp_port"])
    if conf.get("use_ssl", True):
        server = smtplib.SMTP_SSL(conf["smtp_host"], port, timeout=30)
    else:
        server = smtplib.SMTP(conf["smtp_host"], port, timeout=30)
        server.starttls()
    try:
        server.login(conf["username"], conf["password"])
        server.sendmail(conf["from_addr"], [to_addr], msg.as_string())
    finally:
        server.quit()


# ------------------------------------------------------------- execution
def run_deployment(dep_id: str, force: bool = False) -> Dict[str, Any]:
    """Run one deployment now. Returns the signal payload."""
    deployments = _load_all()
    dep = next((d for d in deployments if d.id == dep_id), None)
    if dep is None:
        raise DeployError(f"deployment {dep_id} not found")

    today = _dt.datetime.now(TZ).date().isoformat()
    if not force and dep.last_run_date == today:
        return {"skipped": True, "reason": f"already ran on {today}"}

    spec = TaskSpec.from_yaml(dep.task_yaml)
    signals = compute_signals(spec)
    report = _format_report(dep, spec, signals)

    archive = _signals_dir() / f"{today}_{dep.id}.txt"
    archive.write_text(report, encoding="utf-8")

    status = "signals archived"
    if dep.email_to:
        try:
            send_email(
                f"[VibeQuant] {dep.name} signals {today}", report, dep.email_to
            )
            status = f"emailed to {dep.email_to}"
        except DeployError as exc:
            status = f"email failed: {exc}"
        except Exception as exc:  # smtp errors
            status = f"email failed: {exc}"

    dep.last_run_date = today
    dep.last_signals = signals
    dep.last_status = status
    _save_all(deployments)
    logger.info("deployment %s ran: %s", dep.id, status)
    return {
        "skipped": False,
        "deployment": dep.id,
        "signals": signals,
        "status": status,
        "archive": str(archive),
        "report": report,
    }


def run_due_deployments(now: Optional[_dt.datetime] = None) -> List[Dict[str, Any]]:
    """Run every enabled deployment whose scheduled time has passed today.

    Called by the scheduler thread once a minute. Weekdays only (A-share
    trading calendar approximated by Mon-Fri; a holiday run just re-sends
    the latest available signals, which is harmless).
    """
    now = now or _dt.datetime.now(TZ)
    if now.weekday() >= 5:
        return []
    results = []
    today = now.date().isoformat()
    for dep in _load_all():
        if not dep.enabled or dep.last_run_date == today:
            continue
        run_time = _dt.time.fromisoformat(dep.run_at)
        if now.time() >= run_time:
            try:
                results.append(run_deployment(dep.id))
            except Exception as exc:
                logger.error("deployment %s failed: %s", dep.id, exc)
                results.append({"deployment": dep.id, "error": str(exc)})
    return results
