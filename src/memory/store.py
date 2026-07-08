"""Persistent memory: run artifacts + experiment log + memory bank.

Layout (under the workspace directory):

    workspace/
    ├── runs/<run_id>/          task.yaml, result.json, report.md,
    │                           equity.csv, report.html (optional)
    ├── experiments.jsonl       one line per run — machine-readable history
    └── memory_bank/
        └── experiment_log.md   human-readable rolling log (PDF §7.2 style)

This is the "memory persistence" principle from the design report kept
to its minimal useful core: every run is reproducible (spec + data seed)
and findable later.
"""

from __future__ import annotations

import datetime as _dt
import json
import re
import secrets
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


@dataclass
class RunRecord:
    run_id: str
    directory: Path

    def path(self, name: str) -> Path:
        return self.directory / name


class MemoryStore:
    def __init__(self, workspace: Path) -> None:
        self.workspace = Path(workspace)
        self.runs_dir = self.workspace / "runs"
        self.bank_dir = self.workspace / "memory_bank"
        self.log_file = self.workspace / "experiments.jsonl"
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.bank_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------ runs
    def new_run(self, task_name: str) -> RunRecord:
        slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", task_name).strip("-")[:40] or "task"
        stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        run_id = f"{stamp}-{slug}-{secrets.token_hex(2)}"
        directory = self.runs_dir / run_id
        directory.mkdir(parents=True)
        return RunRecord(run_id=run_id, directory=directory)

    def save_artifact(self, run: RunRecord, name: str, content: str) -> Path:
        path = run.path(name)
        path.write_text(content, encoding="utf-8")
        return path

    def save_equity_curve(self, run: RunRecord, curve: pd.Series) -> Optional[Path]:
        if curve is None or curve.empty:
            return None
        path = run.path("equity.csv")
        curve.rename("equity").to_csv(path, index_label="timestamp")
        return path

    # ----------------------------------------------------- experiment log
    def append_experiment(self, entry: Dict[str, Any]) -> None:
        entry = dict(entry, logged_at=_dt.datetime.now().isoformat(timespec="seconds"))
        with self.log_file.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
        self._append_bank_line(entry)

    def _append_bank_line(self, entry: Dict[str, Any]) -> None:
        bank = self.bank_dir / "experiment_log.md"
        if not bank.exists():
            bank.write_text(
                "# Experiment Log / 实验日志\n\n"
                "| time | run | strategy | symbols | return% | sharpe | mdd% | risk |\n"
                "|------|-----|----------|---------|---------|--------|------|------|\n",
                encoding="utf-8",
            )
        metrics = entry.get("metrics") or {}

        def fmt(key: str) -> str:
            val = metrics.get(key)
            return f"{val:.2f}" if isinstance(val, (int, float)) else "-"

        line = (
            f"| {entry.get('logged_at', '')} "
            f"| {entry.get('run_id', '')} "
            f"| {entry.get('strategy', '')} "
            f"| {','.join(entry.get('symbols', []))} "
            f"| {fmt('total_return_pct')} "
            f"| {fmt('sharpe_ratio')} "
            f"| {fmt('max_drawdown_pct')} "
            f"| {'PASS' if entry.get('risk_passed') else 'FLAG'} |\n"
        )
        with bank.open("a", encoding="utf-8") as fh:
            fh.write(line)

    # ---------------------------------------------------------- queries
    def list_runs(self, limit: int = 100) -> List[Dict[str, Any]]:
        if not self.log_file.exists():
            return []
        entries: List[Dict[str, Any]] = []
        for line in self.log_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return list(reversed(entries))[:limit]

    def rename_run(self, run_id: str, new_name: str) -> bool:
        """Update the display name (task_name) of a logged run in-place."""
        if not self.log_file.exists():
            return False
        lines = self.log_file.read_text(encoding="utf-8").splitlines()
        found = False
        rewritten: List[str] = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if entry.get("run_id") == run_id:
                entry["task_name"] = new_name
                found = True
            rewritten.append(json.dumps(entry, ensure_ascii=False, default=str))
        if found:
            self.log_file.write_text("\n".join(rewritten) + "\n", encoding="utf-8")
        return found

    def delete_run(self, run_id: str) -> bool:
        """Remove a run's artifact directory and its experiment-log entry."""
        directory = self.runs_dir / run_id
        existed = directory.exists()
        if existed:
            shutil.rmtree(directory)
        if self.log_file.exists():
            lines = self.log_file.read_text(encoding="utf-8").splitlines()
            kept = [
                line
                for line in lines
                if line.strip() and json.loads(line).get("run_id") != run_id
            ]
            if len(kept) != len([l for l in lines if l.strip()]):
                existed = True
                self.log_file.write_text(
                    ("\n".join(kept) + "\n") if kept else "", encoding="utf-8"
                )
        return existed

    def load_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        directory = self.runs_dir / run_id
        result_file = directory / "result.json"
        if not result_file.exists():
            return None
        payload = json.loads(result_file.read_text(encoding="utf-8"))
        report_file = directory / "report.md"
        if report_file.exists():
            payload["report_markdown"] = report_file.read_text(encoding="utf-8")
        equity_file = directory / "equity.csv"
        if equity_file.exists():
            df = pd.read_csv(equity_file)
            payload["equity_curve"] = {
                "timestamps": df["timestamp"].astype(str).tolist(),
                "equity": df["equity"].astype(float).tolist(),
            }
        task_file = directory / "task.yaml"
        if task_file.exists():
            payload["task_yaml"] = task_file.read_text(encoding="utf-8")
        return payload
