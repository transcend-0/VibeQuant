"""Runner: execute a TaskSpec end to end.

    spec -> plan -> tools (risk_gate, load_data, backtest,
                           risk_assess, report, memorize) -> RunResult
"""

from __future__ import annotations

import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import workspace_dir
from .dsl import TaskSpec
from .memory import MemoryStore
from .planner import Plan, make_plan
from .tools import get_tool
from .tools.context import RunContext


@dataclass
class RunResult:
    run_id: str
    ok: bool
    spec: TaskSpec
    plan: Plan
    kind: str = "strategy"
    metrics: Dict[str, Optional[float]] = field(default_factory=dict)
    num_trades: int = 0
    factor: Dict[str, Any] = field(default_factory=dict)  # FactorReport payload
    risk: Dict[str, Any] = field(default_factory=dict)
    validation: Dict[str, Any] = field(default_factory=dict)
    report_markdown: str = ""
    artifacts: Dict[str, str] = field(default_factory=dict)
    error: Optional[str] = None
    failed_step: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "ok": self.ok,
            "kind": self.kind,
            "metrics": self.metrics,
            "num_trades": self.num_trades,
            "factor": self.factor,
            "risk": self.risk,
            "validation": self.validation,
            "artifacts": self.artifacts,
            "error": self.error,
            "failed_step": self.failed_step,
            # the clean Python/expressions, not the YAML-escaped form
            # (task_yaml's params.source often renders as a quoted,
            # \n-escaped one-liner) -- for display in the UI's code panel.
            "strategy_source": (
                self.spec.strategy.params.get("source", "")
                if self.kind == "strategy" else ""
            ),
            "factor_expressions": (
                list(self.spec.factor.expressions) if self.kind == "factor" else []
            ),
        }


def run_task(spec: TaskSpec, workspace: Optional[Path] = None) -> RunResult:
    spec.validate()
    store = MemoryStore(workspace or workspace_dir())
    run = store.new_run(spec.name)
    plan = make_plan(spec)
    ctx = RunContext(spec=spec, store=store, run=run)

    for step in plan.steps:
        try:
            get_tool(step.tool)(ctx)
        except Exception as exc:
            # persist the failure so it is debuggable later
            store.save_artifact(run, "task.yaml", spec.to_yaml())
            store.save_artifact(
                run,
                "error.txt",
                f"step: {step.tool}\n{exc}\n\n{traceback.format_exc()}",
            )
            return RunResult(
                run_id=run.run_id,
                ok=False,
                spec=spec,
                plan=plan,
                kind=spec.kind,
                error=str(exc),
                failed_step=step.tool,
                artifacts=dict(ctx.artifacts),
            )

    assert ctx.risk_assessment is not None
    result = RunResult(
        run_id=run.run_id,
        ok=True,
        spec=spec,
        plan=plan,
        kind=spec.kind,
        risk=ctx.risk_assessment.to_dict(),
        validation=ctx.validation or {},
        report_markdown=ctx.report_markdown or "",
        artifacts=dict(ctx.artifacts),
    )
    if spec.kind == "strategy":
        assert ctx.backtest is not None
        result.metrics = ctx.backtest.metrics
        result.num_trades = ctx.backtest.num_trades
    else:
        assert ctx.factor_report is not None
        result.factor = ctx.factor_report.to_dict()
    return result
