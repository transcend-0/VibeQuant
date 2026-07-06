"""Risk layer: pre-run gate (safe by default) and post-run assessment.

The gate runs BEFORE any engine call. Live/paper execution is refused
outright — VibeQuant currently implements research workflows only, and
turning on real execution must be a deliberate, code-level decision,
not a config flag slipped into a YAML file.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .dsl import TaskSpec


class RiskGateError(RuntimeError):
    """Raised when a task is refused by the pre-run safety gate."""


@dataclass
class RiskAssessment:
    passed: bool
    flags: List[str] = field(default_factory=list)  # hard problems
    warnings: List[str] = field(default_factory=list)  # soft advice

    def to_dict(self) -> Dict:
        return {
            "passed": self.passed,
            "flags": self.flags,
            "warnings": self.warnings,
        }


def pre_run_gate(spec: TaskSpec) -> List[str]:
    """Validate a task before execution. Returns advisory notes; raises on refusal."""
    mode = spec.execution.mode
    if mode == "live":
        raise RiskGateError(
            "live trading is disabled in VibeQuant. Backtest first; live "
            "execution requires a broker adapter, human approval flow and "
            "explicit enablement in code (see docs/ARCHITECTURE.md)."
        )
    if mode == "paper":
        raise RiskGateError(
            "paper trading is not implemented yet; use mode: backtest."
        )

    notes: List[str] = []
    if spec.execution.initial_cash > 1e9:
        raise RiskGateError("initial_cash exceeds sanity bound (1e9)")
    if spec.risk.max_position_pct > 0.95:
        notes.append(
            "max_position_pct > 0.95 leaves no cash buffer; consider <= 0.95"
        )
    if spec.execution.commission_rate == 0 and spec.execution.slippage_bps == 0:
        notes.append(
            "zero commission and zero slippage: results will be optimistic"
        )
    if len(spec.data.symbols) > 2000:
        raise RiskGateError("more than 2000 symbols in one task; split the task")
    if len(spec.data.symbols) > 60:
        notes.append(
            f"{len(spec.data.symbols)} symbols: first run downloads each one "
            "(~1s/symbol, cached afterwards) — expect a wait"
        )
    return notes


def post_run_assess(
    spec: TaskSpec,
    metrics: Dict[str, Optional[float]],
    num_trades: int,
    total_bars: int,
) -> RiskAssessment:
    """Turn raw metrics into pass/flag/warn signals for the report."""
    flags: List[str] = []
    warnings: List[str] = []

    mdd = metrics.get("max_drawdown_pct")
    if mdd is not None and abs(mdd) > spec.risk.max_drawdown_pct:
        flags.append(
            f"max drawdown {abs(mdd):.1f}% exceeds limit "
            f"{spec.risk.max_drawdown_pct:.1f}%"
        )

    if num_trades < spec.risk.min_trades_warn:
        warnings.append(
            f"only {num_trades} trades — statistics are not meaningful; "
            "treat metrics as anecdotal"
        )

    sharpe = metrics.get("sharpe_ratio")
    if sharpe is not None and sharpe > 3.0:
        warnings.append(
            f"sharpe {sharpe:.2f} is suspiciously high — check for "
            "look-ahead bias, unrealistic costs or overfitting"
        )

    if total_bars and num_trades > total_bars * 0.5:
        warnings.append(
            "trade count exceeds half the bar count — likely over-trading; "
            "costs may dominate"
        )

    return RiskAssessment(passed=not flags, flags=flags, warnings=warnings)
