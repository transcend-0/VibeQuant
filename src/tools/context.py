"""Shared run context passed between tools."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..dsl import TaskSpec
from ..memory import MemoryStore, RunRecord


@dataclass
class RunContext:
    spec: TaskSpec
    store: MemoryStore
    run: RunRecord
    data: Optional[Dict[str, Any]] = None  # {symbol: DataFrame}
    backtest: Optional[Any] = None  # adapters BacktestOutput
    factor_panel: Optional[Any] = None  # wide DataFrame [date, symbol, <name>...]
    factor_names: List[str] = field(default_factory=list)
    factor_report: Optional[Any] = None  # factors.analysis.FactorReport
    risk_assessment: Optional[Any] = None  # RiskAssessment
    validation: Optional[Any] = None  # validation v1 dict
    report_markdown: Optional[str] = None
    gate_notes: List[str] = field(default_factory=list)
    artifacts: Dict[str, str] = field(default_factory=dict)  # name -> path
