"""VibeQuant DSL: the structured task spec at the center of the pipeline.

prompt / YAML  ->  TaskSpec  ->  planner  ->  tools  ->  akquant adapter

TaskSpec is deliberately small and declarative. It is the single contract
between intent parsing, planning, execution and reporting, so every layer
can evolve independently.
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import yaml

DSL_VERSION = "0.1"

VALID_MODES = ("backtest", "paper", "live")


class DSLError(ValueError):
    """Raised when a task spec is invalid."""


@dataclass
class DataSpec:
    """Where bars come from. source: synthetic | csv | akshare."""

    source: str = "synthetic"
    universe: str = "custom"  # universe preset key (display + PIT pools)
    symbols: List[str] = field(default_factory=lambda: ["DEMO"])
    start: Optional[str] = None  # "YYYY-MM-DD"
    end: Optional[str] = None
    path: Optional[str] = None  # for csv source
    seed: int = 42  # for synthetic source
    adjust: str = "qfq"  # for akshare source


@dataclass
class StrategySpec:
    """Which strategy template to instantiate and with what params."""

    name: str = "ma_cross"
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FactorSpec:
    """Factor research settings (kind: factor).

    expressions: akquant factor expressions (Alpha101 style), optionally
    named via "Name = Expr", e.g. "Mom20 = -Delta(Close,20)/Delay(Close,20)".
    """

    expressions: List[str] = field(default_factory=list)
    universe: str = "custom"  # preset name, informational (symbols live in data.symbols)
    forward_days: int = 1  # forward-return horizon: factor predicts next-day return
    quantiles: int = 5  # number of layers in the quantile backtest
    # WorldQuant-Brain style post-processing, applied in this order:
    truncation: float = 0.0  # winsorize each date's cross-section, per tail (0=off)
    neutralization: str = "none"  # none | demean (market) | industry | zscore | rank
    decay: int = 0  # linear decay over N days (0/1 = off), smooths turnover
    # WorldQuant-style liquidity constraints on the LS portfolio (0 = off):
    # both are fractions of each name's ADV20 (20-day average dollar volume),
    # converted to weight caps against the book size (execution.initial_cash)
    max_position: float = 0.0  # position notional <= max_position * ADV20
    max_trade: float = 0.0  # traded notional per rebalance <= max_trade * ADV20


@dataclass
class RiskSpec:
    """Declarative risk limits, enforced pre-run (gate) and in-engine."""

    max_position_pct: float = 0.95  # max fraction of equity in one symbol
    max_drawdown_pct: float = 30.0  # post-run: flag if exceeded
    max_order_value: Optional[float] = None
    min_trades_warn: int = 5  # post-run: overfit / low-signal warning


@dataclass
class ExecutionSpec:
    """Execution settings. Only backtest is executable; paper/live are gated."""

    mode: str = "backtest"
    initial_cash: float = 1_000_000.0
    commission_rate: float = 0.0003
    stamp_tax_rate: float = 0.0
    slippage_bps: float = 1.0
    t_plus_one: bool = False
    confirm_live: bool = False  # must be true (and mode impl exists) to go live


@dataclass
class ReportSpec:
    """What artifacts to produce."""

    formats: List[str] = field(default_factory=lambda: ["markdown", "json"])
    html: bool = True  # akquant native HTML report (strategy runs)
    benchmark: Optional[str] = None  # symbol to compare against
    language: str = "en"  # en | zh, for the generated narrative


@dataclass
class TaskSpec:
    """The full unit of work."""

    name: str = "untitled-task"
    intent: str = ""  # original natural-language intent, kept for the record
    kind: str = "strategy"  # strategy | factor
    data: DataSpec = field(default_factory=DataSpec)
    strategy: StrategySpec = field(default_factory=StrategySpec)
    factor: FactorSpec = field(default_factory=FactorSpec)
    risk: RiskSpec = field(default_factory=RiskSpec)
    execution: ExecutionSpec = field(default_factory=ExecutionSpec)
    report: ReportSpec = field(default_factory=ReportSpec)
    notes: List[str] = field(default_factory=list)  # parser/planner annotations
    version: str = DSL_VERSION

    # ---------------------------------------------------------------- io
    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.to_dict(), allow_unicode=True, sort_keys=False)

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "TaskSpec":
        if not isinstance(raw, dict):
            raise DSLError("task spec must be a mapping")
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = set(raw) - known
        if unknown:
            raise DSLError(f"unknown top-level keys: {sorted(unknown)}")

        def build(klass, key):
            sub = raw.get(key) or {}
            if not isinstance(sub, dict):
                raise DSLError(f"'{key}' must be a mapping")
            names = {f.name for f in dataclasses.fields(klass)}
            bad = set(sub) - names
            if bad:
                raise DSLError(f"unknown keys under '{key}': {sorted(bad)}")
            return klass(**sub)

        spec = cls(
            name=str(raw.get("name", "untitled-task")),
            intent=str(raw.get("intent", "")),
            kind=str(raw.get("kind", "strategy")),
            data=build(DataSpec, "data"),
            strategy=build(StrategySpec, "strategy"),
            factor=build(FactorSpec, "factor"),
            risk=build(RiskSpec, "risk"),
            execution=build(ExecutionSpec, "execution"),
            report=build(ReportSpec, "report"),
            notes=list(raw.get("notes", []) or []),
            version=str(raw.get("version", DSL_VERSION)),
        )
        spec.validate()
        return spec

    @classmethod
    def from_yaml(cls, text: str) -> "TaskSpec":
        try:
            raw = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise DSLError(f"invalid YAML: {exc}") from exc
        return cls.from_dict(raw or {})

    # ---------------------------------------------------------- validation
    def validate(self) -> None:
        # YAML turns bare numeric codes (510300) into ints — normalize early.
        # NOTE: quote leading-zero codes in YAML ("000333"); unquoted they are
        # parsed as octal. We zero-pad short digit strings for cn sources as a
        # best effort, but octal mangling is not reversible.
        self.data.symbols = [str(s) for s in self.data.symbols]
        if self.data.source in ("etf", "stock", "index"):
            self.data.symbols = [
                s.zfill(6) if s.isdigit() and len(s) < 6 else s
                for s in self.data.symbols
            ]
        self.factor.expressions = [str(e) for e in self.factor.expressions]
        if self.kind not in ("strategy", "factor"):
            raise DSLError(f"kind must be 'strategy' or 'factor', got {self.kind!r}")
        if self.kind == "factor":
            if not self.factor.expressions:
                raise DSLError("kind=factor requires factor.expressions")
            if len(self.data.symbols) < 2:
                raise DSLError(
                    "factor research needs >= 2 symbols for cross-sectional "
                    "analysis (5+ recommended)"
                )
            if self.factor.forward_days < 1:
                raise DSLError("factor.forward_days must be >= 1")
            if not 2 <= self.factor.quantiles <= 10:
                raise DSLError("factor.quantiles must be in [2, 10]")
            if not 0.0 <= self.factor.truncation < 0.5:
                raise DSLError("factor.truncation must be in [0, 0.5)")
            if self.factor.neutralization not in (
                "none", "demean", "industry", "zscore", "rank"
            ):
                raise DSLError(
                    "factor.neutralization must be "
                    "none|demean|industry|zscore|rank"
                )
            if not 0 <= self.factor.decay <= 60:
                raise DSLError("factor.decay must be in [0, 60]")
            if not 0.0 <= self.factor.max_position <= 1.0:
                raise DSLError("factor.max_position must be in [0, 1]")
            if not 0.0 <= self.factor.max_trade <= 1.0:
                raise DSLError("factor.max_trade must be in [0, 1]")
        if self.execution.mode not in VALID_MODES:
            raise DSLError(
                f"execution.mode must be one of {VALID_MODES}, "
                f"got {self.execution.mode!r}"
            )
        if self.execution.initial_cash <= 0:
            raise DSLError("execution.initial_cash must be positive")
        if not self.data.symbols:
            raise DSLError("data.symbols must not be empty")
        if self.data.source not in (
            "synthetic", "csv", "etf", "stock", "index", "hk", "us", "crypto",
            "akshare"
        ):
            raise DSLError(f"unknown data.source {self.data.source!r}")
        if self.data.source == "csv" and not self.data.path:
            raise DSLError("data.source=csv requires data.path")
        for key in ("start", "end"):
            val = getattr(self.data, key)
            if val is not None:
                try:
                    _dt.date.fromisoformat(str(val))
                except ValueError as exc:
                    raise DSLError(f"data.{key} must be YYYY-MM-DD") from exc
        if self.data.start and self.data.end and self.data.start >= self.data.end:
            raise DSLError("data.start must be before data.end")
        if not 0 < self.risk.max_position_pct <= 1:
            raise DSLError("risk.max_position_pct must be in (0, 1]")
        if self.report.language not in ("en", "zh"):
            raise DSLError("report.language must be 'en' or 'zh'")
