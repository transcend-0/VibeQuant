"""Planner: TaskSpec -> ordered list of tool steps.

Deliberately transparent: the plan is a plain list the user can inspect
before execution (Plan-and-Act). Today plans are linear; the step model
leaves room for DAGs later without changing callers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from .dsl import TaskSpec


@dataclass
class PlanStep:
    tool: str
    title_en: str
    title_zh: str
    params: Dict = field(default_factory=dict)


@dataclass
class Plan:
    steps: List[PlanStep]

    def describe(self, lang: str = "en") -> str:
        lines = []
        for i, step in enumerate(self.steps, 1):
            title = step.title_zh if lang == "zh" else step.title_en
            lines.append(f"{i}. [{step.tool}] {title}")
        return "\n".join(lines)


def make_plan(spec: TaskSpec) -> Plan:
    head = [
        PlanStep(
            tool="risk_gate",
            title_en="Pre-run safety gate (mode, limits, sanity checks)",
            title_zh="运行前安全闸（模式、限额、健全性检查）",
        ),
        PlanStep(
            tool="load_data",
            title_en=(
                f"Load {spec.data.source} bars for "
                f"{', '.join(spec.data.symbols)}"
            ),
            title_zh=f"加载 {spec.data.source} 行情：{', '.join(spec.data.symbols)}",
        ),
    ]
    tail = [
        PlanStep(
            tool="memorize",
            title_en="Persist run artifacts and append to experiment memory",
            title_zh="保存运行产物并写入实验记忆",
        ),
    ]

    if spec.kind == "factor":
        n = len(spec.factor.expressions)
        return Plan(
            steps=head
            + [
                PlanStep(
                    tool="factor_compute",
                    title_en=(
                        f"Evaluate {n} factor expression(s) via akquant "
                        "factor engine"
                    ),
                    title_zh=f"通过 akquant 因子引擎计算 {n} 个因子表达式",
                ),
                PlanStep(
                    tool="factor_analyze",
                    title_en=(
                        f"IC/ICIR + {spec.factor.quantiles}-quantile layered "
                        f"returns (forward {spec.factor.forward_days}d)"
                    ),
                    title_zh=(
                        f"IC/ICIR 与 {spec.factor.quantiles} 分层收益分析"
                        f"（前瞻 {spec.factor.forward_days} 日）"
                    ),
                ),
                PlanStep(
                    tool="validate",
                    title_en="Validate: trial-deflated permutation + "
                    "sub-period consistency",
                    title_zh="验证：试验折减置换检验 + 子区间一致性",
                ),
                PlanStep(
                    tool="factor_report",
                    title_en="Generate factor research report",
                    title_zh="生成因子研究报告",
                ),
            ]
            + tail
        )

    steps = head + [
        PlanStep(
            tool="backtest",
            title_en=(
                f"Backtest strategy '{spec.strategy.name}' via akquant "
                f"(cash {spec.execution.initial_cash:,.0f})"
            ),
            title_zh=(
                f"通过 akquant 回测策略 “{spec.strategy.name}”"
                f"（初始资金 {spec.execution.initial_cash:,.0f}）"
            ),
        ),
        PlanStep(
            tool="risk_assess",
            title_en="Post-run risk assessment (drawdown, trade count, sanity)",
            title_zh="运行后风险评估（回撤、交易数、健全性）",
        ),
        PlanStep(
            tool="validate",
            title_en="Validate: equity-curve sub-period consistency",
            title_zh="验证：净值曲线子区间一致性",
        ),
        PlanStep(
            tool="report",
            title_en="Generate report (markdown/json"
            + (" + akquant html" if spec.report.html else "")
            + ")",
            title_zh="生成报告（markdown/json"
            + (" + akquant html" if spec.report.html else "")
            + "）",
        ),
    ]
    return Plan(steps=steps + tail)
