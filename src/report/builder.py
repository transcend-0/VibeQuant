"""Natural-language backtest report (Markdown), bilingual EN/ZH.

Rule-based narrative — no LLM required — so the minimal loop works
offline. The structure mirrors the design report's §5.5.3: strategy
recap, metric read-out, risk verdict, caveats.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from ..dsl import TaskSpec
from ..risk import RiskAssessment

_L = {
    "en": {
        "title": "Backtest Report",
        "task": "Task",
        "strategy": "Strategy",
        "symbols": "Symbols",
        "period": "Period",
        "mode": "Mode",
        "metrics": "Core Metrics",
        "metric": "Metric",
        "value": "Value",
        "total_return_pct": "Total return (%)",
        "annualized_return": "Annualized return",
        "sharpe_ratio": "Sharpe ratio",
        "sortino_ratio": "Sortino ratio",
        "max_drawdown_pct": "Max drawdown (%)",
        "win_rate": "Win rate (%)",
        "num_trades": "Number of trades",
        "risk": "Risk Assessment",
        "risk_pass": "PASSED — no hard risk flags.",
        "risk_fail": "FLAGGED — review before trusting this result:",
        "warnings": "Warnings",
        "verdict": "Summary",
        "notes": "Notes",
        "na": "n/a",
    },
    "zh": {
        "title": "回测报告",
        "task": "任务",
        "strategy": "策略",
        "symbols": "标的",
        "period": "区间",
        "mode": "模式",
        "metrics": "核心指标",
        "metric": "指标",
        "value": "数值",
        "total_return_pct": "总收益率 (%)",
        "annualized_return": "年化收益率",
        "sharpe_ratio": "夏普比率",
        "sortino_ratio": "索提诺比率",
        "max_drawdown_pct": "最大回撤 (%)",
        "win_rate": "胜率 (%)",
        "num_trades": "交易次数",
        "risk": "风险评估",
        "risk_pass": "通过 —— 无硬性风险标记。",
        "risk_fail": "存在风险标记 —— 采信结果前请先复核：",
        "warnings": "提示",
        "verdict": "结论",
        "notes": "备注",
        "na": "无",
    },
}

_METRIC_KEYS = [
    "total_return_pct",
    "annualized_return",
    "sharpe_ratio",
    "sortino_ratio",
    "max_drawdown_pct",
    "win_rate",
]


def _fmt(value: Optional[float]) -> str:
    if value is None:
        return "-"
    return f"{value:.4f}" if abs(value) < 10 else f"{value:.2f}"


def _verdict(lang: str, metrics: Dict[str, Optional[float]], risk: RiskAssessment) -> str:
    ret = metrics.get("total_return_pct")
    sharpe = metrics.get("sharpe_ratio")
    mdd = metrics.get("max_drawdown_pct")
    if lang == "zh":
        parts = []
        if ret is not None:
            parts.append(f"区间总收益 {ret:.2f}%")
        if sharpe is not None:
            parts.append(f"夏普 {sharpe:.2f}")
        if mdd is not None:
            parts.append(f"最大回撤 {abs(mdd):.2f}%")
        text = "，".join(parts) if parts else "指标不足"
        tail = "风险检查通过。" if risk.passed else "存在风险标记，谨慎采信。"
        return f"{text}。{tail}"
    parts = []
    if ret is not None:
        parts.append(f"total return {ret:.2f}%")
    if sharpe is not None:
        parts.append(f"Sharpe {sharpe:.2f}")
    if mdd is not None:
        parts.append(f"max drawdown {abs(mdd):.2f}%")
    text = ", ".join(parts) if parts else "insufficient metrics"
    tail = (
        "Risk checks passed."
        if risk.passed
        else "Risk flags raised — treat with caution."
    )
    return f"{text.capitalize()}. {tail}"


def build_markdown_report(
    spec: TaskSpec,
    metrics: Dict[str, Optional[float]],
    num_trades: int,
    risk: RiskAssessment,
    run_id: str,
    notes: Optional[List[str]] = None,
    validation: Optional[Dict] = None,
) -> str:
    lang = spec.report.language if spec.report.language in _L else "en"
    t = _L[lang]
    period = f"{spec.data.start or '...'} ~ {spec.data.end or '...'}"
    params = ", ".join(f"{k}={v}" for k, v in sorted(spec.strategy.params.items()))
    strategy_line = spec.strategy.name + (f" ({params})" if params else "")

    lines = [
        f"# {t['title']}: {spec.name}",
        "",
        f"- **{t['task']}**: `{run_id}`",
        f"- **{t['strategy']}**: {strategy_line}",
        f"- **{t['symbols']}**: {', '.join(spec.data.symbols)} "
        f"({spec.data.source})",
        f"- **{t['period']}**: {period}",
        f"- **{t['mode']}**: {spec.execution.mode}",
        "",
        f"## {t['metrics']}",
        "",
        f"| {t['metric']} | {t['value']} |",
        "|---|---|",
    ]
    for key in _METRIC_KEYS:
        lines.append(f"| {t[key]} | {_fmt(metrics.get(key))} |")
    lines.append(f"| {t['num_trades']} | {num_trades} |")

    lines += ["", f"## {t['risk']}", ""]
    if risk.passed:
        lines.append(t["risk_pass"])
    else:
        lines.append(t["risk_fail"])
        lines += [f"- ❌ {flag}" for flag in risk.flags]
    if risk.warnings:
        lines += ["", f"**{t['warnings']}:**"]
        lines += [f"- ⚠️ {warning}" for warning in risk.warnings]

    lines += ["", f"## {t['verdict']}", "", _verdict(lang, metrics, risk)]
    lines += _validation_section(validation, lang)

    all_notes = (notes or []) + (spec.notes or [])
    if all_notes:
        lines += ["", f"## {t['notes']}", ""]
        lines += [f"- {note}" for note in all_notes]

    return "\n".join(lines) + "\n"


# ------------------------------------------------------- factor report
_LF = {
    "en": {
        "title": "Factor Research Report",
        "universe": "Universe",
        "horizon": "Forward horizon",
        "days": "days",
        "quantiles": "Quantiles",
        "factor": "Factor",
        "verdict_strong": "shows a promising signal",
        "verdict_weak": "shows a weak/unstable signal",
        "verdict_none": "shows no meaningful signal",
        "ls": "long-short",
        "interp": (
            "Rule of thumb: |IC mean| > 0.03 with |ICIR| > 0.5 is worth deeper "
            "study; positive rate near 0.5 means the sign is unstable."
        ),
    },
    "zh": {
        "title": "因子研究报告",
        "universe": "标的池",
        "horizon": "前瞻窗口",
        "days": "日",
        "quantiles": "分层数",
        "factor": "因子",
        "verdict_strong": "信号较为显著，值得深入研究",
        "verdict_weak": "信号偏弱或不稳定",
        "verdict_none": "未发现有意义的信号",
        "ls": "多空组合",
        "interp": (
            "经验法则：|IC均值| > 0.03 且 |ICIR| > 0.5 值得深入研究；"
            "IC 正率接近 0.5 说明方向不稳定。"
        ),
    },
}


def _f(value, pct: bool = False) -> str:
    if value is None:
        return "-"
    if pct:
        return f"{value * 100:.2f}%"
    return f"{value:.4f}"


def build_factor_markdown_report(
    spec: TaskSpec,
    factor_report,  # factors.analysis.FactorReport
    run_id: str,
    notes: Optional[List[str]] = None,
    validation: Optional[Dict] = None,
) -> str:
    lang = spec.report.language if spec.report.language in _LF else "en"
    t, tt = _LF[lang], _L[lang]
    period = f"{spec.data.start or '...'} ~ {spec.data.end or '...'}"

    lines = [
        f"# {t['title']}: {spec.name}",
        "",
        f"- **{tt['task']}**: `{run_id}`",
        f"- **{t['universe']}**: {', '.join(spec.data.symbols)} ({spec.data.source})",
        f"- **{tt['period']}**: {period}",
        f"- **{t['horizon']}**: {spec.factor.forward_days} {t['days']}"
        f" · **{t['quantiles']}**: {spec.factor.quantiles}",
        "",
        "| " + t["factor"] + " | IC mean | IC std | ICIR | IC>0 | t-stat "
        "| LS ret | Q_top | Q_bottom | rank AC | n |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for s in factor_report.stats:
        lines.append(
            f"| {s.name} | {_f(s.ic_mean)} | {_f(s.ic_std)} | {_f(s.icir)} "
            f"| {_f(s.ic_positive_rate)} | {_f(s.ic_t_stat)} "
            f"| {_f(s.long_short_total_return, pct=True)} "
            f"| {_f(s.top_layer_total_return, pct=True)} "
            f"| {_f(s.bottom_layer_total_return, pct=True)} "
            f"| {_f(s.rank_autocorr)} | {s.n_periods} |"
        )

    lines += ["", f"## {tt['verdict']}", ""]
    for s in factor_report.stats:
        if s.ic_mean is None or s.icir is None:
            verdict = t["verdict_none"]
        elif abs(s.ic_mean) > 0.03 and abs(s.icir) > 0.5:
            verdict = t["verdict_strong"]
        elif abs(s.ic_mean) > 0.01:
            verdict = t["verdict_weak"]
        else:
            verdict = t["verdict_none"]
        direction = ""
        if s.ic_mean is not None and s.ic_mean < 0:
            direction = (
                "（负向因子：低值组合表现更好）"
                if lang == "zh"
                else " (negative factor: low values outperform)"
            )
        lines.append(f"- **{s.name}** {verdict}{direction}")
    lines += ["", t["interp"]]

    lines += _validation_section(validation, lang)

    if factor_report.warnings:
        lines += ["", f"**{tt['warnings']}:**"]
        lines += [f"- ⚠️ {w}" for w in factor_report.warnings]

    all_notes = (notes or []) + (spec.notes or [])
    if all_notes:
        lines += ["", f"## {tt['notes']}", ""]
        lines += [f"- {note}" for note in all_notes]

    return "\n".join(lines) + "\n"


# ---------------------------------------------------- validation section
_LV = {
    "en": {
        "title": "Validation",
        "risk": {"low": "overfit risk: LOW", "medium": "overfit risk: MEDIUM",
                 "high": "overfit risk: HIGH"},
        "perm": "permutation p (raw)",
        "trials": "trials on this universe",
        "deflated": "deflated threshold",
        "sig": "significant after deflation",
        "consistency": "sub-period sign consistency",
        "windows": "equity windows positive",
        "note": ("Sub-period checks measure regime robustness, not true "
                 "out-of-sample; only future data is uncontaminated."),
    },
    "zh": {
        "title": "验证",
        "risk": {"low": "过拟合风险：低", "medium": "过拟合风险：中",
                 "high": "过拟合风险：高"},
        "perm": "置换检验 p（原始）",
        "trials": "该池累计尝试次数",
        "deflated": "折减后阈值",
        "sig": "折减后仍显著",
        "consistency": "子区间同号率",
        "windows": "净值窗口为正比例",
        "note": "子区间检验衡量区制稳健性而非真样本外；只有未来数据不被污染。",
    },
}


def _validation_section(validation, lang: str) -> List[str]:
    if not validation:
        return []
    t = _LV[lang if lang in _LV else "en"]
    risk = validation.get("overfit_risk", "high")
    lines = ["", f"## {t['title']}", "", f"**{t['risk'].get(risk, risk)}**", ""]
    if validation.get("kind") == "factor":
        for name, v in (validation.get("per_factor") or {}).items():
            perm = v.get("permutation") or {}
            cons = v.get("consistency") or {}
            lines.append(
                f"- **{name}**: {t['perm']}={perm.get('p_raw', '-')}, "
                f"{t['trials']}={v.get('trial_count', '-')}, "
                f"{t['deflated']}={v.get('alpha_deflated', '-')}, "
                f"{t['sig']}={'✓' if v.get('significant_deflated') else '✗'}, "
                f"{t['consistency']}={cons.get('sign_consistency', '-')}"
            )
    else:
        lines.append(
            f"- {t['windows']}: {validation.get('positive_fraction', '-')} "
            f"({len(validation.get('windows') or [])} windows)"
        )
    lines += ["", f"_{t['note']}_"]
    return lines
