"""Agent self-iteration: reflect on the current result, revise, run once.

The autonomous counterpart to the manual refine loop. Given the task's
current result (already known to the caller -- e.g. the WebUI's last run
-- so it is NOT re-run here) the agent:

    1. asks the refine engine for ONE revision aimed at the objective
       (factor: |ICIR|; strategy: Sharpe), grounded in the current result
       or, if the current result failed, in its error,
    2. runs that ONE revision through the normal pipeline (a persisted,
       reproducible run in the experiment history).

Guardrails: the candidate passes DSL + expression validation (inside
refine_task), and the universe/period/kind are pinned so the agent can't
"improve" results by changing the data.

This does exactly one new backtest per call. An earlier version re-ran
the current (unchanged) task as its own "round 1" before revising it,
silently doubling the number of backtests behind a single button click.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from ..dsl import TaskSpec
from ..llm import LLMError, get_client
from ..runner import run_task
from .refine import refine_task

logger = logging.getLogger(__name__)


def _objective(result) -> Optional[float]:
    """Single scalar to maximize. None when the run produced nothing."""
    if result.kind == "factor":
        stats = (result.factor or {}).get("stats") or []
        icirs = [abs(s["icir"]) for s in stats if s.get("icir") is not None]
        return max(icirs) if icirs else None
    sharpe = (result.metrics or {}).get("sharpe_ratio")
    return sharpe


def _objective_from_summary(kind: str, result_summary: Optional[Dict[str, Any]]) -> Optional[float]:
    """Same objective, computed from a caller-supplied summary instead of a
    freshly run result -- so the caller's already-known current result can
    seed the reflection without re-running it."""
    if not result_summary or result_summary.get("ok") is False:
        return None
    if kind == "factor":
        stats = result_summary.get("factor_stats") or []
        icirs = [abs(s["icir"]) for s in stats if s.get("icir") is not None]
        return max(icirs) if icirs else None
    metrics = result_summary.get("metrics") or {}
    return metrics.get("sharpe_ratio")


def _objective_name(kind: str) -> str:
    return "|ICIR|" if kind == "factor" else "Sharpe"


def _auto_question(kind: str, language: str, score: Optional[float], failed: bool) -> str:
    obj = _objective_name(kind)
    shown = "n/a" if score is None else f"{score:.3f}"
    if failed:
        return (
            "上一轮回测运行失败（见附带的报错信息），请先修复这个 bug，"
            "使其能够正常回测；不得更改 data 部分（标的、时间区间、数据源）"
            "和 kind。" if language == "zh" else
            "The previous run failed (see the attached error). Fix the bug "
            "so it backtests successfully; do not change the data section "
            "(symbols, period, source) or the kind."
        )
    if language == "zh":
        return (
            f"当前目标 {obj} = {shown}。请提出一个改进方案以提高 {obj}："
            "可调整因子表达式/中性化/衰减/截断/参数等，但不得更改 data 部分"
            "（标的、时间区间、数据源必须保持不变），kind 也不得更改。"
        )
    return (
        f"Current objective {obj} = {shown}. Propose ONE revision to improve "
        f"{obj}. You may adjust expressions/ops/params, but the data section "
        "(symbols, period, source) and the kind must stay unchanged."
    )


def _pin_data(candidate_yaml: str, reference: TaskSpec) -> Optional[str]:
    """Force the candidate to keep the reference's data section and kind."""
    try:
        spec = TaskSpec.from_yaml(candidate_yaml)
    except Exception:
        return None
    if spec.kind != reference.kind:
        return None
    spec.data = reference.data
    spec.validate()
    return spec.to_yaml()


def auto_optimize(
    task_yaml: str,
    result_summary: Optional[Dict[str, Any]] = None,
    language: str = "en",
    workspace=None,
) -> Dict[str, Any]:
    """Reflect on `result_summary` (the caller's current result), propose
    and run ONE revision. Returns {"history": [entry], "best": entry} for
    compatibility with callers/UI written for the old multi-round shape."""
    if get_client() is None:
        raise LLMError(
            "LLM not configured — set up config/llm.yaml to run auto-optimize."
        )
    reference = TaskSpec.from_yaml(task_yaml)  # DSLError -> caller's 422

    failed = bool(result_summary) and result_summary.get("ok") is False
    baseline_score = _objective_from_summary(reference.kind, result_summary)
    question = _auto_question(reference.kind, language, baseline_score, failed)
    if failed:
        detail: Dict[str, Any] = {
            "error": result_summary.get("error"),
            "failed_step": result_summary.get("failed_step"),
        }
    elif reference.kind == "factor":
        detail = {"factor_stats": (result_summary or {}).get("factor_stats") or []}
    else:
        detail = {"metrics": (result_summary or {}).get("metrics") or {}}

    revision = refine_task(
        task_yaml, question, {"objective": baseline_score, "detail": detail}, language,
    )
    pinned = _pin_data(revision["yaml"], reference)
    if pinned is None:
        raise LLMError(
            "agent revision changed the data section or kind — rejected"
        )

    spec = TaskSpec.from_yaml(pinned)
    spec.name = f"{reference.name}-auto"[:48]
    result = run_task(spec, workspace=workspace)
    score = _objective(result) if result.ok else None
    entry = {
        "round": 1,
        "run_id": result.run_id,
        "ok": result.ok,
        "error": result.error,
        "failed_step": result.failed_step,
        "kind": result.kind,
        "objective": None if score is None else round(score, 4),
        "yaml": spec.to_yaml(),
        "task": spec.to_dict(),
        "explanation": revision["explanation"],
        "engine": revision["engine"],
    }
    return {
        "objective_name": _objective_name(reference.kind),
        "baseline_objective": baseline_score,
        "rounds": 1,
        "history": [entry],
        "best": entry,
    }
