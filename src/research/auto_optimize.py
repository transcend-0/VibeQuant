"""Agent self-iteration: run → reflect → revise → run, keep the best.

The autonomous counterpart to the manual refine loop. Given a starting
task and a round budget, the agent:

    1. runs the task,
    2. scores it on a single objective (factor: |ICIR|; strategy: Sharpe),
    3. asks the refine engine (LLM when configured, heuristics otherwise)
       for a revision aimed at that objective,
    4. repeats, keeping the best-scoring task.

Guardrails: rounds are capped, every candidate passes DSL + expression
validation (inside refine_task), the universe and period are pinned so
the agent can't "improve" results by changing the data, and everything
runs through the normal pipeline — each iteration is a persisted,
reproducible run in the experiment history.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from ..dsl import TaskSpec
from ..runner import run_task
from .refine import refine_task

logger = logging.getLogger(__name__)

MAX_ROUNDS = 5


def _objective(result) -> Optional[float]:
    """Single scalar to maximize. None when the run produced nothing."""
    if result.kind == "factor":
        stats = (result.factor or {}).get("stats") or []
        icirs = [abs(s["icir"]) for s in stats if s.get("icir") is not None]
        return max(icirs) if icirs else None
    sharpe = (result.metrics or {}).get("sharpe_ratio")
    return sharpe


def _objective_name(kind: str) -> str:
    return "|ICIR|" if kind == "factor" else "Sharpe"


def _auto_question(kind: str, language: str, score: Optional[float]) -> str:
    obj = _objective_name(kind)
    shown = "n/a" if score is None else f"{score:.3f}"
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
    rounds: int = 3,
    language: str = "en",
    workspace=None,
) -> Dict[str, Any]:
    """Iterate up to `rounds` times. Returns history + the best round."""
    rounds = max(1, min(int(rounds), MAX_ROUNDS))
    reference = TaskSpec.from_yaml(task_yaml)  # DSLError -> caller's 422

    history: List[Dict[str, Any]] = []
    current_yaml = reference.to_yaml()
    explanation = "baseline / 基线"
    engine = "-"

    for round_no in range(1, rounds + 1):
        spec = TaskSpec.from_yaml(current_yaml)
        spec.name = f"{reference.name}-auto{round_no}"[:48]
        result = run_task(spec, workspace=workspace)
        score = _objective(result) if result.ok else None
        entry = {
            "round": round_no,
            "run_id": result.run_id,
            "ok": result.ok,
            "error": result.error,
            "objective": None if score is None else round(score, 4),
            "yaml": spec.to_yaml(),
            "explanation": explanation,
            "engine": engine,
        }
        if result.ok:
            if spec.kind == "factor":
                entry["summary"] = [
                    {k: s.get(k) for k in
                     ("name", "ic_mean", "icir", "long_short_total_return")}
                    for s in (result.factor or {}).get("stats") or []
                ]
            else:
                entry["summary"] = result.metrics
        history.append(entry)

        if round_no == rounds:
            break

        # reflect & revise
        summary = entry.get("summary") or {"error": result.error}
        question = _auto_question(reference.kind, language, score)
        try:
            revision = refine_task(
                entry["yaml"], question,
                {"objective": entry["objective"], "detail": summary},
                language,
            )
        except Exception as exc:
            logger.warning("auto-optimize revision failed: %s", exc)
            break
        pinned = _pin_data(revision["yaml"], reference)
        if pinned is None:
            logger.warning("auto-optimize candidate rejected (data/kind drift)")
            break
        current_yaml = pinned
        explanation = revision["explanation"]
        engine = revision["engine"]

    scored = [h for h in history if h["objective"] is not None]
    best = max(scored, key=lambda h: h["objective"]) if scored else (
        history[0] if history else None
    )
    return {
        "objective_name": _objective_name(reference.kind),
        "rounds": len(history),
        "history": history,
        "best": best,
    }
