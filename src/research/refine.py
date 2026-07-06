"""Refine loop: follow-up questions that optimize an existing task.

Given the current task YAML, its latest result summary and a free-text
question ("试试行业中性化", "why is the drawdown so large?", "use a
shorter window"), propose a *revised task* plus an explanation. The LLM
does the thinking when configured; a deterministic heuristic fallback
keeps the loop alive without one. The revised YAML is always validated
(TaskSpec + factor-expression grammar) before it reaches the user, and
it is never auto-run — the UI loads it into the plan review step.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, Optional

from ..dsl import DSLError, TaskSpec
from ..llm import LLMError, get_client
from .llm_ideas import _expression_valid, _strip_fence

logger = logging.getLogger(__name__)

_SYSTEM = """You are a quantitative research assistant improving research tasks for
VibeQuant, a system with a YAML task DSL. You receive the current task, its
latest results, and the researcher's question. Propose ONE revised task.

DSL rules you must respect:
- top-level: name, kind ("strategy"|"factor"), data, strategy, factor, risk,
  execution, report. Do not invent keys.
- data.source: "etf"|"stock"|"index"|"hk"|"us"|"synthetic"|"csv";
  data.symbols: list; data.start/end: "YYYY-MM-DD".
- strategy.name: ma_cross{fast,slow} | rsi_reversion{period,oversold,overbought}
  | momentum{lookback,threshold} | bollinger{period,num_std} | buy_hold{}.
- factor.expressions: list of "Name = Expr" using ONLY
  Ts_Mean/Ts_Std/Ts_Max/Ts_Min/Ts_Sum/Ts_Corr/Ts_Cov/Ts_Rank/Ts_ArgMax/
  Ts_ArgMin/Delay/Delta/Rank/Scale/Log/Abs/Sign/If over
  Open/High/Low/Close/Volume, +-*/ and numbers.
- factor fields: forward_days (default 1), quantiles (2-10),
  truncation [0,0.5), neutralization none|demean|industry|zscore|rank,
  decay [0,60], max_position [0,1], max_trade [0,1].
- execution.mode must stay "backtest".

Reply with ONLY JSON (no markdown fence):
{"explanation": "<2-4 sentences in the researcher's language explaining what
you changed and why, grounded in the results>",
 "task_yaml": "<the complete revised YAML task>"}"""


def _heuristic_refine(spec: TaskSpec, question: str, language: str) -> Dict[str, Any]:
    """Rule-based fallback: one sensible next experiment per kind."""
    changes = []
    if spec.kind == "factor":
        f = spec.factor
        if f.neutralization == "none":
            f.neutralization = "rank"
            changes.append("neutralization: none → rank")
        elif f.decay == 0:
            f.decay = 5
            changes.append("decay: 0 → 5")
        elif f.truncation == 0:
            f.truncation = 0.05
            changes.append("truncation: 0 → 0.05")
        else:
            f.quantiles = max(2, f.quantiles - 1)
            changes.append(f"quantiles → {f.quantiles}")
    else:
        params = spec.strategy.params
        if spec.strategy.name == "ma_cross":
            fast = int(params.get("fast", 5))
            slow = int(params.get("slow", 20))
            params["fast"], params["slow"] = max(2, fast // 2 + 1), slow * 2
            changes.append(f"ma windows → {params['fast']}/{params['slow']}")
        elif spec.strategy.name == "momentum":
            lb = int(params.get("lookback", 20))
            params["lookback"] = lb * 2 if lb <= 30 else lb // 2
            changes.append(f"lookback → {params['lookback']}")
        else:
            spec.execution.slippage_bps = max(spec.execution.slippage_bps, 2.0)
            changes.append("stress-test costs: slippage → 2 bps")
    spec.name = (spec.name + "-v2")[:48]
    spec.validate()
    if language == "zh":
        explanation = (
            "（未配置 LLM，使用规则建议）下一步实验：" + "；".join(changes)
            + "。运行后对比两次结果再决定方向。"
        )
    else:
        explanation = (
            "(LLM not configured — heuristic suggestion) Next experiment: "
            + "; ".join(changes)
            + ". Run it and compare against the previous result."
        )
    return {"yaml": spec.to_yaml(), "explanation": explanation, "engine": "rules"}


def _validate_yaml(task_yaml: str) -> Optional[TaskSpec]:
    try:
        spec = TaskSpec.from_yaml(task_yaml)
    except DSLError:
        return None
    if spec.execution.mode != "backtest":
        return None
    if spec.kind == "factor":
        for expr in spec.factor.expressions:
            if not _expression_valid(expr):
                return None
    return spec


def refine_task(
    task_yaml: str,
    question: str,
    result_summary: Optional[Dict[str, Any]] = None,
    language: str = "en",
) -> Dict[str, Any]:
    """Return {"yaml", "explanation", "engine"} — never raises for LLM issues."""
    spec = TaskSpec.from_yaml(task_yaml)  # DSLError propagates: bad input is a 422

    client = get_client()
    if client is not None and question.strip():
        user_prompt = (
            f"Current task YAML:\n```yaml\n{task_yaml}\n```\n\n"
            f"Latest results (JSON):\n{json.dumps(result_summary or {}, ensure_ascii=False, default=str)[:4000]}\n\n"
            f"Researcher's language: {'Chinese' if language == 'zh' else 'English'}\n"
            f"Researcher's question / instruction:\n{question.strip()[:2000]}"
        )
        try:
            reply = client.query(user_prompt, system_prompt=_SYSTEM)
            payload = json.loads(_strip_fence(reply))
            revised = str(payload.get("task_yaml", ""))
            explanation = str(payload.get("explanation", "")).strip()
            new_spec = _validate_yaml(revised)
            if new_spec is not None and explanation:
                new_spec.report.language = language
                return {
                    "yaml": new_spec.to_yaml(),  # normalized
                    "explanation": explanation,
                    "engine": f"llm ({client.model_name})",
                }
            logger.warning("LLM refine produced invalid task; falling back")
        except (LLMError, json.JSONDecodeError, TypeError) as exc:
            logger.warning("LLM refine unavailable: %s", exc)

    return _heuristic_refine(spec, question, language)


_QUESTION_ONLY = re.compile(r"^\s*(why|how come|什么原因|为什么|为何)", re.I)


def is_pure_question(question: str) -> bool:
    """Heuristic: 'why…' questions may deserve an answer, not a new task."""
    return bool(_QUESTION_ONLY.match(question))
