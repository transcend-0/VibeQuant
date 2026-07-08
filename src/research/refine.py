"""Refine loop: follow-up questions that optimize an existing task.

Given the current task YAML, its latest result summary and a free-text
question ("试试行业中性化", "why is the drawdown so large?", "use a
shorter window"), propose a *revised task* plus an explanation. The LLM
does the thinking — there is no rule-based fallback, so an unconfigured
or failing LLM raises rather than returning a guessed revision. The
revised YAML is always validated (TaskSpec + factor-expression grammar)
before it reaches the user, and it is never auto-run — the UI loads it
into the plan review step.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional

from ..dsl import DSLError, TaskSpec
from ..llm import LLMError, get_client, query_structured
from ..strategy_source import StrategySourceError, validate_strategy_params
from .llm_ideas import _expression_valid, _FUNCTION_CATALOG, _strip_fence

_SYSTEM = f"""You are a quantitative research assistant improving research tasks for
VibeQuant, a system with a YAML task DSL. You receive the current task, its
latest results (or, if the last run failed, the error and failed step), and
the researcher's question. Propose ONE revised task.

If the latest result shows ok=false, the researcher's question may just be
asking you to fix the failure (e.g. "why did this fail?", "fix it") rather
than improve performance — read the error/failed_step and correct the
actual bug (syntax error, wrong method name, class not inheriting
Strategy, etc.) before trying to improve anything else.

DSL rules you must respect:
- top-level: name, kind ("strategy"|"factor"), data, strategy, factor, risk,
  execution, report. Do not invent keys.
- data.source: "etf"|"stock"|"index"|"hk"|"us"|"synthetic"|"csv";
  data.symbols: list; data.start/end: "YYYY-MM-DD".
- strategy.name: a short descriptive label only, purely informational.
- strategy.params.source: THE strategy. Complete Python source defining a
  class that inherits `Strategy` (already in scope — akquant's own
  Strategy base class), using akquant's real API directly on `self`:
  get_position(symbol), get_portfolio_value() -> float (total equity),
  get_cash() -> float (free cash, not equity -- there is NO self.portfolio/
  .equity attribute, use these methods), get_history(count, symbol,
  field="close") (already a plain numpy array -- do NOT call .values on it,
  that's a pandas Series/DataFrame method and raises AttributeError here;
  index/slice it directly), buy/sell(symbol=, quantity=), close_position(symbol=),
  order_target_percent(symbol=, target_percent=0..1, must be >= 0),
  order_target_weights(target_weights={{sym: pct}}, liquidate_unmentioned=False,
  rebalance_tolerance=0.01, weights must all be >= 0 -- short selling is
  disabled by default, matching real A-share retail constraints; a negative
  weight raises "target weight ... must be >= 0", implement long-short ideas
  long-only instead). Override on_bar(self, bar) for per-bar logic --
  bar has .symbol/.open/.high/.low/.close/.volume and .timestamp_iso (str,
  ISO 8601 "2022-01-02T16:00:00Z", the easy way to get date/time; there is
  NO bar.date/.datetime/.time, don't guess one; bar.timestamp is also there
  but a raw nanosecond int, not directly comparable to a month/day) --
  and/or on_daily_rebalance(self, trading_date, timestamp) for once-a-day
  cross-sectional logic. A custom __init__(self) must call super().__init__()
  and take no extra arguments. numpy (`np`)/pandas (`pd`) are in scope, as
  are globals SYMBOLS/START/END (also set as self.symbols/self.start/
  self.end automatically before __init__ runs, so either style works) and
  (when params.expressions is set) FACTOR_SCORES — a
  {{"YYYY-MM-DD": {{symbol: score}}}} map precomputed from those factor
  expressions, for on_daily_rebalance to rank on (its lookup tolerates
  trading_date directly, but str(trading_date)[:10] is the documented
  form). There is no library of rule templates (no
  ma_cross/rsi/momentum/bollinger/buy_hold) and no restrictive per-symbol
  callback — write the logic directly as an akquant Strategy class,
  single-symbol or cross-sectional alike.
- strategy.params.expressions: OPTIONAL, list of "Name = Expr" using ONLY
  these functions: {_FUNCTION_CATALOG} over Open/High/Low/Close/Volume, plus
  +-*/ and numbers — only meaningful together with a source that reads
  FACTOR_SCORES.
- factor.expressions: list of "Name = Expr" using ONLY these functions:
  {_FUNCTION_CATALOG} over Open/High/Low/Close/Volume, plus +-*/ and numbers.
- factor fields: forward_days (default 1), quantiles (2-10),
  truncation [0,0.5), neutralization none|demean|industry|zscore|rank,
  decay [0,60], max_position [0,1], max_trade [0,1].
- execution.mode must stay "backtest".

Reply with ONLY JSON (no markdown fence):
{{"explanation": "<2-4 sentences in the researcher's language explaining what
you changed and why, grounded in the results>",
 "task_yaml": "<the complete revised YAML task>"}}"""


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
    else:
        try:
            validate_strategy_params(spec.strategy.params)
        except StrategySourceError:
            return None
    return spec


def refine_task(
    task_yaml: str,
    question: str,
    result_summary: Optional[Dict[str, Any]] = None,
    language: str = "en",
) -> Dict[str, Any]:
    """Return {"yaml", "explanation", "engine"}. Raises LLMError when the LLM
    is unconfigured, unreachable, or produces an unusable revision."""
    TaskSpec.from_yaml(task_yaml)  # DSLError propagates: bad input is a 422

    client = get_client()
    if client is None:
        raise LLMError("LLM not configured — set up config/llm.yaml to refine tasks.")

    user_prompt = (
        f"Current task YAML:\n```yaml\n{task_yaml}\n```\n\n"
        f"Latest results (JSON):\n{json.dumps(result_summary or {}, ensure_ascii=False, default=str)[:4000]}\n\n"
        f"Researcher's language: {'Chinese' if language == 'zh' else 'English'}\n"
        f"Researcher's question / instruction:\n{question.strip()[:2000]}"
    )

    def _parse_revision(reply: str) -> Dict[str, Any]:
        try:
            payload = json.loads(_strip_fence(reply))
        except (json.JSONDecodeError, TypeError) as exc:
            raise ValueError(f"reply was not valid JSON: {exc}") from exc
        revised = str(payload.get("task_yaml", ""))
        explanation = str(payload.get("explanation", "")).strip()
        new_spec = _validate_yaml(revised)
        if new_spec is None:
            raise ValueError("revised task_yaml failed DSL/expression validation")
        if not explanation:
            raise ValueError("missing explanation")
        return {"spec": new_spec, "explanation": explanation}

    try:
        result = query_structured(client, user_prompt, _SYSTEM, _parse_revision)
    except LLMError as exc:
        raise LLMError(f"LLM refine failed: {exc}") from exc
    except ValueError as exc:
        raise LLMError(f"LLM refine failed after 3 attempts: {exc}") from exc

    new_spec = result["spec"]
    new_spec.report.language = language
    return {
        "yaml": new_spec.to_yaml(),  # normalized
        "explanation": result["explanation"],
        "engine": f"llm ({client.model_name})",
    }


_QUESTION_ONLY = re.compile(r"^\s*(why|how come|什么原因|为什么|为何)", re.I)


def is_pure_question(question: str) -> bool:
    """Heuristic: 'why…' questions may deserve an answer, not a new task."""
    return bool(_QUESTION_ONLY.match(question))
