"""VibeQuant — intent-driven quant research on top of akquant.

Pipeline: prompt / YAML task -> TaskSpec (DSL) -> planner -> tools
          -> akquant adapter -> results / report / memory
"""

__version__ = "0.1.0"

from .dsl import TaskSpec, DSLError  # noqa: F401
from .intent import parse_prompt, ParseResult  # noqa: F401
from .planner import make_plan, Plan  # noqa: F401
from .runner import run_task, RunResult  # noqa: F401
