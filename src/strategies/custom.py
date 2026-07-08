"""Default "blank" skeleton: a bare per-bar akquant Strategy class.

`params["source"]` is executed directly by the adapter (compile + exec, no
AST allowlist, no sandbox) -- an accepted, explicit risk decision
(2026-07-07): VibeQuant is a single-user, 127.0.0.1-only local tool, and
the user chose direct exec over building a static-analysis gate or process
sandbox. The known exposure this leaves open: `src/research/ingest.py`
feeds arbitrary PDF/URL/text content into the LLM, so adversarial content
in an ingested source could in principle prompt-inject the model into
emitting malicious Python here, which would then run with the full
privileges of the local process (file/network access, no restrictions).
Do not expose this server beyond localhost without revisiting this.

Widening the executed surface from a `signal(closes, position)` callback
to a full `Strategy` subclass (direct order/broker access) is the same
accepted risk, just larger -- there is no additional sandboxing here
either.
"""

from __future__ import annotations

from . import StrategySkeleton, register

DEFAULT_SOURCE = '''class Strategy(BaseStrategy):
    """Write your trading logic here."""

    def on_bar(self, bar):
        # bar.symbol / bar.open / bar.high / bar.low / bar.close / bar.volume
        # self.get_position(bar.symbol) -> current quantity held (0 if flat)
        # self.buy(symbol=..., quantity=...) / self.close_position(symbol=...)
        # self.order_target_percent(symbol=..., target_percent=0.95)
        pass
'''

register(
    StrategySkeleton(
        name="custom",
        summary_en="Blank akquant Strategy class -- write your own on_bar logic.",
        summary_zh="空白 akquant Strategy 类——自行编写 on_bar 交易逻辑。",
        source=DEFAULT_SOURCE,
    )
)
