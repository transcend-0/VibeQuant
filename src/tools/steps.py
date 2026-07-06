"""Built-in tools for both pipelines.

strategy: risk_gate, load_data, backtest, risk_assess, report, memorize
factor:   risk_gate, load_data, factor_compute, factor_analyze,
          factor_report, memorize
"""

from __future__ import annotations

import json

from .. import data as data_mod
from ..adapters import akquant_engine
from ..report import build_factor_markdown_report, build_markdown_report
from ..risk import RiskAssessment, post_run_assess, pre_run_gate
from . import register
from .context import RunContext


@register("risk_gate")
def risk_gate(ctx: RunContext) -> None:
    ctx.gate_notes = pre_run_gate(ctx.spec)


@register("load_data")
def load_data(ctx: RunContext) -> None:
    ctx.data = data_mod.load(ctx.spec.data)


@register("backtest")
def backtest(ctx: RunContext) -> None:
    assert ctx.data is not None, "load_data must run before backtest"
    ctx.backtest = akquant_engine.run_backtest(ctx.spec, ctx.data)


@register("risk_assess")
def risk_assess(ctx: RunContext) -> None:
    assert ctx.backtest is not None, "backtest must run before risk_assess"
    total_bars = sum(len(df) for df in (ctx.data or {}).values())
    ctx.risk_assessment = post_run_assess(
        ctx.spec,
        ctx.backtest.metrics,
        ctx.backtest.num_trades,
        total_bars,
    )


@register("report")
def report(ctx: RunContext) -> None:
    assert ctx.backtest is not None and ctx.risk_assessment is not None
    ctx.report_markdown = build_markdown_report(
        ctx.spec,
        ctx.backtest.metrics,
        ctx.backtest.num_trades,
        ctx.risk_assessment,
        ctx.run.run_id,
        notes=ctx.gate_notes,
        validation=ctx.validation,
    )
    if ctx.spec.report.html:
        html_path = str(ctx.run.path("report.html"))
        written = akquant_engine.write_html_report(
            ctx.backtest,
            html_path,
            title=f"VibeQuant · {ctx.spec.name}",
            market_data=ctx.data,
            benchmark=_load_benchmark(ctx),
        )
        if written:
            ctx.artifacts["report.html"] = written


# default benchmark per market; report.benchmark overrides
_DEFAULT_BENCHMARKS = {
    "etf": ("index", "sh000300"),
    "stock": ("index", "sh000300"),
    "index": ("index", "sh000300"),
    "us": ("us", ".INX"),
    "hk": ("hk", "HSI"),
    "crypto": ("crypto", "BTC"),
}


def _load_benchmark(ctx: RunContext):
    """Benchmark daily-return series for the report, or None."""
    import re as _re

    import pandas as pd

    spec = ctx.spec
    override = (spec.report.benchmark or "").strip()
    if override:
        if override.startswith("."):
            kind, symbol = "us", override
        elif _re.fullmatch(r"(?:sh|sz)\d{6}", override, _re.I):
            kind, symbol = "index", override
        else:
            kind, symbol = spec.data.source, override
    else:
        pair = _DEFAULT_BENCHMARKS.get(spec.data.source)
        if pair is None:
            return None  # synthetic/csv: no meaningful benchmark
        kind, symbol = pair

    try:
        from ..config import raw_data_dir
        from ..data_sources import market

        # benchmarking a symbol against itself is noise
        if market.canonical(symbol, kind) in (ctx.data or {}):
            return None
        df = market.fetch_daily(
            symbol, kind,
            spec.data.start or "2020-01-01",
            spec.data.end or pd.Timestamp.today().strftime("%Y-%m-%d"),
            cache_dir=raw_data_dir(kind),
        )
    except Exception as exc:
        ctx.gate_notes.append(f"benchmark unavailable ({symbol}): {exc}")
        return None
    series = df.set_index("date")["close"].pct_change().fillna(0.0)
    return series.rename(market.canonical(symbol, kind))


@register("factor_compute")
def factor_compute(ctx: RunContext) -> None:
    from ..adapters import akquant_factor

    assert ctx.data is not None, "load_data must run before factor_compute"
    names = [
        akquant_factor.split_named_expression(raw, i)[0]
        for i, raw in enumerate(ctx.spec.factor.expressions)
    ]
    ctx.factor_names = names
    ctx.factor_panel = akquant_factor.compute_factors(
        ctx.spec.factor.expressions, ctx.data
    )

    # point-in-time universes (e.g. hs300): a symbol only participates in
    # the cross-section on dates when it was actually a pool member
    from ..data_sources.constituents import available_pools, membership_mask

    if ctx.spec.data.universe in available_pools():
        mask = membership_mask(ctx.spec.data.universe, ctx.factor_panel)
        dropped = int((~mask).sum())
        ctx.factor_panel = ctx.factor_panel[mask].reset_index(drop=True)
        if dropped:
            ctx.gate_notes.append(
                f"point-in-time universe '{ctx.spec.data.universe}': "
                f"{dropped} symbol-days outside membership were excluded"
            )


@register("factor_analyze")
def factor_analyze(ctx: RunContext) -> None:
    from ..factors import analyze_factor

    assert ctx.factor_panel is not None, "factor_compute must run first"
    groups = None
    if ctx.spec.factor.neutralization == "industry":
        from ..data_sources.etf_pool import industry_groups

        symbols = list((ctx.data or {}).keys())
        groups = industry_groups(symbols)

    adv = None
    if ctx.spec.factor.max_position > 0 or ctx.spec.factor.max_trade > 0:
        from ..factors.analysis import adv20_panel

        # A-share volume comes in lots of 100 shares; crypto/us in units
        lot = 100.0 if ctx.spec.data.source in ("etf", "stock", "index") else 1.0
        adv = adv20_panel(ctx.data or {}, lot_multiplier=lot)

    ctx.factor_report = analyze_factor(
        ctx.factor_panel,
        ctx.data or {},
        ctx.factor_names,
        forward_days=ctx.spec.factor.forward_days,
        quantiles=ctx.spec.factor.quantiles,
        truncation=ctx.spec.factor.truncation,
        neutralization=ctx.spec.factor.neutralization,
        decay=ctx.spec.factor.decay,
        groups=groups,
        max_position=ctx.spec.factor.max_position,
        max_trade=ctx.spec.factor.max_trade,
        adv=adv,
        book=ctx.spec.execution.initial_cash,
    )
    # a factor study has no order flow; risk = analysis-quality warnings
    ctx.risk_assessment = RiskAssessment(
        passed=True, flags=[], warnings=list(ctx.factor_report.warnings)
    )


@register("factor_report")
def factor_report(ctx: RunContext) -> None:
    assert ctx.factor_report is not None
    ctx.report_markdown = build_factor_markdown_report(
        ctx.spec, ctx.factor_report, ctx.run.run_id, notes=ctx.gate_notes,
        validation=ctx.validation,
    )


@register("validate")
def validate(ctx: RunContext) -> None:
    """Validation v1: luck vs signal, from the single run's artifacts."""
    from ..factors import validation as val

    spec = ctx.spec
    if spec.kind == "factor":
        assert ctx.factor_report is not None and ctx.factor_panel is not None
        from ..factors.analysis import _forward_returns

        fwd = _forward_returns(ctx.data or {}, spec.factor.forward_days)
        trials = val.count_prior_trials(ctx.store, spec, kind="factor")
        results = {}
        for name in ctx.factor_names:
            series = ctx.factor_report.ic_series.get(name) or {}
            results[name] = val.validate_factor(
                ctx.factor_panel, fwd, name,
                series.get("dates") or [], series.get("ic") or [],
                trial_count=trials,
            )
        worst = max(
            results.values(),
            key=lambda r: ["low", "medium", "high"].index(r["overfit_risk"]),
        )
        ctx.validation = {
            "kind": "factor",
            "per_factor": results,
            "trial_count": trials,
            "overfit_risk": worst["overfit_risk"],
        }
    else:
        assert ctx.backtest is not None
        ctx.validation = val.validate_strategy(
            ctx.backtest.equity_curve, ctx.backtest.num_trades
        )


@register("memorize")
def memorize(ctx: RunContext) -> None:
    spec, run, store = ctx.spec, ctx.run, ctx.store
    risk = ctx.risk_assessment
    assert risk is not None and ctx.report_markdown

    store.save_artifact(run, "task.yaml", spec.to_yaml())
    store.save_artifact(run, "report.md", ctx.report_markdown)

    result = {
        "run_id": run.run_id,
        "kind": spec.kind,
        "task_name": spec.name,
        "intent": spec.intent,
        "symbols": spec.data.symbols,
        "data_source": spec.data.source,
        "universe": spec.data.universe,
        "period": {"start": spec.data.start, "end": spec.data.end},
        "mode": spec.execution.mode,
        "risk": risk.to_dict(),
        "validation": ctx.validation,
        "gate_notes": ctx.gate_notes,
    }
    experiment = {
        "run_id": run.run_id,
        "kind": spec.kind,
        "task_name": spec.name,
        "symbols": spec.data.symbols,
        "data_source": spec.data.source,
        "universe": spec.data.universe,
        "risk_passed": risk.passed,
        "overfit_risk": (ctx.validation or {}).get("overfit_risk"),
    }

    if spec.kind == "strategy":
        bt = ctx.backtest
        assert bt is not None
        equity_path = store.save_equity_curve(run, bt.equity_curve)
        if equity_path:
            ctx.artifacts["equity.csv"] = str(equity_path)
        result["strategy"] = spec.strategy.name
        result["params"] = spec.strategy.params
        result["summary"] = bt.to_summary()
        experiment.update(
            strategy=spec.strategy.name,
            params=spec.strategy.params,
            metrics=bt.metrics,
            num_trades=bt.num_trades,
        )
    else:
        fr = ctx.factor_report
        assert fr is not None
        store.save_artifact(
            run,
            "factor_analysis.json",
            json.dumps(fr.to_dict(), ensure_ascii=False, indent=2),
        )
        ctx.artifacts["factor_analysis.json"] = str(run.path("factor_analysis.json"))
        result["factor"] = {
            "expressions": spec.factor.expressions,
            "forward_days": spec.factor.forward_days,
            "quantiles": spec.factor.quantiles,
            "stats": [s.to_dict() for s in fr.stats],
        }
        # persist each factor's values to the library (data/factors/)
        from ..adapters.akquant_factor import split_named_expression
        from ..factors.registry import register_factor

        stats_by_name = {st.name: st.to_dict() for st in fr.stats}
        for i, raw_expr in enumerate(spec.factor.expressions):
            fname, expr = split_named_expression(raw_expr, i)
            if ctx.factor_panel is None or fname not in ctx.factor_panel.columns:
                continue
            register_factor(
                fname,
                expr,
                ctx.factor_panel,
                run.run_id,
                spec_summary={
                    "symbols": spec.data.symbols,
                    "source": spec.data.source,
                    "start": spec.data.start,
                    "end": spec.data.end,
                    "forward_days": spec.factor.forward_days,
                    "quantiles": spec.factor.quantiles,
                },
                stats=stats_by_name.get(fname, {}),
            )

        best = _best_factor(fr)
        experiment.update(
            strategy="factor:" + ",".join(ctx.factor_names)[:60],
            metrics={
                "ic_mean": best.ic_mean if best else None,
                "icir": best.icir if best else None,
                "long_short_total_return": (
                    best.long_short_total_return if best else None
                ),
            },
            num_trades=0,
        )

    store.save_artifact(
        run, "result.json", json.dumps(result, ensure_ascii=False, indent=2)
    )
    ctx.artifacts["task.yaml"] = str(run.path("task.yaml"))
    ctx.artifacts["report.md"] = str(run.path("report.md"))
    ctx.artifacts["result.json"] = str(run.path("result.json"))

    store.append_experiment(experiment)


def _best_factor(fr):
    scored = [s for s in fr.stats if s.icir is not None]
    if not scored:
        return fr.stats[0] if fr.stats else None
    return max(scored, key=lambda s: abs(s.icir))
