"""VibeQuant CLI.

    vq ask "在 600000 上做 5/20 双均线策略回测"   # parse -> show plan -> confirm
    vq ask "ma cross 5/20 on DEMO" --yes         # parse and run
    vq run tasks/ma_cross_demo.yaml              # run a YAML task
    vq plan tasks/ma_cross_demo.yaml             # show plan only
    vq strategies                                # list strategy templates
    vq runs                                      # recent run history
    vq ui                                        # start the web UI
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import workspace_dir
from .dsl import TaskSpec
from .intent import IntentError, parse_prompt
from .memory import MemoryStore
from .planner import make_plan
from .runner import run_task
from .strategies import REGISTRY


def _print_result(result) -> None:
    print()
    if not result.ok:
        print(f"❌ run {result.run_id} FAILED at step [{result.failed_step}]")
        print(f"   {result.error}")
        sys.exit(1)
    print(result.report_markdown)
    print("Artifacts:")
    for name, path in sorted(result.artifacts.items()):
        print(f"  - {name}: {path}")


def cmd_ask(args) -> None:
    try:
        parsed = parse_prompt(args.prompt)
    except IntentError as exc:
        sys.exit(f"❌ {exc}")
    spec = parsed.spec
    lang = spec.report.language

    print("── Parsed task (DSL) " + "─" * 40)
    print(spec.to_yaml())
    if parsed.clarifications:
        print("── Assumptions / 假设 " + "─" * 39)
        for note in parsed.clarifications:
            print(f"  • {note}")
    print("── Plan " + "─" * 53)
    print(make_plan(spec).describe(lang))
    print()

    if not args.yes:
        reply = input("Run this task? / 执行该任务？ [y/N] ").strip().lower()
        if reply not in ("y", "yes"):
            print("Aborted. Save the YAML above and edit it, then `vq run <file>`.")
            return
    _print_result(run_task(spec))


def cmd_run(args) -> None:
    path = Path(args.task)
    if not path.exists():
        sys.exit(f"task file not found: {path}")
    spec = TaskSpec.from_yaml(path.read_text(encoding="utf-8"))
    _print_result(run_task(spec))


def cmd_plan(args) -> None:
    path = Path(args.task)
    if not path.exists():
        sys.exit(f"task file not found: {path}")
    spec = TaskSpec.from_yaml(path.read_text(encoding="utf-8"))
    print(spec.to_yaml())
    print(make_plan(spec).describe(spec.report.language))


def cmd_strategies(_args) -> None:
    for name, tpl in sorted(REGISTRY.items()):
        print(f"{name:>15}  {tpl.summary_en}")
        print(f"{'':>15}  {tpl.summary_zh}")
        if tpl.defaults:
            print(f"{'':>15}  defaults: {tpl.defaults}")
        print()


def cmd_runs(args) -> None:
    store = MemoryStore(workspace_dir())
    entries = store.list_runs(limit=args.limit)
    if not entries:
        print("no runs yet")
        return
    for entry in entries:
        metrics = entry.get("metrics") or {}
        ret = metrics.get("total_return_pct")
        sharpe = metrics.get("sharpe_ratio")
        print(
            f"{entry.get('run_id', '?'):<45} "
            f"{entry.get('strategy', '?'):<15} "
            f"ret={ret if ret is None else f'{ret:.2f}%':<9} "
            f"sharpe={sharpe if sharpe is None else f'{sharpe:.2f}'}"
        )


def cmd_deploy(args) -> None:
    from .live import (
        create_deployment,
        delete_deployment,
        list_deployments,
        run_deployment,
    )

    if args.action == "list":
        deps = list_deployments()
        if not deps:
            print("no deployments")
            return
        for d in deps:
            state = "on " if d.enabled else "off"
            print(
                f"{d.id}  [{state}] {d.run_at}  {d.name:<28} "
                f"email={d.email_to or '-'}  last={d.last_run_date or '-'} "
                f"({d.last_status or '-'})"
            )
    elif args.action == "add":
        yaml_text = Path(args.task).read_text(encoding="utf-8")
        dep = create_deployment(
            yaml_text, email_to=args.email or "", run_at=args.at
        )
        print(f"created {dep.id} (runs weekdays at {dep.run_at} Asia/Shanghai)")
    elif args.action == "run":
        out = run_deployment(args.id, force=True)
        print(out.get("report", out))
    elif args.action == "rm":
        print("deleted" if delete_deployment(args.id) else "not found")


def cmd_ui(args) -> None:
    import uvicorn

    uvicorn.run(
        "webui.server:app",
        host=args.host,
        port=args.port,
        app_dir=str(Path(__file__).resolve().parent.parent),
    )


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        prog="vq", description="VibeQuant — intent-driven quant research"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_ask = sub.add_parser("ask", help="natural-language prompt -> plan -> run")
    p_ask.add_argument("prompt")
    p_ask.add_argument("--yes", "-y", action="store_true", help="skip confirmation")
    p_ask.set_defaults(fn=cmd_ask)

    p_run = sub.add_parser("run", help="run a YAML task file")
    p_run.add_argument("task")
    p_run.set_defaults(fn=cmd_run)

    p_plan = sub.add_parser("plan", help="show the plan for a YAML task file")
    p_plan.add_argument("task")
    p_plan.set_defaults(fn=cmd_plan)

    p_str = sub.add_parser("strategies", help="list strategy templates")
    p_str.set_defaults(fn=cmd_strategies)

    p_runs = sub.add_parser("runs", help="list recent runs")
    p_runs.add_argument("--limit", type=int, default=20)
    p_runs.set_defaults(fn=cmd_runs)

    p_dep = sub.add_parser("deploy", help="manage daily signal deployments")
    dep_sub = p_dep.add_subparsers(dest="action", required=True)
    dep_sub.add_parser("list", help="list deployments")
    p_add = dep_sub.add_parser("add", help="deploy a strategy task YAML")
    p_add.add_argument("task")
    p_add.add_argument("--email", help="recipient address")
    p_add.add_argument("--at", default="16:30", help="run time (Asia/Shanghai)")
    p_drun = dep_sub.add_parser("run", help="run one deployment now")
    p_drun.add_argument("id")
    p_drm = dep_sub.add_parser("rm", help="delete a deployment")
    p_drm.add_argument("id")
    p_dep.set_defaults(fn=cmd_deploy)

    p_ui = sub.add_parser("ui", help="start the web UI")
    p_ui.add_argument("--host", default="127.0.0.1")
    p_ui.add_argument("--port", type=int, default=8321)
    p_ui.set_defaults(fn=cmd_ui)

    args = parser.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
