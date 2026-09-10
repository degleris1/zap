"""Command line interface: ``plan``, ``run``, ``evaluate``, ``aggregate``, ``show``.

Local execution and SLURM execution use the same entry point and differ only in
which shard of the task list a process claims::

    python -m experiments.ra.cli plan      --config CFG [--set k=v ...]
    python -m experiments.ra.cli run       --config CFG [--shard k/n] [--force] [--only method=lp]
    python -m experiments.ra.cli evaluate  --config CFG --design-run RUN_ID [--design-file PATH]
                                           [--shard k/n] [--force] [--preflight-only]
    python -m experiments.ra.cli aggregate --run-id ID | --config CFG
    python -m experiments.ra.cli show      --run-id ID
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from . import metrics as metrics_mod
from . import paths, runcard
from . import system as system_mod
from . import tasks as tasks_mod
from .config import (
    DISPATCH_SELECTION_KEYS,
    PLAN_SELECTION_KEYS,
    ConfigError,
    base_config,
    deep_merge,
    dump_config,
    is_plan_mode,
    load_config,
)
from .identity import config_hash, env_info, run_id

logger = logging.getLogger("experiments.ra")

STUB_OVERRIDES = ("methods.lp.solver=STUB", "methods.admm.solver=STUB")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="experiments.ra.cli", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def add_config_args(p):
        p.add_argument("--config", required=True, help="path to a run config YAML")
        p.add_argument(
            "--set",
            dest="overrides",
            action="append",
            default=[],
            metavar="KEY=VALUE",
            help="override a config key, e.g. --set system.voll=5000",
        )
        p.add_argument(
            "--stub",
            action="store_true",
            help="use the deterministic STUB solver (no data, no solver)",
        )
        p.add_argument("--runs-root", default=None, help="override the runs directory")

    plan = sub.add_parser("plan", help="print the run id and the task list")
    add_config_args(plan)
    plan.add_argument("--limit", type=int, default=20, help="task ids to print (-1 for all)")

    run = sub.add_parser("run", help="execute this process's shard of the run")
    add_config_args(run)
    run.add_argument("--shard", default=None, metavar="K/N")
    run.add_argument("--force", action="store_true", help="re-run tasks that already have results")
    run.add_argument("--max-tasks", type=int, default=None)
    run.add_argument(
        "--only",
        action="append",
        default=[],
        metavar="FIELD=VALUE",
        help="only run matching tasks, e.g. --only method=lp",
    )

    evaluate = sub.add_parser(
        "evaluate", help="score one or more designs on weather years x outage draws"
    )
    add_config_args(evaluate)
    evaluate.add_argument(
        "--design-run",
        dest="design_runs",
        action="append",
        default=[],
        metavar="RUN_ID",
        help="a planning run id (or run directory): every designs/*.json in it is scored",
    )
    evaluate.add_argument(
        "--design-file",
        dest="design_files",
        action="append",
        default=[],
        metavar="PATH",
        help="one design.json to score",
    )
    evaluate.add_argument("--shard", default=None, metavar="K/N")
    evaluate.add_argument("--force", action="store_true", help="re-run this shard's cases")
    evaluate.add_argument("--max-tasks", type=int, default=None)
    evaluate.add_argument(
        "--only",
        action="append",
        default=[],
        metavar="FIELD=VALUE",
        help="only run matching tasks, e.g. --only draw=3",
    )
    evaluate.add_argument(
        "--preflight-only",
        action="store_true",
        help="run the preflight checks, write preflight.json and stop",
    )

    aggregate = sub.add_parser("aggregate", help="rebuild metrics.csv and CARD.md")
    aggregate.add_argument("--config", default=None)
    aggregate.add_argument("--set", dest="overrides", action="append", default=[])
    aggregate.add_argument("--stub", action="store_true")
    aggregate.add_argument("--run-id", default=None)
    aggregate.add_argument("--runs-root", default=None)

    show = sub.add_parser("show", help="print a run's card and status counts")
    show.add_argument("--run-id", required=True)
    show.add_argument("--runs-root", default=None)

    plot = sub.add_parser("plot", help="render figures from one or more runs")
    from .plots.cli import add_arguments as add_plot_arguments

    add_plot_arguments(plot)

    design = sub.add_parser("design", help="print a summary of a run's design(s)")
    design.add_argument("--run-id", required=True)
    design.add_argument("--design-id", default=None, help="one design; default: all of them")
    design.add_argument("--runs-root", default=None)

    return parser


def resolve_config(args) -> dict:
    overrides = list(args.overrides or [])
    if getattr(args, "stub", False):
        overrides = list(STUB_OVERRIDES) + overrides
    cfg = load_config(args.config, overrides)
    if getattr(args, "shard", None) is not None:
        cfg["execution"]["shard"] = args.shard
    if getattr(args, "max_tasks", None) is not None:
        cfg["execution"]["max_tasks"] = args.max_tasks
    if getattr(args, "force", False):
        cfg["execution"]["force"] = True
    if getattr(args, "runs_root", None) is not None:
        cfg["output"]["runs_root"] = str(args.runs_root)
    return cfg


def run_directory(cfg: dict) -> Path:
    return paths.run_dir(run_id(cfg), cfg["output"].get("runs_root"))


def parse_pairs(items) -> dict[str, str]:
    out = {}
    for item in items or []:
        if "=" not in item:
            raise ConfigError(f"--only expects field=value, got {item!r}")
        key, _, value = item.partition("=")
        out[key.strip()] = value
    return out


def _configure_logging(run_dir: Path | None = None) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if run_dir is not None:
        run_dir.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(run_dir / "log.txt"))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )


def adopt_task_granularity(cfg: dict, run_dir: Path) -> str:
    """Make ``cfg`` use whatever granularity this run directory already holds.

    ``execution`` is excluded from the run-id hash, so a ``block`` invocation
    and a ``case`` invocation of the same config share a run directory. Each
    only resume-skips *its own* file names, so both sets of files would survive
    side by side and every block would be counted twice in ``metrics.csv``
    (verifier F1, 2026-09-09). The files already on disk win: a re-entry that
    asks for the other granularity is coerced back, loudly, so the run stays
    resumable and its ledger stays one row per block. To score the same config
    at the other granularity, use a different ``--runs-root`` (or a fresh
    directory) -- which is exactly what the equivalence test does.
    """
    wanted = tasks_mod.task_granularity(cfg)
    found = tasks_mod.existing_granularity(run_dir)
    if found is None or found == wanted:
        return wanted
    logger.warning(
        "%s already holds `%s` task files; ignoring execution.task_granularity=%r for this "
        "invocation and continuing at `%s`. Running both granularities in one run directory "
        "would double-count every block. Use a separate runs root to compare them.",
        run_dir,
        found,
        wanted,
        found,
    )
    cfg["execution"]["task_granularity"] = found
    return found


def touch_run_dir(cfg: dict, run_dir: Path) -> None:
    """Write ``config.resolved.yaml`` and ``env.json``, or verify the hash on re-entry."""
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "tasks").mkdir(exist_ok=True)
    granularity = adopt_task_granularity(cfg, run_dir)

    resolved = run_dir / "config.resolved.yaml"
    if resolved.exists():
        existing = load_config(resolved)
        if config_hash(existing) != config_hash(cfg):
            raise ConfigError(
                f"{resolved} hashes to {config_hash(existing)} but this invocation resolves "
                f"to {config_hash(cfg)}; the run directory belongs to a different config"
            )
    else:
        # Store the hashed content; execution/output are excluded from the hash,
        # so they are written at their defaults rather than this shard's values.
        # `task_granularity` is the exception: it decides the *shape* of the
        # files in `tasks/`, so the resolved config has to say which shape they
        # are (and `ra aggregate --run-id` reads it back from here).
        base = base_config()
        execution = dict(base["execution"])
        execution["task_granularity"] = granularity
        dump_config(
            deep_merge(cfg, {"execution": execution, "output": base["output"]}), resolved
        )

    env_path = run_dir / "env.json"
    if not env_path.exists():
        with open(env_path, "w") as f:
            json.dump(env_info(cfg), f, indent=2, sort_keys=True, default=str)


def cmd_plan(args) -> int:
    cfg = resolve_config(args)
    all_tasks = tasks_mod.enumerate_tasks(cfg)
    print(f"run_id: {run_id(cfg)}")
    print(f"run_dir: {run_directory(cfg)}")
    print(f"tasks: {len(all_tasks)}")
    by_key: dict[tuple, int] = {}
    for task in all_tasks:
        key = (task.method, task.block_size)
        by_key[key] = by_key.get(key, 0) + 1
    for (method, size), count in sorted(by_key.items(), key=lambda kv: str(kv[0])):
        print(f"  {method:5s} block_size={size!s:>9s}  {count} tasks")
    limit = len(all_tasks) if args.limit is not None and args.limit < 0 else args.limit
    for task in all_tasks[:limit]:
        print(f"  {task.task_id}  hours=[{task.block.start}, {task.block.stop})")
    if limit is not None and len(all_tasks) > limit:
        print(f"  ... {len(all_tasks) - limit} more")
    return 0


def log_selection_keys(cfg: dict) -> None:
    """Say which half of the ``selection`` block this mode reads (R-W9).

    ``mode: dispatch`` reads ``blocks / reference / reference_window``;
    ``mode: plan`` reads ``strategy / block_size / num_blocks / seed /
    avoid_year_boundaries``.  The other half is silently ignored, so a config
    that sets it non-default gets a warning in ``log.txt`` rather than a
    surprise.
    """
    plan_mode = is_plan_mode(cfg)
    read_keys = PLAN_SELECTION_KEYS if plan_mode else DISPATCH_SELECTION_KEYS
    ignored_keys = DISPATCH_SELECTION_KEYS if plan_mode else PLAN_SELECTION_KEYS
    sel = cfg["selection"]
    logger.info(
        "mode=%s reads selection keys %s: %s",
        cfg["mode"],
        list(read_keys),
        {k: sel[k] for k in read_keys},
    )
    defaults = base_config()["selection"]
    non_default = {k: sel[k] for k in ignored_keys if sel[k] != defaults[k]}
    if non_default:
        logger.warning(
            "mode=%s ignores these selection keys, which are set to non-default values: %s",
            cfg["mode"],
            non_default,
        )


def cmd_run(args) -> int:
    cfg = resolve_config(args)
    run_dir = run_directory(cfg)
    _configure_logging(run_dir)
    touch_run_dir(cfg, run_dir)
    log_selection_keys(cfg)

    all_tasks = tasks_mod.enumerate_tasks(cfg)
    return execute(cfg, run_dir, all_tasks, only=parse_pairs(args.only))


def execute(cfg: dict, run_dir: Path, all_tasks, *, only: dict | None = None, designs=None) -> int:
    """Run this process's shard of ``all_tasks`` and, when unsharded, aggregate.

    Shared by ``ra run`` and ``ra evaluate``: the two differ only in what they
    enumerate (as-built blocks vs a set of designs' cases) and in the preflight
    the evaluation runs first.
    """
    selected = tasks_mod.select_shard(all_tasks, cfg["execution"]["shard"])
    selected = tasks_mod.filter_tasks(selected, only)
    if cfg["execution"]["max_tasks"] is not None:
        selected = selected[: int(cfg["execution"]["max_tasks"])]

    logger.info(
        "run %s: %d of %d tasks in this shard (%s)",
        run_id(cfg),
        len(selected),
        len(all_tasks),
        cfg["execution"]["shard"] or "all",
    )
    records = tasks_mod.run_tasks(
        selected, cfg, run_dir, force=bool(cfg["execution"]["force"]), designs=designs
    )

    # run_task records the provenance from whichever process built the system
    # (the reference solve builds it in a child); this is the in-process fallback.
    meta = system_mod.last_meta()
    if meta and not (run_dir / "system_meta.json").exists():
        with open(run_dir / "system_meta.json", "w") as f:
            json.dump(meta, f, indent=2, sort_keys=True, default=str)

    failures = tasks_mod.failed_required(records, cfg)
    optional_failures = [
        r for r in records if r.get("status") != tasks_mod.STATUS_OK and r not in failures
    ]
    if optional_failures:
        logger.warning(
            "%d optional task(s) did not succeed (they do not fail the run): %s",
            len(optional_failures),
            ", ".join(r["task_id"] for r in optional_failures[:10]),
        )

    if cfg["execution"]["shard"] is None:
        frame = metrics_mod.aggregate(run_dir, cfg)
        # The evaluation tables come *before* the card: the card's ranking
        # section reads `eval_summary.parquet` off disk (WP-E4), so writing it
        # afterwards would print the previous invocation's ranking, or none.
        write_eval_tables(run_dir, frame, cfg)
        runcard.write_card(run_dir, cfg, frame)
        logger.info("wrote %s and %s", run_dir / "metrics.csv", run_dir / "CARD.md")
        write_debug_figures(run_dir, cfg)
    else:
        logger.info("shard finished; run `aggregate` once every shard is done")

    if failures:
        logger.error(
            "%d required task(s) failed: %s",
            len(failures),
            ", ".join(r["task_id"] for r in failures[:10]),
        )
        return 1
    return 0


def cmd_evaluate(args) -> int:
    """``ra evaluate``: the single-source-of-truth scoring of a set of designs.

    Order matters and is the whole point of the command: resolve the config,
    read the designs, **preflight** (config gates, outage-store coverage, pool
    capacity, as-built match) and only then enumerate a single task.  A campaign
    that discovers a missing outage chunk on task 4,000 of 26,000 has wasted
    hours; every failure this command can foresee is raised before the first
    solve (spec 2.1, E9).
    """
    from . import evaluate as evaluate_mod

    cfg = resolve_config(args)
    run_dir = run_directory(cfg)
    _configure_logging(run_dir)

    sources = evaluate_mod.load_designs(
        args.design_runs, args.design_files, cfg, runs_root=cfg["output"].get("runs_root")
    )
    report = evaluate_mod.preflight(sources, cfg, run_dir=run_dir)
    if not report["ok"]:
        failed = [c for c in report["checks"] if not c["ok"]]
        raise ConfigError(
            "evaluation preflight failed; refusing to run "
            f"({run_dir / evaluate_mod.PREFLIGHT_NAME}):\n  "
            + "\n  ".join(f"{c['name']}: {c['detail']}" for c in failed)
        )
    if args.preflight_only:
        print(f"preflight OK -> {run_dir / evaluate_mod.PREFLIGHT_NAME}")
        return 0

    touch_run_dir(cfg, run_dir)
    evaluate_mod.record_splits(run_dir, cfg)
    evaluate_mod.record_sources(run_dir, sources)
    log_selection_keys(cfg)

    designs = {s.design_id: s.design for s in sources}
    all_tasks = tasks_mod.enumerate_tasks(cfg, design_ids=tuple(designs))
    return execute(cfg, run_dir, all_tasks, only=parse_pairs(args.only), designs=designs)


def cmd_aggregate(args) -> int:
    if args.run_id:
        run_dir = paths.run_dir(args.run_id, args.runs_root)
        resolved = run_dir / "config.resolved.yaml"
        if not resolved.exists():
            print(f"no config.resolved.yaml in {run_dir}", file=sys.stderr)
            return 1
        cfg = load_config(resolved)
    else:
        if not args.config:
            print("aggregate needs --run-id or --config", file=sys.stderr)
            return 1
        cfg = resolve_config(args)
        run_dir = run_directory(cfg)

    frame = metrics_mod.aggregate(run_dir, cfg)
    write_eval_tables(run_dir, frame, cfg)  # before the card; see `execute`
    card = runcard.write_card(run_dir, cfg, frame)
    print(f"{len(frame)} task rows -> {run_dir / 'metrics.csv'}")
    print(f"card -> {card}")
    write_debug_figures(run_dir, cfg)
    return 0


def write_eval_tables(run_dir: Path, frame, cfg: dict | None = None) -> None:
    """Rebuild ``eval.parquet`` at aggregate time for an evaluation run.

    A sharded campaign never calls ``evaluate.evaluate_designs`` in one process,
    so the only place the whole ledger is visible is ``ra aggregate``.  Skipped
    for a benchmark run (nothing but ``asbuilt``), and never fatal.
    """
    from . import evaluate as evaluate_mod

    if not evaluate_mod.is_evaluation_run(frame, cfg):
        return
    try:
        eval_path, _profile = evaluate_mod.write_eval_tables(run_dir)
        logger.info("wrote %s", eval_path)
    except Exception as exc:  # a summary table must never fail an aggregate
        logger.warning("could not write the evaluation tables: %s", exc, exc_info=True)


def write_debug_figures(run_dir: Path, cfg: dict) -> None:
    """The ``output.figures`` hook: debug figures into ``<run_dir>/figures/``.

    Never fatal -- a missing matplotlib backend or a plot that has no data must
    not fail a run whose numbers are already on disk.
    """
    if not (cfg.get("output") or {}).get("figures", False):
        return
    try:
        from .plots.cli import debug_figures

        written = debug_figures(run_dir, cfg)
        logger.info("wrote %d debug figure(s) to %s", len(written), run_dir / "figures")
    except Exception as exc:  # see the docstring: this is never fatal
        logger.warning("could not write the debug figures: %s", exc, exc_info=True)


def cmd_show(args) -> int:
    run_dir = paths.run_dir(args.run_id, args.runs_root)
    if not run_dir.exists():
        print(f"no such run directory: {run_dir}", file=sys.stderr)
        return 1
    card = run_dir / "CARD.md"
    if card.exists():
        print(card.read_text())
    else:
        print(f"(no CARD.md in {run_dir})")
    frame = metrics_mod.records_to_frame(metrics_mod.read_task_records(run_dir))
    if not frame.empty:
        print("Task status counts:")
        print(frame["status"].value_counts().to_string())
    return 0


def cmd_plot(args) -> int:
    from .plots.cli import cmd_plot as _cmd_plot

    return _cmd_plot(args)


def cmd_design(args) -> int:
    run_dir = paths.run_dir(args.run_id, args.runs_root)
    if not run_dir.exists():
        print(f"no such run directory: {run_dir}", file=sys.stderr)
        return 1
    from .planning.design import design_paths, read_design_record

    found = design_paths(run_dir)
    if args.design_id:
        found = [p for p in found if p.stem == args.design_id]
    if not found:
        print(f"no designs in {run_dir / 'designs'}", file=sys.stderr)
        return 1

    for path in found:
        record = read_design_record(path)
        print(f"# {record.get('design_id')}  ({path})")
        print(f"  schema_version: {record.get('schema_version')}")
        print(
            f"  method:         {record.get('method')} (preset {record.get('preset')}, "
            f"kind {record.get('kind')})"
        )
        print(
            f"  dataset:        {record.get('dataset')} years={record.get('years')} "
            f"window={record.get('window')}"
        )
        print(
            f"  selection:      {record.get('selection', {}).get('strategy')} "
            f"block_size={record.get('selection', {}).get('block_size')} "
            f"n_blocks={len(record.get('selection', {}).get('blocks') or [])}"
        )
        print(f"  annualization:  {record.get('annualization')}")
        print(f"  objective:      {record.get('objective')}")
        print(f"  emissions:      {record.get('emissions')}")
        print(f"  solver:         {record.get('solver')}")
        print(f"  timing:         {record.get('timing')}")
        for cls_name, entry in (record.get("capacities") or {}).items():
            attr = next((k for k in entry if k != "names"), None)
            values = entry.get(attr) or []
            total = sum(float(v) for v in values)
            print(f"  {cls_name:<14s} {len(values):4d} rows, total {attr} = {total:,.4g}")
        print()
    return 0


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.command not in ("run", "evaluate"):
        _configure_logging()
    handlers = {
        "plan": cmd_plan,
        "run": cmd_run,
        "evaluate": cmd_evaluate,
        "aggregate": cmd_aggregate,
        "show": cmd_show,
        "design": cmd_design,
        "plot": cmd_plot,
    }
    return handlers[args.command](args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
