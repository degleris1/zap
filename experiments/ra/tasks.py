"""Tasks: the smallest independently re-runnable unit of a run, and the ledger.

A task is one block solve, by one method, for one (year, draw) and one design.
Each task writes exactly one JSON file, written to ``<task>.json.tmp`` and
``os.replace``d, so a killed process never leaves a half-written result and a
task whose file exists is skipped on the next invocation (resumability).
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
import os
import time
import traceback
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import dispatch
from .blocks import REFERENCE, Block, make_blocks
from .config import is_plan_mode, parse_shard

logger = logging.getLogger(__name__)

METHOD_ORDER = ("lp", "admm")

#: ``task.method`` of a ``mode: plan`` task.  A planning run enumerates one task
#: per outage draw -- the whole capacity-expansion solve -- through this same
#: ledger (D-W1), so `method` is not a `cfg["methods"]` key and the timeout and
#: `required` flag come from ``cfg["planning"]`` instead.
PLAN_METHOD = "plan"

#: ``selection.block_size: null`` (the monolithic sentinel) in a task id.
FULL_HORIZON = "full"

STATUS_OK = "ok"
STATUS_FAILED = "failed"
STATUS_TIMEOUT = "timeout"
#: The solve finished but its answer is not usable (an ADMM iterate that misses
#: nodal power balance by more than `methods.admm.max_imbalance_mw`). Metrics are
#: kept on the record; `runcard.summarize` excludes the row from the headline.
STATUS_INFEASIBLE = "infeasible"


#: Worst-first precedence used to fold a case's block statuses into one status
#: (WP-E3): a case is ``ok`` only if every one of its blocks is.
_STATUS_SEVERITY = {
    STATUS_OK: 0,
    STATUS_INFEASIBLE: 1,
    STATUS_TIMEOUT: 2,
    STATUS_FAILED: 3,
}

#: ``execution.task_granularity`` values.  ``block`` (the default) is one task
#: per block solve, which every phase-1 dispatch benchmark ran with; ``case`` is
#: one task per (design, year, draw), which is the natural unit of an evaluation
#: row and keeps the file count survivable (spec 2.2 / D1).
GRANULARITY_BLOCK = "block"
GRANULARITY_CASE = "case"
VALID_GRANULARITIES = (GRANULARITY_BLOCK, GRANULARITY_CASE)


def task_granularity(cfg: dict) -> str:
    return str((cfg.get("execution") or {}).get("task_granularity", GRANULARITY_BLOCK))


def record_granularity(record: dict) -> str:
    """``case`` for a nested record, ``block`` for a per-block one."""
    return GRANULARITY_CASE if isinstance(record.get("blocks"), list) else GRANULARITY_BLOCK


def existing_granularity(run_dir: Path) -> str | None:
    """The granularity of the task files already in ``run_dir``, or ``None``.

    ``execution`` is excluded from the run-id hash (changing how a run is cut
    into tasks must not renumber it), so one run directory accepts an
    invocation of either granularity.  Left alone, a ``block`` run followed by a
    ``case`` run leaves *both* files for the same blocks in ``tasks/`` --
    resume-skip only ever looks for its own file name -- and
    ``metrics.read_task_records`` then yields every block twice: doubled sums
    and ``coverage == 2`` (verifier F1, 2026-09-09).  So the directory, not the
    config, decides: :func:`experiments.ra.cli.touch_run_dir` adopts what is
    already there.  Reads one file, not the whole ledger, so this stays O(1) on
    a 3 M-task campaign.
    """
    for path in sorted(tasks_dir(run_dir).glob("*.json")):
        try:
            with open(path, "r") as f:
                record = json.load(f)
        except (OSError, json.JSONDecodeError):  # pragma: no cover - defensive
            continue
        if isinstance(record, dict):
            return record_granularity(record)
    return None


@dataclass(frozen=True)
class Task:
    task_id: str
    method: str
    block: Block
    block_size: int | str
    draw: int | None = None
    design_id: str = "asbuilt"


@dataclass(frozen=True)
class CaseTask:
    """One (design, year, draw) case: its blocks solved in one process (WP-E3).

    Every number in ``eval.parquet`` is a sum over a case's blocks and every
    reliability average is over cases, so the case is the natural atom of an
    evaluation.  It is also one system *build* by construction rather than by
    cache luck at a shard boundary.  The record it writes nests today's
    per-block records under ``blocks``, and ``metrics.read_task_records``
    flattens them, so ``metrics.csv`` stays exactly per block.
    """

    task_id: str
    method: str
    block_size: int | str
    year: int
    draw: int | None
    design_id: str
    tasks: tuple[Task, ...]

    @property
    def block(self) -> Block:
        """The span the case covers, for ``filter_tasks`` and ``cli plan``."""
        return Block(
            index=0,
            year=self.year,
            start=min(t.block.start for t in self.tasks),
            stop=max(t.block.stop for t in self.tasks),
        )


def make_task_id(
    *, design_id: str, method: str, block_size: int | str, year: int, index: int, draw: int | None
) -> str:
    task_id = f"{design_id}-{method}-{block_size}-y{year}-b{index:05d}"
    if draw is not None:
        task_id += f"-d{draw}"
    return task_id


def make_plan_task_id(
    *, preset: str, strategy: str, block_size, seed: int, draw: int | None, batch_size: int = 0
) -> str:
    """The task id -- and hence the design id -- of one planning solve.

    ``batch_size`` is part of the id because a deterministic gradient run
    (``batch_size: 0``) and a stochastic one over the same block pool differ in
    nothing else, and ``evaluate.evaluate_designs`` requires unique design ids.
    A zero batch size adds no suffix, so the ids of the single-level presets are
    unchanged.  Gradient / ADMM configs are **not**: ``methods/plan_gradient.yaml``
    and ``methods/plan_admm.yaml`` set ``batch_size: 4``, so every config that
    inherits from them gains a ``-B4`` and its design id (and hence its run id,
    which hashes the config) changes.  No planning run existed when this landed,
    so nothing was invalidated.
    """
    size = FULL_HORIZON if block_size is None else int(block_size)
    task_id = f"plan-{preset}-{strategy}-b{size}-s{int(seed)}"
    if int(batch_size or 0) > 0:
        task_id += f"-B{int(batch_size)}"
    if draw is not None:
        task_id += f"-d{draw}"
    return task_id


def enumerate_plan_tasks(cfg: dict) -> list[Task]:
    """The tasks of a ``mode: plan`` run: one whole planning solve per draw."""
    draws = [int(d) for d in cfg["heuristics"]["outage_draws"]] or [None]
    sel = cfg["selection"]
    plan = cfg["planning"]
    start, stop = int(cfg["dataset"]["window"]["start"]), int(cfg["dataset"]["window"]["stop"])
    year = int(cfg["dataset"]["years"][0])
    block_size = sel["block_size"]
    size: int | str = FULL_HORIZON if block_size is None else int(block_size)

    tasks = []
    for draw in draws:
        task_id = make_plan_task_id(
            preset=plan["method"],
            strategy=sel["strategy"],
            block_size=block_size,
            seed=sel["seed"],
            draw=draw,
            batch_size=(plan.get("optimizer") or {}).get("batch_size", 0),
        )
        tasks.append(
            Task(
                task_id=task_id,
                method=PLAN_METHOD,
                # The "block" of a planning task is the whole loaded window; it
                # is what the ledger records as `start` / `stop` / `hours`.
                block=Block(index=0, year=year, start=start, stop=stop),
                block_size=size,
                draw=draw,
                # The design this task produces is named after the task (3.4).
                design_id=task_id,
            )
        )
    return tasks


def make_case_task_id(
    *, design_id: str, method: str, block_size: int | str, year: int, draw: int | None
) -> str:
    """The task id of one case, e.g. ``asbuilt-lp-168-y2020-d7`` (spec 2.2).

    No collision with :func:`make_task_id`: a block id always carries a
    ``-bNNNNN`` segment that a case id never has.
    """
    task_id = f"{design_id}-{method}-{block_size}-y{year}"
    if draw is not None:
        task_id += f"-d{draw}"
    return task_id


def _block_sizes(cfg: dict) -> list[int | str]:
    block_sizes: list[int | str] = [int(b) for b in cfg["selection"]["blocks"]]
    if cfg["selection"]["reference"] != "none":
        block_sizes.append(REFERENCE)
    return block_sizes


def enumerate_case_tasks(cfg: dict, design_ids: Sequence[str] = ("asbuilt",)) -> list[CaseTask]:
    """Case tasks in **design-inner** order: year, draw, design, method, size.

    Design-inner is what makes WP-E0's unit-slice cache pay: each (year, draw)
    zarr slice is decompressed once per process and re-weighted per design,
    instead of once per (design, draw) (spec 2.2).  Shards are contiguous over
    this list, so a shard never splits a case, and with ``n_shards`` dividing the
    number of draws it never splits a (year, draw) group either.
    """
    draws = [int(d) for d in cfg["heuristics"]["outage_draws"]] or [None]
    years = [int(y) for y in cfg["dataset"]["years"]]
    sizes = _block_sizes(cfg)
    blocks_by_size = {size: make_blocks(cfg, size) for size in sizes}

    cases: list[CaseTask] = []
    for year in years:
        for draw in draws:
            for design_id in design_ids:
                for method in METHOD_ORDER:
                    if not cfg["methods"][method]["enabled"]:
                        continue
                    for block_size in sizes:
                        blocks = [b for b in blocks_by_size[block_size] if b.year == year]
                        if not blocks:
                            continue
                        tasks = tuple(
                            Task(
                                task_id=make_task_id(
                                    design_id=design_id,
                                    method=method,
                                    block_size=block_size,
                                    year=block.year,
                                    index=block.index,
                                    draw=draw,
                                ),
                                method=method,
                                block=block,
                                block_size=block_size,
                                draw=draw,
                                design_id=design_id,
                            )
                            for block in blocks
                        )
                        cases.append(
                            CaseTask(
                                task_id=make_case_task_id(
                                    design_id=design_id,
                                    method=method,
                                    block_size=block_size,
                                    year=year,
                                    draw=draw,
                                ),
                                method=method,
                                block_size=block_size,
                                year=year,
                                draw=draw,
                                design_id=design_id,
                                tasks=tasks,
                            )
                        )
    return cases


def enumerate_tasks(cfg: dict, design_ids: Sequence[str] = ("asbuilt",)) -> list:
    """All tasks of a run, in a deterministic order.

    ``execution.task_granularity: case`` returns :class:`CaseTask` objects
    instead of one :class:`Task` per block; ``execution`` is excluded from the
    run-id hash, so the granularity changes the *files* a run writes and never
    its numbers (the equivalence is spec section 5.7).
    """
    if is_plan_mode(cfg):
        return enumerate_plan_tasks(cfg)
    if task_granularity(cfg) == GRANULARITY_CASE:
        return enumerate_case_tasks(cfg, design_ids)
    draws = [int(d) for d in cfg["heuristics"]["outage_draws"]] or [None]
    block_sizes: list[int | str] = [int(b) for b in cfg["selection"]["blocks"]]
    if cfg["selection"]["reference"] != "none":
        block_sizes.append(REFERENCE)

    tasks: list[Task] = []
    for design_id in design_ids:
        for method in METHOD_ORDER:
            if not cfg["methods"][method]["enabled"]:
                continue
            for block_size in block_sizes:
                for draw in draws:
                    for block in make_blocks(cfg, block_size):
                        tasks.append(
                            Task(
                                task_id=make_task_id(
                                    design_id=design_id,
                                    method=method,
                                    block_size=block_size,
                                    year=block.year,
                                    index=block.index,
                                    draw=draw,
                                ),
                                method=method,
                                block=block,
                                block_size=block_size,
                                draw=draw,
                                design_id=design_id,
                            )
                        )
    return tasks


def select_shard(tasks: Sequence[Task], shard: str | None) -> list[Task]:
    """Contiguous, disjoint shards whose union is the whole task list."""
    if shard is None:
        return list(tasks)
    k, n = parse_shard(shard)
    total = len(tasks)
    base, extra = divmod(total, n)
    start = (k - 1) * base + min(k - 1, extra)
    stop = start + base + (1 if k <= extra else 0)
    return list(tasks[start:stop])


def filter_tasks(tasks: Iterable[Task], only: dict[str, str] | None) -> list[Task]:
    """Filter by simple ``field=value`` predicates (e.g. ``method=lp``)."""
    tasks = list(tasks)
    if not only:
        return tasks
    out = []
    for task in tasks:
        keep = True
        for key, value in only.items():
            actual = getattr(task, key, None)
            if actual is None:
                actual = getattr(task.block, key, None)
            if str(actual) != str(value):
                keep = False
                break
        if keep:
            out.append(task)
    return out


def tasks_dir(run_dir: Path) -> Path:
    return Path(run_dir) / "tasks"


def task_path(run_dir: Path, task: Task) -> Path:
    return tasks_dir(run_dir) / f"{task.task_id}.json"


def is_done(run_dir: Path, task: Task) -> bool:
    path = task_path(run_dir, task)
    if not path.exists():
        return False
    try:
        with open(path, "r") as f:
            record = json.load(f)
    except (OSError, json.JSONDecodeError):  # pragma: no cover - defensive
        return False
    return bool(record.get("status"))


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json_atomic(path: Path, payload: dict) -> Path:
    """Serialize first, then write a temp file and ``os.replace`` it into place."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = json.dumps(payload, indent=2, sort_keys=True, default=str)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        with open(tmp, "w") as f:
            f.write(blob)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        if tmp.exists():
            tmp.unlink()
        raise
    return path


def task_timeout_s(task: Task, cfg: dict) -> float:
    """The timeout that applies to a task, from whichever config block owns it."""
    if task.method == PLAN_METHOD:
        return float(cfg["planning"]["timeout_s"])
    return float(cfg["methods"][task.method]["timeout_s"])


def task_is_required(method: str, cfg: dict) -> bool:
    if method == PLAN_METHOD:
        return bool(cfg["planning"].get("required", True))
    return bool(cfg["methods"][method].get("required", True))


def solve_plan(task: Task, cfg: dict, run_dir: Path) -> dict:
    """Run one capacity-expansion solve and write its ``design.json`` (3.4).

    Imported lazily: the planning core pulls in cvxpy, torch and WP1's reader,
    none of which a dispatch run needs.
    """
    from . import metrics as metrics_mod
    from . import planning as planning_mod
    from . import system as system_mod
    from .identity import run_id

    loaded = system_mod.build_system(cfg, draw=task.draw)

    started = time.perf_counter()
    result = planning_mod.plan(loaded, cfg)
    solve_wall = time.perf_counter() - started

    result.design_id = task.design_id
    result.run_id = run_id(cfg)
    design_path = result.write(run_dir)

    iteration_paths: list[str] = []
    if (cfg.get("output") or {}).get("save_iterations", True):
        from .planning.history import write_iteration_tables

        try:
            iteration_paths = [
                str(Path(p).relative_to(run_dir))
                for p in write_iteration_tables(result, run_dir, cfg, task=task)
            ]
        except Exception as exc:  # never lose a design over a summary table
            logger.warning("could not write the iteration tables: %s", exc, exc_info=True)

    try:
        from . import persist

        persist.write_system_static(run_dir, loaded, cfg)
    except Exception as exc:  # noqa: BLE001 - provenance is best-effort
        logger.warning("could not write system_static.json: %s", exc)

    payload_metrics = metrics_mod.planning_metrics(result)
    payload_metrics["solve_wall_clock_s"] = solve_wall
    return {
        "metrics": payload_metrics,
        "solver_status": result.solver.get("status"),
        "n_variables": result.solver.get("n_variables"),
        "n_constraints": result.solver.get("n_constraints"),
        "design_path": str(Path(design_path).relative_to(run_dir)),
        "iteration_paths": iteration_paths,
        "system_meta": dict(getattr(loaded, "meta", {}) or {}),
    }


def _solve_entry(task: Task, cfg: dict, design=None, run_dir=None) -> dict:
    """Module-level entry point so the reference solve can run in a subprocess."""
    return dispatch.solve_block(task, cfg, design=design, run_dir=run_dir)


def _run_in_subprocess(fn, args: tuple, timeout_s: float, label: str) -> dict:
    """Run ``fn(*args)`` in a worker process, killed at ``timeout_s``."""
    executor = concurrent.futures.ProcessPoolExecutor(max_workers=1)
    try:
        future = executor.submit(fn, *args)
        return future.result(timeout=timeout_s)
    except concurrent.futures.TimeoutError as exc:
        for process in list(getattr(executor, "_processes", {}).values()):
            process.kill()
        raise TimeoutError(f"{label} exceeded timeout_s={timeout_s}") from exc
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


def _use_subprocess(task: Task, cfg: dict) -> bool:
    """Isolate the (large, possibly runaway) reference solve behind a timeout."""
    if task.method == PLAN_METHOD:  # handled directly by `run_task`
        return False
    if str(task.block_size) != REFERENCE:
        return False
    if dispatch.is_stub(cfg, task.method):
        # The stub is instantaneous, so it stays in-process (and stays
        # monkeypatchable) unless it was deliberately told to sleep.
        return dispatch.stub_sleep_s(cfg, task.method) > 0
    return True


def _solve_with_timeout(task: Task, cfg: dict, timeout_s: float, design=None, run_dir=None) -> dict:
    return _run_in_subprocess(
        _solve_entry, (task, cfg, design, run_dir), timeout_s, label=task.task_id
    )


def run_task(task: Task, cfg: dict, run_dir: Path, design=None) -> dict[str, Any]:
    """Run one task and write its result file. Never raises on a solver failure."""
    record = block_record(task, cfg, run_dir, design=design)
    write_json_atomic(task_path(run_dir, task), record)
    return record


def block_record(task: Task, cfg: dict, run_dir: Path, design=None) -> dict[str, Any]:
    """Solve one block and return its ledger record, **without writing a file**.

    Split out of :func:`run_task` so a case task (WP-E3) can produce exactly
    today's per-block records and nest them, which is what keeps ``metrics.csv``
    byte-for-byte identical across the two granularities.
    """
    run_dir = Path(run_dir)
    timeout_s = task_timeout_s(task, cfg)

    started = _utcnow()
    t0 = time.perf_counter()
    status = STATUS_OK
    error = None
    tb = None
    payload: dict[str, Any] = {}

    try:
        if task.method == PLAN_METHOD:
            # A planning solve runs in a worker process so that
            # `planning.timeout_s` is *enforced* -- a gradient loop or an LP that
            # runs past the budget is killed at the cap, not merely reported late
            # afterwards. The child writes `designs/<id>.json` itself and returns
            # the same payload the in-process path did.
            payload = _run_in_subprocess(
                solve_plan, (task, cfg, run_dir), timeout_s, label=task.task_id
            )
        elif _use_subprocess(task, cfg):
            payload = _solve_with_timeout(task, cfg, timeout_s, design, run_dir)
        else:
            payload = dispatch.solve_block(task, cfg, design=design, run_dir=run_dir)
    except TimeoutError as exc:
        status = STATUS_TIMEOUT
        error = str(exc)
        tb = traceback.format_exc()
    except Exception as exc:  # noqa: BLE001 - a failed task must not kill the run
        status = STATUS_FAILED
        error = f"{type(exc).__name__}: {exc}"
        tb = traceback.format_exc()
        logger.warning("task %s failed: %s", task.task_id, error)

    # A solver may report that its answer is unusable without raising.
    if status == STATUS_OK and payload.get("status"):
        status = str(payload["status"])
        error = payload.get("error") or error

    wall = time.perf_counter() - t0
    if status == STATUS_OK and wall > timeout_s:
        status = STATUS_TIMEOUT
        error = f"{task.task_id} took {wall:.1f}s > timeout_s={timeout_s}"

    # A solve that finished late is marked `timeout` but keeps its metrics.
    metrics = payload.get("metrics", {})
    record = {
        "task_id": task.task_id,
        "status": status,
        "method": task.method,
        "block_size": task.block_size,
        "year": task.block.year,
        "block_index": task.block.index,
        "start": task.block.start,
        "stop": task.block.stop,
        "hours": task.block.hours,
        "draw": task.draw,
        "design_id": task.design_id,
        "design_path": payload.get("design_path"),
        # Per-task artefacts written by `persist` (relative to run_dir), so the
        # ledger says exactly which files belong to this task.
        "hourly_path": payload.get("hourly_path"),
        "ens_profile_path": payload.get("ens_profile_path"),
        "admm_trace_path": payload.get("admm_trace_path"),
        "iteration_paths": payload.get("iteration_paths"),
        "metrics": metrics,
        "wall_clock_s": wall,
        "solver_status": payload.get("solver_status"),
        "n_variables": payload.get("n_variables"),
        "n_constraints": payload.get("n_constraints"),
        "admm_iterations": payload.get("admm_iterations"),
        "admm_primal_residual": payload.get("admm_primal_residual"),
        "admm_dual_residual": payload.get("admm_dual_residual"),
        # Inner-prox and dual-accuracy diagnostics (A3). The metrics dict already
        # flattens into `metrics.csv` via `records_to_frame`; these top-level
        # mirrors are for the ledger, which is read without the metrics block.
        "admm_max_inner_prox_residual_mw": payload.get("admm_max_inner_prox_residual_mw"),
        "admm_price_error_max_usd_per_mwh": payload.get("admm_price_error_max_usd_per_mwh"),
        "error": error,
        "traceback": tb,
        "started_utc": started,
        "finished_utc": _utcnow(),
    }
    _record_system_meta(run_dir, payload.get("system_meta"))
    return record


def case_status(block_records: Sequence[dict]) -> str:
    """The worst block status of a case: ``ok`` only when every block is ``ok``.

    Keeps the existing "never average a failed row" rule intact -- the failing
    block keeps its own status in the flattened ``metrics.csv`` and the other
    blocks keep theirs, so one bad block does not throw away 51 good rows.
    """
    worst = STATUS_OK
    for record in block_records:
        status = str(record.get("status") or STATUS_OK)
        if _STATUS_SEVERITY.get(status, 3) > _STATUS_SEVERITY.get(worst, 0):
            worst = status
    return worst


def _prebuild_system(case: CaseTask, cfg: dict, design=None) -> tuple[float, str | None]:
    """Build the case's system once and time it; ``(seconds, error)``.

    The blocks then hit ``system.build_system``'s process cache, so the case's
    build cost is measured rather than smeared over 52 block solves (spec 2.2 and
    the ``build_wall_clock_s`` column of section 4.1).  A failure here is *not*
    raised: the per-block solves below will fail with the same error and be
    recorded one by one, exactly as they would under block granularity.
    """
    if dispatch.is_stub(cfg, case.method):
        return 0.0, None
    from . import system as system_mod

    if not system_mod.WP1_AVAILABLE:  # pragma: no cover - defensive
        return 0.0, None
    t0 = time.perf_counter()
    try:
        system_mod.build_system(cfg, draw=case.draw, design=design)
    except Exception as exc:  # noqa: BLE001 - see the docstring
        logger.warning("case %s could not build its system: %s", case.task_id, exc)
        return time.perf_counter() - t0, f"{type(exc).__name__}: {exc}"
    return time.perf_counter() - t0, None


def run_case_task(case: CaseTask, cfg: dict, run_dir: Path, design=None) -> dict[str, Any]:
    """Solve every block of one case in this process and write **one** record.

    The record nests the per-block records under ``blocks``;
    ``metrics.read_task_records`` flattens them, so ``eval.parquet``,
    ``deviation_vs_reference`` and every plot are unchanged (spec 2.2).
    """
    run_dir = Path(run_dir)
    started = _utcnow()
    t0 = time.perf_counter()

    build_wall, build_error = _prebuild_system(case, cfg, design=design)
    records = []
    for i, task in enumerate(case.tasks):
        record = block_record(task, cfg, run_dir, design=design)
        # The build is charged to the case, so only the first block carries it:
        # `eval.parquet` sums this column over a case's blocks.
        record["build_wall_clock_s"] = float(build_wall) if i == 0 else 0.0
        records.append(record)

    status = case_status(records)
    errors = [r.get("error") for r in records if r.get("error")]
    payload = {
        "task_id": case.task_id,
        "status": status,
        "method": case.method,
        "block_size": case.block_size,
        "year": case.year,
        "draw": case.draw,
        "design_id": case.design_id,
        "n_blocks": len(records),
        "n_blocks_ok": sum(1 for r in records if r.get("status") == STATUS_OK),
        "wall_clock_s": time.perf_counter() - t0,
        "build_wall_clock_s": float(build_wall),
        "build_error": build_error,
        "error": errors[0] if errors else None,
        "n_errors": len(errors),
        "started_utc": started,
        "finished_utc": _utcnow(),
        "blocks": records,
    }
    write_json_atomic(task_path(run_dir, case), payload)
    return payload


def _record_system_meta(run_dir: Path, meta: dict | None) -> None:
    """Persist the dataset/demand-scaling provenance from whichever process has it.

    The reference solve runs in a child process, so a reference-only run would
    otherwise leave the card without a demand-scaling block (verifier, item 2).
    """
    if not meta:
        return
    path = Path(run_dir) / "system_meta.json"
    if path.exists():
        return
    write_json_atomic(path, dict(meta))


def run_tasks(
    tasks: Sequence[Task],
    cfg: dict,
    run_dir: Path,
    *,
    force: bool = False,
    designs: dict | None = None,
) -> list[dict]:
    """Run a list of tasks, skipping those already done unless ``force``.

    Accepts :class:`Task` and :class:`CaseTask` entries; resume-skip is at the
    granularity of whatever was enumerated, so a finished case is skipped whole
    and an unfinished one re-runs all of its blocks.
    """
    records = []
    for i, task in enumerate(tasks, start=1):
        if not force and is_done(run_dir, task):
            logger.info("[%d/%d] skipping completed task %s", i, len(tasks), task.task_id)
            continue
        logger.info("[%d/%d] running task %s", i, len(tasks), task.task_id)
        design = (designs or {}).get(task.design_id)
        if isinstance(task, CaseTask):
            records.append(run_case_task(task, cfg, run_dir, design=design))
        else:
            records.append(run_task(task, cfg, run_dir, design=design))
    return records


def failed_required(records: Iterable[dict], cfg: dict) -> list[dict]:
    """Records that fail the run: a hard failure of a method marked ``required``.

    A ``timeout`` is *not* a required failure: the full-year reference solve is
    expected to hit its cap (spec R2), and the run must still complete and say so
    on the card (verifier, item 7).
    """
    return [
        r
        for r in records
        if r.get("status") == STATUS_FAILED and task_is_required(r["method"], cfg)
    ]
