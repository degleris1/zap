"""Block solves: LP (cvxpy), ADMM, and a deterministic stub.

Every solver returns the same payload dict:

``{"metrics": {...}, "solver_status": str, "n_variables": int | None,
   "n_constraints": int | None, "admm_iterations": int | None,
   "admm_primal_residual": float | None, "admm_dual_residual": float | None}``

The ``STUB`` solver returns fabricated but deterministic metrics without
touching the data or a solver, so the whole pipeline (config -> run id -> tasks
-> ledger -> metrics.csv -> CARD.md -> resume) can be exercised without WP1.
"""

from __future__ import annotations

import hashlib
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

from . import persist
from . import system as system_mod
from .blocks import prorate_energy_budgets, window_bounds
from .metrics import block_metrics, numpyify

logger = logging.getLogger(__name__)

STUB_SOLVER = "STUB"

#: Cost of one hour of a stubbed block, and the penalty charged per block, so
#: that summing many small blocks costs more than one big block -- the
#: qualitative property the real benchmark measures.
STUB_COST_PER_HOUR = 1000.0
STUB_COST_PER_BLOCK = 250.0


#: Default hours per weather year, used only if the store does not say.
DEFAULT_HOURS_PER_YEAR = 8760


def is_stub(cfg: dict, method: str) -> bool:
    return str(cfg["methods"][method]["solver"]).upper() == STUB_SOLVER


def stub_sleep_s(cfg: dict, method: str) -> float:
    """Test hook: make the stub solver sleep, to exercise the timeout path."""
    kwargs = cfg["methods"][method].get("solver_kwargs") or {}
    return float(kwargs.get("stub_sleep_s", 0.0))


def hours_per_year(loaded) -> int:
    """Hours in one weather year, from the store's provenance (D12), else 8760.

    This is the denominator for pro-rating an annual energy budget to a block --
    the window length is *not* a year and must not be used (verifier, item 5).
    """
    meta = getattr(loaded, "meta", None) or {}
    store_attrs = meta.get("weather_store_attrs") or {}
    for source in (store_attrs, meta):
        value = source.get("hours_per_year")
        if value:
            return int(value)
    return DEFAULT_HOURS_PER_YEAR


def block_time_periods(cfg: dict, block) -> np.ndarray:
    """Indices of the block's hours inside the loaded (multi-year) window."""
    win_start, win_stop = window_bounds(cfg)
    window_len = win_stop - win_start
    years = [int(y) for y in cfg["dataset"]["years"]]
    if block.year not in years:
        raise ValueError(f"block year {block.year} is not in dataset.years {years}")
    offset = years.index(block.year) * window_len + (block.start - win_start)
    if block.start < win_start or block.stop > win_stop:
        raise ValueError(f"block {block} is not inside the window [{win_start}, {win_stop})")
    return np.arange(offset, offset + block.hours)


def slice_devices(loaded, cfg: dict, block, design=None) -> list:
    """Per-block devices: sample the window's devices down to the block's hours."""
    devices = design.apply(loaded) if design is not None else loaded.devices
    win_start, win_stop = window_bounds(cfg)
    total_hours = (win_stop - win_start) * len(cfg["dataset"]["years"])
    periods = block_time_periods(cfg, block)
    sliced = [d.sample_time(periods, total_hours) for d in devices]
    check_block_horizon(sliced, block.hours)
    return prorate_energy_budgets(sliced, block.hours, hours_per_year(loaded))


def check_block_horizon(devices, hours: int) -> None:
    """Fail loudly if a device did not follow ``sample_time`` down to the block.

    ``PowerNetwork.dispatch`` asserts ``d.time_horizon in (0, T)`` with a bare
    ``AssertionError``; catching it here names the device whose time-varying
    attribute was not sliced (a missing ``sample_time`` override, or a missing
    entry in ``zap.importers.multi_year.TIME_VARYING_ATTRS``).
    """
    bad = [
        f"{type(d).__name__}(time_horizon={d.time_horizon})"
        for d in devices
        if d.time_horizon not in (0, hours)
    ]
    if bad:
        raise ValueError(
            f"device(s) kept a time horizon other than the block's {hours} hours after "
            f"sample_time: {', '.join(bad)}. Their time-varying attributes are not being "
            "sliced; blocked dispatch cannot be built from this system."
        )


def solve_block(task, cfg: dict, design=None, run_dir=None) -> dict[str, Any]:
    """Solve one task's block with its method. Raises on solver failure.

    ``run_dir`` is optional: without it nothing is persisted beyond the metrics
    payload, which keeps ``solve_block`` usable as a pure function in tests.
    """
    if is_stub(cfg, task.method):
        return solve_block_stub(task, cfg)

    loaded = system_mod.build_system(cfg, draw=task.draw)
    devices = slice_devices(loaded, cfg, task.block, design=design)

    if task.method == "lp":
        return solve_block_lp(loaded, devices, task, cfg, run_dir=run_dir)
    if task.method == "admm":
        return solve_block_admm(loaded, devices, task, cfg, run_dir=run_dir)
    raise ValueError(f"unknown method {task.method!r}")


def _persist_block(run_dir, task, cfg, loaded, devices, outcome, block, payload) -> None:
    """Write the per-block artefacts of a solved block (WP-P1).

    Persistence is never allowed to fail a solve: a broken writer would throw
    away a solution that is already computed.
    """
    if run_dir is None:
        return
    for key, fn in (
        ("hourly_path", persist.write_hourly),
        ("ens_profile_path", persist.write_ens_profile),
    ):
        try:
            path = fn(run_dir, task, cfg, loaded, devices, outcome, block)
        except Exception as exc:  # see the docstring: persistence never fails a solve
            logger.warning("could not write %s for %s: %s", key, task.task_id, exc, exc_info=True)
            continue
        if path is not None:
            payload[key] = str(Path(path).relative_to(Path(run_dir)))
    try:
        persist.write_system_static(run_dir, loaded, cfg)
    except Exception as exc:  # noqa: BLE001 - provenance is best-effort
        logger.warning("could not write system_static.json: %s", exc)


def solve_block_lp(loaded, devices, task, cfg: dict, run_dir=None) -> dict[str, Any]:
    method_cfg = cfg["methods"]["lp"]
    solver = str(method_cfg["solver"]).upper()
    solver_kwargs = dict(method_cfg.get("solver_kwargs") or {})

    start = time.perf_counter()
    outcome = loaded.network.dispatch(
        devices,
        time_horizon=task.block.hours,
        solver=solver,
        solver_kwargs=solver_kwargs,
        add_ground=True,
    )
    wall = time.perf_counter() - start

    problem = outcome.problem
    n_variables = int(sum(int(np.prod(v.shape)) for v in problem.variables())) if problem else None
    n_constraints = (
        int(sum(int(np.prod(c.shape)) for c in problem.constraints)) if problem else None
    )

    metrics = block_metrics(loaded, devices, outcome, task.block)
    metrics["solve_wall_clock_s"] = wall
    metrics["solver_status"] = str(problem.status) if problem else "unknown"
    metrics["n_variables"] = n_variables
    metrics["n_constraints"] = n_constraints

    payload = {
        "metrics": metrics,
        "solver_status": metrics["solver_status"],
        "n_variables": n_variables,
        "n_constraints": n_constraints,
        "admm_iterations": None,
        "admm_primal_residual": None,
        "admm_dual_residual": None,
        "system_meta": dict(getattr(loaded, "meta", {}) or {}),
    }
    _persist_block(run_dir, task, cfg, loaded, devices, outcome, task.block, payload)
    return payload


#: Default gate on the ADMM iterate's worst nodal power-balance violation. An
#: iterate that misses balance by more than this is not a dispatch, and its cost
#: must not be averaged into a headline (benchmark review, section 3.3).
DEFAULT_MAX_IMBALANCE_MW = 1.0

STATUS_INFEASIBLE = "infeasible"
DEFAULT_MAX_SOC_RESIDUAL_MWH = 1.0


#: Length of one time step of a block, in hours.  Every dataset in this chapter
#: is hourly, so a MW-valued residual integrates to MWh with a factor of 1.
HOURS_PER_STEP = 1.0


def admm_imbalance_stats(state) -> tuple[float, float]:
    """``(max |nodal imbalance| in MW, integral of |nodal imbalance| in MWh)``.

    ``state.avg_power`` is the per-node average of the terminal powers; multiplying
    by the terminal count recovers the nodal sum, which is zero exactly when power
    balance holds. Residual *norms* are uninterpretable on a run card; these two
    are: the max is the worst single bus-hour, and the sum over every bus and hour
    (times the step length) is the total absolute energy the iterate fails to
    balance -- the quantity to compare against the block's served energy.
    """
    import torch

    from zap.admm.util import unsqueeze_terminals_times_x

    imbalance = torch.abs(unsqueeze_terminals_times_x(state.num_terminals, state.avg_power))
    return float(torch.max(imbalance).item()), float(torch.sum(imbalance).item()) * HOURS_PER_STEP


def admm_max_imbalance_mw(state) -> float:
    """Worst nodal power-balance violation of an ADMM iterate, in MW."""
    return admm_imbalance_stats(state)[0]


def admm_storage_residual_stats(devices, outcome) -> tuple[float, float]:
    """``(worst, summed)`` storage equality-constraint violation of an ADMM
    iterate, in device energy units (MWh at ``power_unit`` 1).

    The battery prox returns the box-projected iterate, so its state of charge
    can violate the SoC recursion ``s[t+1] = s[t] + eta_c c - d / eta_d`` (and
    the initial / final SoC pins) by an amount the nodal-balance gate cannot
    see. Evaluates every StorageUnit's own ``equality_constraints`` on the
    numpy outcome; ``(0.0, 0.0)`` when the system has no storage. The second
    value sums |residual| over units and hours -- the SoC recursion's residual is
    already an energy, so no step length enters.
    """
    from zap.devices.storage_unit import StorageUnit

    worst = 0.0
    total = 0.0
    for i, dev in enumerate(devices):
        if not isinstance(dev, StorageUnit):
            continue
        residuals = dev.equality_constraints(
            outcome.power[i], outcome.angle[i], outcome.local_variables[i], la=np
        )
        for r in residuals[1:]:  # [0] is the terminal-power identity, always exact
            if np.size(r):
                absolute = np.abs(np.asarray(r, dtype=float))
                worst = max(worst, float(np.max(absolute)))
                total += float(np.sum(absolute))
    return worst, total


def admm_max_storage_residual_mwh(devices, outcome) -> float:
    """Worst storage equality-constraint violation; see :func:`admm_storage_residual_stats`."""
    return admm_storage_residual_stats(devices, outcome)[0]


#: Relative gates (A2): a fraction of the block's own scale, so a 24 h block and
#: a 168 h block are held to the same *standard* rather than the same number.
#: PLACEHOLDER: 2.5e-5 is back-derived from the interim 2 MW absolute gate at
#: ca2040_z4's ~81.9 GW unscaled peak load, not measured. It must be set from the
#: measured distribution of `admm_imbalance_rel` on converged blocks and recorded
#: in `memory/numbers.yaml` (spec "Decisions that need Kamran", item 1).
DEFAULT_MAX_IMBALANCE_REL = 2.5e-5
DEFAULT_MAX_SOC_RESIDUAL_REL = 2.5e-5

GATE_ABSOLUTE = "absolute"
GATE_RELATIVE = "relative"


def _gate_value(method_cfg: dict, key: str, default: float) -> float:
    value = method_cfg.get(key)
    return float(default if value is None else value)


def block_peak_load_mw(devices) -> float:
    """Max over hours of the summed ``Load`` demand, in the devices' own units.

    ``load * nominal_capacity`` is the same gross-demand definition
    ``metrics.block_metrics`` uses for the price weights.  0.0 when the block has
    no ``Load`` (a synthetic test system), which the caller must treat as "no
    scale" rather than "a zero gate".
    """
    from zap.devices import Load

    per_hour = None
    for device in devices:
        if not isinstance(device, Load):
            continue
        load = np.asarray(device.load, dtype=np.float64)
        nominal = getattr(device, "nominal_capacity", None)
        nominal = 1.0 if nominal is None else np.asarray(nominal, dtype=np.float64)
        demand = np.atleast_2d(load * nominal)
        hourly = demand.sum(axis=0)
        if per_hour is None:
            per_hour = hourly
        elif hourly.size == per_hour.size:
            per_hour = per_hour + hourly
        else:  # one of the two is a single column: broadcast the shorter one
            width = max(hourly.size, per_hour.size)
            per_hour = np.broadcast_to(per_hour, (width,)) + np.broadcast_to(hourly, (width,))
    if per_hour is None or per_hour.size == 0:
        return 0.0
    return float(np.max(per_hour))


def block_storage_energy_mwh(devices) -> float:
    """Total storage energy capacity (``power_capacity * duration``) of the block.

    In the devices' own units; 0.0 when the block has no storage.
    """
    from zap.devices.storage_unit import StorageUnit

    total = 0.0
    for device in devices:
        if not isinstance(device, StorageUnit):
            continue
        total += float(
            np.sum(
                np.asarray(device.power_capacity, dtype=np.float64)
                * np.asarray(device.duration, dtype=np.float64)
            )
        )
    return total


def resolve_gate(
    method_cfg: dict,
    abs_key: str,
    rel_key: str,
    scale: float,
    abs_default: float,
    rel_default: float,
) -> tuple[float, str]:
    """``(threshold, mode)`` for one ADMM gate, resolved in this order:

    1. ``method_cfg[abs_key]`` present and not None -> an absolute gate. This is
       the backwards-compatibility path, so every existing
       ``experiments/runs/*/config.resolved.yaml`` keeps the exact gate it ran
       with (including an explicit 0.0, which is not treated as "unset").
    2. else ``method_cfg[rel_key]`` present and not None -> ``rel * scale``.
    3. else ``rel_default * scale``, or ``abs_default`` when ``scale <= 0`` -- a
       test system with no load (or no storage) must not get a zero gate.

    ``scale`` and ``abs_default`` must be in the same units as the quantity being
    gated (physical MW / MWh, i.e. already multiplied by ``power_unit``).
    """
    absolute = method_cfg.get(abs_key)
    if absolute is not None:
        return float(absolute), GATE_ABSOLUTE
    relative = method_cfg.get(rel_key)
    if relative is None:
        relative = rel_default
    if not (scale > 0.0):
        return float(abs_default), GATE_ABSOLUTE
    return float(relative) * float(scale), GATE_RELATIVE


def solve_block_admm(loaded, devices, task, cfg: dict, run_dir=None) -> dict[str, Any]:
    import torch

    from zap.admm import ADMMSolver

    method_cfg = cfg["methods"]["admm"]
    solver_kwargs = dict(method_cfg.get("solver_kwargs") or {})
    if solver_kwargs.get("battery_window"):
        # Backstop: `config.validate` rejects this at load time for `mode: dispatch`,
        # so reaching here means a hand-built cfg dict (a test, or a caller that
        # bypassed the loader). `0` / `None` are the "off" values and stay legal.
        raise ValueError(
            "methods.admm.solver_kwargs.battery_window makes each window its own SoC "
            "problem: that is a different model, not a solver setting, and must not "
            "appear on a dispatch-benchmark or blocking-error row"
        )
    dtype = getattr(torch, str(method_cfg.get("dtype", "float64")))
    machine = solver_kwargs.pop("machine", "cpu")
    solver_kwargs.setdefault("verbose", 0)

    # In physical MW: the importer may have scaled the system by `power_unit`.
    power_unit = float((getattr(loaded, "meta", {}) or {}).get("power_unit", 1.0) or 1.0)
    # Both scales are computed on the *sliced* block devices, i.e. in solver
    # units, then converted to physical MW / MWh so the gates, the recorded
    # residuals and the absolute config keys are all in the same units.
    peak_load = block_peak_load_mw(devices) * power_unit
    storage_energy = block_storage_energy_mwh(devices) * power_unit
    max_imbalance, imbalance_gate_mode = resolve_gate(
        method_cfg,
        "max_imbalance_mw",
        "max_imbalance_rel",
        peak_load,
        DEFAULT_MAX_IMBALANCE_MW,
        DEFAULT_MAX_IMBALANCE_REL,
    )
    max_soc_residual, soc_gate_mode = resolve_gate(
        method_cfg,
        "max_soc_residual_mwh",
        "max_soc_residual_rel",
        storage_energy,
        DEFAULT_MAX_SOC_RESIDUAL_MWH,
        DEFAULT_MAX_SOC_RESIDUAL_REL,
    )

    torch_devices = [d.torchify(machine=machine, dtype=dtype) for d in devices]
    solver = ADMMSolver(machine=machine, dtype=dtype, **solver_kwargs)

    start = time.perf_counter()
    # Deliberately cold-started: no `initial_state=`.  Each SLURM array task
    # solves one block exactly once, so there is nothing to warm-start from, and
    # a cold start keeps the reported `admm_iterations` comparable across blocks
    # and against the LP row.  (Warm starts live in the planning path only --
    # `ADMMLayer.warm_start` / `planning.admm.warm_start`; spec section 5.1.)
    state, history = solver.solve(loaded.network, torch_devices, task.block.hours)
    wall = time.perf_counter() - start

    outcome = state.as_outcome()
    outcome.power = numpyify(outcome.power)
    outcome.angle = numpyify(outcome.angle)
    outcome.local_variables = numpyify(outcome.local_variables)
    outcome.prices = numpyify(outcome.prices)

    metrics = block_metrics(loaded, devices, outcome, task.block)
    primal = float(history.power[-1]) if getattr(history, "power", None) else float("nan")
    dual = float(history.dual_power[-1]) if getattr(history, "dual_power", None) else float("nan")
    iterations = len(getattr(history, "power", []) or [])
    converged = bool(getattr(solver, "converged", False))
    status = "converged" if converged else "max_iterations"
    imbalance, sum_abs_imbalance = admm_imbalance_stats(state)
    imbalance *= power_unit
    sum_abs_imbalance *= power_unit
    soc_residual, sum_abs_soc_residual = admm_storage_residual_stats(devices, outcome)
    soc_residual *= power_unit
    sum_abs_soc_residual *= power_unit

    def _last(name):
        values = getattr(history, name, None) or []
        return float(values[-1]) if values else float("nan")

    metrics["solve_wall_clock_s"] = wall
    metrics["solver_status"] = status
    metrics["admm_iterations"] = iterations
    metrics["admm_primal_residual"] = primal
    metrics["admm_dual_residual"] = dual
    metrics["admm_primal_tol"] = _last("primal_tol")
    metrics["admm_dual_tol"] = _last("dual_tol")
    metrics["admm_converged"] = converged
    metrics["admm_max_imbalance_mw"] = imbalance
    metrics["admm_max_soc_residual_mwh"] = soc_residual
    # Diagnostics, never gated (Kamran, 2026-09-09): the max is one bus-hour, while
    # these integrate the violation over every bus / unit and hour, so they say how
    # much energy the iterate fails to balance in total -- read them next to the
    # block's served energy, not against a threshold.
    metrics["admm_sum_abs_imbalance_mwh"] = sum_abs_imbalance
    metrics["admm_sum_abs_soc_residual_mwh"] = sum_abs_soc_residual
    # Worst inner-prox residual (storage) over devices and iterations, physical MW.
    inner = float(getattr(solver, "max_inner_prox_residual", float("nan")))
    metrics["admm_max_inner_prox_residual_mw"] = inner * power_unit

    # --- Gates (A2) ------------------------------------------------------------
    # Every key below is set *after* `block_metrics`, which is the function that
    # applies `metrics._to_physical_units`, so the `power_unit` conversion is done
    # by hand here and these values are already physical MW / MWh. They are still
    # listed in `metrics._POWER_METRICS` / `_ENERGY_METRICS` so that their unit is
    # declared in one place; that registration cannot double-scale them, because
    # `_to_physical_units` has already run by the time they exist.
    metrics["block_peak_load_mw"] = peak_load
    metrics["storage_energy_capacity_mwh"] = storage_energy
    # The scale-free residuals: these, not the MW numbers, are what makes a 24 h
    # block comparable to a 168 h one.
    metrics["admm_imbalance_rel"] = imbalance / peak_load if peak_load > 0 else float("nan")
    metrics["admm_soc_residual_rel"] = (
        soc_residual / storage_energy if storage_energy > 0 else float("nan")
    )
    metrics["admm_imbalance_gate_mw"] = max_imbalance
    metrics["admm_soc_residual_gate_mwh"] = max_soc_residual
    # Mode of the *imbalance* gate; the SoC gate resolves independently, so its
    # mode is recorded separately rather than conflated into one label.
    metrics["admm_gate_mode"] = imbalance_gate_mode
    metrics["admm_soc_residual_gate_mode"] = soc_gate_mode

    # --- Dual accuracy (A3 records the columns; A4 fills them) -----------------
    # The gradient planning methods consume ADMM duals, so a block whose prices
    # are inaccurate is unusable for planning even when its cost row passes the
    # gates above. These columns exist from A3 onward so that no run has to be
    # repeated to get them; `price_reference` and the price-stability criterion
    # that populate them are A4.
    #
    # Set here rather than in `block_metrics`, so the $/MWh values are scaled by
    # `cost_unit` by hand (`_to_physical_units` has already run).
    cost_unit = float((getattr(loaded, "meta", {}) or {}).get("cost_unit", 1.0) or 1.0)
    price_movement = float(getattr(solver, "price_movement", float("nan")))
    metrics["admm_price_movement_usd_per_mwh"] = price_movement * cost_unit
    metrics["admm_duals_usable"] = None
    price_reference = str(method_cfg.get("price_reference", "none") or "none")
    metrics["admm_price_reference"] = price_reference
    # `nan` unless `methods.admm.price_reference: same_block_lp` (A4): the
    # same-block LP duals are the only reference, and they are a diagnostic, never
    # a production stopping test.
    metrics["admm_price_error_max_usd_per_mwh"] = float("nan")

    payload = {
        "metrics": metrics,
        "solver_status": status,
        "n_variables": None,
        "n_constraints": None,
        "admm_iterations": iterations,
        "admm_primal_residual": primal,
        "admm_dual_residual": dual,
        # Mirrored onto the ledger record so the run card can read them without
        # re-parsing `metrics.csv` (A3).
        "admm_max_inner_prox_residual_mw": metrics["admm_max_inner_prox_residual_mw"],
        "admm_price_error_max_usd_per_mwh": metrics["admm_price_error_max_usd_per_mwh"],
        "system_meta": dict(getattr(loaded, "meta", {}) or {}),
    }

    _persist_block(run_dir, task, cfg, loaded, devices, outcome, task.block, payload)
    try:
        trace = persist.write_admm_trace(run_dir, task, cfg, loaded, task.block, history, solver)
    except Exception as exc:  # noqa: BLE001 - persistence must not fail a solve
        logger.warning("could not write the ADMM trace for %s: %s", task.task_id, exc)
        trace = None
    if trace is not None:
        payload["admm_trace_path"] = str(Path(trace).relative_to(Path(run_dir)))

    # The ADMM iterate's cost is the cost of a *dispatch heuristic*: it neither
    # upper- nor lower-bounds the LP. If the iterate is not even a dispatch --
    # nodal power balance missed by more than the gate -- mark the task so
    # `runcard.summarize` drops it instead of averaging it into the headline.
    if not np.isfinite(imbalance) or imbalance > max_imbalance:
        payload["status"] = STATUS_INFEASIBLE
        payload["error"] = (
            f"ADMM iterate violates nodal power balance by {imbalance:.4g} MW "
            f"> {imbalance_gate_mode} gate {max_imbalance:g} MW "
            f"(block peak load {peak_load:.4g} MW)"
        )
    elif not np.isfinite(soc_residual) or soc_residual > max_soc_residual:
        payload["status"] = STATUS_INFEASIBLE
        payload["error"] = (
            f"ADMM iterate violates storage energy balance by {soc_residual:.4g} MWh "
            f"> {soc_gate_mode} gate {max_soc_residual:g} MWh "
            f"(block storage energy {storage_energy:.4g} MWh)"
        )

    return payload


def solve_block_stub(task, cfg: dict) -> dict[str, Any]:
    """Deterministic fake metrics: no data, no solver, no randomness."""
    sleep_s = stub_sleep_s(cfg, task.method)
    started = time.perf_counter()
    if sleep_s > 0:
        time.sleep(sleep_s)
    digest = hashlib.blake2b(task.task_id.encode("utf-8"), digest_size=8).digest()
    jitter = int.from_bytes(digest, "big") % 1000 / 1000.0  # in [0, 1)
    hours = task.block.hours

    cost = STUB_COST_PER_HOUR * hours + STUB_COST_PER_BLOCK
    metrics = {
        "operational_cost": cost,
        "generation_cost": cost,
        "voll_cost": 0.0,
        "export_revenue": 0.0,
        "unserved_energy_mwh": 0.0,
        "lost_load_hours": 0,
        "co2_tonnes": 10.0 * hours,
        "generation_mwh_by_carrier": '{"stub": %.1f}' % (100.0 * hours),
        "curtailment_mwh": 0.0,
        "imports_mwh": 0.0,
        "exports_mwh": 0.0,
        "storage_cycles": 0.0,
        "mean_price": 50.0 + jitter,
        "max_price": 100.0 + jitter,
        "hours": hours,
        "solve_wall_clock_s": time.perf_counter() - started,
        "solver_status": "stub",
        "n_variables": 10 * hours,
        "n_constraints": 20 * hours,
    }
    return {
        "metrics": metrics,
        "solver_status": "stub",
        "n_variables": metrics["n_variables"],
        "n_constraints": metrics["n_constraints"],
        "admm_iterations": None,
        "admm_primal_residual": None,
        "admm_dual_residual": None,
    }
