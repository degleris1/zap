"""The ``ra plot`` catalogue: a registry of figures, each with a pinned CSV schema.

Every plot is a function ``fn(runs, **opts) -> (Figure, DataFrame)`` registered
with the exact columns of the table it returns.  :func:`plot` asserts the
returned frame's columns against that declaration, so a schema drift is a
run-time failure rather than a silently changed CSV (D9).

``experiments.ra.plots.PLOTS`` is the catalogue; ``plot`` / ``render`` are the
public entry points and ``catalogue()`` is what ``ra plot --list`` prints.

Phase-A plots are implemented; phase-B plots are registered with their title,
tier, needs and columns, and a body that raises ``NotImplementedError`` (see
``memory/plans/2026-09-09-plots-spec.md`` D8).
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from . import style
from .loader import MissingDataError, RunHandle, load_runs, resolve_run_dir

logger = logging.getLogger(__name__)

TIERS = ("debug", "report")
PHASES = ("A", "B")

__all__ = [
    "PLOTS",
    "MissingDataError",
    "PlotSpec",
    "RunCountError",
    "RunHandle",
    "catalogue",
    "load_runs",
    "plot",
    "register",
    "render",
    "resolve_run_dir",
    "style",
]


@dataclass(frozen=True)
class PlotSpec:
    plot_id: str
    title: str
    tier: str
    needs: tuple[str, ...]
    columns: tuple[str, ...]
    min_runs: int = 1
    max_runs: int | None = None
    phase: str = "A"
    fn: Callable[..., tuple] = field(default=None, compare=False, repr=False)


class RunCountError(ValueError):
    """A plot was given a number of runs its ``min_runs`` / ``max_runs`` forbid.

    A ``ValueError`` (so a caller can catch it as one), but distinguishable, so
    ``ra plot --tier report`` on a single run *skips* O5 rather than failing:
    "this plot does not apply to this selection" is the same kind of condition
    as "this run has no hourly data", not a bug.
    """


PLOTS: dict[str, PlotSpec] = {}


def register(
    plot_id: str,
    *,
    title: str,
    tier: str,
    needs: Sequence[str],
    columns: Sequence[str],
    min_runs: int = 1,
    max_runs: int | None = None,
    phase: str = "A",
):
    """Decorator: add a plot function to :data:`PLOTS`."""
    if tier not in TIERS:
        raise ValueError(f"tier must be one of {TIERS}, got {tier!r}")
    if phase not in PHASES:
        raise ValueError(f"phase must be one of {PHASES}, got {phase!r}")

    def decorator(fn):
        if plot_id in PLOTS:
            raise ValueError(f"plot id {plot_id!r} is already registered")
        PLOTS[plot_id] = PlotSpec(
            plot_id=plot_id,
            title=title,
            tier=tier,
            needs=tuple(needs),
            columns=tuple(columns),
            min_runs=int(min_runs),
            max_runs=None if max_runs is None else int(max_runs),
            phase=phase,
            fn=fn,
        )
        return fn

    return decorator


def phase_b(plot_id: str, **kwargs):
    """Register a phase-B plot: schema now, body later."""

    @register(plot_id, phase="B", **kwargs)
    def _not_implemented(runs, **opts):
        raise NotImplementedError(
            f"{plot_id} is phase B (see memory/plans/2026-09-09-plots-spec.md)"
        )

    return _not_implemented


def get(plot_id: str) -> PlotSpec:
    try:
        return PLOTS[plot_id]
    except KeyError:
        raise KeyError(
            f"unknown plot id {plot_id!r}; known ids: {', '.join(sorted(PLOTS))}"
        ) from None


def plot(plot_id: str, runs: Sequence[RunHandle], **opts):
    """Run one plot: check the run count and the data, then pin the CSV schema."""
    spec = get(plot_id)
    runs = list(runs)
    exact = spec.max_runs is not None and spec.min_runs == spec.max_runs
    if len(runs) < spec.min_runs or (spec.max_runs is not None and len(runs) > spec.max_runs):
        if exact:
            raise RunCountError(
                f"{plot_id} compares exactly {spec.max_runs} runs, got {len(runs)}"
            )
        if len(runs) < spec.min_runs:
            raise RunCountError(
                f"{plot_id} needs at least {spec.min_runs} run(s), got {len(runs)}"
            )
        raise RunCountError(f"{plot_id} takes at most {spec.max_runs} run(s), got {len(runs)}")
    for run in runs:
        for need in spec.needs:
            run.require(need)

    style.apply_rc()
    fig, table = spec.fn(runs, **opts)
    if not isinstance(table, pd.DataFrame):
        raise TypeError(f"{plot_id} returned {type(table).__name__}, not a DataFrame")
    if tuple(table.columns) != spec.columns:
        raise AssertionError(
            f"{plot_id} returned columns {tuple(table.columns)} but declares "
            f"{spec.columns}; the CSV schema is pinned by PlotSpec.columns (D9)"
        )
    return fig, table


def render(plot_id: str, runs: Sequence[RunHandle], out_dir, *, stem: str | None = None, **opts):
    """:func:`plot` plus the PNG / CSV pair on disk."""
    fig, table = plot(plot_id, runs, **opts)
    if stem is None:
        stem = "-".join(run.label for run in runs) or "run"
    return style.save(fig, table, plot_id, Path(out_dir), stem)


def catalogue() -> pd.DataFrame:
    """One row per registered plot: id, title, tier, phase, needs, columns."""
    rows = [
        {
            "plot_id": spec.plot_id,
            "title": spec.title,
            "tier": spec.tier,
            "phase": spec.phase,
            "needs": ",".join(spec.needs),
            "columns": ",".join(spec.columns),
        }
        for spec in PLOTS.values()
    ]
    frame = pd.DataFrame(rows, columns=["plot_id", "title", "tier", "phase", "needs", "columns"])
    if frame.empty:
        return frame
    return frame.sort_values("plot_id", key=lambda s: s.map(_sort_key)).reset_index(drop=True)


def _sort_key(plot_id: str) -> tuple[str, int]:
    return (str(plot_id)[0], int(str(plot_id)[1:] or 0))


def ids(*, tier: str | None = None, phase: str | None = None) -> list[str]:
    out = [
        pid
        for pid, spec in PLOTS.items()
        if (tier in (None, "all", spec.tier)) and (phase is None or spec.phase == phase)
    ]
    return sorted(out, key=_sort_key)


# Registration happens on import (the three modules only register).
from . import adequacy, operational, planning  # noqa: F401
