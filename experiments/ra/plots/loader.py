"""Reading a run directory for the plot catalogue.

A :class:`RunHandle` is the only thing a plot function is given.  Every reader
raises :class:`MissingDataError` naming the file *and* the config flag that
would have produced it, rather than returning an empty frame -- a plot must
never silently draw nothing.
"""

from __future__ import annotations

import functools
import json
import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from .. import metrics as metrics_mod
from .. import paths, persist
from ..config import load_config

logger = logging.getLogger(__name__)

#: ``artefact -> (path shown in the error, the config flag that writes it)``.
ARTEFACTS = {
    "metrics": ("metrics.csv", "always written by `ra aggregate`"),
    "hourly": ("hourly.parquet", "output.save_hourly"),
    "ens_profile": ("ens_profile.parquet", "output.save_ens_profile"),
    "admm_trace": ("admm_trace.parquet", "output.admm_trace_every"),
    "iterations": ("iterations/<task_id>.iterations.parquet", "output.save_iterations"),
    "iteration_blocks": (
        "iterations/<task_id>.iteration_blocks.parquet",
        "output.save_iterations",
    ),
    "iteration_capacity": (
        "iterations/<task_id>.iteration_capacity.parquet",
        "output.save_iterations (with planning.optimizer.save_param_history)",
    ),
    "designs": ("designs/*.json", "mode: plan"),
    "eval": ("eval.parquet", "evaluate.write_eval_tables"),
    "system_static": ("system_static.json", "always written by a solved task"),
    "deviation": ("metrics.csv", "selection.reference"),
    "price_error": ("price_error.parquet", "output.save_price_error"),
}

#: Columns of ``hourly.parquet`` that come back as pandas categoricals and are
#: cast to plain strings so string operations in the plots behave.
_HOURLY_STRING_COLUMNS = (
    "task_id",
    "design_id",
    "method",
    "block_size",
    "quantity",
    "carrier",
    "bus",
    "name",
    "unit",
)


class MissingDataError(RuntimeError):
    """A run does not carry the artefact a plot declared it needs."""

    def __init__(self, run_id: str, artefact: str, message: str | None = None):
        filename, flag = ARTEFACTS.get(artefact, (artefact, "?"))
        self.run_id = run_id
        self.artefact = artefact
        self.filename = filename
        self.flag = flag
        super().__init__(
            message
            or (
                f"run {run_id} has no {filename}; it is written when {flag} is set "
                f"(e.g. re-run with --set {flag}=carrier_bus)"
                if artefact == "hourly"
                else f"run {run_id} has no {filename} ({flag})"
            )
        )


def resolve_run_dir(run_id: str, runs_root=None) -> Path:
    """A run id, a ``name`` prefix that matches exactly one run, or a path."""
    candidate = Path(run_id).expanduser()
    if candidate.is_dir() and (candidate / "config.resolved.yaml").exists():
        return candidate.resolve()
    direct = paths.run_dir(run_id, runs_root)
    if direct.is_dir():
        return direct
    root = Path(runs_root).expanduser() if runs_root is not None else paths.runs_root()
    matches = sorted(p for p in root.glob(f"{run_id}*") if p.is_dir())
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(f"no run directory for {run_id!r} under {root}")
    raise ValueError(
        f"run id {run_id!r} is ambiguous under {root}: " + ", ".join(p.name for p in matches)
    )


@dataclass(frozen=True)
class RunHandle:
    """One run directory, with cached readers for everything a plot may need."""

    run_id: str
    run_dir: Path
    cfg: dict
    label: str
    _cache: dict = field(default_factory=dict, repr=False, compare=False)

    # -- generic ----------------------------------------------------------
    def _cached(self, key: str, fn):
        if key not in self._cache:
            self._cache[key] = fn()
        return self._cache[key]

    def path(self, *parts) -> Path:
        return self.run_dir.joinpath(*parts)

    def has(self, artefact: str) -> bool:
        if artefact in ("hourly", "ens_profile", "admm_trace"):
            return persist.has_artefact(self.run_dir, artefact)
        if artefact in ("iterations", "iteration_blocks", "iteration_capacity"):
            from ..planning.history import has_table

            return has_table(self.run_dir, artefact)
        if artefact == "metrics":
            return self.path("metrics.csv").exists()
        if artefact == "deviation":
            return self.path("metrics.csv").exists()
        if artefact == "designs":
            return bool(self._design_paths())
        if artefact == "eval":
            return self.path("eval.parquet").exists()
        if artefact == "system_static":
            return self.path("system_static.json").exists()
        if artefact == "price_error":
            return self.path("price_error.parquet").exists()
        raise KeyError(f"unknown artefact {artefact!r}; known: {sorted(ARTEFACTS)}")

    def require(self, artefact: str) -> None:
        if not self.has(artefact):
            raise MissingDataError(self.run_id, artefact)

    # -- readers ----------------------------------------------------------
    def metrics(self) -> pd.DataFrame:
        self.require("metrics")
        return self._cached("metrics", lambda: pd.read_csv(self.path("metrics.csv"))).copy()

    def hourly(
        self,
        quantities: Iterable[str] | None = None,
        hours: tuple[int, int] | None = None,
        years: Iterable[int] | None = None,
        methods: Iterable[str] | None = None,
        block_sizes: Iterable[str] | None = None,
    ) -> pd.DataFrame:
        """The long-format hourly table, filtered at read time where possible."""
        self.require("hourly")
        filters = []
        if quantities is not None:
            filters.append(("quantity", "in", [str(q) for q in quantities]))
        if years is not None:
            filters.append(("year", "in", [int(y) for y in years]))
        if hours is not None:
            filters.append(("hour", ">=", int(hours[0])))
            filters.append(("hour", "<", int(hours[1])))
        if methods is not None:
            filters.append(("method", "in", [str(m) for m in methods]))
        if block_sizes is not None:
            filters.append(("block_size", "in", [str(b) for b in block_sizes]))
        kwargs = {"filters": filters} if filters else {}
        frame = persist.read_combined(self.run_dir, "hourly", **kwargs)
        for col in _HOURLY_STRING_COLUMNS:
            if col in frame.columns:
                frame[col] = frame[col].astype(str)
        key = ["method", "block_size", "draw", "design_id", "year", "hour", "quantity",
               "carrier", "bus", "name"]
        duplicated = frame.duplicated(subset=key)
        if duplicated.any():
            raise ValueError(
                f"run {self.run_id}: {int(duplicated.sum())} duplicate rows in the hourly "
                "store; two block files overlap (re-run the affected tasks with --force)"
            )
        return frame

    def price_error(self) -> pd.DataFrame:
        """``price_error.parquet``: the per-(bus, hour) ADMM dual error vs the LP."""
        self.require("price_error")

        def read():
            frame = pd.read_parquet(self.path("price_error.parquet"))
            for col in ("task_id", "method", "block_size", "bus"):
                if col in frame.columns:
                    frame[col] = frame[col].astype(str)
            return frame

        return self._cached("price_error", read).copy()

    def ens_profile(self) -> pd.DataFrame:
        self.require("ens_profile")
        frame = persist.read_combined(self.run_dir, "ens_profile")
        for col in ("task_id", "design_id", "method", "block_size", "bus"):
            if col in frame.columns:
                frame[col] = frame[col].astype(str)
        return frame

    def admm_trace(self, block_index: int | None = None) -> pd.DataFrame:
        self.require("admm_trace")
        frame = persist.read_combined(self.run_dir, "admm_trace")
        for col in ("task_id", "method", "block_size"):
            if col in frame.columns:
                frame[col] = frame[col].astype(str)
        if block_index is not None:
            frame = frame[frame["block_index"] == int(block_index)]
        return frame

    def _iteration_table(self, name: str) -> pd.DataFrame:
        """One iteration table, concatenated over every task that wrote one."""
        self.require(name)
        from ..planning.history import read_table

        frame = read_table(self.run_dir, name)
        for col in frame.columns:
            if str(frame[col].dtype) == "category":
                frame[col] = frame[col].astype(str)
        return frame

    def iterations(self) -> pd.DataFrame:
        return self._iteration_table("iterations")

    def iteration_blocks(self) -> pd.DataFrame:
        return self._iteration_table("iteration_blocks")

    def iteration_capacity(self) -> pd.DataFrame:
        return self._iteration_table("iteration_capacity")

    def _design_paths(self) -> list[Path]:
        from ..planning.design import design_paths

        return design_paths(self.run_dir)

    def designs(self) -> list[dict]:
        self.require("designs")
        from ..planning.design import read_design_record

        return self._cached(
            "designs", lambda: [read_design_record(p) for p in self._design_paths()]
        )

    def eval_table(self) -> pd.DataFrame:
        self.require("eval")
        return pd.read_parquet(self.path("eval.parquet"))

    def system_static(self) -> dict:
        self.require("system_static")
        return self._cached(
            "system_static", lambda: json.loads(self.path("system_static.json").read_text())
        )

    def system_meta(self) -> dict:
        path = self.path("system_meta.json")
        if not path.exists():
            return {}
        return self._cached("system_meta", lambda: json.loads(path.read_text()))

    def env(self) -> dict:
        path = self.path("env.json")
        if not path.exists():
            return {}
        return self._cached("env", lambda: json.loads(path.read_text()))

    def deviation(self) -> pd.DataFrame:
        """``metrics.deviation_vs_reference`` of this run, or an empty frame."""

        def compute():
            try:
                return metrics_mod.deviation_vs_reference(self.metrics())
            except Exception as exc:  # noqa: BLE001 - a misaligned window is not fatal
                logger.warning("run %s: no blocking error (%s)", self.run_id, exc)
                return metrics_mod.deviation_vs_reference(pd.DataFrame())

        return self._cached("deviation", compute).copy()

    def row_carriers(self) -> dict[str, dict[str, str]]:
        """``{device class: {row name: carrier}}`` from the dataset's static tables.

        ``design.json`` records row names but not carriers, and the dataset is
        the only place the mapping lives.  Missing dataset -> empty mapping and
        the caller falls back to ``"unknown"``.
        """

        def compute():
            from ..system import dataset_path

            try:
                from zap.importers.wy_store import read_static

                static = read_static(dataset_path(self.cfg))
            except Exception as exc:  # noqa: BLE001 - the dataset may be absent
                logger.warning("run %s: cannot read carriers from the dataset (%s)",
                               self.run_id, exc)
                return {}
            out: dict[str, dict[str, str]] = {}
            for cls_name, key in (
                ("Generator", "generators"),
                ("StorageUnit", "storage_units"),
                ("DirectedLine", "links"),
                ("Load", "loads"),
            ):
                table = static.get(key)
                if table is None or "carrier" not in table.columns:
                    continue
                out[cls_name] = {str(k): str(v) for k, v in table["carrier"].items()}
            return out

        return self._cached("row_carriers", compute)

    def row_durations(self) -> dict[str, float]:
        """``{storage row name: max_hours}`` from the dataset's static table."""

        def compute():
            from ..system import dataset_path

            try:
                from zap.importers.wy_store import read_static

                table = read_static(dataset_path(self.cfg)).get("storage_units")
            except Exception:  # noqa: BLE001 - the dataset may be absent
                return {}
            if table is None or "max_hours" not in table.columns:
                return {}
            return {str(k): float(v) for k, v in table["max_hours"].items()}

        return self._cached("row_durations", compute)

    # -- convenience ------------------------------------------------------
    @property
    def voll(self) -> float:
        return float((self.cfg.get("system") or {}).get("voll", 0.0) or 0.0)

    @property
    def window(self) -> tuple[int, int]:
        win = (self.cfg.get("dataset") or {}).get("window") or {}
        return int(win.get("start", 0)), int(win.get("stop", 0))

    @property
    def years(self) -> list[int]:
        return [int(y) for y in ((self.cfg.get("dataset") or {}).get("years") or [])]

    @property
    def cores(self) -> float:
        env = self.env()
        value = env.get("slurm_cpus_per_task") or env.get("cpu_count")
        if value in (None, ""):
            return float("nan")
        try:
            return float(value)
        except (TypeError, ValueError):
            return float("nan")


@functools.lru_cache(maxsize=64)
def _load_cfg(path: str) -> dict:
    return load_config(Path(path))


def load_runs(
    run_ids: Sequence[str], *, runs_root=None, labels: Sequence[str] | None = None
) -> list[RunHandle]:
    """Resolve run ids (or name prefixes, or paths) to :class:`RunHandle`s."""
    if labels is not None and len(labels) not in (0, len(run_ids)):
        raise ValueError(
            f"--label was given {len(labels)} times for {len(run_ids)} run(s); "
            "pass one label per --run-id or none at all"
        )
    handles = []
    for i, run_id in enumerate(run_ids):
        run_dir = resolve_run_dir(run_id, runs_root)
        resolved = run_dir / "config.resolved.yaml"
        if not resolved.exists():
            raise FileNotFoundError(f"{run_dir} has no config.resolved.yaml")
        cfg = _load_cfg(str(resolved))
        label = None
        if labels:
            label = labels[i]
        handles.append(
            RunHandle(
                run_id=run_dir.name,
                run_dir=run_dir,
                cfg=cfg,
                label=str(label or cfg.get("name") or run_dir.name),
            )
        )
    return handles


def as_frame(value: Any) -> pd.DataFrame:  # pragma: no cover - tiny helper
    return value if isinstance(value, pd.DataFrame) else pd.DataFrame(value)
