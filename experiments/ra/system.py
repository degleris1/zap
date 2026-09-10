"""Building the zap system a run dispatches, and the design applied to it.

This module is a thin wrapper over WP1's ``zap.importers.wy_store.load_system``.
The import is guarded so that the rest of the harness (config, identity, tasks,
ledger, aggregation, card) is usable -- and testable with the ``STUB`` solver --
before WP1 lands.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .paths import dataset_dir

try:  # WP1
    from zap.importers.wy_store import (
        DESIGN_CAPACITY_TABLES,
        HourWindow,
        LoadOptions,
        design_capacity_digest,
        load_system,
        read_static,
    )

    WP1_AVAILABLE = True
    _WP1_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # noqa: BLE001 - any import failure means WP1 is unusable
    HourWindow = LoadOptions = load_system = read_static = None  # type: ignore[assignment]
    design_capacity_digest = None  # type: ignore[assignment]
    DESIGN_CAPACITY_TABLES = {  # type: ignore[assignment]
        "Generator": "generators",
        "StorageUnit": "storage_units",
        "DirectedLine": "links",
    }
    WP1_AVAILABLE = False
    _WP1_IMPORT_ERROR = exc


WP1_MESSAGE = (
    "WP1 not available: could not import `zap.importers.wy_store` "
    "(load_system / LoadOptions / HourWindow). Build the weather store reader, "
    "or run the harness with a STUB solver (`--stub`)."
)


def require_wp1() -> None:
    if not WP1_AVAILABLE:
        raise RuntimeError(f"{WP1_MESSAGE} Original import error: {_WP1_IMPORT_ERROR!r}")


@dataclass(frozen=True)
class Design:
    """A candidate system design: capacities to impose on the as-built system.

    Phase 1 only ever evaluates the as-built design, so ``capacities`` is empty
    and :meth:`apply` is the identity.  Phase 2 (the evaluation pipeline) hands
    the same task machinery a design with capacities per device class.
    """

    design_id: str = "asbuilt"
    capacities: Mapping[str, Any] = field(default_factory=dict)

    def apply(self, loaded) -> list:
        """Return the device list with this design's capacities imposed.

        **Never use this for an evaluation with outage draws.** The pool
        weighting is computed at *load* time, so patching capacities onto an
        already-loaded system leaves the availability multipliers weighted over
        the as-built units: built capacity on a retired or greenfield row would
        be outage-free and an expanded row derated by its as-built units only
        (WP-E1).  Pass the design to :func:`build_system` instead, which sets
        ``LoadOptions.design_capacity``.  This method stays for the planning
        path (where the design *is* the decision variable and there is no draw)
        and for tests.
        """
        devices = loaded.devices
        if not self.capacities:
            return devices

        devices = [copy.deepcopy(d) for d in devices]
        index = loaded.index
        for cls_name, values in self.capacities.items():
            i = index.device_index[cls_name]
            device = devices[i]
            values = np.asarray(values, dtype=float).reshape(-1, 1)
            for attr in ("nominal_capacity", "power_capacity"):
                if getattr(device, attr, None) is not None:
                    setattr(device, attr, values)
            for attr in (
                "min_nominal_capacity",
                "max_nominal_capacity",
                "min_power_capacity",
                "max_power_capacity",
            ):
                if getattr(device, attr, None) is not None:
                    setattr(device, attr, values)
        return devices


def design_capacity_map(design: Design | None) -> dict[str, np.ndarray] | None:
    """A :class:`Design` as ``LoadOptions.design_capacity``.

    ``Design.capacities`` is already keyed by device-class name in source-row
    order, so this only coerces to float arrays and rejects classes the loader
    cannot impose (``ExportSink`` capacity is derived from the export links).
    """
    if design is None or not design.capacities:
        return None
    out: dict[str, np.ndarray] = {}
    for cls_name, values in design.capacities.items():
        if cls_name not in DESIGN_CAPACITY_TABLES:
            raise ValueError(
                f"design {design.design_id!r} sets capacity on device class {cls_name!r}, "
                f"which the loader cannot impose (expected any of "
                f"{sorted(DESIGN_CAPACITY_TABLES)})"
            )
        out[cls_name] = np.asarray(values, dtype=float).reshape(-1)
    return out


def read_design_capacity(path, dataset: Path | None = None) -> dict[str, np.ndarray]:
    """Read a ``design.json`` into a ``LoadOptions.design_capacity`` mapping.

    ``dataset`` -- a dataset directory, when given -- checks the recorded row
    names against that dataset's static tables, so a design built on another
    dataset fails here rather than mis-mapping row by row.  This is the
    load-time counterpart of ``planning.design.check_design_names``, which needs
    a built system; the static tables are enough and cost nothing.
    """
    record = json.loads(Path(path).read_text())
    if dataset is not None:
        check_design_capacity_names(record, dataset)
    return design_capacity_map(design_from_record(record)) or {}


def design_from_record(record: dict) -> Design:
    """``design.json`` document -> :class:`Design` (no system needed)."""
    from .planning.design import record_to_design  # local: planning imports this module

    return record_to_design(record)


def check_design_capacity_names(record: dict, dataset: Path) -> None:
    """Assert a ``design.json`` record's row names match the dataset's static tables."""
    require_wp1()
    dataset = Path(dataset)
    static = read_static(dataset)
    for cls_name, entry in (record.get("capacities") or {}).items():
        if cls_name not in DESIGN_CAPACITY_TABLES:
            raise ValueError(
                f"design references device class {cls_name!r}, which the loader cannot "
                f"impose (expected any of {sorted(DESIGN_CAPACITY_TABLES)})"
            )
        names = entry.get("names")
        if names is None:
            raise ValueError(f"design has no row names for {cls_name!r}; refusing to apply it")
        expected = [str(n) for n in static[DESIGN_CAPACITY_TABLES[cls_name]].index]
        got = [str(n) for n in names]
        if got != expected:
            raise ValueError(
                f"design row names for {cls_name!r} do not match "
                f"{dataset}/static/{DESIGN_CAPACITY_TABLES[cls_name]}.csv "
                f"({len(got)} vs {len(expected)} rows). "
                "This design was built on a different dataset."
            )


def load_options(cfg: dict, *, draw: int | None = None, design: Design | None = None):
    """Translate a run config into WP1's ``LoadOptions``.

    ``design`` is imposed **at load time** (``LoadOptions.design_capacity``), so
    outage availability is weighted over the units backing the designed
    capacity.  See :meth:`Design.apply` for why the post-hoc path is wrong for
    any run with outage draws.
    """
    require_wp1()
    sysc = cfg["system"]
    win = cfg["dataset"]["window"]
    power_unit = float(sysc.get("power_unit", 1.0))
    if design is not None and design.capacities and power_unit != 1.0:
        # `design_capacity` overwrites the static tables' `p_nom`, i.e. MW, while a
        # planning run on a scaled system records its parameters in MW/power_unit.
        # Guessing which one a design.json holds is exactly the silent unit bug
        # this project keeps paying for, so refuse instead.
        raise ValueError(
            f"system.power_unit is {power_unit}, not 1.0, and a design was supplied: "
            "design capacities overwrite the static tables' p_nom (MW) while a scaled "
            "planning run records parameters in MW/power_unit. Evaluate with "
            "power_unit: 1.0, or convert the design to MW before loading."
        )
    return LoadOptions(
        design_capacity=design_capacity_map(design),
        design_id=None if design is None else str(design.design_id),
        years=tuple(int(y) for y in cfg["dataset"]["years"]),
        window=HourWindow(start=int(win["start"]), stop=int(win["stop"])),
        voll=float(sysc["voll"]),
        demand_scaling=sysc["demand_scaling"],
        scale_load=float(sysc["scale_load"]),
        peak_capacity_fraction=float(sysc["peak_capacity_fraction"]),
        clip_scale_to_one=bool(sysc["clip_scale_to_one"]),
        ucap_derate=bool(cfg["heuristics"]["ucap_derate"]),
        outage_draw=draw,
        link_losses=bool(sysc["link_losses"]),
        export_mode=sysc["export_mode"],
        carbon_tax=float(sysc["carbon_tax"]),
        storage_init_soc=float(sysc["storage_init_soc"]),
        storage_final_soc=float(sysc["storage_final_soc"]),
        storage_soc_mode=str(sysc.get("storage_soc_mode", "fixed")),
        power_unit=float(sysc.get("power_unit", 1.0)),
        cost_unit=float(sysc.get("cost_unit", 1.0)),
    )


def system_key(
    cfg: dict,
    draw: int | None,
    design_id: str | None = None,
    capacity_digest: str | None = None,
) -> tuple:
    """Everything that identifies the loaded system, for the process-local cache.

    The design is part of the identity: a cached system built for one design
    must never be handed to another (the availability arrays differ).  The
    digest, not the id, is what makes that sound.
    """
    win = cfg["dataset"]["window"]
    return (
        str(cfg["dataset"]["dir"]),
        tuple(int(y) for y in cfg["dataset"]["years"]),
        int(win["start"]),
        int(win["stop"]),
        tuple(sorted((k, repr(v)) for k, v in cfg["system"].items())),
        bool(cfg["heuristics"]["ucap_derate"]),
        draw,
        design_id,
        capacity_digest,
    )


_SYSTEM_CACHE: dict[tuple, Any] = {}

#: Provenance of the most recently built system in this process, so the run card
#: can report the dataset attrs and the demand-scaling block without reloading.
_LAST_META: dict | None = None


def last_meta() -> dict | None:
    return _LAST_META


def build_system(
    cfg: dict,
    *,
    draw: int | None = None,
    design: Design | None = None,
    cache: bool = True,
):
    """Load the whole window once per process; blocks are sliced from it.

    Returns WP1's ``LoadedSystem`` (``network``, ``devices``, ``index``, ``meta``).
    With ``design`` the capacities are imposed *inside* the load, before the
    outage lookup (WP-E1) -- the returned system is already the designed one and
    must not be patched again with :meth:`Design.apply`.
    """
    require_wp1()
    options = load_options(cfg, draw=draw, design=design)
    key = system_key(
        cfg,
        draw,
        options.design_id,
        design_capacity_digest(options.design_capacity),
    )
    if cache and key in _SYSTEM_CACHE:
        return _SYSTEM_CACHE[key]

    loaded = load_system(dataset_path(cfg), options)
    global _LAST_META
    _LAST_META = dict(getattr(loaded, "meta", {}) or {})
    if cache:
        _SYSTEM_CACHE.clear()  # one window at a time; these objects are large
        _SYSTEM_CACHE[key] = loaded
    return loaded


def dataset_path(cfg: dict) -> Path:
    return dataset_dir(cfg["dataset"]["dir"])


def clear_system_cache() -> None:
    _SYSTEM_CACHE.clear()
