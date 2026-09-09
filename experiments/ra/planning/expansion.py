"""Capacity-expansion bounds (spec section 4).

WP1's ``load_system`` pins ``min_* == max_* == p_nom`` on every device: the
system it builds is the as-built one, for operations.  Planning needs real
bounds, which live in the source CSVs (``p_nom_extendable``, ``p_nom_min``,
``p_nom_max``).  This module reads them back and writes them onto a deep copy of
the device list.

This is the *only* place in WP5 where an upper bound is invented: PyPSA writes
``p_nom_max = +inf`` for an unconstrained candidate, and an LP needs a number.

**No retirements in phase 1/2.**  ``AbstractDevice.get_investment_cost`` is
``capital_cost * (x - nominal_capacity)``, a two-sided linear term: zap assumes
the lower bound *is* the as-built capacity, so any ``x < p_nom`` earns a capital
*refund* and the planner is paid to retire (on ``ca2040_z4`` 11 rows have
``p_nom_min < p_nom``; on the tiny fixture the LP retired solar to the floor and
booked a -1.97M "investment").  Modelling retirement properly needs a one-sided
capex term (``capital_cost * max(x - p_nom, 0)`` plus, if wanted, a separate
decommissioning cost), which zap does not have.  Until it does, every existing
row is floored at its as-built capacity: ``lower = max(p_nom_min, p_nom)`` for
extendable and frozen rows alike, so designs may only expand.
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import ConfigError
from ..paths import dataset_dir as resolve_dataset_dir
from .base import planning_options

#: device class name -> (static table key, capacity attr, min attr, max attr)
EXPANDABLE: dict[str, tuple[str, str, str, str]] = {
    "Generator": ("generators", "nominal_capacity", "min_nominal_capacity", "max_nominal_capacity"),
    "DirectedLine": ("links", "nominal_capacity", "min_nominal_capacity", "max_nominal_capacity"),
    "StorageUnit": ("storage_units", "power_capacity", "min_power_capacity", "max_power_capacity"),
}

MODES = ("none", "pypsa")


def dataset_path(cfg: dict, loaded) -> Path:
    """Where the static CSVs of this system live."""
    ds = (cfg or {}).get("dataset") or {}
    name = ds.get("dir")
    if name is None:
        name = (getattr(loaded, "meta", {}) or {}).get("dataset")
    if name is None:
        raise ConfigError(
            "cannot locate the dataset directory: set cfg['dataset']['dir'] or make sure "
            "the loaded system's meta carries a 'dataset' name."
        )
    return resolve_dataset_dir(str(name))


def _column(df: pd.DataFrame, name: str, default: float) -> np.ndarray:
    if name in df.columns:
        return df[name].to_numpy(dtype=float)
    return np.full(len(df), float(default))


def _as_column(values: np.ndarray, like) -> np.ndarray:
    """Match the ``(n, 1)`` shape ``make_dynamic`` gave the device attribute."""
    arr = np.asarray(values, dtype=float).reshape(-1)
    ref = np.asarray(like)
    if ref.ndim == 2:
        return arr.reshape(-1, 1)
    return arr


def apply_expansion(loaded, cfg: dict):
    """Return a copy of ``loaded`` whose capacity bounds reflect PyPSA extendability.

    ``mode == "none"`` returns ``loaded`` unchanged (operations-only sanity runs).
    ``mode == "pypsa"`` deep-copies the devices and, per device class:

    * ``p_nom_extendable == False`` -> ``min == max == p_nom`` (frozen);
    * ``p_nom_extendable == True``  -> ``min = max(p_nom_min, p_nom)``,
      ``max = p_nom_max`` with ``+inf`` replaced by
      ``max(max_capacity_mw, (p_nom + 1000) * max_capacity_multiple)``.

    The ``max(..., p_nom)`` on the lower bound is the no-retirement decision: see
    the module docstring.  ``report["<class>"]["retirement_blocked"]`` counts the
    rows it raised.

    ``loaded.network`` and ``loaded.index`` are shared; ``meta`` is copied and
    gains an ``"expansion"`` block.
    """
    options = planning_options(cfg)["expansion"]
    mode = options["mode"]
    if mode not in MODES:
        raise ConfigError(f"planning.expansion.mode must be one of {MODES}, got {mode!r}")
    if mode == "none":
        return loaded

    from zap.importers.wy_store import LoadedSystem, read_static

    multiple = float(options["max_capacity_multiple"])
    cap_floor: float | None = options["max_capacity_mw"]
    cap_floor = None if cap_floor is None else float(cap_floor)

    static = read_static(dataset_path(cfg, loaded))
    devices = [copy.deepcopy(d) for d in loaded.devices]
    index = loaded.index

    report: dict[str, dict] = {}
    for cls_name, (table_key, cap_attr, min_attr, max_attr) in EXPANDABLE.items():
        if cls_name not in index.device_index:
            continue
        device = devices[index.device_index[cls_name]]
        df = static[table_key]

        if len(df) != device.num_devices:
            raise ValueError(
                f"{cls_name}: static/{table_key}.csv has {len(df)} rows but the device has "
                f"{device.num_devices}; the CSV and the loaded system disagree."
            )
        names = getattr(device, "name", None)
        if names is None:
            names = index.names.get(cls_name)
        if names is None or not df.index.equals(pd.Index(names)):
            raise ValueError(
                f"{cls_name}: static/{table_key}.csv row order does not match the device row "
                "order; expansion bounds would be mis-assigned."
            )

        p_nom = np.asarray(getattr(device, cap_attr), dtype=float).reshape(-1)
        extendable = _column(df, "p_nom_extendable", 0.0).astype(bool)
        p_nom_min = _column(df, "p_nom_min", 0.0)
        p_nom_max = _column(df, "p_nom_max", np.inf)

        pypsa_lower = np.where(extendable, p_nom_min, p_nom)
        upper = np.where(extendable, p_nom_max, p_nom)

        invented = np.zeros(len(df), dtype=bool)
        bad = ~np.isfinite(upper)
        if np.any(bad):
            fallback = (p_nom + 1000.0) * multiple
            if cap_floor is not None:
                fallback = np.maximum(fallback, cap_floor)
            upper = np.where(bad, fallback, upper)
            invented = bad

        if np.any(pypsa_lower > upper + 1e-12):
            row = int(np.argmax(pypsa_lower - upper))
            raise ValueError(
                f"{cls_name} row {df.index[row]!r}: p_nom_min ({pypsa_lower[row]}) exceeds "
                f"p_nom_max ({upper[row]})."
            )

        # --- no retirements in phase 1/2 (spec section 4) -------------------
        # zap's `get_investment_cost` is `capital_cost * (x - nominal_capacity)`
        # with no one-sided term: it *refunds* capital cost for any x below the
        # as-built capacity, so an LP handed `lower = p_nom_min < p_nom` is paid
        # to retire and books a negative investment objective.  Until a one-sided
        # capex term exists in zap, every existing row is pinned at or above its
        # as-built capacity; retiring is out of scope, expanding is not.
        lower = np.maximum(pypsa_lower, p_nom)
        retirement_blocked = int(np.count_nonzero(lower > pypsa_lower + 1e-12))
        # A row whose p_nom_max sits below p_nom (PyPSA planning a retirement)
        # would otherwise leave an empty box; the as-built capacity wins.
        upper_raised = int(np.count_nonzero(upper < lower - 1e-12))
        upper = np.maximum(upper, lower)

        setattr(device, min_attr, _as_column(lower, getattr(device, min_attr, p_nom)))
        setattr(device, max_attr, _as_column(upper, getattr(device, max_attr, p_nom)))

        report[cls_name] = {
            "rows": len(df),
            "extendable": int(np.count_nonzero(extendable)),
            "frozen": int(np.count_nonzero(~extendable)),
            "upper_bounds_invented": int(np.count_nonzero(invented)),
            "retirement_blocked": retirement_blocked,
            "upper_bounds_raised_to_p_nom": upper_raised,
        }

    meta = dict(getattr(loaded, "meta", {}) or {})
    meta["expansion"] = {
        "mode": mode,
        "max_capacity_multiple": multiple,
        "max_capacity_mw": cap_floor,
        "classes": report,
    }
    return LoadedSystem(network=loaded.network, devices=devices, index=index, meta=meta)
