"""Building the zap system a run dispatches, and the design applied to it.

This module is a thin wrapper over WP1's ``zap.importers.wy_store.load_system``.
The import is guarded so that the rest of the harness (config, identity, tasks,
ledger, aggregation, card) is usable -- and testable with the ``STUB`` solver --
before WP1 lands.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .paths import dataset_dir

try:  # WP1
    from zap.importers.wy_store import HourWindow, LoadOptions, load_system

    WP1_AVAILABLE = True
    _WP1_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # noqa: BLE001 - any import failure means WP1 is unusable
    HourWindow = LoadOptions = load_system = None  # type: ignore[assignment]
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
        """Return the device list with this design's capacities imposed."""
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


def load_options(cfg: dict, *, draw: int | None = None):
    """Translate a run config into WP1's ``LoadOptions``."""
    require_wp1()
    sysc = cfg["system"]
    win = cfg["dataset"]["window"]
    return LoadOptions(
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


def system_key(cfg: dict, draw: int | None) -> tuple:
    """Everything that identifies the loaded system, for the process-local cache."""
    win = cfg["dataset"]["window"]
    return (
        str(cfg["dataset"]["dir"]),
        tuple(int(y) for y in cfg["dataset"]["years"]),
        int(win["start"]),
        int(win["stop"]),
        tuple(sorted((k, repr(v)) for k, v in cfg["system"].items())),
        bool(cfg["heuristics"]["ucap_derate"]),
        draw,
    )


_SYSTEM_CACHE: dict[tuple, Any] = {}

#: Provenance of the most recently built system in this process, so the run card
#: can report the dataset attrs and the demand-scaling block without reloading.
_LAST_META: dict | None = None


def last_meta() -> dict | None:
    return _LAST_META


def build_system(cfg: dict, *, draw: int | None = None, cache: bool = True):
    """Load the whole window once per process; blocks are sliced from it.

    Returns WP1's ``LoadedSystem`` (``network``, ``devices``, ``index``, ``meta``).
    """
    require_wp1()
    key = system_key(cfg, draw)
    if cache and key in _SYSTEM_CACHE:
        return _SYSTEM_CACHE[key]

    loaded = load_system(dataset_path(cfg), load_options(cfg, draw=draw))
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
