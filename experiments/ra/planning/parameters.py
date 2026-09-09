"""Planning parameters and their bounds.

``setup_parameter_names`` is ported verbatim from
``experiments/multi_year/runner.py``.  ``setup_bounds`` is the same function
with the ``inf``/``None`` upper-bound fallback removed: after WP5 there is
exactly one place where an upper bound is invented, and that place is
:mod:`experiments.ra.planning.expansion` (spec section 4).

The two floors (``min_capacity_mw`` / ``min_storage_mw``) keep the old
defaults but are explicit config, because they bias every design: no design can
retire a unit below the floor (spec section 9.5).  ``floor_report`` counts how
many rows each floor actually raised, for the run card.
"""

from __future__ import annotations

import numpy as np

from zap.devices.storage_unit import StorageUnit


def setup_parameter_names(devices: list) -> dict[str, tuple[int, str]]:
    """Map parameter name -> ``(device_index, attribute_name)``.

    Ported verbatim from ``runner.py``: a device contributes a parameter only
    if it has both a capacity attribute and a capital cost.
    """
    parameter_names: dict[str, tuple[int, str]] = {}

    for i, dev in enumerate(devices):
        dev_type = type(dev).__name__.lower()

        if hasattr(dev, "nominal_capacity"):
            cap = dev.nominal_capacity
            if cap is not None and hasattr(dev, "capital_cost") and dev.capital_cost is not None:
                parameter_names[f"{dev_type}_capacity"] = (i, "nominal_capacity")

        if hasattr(dev, "power_capacity"):
            cap = dev.power_capacity
            if cap is not None and hasattr(dev, "capital_cost") and dev.capital_cost is not None:
                parameter_names[f"{dev_type}_power"] = (i, "power_capacity")

    return parameter_names


def _bound_attrs(device) -> tuple[str, str]:
    if isinstance(device, StorageUnit):
        return "min_power_capacity", "max_power_capacity"
    return "min_nominal_capacity", "max_nominal_capacity"


def setup_bounds(
    devices: list,
    parameter_names: dict[str, tuple[int, str]],
    *,
    min_capacity_mw: float = 0.1,
    min_storage_mw: float = 10.0,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Explicit lower/upper bounds for every planning parameter.

    Reads ``min_*``/``max_*`` off the devices (which
    :func:`~experiments.ra.planning.expansion.apply_expansion` has already made
    finite) and applies the configured floors.  A missing or infinite bound is
    an error here, not a silently invented number.  A floor is clipped to the
    row's upper bound, so a row that is retired (upper bound 0, see
    ``LoadOptions.apply_lifetimes``) stays at 0 instead of being floored above
    its own maximum.
    """
    lower_bounds: dict[str, np.ndarray] = {}
    upper_bounds: dict[str, np.ndarray] = {}

    for param_name, (device_idx, attr_name) in parameter_names.items():
        device = devices[device_idx]
        current_cap = getattr(device, attr_name)
        min_attr, max_attr = _bound_attrs(device)

        lb = getattr(device, min_attr, None)
        if lb is None:
            lb = np.zeros_like(np.asarray(current_cap, dtype=float))
        else:
            lb = np.asarray(lb, dtype=float).copy()

        ub = getattr(device, max_attr, None)
        if ub is None or np.any(~np.isfinite(np.asarray(ub, dtype=float))):
            raise ValueError(
                f"parameter {param_name!r} ({type(device).__name__}.{max_attr}) has no finite "
                "upper bound. Run planning.expansion (mode 'pypsa' or 'none') before "
                "setup_bounds; inventing an upper bound is expansion.py's job (spec 4)."
            )
        ub = np.asarray(ub, dtype=float).copy()

        floor = float(min_storage_mw if isinstance(device, StorageUnit) else min_capacity_mw)
        # The floor keeps an *existing* unit from being retired below it; it must
        # not raise a row that cannot exist at all in this model year -- a row
        # retired by the lifetime rule (`wy_store.retired_mask`), or any other
        # row whose upper bound is 0, keeps a lower bound of 0.
        lb = np.maximum(lb, np.minimum(floor, ub))

        if np.any(lb > ub + 1e-12):
            bad = int(np.argmax(lb - ub))
            raise ValueError(
                f"parameter {param_name!r}: lower bound exceeds upper bound at row {bad} "
                f"({float(lb.reshape(-1)[bad])} > {float(ub.reshape(-1)[bad])}); the capacity "
                "floor is above this row's maximum capacity."
            )

        lower_bounds[param_name] = lb
        upper_bounds[param_name] = ub

    return lower_bounds, upper_bounds


def floor_report(
    devices: list,
    parameter_names: dict[str, tuple[int, str]],
    *,
    min_capacity_mw: float = 0.1,
    min_storage_mw: float = 10.0,
) -> dict:
    """How many rows each capacity floor actually raised, per parameter (9.5)."""
    raised: dict[str, int] = {}
    for param_name, (device_idx, _) in parameter_names.items():
        device = devices[device_idx]
        min_attr, _ = _bound_attrs(device)
        lb = getattr(device, min_attr, None)
        if lb is None:
            continue
        lb = np.asarray(lb, dtype=float)
        floor = float(min_storage_mw if isinstance(device, StorageUnit) else min_capacity_mw)
        raised[param_name] = int(np.count_nonzero(lb < floor))
    return {"rows_raised_by_floor": raised}


def initial_parameters(
    devices: list, parameter_names: dict[str, tuple[int, str]]
) -> dict[str, np.ndarray]:
    """Pre-optimisation capacities, one array per parameter."""
    return {
        param: np.asarray(getattr(devices[i], attr), dtype=float).copy()
        for param, (i, attr) in parameter_names.items()
    }


def carrier_labels(
    devices: list, parameter_names: dict[str, tuple[int, str]]
) -> dict[str, list[str]]:
    """Per-parameter carrier labels, for capacity-by-carrier reporting."""
    labels: dict[str, list[str]] = {}
    for param, (i, _) in parameter_names.items():
        dev = devices[i]
        fuel = getattr(dev, "fuel_type", None)
        if fuel is not None:
            labels[param] = [str(f) for f in np.asarray(fuel).reshape(-1)]
        else:
            labels[param] = [type(dev).__name__] * int(dev.num_devices)
    return labels
