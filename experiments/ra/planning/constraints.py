"""Budget constraints and the emissions switch (spec section 5.4, D-W5)."""

from __future__ import annotations

from pathlib import Path

from zap.planning.constraints import BudgetConstraintSet

from ..config import ConfigError
from ..paths import config_root
from .base import planning_options

#: ``planning.method`` presets that map onto ``SingleLevelMethod``.
_SINGLE_LEVEL = ("monolithic", "stochastic", "relaxed")
#: ``planning.method`` presets that run a gradient loop.
_GRADIENT = ("gradient", "admm")


def budget_csv_path(cfg: dict) -> Path | None:
    """Resolve ``planning.budget_constraints`` to a path, or None."""
    value = planning_options(cfg)["budget_constraints"]
    if value in (None, "", "none"):
        return None
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = config_root() / path
    if not path.exists():
        raise ConfigError(f"planning.budget_constraints: no such file {path}")
    return path


def load_budget_constraints(
    cfg: dict, parameter_names: dict, devices: list
) -> BudgetConstraintSet | None:
    """Build the configured ``BudgetConstraintSet``, or None."""
    path = budget_csv_path(cfg)
    if path is None:
        return None
    return BudgetConstraintSet.from_csv(str(path), parameter_names, devices)


def emissions_limit(cfg: dict, annualization_factor: float) -> float | None:
    """The hard cap handed to ``MonolithicPlanningProblem``, in horizon units.

    ``cap_basis: annual`` means the configured cap is an annual (full-horizon)
    quantity, so it is divided by the annualization factor before it reaches a
    solver that only sees ``sampled_hours`` of operations — the same YAML then
    means the same physical target at any block coverage (D-W5).
    """
    from .objectives import emissions_options

    opts = emissions_options(cfg)
    if opts["mode"] != "cap":
        return None
    if opts["cap"] is None:
        raise ConfigError("planning.emissions.mode is 'cap' but planning.emissions.cap is null")
    cap = float(opts["cap"])
    if opts["cap_basis"] == "annual":
        return cap / float(annualization_factor)
    return cap


def validate_emissions(cfg: dict) -> None:
    """The D-W5 compatibility matrix, checked at config time."""
    from .objectives import emissions_options

    opts = planning_options(cfg)
    method = opts["method"]
    emis = emissions_options(cfg)
    mode = emis["mode"]

    if mode == "cap":
        if emis["cap"] is None:
            raise ConfigError("planning.emissions.mode is 'cap' but planning.emissions.cap is null")
        if method not in _SINGLE_LEVEL:
            raise ConfigError(
                f"planning.emissions.mode 'cap' requires a single-level method "
                f"({', '.join(_SINGLE_LEVEL)}), got planning.method={method!r}. "
                "A gradient method has no hard-constraint mechanism (D-W5)."
            )
        if opts["single_level"]["kind"] != "primal":
            raise ConfigError(
                "planning.emissions.mode 'cap' requires planning.single_level.kind "
                "'primal': RelaxedPlanningProblem has no emissions_limit parameter (D-W5)."
            )

    if mode == "dual_ascent":
        if method not in _GRADIENT:
            raise ConfigError(
                f"planning.emissions.mode 'dual_ascent' is valid only for "
                f"{', '.join(_GRADIENT)}, got planning.method={method!r} (D-W5)."
            )
        if emis["dual_ascent"]["target"] is None:
            raise ConfigError(
                "planning.emissions.mode is 'dual_ascent' but "
                "planning.emissions.dual_ascent.target is null"
            )

    if mode == "price" and float(emis["price"]) <= 0:
        raise ConfigError(
            f"planning.emissions.mode is 'price' but planning.emissions.price is "
            f"{emis['price']}; use mode 'none' for an unpriced run."
        )


def emissions_target(cfg: dict, annualization_factor: float) -> float | None:
    """The dual-ascent target in horizon units (annualized like the cap)."""
    from .objectives import emissions_options

    emis = emissions_options(cfg)
    if emis["mode"] != "dual_ascent":
        return None
    target = emis["dual_ascent"]["target"]
    if target is None:
        return None
    target = float(target)
    if emis["cap_basis"] == "annual":
        return target / float(annualization_factor)
    return target
