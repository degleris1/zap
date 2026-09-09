"""Objective factories and the emissions plumbing (spec section 5.4).

``update_operation_objectives`` and ``evaluate_emissions`` are ported from
``experiments/multi_year/runner.py``; they are what the dual-ascent outer loop
(Task B) needs.
"""

from __future__ import annotations

from zap.planning import (
    DispatchCostObjective,
    EmissionsObjective,
    InvestmentObjective,
)

from ..config import ConfigError
from .base import planning_options

EMISSIONS_MODES = ("none", "price", "cap", "dual_ascent")


def emissions_options(cfg: dict) -> dict:
    opts = planning_options(cfg)["emissions"]
    if opts["mode"] not in EMISSIONS_MODES:
        raise ConfigError(
            f"planning.emissions.mode must be one of {EMISSIONS_MODES}, got {opts['mode']!r}"
        )
    if opts["cap_basis"] not in ("annual", "horizon"):
        raise ConfigError(
            f"planning.emissions.cap_basis must be 'annual' or 'horizon', got {opts['cap_basis']!r}"
        )
    return opts


def emissions_price(cfg: dict) -> float:
    """The carbon price added to the operation objective at build time.

    ``price`` mode uses the configured price; ``dual_ascent`` starts from
    ``initial_weight`` and is then updated in place by the outer loop.
    """
    opts = emissions_options(cfg)
    if opts["mode"] == "price":
        return float(opts["price"])
    if opts["mode"] == "dual_ascent":
        return float(opts["dual_ascent"]["initial_weight"])
    return 0.0


def operation_objective_factory(network, cfg: dict, *, price: float | None = None):
    """``devices -> DispatchCostObjective [+ price * EmissionsObjective]``."""
    weight = emissions_price(cfg) if price is None else float(price)

    def build(devices):
        f_cost = DispatchCostObjective(network, devices)
        if weight > 0:
            return f_cost + weight * EmissionsObjective(devices)
        return f_cost

    return build


def investment_objective_factory():
    """``devices, layer -> InvestmentObjective``."""

    def build(devices, layer):
        return InvestmentObjective(devices, layer)

    return build


def emissions_objectives(problem) -> list:
    """One reusable ``EmissionsObjective`` per subproblem."""
    return [EmissionsObjective(sub.layer.devices) for sub in problem.subproblems]


def update_operation_objectives(problem, network, price: float) -> None:
    """Swap every subproblem's operation objective in place.

    Ported from ``runner.py``.  Cheap: it creates Python wrapper objects
    without touching the CVXPY dispatch layer.
    """
    for sub in problem.subproblems:
        f_cost = DispatchCostObjective(network, sub.layer.devices)
        if price > 0:
            sub.operation_objective = f_cost + price * EmissionsObjective(sub.layer.devices)
        else:
            sub.operation_objective = f_cost


def evaluate_emissions(problem, emissions_objs: list, params: dict) -> float:
    """Total weighted emissions at ``params``, by a full forward pass.

    Ported from ``runner.py``.  Weighted by subproblem weight and snapshot
    weight so the result is consistent with the Lagrangian cost.
    """
    problem.forward(requires_grad=False, **params)

    total = 0.0
    for i, (sub, eo) in enumerate(zip(problem.subproblems, emissions_objs)):
        # `la` has to match the subproblem's array module: the ADMM path stores
        # a torch dispatch state, and `numpy.sum` on a torch tensor raises.
        value = eo(sub.state, parameters=sub.params, la=sub.la)
        total += problem.weights[i] * sub.snapshot_weight * float(value)
    return total


def emissions_record(cfg: dict, annualization_factor: float) -> dict:
    """The ``emissions`` block of ``design.json`` (spec section 3.4)."""
    from .constraints import emissions_limit

    opts = emissions_options(cfg)
    return {
        "mode": opts["mode"],
        "price": float(opts["price"]),
        "cap": None if opts["cap"] is None else float(opts["cap"]),
        "cap_basis": opts["cap_basis"],
        "cap_applied": emissions_limit(cfg, annualization_factor),
        "dual_ascent": dict(opts["dual_ascent"]) if opts["mode"] == "dual_ascent" else None,
    }
