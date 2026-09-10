"""Objective factories and the emissions plumbing (spec section 5.4).

``update_operation_objectives`` and ``evaluate_emissions`` are ported from
``experiments/multi_year/runner.py``; they are what the dual-ascent outer loop
(Task B) needs.
"""

from __future__ import annotations

import numpy as np

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


# ---------------------------------------------------------------------------
# Cost decomposition (the negative-price half of the operation cost)
# ---------------------------------------------------------------------------


def _numeric(obj):
    """A dispatch state as float64 numpy, from cvxpy expressions or torch tensors."""
    if obj is None:
        return None
    if isinstance(obj, (list, tuple)):
        converted = [_numeric(o) for o in obj]
        if isinstance(obj, tuple) and hasattr(obj, "_fields"):  # namedtuple
            return type(obj)(*converted)
        return converted
    if not isinstance(obj, np.ndarray) and hasattr(obj, "value"):  # cvxpy
        obj = obj.value
    if obj is None:
        return None
    if hasattr(obj, "detach"):  # torch
        obj = obj.detach().cpu().numpy()
    return np.asarray(obj, dtype=np.float64)


#: Device classes whose negative-price cost is a *production credit* (a PTC, say).
#: Kept Generator-only so the design-level `opex_credit_raw` means the same thing
#: as the block-level `metrics.generation_credit` (verifier V-S3-b).
CREDIT_CLASSES = ("Generator",)
#: Device classes whose negative-price cost is *export revenue* -- a different
#: economic object, reported on its own line (`export_mode: sink` prices the
#: export links at about -178 $/MWh on the CA2040 exports).
EXPORT_CLASSES = ("DirectedLine",)


def dispatch_cost_credit(devices, power, angle=None, local_variables=None, classes=None) -> float:
    """Sum over ``devices`` of the negative-price part of their operation cost.

    Always ``<= 0``.  ``power`` / ``angle`` / ``local_variables`` are one
    block's dispatch state, as cvxpy expressions (the single-level LP), torch
    tensors or numpy arrays (the gradient / ADMM paths).  Used to split the
    reported opex into a gross cost and a credit: on ca2040_z4 the four
    extendable 2040 ``onwind`` rows are priced at -12.76 $/MWh, so net opex is a
    small difference of two ~2.4 B$ numbers and the net alone hides large
    changes.

    ``classes`` restricts the sum to those device class names; ``None`` sums
    every device.
    """
    from ..metrics import negative_cost_credit

    power = _numeric(power)
    if power is None:
        return 0.0
    angle = _numeric(angle)
    local_variables = _numeric(local_variables)

    total = 0.0
    for i, device in enumerate(devices):
        if i >= len(power):  # e.g. the Ground appended by the network
            break
        if classes is not None and type(device).__name__ not in classes:
            continue
        total += negative_cost_credit(
            device,
            power[i],
            None if angle is None else angle[i],
            None if local_variables is None else local_variables[i],
        )
    return float(total)


def problem_cost_credit(problem, classes=None) -> float | None:
    """``sum_i w_i * credit(sub_i)`` at the subproblems' current dispatch states.

    Weighted exactly as :attr:`StochasticPlanningProblem.op_cost` is --
    ``sum_i w_i * sub_i.get_op_cost()`` with the *unweighted* subproblem cost,
    i.e. **without** ``snapshot_weight`` (which `problem_cvx.py` applies only to
    ``sub.cost``, never to ``sub.op_cost``).  Using a different weighting here
    would break ``opex == gross + credit`` the moment snapshot weights stop
    being 1.0 (verifier V-S3-a).  Returns ``None`` if any subproblem has no
    dispatch state yet.
    """
    total = 0.0
    for weight, sub in zip(problem.weights, problem.subproblems):
        state = getattr(sub, "state", None)
        if state is None:
            return None
        total += float(weight) * dispatch_cost_credit(
            sub.layer.devices, state.power, state.angle, state.local_variables, classes=classes
        )
    return float(total)


def opex_split(opex: float | None, credit: float | None, export_revenue=None) -> dict:
    """The opex decomposition written to ``design.json``.

    ``opex_gross_raw`` is the residual, so ``opex_raw == opex_gross_raw +
    opex_credit_raw`` holds exactly whatever the device mix.  ``opex_credit_raw``
    is the *generator* credit only; ``opex_export_revenue_raw``, when a run has
    export sinks, is the negative-price part contributed by the export links and
    is a component **of** ``opex_gross_raw``, not a third term of the identity.
    """
    out: dict = {"opex_gross_raw": None, "opex_credit_raw": None}
    if opex is not None and credit is not None:
        out["opex_gross_raw"] = float(opex) - float(credit)
        out["opex_credit_raw"] = float(credit)
    if export_revenue is not None:
        out["opex_export_revenue_raw"] = float(export_revenue)
    return out
