"""Capacity-expansion planning for the RA harness (WP5).

``mode: plan`` runs one task — the whole capacity-expansion solve — through the
same ledger, task JSON, ``metrics.csv`` and ``CARD.md`` as a dispatch run
(D-W1).  This package owns the planning path itself:

* :mod:`.expansion`   — PyPSA extendability -> real capacity bounds (section 4)
* :mod:`.sampler`     — ``SystemBlockSampler`` over a ``LoadedSystem`` (5.1)
* :mod:`.selection`   — the period-selection registry (D-W3)
* :mod:`.parameters`  — planning parameters, bounds and their floors
* :mod:`.objectives`  / :mod:`.constraints` — objectives, budgets, emissions (D-W5)
* :mod:`.base`        — ``PlanningMethod.build()``, ``PlanningResult``, ``plan()``
* :mod:`.design`      — ``design.json``, the evaluation seam (3.4)
* ``methods/``        — the concrete methods (single level, gradient, ADMM)
"""

from __future__ import annotations

from . import constraints, design, expansion, objectives, parameters, selection
from .base import (
    METHOD_PRESETS,
    METHODS,
    PLANNING_DEFAULTS,
    SELECTION_DEFAULTS,
    PlanningContext,
    PlanningMethod,
    PlanningResult,
    annualization_factor,
    make_method,
    plan,
    planning_options,
    register_method,
    require_solver,
    selection_options,
    solver_object,
)
from .design import design_paths, read_design, read_design_record
from .sampler import SystemBlockSampler
from .selection import SELECTORS, PeriodSelector, SelectionSpec, make_selector

__all__ = [
    "METHODS",
    "METHOD_PRESETS",
    "PLANNING_DEFAULTS",
    "SELECTION_DEFAULTS",
    "SELECTORS",
    "PeriodSelector",
    "PlanningContext",
    "PlanningMethod",
    "PlanningResult",
    "SelectionSpec",
    "SystemBlockSampler",
    "annualization_factor",
    "constraints",
    "design",
    "design_paths",
    "expansion",
    "make_method",
    "make_selector",
    "objectives",
    "parameters",
    "plan",
    "planning_options",
    "read_design",
    "read_design_record",
    "register_method",
    "require_solver",
    "selection",
    "selection_options",
    "solver_object",
]
