"""
Claim:
Budget projections used by the planning solver compute Euclidean projections onto
their stated feasible sets, and the planner refuses the known unsafe combination
of a simplex budget projection with binding per-coordinate upper caps.

Plausible wrong implementations:
- Clip negative values before simplex projection, losing mass that should be
  redistributed to less-negative coordinates.
- Enforce the budget but ignore per-site upper caps.
- Accept a simplex projection even though finite upper bounds make it the wrong
  feasible set.
"""

import numpy as np
import pytest

from zap.planning.problem_abstract import AbstractPlanningProblem
from zap.planning.projection import BoxBudgetProjection, SimplexBudgetProjection


def test_simplex_projection_uses_raw_negative_inputs():
    projection = SimplexBudgetProjection(2.5, strict=True)

    got = projection(np.array([-1.0, -2.0, -0.5, -3.0, -0.1]))

    expected = np.array([0.3666666667, 0.0, 0.8666666667, 0.0, 1.2666666667])
    np.testing.assert_allclose(got, expected, atol=1e-9)
    assert got.sum() == pytest.approx(2.5)
    assert np.all(got >= 0.0)


def test_box_budget_projection_enforces_caps_and_budget():
    projection = BoxBudgetProjection(
        budget=1.0,
        lower_bounds=np.zeros(4),
        upper_bounds=np.full(4, 0.4),
    )

    got = projection(np.array([5.0, 0.0, 0.0, 0.0]))

    expected = np.array([0.4, 0.2, 0.2, 0.2])
    np.testing.assert_allclose(got, expected, atol=1e-7)
    assert got.sum() == pytest.approx(1.0)
    assert np.max(got) <= 0.4 + 1e-9


def test_box_budget_projection_rejects_infeasible_bounds():
    with pytest.raises(ValueError, match="too small"):
        BoxBudgetProjection(0.5, np.array([0.4, 0.4]), np.ones(2))

    with pytest.raises(ValueError, match="too large"):
        BoxBudgetProjection(2.5, np.zeros(2), np.ones(2))


class _GuardOnlyProblem(AbstractPlanningProblem):
    def forward(self, requires_grad=False, batch=None, **kwargs):
        raise AssertionError("guard should run before forward")

    def backward(self):
        raise AssertionError("guard should run before backward")


def test_planner_rejects_simplex_projection_with_binding_upper_caps():
    problem = _GuardOnlyProblem(
        operation_objective=object(),
        investment_objective=object(),
        layer=object(),
        lower_bounds={"dc_capacity": np.zeros(3)},
        upper_bounds={"dc_capacity": np.array([0.5, 0.5, 0.5])},
        extra_projections={"dc_capacity": SimplexBudgetProjection(1.0, strict=True)},
    )

    with pytest.raises(ValueError, match="SimplexBudgetProjection"):
        problem.solve(initial_state={"dc_capacity": np.zeros(3)}, init_full_loss=False)
