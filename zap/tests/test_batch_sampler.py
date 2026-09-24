"""The caller-chosen batch seam of ``AbstractPlanningProblem.solve``.

``solve(batch_sampler=...)`` lets a caller pick each iteration's minibatch and
attach per-entry expansion factors (``set_batch_expansion``), so a sampling
design that is not zap's own -- an outage draw installed per block, a
Horvitz-Thompson weight ``1/pi`` -- can drive the descent loop.  Without a
sampler every code path must be exactly today's; these tests pin both halves.

Fixture: one bus, one extendable generator and a VOLL load, 4 hours per
subproblem, six subproblems with different demand levels -- small enough that a
forward-and-backward pass is a few milliseconds with HiGHS.
"""

import copy

import numpy as np
import pytest

torch = pytest.importorskip("torch")
cp = pytest.importorskip("cvxpy")

from zap.devices import Generator, Load
from zap.layer import DispatchLayer
from zap.network import PowerNetwork
from zap.planning import (
    DispatchCostObjective,
    GradientDescent,
    InvestmentObjective,
    PlanningProblem,
    StochasticPlanningProblem,
)

T = 4
PARAMETER_NAMES = {"generator_capacity": (0, "nominal_capacity")}
DEMANDS = [30.0, 45.0, 60.0, 35.0, 50.0, 70.0]


def _subproblem(level: float):
    network = PowerNetwork(1)
    gen = Generator(
        num_nodes=1,
        name=np.array(["g_cheap", "g_peak"], dtype=object),
        terminal=np.array([0, 0]),
        nominal_capacity=np.array([[40.0], [10.0]]),
        dynamic_capacity=np.ones((2, T)),
        linear_cost=np.array([[10.0], [80.0]]) * np.ones((1, T)),
        emission_rates=np.zeros((2, 1)),
        capital_cost=np.array([[5.0], [1.0]]),
        min_nominal_capacity=np.array([[0.0], [0.0]]),
        max_nominal_capacity=np.array([[200.0], [200.0]]),
    )
    load = Load(
        num_nodes=1,
        name=np.array(["load"], dtype=object),
        terminal=np.array([0]),
        load=np.array([[level, level * 1.1, level * 0.9, level]]),
        linear_cost=np.array([[1000.0]]),
    )
    devices = [gen, load]
    layer = DispatchLayer(
        network, devices, PARAMETER_NAMES, time_horizon=T, solver=cp.HIGHS, solver_kwargs={}
    )
    return PlanningProblem(
        operation_objective=DispatchCostObjective(network, devices),
        investment_objective=InvestmentObjective(devices, layer),
        layer=layer,
        lower_bounds={"generator_capacity": np.zeros((2, 1))},
        upper_bounds={"generator_capacity": 200.0 * np.ones((2, 1))},
    )


def _problem(weights=None):
    return StochasticPlanningProblem([_subproblem(d) for d in DEMANDS], weights=weights)


STATE = {"generator_capacity": np.array([[42.0], [7.0]])}


def _solve(problem, **kwargs):
    return problem.solve(
        algorithm=GradientDescent(step_size=1e-3, clip=1e3),
        initial_state=copy.deepcopy(STATE),
        num_iterations=3,
        trackers=["loss", "batch", "param"],
        verbosity=0,
        init_full_loss=False,
        **kwargs,
    )


def test_default_solve_unchanged_without_batch_sampler():
    """zap's own `random` strategy: the same batches and losses as a reference run.

    The reference is re-derived here from the RNG contract (`default_rng(seed)`,
    `choice(N, B, replace=False)` sorted, once per iteration) and the ratio
    weights, so the test pins today's behaviour rather than comparing the code
    with itself.
    """
    problem = _problem()
    _state, history = _solve(problem, batch_size=2, batch_strategy="random", batch_seed=7)

    rng = np.random.default_rng(7)
    expected = [sorted(rng.choice(6, size=2, replace=False).tolist()) for _ in range(4)]
    assert [list(map(int, b)) for b in history["batch"]] == expected

    # Every loss is the ratio-weighted batch objective: with unit weights that
    # is (N / B) * sum of the batch's subproblem costs.
    reference = _problem()
    for batch, param, loss in zip(history["batch"], history["param"], history["loss"]):
        costs = [
            float(reference.subproblems[b].forward(requires_grad=False, **param))
            for b in batch
        ]
        assert loss == pytest.approx(3.0 * sum(costs), rel=1e-12)
    assert problem._batch_expansion is None


def test_batch_sampler_called_with_post_projection_state():
    calls = []

    def sampler(iteration, state):
        calls.append((iteration, {k: np.array(v, copy=True) for k, v in state.items()}))
        return [iteration % 6], None

    problem = _problem()
    _state, history = _solve(problem, batch_sampler=sampler)
    assert [i for i, _ in calls] == [0, 1, 2, 3]
    assert len(history["param"]) == len(calls)
    for (iteration, state), recorded in zip(calls, history["param"]):
        np.testing.assert_array_equal(
            state["generator_capacity"], np.asarray(recorded["generator_capacity"])
        )
    assert [list(map(int, b)) for b in history["batch"]] == [[0], [1], [2], [3]]


def test_expansion_weights_forward_and_gradient():
    weights = [1.0, 2.0, 3.0, 1.0, 2.0, 3.0]
    problem = _problem(weights)
    batch = [1, 4]
    expansion = np.array([2.5, 4.0])
    problem.set_batch_expansion(batch, expansion)
    J = float(problem.forward(requires_grad=True, batch=batch, **STATE))
    grad = problem.backward()

    parts, grads = [], []
    for b in batch:
        sub = _problem(weights).subproblems[b]
        parts.append(float(sub.forward(requires_grad=True, **STATE)))
        grads.append(sub.backward()["generator_capacity"])
    applied = expansion * np.array([weights[b] for b in batch])
    assert J == pytest.approx(float(np.dot(applied, parts)), rel=1e-12)
    np.testing.assert_allclose(
        grad["generator_capacity"], sum(a * g for a, g in zip(applied, grads)), rtol=1e-12
    )
    np.testing.assert_allclose(problem.batch_weights(batch), applied, rtol=0, atol=0)


def test_expansion_ignored_for_other_batches():
    weights = [1.0, 2.0, 3.0, 1.0, 2.0, 3.0]
    problem = _problem(weights)
    problem.set_batch_expansion([1, 4], [2.5, 4.0])
    full = float(problem.forward(requires_grad=False, batch=None, **STATE))
    costs = [float(sub.forward(requires_grad=False, **STATE)) for sub in _problem().subproblems]
    assert full == pytest.approx(float(np.dot(weights, costs)), rel=1e-12)
    # Another batch: the ratio form, unchanged.
    ratio = problem.batch_weights([0, 2])
    np.testing.assert_allclose(ratio, (12.0 / 4.0) * np.array([1.0, 3.0]))
    # Clearing it restores the ratio form on the stamped batch too.
    problem.set_batch_expansion(None, None)
    np.testing.assert_allclose(problem.batch_weights([1, 4]), (12.0 / 4.0) * np.array([2.0, 2.0]))


def test_batch_sampler_rejects_peak_net_load():
    problem = _problem()
    with pytest.raises(ValueError, match="peak_net_load_k"):
        _solve(problem, batch_sampler=lambda i, s: ([0], None), peak_net_load_k=1)


def test_batch_sampler_rejects_unsorted_or_out_of_range_batches():
    for bad in ([2, 1], [1, 1], [6], []):
        problem = _problem()
        with pytest.raises(ValueError):
            _solve(problem, batch_sampler=lambda i, s, bad=bad: (bad, None))
