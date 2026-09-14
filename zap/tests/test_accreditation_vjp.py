"""The second adjoint: ``PlanningProblemCVX.backward_objective`` (WP-A4, D9/D10).

The planner-side marginal reliability impact of
``memory/plans/2026-09-11-accreditation-spec.md`` is one extra VJP on the
forward solve the planning iteration already took: seed
:class:`~zap.planning.operation_objectives.UnservedEnergyObjective` instead of
the cost objective, push it back through the same KKT factorization, and read
both halves of the result --

* ``dtheta[param] = d EUE / d param`` (<= 0 for a capacity), the **numerator**
  of an accreditation factor, and
* ``adjoint.prices[n, t] = + d EUE / d D_{n,t}``, the sensitivity to marginal
  **demand** at a node and hour (equivalently ``-d/d injection``), whose
  contraction with a load direction ``f_n`` is the **denominator**.

That sign convention is the one thing here that cannot be derived from the
source: it was measured against finite differences on 2026-09-13 (spec
implementation note 29), and every assertion below re-measures it.

The fixture is a **ground-free** 3-bus, 6-hour toy built in this file rather
than borrowed from ``tests/test_accreditation.py`` in the CH3 repo: that
system is dispatched with ``add_ground=True``, and the resulting ``Ground``
device carries an ``angle`` of shape ``(1, 1, 6)`` against an empty
``phase_dual``, which trips ``DispatchOutcome.shape``'s ``mu_shape ==
a_shape`` assertion before the KKT Jacobian is ever built (spec note 33).
Every bus here carries an injector, so no ground is needed.

Fixture, all six hours::

    bus1: gen_b  40 MW @ 10 $/MWh --(DirectedLine, eff 0.90)--> bus0
    bus2: gen_c  30 MW @ 30 $/MWh --(DirectedLine, eff 0.95)--> bus0,
          and 10 MW of its own load
    bus0: gen_a  50 MW @ 20 $/MWh, gen_zero 0 MW @ 15 $/MWh,
          a 20 MW / 2 h battery, and the load that sheds in hour 3.

Hour 3's demand of 200 MW exceeds everything the three generators, the two
lines and the battery can deliver, so the block sheds and every capacity row
has a derivative that can be read off the network by hand: 1.0 MWh/MW for a
row at the shedding bus, 0.90 behind the lossy line, 0.95 behind the other.
"""

import copy

import numpy as np
import pytest

torch = pytest.importorskip("torch")
cp = pytest.importorskip("cvxpy")

from zap.devices import Generator, Load, StorageUnit
from zap.devices.transporter import DirectedLine
from zap.layer import DispatchLayer
from zap.network import PowerNetwork
from zap.planning import InvestmentObjective, StochasticPlanningProblem
from zap.planning.operation_objectives import UnservedEnergyObjective
from zap.planning.problem_cvx import PlanningProblemCVX

torch.set_default_dtype(torch.float64)

T = 6
VOLL = 10_000.0
C_MAX = 30.0
EPS_B1 = 0.9
EPS_B2 = 0.95

#: The two planning parameters of the toy, in the layer's own naming.
PARAMETER_NAMES = {
    "generator_capacity": (0, "nominal_capacity"),
    "storage_power": (2, "power_capacity"),
}

#: The as-built point every test differentiates at.
THETA = {
    "generator_capacity": np.array([[50.0], [40.0], [30.0], [0.0]]),
    "storage_power": np.array([[20.0]]),
}

GEN_ROWS = {"gen_a": 0, "gen_b": 1, "gen_c": 2, "gen_zero": 3}

#: The shedding hour, and the only one with a nonzero derivative here.
SHED_HOUR = 3


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


def toy_devices(demand0=None, load2=10.0):
    """The 3-bus, 6-hour, ground-free system described in the module docstring."""
    network = PowerNetwork(3)
    ones = np.ones((1, T))

    generators = Generator(
        num_nodes=3,
        name=np.array(["gen_a", "gen_b", "gen_c", "gen_zero"], dtype=object),
        terminal=np.array([0, 1, 2, 0]),
        nominal_capacity=np.array([[50.0], [40.0], [30.0], [0.0]]),
        dynamic_capacity=np.ones((4, T)),
        linear_cost=np.array([[20.0], [10.0], [C_MAX], [15.0]]) * ones,
        emission_rates=np.zeros((4, 1)),
        capital_cost=np.ones((4, 1)),
    )
    demand = (
        np.array([80.0, 80.0, 80.0, 200.0, 80.0, 80.0])
        if demand0 is None
        else np.asarray(demand0, dtype=float)
    )
    loads = Load(
        num_nodes=3,
        name=np.array(["load_b0", "load_b2"], dtype=object),
        terminal=np.array([0, 2]),
        load=np.vstack([demand, np.full(T, float(load2))]),
        linear_cost=VOLL * np.ones((2, 1)),
    )
    storage = StorageUnit(
        num_nodes=3,
        name=np.array(["batt_b0"], dtype=object),
        terminal=np.array([0]),
        power_capacity=np.array([[20.0]]),
        duration=np.array([[2.0]]),
        charge_efficiency=np.array([[0.9]]),
        discharge_efficiency=np.array([[0.9]]),
        initial_soc=np.array([[0.0]]),
        final_soc=np.array([[0.0]]),
        linear_cost=np.array([[0.0]]),
        soc_mode="fixed",
        capital_cost=np.ones((1, 1)),
    )
    lines = DirectedLine(
        num_nodes=3,
        name=np.array(["ln_b1", "ln_b2"], dtype=object),
        source_terminal=np.array([1, 2]),
        sink_terminal=np.array([0, 0]),
        min_power=np.zeros((2, 1)),
        max_power=np.array([[100.0], [100.0]]),
        linear_cost=np.zeros((2, 1)),
        efficiency=np.array([[EPS_B1], [EPS_B2]]),
    )
    return network, [generators, loads, storage, lines]


def make_problem(devices=None, network=None):
    """A :class:`PlanningProblemCVX` whose *operation* objective is unserved energy."""
    if devices is None:
        network, devices = toy_devices()
    layer = DispatchLayer(
        network,
        devices,
        PARAMETER_NAMES,
        time_horizon=T,
        solver=cp.HIGHS,
        solver_kwargs={},
        add_ground=False,
    )
    lower = {k: np.zeros_like(v) for k, v in THETA.items()}
    upper = {k: np.full_like(v, 1e4) for k, v in THETA.items()}
    problem = PlanningProblemCVX(
        UnservedEnergyObjective(devices),
        InvestmentObjective(devices, layer),
        layer,
        lower,
        upper,
    )
    return problem, devices


def solved_problem():
    """A problem with a live ``forward(requires_grad=True)`` + ``backward()``."""
    problem, devices = make_problem()
    problem.forward(requires_grad=True, **theta())
    problem.backward()
    return problem, devices


def theta(**overrides):
    """A fresh copy of :data:`THETA`, optionally with one entry replaced."""
    out = {k: np.array(v, dtype=float, copy=True) for k, v in THETA.items()}
    out.update({k: np.asarray(v, dtype=float) for k, v in overrides.items()})
    return out


def eue_at(devices, network, values):
    """The block's unserved energy at a parameter point, from a fresh dispatch."""
    layer = DispatchLayer(
        network,
        devices,
        PARAMETER_NAMES,
        time_horizon=T,
        solver=cp.HIGHS,
        solver_kwargs={},
        add_ground=False,
    )
    outcome = layer.forward(**values)
    return float(UnservedEnergyObjective(devices)(outcome, la=np))


def with_demand(devices, profile):
    """A deep copy of ``devices`` whose ``Load.load`` is ``profile``.

    A ``Load`` is rebuilt by mutation rather than by construction so that every
    other attribute of the fixture (names, VOLL, terminals) is carried over
    unexamined -- the perturbation must differ in exactly one array.
    """
    out = copy.deepcopy(list(devices))
    out[1].load = np.asarray(profile, dtype=float)
    if hasattr(out[1], "has_changed"):
        out[1].has_changed = True
    return out


def load_shares(devices):
    """``f_n``: the firm-load direction, pro-rata to each bus's window peak.

    The same rule the evaluator uses (spec D15 / :func:`ch3.ra.accreditation.
    load_direction`), restated here so the zap test does not depend on ch3.
    """
    load = devices[1]
    profile = np.atleast_2d(np.asarray(load.load, dtype=float))
    peaks = np.zeros(3)
    for row, node in enumerate(np.asarray(load.terminal).reshape(-1)):
        peaks[int(node)] = max(peaks[int(node)], float(profile[row].max()))
    return peaks / peaks.sum()


# ---------------------------------------------------------------------------
# (i) the adjoint's prices are +d EUE / d demand
# ---------------------------------------------------------------------------


def test_adjoint_prices_are_the_demand_derivative():
    """``adjoint.prices[n, t]`` == one-sided FD of EUE in ``D_{n, t}`` (D10 i)."""
    problem, devices = solved_problem()
    network = problem.layer.network
    _dtheta, adjoint, base_eue = problem.backward_objective(
        UnservedEnergyObjective(devices), return_value=True
    )
    prices = np.asarray(adjoint.prices, dtype=float)
    assert prices.shape == (3, T)

    load = devices[1]
    terminals = np.asarray(load.terminal).reshape(-1)
    checked = 0
    for row, node in enumerate(terminals):
        for hour in (0, SHED_HOUR, T - 1):
            profile = np.array(load.load, dtype=float, copy=True)
            profile[row, hour] += 1.0
            fd = eue_at(with_demand(devices, profile), network, theta()) - base_eue
            assert prices[int(node), hour] == pytest.approx(fd, rel=1e-3, abs=1e-6), (
                f"node {node} hour {hour}: adjoint {prices[int(node), hour]} vs FD {fd}"
            )
            checked += 1
    assert checked == 2 * 3


def test_adjoint_prices_match_the_evaluator_weight_at_load_buses():
    """The two venues compute one derivative (D10 i, second half).

    The evaluator's rule is the nodal price over VOLL above the scarcity
    threshold; the planner's is this adjoint.  Guarded by ``importorskip``
    because a zap test may import ``ch3`` only when it happens to be installed.
    """
    acc = pytest.importorskip("ch3.ra.accreditation")

    problem, devices = solved_problem()
    _dtheta, adjoint = problem.backward_objective(UnservedEnergyObjective(devices))
    prices = np.asarray(adjoint.prices, dtype=float)
    weights = acc.price_weights(
        None, problem.state, voll=VOLL, c_max=C_MAX, devices=devices, restrict=False
    )

    load_nodes = sorted({int(n) for n in np.asarray(devices[1].terminal).reshape(-1)})
    for node in load_nodes:
        np.testing.assert_allclose(weights[node], prices[node], atol=1e-3)


# ---------------------------------------------------------------------------
# (ii) dtheta is the capacity derivative, and f . nu the load derivative
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("row_name", ["gen_a", "gen_b", "gen_c"])
@pytest.mark.parametrize("delta", [1.0, 10.0])
def test_dtheta_matches_finite_differences(row_name, delta):
    """``dtheta[generator_capacity][r]`` == ``dEUE/dcap`` by FD (D10 ii).

    The three rows sit at the shedding bus and behind the two lines, so their
    analytic values are 1.0, ``EPS_B1`` and ``EPS_B2`` MWh per MW -- which is
    what makes this a test of the *sign convention* and not just of internal
    consistency.
    """
    expected = {"gen_a": -1.0, "gen_b": -EPS_B1, "gen_c": -EPS_B2}[row_name]
    index = GEN_ROWS[row_name]

    problem, devices = solved_problem()
    network = problem.layer.network
    dtheta, _adjoint, base_eue = problem.backward_objective(
        UnservedEnergyObjective(devices), return_value=True
    )
    grad = np.asarray(dtheta["generator_capacity"], dtype=float).reshape(-1)

    bumped = theta()
    bumped["generator_capacity"][index, 0] += delta
    fd = (eue_at(devices, network, bumped) - base_eue) / delta

    assert grad[index] == pytest.approx(expected, rel=1e-6)
    assert grad[index] == pytest.approx(fd, rel=1e-2)


@pytest.mark.xfail(
    strict=False,
    reason=(
        "zero-capacity gradients under scarcity are not directional derivatives "
        "(LESSONS 2026-09-09, issue #10); recorded, not fixed"
    ),
)
@pytest.mark.parametrize("delta", [1.0, 10.0])
def test_dtheta_at_a_zero_capacity_row(delta):
    """``gen_zero`` is at 0 MW: its implicit gradient need not be the derivative."""
    problem, devices = solved_problem()
    network = problem.layer.network
    dtheta, _adjoint, base_eue = problem.backward_objective(
        UnservedEnergyObjective(devices), return_value=True
    )
    grad = np.asarray(dtheta["generator_capacity"], dtype=float).reshape(-1)

    bumped = theta()
    bumped["generator_capacity"][GEN_ROWS["gen_zero"], 0] += delta
    fd = (eue_at(devices, network, bumped) - base_eue) / delta
    assert grad[GEN_ROWS["gen_zero"]] == pytest.approx(fd, rel=1e-2)


def test_load_direction_denominator_matches_finite_differences():
    """``+ sum_n f_n sum_t nu_{n,t}`` == ``dEUE/dL`` for a firm-load increment (D5).

    Note the sign: the spec's D9 wrote this denominator as ``- sum f nu`` and
    was corrected on 2026-09-13.  A firm 1 MW of extra load in every hour
    *raises* EUE, so the FD is positive and so is the contraction.
    """
    problem, devices = solved_problem()
    network = problem.layer.network
    _dtheta, adjoint, base_eue = problem.backward_objective(
        UnservedEnergyObjective(devices), return_value=True
    )
    nu = np.asarray(adjoint.prices, dtype=float).sum(axis=1)
    f = load_shares(devices)
    m_L = float(np.dot(f, nu))
    assert m_L > 0.0

    load = devices[1]
    terminals = np.asarray(load.terminal).reshape(-1)
    profile = np.array(load.load, dtype=float, copy=True)
    for row, node in enumerate(terminals):
        profile[row, :] += f[int(node)]
    fd = eue_at(with_demand(devices, profile), network, theta()) - base_eue
    assert m_L == pytest.approx(fd, rel=1e-3)


# ---------------------------------------------------------------------------
# (iii) it is repeatable and it does not touch .grad
# ---------------------------------------------------------------------------


def test_backward_objective_is_repeatable_and_leaves_grad_alone():
    """Two more calls on the same forward agree, and ``.grad`` is untouched.

    ``backward_objective`` must use :func:`torch.autograd.grad`, never
    ``.backward()``: the ``.grad`` fields hold the *planning* gradient the
    descent step is about to read, and accumulating an EUE seed into them would
    silently turn a cost minimiser into something else.
    """
    problem, devices = solved_problem()
    before = {
        k: None if v.grad is None else v.grad.detach().clone()
        for k, v in problem.torch_kwargs.items()
    }
    assert any(v is not None for v in before.values()), "the fixture never ran backward()"

    objective = UnservedEnergyObjective(devices)
    first = problem.backward_objective(objective)
    second = problem.backward_objective(objective)
    third = problem.backward_objective(objective)

    for key in first[0]:
        np.testing.assert_allclose(
            np.asarray(second[0][key], dtype=float), np.asarray(first[0][key], dtype=float)
        )
        np.testing.assert_allclose(
            np.asarray(third[0][key], dtype=float), np.asarray(first[0][key], dtype=float)
        )
    np.testing.assert_allclose(
        np.asarray(second[1].prices, dtype=float), np.asarray(first[1].prices, dtype=float)
    )

    for key, value in problem.torch_kwargs.items():
        if before[key] is None:
            assert value.grad is None
        else:
            np.testing.assert_allclose(value.grad.detach().numpy(), before[key].numpy())


def test_return_value_reports_the_tracked_pass_eue():
    """``return_value=True`` hands back the objective's own value on this pass."""
    problem, devices = solved_problem()
    objective = UnservedEnergyObjective(devices)
    dtheta, adjoint, value = problem.backward_objective(objective, return_value=True)
    assert value == pytest.approx(float(objective(problem.state, la=np)), rel=1e-9)
    assert value > 0.0

    only_grad, only_value = problem.backward_objective(
        objective, return_adjoint=False, return_value=True
    )
    assert only_value == pytest.approx(value, rel=1e-12)
    np.testing.assert_allclose(
        np.asarray(only_grad["generator_capacity"], dtype=float),
        np.asarray(dtheta["generator_capacity"], dtype=float),
    )
    assert adjoint is not None


# ---------------------------------------------------------------------------
# (iv) a numpy forward pass is refused
# ---------------------------------------------------------------------------


def test_numpy_forward_pass_is_refused():
    """``forward(requires_grad=False)`` leaves no graph; say so, don't crash on it.

    This is the failure the CH3 checkpointer produced: its full pass replaces
    every subproblem's ``torch_state`` with numpy, and the tracker's next call
    died inside autograd with ``'numpy.ndarray' object has no attribute
    'requires_grad'``, swallowed as a generic failure.
    """
    problem, devices = make_problem()
    problem.forward(requires_grad=False, **theta())
    with pytest.raises(RuntimeError, match="numpy"):
        problem.backward_objective(UnservedEnergyObjective(devices))

    # And it recovers: a differentiable pass makes it work again.
    problem.forward(requires_grad=True, **theta())
    dtheta, _adjoint = problem.backward_objective(UnservedEnergyObjective(devices))
    assert np.asarray(dtheta["generator_capacity"], dtype=float).min() < 0.0


def test_no_forward_pass_at_all_is_refused():
    problem, devices = make_problem()
    with pytest.raises(RuntimeError, match="forward"):
        problem.backward_objective(UnservedEnergyObjective(devices))


# ---------------------------------------------------------------------------
# (v) the stochastic problem sums its subproblems with the batch weights
# ---------------------------------------------------------------------------


def _stochastic():
    """Two subproblems over two different demand blocks, unit weights."""
    subs = []
    device_sets = []
    for demand in (
        [80.0, 80.0, 80.0, 200.0, 80.0, 80.0],
        [80.0, 80.0, 80.0, 170.0, 90.0, 80.0],
    ):
        network, devices = toy_devices(demand0=demand)
        problem, devices = make_problem(devices=devices, network=network)
        subs.append(problem)
        device_sets.append(devices)
    return StochasticPlanningProblem(subs, weights=[1.0, 1.0]), device_sets


@pytest.mark.parametrize("batch", [[0, 1], [0], [1]])
def test_stochastic_backward_objective_is_the_batch_weighted_sum(batch):
    problem, device_sets = _stochastic()
    problem.forward(requires_grad=True, batch=batch, **theta())
    problem.backward()

    objectives = [UnservedEnergyObjective(d) for d in device_sets]
    dtheta, adjoints, value = problem.backward_objective(objectives, return_value=True)
    assert len(adjoints) == len(batch)

    weights = problem.batch_weights(batch)
    assert len(weights) == len(batch)

    expected_grad = None
    expected_value = 0.0
    for weight, b, adjoint in zip(weights, batch, adjoints):
        sub_grad, sub_adjoint, sub_value = problem.subproblems[b].backward_objective(
            objectives[b], return_value=True
        )
        expected_value += float(weight) * sub_value
        scaled = {
            k: float(weight) * np.asarray(v, dtype=float) for k, v in sub_grad.items()
        }
        expected_grad = (
            scaled
            if expected_grad is None
            else {k: expected_grad[k] + v for k, v in scaled.items()}
        )
        np.testing.assert_allclose(
            np.asarray(adjoint.prices, dtype=float),
            np.asarray(sub_adjoint.prices, dtype=float),
        )

    for key, expected in expected_grad.items():
        np.testing.assert_allclose(np.asarray(dtheta[key], dtype=float), expected, rtol=1e-9)
    assert value == pytest.approx(expected_value, rel=1e-9)
