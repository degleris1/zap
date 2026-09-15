"""The aggregate interface limit on ``DirectedLine`` (import-limit spec, 2026-09-14).

``memory/plans/2026-09-14-import-limit-spec.md`` section 3.1: an optional group
index per row plus one MW cap per group, constraining
``sum_{i in g} power[1][i, t] <= group_limit[g]`` in every hour -- the delivered
(sink-end) MW, which is what ``metrics.imports_mwh`` sums and what the CPUC /
SERVM "simultaneous import limit" is a limit on.

Numbered as the spec's section 5 numbers them:

1. validation (both-or-neither, length, empty group, non-positive, must-flow);
2. a binding cap on a 3-bus cvxpy toy -- equality at the cap, a positive family-2
   dual, a strictly higher objective;
3. inert when the fields are absent -- every path is ``Transporter``'s;
4. ``_inequality_matrices`` against one-sided finite differences of
   ``inequality_constraints``;
5. the implicit gradient at a binding cap against one-sided FD of a re-solve,
   on **HiGHS simplex** (LESSONS 2026-09-09) at a point where no per-link bound
   is simultaneously active (risk R1);
6. ``scale_power`` / ``sample_time``;
7. ``admm_prox_update`` refuses (spec D5).

Plus the importer wiring: ``LoadOptions.import_limit_mw`` on the tiny dataset.
"""

from __future__ import annotations

import unittest

import numpy as np
import pytest

cp = pytest.importorskip("cvxpy")
torch = pytest.importorskip("torch")

from zap.devices import Generator, Load  # noqa: E402
from zap.devices.transporter import DirectedLine  # noqa: E402
from zap.devices.transporter.transporter import Transporter  # noqa: E402
from zap.layer import DispatchLayer  # noqa: E402
from zap.network import PowerNetwork  # noqa: E402
from zap.planning import DispatchCostObjective, InvestmentObjective  # noqa: E402
from zap.planning.problem_cvx import PlanningProblemCVX  # noqa: E402

torch.set_default_dtype(torch.float64)


# ---------------------------------------------------------------------------
# The 3-bus toy
# ---------------------------------------------------------------------------

T = 6
#: Bus 0 carries the load; buses 1 and 2 are "out of state" and reach bus 0 only
#: through the two grouped lines.
DEMAND = 520.0
#: (control capacity, cycling-free -- the parameter of the FD test).
LOCAL_CAPACITY = 200.0
#: Cheap import gen at bus 1, capped below the group limit so that the *second*
#: line has to carry the balance; that is what makes the ``efficiency != 1``
#: distinction between ``power[0]`` and ``power[1]`` observable.
BUS1_CAPACITY = 150.0
BUS2_CAPACITY = 400.0
LINE_CAPACITY = 1000.0
ETA2 = 0.9
GROUP_CAP = 300.0

PARAMETERS = {"generator_capacity": (0, "nominal_capacity")}
THETA = {
    "generator_capacity": np.array([[LOCAL_CAPACITY], [BUS1_CAPACITY], [BUS2_CAPACITY], [1000.0]])
}


def toy_devices(cap: float | None = GROUP_CAP):
    """A ground-free 3-bus system whose two import lines share one group cap.

    Every bus carries an injector, so no ``Ground`` is needed (LESSONS
    2026-09-13: the ``Ground`` device trips ``DispatchOutcome.shape`` in the KKT
    adjoint).  ``cap=None`` builds the same system with no group at all.
    """
    network = PowerNetwork(3)
    ones = np.ones((1, T))
    generators = Generator(
        num_nodes=3,
        name=np.array(["local", "import_b1", "import_b2", "backstop"], dtype=object),
        terminal=np.array([0, 1, 2, 0]),
        nominal_capacity=np.array(THETA["generator_capacity"], copy=True),
        dynamic_capacity=np.ones((4, T)),
        linear_cost=np.array([[40.0], [10.0], [12.0], [500.0]]) * ones,
        emission_rates=np.zeros((4, 1)),
        capital_cost=np.ones((4, 1)),
    )
    loads = Load(
        num_nodes=3,
        name=np.array(["load_b0"], dtype=object),
        terminal=np.array([0]),
        load=np.full((1, T), DEMAND),
        linear_cost=10_000.0 * np.ones((1, 1)),
    )
    group_kwargs = {}
    if cap is not None:
        group_kwargs = {
            "group": np.array([0, 0]),
            "group_limit": np.array([cap]),
            "group_name": np.array(["imports"], dtype=object),
        }
    lines = DirectedLine(
        num_nodes=3,
        name=np.array(["ln_b1", "ln_b2"], dtype=object),
        source_terminal=np.array([1, 2]),
        sink_terminal=np.array([0, 0]),
        min_power=np.zeros(2),
        max_power=np.array([LINE_CAPACITY, LINE_CAPACITY]),
        linear_cost=np.zeros(2),
        efficiency=np.array([1.0, ETA2]),
        **group_kwargs,
    )
    return network, [generators, loads, lines]


def solve_toy(cap: float | None = GROUP_CAP):
    network, devices = toy_devices(cap)
    outcome = network.dispatch(devices, time_horizon=T, solver=cp.HIGHS, add_ground=False)
    return devices, outcome


def bare_line(**kwargs) -> DirectedLine:
    defaults = dict(
        num_nodes=3,
        name=np.array(["ln_b1", "ln_b2"], dtype=object),
        source_terminal=np.array([1, 2]),
        sink_terminal=np.array([0, 0]),
        min_power=np.zeros(2),
        max_power=np.array([LINE_CAPACITY, LINE_CAPACITY]),
        linear_cost=np.zeros(2),
        efficiency=np.array([1.0, ETA2]),
    )
    defaults.update(kwargs)
    return DirectedLine(**defaults)


# ---------------------------------------------------------------------------
# 1. Validation
# ---------------------------------------------------------------------------


class ValidationTests(unittest.TestCase):
    def test_group_without_limit(self):
        with self.assertRaises(ValueError) as ctx:
            bare_line(group=np.array([0, 0]))
        self.assertIn("together", str(ctx.exception))

    def test_limit_without_group(self):
        with self.assertRaises(ValueError) as ctx:
            bare_line(group_limit=np.array([100.0]))
        self.assertIn("together", str(ctx.exception))

    def test_wrong_length(self):
        with self.assertRaises(ValueError) as ctx:
            bare_line(group=np.array([0, 0, 0]), group_limit=np.array([100.0]))
        self.assertIn("3 entries", str(ctx.exception))

    def test_out_of_range_index(self):
        with self.assertRaises(ValueError) as ctx:
            bare_line(group=np.array([0, 1]), group_limit=np.array([100.0]))
        self.assertIn("must lie in", str(ctx.exception))

    def test_empty_group(self):
        """A group with no rows is a constraint that silently does nothing."""
        with self.assertRaises(ValueError) as ctx:
            bare_line(group=np.array([0, 0]), group_limit=np.array([100.0, 200.0]))
        self.assertIn("no rows", str(ctx.exception))

    def test_non_positive_limit(self):
        with self.assertRaises(ValueError) as ctx:
            bare_line(group=np.array([0, 0]), group_limit=np.array([0.0]))
        self.assertIn("> 0", str(ctx.exception))

    def test_non_finite_limit(self):
        with self.assertRaises(ValueError):
            bare_line(group=np.array([0, 0]), group_limit=np.array([np.inf]))

    def test_must_flow_above_the_limit_is_infeasible_by_construction(self):
        with self.assertRaises(ValueError) as ctx:
            bare_line(
                min_power=np.array([0.5, 0.5]),
                nominal_capacity=np.array([1000.0, 1000.0]),
                group=np.array([0, 0]),
                group_limit=np.array([100.0]),
            )
        self.assertIn("infeasible by construction", str(ctx.exception))

    def test_group_name_length(self):
        with self.assertRaises(ValueError) as ctx:
            bare_line(
                group=np.array([0, 0]),
                group_limit=np.array([100.0]),
                group_name=np.array(["a", "b"], dtype=object),
            )
        self.assertIn("group_name", str(ctx.exception))

    def test_ungrouped_rows_are_allowed(self):
        line = bare_line(group=np.array([-1, 0]), group_limit=np.array([100.0]))
        np.testing.assert_allclose(line.group_matrix, np.array([[0.0, 1.0]]))

    def test_group_limit_is_made_dynamic(self):
        line = bare_line(group=np.array([0, 0]), group_limit=np.array([100.0]))
        self.assertEqual(np.asarray(line.group_limit).shape, (1, 1))


# ---------------------------------------------------------------------------
# 2. A binding cap
# ---------------------------------------------------------------------------


class BindingCapTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.devices, cls.outcome = solve_toy(GROUP_CAP)
        cls.free_devices, cls.free_outcome = solve_toy(None)

    def sink(self, outcome):
        return np.asarray(outcome.power[2][1], dtype=float)

    def test_the_cap_binds_exactly(self):
        delivered = self.sink(self.outcome).sum(axis=0)
        np.testing.assert_allclose(delivered, np.full(T, GROUP_CAP), atol=1e-9)

    def test_it_is_the_delivered_mw_that_is_capped_not_the_withdrawn(self):
        """D2: the cap is on ``power[1]``, and ``eta != 1`` makes that visible."""
        withdrawn = -np.asarray(self.outcome.power[2][0], dtype=float).sum(axis=0)
        # bus1 delivers 150 at eta = 1, bus2 the other 150 at eta = 0.9.
        self.assertGreater(float(withdrawn[0]), GROUP_CAP + 1.0)
        np.testing.assert_allclose(
            withdrawn, np.full(T, BUS1_CAPACITY + (GROUP_CAP - BUS1_CAPACITY) / ETA2), atol=1e-6
        )

    def test_the_family_2_dual_is_positive(self):
        duals = self.outcome.local_inequality_duals[2]
        self.assertEqual(len(duals), 3)
        group_dual = np.asarray(duals[2], dtype=float)
        self.assertEqual(group_dual.shape, (1, T))
        self.assertTrue(np.all(group_dual > 1e-6), group_dual)

    def test_the_objective_rises(self):
        self.assertGreater(
            float(self.outcome.problem.value), float(self.free_outcome.problem.value) + 1.0
        )

    def test_the_uncapped_solution_would_have_violated_it(self):
        delivered = self.sink(self.free_outcome).sum(axis=0)
        self.assertTrue(np.all(delivered > GROUP_CAP + 1.0), delivered)

    def test_a_non_binding_cap_reproduces_the_uncapped_objective(self):
        _devices, outcome = solve_toy(2 * LINE_CAPACITY)
        self.assertAlmostEqual(
            float(outcome.problem.value), float(self.free_outcome.problem.value), places=6
        )


# ---------------------------------------------------------------------------
# 3. Inert when the fields are absent
# ---------------------------------------------------------------------------


class InertWhenAbsentTests(unittest.TestCase):
    def setUp(self):
        self.line = bare_line()
        self.power = [
            -np.linspace(10.0, 60.0, 2 * T).reshape(2, T),
            np.linspace(10.0, 55.0, 2 * T).reshape(2, T),
        ]

    def test_no_group_matrix(self):
        self.assertIsNone(self.line.group_matrix)

    def test_inequality_constraints_are_the_transporters(self):
        ours = self.line.inequality_constraints(self.power, None, None)
        theirs = Transporter.inequality_constraints(self.line, self.power, None, None)
        self.assertEqual(len(ours), 2)
        for a, b in zip(ours, theirs):
            np.testing.assert_array_equal(np.asarray(a), np.asarray(b))

    def test_inequality_matrices_are_the_transporters(self):
        families = self.line.inequality_constraints(self.power, None, None)
        ours = self.line.inequality_matrices(families, self.power, None, None)
        theirs = Transporter._inequality_matrices(
            self.line, self.line.get_empty_constraint_matrix(families, self.power, None, None)
        )
        self.assertEqual(len(ours), len(theirs))
        for a, b in zip(ours, theirs):
            np.testing.assert_allclose(a.power[1].toarray(), b.power[1].toarray())

    def test_the_solved_objective_matches_a_non_binding_cap(self):
        _devices, free = solve_toy(None)
        _devices, huge = solve_toy(1e6)
        self.assertAlmostEqual(float(free.problem.value), float(huge.problem.value), places=6)

    def test_admm_prox_still_runs(self):
        line = bare_line().torchify(machine="cpu", dtype=torch.float64)
        power = [torch.zeros(2, T, dtype=torch.float64) for _ in range(2)]
        out, _a, _b = line.admm_prox_update(0.5, None, power, None)
        self.assertEqual(len(out), 2)


# ---------------------------------------------------------------------------
# 4. The KKT block against finite differences of the constraint itself
# ---------------------------------------------------------------------------


class InequalityMatrixTests(unittest.TestCase):
    """``d(families)/d(power)`` by one-sided differences, per family.

    The pattern of ``test_unit_commitment.py``'s ``numerical_jacobian``: the
    families are affine, so a one-sided difference is exact up to round-off.
    """

    STEP = 1e-6

    def line(self):
        return bare_line(group=np.array([0, -1]), group_limit=np.array([GROUP_CAP]))

    def numerical_jacobian(self, line, power, slot, n_families):
        target = power[slot]
        columns = []
        base = [np.asarray(f, dtype=float) for f in line.inequality_constraints(power, None, None)]
        for j in range(target.size):
            bumped = [np.array(p, copy=True) for p in power]
            bumped[slot].ravel()[j] += self.STEP
            plus = [
                np.asarray(f, dtype=float) for f in line.inequality_constraints(bumped, None, None)
            ]
            columns.append([(a - b).ravel() / self.STEP for a, b in zip(plus, base)])
        return [np.column_stack([col[f] for col in columns]) for f in range(n_families)]

    def test_all_three_families_in_both_power_slots(self):
        line = self.line()
        rng = np.random.default_rng(0)
        power = [
            -rng.uniform(1.0, 40.0, (2, T)),
            rng.uniform(1.0, 40.0, (2, T)),
        ]
        families = line.inequality_constraints(power, None, None)
        self.assertEqual(len(families), 3)
        self.assertEqual(np.asarray(families[2]).shape, (1, T))
        matrices = line.inequality_matrices(families, power, None, None)
        for slot in (0, 1):
            expected = self.numerical_jacobian(line, power, slot, len(families))
            for f in range(len(families)):
                analytic = matrices[f].power[slot]
                analytic = (
                    analytic.toarray() if hasattr(analytic, "toarray") else np.asarray(analytic)
                )
                with self.subTest(slot=slot, family=f):
                    np.testing.assert_allclose(analytic, expected[f], atol=1e-6)

    def test_the_kron_maps_hour_t_to_hour_t(self):
        """Row-major flattening: constraint ``g*T + t`` meets variable ``i*T + t``."""
        line = bare_line(group=np.array([0, 0]), group_limit=np.array([GROUP_CAP]))
        power = [np.zeros((2, T)), np.zeros((2, T))]
        families = line.inequality_constraints(power, None, None)
        block = line.inequality_matrices(families, power, None, None)[2].power[1].toarray()
        expected = np.kron(np.ones((1, 2)), np.eye(T))
        np.testing.assert_allclose(block, expected)


# ---------------------------------------------------------------------------
# 5. The implicit gradient at a binding cap
# ---------------------------------------------------------------------------


#: The FD fixture is its own 4-bus system, not ``toy_devices``: test 5 needs the
#: ``DirectedLine`` **as a planning parameter** (ch3's ``setup_parameter_names``
#: always registers ``directedline_capacity``, because ``_build_links`` gives
#: every link a ``capital_cost``), and it needs a link whose own bound binds so
#: that the link gradient is non-trivially non-zero -- without breaking the
#: non-degeneracy the FD gate rests on.  Bus 3's link is therefore **ungrouped**
#: and tight, while both grouped links stay slack behind the interface limit.
FD_T = 6
FD_DEMAND = 520.0
FD_PARAMETERS = {
    "generator_capacity": (0, "nominal_capacity"),
    "line_capacity": (2, "nominal_capacity"),
}
#: (local @40, import_b1 @10, import_b2 @12, import_b3 @20, backstop @500).
FD_GEN_ROWS = {"local": 0, "import_b1": 1, "import_b2": 2, "import_b3": 3, "backstop": 4}
#: (ln_b1 grouped/slack, ln_b2 grouped/slack, ln_b3 ungrouped/**binding**).
FD_LINE_ROWS = {"ln_b1": 0, "ln_b2": 1, "ln_b3": 2}
FD_THETA = {
    "generator_capacity": np.array([[200.0], [150.0], [400.0], [400.0], [1000.0]]),
    "line_capacity": np.array([[1000.0], [1000.0], [50.0]]),
}


def fd_devices(cap: float | None = GROUP_CAP):
    """Four buses: the load bus plus three sources, two of them behind the cap.

    Ground-free (every bus carries an injector), as the KKT adjoint requires
    (LESSONS 2026-09-13).  At ``cap = 300`` the unique optimum is
    ``ln_b1 = 150`` (bus 1's generator is at its cap), ``ln_b2 = 150`` (the
    balance of the interface limit), ``ln_b3 = 50`` (**its own** bound), and the
    local generator interior at 170 MW -- so the group limit, one *ungrouped*
    link bound and one generator bound are active and nothing else is.
    """
    network = PowerNetwork(4)
    ones = np.ones((1, FD_T))
    generators = Generator(
        num_nodes=4,
        name=np.array(["local", "import_b1", "import_b2", "import_b3", "backstop"], dtype=object),
        terminal=np.array([0, 1, 2, 3, 0]),
        nominal_capacity=np.array(FD_THETA["generator_capacity"], copy=True),
        dynamic_capacity=np.ones((5, FD_T)),
        linear_cost=np.array([[40.0], [10.0], [12.0], [20.0], [500.0]]) * ones,
        emission_rates=np.zeros((5, 1)),
        capital_cost=np.ones((5, 1)),
    )
    loads = Load(
        num_nodes=4,
        name=np.array(["load_b0"], dtype=object),
        terminal=np.array([0]),
        load=np.full((1, FD_T), FD_DEMAND),
        linear_cost=10_000.0 * np.ones((1, 1)),
    )
    group_kwargs = {}
    if cap is not None:
        group_kwargs = {
            # Row 2 (`ln_b3`) is deliberately **outside** the group.
            "group": np.array([0, 0, -1]),
            "group_limit": np.array([cap]),
            "group_name": np.array(["imports"], dtype=object),
        }
    efficiency = np.array([1.0, ETA2, 1.0])
    lines = DirectedLine(
        num_nodes=4,
        name=np.array(["ln_b1", "ln_b2", "ln_b3"], dtype=object),
        source_terminal=np.array([1, 2, 3]),
        sink_terminal=np.array([0, 0, 0]),
        min_power=np.zeros(3),
        # Per-unit, folding in the efficiency exactly as `wy_store._build_links`
        # does, so `nominal_capacity` is the MW the parameter carries.
        max_power=efficiency.copy(),
        linear_cost=np.zeros(3),
        efficiency=efficiency,
        nominal_capacity=np.array(FD_THETA["line_capacity"], copy=True),
        capital_cost=np.ones((3, 1)),
        **group_kwargs,
    )
    return network, [generators, loads, lines]


def fd_problem(cap: float | None = GROUP_CAP):
    network, devices = fd_devices(cap)
    layer = DispatchLayer(
        network,
        devices,
        FD_PARAMETERS,
        time_horizon=FD_T,
        # HiGHS simplex, never CLARABEL: interior duals at load-less buses
        # corrupt zero-capacity gradients (LESSONS 2026-09-09).
        solver=cp.HIGHS,
        solver_kwargs={},
        add_ground=False,
    )
    lower = {k: np.zeros_like(v) for k, v in FD_THETA.items()}
    upper = {k: np.full_like(v, 1e4) for k, v in FD_THETA.items()}
    return PlanningProblemCVX(
        DispatchCostObjective(network, devices),
        InvestmentObjective(devices, layer),
        layer,
        lower,
        upper,
    )


def theta(**overrides):
    out = {k: np.array(v, dtype=float, copy=True) for k, v in FD_THETA.items()}
    out.update({k: np.asarray(v, dtype=float) for k, v in overrides.items()})
    return out


class ImplicitGradientTests(unittest.TestCase):
    """``PlanningProblemCVX.backward()`` vs one-sided FD, with the cap binding.

    Both parameter blocks are differentiated, the ``DirectedLine`` included:
    that is the path ``PowerNetwork.kkt_vjp_parameters`` takes, where the device
    is torchified at float32 and the state at float64, and where a ``matmul``
    that does not promote dtypes crashes the first ``backward()`` of every
    capped planning run.

    The point is deliberately **non-degenerate** (risk R1): the group limit is
    active, and the only active per-link bound belongs to a link *outside* the
    group, so no shadow price is split between two families and the implicit
    gradient is the derivative.
    """

    @classmethod
    def setUpClass(cls):
        problem = fd_problem()
        cls.base_value = float(problem.forward(requires_grad=True, **theta()))
        grads = problem.backward()
        cls.dtheta = {k: np.asarray(v, dtype=float).reshape(-1) for k, v in grads.items()}
        cls.state = problem.state
        cls.fd = {}
        for delta in (0.1, 1.0):
            values = {}
            for key, block in FD_THETA.items():
                row_values = []
                for index in range(block.shape[0]):
                    bumped = theta()
                    bumped[key][index, 0] += delta
                    other = fd_problem()
                    row_values.append(
                        (float(other.forward(requires_grad=False, **bumped)) - cls.base_value)
                        / delta
                    )
                values[key] = np.asarray(row_values)
            cls.fd[delta] = values

    # --- the fixture is the point it claims to be ------------------------

    def test_the_active_set_is_the_one_the_docstring_claims(self):
        sink = np.asarray(self.state.power[2][1], dtype=float)
        grouped = sink[[FD_LINE_ROWS["ln_b1"], FD_LINE_ROWS["ln_b2"]], :]
        np.testing.assert_allclose(grouped.sum(axis=0), np.full(FD_T, GROUP_CAP), atol=1e-6)
        # The grouped links are slack: no family-0/1 bound binds with the cap.
        bounds = (np.asarray(FD_THETA["line_capacity"]).reshape(-1) * np.array([1.0, ETA2, 1.0]))[
            :2
        ]
        self.assertTrue(np.all(grouped < bounds[:, None] - 1.0))
        # The ungrouped one is at its bound, which is what makes its gradient
        # non-zero without making the vertex degenerate.
        np.testing.assert_allclose(sink[FD_LINE_ROWS["ln_b3"], :], np.full(FD_T, 50.0), atol=1e-6)

    def test_backward_returns_a_gradient_for_every_parameter_block(self):
        self.assertEqual(set(self.dtheta), set(FD_THETA))

    # --- the gate --------------------------------------------------------

    def assert_matches_fd(self, key: str, index: int):
        for delta, fd in self.fd.items():
            with self.subTest(parameter=key, row=index, delta=delta):
                self.assertLessEqual(
                    abs(self.dtheta[key][index] - fd[key][index]),
                    1e-5 * max(1.0, abs(fd[key][index])),
                )

    def test_generator_gradients_match_finite_differences(self):
        for index in FD_GEN_ROWS.values():
            self.assert_matches_fd("generator_capacity", index)

    def test_line_gradients_match_finite_differences(self):
        """The crash path: the ``DirectedLine`` is the differentiated device."""
        for index in FD_LINE_ROWS.values():
            self.assert_matches_fd("line_capacity", index)

    def test_the_binding_ungrouped_line_is_worth_something(self):
        """A zero-vs-zero FD match would prove nothing; this one is not zero."""
        value = self.dtheta["line_capacity"][FD_LINE_ROWS["ln_b3"]]
        self.assertLess(value, -10.0 * FD_T)

    def test_a_grouped_line_is_worth_nothing_once_the_cap_binds(self):
        """The economics the cap imposes: more import *wire* buys no import."""
        for name in ("ln_b1", "ln_b2"):
            with self.subTest(line=name):
                # Capital cost 1 per MW is all that is left.
                self.assertAlmostEqual(
                    self.dtheta["line_capacity"][FD_LINE_ROWS[name]], 1.0, places=6
                )

    def test_relieving_the_upstream_generator_is_worth_something(self):
        """With the cap binding, cheaper MW *inside* the group still displaces."""
        self.assertLess(self.dtheta["generator_capacity"][FD_GEN_ROWS["import_b1"]], 0.0)

    def test_the_capped_gradient_differs_from_the_uncapped_one(self):
        free = fd_problem(None)
        free.forward(requires_grad=True, **theta())
        free_dtheta = {
            k: np.asarray(v, dtype=float).reshape(-1) for k, v in free.backward().items()
        }
        self.assertGreater(
            abs(
                free_dtheta["generator_capacity"][FD_GEN_ROWS["import_b1"]]
                - self.dtheta["generator_capacity"][FD_GEN_ROWS["import_b1"]]
            ),
            1.0,
        )

    def test_the_dtype_mismatch_the_kkt_path_creates_is_handled(self):
        """The defect itself, in one assertion, without a solve.

        ``kkt_vjp_parameters`` torchifies the devices at
        ``AbstractDevice.DEFAULT_DTYPE`` and the state at float64.  Family 2 must
        survive that, as families 0 and 1 do.
        """
        _network, devices = fd_devices()
        line = devices[2].torchify(machine="cpu")  # DEFAULT_DTYPE, i.e. float32
        self.assertNotEqual(line.group_matrix.dtype, torch.float64)
        power = [
            torch.zeros(3, FD_T, dtype=torch.float64),
            torch.full((3, FD_T), 10.0, dtype=torch.float64),
        ]
        families = line.inequality_constraints(power, None, None, la=torch)
        self.assertEqual(len(families), 3)
        self.assertEqual(families[2].dtype, torch.float64)
        np.testing.assert_allclose(
            families[2].numpy(), np.full((1, FD_T), 20.0 - GROUP_CAP), atol=1e-12
        )


# ---------------------------------------------------------------------------
# 6. scale_power / sample_time
# ---------------------------------------------------------------------------


class ScalingTests(unittest.TestCase):
    def test_scale_power_scales_the_limit(self):
        line = bare_line(group=np.array([0, 0]), group_limit=np.array([GROUP_CAP]))
        line.scale_power(2.0)
        np.testing.assert_allclose(np.asarray(line.group_limit), [[GROUP_CAP / 2.0]])
        # ... and the per-row bounds are untouched by the new code path.
        np.testing.assert_allclose(np.asarray(line.max_power).reshape(-1), [LINE_CAPACITY] * 2)

    def test_scale_power_is_a_no_op_without_a_group(self):
        line = bare_line()
        line.scale_power(2.0)
        self.assertIsNone(line.group_limit)

    def test_sample_time_preserves_the_group(self):
        line = bare_line(group=np.array([0, 0]), group_limit=np.array([GROUP_CAP]))
        block = line.sample_time(np.arange(3), T)
        np.testing.assert_allclose(np.asarray(block.group_limit), [[GROUP_CAP]])
        np.testing.assert_allclose(block.group_matrix, line.group_matrix)
        self.assertEqual(block.time_horizon, 0)

    def test_torchify_moves_the_group_matrix(self):
        line = bare_line(group=np.array([0, 0]), group_limit=np.array([GROUP_CAP])).torchify(
            machine="cpu", dtype=torch.float64
        )
        self.assertIsInstance(line.group_matrix, torch.Tensor)
        self.assertIsInstance(line.group_limit, torch.Tensor)
        power = [torch.zeros(2, T, dtype=torch.float64) for _ in range(2)]
        families = line.inequality_constraints(power, None, None, la=torch)
        self.assertEqual(len(families), 3)
        np.testing.assert_allclose(families[2].numpy(), np.full((1, T), -GROUP_CAP), atol=1e-12)


# ---------------------------------------------------------------------------
# 7. ADMM refuses
# ---------------------------------------------------------------------------


class AdmmRefusesTests(unittest.TestCase):
    def test_prox_raises(self):
        line = bare_line(group=np.array([0, 0]), group_limit=np.array([GROUP_CAP])).torchify(
            machine="cpu", dtype=torch.float64
        )
        power = [torch.zeros(2, T, dtype=torch.float64) for _ in range(2)]
        with self.assertRaises(NotImplementedError) as ctx:
            line.admm_prox_update(0.5, None, power, None)
        self.assertIn("import_limit", str(ctx.exception))


# ---------------------------------------------------------------------------
# The importer wiring
# ---------------------------------------------------------------------------


class ImporterTests(unittest.TestCase):
    """``LoadOptions.import_limit_mw`` on the tiny dataset (spec 3.2)."""

    def setUp(self):
        import tempfile
        from pathlib import Path

        from zap.importers.wy_store import convert_dataset
        from zap.tests.fixtures.tiny_dataset import write_tiny_dataset

        self._tmp = tempfile.TemporaryDirectory()
        self.root = write_tiny_dataset(Path(self._tmp.name) / "tiny", n_hours=24, years=(2020,))
        convert_dataset(self.root)

    def tearDown(self):
        self._tmp.cleanup()

    def system(self, **kwargs):
        from zap.importers.wy_store import HourWindow, LoadOptions, load_system

        return load_system(
            self.root,
            LoadOptions(years=(2020,), window=HourWindow(start=0, stop=24), **kwargs),
        )

    def line_device(self, loaded):
        return loaded.devices[loaded.index.device_index["DirectedLine"]]

    def test_no_cap_leaves_the_device_ungrouped(self):
        loaded = self.system()
        self.assertIsNone(self.line_device(loaded).group_matrix)
        self.assertIsNone(loaded.meta["import_limit_mw"])
        self.assertGreater(loaded.meta["import_link_capacity_mw"], 0.0)

    def test_the_cap_groups_exactly_the_import_carrier_rows(self):
        from zap.importers.wy_store import IMPORT_LINK_CARRIER

        loaded = self.system(import_limit_mw=30.0)
        line = self.line_device(loaded)
        carriers = np.asarray(loaded.index.carrier["DirectedLine"]).astype(str)
        expected = (carriers == IMPORT_LINK_CARRIER).astype(float)
        np.testing.assert_allclose(line.group_matrix, expected.reshape(1, -1))
        np.testing.assert_allclose(np.asarray(line.group_limit), [[30.0]])
        self.assertEqual(loaded.meta["import_limit_mw"], 30.0)

    def test_the_cap_is_scaled_with_power_unit(self):
        loaded = self.system(import_limit_mw=30.0, power_unit=0.001)
        np.testing.assert_allclose(np.asarray(self.line_device(loaded).group_limit), [[30_000.0]])

    def test_a_non_binding_cap_warns(self):
        with self.assertLogs("zap.importers.wy_store", level="WARNING") as logs:
            self.system(import_limit_mw=1e6)
        self.assertTrue(any("cannot bind" in line for line in logs.output))

    def test_a_dataset_with_no_import_links_is_refused(self):
        import pandas as pd

        from zap.importers.wy_store import _build_links

        static = {
            "links": pd.DataFrame(
                {
                    "bus0": ["a"],
                    "bus1": ["b"],
                    "carrier": ["AC"],
                    "p_nom": [10.0],
                    "efficiency": [1.0],
                    "marginal_cost": [0.0],
                    "capital_cost": [0.0],
                    "p_min_pu": [0.0],
                    "p_max_pu": [1.0],
                },
                index=pd.Index(["ln"], name="name"),
            )
        }
        from zap.importers.wy_store import HourWindow, LoadOptions

        options = LoadOptions(
            years=(2020,), window=HourWindow(start=0, stop=2), import_limit_mw=5.0
        )
        with self.assertRaises(ValueError) as ctx:
            _build_links(static, None, {"a": 0, "b": 1}, options)
        self.assertIn("nothing to cap", str(ctx.exception))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
