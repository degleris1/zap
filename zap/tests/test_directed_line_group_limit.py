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
        # A scalar builds the flat `(1,)` cap; an array-like is passed through,
        # so the same fixture serves the `(1, T)` hour-of-day profile.
        limit = np.asarray(cap, dtype=float)
        group_kwargs = {
            # Row 2 (`ln_b3`) is deliberately **outside** the group.
            "group": np.array([0, 0, -1]),
            "group_limit": limit if limit.ndim else np.array([float(cap)]),
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


#: An **hour-of-day** cap on the same fixture (import-limit profile spec T-G7):
#: the group binds in hours 0-2 at 300 MW and is slack in hours 3-5 at 800 MW,
#: so one `backward()` covers both regimes.  At 800 the grouped links deliver
#: 150 + 360 = 510 MW (both upstream generators at their capacity) and the
#: *ungrouped* `ln_b3` picks up the remaining 10 MW **interior** to its bound,
#: which keeps the vertex non-degenerate in those hours too.
HOURLY_CAP = np.array([[300.0, 300.0, 300.0, 800.0, 800.0, 800.0]])


def fd_problem(cap=GROUP_CAP):
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



# ===========================================================================
# The hour-of-day import-limit profile
# (memory/plans/2026-09-17-import-limit-profile-spec.md, tests T-G1 ... T-G11)
# ===========================================================================


def hourly_toy(limit):
    """A 2-bus, 4-hour toy: cheap imports behind a per-hour cap, expensive local.

    Bus 0 carries the 10 MW load and a 100 $/MWh local generator; bus 1 carries a
    **5 MW** 10 $/MWh generator that reaches bus 0 only through one grouped line.
    Capping the import generator below the off-peak limit is deliberate: it makes
    the group constraint *strictly slack* in hours 0-1 and *strictly binding* in
    hours 2-3, so the family-2 dual is uniquely 0 in the first pair and uniquely
    positive in the second.  (With imports pinned at the cap in every hour the
    dual is degenerate -- exactly risk R1 -- and a dual test would be reading a
    simplex vertex, not a derivative.)
    """
    hours = 4
    network = PowerNetwork(2)
    generators = Generator(
        num_nodes=2,
        name=np.array(["local", "import"], dtype=object),
        terminal=np.array([0, 1]),
        nominal_capacity=np.array([[20.0], [5.0]]),
        dynamic_capacity=np.ones((2, hours)),
        linear_cost=np.array([[100.0], [10.0]]) * np.ones((1, hours)),
        emission_rates=np.zeros((2, 1)),
    )
    loads = Load(
        num_nodes=2,
        name=np.array(["load_b0"], dtype=object),
        terminal=np.array([0]),
        load=np.full((1, hours), 10.0),
        linear_cost=10_000.0 * np.ones((1, 1)),
    )
    lines = DirectedLine(
        num_nodes=2,
        name=np.array(["ln"], dtype=object),
        source_terminal=np.array([1]),
        sink_terminal=np.array([0]),
        min_power=np.zeros(1),
        max_power=np.array([20.0]),
        linear_cost=np.zeros(1),
        efficiency=np.array([1.0]),
        group=np.array([0]),
        group_limit=np.asarray(limit, dtype=float),
        group_name=np.array(["imports"], dtype=object),
    )
    return network, [generators, loads, lines], hours


def solve_hourly_toy(limit):
    network, devices, hours = hourly_toy(limit)
    return devices, network.dispatch(
        devices, time_horizon=hours, solver=cp.HIGHS, add_ground=False
    )


# ---------------------------------------------------------------------------
# T-G1 / T-G2 / T-G3 / T-G4: shapes, slicing, scaling
# ---------------------------------------------------------------------------


class HourlyLimitShapeTests(unittest.TestCase):
    """T-G1: ``(G,)``, ``(G, 1)`` and ``(G, T)`` are all accepted; nothing else."""

    def test_the_three_accepted_shapes(self):
        for limit, expected in (
            (np.array([GROUP_CAP]), (1, 1)),
            (np.array([[GROUP_CAP]]), (1, 1)),
            (np.full((1, T), GROUP_CAP), (1, T)),
        ):
            with self.subTest(shape=np.shape(limit)):
                line = bare_line(group=np.array([0, 0]), group_limit=limit)
                self.assertEqual(np.asarray(line.group_limit).shape, expected)

    def test_a_3d_limit_is_refused(self):
        with self.assertRaises(ValueError) as ctx:
            bare_line(group=np.array([0, 0]), group_limit=np.ones((1, 1, T)))
        self.assertIn("(G,), (G, 1) or (G, T)", str(ctx.exception))

    def test_the_group_count_is_the_first_axis_not_the_size(self):
        """A ``(1, T)`` limit is **one** group, not ``T`` of them."""
        line = bare_line(group=np.array([0, 0]), group_limit=np.full((1, T), GROUP_CAP))
        self.assertEqual(np.asarray(line.group_matrix).shape, (1, 2))

    def test_an_empty_limit_is_refused(self):
        with self.assertRaises(ValueError) as ctx:
            bare_line(group=np.array([-1, -1]), group_limit=np.zeros((0, T)))
        self.assertIn("empty", str(ctx.exception))

    def test_one_non_finite_hour_is_refused(self):
        limit = np.full((1, T), GROUP_CAP)
        limit[0, 3] = np.inf
        with self.assertRaises(ValueError) as ctx:
            bare_line(group=np.array([0, 0]), group_limit=limit)
        self.assertIn("finite", str(ctx.exception))

    def test_one_non_positive_hour_is_refused(self):
        limit = np.full((1, T), GROUP_CAP)
        limit[0, 2] = 0.0
        with self.assertRaises(ValueError):
            bare_line(group=np.array([0, 0]), group_limit=limit)

    def test_must_flow_above_the_limit_in_one_hour_only_is_refused(self):
        """The ``min(axis=1)`` rule: infeasible in *any* hour is infeasible."""
        limit = np.full((1, T), 500.0)
        limit[0, 4] = 10.0
        with self.assertRaises(ValueError) as ctx:
            bare_line(
                group=np.array([0, 0]),
                group_limit=limit,
                min_power=np.array([0.1, 0.0]),
                nominal_capacity=np.array([[1000.0], [1000.0]]),
            )
        self.assertIn("infeasible by construction", str(ctx.exception))

    def test_an_hourly_min_power_is_reduced_before_the_comparison(self):
        """``min_power`` may be ``(N, T)`` too -- e.g. the rolling-horizon fixture.

        The guard reduces to one number per group (largest must-flow against
        tightest hour); before the profile work it flattened ``min_power`` and
        the shapes did not even multiply.
        """
        ramp = np.vstack([np.linspace(0.0, 0.05, T), np.zeros(T)])
        line = bare_line(
            group=np.array([0, 0]),
            group_limit=np.full((1, T), 500.0),
            min_power=ramp,
            nominal_capacity=np.array([[1000.0], [1000.0]]),
        )
        self.assertEqual(np.asarray(line.group_limit).shape, (1, T))
        with self.assertRaises(ValueError) as ctx:
            bare_line(
                group=np.array([0, 0]),
                group_limit=np.full((1, T), 10.0),
                min_power=ramp,
                nominal_capacity=np.array([[1000.0], [1000.0]]),
            )
        self.assertIn("infeasible by construction", str(ctx.exception))

    def test_a_must_flow_below_every_hour_is_accepted(self):
        limit = np.full((1, T), 500.0)
        limit[0, 4] = 200.0
        line = bare_line(
            group=np.array([0, 0]),
            group_limit=limit,
            min_power=np.array([0.1, 0.0]),
            nominal_capacity=np.array([[1000.0], [1000.0]]),
        )
        self.assertEqual(np.asarray(line.group_limit).shape, (1, T))


class WrongWidthLimitTests(unittest.TestCase):
    """T-G2: a ``(G, T')`` limit with ``T' != T`` must fail loudly at solve.

    ``Transporter.time_horizon`` is a hard-coded ``0``, so neither
    ``check_block_horizon`` nor ``network.py``'s horizon assertion sees an
    unsliced ``group_limit``; only cvxpy's broadcast does (profile spec R4).
    Pinning it here is what stands in for a structural guard.
    """

    def test_it_raises(self):
        limit = np.full((1, T - 2), GROUP_CAP)
        network, devices = toy_devices(None)
        line = bare_line(group=np.array([0, 0]), group_limit=limit)
        devices = devices[:2] + [line]
        with self.assertRaises(ValueError):
            network.dispatch(devices, time_horizon=T, solver=cp.HIGHS, add_ground=False)

    def test_numpy_evaluation_raises_too(self):
        line = bare_line(group=np.array([0, 0]), group_limit=np.full((1, T - 2), GROUP_CAP))
        power = [np.zeros((2, T)), np.zeros((2, T))]
        with self.assertRaises(ValueError):
            line.inequality_constraints(power, None, None)


class HourlySampleTimeTests(unittest.TestCase):
    """T-G3: ``sample_time`` slices a ``(G, T)`` limit and leaves ``(G, 1)`` alone."""

    def test_an_hourly_limit_is_sliced(self):
        limit = np.arange(1.0, T + 1.0).reshape(1, T) * 100.0
        line = bare_line(group=np.array([0, 0]), group_limit=limit)
        periods = np.array([1, 3, 4])
        block = line.sample_time(periods, T)
        np.testing.assert_allclose(np.asarray(block.group_limit), limit[:, periods])
        np.testing.assert_allclose(block.group_matrix, line.group_matrix)
        # ... and the original is untouched.
        np.testing.assert_allclose(np.asarray(line.group_limit), limit)

    def test_a_flat_limit_is_untouched(self):
        line = bare_line(group=np.array([0, 0]), group_limit=np.array([GROUP_CAP]))
        block = line.sample_time(np.arange(3), T)
        np.testing.assert_allclose(np.asarray(block.group_limit), [[GROUP_CAP]])

    def test_the_sliced_device_solves_and_respects_the_sliced_cap(self):
        limit = np.array([[8.0, 8.0, 2.0, 2.0]])
        network, devices, hours = hourly_toy(limit)
        periods = np.array([2, 3])
        sliced = [d.sample_time(periods, hours) for d in devices]
        outcome = network.dispatch(sliced, time_horizon=2, solver=cp.HIGHS, add_ground=False)
        np.testing.assert_allclose(
            np.asarray(outcome.power[2][1])[0], np.full(2, 2.0), atol=1e-6
        )

    def test_a_parametrised_limit_refuses_to_be_sliced(self):
        line = bare_line(group=np.array([0, 0]), group_limit=np.full((1, T), GROUP_CAP))
        line.group_limit = cp.Parameter((1, T), value=np.full((1, T), GROUP_CAP))
        with self.assertRaises(ValueError) as ctx:
            line.sample_time(np.arange(3), T)
        self.assertIn("cp.Parameter", str(ctx.exception))


class HourlyScalingTests(unittest.TestCase):
    """T-G4: ``scale_power`` is elementwise on a ``(G, T)`` limit."""

    def test_every_entry_is_scaled(self):
        limit = np.arange(1.0, T + 1.0).reshape(1, T) * 100.0
        line = bare_line(group=np.array([0, 0]), group_limit=limit)
        line.scale_power(2.0)
        np.testing.assert_allclose(np.asarray(line.group_limit), limit / 2.0)


# ---------------------------------------------------------------------------
# T-G5: the economics of an hourly cap
# ---------------------------------------------------------------------------


class HourlyEconomicsTests(unittest.TestCase):
    """The cap has to bite hour by hour, not once per block."""

    HOURLY = np.array([[8.0, 8.0, 2.0, 2.0]])

    @classmethod
    def setUpClass(cls):
        cls.devices, cls.outcome = solve_hourly_toy(cls.HOURLY)
        cls.flat_loose = solve_hourly_toy(np.array([8.0]))[1]
        cls.flat_tight = solve_hourly_toy(np.array([2.0]))[1]

    def imports(self, outcome):
        return np.asarray(outcome.power[2][1], dtype=float)[0]

    def test_the_imports_follow_the_profile(self):
        # 5 MW (the generator's own cap) while the limit is 8, then 2 MW.
        np.testing.assert_allclose(
            self.imports(self.outcome), np.array([5.0, 5.0, 2.0, 2.0]), atol=1e-6
        )

    def test_the_cap_is_respected_in_every_hour(self):
        self.assertTrue(np.all(self.imports(self.outcome) <= self.HOURLY[0] + 1e-6))

    def test_the_family_2_dual_is_zero_off_peak_and_positive_on_peak(self):
        dual = np.asarray(self.outcome.local_inequality_duals[2][2], dtype=float)[0]
        self.assertEqual(dual.shape, (4,))
        np.testing.assert_allclose(dual[:2], np.zeros(2), atol=1e-6)
        self.assertTrue(np.all(dual[2:] > 1.0))

    def test_the_objective_sits_between_the_two_flat_caps(self):
        hourly = float(self.outcome.problem.value)
        self.assertGreater(hourly, float(self.flat_loose.problem.value) + 1.0)
        self.assertLess(hourly, float(self.flat_tight.problem.value) - 1.0)

    def test_a_flat_profile_reproduces_the_flat_cap_exactly(self):
        """A ``(G, T)`` limit of one repeated value is the ``(G, 1)`` problem."""
        repeated = solve_hourly_toy(np.full((1, 4), 2.0))[1]
        self.assertAlmostEqual(
            float(repeated.problem.value) / float(self.flat_tight.problem.value), 1.0, places=9
        )


# ---------------------------------------------------------------------------
# T-G6: the RHS does not enter the Jacobian
# ---------------------------------------------------------------------------


class HourlyInequalityMatrixTests(InequalityMatrixTests):
    """Same finite-difference gate as test 4, with a ``(G, T)`` limit."""

    def line(self):
        limit = np.arange(1.0, T + 1.0).reshape(1, T) * 50.0 + GROUP_CAP
        return bare_line(group=np.array([0, -1]), group_limit=limit)

    def test_the_jacobian_is_identical_to_the_flat_one(self):
        flat = bare_line(group=np.array([0, -1]), group_limit=np.array([GROUP_CAP]))
        hourly = self.line()
        power = [np.zeros((2, T)), np.full((2, T), 5.0)]
        blocks = []
        for line in (flat, hourly):
            families = line.inequality_constraints(power, None, None)
            blocks.append(line.inequality_matrices(families, power, None, None)[2].power[1])
        np.testing.assert_allclose(blocks[0].toarray(), blocks[1].toarray())


# ---------------------------------------------------------------------------
# T-G7: the implicit gradient with an hourly cap
# ---------------------------------------------------------------------------


class HourlyImplicitGradientTests(ImplicitGradientTests):
    """``ImplicitGradientTests`` on :data:`HOURLY_CAP` -- binding in some hours only.

    Inherits the FD gate and re-derives the fixture's own active-set claim,
    which differs: the cap binds in hours 0-2 (300 MW) and is slack in hours 3-5
    (800 MW), where ``ln_b3`` is interior instead of at its bound.
    """

    @classmethod
    def setUpClass(cls):
        problem = fd_problem(HOURLY_CAP)
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
                    other = fd_problem(HOURLY_CAP)
                    row_values.append(
                        (float(other.forward(requires_grad=False, **bumped)) - cls.base_value)
                        / delta
                    )
                values[key] = np.asarray(row_values)
            cls.fd[delta] = values

    def test_the_active_set_is_the_one_the_docstring_claims(self):
        sink = np.asarray(self.state.power[2][1], dtype=float)
        grouped = sink[[FD_LINE_ROWS["ln_b1"], FD_LINE_ROWS["ln_b2"]], :].sum(axis=0)
        # Binding in the first three hours, strictly slack in the last three.
        np.testing.assert_allclose(grouped[:3], np.full(3, 300.0), atol=1e-6)
        self.assertTrue(np.all(grouped[3:] < HOURLY_CAP[0, 3:] - 1.0))
        # The ungrouped link is at its bound only while the cap binds.
        np.testing.assert_allclose(sink[FD_LINE_ROWS["ln_b3"], :3], np.full(3, 50.0), atol=1e-6)
        self.assertTrue(np.all(sink[FD_LINE_ROWS["ln_b3"], 3:] < 49.0))

    def test_a_grouped_line_is_worth_nothing_once_the_cap_binds(self):
        """Still nothing: the grouped links are slack in the loose hours too."""
        for name in ("ln_b1", "ln_b2"):
            with self.subTest(line=name):
                self.assertAlmostEqual(
                    self.dtheta["line_capacity"][FD_LINE_ROWS[name]], 1.0, places=6
                )

    def test_the_binding_ungrouped_line_is_worth_something(self):
        value = self.dtheta["line_capacity"][FD_LINE_ROWS["ln_b3"]]
        self.assertLess(value, -10.0)

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

    def test_the_gradient_differs_from_the_flat_capped_one(self):
        """The hourly cap is a different model from the flat one it is built on."""
        flat = fd_problem(GROUP_CAP)
        flat.forward(requires_grad=True, **theta())
        flat_dtheta = {
            k: np.asarray(v, dtype=float).reshape(-1) for k, v in flat.backward().items()
        }
        self.assertGreater(
            abs(
                flat_dtheta["line_capacity"][FD_LINE_ROWS["ln_b3"]]
                - self.dtheta["line_capacity"][FD_LINE_ROWS["ln_b3"]]
            ),
            1.0,
        )

    def test_the_dtype_mismatch_the_kkt_path_creates_is_handled(self):
        """float32 device, float64 state -- now with a ``(G, T)`` limit."""
        _network, devices = fd_devices(HOURLY_CAP)
        line = devices[2].torchify(machine="cpu")  # DEFAULT_DTYPE, i.e. float32
        self.assertNotEqual(line.group_matrix.dtype, torch.float64)
        self.assertEqual(tuple(line.group_limit.shape), (1, FD_T))
        power = [
            torch.zeros(3, FD_T, dtype=torch.float64),
            torch.full((3, FD_T), 10.0, dtype=torch.float64),
        ]
        families = line.inequality_constraints(power, None, None, la=torch)
        self.assertEqual(len(families), 3)
        self.assertEqual(families[2].dtype, torch.float64)
        np.testing.assert_allclose(
            families[2].numpy(), 20.0 - HOURLY_CAP, atol=1e-3
        )


# ---------------------------------------------------------------------------
# T-G8: group_limit as a cp.Parameter on a retained problem
# ---------------------------------------------------------------------------


class ParametrizedGroupLimitTests(unittest.TestCase):
    """``parametrize_devices`` accepts ``group_limit`` and the problem stays DPP."""

    LOW = np.array([[8.0, 8.0, 2.0, 2.0]])
    HIGH = np.array([[8.0, 8.0, 6.0, 6.0]])

    def build(self, limit):
        network, devices, hours = hourly_toy(limit)
        return network.build_dispatch(
            devices, time_horizon=hours, add_ground=False, parametrize={2: ["group_limit"]}
        )

    def test_it_is_registered(self):
        from zap.network import PARAMETRIZABLE_ATTRS

        self.assertIn("group_limit", PARAMETRIZABLE_ATTRS[DirectedLine])

    def test_the_built_problem_is_dpp(self):
        self.assertTrue(self.build(self.LOW).is_dpp())

    def test_rewriting_the_limit_matches_a_rebuilt_problem(self):
        retained = self.build(self.LOW)
        retained.set_parameters({(2, "group_limit"): self.HIGH})
        theirs = retained.solve(solver=cp.HIGHS)
        rebuilt = solve_hourly_toy(self.HIGH)[1]
        self.assertAlmostEqual(
            float(theirs.problem.value) / float(rebuilt.problem.value), 1.0, places=9
        )
        np.testing.assert_allclose(
            np.asarray(theirs.power[2][1], dtype=float),
            np.asarray(rebuilt.power[2][1], dtype=float),
            atol=1e-9,
        )

    def test_the_first_solve_is_the_value_it_was_built_with(self):
        retained = self.build(self.LOW)
        theirs = retained.solve(solver=cp.HIGHS)
        rebuilt = solve_hourly_toy(self.LOW)[1]
        self.assertAlmostEqual(
            float(theirs.problem.value) / float(rebuilt.problem.value), 1.0, places=9
        )


# ---------------------------------------------------------------------------
# T-G9 / T-G10: the importer's calendar and expansion helpers
# ---------------------------------------------------------------------------


SERVM_2025 = (11040.0,) * 14 + (8700.0, 6300.0) + (4000.0,) * 6 + (6300.0, 8700.0)


class CalendarHelperTests(unittest.TestCase):
    """T-G10: ``local_hour_of_day`` / ``local_month`` on the fixed PDT grid."""

    def setUp(self):
        from zap.importers import wy_store

        self.mod = wy_store

    def test_local_hour_of_day(self):
        np.testing.assert_array_equal(
            self.mod.local_hour_of_day(np.array([7, 6, 23, 8742])), np.array([0, 23, 16, 23])
        )

    def test_the_offset_matches_the_windows_local_midnight(self):
        """Window start 7 is local hour 0 -- the convention `lole_days` uses."""
        self.assertEqual(int(self.mod.local_hour_of_day(np.array([7]))[0]), 0)
        self.assertEqual(self.mod.DEFAULT_LOCAL_OFFSET_HOURS, 7)

    def test_local_month_wraps_the_first_hours_into_december(self):
        np.testing.assert_array_equal(self.mod.local_month(np.array([0, 6])), np.array([12, 12]))
        np.testing.assert_array_equal(self.mod.local_month(np.array([7])), np.array([1]))

    def test_month_boundaries_match_the_non_leap_table(self):
        edges = np.cumsum(self.mod.NON_LEAP_MONTH_HOURS)
        self.assertEqual(int(edges[-1]), 8760)
        self.assertEqual(int(self.mod.NON_LEAP_MONTH_HOURS[1]), 672)  # 28-day February
        for month, edge in enumerate(edges[:-1], start=1):
            with self.subTest(month=month):
                # Local hour `edge - 1` is the last of `month`, `edge` the first
                # of the next; add the offset to get the stored UTC hour.
                last = int(self.mod.local_month(np.array([edge - 1 + 7]))[0])
                first = int(self.mod.local_month(np.array([edge + 7]))[0])
                self.assertEqual(last, month)
                self.assertEqual(first, month + 1)

    def test_a_leap_length_year_is_refused(self):
        with self.assertRaises(ValueError) as ctx:
            self.mod.local_month(np.array([0]), hours_per_year=8784)
        self.assertIn("NON_LEAP_MONTH_HOURS", str(ctx.exception))


class HourlyImportLimitHelperTests(unittest.TestCase):
    """T-G9: ``hourly_import_limit`` builds the loaded grid, year-major."""

    def setUp(self):
        from zap.importers.wy_store import HourWindow, hourly_import_limit

        self.window = HourWindow(start=7, stop=8743)
        self.fn = hourly_import_limit
        self.HourWindow = HourWindow

    def call(self, **kwargs):
        args = dict(
            base_mw=11040.0,
            profile_mw=SERVM_2025,
            months=None,
            years=(2013,),
            window=self.window,
            hours_per_year=8760,
        )
        args.update(kwargs)
        return self.fn(**args)

    def test_no_profile_returns_the_flat_base(self):
        out = self.call(profile_mw=None)
        self.assertEqual(out.shape, (1,))
        self.assertEqual(float(out[0]), 11040.0)

    def test_the_shape_is_the_loaded_grid(self):
        self.assertEqual(self.call().shape, (1, 8736))

    def test_the_profile_lands_on_the_right_local_hours(self):
        out = self.call()[0]
        hours = np.arange(self.window.start, self.window.stop)
        local = (hours - 7) % 24
        np.testing.assert_allclose(out, np.asarray(SERVM_2025)[local])
        # The window opens at local midnight, so hour 0 is the base cap.
        self.assertEqual(float(out[0]), 11040.0)
        self.assertEqual(int((out == 4000.0).sum()), 6 * 364)

    def test_two_years_are_tiled_year_major(self):
        one = self.call(years=(2013,))
        two = self.call(years=(2013, 2014))
        self.assertEqual(two.shape, (1, 2 * 8736))
        np.testing.assert_allclose(two[0, :8736], one[0])
        np.testing.assert_allclose(two[0, 8736:], one[0])

    def test_the_month_filter_restores_the_base_cap_outside_its_months(self):
        june_sep = self.call(months=[6, 7, 8, 9])[0]
        hours = np.arange(self.window.start, self.window.stop)
        from zap.importers.wy_store import local_month

        inside = np.isin(local_month(hours), [6, 7, 8, 9])
        np.testing.assert_allclose(june_sep[~inside], np.full(int((~inside).sum()), 11040.0))
        # 122 days x 6 plateau hours.
        self.assertEqual(int((june_sep == 4000.0).sum()), 122 * 6)

    def test_months_on_a_non_8760_year_is_refused(self):
        with self.assertRaises(ValueError) as ctx:
            self.call(months=[6], hours_per_year=8784)
        self.assertIn("NON_LEAP_MONTH_HOURS", str(ctx.exception))

    def test_all_months_needs_no_calendar(self):
        """Which is what lets the shipped default run on a short fixture."""
        out = self.fn(
            base_mw=100.0,
            profile_mw=tuple(np.linspace(10.0, 100.0, 24)),
            months=None,
            years=(2020,),
            window=self.HourWindow(start=0, stop=48),
            hours_per_year=48,
        )
        self.assertEqual(out.shape, (1, 48))

    def test_a_profile_of_the_wrong_length_is_refused(self):
        with self.assertRaises(ValueError) as ctx:
            self.call(profile_mw=SERVM_2025[:23])
        self.assertIn("24", str(ctx.exception))

    def test_a_profile_above_its_base_is_refused(self):
        with self.assertRaises(ValueError) as ctx:
            self.call(base_mw=10_000.0)
        self.assertIn("base cap", str(ctx.exception))

    def test_a_non_positive_profile_entry_is_refused(self):
        bad = list(SERVM_2025)
        bad[3] = 0.0
        with self.assertRaises(ValueError):
            self.call(profile_mw=bad)

    def test_an_empty_month_list_is_refused(self):
        with self.assertRaises(ValueError):
            self.call(months=[])

    def test_a_month_outside_1_12_is_refused(self):
        with self.assertRaises(ValueError):
            self.call(months=[0])


class ImporterProfileTests(ImporterTests):
    """T-G11: the profile reaches the device through ``load_system``."""

    PROFILE = tuple(float(100 + 10 * h) for h in range(24))

    def test_a_profile_gives_the_device_an_hourly_limit(self):
        loaded = self.system(
            import_limit_mw=max(self.PROFILE),
            import_limit_profile_mw=self.PROFILE,
        )
        limit = np.asarray(self.line_device(loaded).group_limit)
        self.assertEqual(limit.shape, (1, 24))
        # The tiny fixture's window is [0, 24) and the offset is 7 hours.
        expected = np.asarray(self.PROFILE)[(np.arange(24) - 7) % 24]
        np.testing.assert_allclose(limit[0], expected)

    def test_no_profile_still_gives_a_flat_limit(self):
        loaded = self.system(import_limit_mw=30.0)
        np.testing.assert_allclose(np.asarray(self.line_device(loaded).group_limit), [[30.0]])

    def test_the_meta_records_the_profile_and_the_tightest_hour(self):
        loaded = self.system(
            import_limit_mw=max(self.PROFILE),
            import_limit_profile_mw=self.PROFILE,
        )
        self.assertEqual(loaded.meta["import_limit_mw"], max(self.PROFILE))
        self.assertEqual(loaded.meta["import_limit_profile_mw"], list(self.PROFILE))
        self.assertIsNone(loaded.meta["import_limit_months"])
        self.assertEqual(loaded.meta["import_limit_local_offset_hours"], 7)
        self.assertEqual(loaded.meta["import_limit_mw_min"], min(self.PROFILE))

    def test_the_meta_min_is_the_cap_itself_without_a_profile(self):
        loaded = self.system(import_limit_mw=30.0)
        self.assertEqual(loaded.meta["import_limit_mw_min"], 30.0)
        self.assertIsNone(loaded.meta["import_limit_profile_mw"])

    def test_the_profile_is_scaled_with_power_unit(self):
        loaded = self.system(
            import_limit_mw=max(self.PROFILE),
            import_limit_profile_mw=self.PROFILE,
            power_unit=0.001,
        )
        limit = np.asarray(self.line_device(loaded).group_limit)
        expected = np.asarray(self.PROFILE)[(np.arange(24) - 7) % 24] * 1000.0
        np.testing.assert_allclose(limit[0], expected)
        # ... but the MW-denominated meta is not.
        self.assertEqual(loaded.meta["import_limit_mw_min"], min(self.PROFILE))

    def test_a_profile_without_a_base_cap_is_refused(self):
        with self.assertRaises(ValueError) as ctx:
            self.system(import_limit_profile_mw=self.PROFILE)
        self.assertIn("silently ignored", str(ctx.exception))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
