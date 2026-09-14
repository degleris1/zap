"""The minimal linearised unit-commitment device.

Spec: ``memory/plans/2026-09-14-uc-minimal-impl-spec.md`` (design rationale in
``memory/plans/2026-09-14-uc-in-zap-feasibility.md``).  Four things are asserted
here, in the spec's own lettering:

* **(a) off is off** -- with the commitment fields ``None`` every new code path
  delegates to :class:`~zap.devices.injector.AbstractInjector`, and the solved
  48 h ``ca2040_z4`` objective is *bit-identical* to the value the device
  produced before the feature existed (measured at zap ``b7a3386``).
* **(c) F3** -- with ``start_up_cost_per_mw`` and ``min_power_fraction`` both
  zero, the committed model reproduces the plain LP objective.
* **(d) boundaries** -- ``cyclic_free`` vs ``pypsa`` on a 2 x 24 h tiling.
* **(e) KKT vs finite differences** on a ground-free toy with one *cycling* and
  one *flat* committable row.  The fixture is modelled on
  ``test_accreditation_vjp.py``'s 3-bus toy (the ch3 4-bus fixture cannot run
  the KKT adjoint: its ``Ground`` device trips ``DispatchOutcome.shape``), but
  it is a separate 24 h system rather than a mutation of that one, which is 6 h
  and shared by a dozen assertions about the second adjoint.

Test (b), the acceptance comparison against PyPSA's own linearised UC, needs the
ch3 harness and lives in the brain repo as ``tests/test_unit_commitment_pypsa.py``.
"""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

cp = pytest.importorskip("cvxpy")
torch = pytest.importorskip("torch")

from zap.devices import Generator, Load, StorageUnit
from zap.devices.injector import (
    AbstractInjector,
    GeneratorCommitVariable,
    previous_hour_matrix,
)
from zap.devices.transporter import DirectedLine
from zap.importers.wy_store import (
    HourWindow,
    LoadOptions,
    commitment_fields,
    load_system,
)
from zap.layer import DispatchLayer
from zap.network import PowerNetwork
from zap.planning import DispatchCostObjective, InvestmentObjective
from zap.planning.problem_cvx import PlanningProblemCVX
from zap.tests.fixtures.tiny_dataset import real_z4_dir

torch.set_default_dtype(torch.float64)

#: The 48 h venue every ch3 experiment smokes on: window [7, 55) of WY2020.
WINDOW = HourWindow(start=7, stop=55)

#: The solved objective of that window with ``export_mode="drop"``,
#: ``storage_soc_mode="cyclic_free"`` and HiGHS, measured on zap ``b7a3386``
#: -- i.e. *before* the commitment device existed.  ``float.hex`` is not needed:
#: the assertion below is an exact ``==`` on these decimal literals, which
#: round-trip through ``repr``.
PRE_CHANGE_OBJECTIVE = {1.0: 57018188.5456866, 1.6: 3588769322.2414174}

requires_z4 = unittest.skipIf(
    real_z4_dir() is None, "data/ca2040_z4/weather.zarr is not present"
)


def z4_system(scale_load: float, **commitment):
    root: Path = real_z4_dir()
    options = LoadOptions(
        years=(2020,),
        window=WINDOW,
        export_mode="drop",
        storage_soc_mode="cyclic_free",
        demand_scaling="fixed",
        scale_load=scale_load,
        **commitment,
    )
    return load_system(root, options)


def solve(system, time_horizon=48, devices=None):
    devices = system.devices if devices is None else devices
    return system.network.dispatch(
        devices, time_horizon=time_horizon, solver=cp.HIGHS, add_ground=True
    )


# ---------------------------------------------------------------------------
# Sprev: the boundary lives in the first column and nowhere else
# ---------------------------------------------------------------------------


class PreviousHourMatrixTests(unittest.TestCase):
    def test_shift_direction(self):
        """``(c @ Sprev)[:, t] == c[:, t - 1]`` for every interior hour."""
        T = 5
        c = np.arange(2 * T, dtype=float).reshape(2, T)
        for mode in ("cyclic_free", "pypsa"):
            shifted = c @ previous_hour_matrix(T, mode).toarray()
            np.testing.assert_allclose(shifted[:, 1:], c[:, :-1])

    def test_first_column_is_the_whole_boundary(self):
        cyclic = previous_hour_matrix(4, "cyclic_free").toarray()
        pypsa = previous_hour_matrix(4, "pypsa").toarray()
        np.testing.assert_allclose(cyclic[:, 1:], pypsa[:, 1:])
        # cyclic_free: c_{-1} = c_{T-1}.  Not c_{T-1} = c_0 -- the transition is
        # one-sided, so the direction of the wrap matters (verifier item 3).
        self.assertEqual(cyclic[3, 0], 1.0)
        self.assertEqual(cyclic[:, 0].sum(), 1.0)
        self.assertEqual(pypsa[:, 0].sum(), 0.0)

    def test_torch_backend_is_dense_and_equal(self):
        like = torch.zeros(2, 4, dtype=torch.float64)
        dense = previous_hour_matrix(4, "cyclic_free", la=torch, like=like)
        self.assertIsInstance(dense, torch.Tensor)
        np.testing.assert_allclose(
            dense.numpy(), previous_hour_matrix(4, "cyclic_free").toarray()
        )

    def test_bad_mode_is_refused(self):
        with self.assertRaises(ValueError):
            previous_hour_matrix(4, "free")


# ---------------------------------------------------------------------------
# (a) off is off
# ---------------------------------------------------------------------------


class OffIsOffTests(unittest.TestCase):
    """With the fields ``None``, every new path is ``super()``'s."""

    def device(self, **kwargs):
        T = 4
        return Generator(
            num_nodes=1,
            name=np.array(["a", "b"], dtype=object),
            terminal=np.array([0, 0]),
            nominal_capacity=np.array([[100.0], [50.0]]),
            dynamic_capacity=np.linspace(0.5, 1.0, 2 * T).reshape(2, T),
            linear_cost=np.array([[10.0], [30.0]]) * np.ones((1, T)),
            emission_rates=np.zeros((2, 1)),
            capital_cost=np.ones((2, 1)),
            **kwargs,
        )

    def test_no_local_variables(self):
        device = self.device()
        self.assertFalse(device.is_committable)
        self.assertIsNone(device.model_local_variables(4))

    def test_all_false_committable_is_also_off(self):
        device = self.device(committable=np.array([False, False]))
        self.assertFalse(device.is_committable)
        self.assertIsNone(device.model_local_variables(4))

    def test_inequality_constraints_are_the_injectors(self):
        device = self.device()
        power = [np.linspace(0.0, 40.0, 8).reshape(2, 4)]
        ours = device.inequality_constraints(power, None, None)
        theirs = AbstractInjector.inequality_constraints(device, power, None, None)
        self.assertEqual(len(ours), 2)
        for a, b in zip(ours, theirs):
            self.assertTrue(np.array_equal(a, b))

    def test_operation_cost_is_the_injectors(self):
        device = self.device()
        power = [np.linspace(0.0, 40.0, 8).reshape(2, 4)]
        self.assertEqual(
            device.operation_cost(power, None, None),
            AbstractInjector.operation_cost(device, power, None, None),
        )

    def test_inequality_matrices_are_the_injectors(self):
        device = self.device()
        power = [np.zeros((2, 4))]
        ours = device.inequality_matrices(
            device.inequality_constraints(power, None, None), power, None, None
        )
        theirs = AbstractInjector._inequality_matrices(
            device,
            device.get_empty_constraint_matrix(
                device.inequality_constraints(power, None, None), power, None, None
            ),
        )
        self.assertEqual(len(ours), len(theirs))
        for a, b in zip(ours, theirs):
            np.testing.assert_allclose(a.power[0].toarray(), b.power[0].toarray())

    def test_admm_prox_delegates(self):
        device = self.device().torchify(machine="cpu")
        device.has_changed = True  # normally set by the ADMM solver's setup
        power = [torch.zeros(2, 4, dtype=torch.float64)]
        out = device.admm_prox_update(0.5, None, power, None)
        self.assertEqual(out[0][0].shape, (2, 4))

    @requires_z4
    def test_the_48h_objective_is_bit_identical(self):
        """The whole point: ``commitment="off"`` is not a re-derivation."""
        for scale, expected in PRE_CHANGE_OBJECTIVE.items():
            with self.subTest(scale=scale):
                system = z4_system(scale, commitment="off")
                outcome = solve(system)
                self.assertEqual(outcome.problem.value, expected)


# ---------------------------------------------------------------------------
# The device itself, on a one-bus toy
# ---------------------------------------------------------------------------


def toy_commitment_device(
    *, committable=(False, True), rho=(0.0, 0.4), k=(0.0, 5.0), mode="cyclic_free", T=6
):
    return Generator(
        num_nodes=1,
        name=np.array(["cheap", "peaker"], dtype=object),
        terminal=np.array([0, 0]),
        nominal_capacity=np.array([[100.0], [50.0]]),
        dynamic_capacity=np.ones((2, T)),
        linear_cost=np.array([[10.0], [30.0]]) * np.ones((1, T)),
        emission_rates=np.zeros((2, 1)),
        capital_cost=np.ones((2, 1)),
        committable=np.asarray(committable),
        min_power_fraction=np.asarray(rho, dtype=float),
        start_up_cost_per_mw=np.asarray(k, dtype=float),
        commitment_mode=mode,
    )


def toy_system(demand, **kwargs):
    T = len(demand)
    generator = toy_commitment_device(T=T, **kwargs)
    load = Load(
        num_nodes=1,
        name=np.array(["l"], dtype=object),
        terminal=np.array([0]),
        load=np.asarray(demand, dtype=float).reshape(1, T),
        linear_cost=1e4 * np.ones((1, 1)),
    )
    return PowerNetwork(1), [generator, load]


class DeviceShapeTests(unittest.TestCase):
    def test_six_families_of_the_right_shape(self):
        T = 6
        device = toy_commitment_device(T=T)
        state = GeneratorCommitVariable(np.zeros((1, T)), np.zeros((1, T)))
        power = [np.zeros((2, T))]
        families = device.inequality_constraints(power, None, state)
        self.assertEqual([np.shape(f) for f in families], [(2, T)] * 2 + [(1, T)] * 4)
        self.assertEqual(device.equality_constraints(power, None, state), [])

    def test_non_committable_rows_keep_todays_bounds(self):
        """Families 0 and 1 *replace* the injector's; they must agree off-commitment."""
        T = 6
        device = toy_commitment_device(T=T)
        power = [np.linspace(0.0, 40.0, 2 * T).reshape(2, T)]
        state = GeneratorCommitVariable(np.zeros((1, T)), np.zeros((1, T)))
        ours = device.inequality_constraints(power, None, state)
        theirs = AbstractInjector.inequality_constraints(device, power, None, None)
        np.testing.assert_allclose(ours[0][0], theirs[0][0])
        np.testing.assert_allclose(ours[1][0], theirs[1][0])

    def test_min_power_must_stay_zero(self):
        T = 6
        device = toy_commitment_device(T=T)
        state = GeneratorCommitVariable(np.zeros((1, T)), np.zeros((1, T)))
        with self.assertRaises(ValueError):
            device.inequality_constraints(
                [np.zeros((2, T))], None, state, min_power=np.full((2, 1), 0.1)
            )

    def test_state_is_rewrapped_from_a_plain_list(self):
        T = 6
        device = toy_commitment_device(T=T)
        state = [np.zeros((1, T)), np.zeros((1, T))]
        families = device.inequality_constraints([np.zeros((2, T))], None, state)
        self.assertEqual(len(families), 6)

    def test_admm_prox_is_refused(self):
        device = toy_commitment_device()
        with self.assertRaises(NotImplementedError) as ctx:
            device.admm_prox_update(0.5, None, [torch.zeros(2, 6)], None)
        self.assertIn("no closed-form ADMM prox", str(ctx.exception))

    def test_sample_time_refuses_a_non_contiguous_block(self):
        device = toy_commitment_device(T=8)
        device.sample_time(np.arange(2, 6), 8)  # contiguous: fine
        with self.assertRaises(ValueError) as ctx:
            device.sample_time(np.array([0, 1, 4, 5]), 8)
        self.assertIn("contiguous", str(ctx.exception))

    def test_sample_time_carries_the_static_commitment_index(self):
        device = toy_commitment_device(T=8, mode="pypsa")
        sliced = device.sample_time(np.arange(2, 6), 8)
        self.assertEqual(sliced.commitment_mode, "pypsa")
        np.testing.assert_array_equal(sliced.commit_rows, device.commit_rows)
        np.testing.assert_allclose(sliced.commit_scatter, device.commit_scatter)
        np.testing.assert_allclose(
            sliced.commit_start_up_cost, device.commit_start_up_cost
        )
        self.assertEqual(sliced.dynamic_capacity.shape[1], 4)

    def test_commitment_mode_survives_torchify(self):
        device = toy_commitment_device(mode="pypsa").torchify(machine="cpu")
        self.assertEqual(device.commitment_mode, "pypsa")
        self.assertTrue(device.is_committable)
        self.assertIsInstance(device.commit_scatter, torch.Tensor)

    def test_bad_commitment_mode_is_refused(self):
        with self.assertRaises(ValueError):
            toy_commitment_device(mode="free")

    def test_a_derate_below_the_minimum_stable_level_forbids_commitment(self):
        """``rho > a`` forces ``c = 0``: the unit is un-committable that hour."""
        T = 4
        network, devices = toy_system([60.0, 60.0, 60.0, 60.0])
        generator = devices[0]
        # The peaker is needed in every hour, but hour 2 derates it to 10 % while
        # its minimum stable level is 40 % -- families 0 and 1 then force c = 0.
        generator.dynamic_capacity = np.array([[0.5] * T, [1.0, 1.0, 0.1, 1.0]])
        devices[1].load = np.full((1, T), 60.0)
        outcome = network.dispatch(devices, time_horizon=T, solver=cp.HIGHS, add_ground=False)
        committed = np.asarray(outcome.local_variables[0][0])
        self.assertAlmostEqual(float(committed[0, 2]), 0.0, places=6)
        self.assertGreater(float(committed[0, 1]), 0.0)


class InequalityMatrixTests(unittest.TestCase):
    """``_inequality_matrices`` is the numerical Jacobian of the six families.

    The one thing in the device that cannot be read off the constraint code:
    the row-major ``kron`` identity ``vec_r(c @ Sprev) = kron(I_{N_c}, Sprev.T)
    vec_r(c)`` and the sign of every block (verifier items 1 and 3).
    """

    T = 5
    N = 3

    def device(self, mode="cyclic_free"):
        rng = np.random.default_rng(0)
        return Generator(
            num_nodes=1,
            name=np.array(["a", "b", "c"], dtype=object),
            terminal=np.zeros(3, dtype=int),
            nominal_capacity=np.array([[100.0], [50.0], [70.0]]),
            dynamic_capacity=rng.uniform(0.3, 1.0, (self.N, self.T)),
            linear_cost=np.array([[10.0], [30.0], [20.0]]) * np.ones((1, self.T)),
            emission_rates=np.zeros((self.N, 1)),
            capital_cost=np.ones((self.N, 1)),
            committable=np.array([False, True, True]),
            min_power_fraction=np.array([0.0, 0.4, 0.25]),
            start_up_cost_per_mw=np.array([0.0, 5.0, 3.0]),
            commitment_mode=mode,
        )

    def numerical_jacobian(self, device, power, state, key, index):
        """``d(families)/d(variable)`` by central differences, per family."""
        arrays = {"power": power[0], "committed": state[0], "start_up": state[1]}
        target = arrays[key]
        size = target.size
        columns = []
        for j in range(size):
            plus = {k: np.array(v, copy=True) for k, v in arrays.items()}
            minus = {k: np.array(v, copy=True) for k, v in arrays.items()}
            plus[key].ravel()[j] += 1e-5
            minus[key].ravel()[j] -= 1e-5
            f_plus = device.inequality_constraints(
                [plus["power"]], None, GeneratorCommitVariable(plus["committed"], plus["start_up"])
            )
            f_minus = device.inequality_constraints(
                [minus["power"]],
                None,
                GeneratorCommitVariable(minus["committed"], minus["start_up"]),
            )
            columns.append(
                [(np.asarray(a) - np.asarray(b)).ravel() / 2e-5 for a, b in zip(f_plus, f_minus)]
            )
        return [np.column_stack([col[f] for col in columns]) for f in range(index)]

    def check(self, mode):
        rng = np.random.default_rng(1)
        device = self.device(mode)
        n_c = int(device.commit_rows.size)
        power = [rng.uniform(0.0, 20.0, (self.N, self.T))]
        state = GeneratorCommitVariable(
            rng.uniform(0.0, 20.0, (n_c, self.T)), rng.uniform(0.0, 5.0, (n_c, self.T))
        )
        families = device.inequality_constraints(power, None, state)
        matrices = device.inequality_matrices(families, power, None, state)

        for key, slot, attribute in (
            ("power", 0, "power"),
            ("committed", 0, "local_variables"),
            ("start_up", 1, "local_variables"),
        ):
            numeric = self.numerical_jacobian(device, power, state, key, len(families))
            for f, expected in enumerate(numeric):
                analytic = np.asarray(
                    getattr(matrices[f], attribute)[slot].todense()
                    if hasattr(getattr(matrices[f], attribute)[slot], "todense")
                    else getattr(matrices[f], attribute)[slot]
                )
                with self.subTest(mode=mode, variable=key, family=f):
                    np.testing.assert_allclose(analytic, expected, atol=1e-6)

    def test_cyclic_free(self):
        self.check("cyclic_free")

    def test_pypsa(self):
        self.check("pypsa")


class DeviceSolveTests(unittest.TestCase):
    def test_start_up_cost_is_charged_once(self):
        demand = [60.0, 60.0, 120.0, 120.0, 60.0, 60.0]
        network, devices = toy_system(demand)
        outcome = network.dispatch(devices, time_horizon=6, solver=cp.HIGHS, add_ground=False)
        committed = np.asarray(outcome.local_variables[0][0])
        start_up = np.asarray(outcome.local_variables[0][1])
        # 20 MW of the peaker in the two peak hours: 10*440 fuel + 30*40 peaker
        # + 5 $/MW * 20 MW started.
        self.assertAlmostEqual(float(outcome.problem.value), 5700.0, places=6)
        np.testing.assert_allclose(committed[0], [0, 0, 20, 20, 0, 0], atol=1e-6)
        np.testing.assert_allclose(start_up[0], [0, 0, 20, 0, 0, 0], atol=1e-6)

    def test_minimum_stable_level_binds(self):
        """``rho * c <= p``: a committed unit cannot be turned down below rho."""
        demand = [60.0, 60.0, 120.0, 61.0, 60.0, 60.0]
        network, devices = toy_system(demand, rho=(0.0, 1.0), k=(0.0, 0.0))
        outcome = network.dispatch(devices, time_horizon=6, solver=cp.HIGHS, add_ground=False)
        power = np.asarray(outcome.power[0][0])
        committed = np.asarray(outcome.local_variables[0][0])
        # rho == 1 pins p == c exactly whenever the unit is on.
        np.testing.assert_allclose(power[1], committed[0], atol=1e-6)

    def test_f3_regression_zero_start_up_and_zero_min_power(self):
        """(c) With ``k = 0`` and ``rho = 0`` the committed LP is the plain LP."""
        demand = [60.0, 60.0, 120.0, 120.0, 60.0, 60.0]
        network, devices = toy_system(demand, rho=(0.0, 0.0), k=(0.0, 0.0))
        committed = network.dispatch(
            devices, time_horizon=6, solver=cp.HIGHS, add_ground=False
        )
        _, plain_devices = toy_system(demand)
        plain_devices[0].committable = None
        plain_devices[0].rebuild_commitment()
        plain = network.dispatch(
            plain_devices, time_horizon=6, solver=cp.HIGHS, add_ground=False
        )
        self.assertAlmostEqual(
            committed.problem.value / plain.problem.value, 1.0, places=9
        )


@requires_z4
class Z4RegressionTests(unittest.TestCase):
    """(c) on the real 48 h slice: the F3 regression at fleet scale."""

    def test_zero_cost_zero_min_power_reproduces_the_plain_lp(self):
        plain = z4_system(1.0, commitment="off")
        plain_value = solve(plain).problem.value

        system = z4_system(1.0, commitment="minimal")
        generator = system.devices[0]
        generator.min_power_fraction = np.zeros_like(generator.min_power_fraction)
        generator.start_up_cost_per_mw = np.zeros_like(generator.start_up_cost_per_mw)
        generator.rebuild_commitment()
        value = solve(system).problem.value

        self.assertAlmostEqual(value / plain_value, 1.0, places=9)

    def test_commitment_is_more_expensive_than_the_relaxation(self):
        plain = solve(z4_system(1.0, commitment="off")).problem.value
        committed = solve(
            z4_system(1.0, commitment="minimal", commitment_mode="cyclic_free")
        ).problem.value
        self.assertGreater(committed, plain)


# ---------------------------------------------------------------------------
# The importer: `k_g` in $/MW started, off the file's nameplate
# ---------------------------------------------------------------------------


class CommitmentFieldTests(unittest.TestCase):
    """:func:`zap.importers.wy_store.commitment_fields` (spec section 3)."""

    def frame(self, **overrides):
        data = {
            "committable": [True, True, False, False],
            "p_min_pu": [0.4, 0.25, 0.3, 0.0],
            "start_up_cost": [1000.0, 0.0, 500.0, 0.0],
            "p_nom_extendable": [False, True, False, False],
        }
        data.update(overrides)
        return pd.DataFrame(data, index=pd.Index(["a", "b", "c", "d"], name="name"))

    def test_dollars_per_mw_from_the_file_nameplate(self):
        fields = commitment_fields(self.frame(), np.array([100.0, 50.0, 200.0, 0.0]))
        np.testing.assert_allclose(fields["start_up_cost_per_mw"], [10.0, 0.0, 0.0, 0.0])
        # `p_min_pu` on a non-committable row is ignored (row "c").
        np.testing.assert_allclose(fields["min_power_fraction"], [0.4, 0.25, 0.0, 0.0])
        np.testing.assert_array_equal(fields["committable"], [True, True, False, False])

    def test_zero_file_capacity_gives_zero_cost(self):
        fields = commitment_fields(
            self.frame(start_up_cost=[0.0, 0.0, 0.0, 0.0]),
            np.array([0.0, 50.0, 200.0, 0.0]),
        )
        self.assertEqual(float(fields["start_up_cost_per_mw"][0]), 0.0)

    def test_a_buildable_row_with_no_nameplate_is_refused(self):
        frame = self.frame(start_up_cost=[1000.0, 7.0, 500.0, 0.0])
        with self.assertRaises(ValueError) as ctx:
            commitment_fields(frame, np.array([100.0, 0.0, 200.0, 0.0]))
        self.assertIn("'b'", str(ctx.exception))


@requires_z4
class Z4ImporterTests(unittest.TestCase):
    def setUp(self):
        self.system = z4_system(1.0, commitment="minimal")
        self.generator = self.system.devices[0]
        self.static = pd.read_csv(
            real_z4_dir() / "static" / "generators.csv", index_col=0
        )

    def test_the_committable_rows_are_the_files(self):
        expected = np.asarray(self.static["committable"], dtype=bool)
        np.testing.assert_array_equal(
            np.asarray(self.generator.committable, dtype=bool), expected
        )
        self.assertEqual(int(expected.sum()), 38)

    def test_k_is_dollars_per_mw_of_the_file_nameplate(self):
        committable = np.asarray(self.static["committable"], dtype=bool)
        expected = (
            self.static.loc[committable, "start_up_cost"].to_numpy(float)
            / self.static.loc[committable, "p_nom"].to_numpy(float)
        )
        np.testing.assert_allclose(
            np.asarray(self.generator.commit_start_up_cost, dtype=float).reshape(-1),
            expected,
        )

    def test_a_retired_committable_row_keeps_its_cost(self):
        """``k_g`` is off the *file*, so the lifetime rule cannot move it.

        Such a row is harmless: ``P_bar = 0 => c = 0 => su = 0``.
        """
        capacities = np.asarray(self.generator.nominal_capacity, dtype=float).reshape(-1)
        committable = np.asarray(self.generator.committable, dtype=bool)
        retired = committable & (capacities == 0.0)
        self.assertGreater(int(retired.sum()), 0)
        rows = np.asarray(self.generator.commit_rows)
        k = np.asarray(self.generator.commit_start_up_cost, dtype=float).reshape(-1)
        self.assertTrue(bool((k[np.isin(rows, np.flatnonzero(retired))] > 0).any()))

    def test_min_power_fraction_is_zero_off_the_committable_rows(self):
        committable = np.asarray(self.generator.committable, dtype=bool)
        rho = np.asarray(self.generator.min_power_fraction, dtype=float).reshape(-1)
        np.testing.assert_allclose(rho[~committable], 0.0)
        self.assertGreater(float(rho[committable].max()), 0.0)

    def test_min_power_is_still_identically_zero(self):
        np.testing.assert_allclose(np.asarray(self.generator.min_power, dtype=float), 0.0)


# ---------------------------------------------------------------------------
# (d) block boundaries
# ---------------------------------------------------------------------------


@requires_z4
class BoundaryModeTests(unittest.TestCase):
    """2 x 24 h tiling of the same 48 h window, under each boundary."""

    @classmethod
    def setUpClass(cls):
        cls.results = {}
        for mode in ("cyclic_free", "pypsa"):
            system = z4_system(1.0, commitment="minimal", commitment_mode=mode)
            blocks = []
            for start in (0, 24):
                devices = [
                    d.sample_time(np.arange(start, start + 24), 48) for d in system.devices
                ]
                outcome = system.network.dispatch(
                    devices, time_horizon=24, solver=cp.HIGHS, add_ground=True
                )
                k = np.asarray(devices[0].commit_start_up_cost, dtype=float)
                committed = np.asarray(outcome.local_variables[0][0], dtype=float)
                start_up = np.asarray(outcome.local_variables[0][1], dtype=float)
                blocks.append(
                    {
                        "objective": float(outcome.problem.value),
                        "startup_cost_usd": float((k * start_up).sum()),
                        "committed": committed,
                        "start_up": start_up,
                        "k": k.reshape(-1),
                    }
                )
            cls.results[mode] = blocks

    def test_pypsa_charges_one_extra_fleet_start_per_block(self):
        for index in (0, 1):
            with self.subTest(block=index):
                pypsa = self.results["pypsa"][index]
                cyclic = self.results["cyclic_free"][index]
                delta = pypsa["startup_cost_usd"] - cyclic["startup_cost_usd"]
                # Under `pypsa` hour 0 pays for all of `c_0`; under `cyclic_free`
                # the wrap covers `min(c_0, c_{T-1})` of it.  Evaluated on the
                # `pypsa` solve, as the spec's section 6(d) asks.
                wrapped = float(
                    (
                        pypsa["k"]
                        * np.minimum(pypsa["committed"][:, 0], pypsa["committed"][:, -1])
                    ).sum()
                )
                self.assertGreater(delta, 0.0)
                self.assertLessEqual(
                    abs(delta - wrapped), 1e-6 * max(abs(delta), abs(wrapped))
                )

    def test_the_cyclic_wrap_is_actually_used(self):
        for index in (0, 1):
            with self.subTest(block=index):
                block = self.results["cyclic_free"][index]
                c, su = block["committed"], block["start_up"]
                # The transition holds with the wrap ...
                self.assertGreaterEqual(float((su[:, 0] - (c[:, 0] - c[:, -1])).min()), -1e-6)
                # ... and at least one row is committed at hour 0 for free.
                wrapped = (c[:, 0] > 1e-6) & (su[:, 0] < c[:, 0] - 1e-6)
                self.assertTrue(bool(wrapped.any()))


# ---------------------------------------------------------------------------
# (e) KKT vs finite differences
# ---------------------------------------------------------------------------

FD_T = 24
FD_PARAMETERS = {"generator_capacity": (0, "nominal_capacity")}
#: (control, cycling row X, flat row Y, an expensive non-binding backstop).
FD_THETA = {"generator_capacity": np.array([[700.0], [90.0], [250.0], [600.0]])}
FD_ROWS = {"control": 0, "cycling": 1, "flat": 2, "backstop": 3}
#: Index into ``local_variables``, which carries the committable rows only.
FD_COMMIT_ROWS = {"cycling": 0, "flat": 1}

#: A two-peak day.  The swing is wider than ``1 / rho`` for the cycling row, so
#: its committed MW *cannot* stay flat; the system is ten times the size of the
#: accreditation toy so that a 1 MW capacity step stays inside one active set
#: and the finite differences are step-independent.
FD_DEMAND = 10.0 * np.array(
    [55.0, 52.0, 50.0, 50.0, 58.0, 70.0, 86.0, 97.0, 104.0, 95.0, 88.0, 82.0]
    + [80.0, 84.0, 92.0, 100.0, 105.0, 103.0, 96.0, 88.0, 78.0, 68.0, 60.0, 56.0]
)


def fd_devices():
    """The ground-free 3-bus toy of ``test_accreditation_vjp.py``, at 24 h and
    ten times the size, with two committable rows.

    * row **cycling** sits at bus 1 behind a lossy line with ``rho = 0.7``.  It
      is the marginal unit, so its committed MW follows the residual demand and
      moves between hours; its capacity binds only over the second peak.
    * row **flat** sits at bus 2, is the cheapest unit, and commits at its cap in
      every hour -- the degenerate case of spec 6(e): ``su`` sits at its own
      lower bound *and* the start transition is tight in every hour.
    * row **control** is the non-committable mid-merit unit at the load bus, and
      **backstop** an expensive non-committable row whose capacity never binds.

    Every bus carries an injector, so no ``Ground`` is needed -- the ch3 4-bus
    fixture cannot run the KKT adjoint because its ``Ground`` device trips
    ``DispatchOutcome.shape`` (LESSONS 2026-09-13).
    """
    network = PowerNetwork(3)
    ones = np.ones((1, FD_T))
    generators = Generator(
        num_nodes=3,
        name=np.array(["control", "cycling", "flat", "backstop"], dtype=object),
        terminal=np.array([0, 1, 2, 0]),
        nominal_capacity=np.array(FD_THETA["generator_capacity"], copy=True),
        dynamic_capacity=np.ones((4, FD_T)),
        linear_cost=np.array([[10.0], [25.0], [8.0], [200.0]]) * ones,
        emission_rates=np.zeros((4, 1)),
        capital_cost=np.ones((4, 1)),
        committable=np.array([False, True, True, False]),
        min_power_fraction=np.array([0.0, 0.7, 0.05, 0.0]),
        start_up_cost_per_mw=np.array([0.0, 4.0, 9.0, 0.0]),
        commitment_mode="cyclic_free",
    )
    loads = Load(
        num_nodes=3,
        name=np.array(["load_b0", "load_b2"], dtype=object),
        terminal=np.array([0, 2]),
        load=np.vstack([FD_DEMAND, np.full(FD_T, 50.0)]),
        linear_cost=10_000.0 * np.ones((2, 1)),
    )
    storage = StorageUnit(
        num_nodes=3,
        name=np.array(["batt_b0"], dtype=object),
        terminal=np.array([0]),
        power_capacity=np.array([[80.0]]),
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
        max_power=np.array([[1000.0], [1000.0]]),
        linear_cost=np.zeros((2, 1)),
        efficiency=np.array([[0.90], [0.95]]),
    )
    return network, [generators, loads, storage, lines]


def fd_problem():
    network, devices = fd_devices()
    layer = DispatchLayer(
        network,
        devices,
        FD_PARAMETERS,
        time_horizon=FD_T,
        solver=cp.HIGHS,
        solver_kwargs={},
        add_ground=False,
    )
    lower = {k: np.zeros_like(v) for k, v in FD_THETA.items()}
    upper = {k: np.full_like(v, 1e4) for k, v in FD_THETA.items()}
    problem = PlanningProblemCVX(
        DispatchCostObjective(network, devices),
        InvestmentObjective(devices, layer),
        layer,
        lower,
        upper,
    )
    return problem, devices


def fd_theta(**overrides):
    out = {k: np.array(v, dtype=float, copy=True) for k, v in FD_THETA.items()}
    out.update({k: np.asarray(v, dtype=float) for k, v in overrides.items()})
    return out


class SingleLevelLPTests(unittest.TestCase):
    """Spec section 5: the monolithic LP needs no change to carry commitment.

    ``setup_inner_problem`` builds every device constraint through
    ``net.model_dispatch_problem``, i.e. through the same
    ``model_local_variables`` / ``inequality_constraints`` / ``operation_cost``
    with ``nominal_capacity`` bound to the outer ``cp.Variable``.  Families 0--5
    are affine in ``(p, c, su, P_bar)``, so the joint problem stays an LP -- and
    the LP optimum must equal the forward pass at the LP's own parameters.
    """

    def test_the_lp_optimum_equals_the_forward_pass(self):
        from zap.planning import MonolithicPlanningProblem

        problem, _devices = fd_problem()
        problem.upper_bounds = {
            k: np.full_like(v, 2000.0) for k, v in problem.upper_bounds.items()
        }
        monolithic = MonolithicPlanningProblem(problem, solver=cp.HIGHS, solver_kwargs={})
        theta, parts = monolithic.solve()
        forward = float(
            problem.forward(
                requires_grad=False, **{k: np.asarray(v) for k, v in theta.items()}
            )
        )
        self.assertAlmostEqual(parts["problem"].value / forward, 1.0, places=9)
        self.assertGreater(float(np.asarray(theta["generator_capacity"]).sum()), 0.0)


class KKTFiniteDifferenceTests(unittest.TestCase):
    """(e) ``PlanningProblemCVX.backward()`` against one-sided FD of a re-solve."""

    @classmethod
    def setUpClass(cls):
        problem, _devices = fd_problem()
        cls.base_value = float(problem.forward(requires_grad=True, **fd_theta()))
        cls.dtheta = np.asarray(
            problem.backward()["generator_capacity"], dtype=float
        ).reshape(-1)
        outcome = problem.state
        cls.committed = np.asarray(outcome.local_variables[0][0], dtype=float)
        cls.start_up = np.asarray(outcome.local_variables[0][1], dtype=float)
        cls.shortfall = float(
            np.maximum(
                np.asarray(outcome.power[1][0], dtype=float)
                + np.asarray(_devices[1].load, dtype=float),
                0.0,
            ).sum()
        )
        cls.fd = {}
        for delta in (0.1, 1.0):
            values = []
            for index in range(len(FD_ROWS)):
                bumped = fd_theta()
                bumped["generator_capacity"][index, 0] += delta
                other, _ = fd_problem()
                values.append(
                    (float(other.forward(requires_grad=False, **bumped)) - cls.base_value)
                    / delta
                )
            cls.fd[delta] = np.asarray(values)

    def assert_matches_fd(self, name: str, tolerance: float = 1e-6):
        index = FD_ROWS[name]
        for delta, fd in self.fd.items():
            with self.subTest(row=name, delta=delta):
                self.assertLessEqual(
                    abs(self.dtheta[index] - fd[index]),
                    tolerance * max(1.0, abs(fd[index])),
                    f"{name}: adjoint {self.dtheta[index]!r} vs FD {fd[index]!r} "
                    f"at delta={delta}",
                )
        with self.subTest(row=name, check="step independence"):
            self.assertLessEqual(
                abs(self.fd[0.1][index] - self.fd[1.0][index]),
                tolerance * max(1.0, abs(self.fd[1.0][index])),
            )

    def test_the_fixture_is_what_it_claims(self):
        """One row cycles, one is flat at its cap, and nothing sheds."""
        cycling = self.committed[FD_COMMIT_ROWS["cycling"]]
        flat = self.committed[FD_COMMIT_ROWS["flat"]]
        self.assertAlmostEqual(self.shortfall, 0.0, places=6)
        # Genuinely cycling: it starts more than once and is off in some hours.
        self.assertGreater(float(cycling.max() - cycling.min()), 1.0)
        self.assertGreater(
            int(np.count_nonzero(self.start_up[FD_COMMIT_ROWS["cycling"]] > 1e-6)), 1
        )
        # ... and its capacity binds somewhere, so its gradient is informative.
        self.assertAlmostEqual(
            float(cycling.max()), float(FD_THETA["generator_capacity"][1, 0]), places=6
        )
        # Flat at its cap in every hour, and never starting: the degenerate case
        # (`su` at its own bound *and* the transition constraint tight).
        np.testing.assert_allclose(flat, flat[0], atol=1e-6)
        self.assertAlmostEqual(
            float(flat[0]), float(FD_THETA["generator_capacity"][2, 0]), places=6
        )
        np.testing.assert_allclose(self.start_up[FD_COMMIT_ROWS["flat"]], 0.0, atol=1e-6)

    def test_control_row_matches_finite_differences(self):
        """The non-committable control: today's protocol, today's tolerance."""
        self.assert_matches_fd("control")
        self.assertLess(self.dtheta[FD_ROWS["control"]], 0.0)

    def test_cycling_committable_row_matches_finite_differences(self):
        """Row X of spec 6(e)."""
        self.assert_matches_fd("cycling")
        self.assertLess(self.dtheta[FD_ROWS["cycling"]], 0.0)

    def test_flat_committable_row_is_within_five_percent_of_its_fd(self):
        """Row Y of spec 6(e), the degenerate one.

        The spec allows 5 % here and asks for the measurement to be printed and
        reported either way.  Measured 2026-09-14: it matches to ~1e-12, i.e. as
        exactly as the control row.
        """
        index = FD_ROWS["flat"]
        grad = self.dtheta[index]
        print(
            "\n[spec 6(e)] flat committable row: adjoint = "
            f"{grad!r}; FD = "
            + ", ".join(f"delta={d}: {self.fd[d][index]!r}" for d in sorted(self.fd))
        )
        self.assertTrue(np.isfinite(grad))
        self.assertLessEqual(grad, 1e-9)
        for delta, fd in self.fd.items():
            with self.subTest(delta=delta):
                self.assertLessEqual(
                    abs(grad - fd[index]), 0.05 * max(1.0, abs(fd[index]))
                )

    def test_flat_row_also_matches_at_the_control_tolerance(self):
        """The stronger statement, recorded because it currently holds."""
        self.assert_matches_fd("flat")


class UnitScalingTests(unittest.TestCase):
    """`scale_power` / `scale_costs` leave the start-up term in the fuel term's units."""

    DEMAND = [60.0, 60.0, 120.0, 120.0, 60.0, 60.0]

    def _objective(self, power_unit=1.0, cost_unit=1.0):
        network, devices = toy_system(self.DEMAND)
        for device in devices:
            device.scale_power(power_unit)
            device.scale_costs(cost_unit)
        outcome = network.dispatch(devices, time_horizon=6, solver=cp.HIGHS, add_ground=False)
        return outcome.problem.value * power_unit * cost_unit

    def test_objective_is_invariant_to_power_and_cost_units(self):
        reference = self._objective()
        for power_unit, cost_unit in ((10.0, 1.0), (1.0, 100.0), (10.0, 100.0)):
            with self.subTest(power_unit=power_unit, cost_unit=cost_unit):
                self.assertAlmostEqual(
                    self._objective(power_unit, cost_unit) / reference, 1.0, places=9
                )
