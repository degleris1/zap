"""rho is a MW-to-dollar conversion factor, not a free knob.

Benchmark review section 7.3: the penalty term ``(rho / 2) ||A p||^2`` is dollars
built out of a MW imbalance, and the multiplier it is conjugate to is a $/MWh
price, so ``rho`` has units of ``$/(MWh * MW)`` and its natural value is the norm
ratio ``||price|| / ||power||``.  Two consequences are pinned here:

* one linear-cost prox step moves dispatch by at most ``c / rho`` MW, so a rho far
  above the norm ratio needs ``O(range * rho / c)`` iterations to cross a device's
  feasible range;
* ``rho`` must be rescaled with the importer's units,
  ``rho_scaled = rho_physical * power_unit / cost_unit``.

Also covers the split of the shared absolute tolerance into ``atol_primal`` (MW)
and ``atol_dual`` ($/MWh) (section 7.5, R3).
"""

import logging
import unittest

import cvxpy as cp
import numpy as np
import torch

import zap
from zap.admm import ADMMSolver
from zap.admm.basic_solver import natural_rho
from zap.tests.test_storage_admm_prox import small_system

torch.set_default_dtype(torch.float64)

#: Marginal cost of the price-setting generator, $/MWh.
GEN_COST = 34.0
#: Its capacity, MW.
GEN_CAPACITY = 5000.0


def two_bus_system(T=12):
    """A cheap 5,000 MW / 34 $/MWh unit at bus 0 serving load at bus 1.

    Sized so the cheap unit is always marginal and never binds: every nodal price
    is 34 $/MWh, which makes the norm ratio easy to reason about by hand.
    """
    net = zap.PowerNetwork(2)
    load_profile = np.array(
        [
            2000.0,
            2200.0,
            2400.0,
            2600.0,
            3000.0,
            3400.0,
            3600.0,
            3400.0,
            3000.0,
            2600.0,
            2200.0,
            2000.0,
        ]
    )[:T]

    generators = zap.Generator(
        num_nodes=2,
        name=np.array(["base", "peaker"]),
        terminal=np.array([0, 1]),
        nominal_capacity=np.array([GEN_CAPACITY, 1000.0]),
        dynamic_capacity=np.ones((2, T)),
        linear_cost=np.array([GEN_COST, 500.0]),
        emission_rates=np.array([0.4, 0.6]),
    )
    load = zap.Load(
        num_nodes=2,
        name=np.array(["l0"]),
        terminal=np.array([1]),
        load=load_profile.reshape(1, T),
        linear_cost=np.array([10000.0]),
    )
    line = zap.DCLine(
        num_nodes=2,
        name=np.array(["ln0"]),
        source_terminal=np.array([0]),
        sink_terminal=np.array([1]),
        capacity=np.array([6000.0]),
        linear_cost=np.array([0.0]),
    )
    return net, [generators, load, line], T


def single_generator(T=12):
    """Just the marginal unit, so its prox step is not clipped by another unit's box."""
    return zap.Generator(
        num_nodes=1,
        name=np.array(["base"]),
        terminal=np.array([0]),
        nominal_capacity=np.array([GEN_CAPACITY]),
        dynamic_capacity=np.ones((1, T)),
        linear_cost=np.array([GEN_COST]),
    )


def solve_admm(net, devices, T, rho, num_iterations, **kwargs):
    torch_devices = [d.torchify(machine="cpu", dtype=torch.float64) for d in devices]
    settings = {
        "machine": "cpu",
        "dtype": torch.float64,
        "num_iterations": num_iterations,
        "rho_power": rho,
        "adaptive_rho": False,
        "battery_inner_iterations": 200,
        "battery_inner_over_relaxation": 1.8,
        "atol": 1e-8,
        "rtol": 1e-8,
        "verbose": 0,
    }
    settings.update(kwargs)
    solver = ADMMSolver(**settings)
    state, history = solver.solve(net, torch_devices, T)
    return solver, state, history


class TestProxStepSize(unittest.TestCase):
    """`p = clip(z - c / rho)`: rho sets the largest step a device can take."""

    def test_step_is_exactly_cost_over_rho_in_the_interior(self):
        T = 12
        device = single_generator(T).torchify(machine="cpu", dtype=torch.float64)

        # z at the capacity: the update wants to back off by c / rho and the box
        # does not interfere, so the move is exactly c / rho.
        z = torch.full((1, T), GEN_CAPACITY, dtype=torch.float64)

        device.has_changed = True
        p, _, _ = device.admm_prox_update(1.0, 1.0, [z], None)
        move = float(torch.max(torch.abs(z - p[0])).item())
        # 34 MW out of a 5,000 MW range: 0.7 % per iteration.
        self.assertAlmostEqual(move, GEN_COST / 1.0, places=9)

        device.has_changed = True
        p, _, _ = device.admm_prox_update(0.1, 0.1, [z], None)
        move = float(torch.max(torch.abs(z - p[0])).item())
        self.assertAlmostEqual(move, GEN_COST / 0.1, places=9)

    def test_step_never_exceeds_cost_over_rho(self):
        T = 12
        device = single_generator(T).torchify(machine="cpu", dtype=torch.float64)

        for rho in (1.0, 0.1, 0.01):
            for level in (0.0, 0.25, 0.5, 0.75, 1.0):
                z = torch.full((1, T), level * GEN_CAPACITY, dtype=torch.float64)
                device.has_changed = True
                p, _, _ = device.admm_prox_update(rho, rho, [z], None)
                move = float(torch.max(torch.abs(z - p[0])).item())
                self.assertLessEqual(move, GEN_COST / rho + 1e-9, f"rho={rho}, z={level}")


class TestNaturalRho(unittest.TestCase):
    def test_helper_is_the_norm_ratio(self):
        prices = np.array([[3.0, 4.0]])
        powers = [np.array([[6.0, 8.0]]), None]
        self.assertAlmostEqual(natural_rho(prices, powers), 5.0 / 10.0, places=12)
        self.assertAlmostEqual(natural_rho(torch.tensor(prices), powers), 0.5, places=12)
        self.assertEqual(natural_rho(prices, np.zeros((1, 2))), float("inf"))

    def test_lp_duals_give_an_order_1e_minus_2_rho(self):
        """The z4 measurement in miniature: prices are O(10), powers O(1e3)."""
        net, devices, T = two_bus_system()
        lp = net.dispatch(devices, time_horizon=T, solver=cp.HIGHS, add_ground=False)

        prices = np.asarray(lp.prices)
        np.testing.assert_allclose(prices, GEN_COST, atol=1e-6)

        rho = natural_rho(lp.prices, lp.power)
        self.assertGreater(rho, 1e-3)
        self.assertLess(rho, 1e-1)

    def test_lp_duals_are_scale_covariant(self):
        """`rho_scaled = rho_physical * power_unit / cost_unit`, measured.

        Dividing every cost by `cost_unit` and every power by `power_unit` -- what
        `scale_costs` / `scale_power` do at import -- multiplies the natural rho by
        `power_unit / cost_unit`, which is why `rho_power: 1.0` was right for
        `runner.py` (power_unit 1e3, cost_unit 10) and ~80x too large here.
        """
        power_unit, cost_unit = 1000.0, 10.0

        net, devices, T = two_bus_system()
        lp = net.dispatch(devices, time_horizon=T, solver=cp.HIGHS, add_ground=False)
        physical = natural_rho(lp.prices, lp.power)

        net_s, devices_s, T = two_bus_system()
        for d in devices_s:
            d.scale_costs(cost_unit)
            d.scale_power(power_unit)
        lp_s = net_s.dispatch(devices_s, time_horizon=T, solver=cp.HIGHS, add_ground=False)
        scaled = natural_rho(lp_s.prices, lp_s.power)

        self.assertAlmostEqual(scaled, physical * power_unit / cost_unit, delta=1e-6 * scaled)


class TestRhoDrivesIterationCount(unittest.TestCase):
    """Review 7.5 R5: the natural rho converges, 100x the natural rho does not.

    Uses the heterogeneous fleet of `test_storage_admm_prox.TestADMMvsLP` -- three
    buses, two generators, a load, two lines and two batteries -- because storage
    is what makes the problem rho-sensitive at all.
    """

    ITERATION_CAP = 3000

    @classmethod
    def setUpClass(cls):
        cls.net, cls.devices, cls.T = small_system()
        cls.lp = cls.net.dispatch(
            cls.devices, time_horizon=cls.T, solver=cp.HIGHS, add_ground=False
        )
        cls.rho_natural = natural_rho(cls.lp.prices, cls.lp.power)

    def _gap(self, state):
        return abs(float(state.objective) - self.lp.problem.value) / abs(self.lp.problem.value)

    def test_natural_rho_converges_and_100x_does_not(self):
        solver, state, _ = solve_admm(
            self.net, self.devices, self.T, self.rho_natural, self.ITERATION_CAP
        )
        self.assertTrue(solver.converged, "the natural rho must reach the tolerances")
        self.assertLess(solver.iteration, self.ITERATION_CAP)
        self.assertLess(self._gap(state), 1e-3)
        natural_iterations = solver.iteration

        solver_big, _state_big, _ = solve_admm(
            self.net, self.devices, self.T, 100.0 * self.rho_natural, self.ITERATION_CAP
        )
        self.assertFalse(
            solver_big.converged, "100x the natural rho should not reach the tolerances"
        )
        self.assertLess(natural_iterations, solver_big.iteration)

    def test_the_natural_rho_of_this_fleet_is_not_z4s(self):
        """Guards against reading `rho_power: 0.1` as a universal constant.

        This toy fleet runs on tens of MW at prices of 5-180 $/MWh, so its norm
        ratio is O(1), not the O(1e-2) of `ca2040_z4`; measured, ADMM needs ~3,100
        iterations here at rho = 1.0 and ~14,600 at rho = 0.1.  The transferable
        quantity is `natural_rho`, not the number in the config.
        """
        self.assertGreater(self.rho_natural, 1.0)
        self.assertLess(self.rho_natural, 10.0)


class TestAbsoluteToleranceSplit(unittest.TestCase):
    """Review 7.5 R3: a MW tolerance and a $/MWh tolerance are separate knobs."""

    def _tolerances(self, **kwargs):
        net, devices, T = two_bus_system()
        # rtol = 0 leaves the tolerances purely absolute, so they are exactly
        # `atol * (total_terminals)^(1/p)` and the ratio is readable.
        solver, _, _ = solve_admm(
            net, devices, T, 1.0, 12, rtol=0.0, minimum_iterations=10, **kwargs
        )
        return solver.primal_tol, solver.dual_tol

    def test_atol_sets_both(self):
        primal, dual = self._tolerances(atol=1e-4)
        self.assertAlmostEqual(primal, dual, places=12)

    def test_split_tolerances_are_independent(self):
        primal, dual = self._tolerances(atol=1e-4, atol_primal=1e-3, atol_dual=1e-6)
        self.assertAlmostEqual(dual / primal, 1e-6 / 1e-3, delta=1e-12)

    def test_one_side_falls_back_to_the_alias(self):
        primal, dual = self._tolerances(atol=1e-4, atol_dual=1e-2)
        self.assertAlmostEqual(dual / primal, 1e-2 / 1e-4, delta=1e-9)

        solver = ADMMSolver(atol=1e-4, atol_primal=1e-7, verbose=0)
        self.assertEqual(solver.absolute_tolerances(), (1e-7, 1e-4))


class TestConfigMirrorsAgree(unittest.TestCase):
    """`rho_power` is written down in three places and they must not drift.

    `experiments/ra/configs/base.yaml` is the declared key space,
    `experiments/ra/planning/base.py`'s `PLANNING_DEFAULTS` is the in-code mirror
    the harness validates against, and
    `experiments/ra/configs/methods/plan_admm.yaml` merges over both.  The retune
    to rho = 0.1 (review 7.5 R1) has to reach all three, or an ADMM planning run
    silently solves at the wrong penalty.
    """

    def test_planning_defaults_mirror_base_yaml_rho(self):
        from experiments.ra import config as ra_config
        from experiments.ra.planning import PLANNING_DEFAULTS

        base = ra_config.base_config()
        self.assertEqual(
            PLANNING_DEFAULTS["admm"]["solver_kwargs"]["rho_power"],
            base["planning"]["admm"]["solver_kwargs"]["rho_power"],
            "experiments/ra/planning/base.py:89 still sets rho_power: 1.0; base.yaml "
            "is at 0.1 (rho has units of $/(MWh*MW) -- see ADMMSolver)",
        )

    def test_plan_admm_preset_does_not_reintroduce_rho_1(self):
        """The preset merges over base.yaml, so it has to restate the new value."""
        import pathlib

        import yaml

        from experiments.ra.paths import config_root

        path = pathlib.Path(config_root()) / "methods" / "plan_admm.yaml"
        cfg = yaml.safe_load(path.read_text())
        self.assertEqual(cfg["planning"]["admm"]["solver_kwargs"]["rho_power"], 0.1)


class TestSuggestRho(unittest.TestCase):
    LOGGER = "zap.admm.basic_solver"

    def test_logs_once_when_rho_is_far_from_the_norm_ratio(self):
        net, devices, T = two_bus_system()

        # rho = 1.0 against a norm ratio of ~9e-3: a factor of ~100.
        with self.assertLogs(self.LOGGER, level=logging.INFO) as caught:
            solver, state, _ = solve_admm(net, devices, T, 1.0, 400)
        messages = [r.getMessage() for r in caught.records]
        self.assertEqual(len(messages), 1, messages)
        self.assertIn("suggest_rho", messages[0])

        suggestion = solver.suggest_rho(state)
        self.assertGreater(suggestion, 1e-3)
        self.assertLess(suggestion, 1e-1)

        # One line per solver, not per block: a reused solver stays quiet.
        self.assertTrue(solver._rho_suggested)
        torch_devices = [d.torchify(machine="cpu", dtype=torch.float64) for d in devices]
        with self.assertNoLogs(self.LOGGER, level=logging.INFO):
            solver.solve(net, torch_devices, T)

    def test_says_nothing_when_rho_is_already_natural(self):
        net, devices, T = two_bus_system()
        lp = net.dispatch(devices, time_horizon=T, solver=cp.HIGHS, add_ground=False)
        rho = natural_rho(lp.prices, lp.power)

        with self.assertNoLogs(self.LOGGER, level=logging.INFO):
            solve_admm(net, devices, T, rho, 400)


if __name__ == "__main__":
    unittest.main()
