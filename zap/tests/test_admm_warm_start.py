"""Warm starting ADMM across solves must be safe, refusable, and reportable.

`ADMMLayer` has always carried its ADMM state between forward passes, silently
and by default, and it measurably works (architect spec
`memory/plans/2026-09-09-admm-warm-start-spec.md`, section 8: -40 % iterations
over five planner passes).  What was missing is everything that makes it safe:
a layout fingerprint, rho reconciliation, a cold-start fallback with a reason
instead of a crash, a periodic reset, and a `copy()` that does not downcast
subclasses.  These tests pin that behaviour.

All CPU, float64, tiny horizons.
"""

import os
import tempfile
import unittest
import warnings

import numpy as np
import torch

from zap.admm import ADMMLayer, ADMMSolver
from zap.admm.basic_solver import ADMMLayout, ADMMState
from zap.admm.weighted_solver import ExtendedADMMState
from zap.planning import (
    DispatchCostObjective,
    InvestmentObjective,
    PlanningProblem,
    StochasticPlanningProblem,
)
from zap.tests.test_admm_convergence import two_bus_system

torch.set_default_dtype(torch.float64)


# ====
# Helpers
# ====


def torchify(devices, dtype=torch.float64):
    return [d.torchify(machine="cpu", dtype=dtype) for d in devices]


def make_solver(**kwargs):
    settings = {
        "machine": "cpu",
        "dtype": torch.float64,
        "num_iterations": 20000,
        "rho_power": 1.0,
        "adaptive_rho": False,
        "atol": 1e-8,
        "rtol": 1e-8,
        "minimum_iterations": 10,
        "verbose": 0,
    }
    settings.update(kwargs)
    return ADMMSolver(**settings)


def max_abs(nested):
    """Largest absolute entry of a nested list-of-lists of tensors."""
    return max(float(torch.max(torch.abs(x))) for dev in nested for x in dev)


def max_rel_move(new, old):
    """Largest entrywise change of a nested primal, relative to its own scale."""
    num = max(
        float(torch.max(torch.abs(a - b)))
        for dev_a, dev_b in zip(new, old)
        for a, b in zip(dev_a, dev_b)
    )
    return num / max_abs(old)


def perturbed_capacity(device, fraction=0.01, seed=0):
    """`nominal_capacity * (1 +/- fraction)`, random sign -- the planner's own move."""
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=tuple(device.nominal_capacity.shape))
    return device.nominal_capacity * torch.tensor(1.0 + fraction * signs, dtype=torch.float64)


PARAMETER_NAMES = {"generator_capacity": (0, "nominal_capacity")}


def planning_block(load_scale=1.0, T=8):
    """A two-bus block whose generator capacity is a planning parameter."""
    net, devices, T = two_bus_system(T=T)
    devices[1].load = devices[1].load * load_scale
    # InvestmentObjective returns a float (not a tensor) without a capital cost.
    devices[0].capital_cost = np.array([100.0, 50.0]).reshape(-1, 1)
    return net, torchify(devices), T


def planning_problem(load_scale=1.0, T=8, **layer_kwargs):
    net, devices, T = planning_block(load_scale, T)
    solver_kwargs = layer_kwargs.pop("solver_kwargs", {})
    layer = ADMMLayer(
        net,
        devices,
        PARAMETER_NAMES,
        time_horizon=T,
        solver=make_solver(**solver_kwargs),
        **layer_kwargs,
    )
    bounds_zero = torch.zeros_like(devices[0].nominal_capacity)
    return PlanningProblem(
        DispatchCostObjective(net, devices),
        InvestmentObjective(devices, layer),
        layer,
        lower_bounds={"generator_capacity": bounds_zero},
        upper_bounds={"generator_capacity": 10.0 * devices[0].nominal_capacity},
    )


def two_block_problem(**layer_kwargs):
    subs = [planning_problem(1.0, **layer_kwargs), planning_problem(0.8, **layer_kwargs)]
    return StochasticPlanningProblem(subs), subs


# ====
# (a) The state is a fixed point of the iteration
# ====


class TestWarmStartFixedPoint(unittest.TestCase):
    """Tolerances are tight (1e-10) so that `S*` really is a fixed point."""

    @classmethod
    def setUpClass(cls):
        cls.net, cls.devices, cls.T = two_bus_system(T=12)
        cls.torch_devices = torchify(cls.devices)
        solver = make_solver(atol=1e-10, rtol=1e-10, num_iterations=50000)
        cls.state, cls.history = solver.solve(cls.net, cls.torch_devices, cls.T)
        assert solver.converged

    def test_zero_iterations_returns_initial_state(self):
        solver = make_solver(atol=1e-10, rtol=1e-10)
        state, history = solver.solve(
            self.net, self.torch_devices, self.T, initial_state=self.state.copy(), num_iterations=0
        )

        self.assertEqual(len(history.power), 0)
        self.assertFalse(solver.converged)
        self.assertTrue(solver.warm_started)
        for dev_new, dev_old in zip(state.clone_power, self.state.clone_power):
            for a, b in zip(dev_new, dev_old):
                self.assertTrue(torch.equal(a, b))
        self.assertTrue(torch.equal(state.dual_power, self.state.dual_power))
        self.assertEqual(state.cumulative_iteration, self.state.cumulative_iteration)

    def test_warm_start_is_idempotent_at_the_fixed_point(self):
        solver = make_solver(atol=1e-10, rtol=1e-10)
        state, _ = solver.solve(
            self.net, self.torch_devices, self.T, initial_state=self.state.copy(), num_iterations=1
        )

        self.assertLess(max_rel_move(state.clone_power, self.state.clone_power), 1e-9)
        self.assertLess(max_rel_move(state.power, self.state.power), 1e-9)
        dual_move = float(torch.max(torch.abs(state.dual_power - self.state.dual_power)))
        self.assertLess(dual_move / float(torch.max(torch.abs(self.state.dual_power))), 1e-9)

    def test_warm_start_from_converged_state_converges_immediately(self):
        """Re-solving the *same* problem from `S*` costs only `minimum_iterations`."""
        solver = make_solver(atol=1e-10, rtol=1e-10, minimum_iterations=10)
        _, history = solver.solve(
            self.net, self.torch_devices, self.T, initial_state=self.state.copy()
        )

        self.assertTrue(solver.converged)
        self.assertTrue(solver.warm_started)
        self.assertLessEqual(len(history.power), solver.minimum_iterations + 5)
        self.assertLess(len(history.power), len(self.history.power))


# ====
# (c) The point of the exercise
# ====


class TestWarmStartSpeedup(unittest.TestCase):
    def test_warm_start_beats_cold_after_capacity_perturbation(self):
        net, devices, T = two_bus_system(T=12)
        torch_devices = torchify(devices)

        base_solver = make_solver()
        base_state, _ = base_solver.solve(net, torch_devices, T)
        self.assertTrue(base_solver.converged)

        parameters = [{"nominal_capacity": perturbed_capacity(torch_devices[0])}, {}, {}]

        cold = make_solver()
        cold_state, cold_history = cold.solve(net, torch_devices, T, parameters=parameters)

        warm = make_solver()
        warm_state, warm_history = warm.solve(
            net, torch_devices, T, parameters=parameters, initial_state=base_state.copy()
        )

        self.assertTrue(cold.converged)
        self.assertTrue(warm.converged)
        self.assertTrue(warm.warm_started)
        self.assertFalse(cold.warm_started)
        self.assertLess(len(warm_history.power), len(cold_history.power))
        self.assertAlmostEqual(
            float(warm_state.objective) / float(cold_state.objective), 1.0, delta=1e-4
        )


# ====
# (d) Guards: a state that does not fit is refused, never crashed on
# ====


class TestWarmStartGuards(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.net, cls.devices, cls.T = two_bus_system(T=12)
        cls.torch_devices = torchify(cls.devices)
        solver = make_solver(num_iterations=200, minimum_iterations=10)
        cls.state, _ = solver.solve(cls.net, cls.torch_devices, cls.T)

    def test_layout_change_falls_back_to_cold_start(self):
        net, devices, T = two_bus_system(T=8)
        solver = make_solver(verbose=1)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _, history = solver.solve(net, torchify(devices), T, initial_state=self.state.copy())

        self.assertFalse(solver.warm_started)
        self.assertIn("time_horizon", solver.warm_start_reason)
        self.assertTrue(any("warm start refused" in str(w.message) for w in caught))
        self.assertTrue(solver.converged)
        self.assertGreater(len(history.power), 0)

    def test_device_set_change_falls_back_to_cold_start(self):
        net, devices, T = two_bus_system(T=12)
        fewer = torchify(devices)[:2]  # drop the line
        solver = make_solver(num_iterations=50, verbose=0)

        solver.solve(net, fewer, T, initial_state=self.state.copy())

        self.assertFalse(solver.warm_started)
        self.assertIn("device_shapes", solver.warm_start_reason)

    def test_dtype_change_falls_back_to_cold_start(self):
        net, devices, T = two_bus_system(T=12)
        float32_devices = torchify(devices, dtype=torch.float32)
        solver = make_solver(dtype=torch.float32, num_iterations=50, verbose=0)

        solver.solve(net, float32_devices, T, initial_state=self.state.copy())

        self.assertFalse(solver.warm_started)
        self.assertIn("dtype", solver.warm_start_reason)

    def test_stateless_state_falls_back(self):
        """A state pickled before the fingerprint existed carries `layout=None`."""
        stale = self.state.copy().update(layout=None)
        solver = make_solver(num_iterations=50, verbose=0)

        solver.solve(self.net, self.torch_devices, self.T, initial_state=stale)

        self.assertFalse(solver.warm_started)
        self.assertIn("no layout fingerprint", solver.warm_start_reason)

    def test_non_state_falls_back(self):
        solver = make_solver(num_iterations=50, verbose=0)

        solver.solve(self.net, self.torch_devices, self.T, initial_state={"not": "a state"})

        self.assertFalse(solver.warm_started)
        self.assertIn("not an ADMMState", solver.warm_start_reason)

    def test_rho_angle_only_change_rescales_and_restamps(self):
        """A change in rho_angle alone must also be treated as a rho change."""
        solver = make_solver(rho_power=1.0, rho_angle=5.0, num_iterations=50)
        layout = ADMMLayout.of(
            self.net, self.torch_devices, self.T, solver.machine, solver.dtype, 0, None
        )
        solver.warm_start_rho_rescaled = False
        stale = self.state.copy().update(rho_power=1.0, rho_angle=1.0)

        accepted, reason = solver._accept_initial_state(stale, layout)

        self.assertIsNone(reason)
        self.assertTrue(solver.warm_start_rho_rescaled)
        self.assertEqual(accepted.rho_angle, 5.0)
        torch.testing.assert_close(accepted.dual_power, self.state.dual_power)

    def test_rho_change_rescales_duals_instead_of_refusing(self):
        """rho is a change of variables on `u = nu / rho`, not an incompatibility."""
        solver = make_solver(rho_power=10.0, num_iterations=50)
        layout = ADMMLayout.of(
            self.net, self.torch_devices, self.T, solver.machine, solver.dtype, 0, None
        )
        solver.warm_start_rho_rescaled = False

        rescaled, reason = solver._accept_initial_state(self.state.copy(), layout)

        self.assertIsNone(reason)
        self.assertTrue(solver.warm_start_rho_rescaled)
        self.assertEqual(rescaled.rho_power, 10.0)
        # nu = rho * u is what must be preserved.
        torch.testing.assert_close(
            rescaled.dual_power * 10.0,
            self.state.dual_power * self.state.rho_power,
            rtol=1e-12,
            atol=1e-12,
        )

        # ... and the full solve at the new rho lands where a cold one does.
        warm = make_solver(rho_power=10.0)
        warm_state, _ = warm.solve(
            self.net, self.torch_devices, self.T, initial_state=self.state.copy()
        )
        cold = make_solver(rho_power=10.0)
        cold_state, _ = cold.solve(self.net, self.torch_devices, self.T)

        self.assertTrue(warm.warm_started)
        self.assertTrue(warm.warm_start_rho_rescaled)
        self.assertAlmostEqual(
            float(warm_state.objective) / float(cold_state.objective), 1.0, delta=1e-4
        )


# ====
# (e) The planning path
# ====


class TestWarmStartInPlanningProblem(unittest.TestCase):
    def test_stochastic_problem_keeps_state_per_block(self):
        problem, subs = two_block_problem(solver_kwargs={"num_iterations": 500})
        problem.initialize_workers(2)
        try:
            theta = {"generator_capacity": subs[0].layer.devices[0].nominal_capacity.clone()}
            problem.forward(**theta)
            problem.forward(**{k: v * 1.01 for k, v in theta.items()})
        finally:
            problem.shutdown_workers()

        for sub in subs:
            stats = sub.layer.warm_start_stats
            self.assertEqual([s["warm_started"] for s in stats], [False, True])
            self.assertEqual(sub.layer.forward_count, 2)

        self.assertNotEqual(id(subs[0].layer.state), id(subs[1].layer.state))
        self.assertNotEqual(id(subs[0].layer), id(subs[1].layer))
        for sub in subs:
            self.assertGreater(
                sub.layer.state.cumulative_iteration,
                sub.layer.warm_start_stats[0]["iterations"] - 1,
            )
            self.assertEqual(
                sub.layer.state.cumulative_iteration,
                sum(s["iterations"] for s in sub.layer.warm_start_stats),
            )

    def test_warm_start_gradient_matches_cold_gradient(self):
        """Warm start is not *exactly* gradient-neutral (the tape starts elsewhere).

        At tight settings the difference must be inside `rtol=1e-3`; if it is not,
        that is a finding about the solve tolerance, not about warm starting.
        """
        solver_kwargs = {"atol": 1e-8, "rtol": 1e-8, "num_iterations": 20000}

        warm_problem, warm_subs = two_block_problem(solver_kwargs=solver_kwargs)
        theta1 = {"generator_capacity": warm_subs[0].layer.devices[0].nominal_capacity.clone()}
        theta2 = {k: v * 1.01 for k, v in theta1.items()}

        warm_problem.forward(**theta1)  # build the carried state
        J_warm, g_warm = warm_problem.forward_and_back(**theta2)
        self.assertTrue(all(s["warm_started"] for s in warm_subs[0].layer.warm_start_stats[1:]))

        cold_problem, _ = two_block_problem(warm_start=False, solver_kwargs=solver_kwargs)
        J_cold, g_cold = cold_problem.forward_and_back(**theta2)

        # The two runs unrolled different numbers of iterations, so the tapes are
        # genuinely different and the agreement below is not vacuous.
        warm_iters = sum(s["iterations"] for s in warm_subs[0].layer.warm_start_stats[1:])
        cold_iters = cold_problem.subproblems[0].layer.warm_start_stats[0]["iterations"]
        self.assertLess(warm_iters, cold_iters)

        self.assertAlmostEqual(float(J_warm) / float(J_cold), 1.0, delta=1e-6)
        for key in g_cold:
            self.assertGreater(float(torch.max(torch.abs(g_cold[key]))), 1.0)
            torch.testing.assert_close(g_warm[key], g_cold[key], rtol=1e-3, atol=1e-6)


# ====
# (f) Configuration
# ====


class TestWarmStartResetEvery(unittest.TestCase):
    """Forward call `i` (0-based) cold-starts iff `i % N == 0`."""

    def _flags(self, **layer_kwargs):
        net, devices, T = planning_block(T=6)
        layer = ADMMLayer(
            net,
            devices,
            PARAMETER_NAMES,
            time_horizon=T,
            solver=make_solver(num_iterations=20),
            **layer_kwargs,
        )
        capacity = devices[0].nominal_capacity
        for i in range(4):
            layer.forward(generator_capacity=capacity * (1.0 + 0.001 * i))
        return [s["warm_started"] for s in layer.warm_start_stats]

    def test_warm_start_reset_every(self):
        self.assertEqual(self._flags(warm_start_reset_every=2), [False, True, False, True])

    def test_no_reset_by_default(self):
        self.assertEqual(self._flags(warm_start_reset_every=None), [False, True, True, True])

    def test_warm_start_disabled(self):
        self.assertEqual(self._flags(warm_start=False), [False, False, False, False])

    def test_reset_warm_start_drops_the_state(self):
        net, devices, T = planning_block(T=6)
        layer = ADMMLayer(
            net, devices, PARAMETER_NAMES, time_horizon=T, solver=make_solver(num_iterations=20)
        )
        capacity = devices[0].nominal_capacity
        layer.forward(generator_capacity=capacity)
        layer.reset_warm_start()
        self.assertIsNone(layer.state)
        layer.forward(generator_capacity=capacity)
        self.assertEqual([s["warm_started"] for s in layer.warm_start_stats], [False, False])


# ====
# (g) Odds and ends
# ====


class TestStateCopyAndGuards(unittest.TestCase):
    def test_extended_state_copy_preserves_type(self):
        """Regression: the old hand-written `copy()` downcast to `ADMMState`."""
        weights = [[torch.ones(2, 3)], [torch.ones(1, 3)]]
        state = ExtendedADMMState(
            num_terminals=None,
            num_ac_terminals=None,
            power=[[torch.zeros(2, 3)], [torch.zeros(1, 3)]],
            phase=[None, None],
            dual_power=torch.zeros(2, 3),
            dual_phase=[None, None],
            _power_weights=weights,
            _angle_weights=[None, None],
        )

        copied = state.copy()

        self.assertIsInstance(copied, ExtendedADMMState)
        self.assertIsNotNone(copied.power_weights)
        self.assertEqual(len(copied.power_weights), 2)
        torch.testing.assert_close(copied.power_weights[0][0], weights[0][0])
        # A deep copy, not an alias.
        self.assertIsNot(copied.power_weights[0][0], weights[0][0])

    def test_copy_carries_every_field(self):
        state = ADMMState(
            num_terminals=None,
            num_ac_terminals=None,
            power=[[torch.zeros(1, 2)]],
            phase=[None],
            dual_power=torch.zeros(1, 2),
            dual_phase=[None],
            cumulative_iteration=7,
            layout=ADMMLayout(1, 1, 0, None, (), "cpu", "torch.float64"),
        )
        copied = state.copy()
        self.assertEqual(copied.cumulative_iteration, 7)
        self.assertEqual(copied.layout, state.layout)

    def test_minimum_iterations_below_ten_warns(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ADMMSolver(minimum_iterations=1)
        self.assertTrue(any("stale dual residual" in str(w.message) for w in caught))

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ADMMSolver(minimum_iterations=100)
        self.assertFalse(any("stale dual residual" in str(w.message) for w in caught))

    def test_layout_explains_the_first_mismatch(self):
        base = ADMMLayout(12, 2, 0, None, (("Generator", 2, 1, False),), "cpu", "torch.float64")
        self.assertIsNone(base.explain_mismatch(base))
        self.assertIn(
            "num_nodes",
            base.explain_mismatch(
                base.__class__(12, 3, 0, None, base.device_shapes, "cpu", "torch.float64")
            ),
        )
        self.assertIn(
            "machine",
            base.explain_mismatch(
                base.__class__(12, 2, 0, None, base.device_shapes, "cuda", "torch.float64")
            ),
        )

    def test_save_and_load_warm_start_round_trip(self):
        net, devices, T = planning_block(T=8)
        capacity = devices[0].nominal_capacity

        source = ADMMLayer(
            net, devices, PARAMETER_NAMES, time_horizon=T, solver=make_solver(num_iterations=400)
        )
        source.forward(generator_capacity=capacity)
        reference = float(source.forward(generator_capacity=capacity).objective)

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "block_0.pt")
            source.save_warm_start(path)

            restored = ADMMLayer(
                net,
                devices,
                PARAMETER_NAMES,
                time_horizon=T,
                solver=make_solver(num_iterations=400),
            )
            self.assertFalse(restored.load_warm_start(os.path.join(tmp, "missing.pt")))
            self.assertIsNone(restored.state)
            self.assertTrue(restored.load_warm_start(path))

            state = restored.forward(generator_capacity=capacity)

        self.assertTrue(restored.warm_start_stats[0]["warm_started"])
        self.assertAlmostEqual(float(state.objective) / reference, 1.0, delta=1e-6)


if __name__ == "__main__":
    unittest.main()


def test_clone_detach_preserves_namedtuples():
    from collections import namedtuple
    import torch
    from zap.admm.basic_solver import _clone_detach

    NT = namedtuple("NT", ["a", "b"])
    out = _clone_detach(NT(torch.ones(2), torch.zeros(2)))
    assert isinstance(out, NT) and out.a.shape == (2,)
