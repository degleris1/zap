"""Free cyclic state of charge (`StorageUnit.soc_mode = "cyclic_free"`).

`fixed` (the default) pins both endpoints of a block at `initial_soc` / `final_soc`.
`cyclic_free` drops both pins and imposes `energy[:, 0] == energy[:, T]` instead,
in the cvx model and in the ADMM prox alike. These tests check that the two paths
agree, that the relaxation is a relaxation, and that a warm start cannot cross
the two modes.
"""

import unittest

import cvxpy as cp
import numpy as np
import torch

import zap
from zap.admm import ADMMSolver
from zap.admm.basic_solver import ADMMLayout
from zap.devices import StorageUnit

torch.set_default_dtype(torch.float64)


def evening_peak_system(soc_mode="fixed", T=12):
    """2 buses, T hours, one battery, a peak at the *start* of the block.

    The expensive peaker is needed in hours 0-3; the cheap generator only has
    spare capacity later in the block. A block that may pick its own level starts
    full, discharges through the peak and refills from the cheap hours, ending
    where it started. A `fixed` block starts at 50 % and can displace only half
    as much peaker energy, so it is strictly more expensive.
    """
    net = zap.PowerNetwork(2)

    load_profile = np.array([60.0, 70.0, 65.0, 55.0] + [10.0] * (T - 4))[:T]

    generators = zap.Generator(
        num_nodes=2,
        name=np.array(["cheap", "peaker"]),
        terminal=np.array([0, 0]),
        nominal_capacity=np.array([30.0, 100.0]),
        dynamic_capacity=np.vstack([np.ones(T), np.ones(T)]),
        linear_cost=np.array([10.0, 400.0]),
        emission_rates=np.array([0.4, 0.6]),
    )
    load = zap.Load(
        num_nodes=2,
        name=np.array(["l0"]),
        terminal=np.array([1]),
        load=load_profile.reshape(1, T),
        linear_cost=np.array([5000.0]),
    )
    line = zap.DCLine(
        num_nodes=2,
        name=np.array(["ln0"]),
        source_terminal=np.array([0]),
        sink_terminal=np.array([1]),
        capacity=np.array([200.0]),
        linear_cost=np.array([0.0]),
    )
    battery = StorageUnit(
        num_nodes=2,
        name=np.array(["b0"]),
        terminal=np.array([1]),
        power_capacity=np.array([20.0]),
        duration=np.array([4.0]),
        charge_efficiency=np.array([1.0]),
        discharge_efficiency=np.array([1.0]),
        linear_cost=np.array([0.0]),
        soc_mode=soc_mode,
    )
    return net, [generators, load, line, battery], T


def heterogeneous_fleet(soc_mode="fixed", T=12):
    """3 buses, 2 generators, 1 load, 2 lines and a 2-unit heterogeneous battery."""
    net = zap.PowerNetwork(3)
    load_profile = np.array(
        [20.0, 20.0, 20.0, 25.0, 40.0, 70.0, 100.0, 110.0, 95.0, 60.0, 30.0, 20.0]
    )[:T]

    generators = zap.Generator(
        num_nodes=3,
        name=np.array(["g0", "g1"]),
        terminal=np.array([0, 1]),
        nominal_capacity=np.array([120.0, 55.0]),
        dynamic_capacity=np.vstack([np.ones(T), np.ones(T)]),
        linear_cost=np.array([180.0, 5.0]),
        emission_rates=np.array([0.4, 0.0]),
    )
    load = zap.Load(
        num_nodes=3,
        name=np.array(["l0"]),
        terminal=np.array([2]),
        load=load_profile.reshape(1, T),
        linear_cost=np.array([500.0]),
    )
    lines = zap.DCLine(
        num_nodes=3,
        name=np.array(["ln0", "ln1"]),
        source_terminal=np.array([0, 1]),
        sink_terminal=np.array([2, 2]),
        capacity=np.array([200.0, 200.0]),
        linear_cost=np.array([0.0, 0.0]),
    )
    battery = StorageUnit(
        num_nodes=3,
        name=np.array(["b0", "b1"]),
        terminal=np.array([2, 1]),
        power_capacity=np.array([30.0, 15.0]),
        duration=np.array([4.0, 8.0]),
        charge_efficiency=np.array([0.90, 0.80]),
        discharge_efficiency=np.array([0.70, 1.00]),
        linear_cost=np.array([0.5, 2.0]),
        initial_soc=np.array([0.4, 0.6]),
        final_soc=np.array([0.55, 0.3]),
        soc_mode=soc_mode,
    )
    return net, [generators, load, lines, battery], T


def battery_state(outcome, index=3):
    state = outcome.local_variables[index]
    energy = np.asarray(state[0] if not hasattr(state, "energy") else state.energy)
    return energy


class TestCyclicFreeCVX(unittest.TestCase):
    """(a), (b): the cvx model."""

    def test_cyclic_free_starts_where_it_ends_and_is_not_pinned(self):
        net, devices, T = evening_peak_system("cyclic_free")
        outcome = net.dispatch(devices, time_horizon=T, solver=cp.HIGHS, add_ground=False)
        energy = battery_state(outcome)

        emax = float(devices[3].power_capacity[0, 0] * devices[3].duration[0, 0])
        self.assertAlmostEqual(float(energy[0, 0]), float(energy[0, -1]), places=6)
        # The optimum wants a full battery at hour 0: it is nowhere near the
        # 50 % the `fixed` mode would have imposed.
        self.assertGreater(float(energy[0, 0]) / emax, 0.9)
        # ... and the level moves inside the block, so this is not a trivially
        # constant trajectory.
        self.assertLess(float(energy[0, 1:-1].min()) / emax, 0.6)

    def test_fixed_mode_is_still_pinned(self):
        net, devices, T = evening_peak_system("fixed")
        outcome = net.dispatch(devices, time_horizon=T, solver=cp.HIGHS, add_ground=False)
        energy = battery_state(outcome)
        emax = float(devices[3].power_capacity[0, 0] * devices[3].duration[0, 0])
        self.assertAlmostEqual(float(energy[0, 0]) / emax, 0.5, places=6)
        self.assertAlmostEqual(float(energy[0, -1]) / emax, 0.5, places=6)

    def test_cyclic_free_is_a_relaxation_of_fixed(self):
        net_f, dev_f, T = evening_peak_system("fixed")
        net_c, dev_c, _ = evening_peak_system("cyclic_free")

        fixed = net_f.dispatch(dev_f, time_horizon=T, solver=cp.HIGHS, add_ground=False)
        free = net_c.dispatch(dev_c, time_horizon=T, solver=cp.HIGHS, add_ground=False)

        self.assertLessEqual(float(free.problem.value), float(fixed.problem.value) + 1e-6)
        # On this instance the relaxation is strict.
        self.assertLess(float(free.problem.value), float(fixed.problem.value) - 1e-3)

    def test_cyclic_free_rejects_a_bad_mode(self):
        with self.assertRaises(ValueError):
            StorageUnit(
                num_nodes=1,
                name=np.array(["b"]),
                terminal=np.array([0]),
                power_capacity=np.array([1.0]),
                duration=np.array([1.0]),
                soc_mode="cyclic",
            )

    def test_mode_survives_sample_time_and_torchify(self):
        _, devices, T = evening_peak_system("cyclic_free")
        battery = devices[3]
        self.assertEqual(battery.sample_time(np.arange(4), T).soc_mode, "cyclic_free")
        self.assertEqual(
            battery.torchify(machine="cpu", dtype=torch.float64).soc_mode, "cyclic_free"
        )
        scaled = battery.sample_time(np.arange(T), T)
        scaled.scale_power(10.0)
        self.assertEqual(scaled.soc_mode, "cyclic_free")


def exact_prox(device: StorageUnit, z: np.ndarray, rho: float):
    """The prox the ADMM update must compute, solved by a conic solver."""
    N, T = z.shape
    power = cp.Variable((N, T))
    energy = cp.Variable((N, T + 1))
    charge = cp.Variable((N, T))
    discharge = cp.Variable((N, T))
    state = [energy, charge, discharge]

    constraints = [c == 0 for c in device.equality_constraints([power], None, state, la=cp)]
    constraints += [c <= 0 for c in device.inequality_constraints([power], None, state, la=cp)]

    objective = device.operation_cost([power], None, state, la=cp)
    objective += 0.5 * rho * cp.sum_squares(power - z)

    problem = cp.Problem(cp.Minimize(objective), constraints)
    problem.solve(solver=cp.CLARABEL)
    assert problem.status == cp.OPTIMAL, problem.status
    return power.value, energy.value


def run_prox(device: StorageUnit, z: np.ndarray, rho: float, inner_iterations: int):
    torch_device = device.torchify(machine="cpu", dtype=torch.float64)
    zt = torch.tensor(z, dtype=torch.float64)
    power, _angle, state = torch_device.admm_prox_update(
        rho,
        rho,
        [zt],
        None,
        inner_iterations=inner_iterations,
        inner_over_relaxation=1.8,
    )
    return power[0].numpy(), state


def prox_battery(soc_mode):
    return StorageUnit(
        num_nodes=1,
        name=np.array([f"b{i}" for i in range(4)]),
        terminal=np.zeros(4, dtype=int),
        power_capacity=np.array([10.0, 25.0, 7.5, 100.0]),
        duration=np.array([4.0, 2.0, 8.0, 1.5]),
        charge_efficiency=np.array([0.95, 0.88, 1.0, 0.92]),
        discharge_efficiency=np.array([0.85, 0.99, 0.90, 1.0]),
        linear_cost=np.array([1.5, 0.0, 3.0, 0.25]),
        initial_soc=np.array([0.3, 0.5, 0.7, 0.4]),
        final_soc=np.array([0.6, 0.2, 0.5, 0.45]),
        soc_mode=soc_mode,
    )


class TestCyclicFreeProx(unittest.TestCase):
    """(c), (d): the ADMM prox."""

    def setUp(self):
        self.T = 12
        self.rho = 1.0
        rng = np.random.default_rng(11)
        device = prox_battery("cyclic_free")
        pmax = np.asarray(device.power_capacity).reshape(-1, 1)
        self.z = rng.uniform(-1.0, 1.0, (device.num_devices, self.T)) * pmax

    def test_prox_matches_cvxpy_cyclic_free(self):
        device = prox_battery("cyclic_free")
        p_exact, _ = exact_prox(device, self.z, self.rho)
        p_admm, _ = run_prox(device, self.z, self.rho, inner_iterations=4000)

        tol = 1e-3 * float(np.max(device.power_capacity))
        self.assertLess(float(np.max(np.abs(p_admm - p_exact))), tol)

    def test_prox_iterate_is_cyclic(self):
        device = prox_battery("cyclic_free")
        _, state = run_prox(device, self.z, self.rho, inner_iterations=4000)
        energy = np.asarray(state.energy)
        emax = np.asarray(device.power_capacity).reshape(-1) * np.asarray(device.duration).reshape(
            -1
        )

        np.testing.assert_allclose(energy[:, 0], energy[:, -1], atol=1e-6 * float(emax.max()))
        # And not pinned to the (ignored) initial_soc / final_soc.
        self.assertTrue(np.all(energy >= -1e-9))
        self.assertTrue(np.all(energy <= emax.reshape(-1, 1) + 1e-9))

    def test_prox_ignores_initial_and_final_soc_in_cyclic_free(self):
        base = prox_battery("cyclic_free")
        moved = prox_battery("cyclic_free")
        moved.initial_soc = np.array([0.9, 0.9, 0.9, 0.9]).reshape(-1, 1)
        moved.final_soc = np.array([0.1, 0.1, 0.1, 0.1]).reshape(-1, 1)

        p_base, _ = run_prox(base, self.z, self.rho, inner_iterations=4000)
        p_moved, _ = run_prox(moved, self.z, self.rho, inner_iterations=4000)
        np.testing.assert_allclose(p_base, p_moved, atol=1e-5)

    def test_mode_change_invalidates_the_prox_cache(self):
        """Flipping `soc_mode` on a torchified device must rebuild ymin/ymax and S."""
        device = prox_battery("fixed").torchify(machine="cpu", dtype=torch.float64)
        zt = torch.tensor(self.z, dtype=torch.float64)
        _, _, fixed_state = device.admm_prox_update(
            self.rho, self.rho, [zt], None, inner_iterations=2000, inner_over_relaxation=1.8
        )
        emax = np.asarray(device.power_capacity).reshape(-1) * np.asarray(device.duration).reshape(
            -1
        )
        np.testing.assert_allclose(
            np.asarray(fixed_state.energy)[:, 0],
            np.asarray(device.initial_soc).reshape(-1) * emax,
            atol=1e-9,
        )

        device.soc_mode = "cyclic_free"
        _, _, free_state = device.admm_prox_update(
            self.rho, self.rho, [zt], None, inner_iterations=4000, inner_over_relaxation=1.8
        )
        energy = np.asarray(free_state.energy)
        np.testing.assert_allclose(energy[:, 0], energy[:, -1], atol=1e-6 * float(emax.max()))

        reference = prox_battery("cyclic_free")
        p_exact, _ = exact_prox(reference, self.z, self.rho)
        np.testing.assert_allclose(
            np.asarray(free_state.discharge - free_state.charge),
            p_exact,
            atol=1e-3 * float(np.max(reference.power_capacity)),
        )


class TestADMMvsLPCyclicFree(unittest.TestCase):
    """(c): a full ADMM solve of a heterogeneous fleet matches the LP."""

    def test_admm_matches_lp(self):
        net, devices, T = heterogeneous_fleet("cyclic_free")

        lp = net.dispatch(devices, time_horizon=T, solver=cp.HIGHS, add_ground=False)

        torch_devices = [d.torchify(machine="cpu", dtype=torch.float64) for d in devices]
        solver = ADMMSolver(
            machine="cpu",
            dtype=torch.float64,
            num_iterations=20000,
            rho_power=1.0,
            adaptive_rho=False,
            battery_inner_iterations=1000,
            battery_inner_over_relaxation=1.8,
            atol=1e-10,
            rtol=1e-10,
            verbose=0,
        )
        state, _ = solver.solve(net, torch_devices, T)

        admm_cost = float(state.objective)
        lp_cost = float(lp.problem.value)
        self.assertLess(abs(admm_cost - lp_cost) / abs(lp_cost), 1e-3)

        imbalance = float(torch.max(torch.abs(state.num_terminals * state.avg_power)).item())
        peak_load = float(np.max(np.asarray(devices[1].load)))
        self.assertLess(imbalance, 1e-4 * peak_load)

        # Dispatch, not just the objective. The *per-unit* battery schedule of
        # this LP is degenerate -- HiGHS and CLARABEL return schedules that
        # differ by ~8 MW at the same optimal value -- so the comparable
        # dispatch quantities are the ones the LP pins down: total generation,
        # and (below) the storage trajectory's cyclicity. The strict per-unit
        # dispatch comparison lives in `TestCyclicFreeProx`, where the prox is
        # strictly convex and its solution unique.
        lp_gen = float(np.asarray(lp.power[0][0]).sum())
        admm_gen = float(np.asarray(state.power[0][0]).sum())
        self.assertLess(abs(lp_gen - admm_gen), 1e-3 * abs(lp_gen))

        # And the storage the ADMM solve found is genuinely cyclic.
        energy = np.asarray(state.local_variables[3].energy)
        emax = np.asarray(devices[3].power_capacity).reshape(-1) * np.asarray(
            devices[3].duration
        ).reshape(-1)
        np.testing.assert_allclose(energy[:, 0], energy[:, -1], atol=1e-4 * float(emax.max()))


class TestWarmStartAcrossModes(unittest.TestCase):
    """(f): a state from the other mode is refused, with a reason naming soc_mode."""

    def _solve(self, soc_mode, initial_state=None):
        net, devices, T = heterogeneous_fleet(soc_mode)
        torch_devices = [d.torchify(machine="cpu", dtype=torch.float64) for d in devices]
        solver = ADMMSolver(
            machine="cpu",
            dtype=torch.float64,
            num_iterations=50,
            minimum_iterations=10,
            rho_power=1.0,
            adaptive_rho=False,
            battery_inner_iterations=20,
            verbose=0,
        )
        state, _ = solver.solve(net, torch_devices, T, initial_state=initial_state)
        return solver, state

    def test_layout_records_the_mode(self):
        net, devices, T = heterogeneous_fleet("cyclic_free")
        layout = ADMMLayout.of(net, devices, T, "cpu", "torch.float64")
        self.assertEqual(layout.storage_soc_modes, ((3, "cyclic_free"),))

        net_f, devices_f, _ = heterogeneous_fleet("fixed")
        other = ADMMLayout.of(net_f, devices_f, T, "cpu", "torch.float64")
        reason = layout.explain_mismatch(other)
        self.assertIsNotNone(reason)
        self.assertIn("soc_mode", reason)

    def test_warm_start_across_modes_is_refused(self):
        _, fixed_state = self._solve("fixed")
        solver, _ = self._solve("cyclic_free", initial_state=fixed_state)

        self.assertFalse(solver.warm_started)
        self.assertIsNotNone(solver.warm_start_reason)
        self.assertIn("soc_mode", solver.warm_start_reason)

    def test_warm_start_within_a_mode_is_accepted(self):
        _, free_state = self._solve("cyclic_free")
        solver, _ = self._solve("cyclic_free", initial_state=free_state)

        self.assertTrue(solver.warm_started)
        self.assertIsNone(solver.warm_start_reason)


class TestHarnessWiring(unittest.TestCase):
    """The config key, the importer option and the run-card record."""

    def test_config_validates_the_enum(self):
        from experiments.ra import config as ra_config

        cfg = ra_config.base_config()
        cfg["system"]["storage_soc_mode"] = "cyclic_free"
        self.assertEqual(ra_config.validate(cfg)["system"]["storage_soc_mode"], "cyclic_free")

        cfg["system"]["storage_soc_mode"] = "cyclic"
        with self.assertRaises(ra_config.ConfigError) as ctx:
            ra_config.validate(cfg)
        self.assertIn("storage_soc_mode", str(ctx.exception))

    def test_load_options_default_and_forwarding(self):
        from experiments.ra import config as ra_config
        from experiments.ra import system as ra_system
        from zap.importers.wy_store import LoadOptions

        self.assertEqual(LoadOptions().storage_soc_mode, "fixed")

        cfg = ra_config.validate(ra_config.base_config())
        self.assertEqual(cfg["system"]["storage_soc_mode"], "fixed")
        self.assertEqual(ra_system.load_options(cfg).storage_soc_mode, "fixed")

        cfg["system"]["storage_soc_mode"] = "cyclic_free"
        self.assertEqual(ra_system.load_options(cfg).storage_soc_mode, "cyclic_free")

    def test_the_mode_is_part_of_the_system_cache_key(self):
        from experiments.ra import config as ra_config
        from experiments.ra import system as ra_system

        cfg = ra_config.validate(ra_config.base_config())
        fixed_key = ra_system.system_key(cfg, None)
        cfg["system"]["storage_soc_mode"] = "cyclic_free"
        self.assertNotEqual(fixed_key, ra_system.system_key(cfg, None))


if __name__ == "__main__":
    unittest.main()
