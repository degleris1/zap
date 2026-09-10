"""The step rules of the planning descent loop.

Spec: ``memory/plans/2026-09-10-step-rule-spec.md`` section 5, tests 1-10.

The rule ``experiments/ra`` shipped through phase 2 normalises the step by the
2-norm of the gradient over **all** parameter rows.  On ``ca2040_z4`` 127 of 166
rows are structurally frozen and carry 98.7 % of that norm, so a reported
1,000 MW step moved the design by 13 MW; from a short system the same rule
moves 1e5-1e6 MW in one iteration, because the step is proportional to the
gradient in units nobody set.  These tests pin the replacement:

* tests 1-3, 9: :class:`~zap.planning.solvers.AdamDescent` -- a learning rate in
  MW, ``eps`` in $/MW-yr, a first step of exactly ``lr`` whatever the gradient
  scale, convergence on a synthetic quadratic with a known optimum, and
  convergence under the minibatch noise measured on cell 5;
* test 4: ``GradientDescent`` is untouched, so cells 4 and 5 reproduce;
* test 5: the active-set trackers that make the step columns honest;
* tests 6, 10: the convergence stops and the per-row gradient table, on the
  48-hour planning fixture;
* test 7: no first-iteration spike from as-built with (effectively) unbounded
  expansion -- the property that removes the invented 10x capacity bound;
* test 8: the trust region expands on a locally linear objective.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ZAP_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(ZAP_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(ZAP_REPO_ROOT))

import zap.planning.trackers as tr
from zap.importers.wy_store import HourWindow, LoadOptions, load_system
from zap.planning.solvers import (
    AdagradDescent,
    AdamDescent,
    CapexScaledDescent,
    GradientDescent,
    TrustRegionDescent,
)
from zap.tests.test_ra_planning_methods import (
    N_HOURS,
    PLANNING_AVAILABLE,
    TASK_A,
    VOLL,
    PlanningFixtureMixin,
    deep_merge,
    plan_config,
    write_planning_dataset,
)

try:
    from experiments.ra import planning
    from experiments.ra.config import ConfigError
    from experiments.ra.planning import history as history_mod
except Exception:  # noqa: BLE001  # pragma: no cover - mirrors test_ra_planning_methods
    planning = None  # type: ignore[assignment]
    ConfigError = ValueError  # type: ignore[misc,assignment]
    history_mod = None  # type: ignore[assignment]


#: The gradient preset the fixture tests start from (CLARABEL, like the rest of
#: the planning tests; the campaign runs HiGHS).
GRADIENT = {
    "method": "gradient",
    "dispatch_solver": "CLARABEL",
    "optimizer": {"num_iterations": 4, "batch_size": 0},
}


def _step(algorithm, eta: np.ndarray, grad: np.ndarray, **kwargs) -> np.ndarray:
    """One step of ``algorithm`` on a single numpy parameter block."""
    state = {"x": np.asarray(eta, dtype=float).copy()}
    algorithm.step(state, {"x": torch.as_tensor(np.asarray(grad, dtype=float))}, **kwargs)
    return np.asarray(state["x"], dtype=float)


# ===========================================================================
# 1-3, 9.  Adam
# ===========================================================================


class TestAdamStep(unittest.TestCase):
    def test_adam_first_step_is_lr(self):
        """Test 1: the first step is ``lr`` per coordinate, at any gradient scale.

        At ``t = 1`` the bias-corrected moments are ``m_hat = g`` and
        ``v_hat = g^2``, so ``Delta eta_j = -lr * g_j / (|g_j| + eps)``, which is
        ``-lr * sign(g_j)`` in the limit ``eps -> 0``.  That is the property that
        makes a first-iteration spike impossible from any starting point, with no
        invented capacity bound: the gradients below span seven orders of
        magnitude and every one of them moves exactly 200 MW.

        Deviation from the spec's wording: it asks for exact equality with the
        *shipped* ``eps = 1.0``, which is impossible by construction -- ``eps``
        is in $/MW-yr and a row whose gradient is 1 $/MW-yr is deliberately
        damped to ``lr/2`` (that is what test 3 checks).  Equality holds in the
        ``eps -> 0`` limit, and to within ``eps/|g|`` at the shipped value, which
        the second half of this test pins at the campaign's gradient scale.
        """
        lr = 200.0
        grad = np.array([1e0, -1e1, 1e2, -1e3, 1e4, -1e5, 1e6, -1e7, 0.0])
        eta = np.zeros_like(grad)

        moved = _step(AdamDescent(step_size=lr, eps=1e-12), eta, grad) - eta
        nonzero = grad != 0.0
        np.testing.assert_allclose(np.abs(moved[nonzero]), lr, rtol=1e-9)
        np.testing.assert_allclose(np.sign(moved[nonzero]), -np.sign(grad[nonzero]))
        self.assertEqual(moved[~nonzero].tolist(), [0.0])

        # At the shipped eps = 1.0 the damping is eps/(|g| + eps), which is
        # below 0.1 % for every gradient at or above 1e3 $/MW-yr -- the whole
        # range the campaign's rows live in.
        moved = _step(AdamDescent(step_size=lr, eps=1.0), eta, grad) - eta
        big = np.abs(grad) >= 1e3
        np.testing.assert_allclose(np.abs(moved[big]), lr, rtol=1e-3)

    def test_adam_eps_units(self):
        """Test 3: ``eps`` is in $/MW-yr, so a 0.5 $/MW-yr row is damped.

        ``|Delta eta| = lr * 0.5 / (0.5 + 1.0) = lr / 3``.  At the ML default
        ``eps = 1e-8`` the same row would take a full ``lr`` step -- 200 MW of
        capacity moved on numerical dust (the ADMM-rho units trap again,
        LESSONS 2026-09-09).
        """
        lr = 200.0
        moved = _step(AdamDescent(step_size=lr, eps=1.0), np.zeros(1), np.array([0.5]))
        self.assertLess(abs(float(moved[0])), lr / 2.0)
        self.assertAlmostEqual(abs(float(moved[0])), lr / 3.0, places=9)

        dust = _step(AdamDescent(step_size=lr, eps=1e-8), np.zeros(1), np.array([0.5]))
        self.assertAlmostEqual(abs(float(dust[0])), lr, places=4)

    def test_adam_max_step_caps_every_row(self):
        """``max_step_mw`` defaults to ``3 * lr`` and is a hard per-row cap."""
        adam = AdamDescent(step_size=100.0)
        self.assertEqual(adam.max_step, 300.0)
        capped = AdamDescent(step_size=100.0, max_step_mw=25.0)
        moved = _step(capped, np.zeros(4), np.array([1e6, -1e6, 1.0, -1.0]))
        self.assertLessEqual(float(np.max(np.abs(moved))), 25.0 + 1e-12)
        self.assertEqual(capped.diagnostics()["n_capped"], 4)

    def test_adam_synthetic_quadratic(self):
        """Test 2: 300 cosine-decayed iterations reach a known optimum.

        ``F(eta) = 0.5 * sum_j h_j (eta_j - eta*_j)^2`` with the curvatures
        spanning two orders of magnitude, box bounds, and the walk starting at
        zero.  Adam has to cross up to 995 MW on some rows and 12 MW on others
        with one learning rate; the assertion is the spec's, 1 % of the initial
        worst-case distance.
        """
        rng = np.random.default_rng(0)
        n = 40
        h = rng.uniform(1.0, 100.0, n)
        target = rng.uniform(0.0, 1000.0, n)
        lower, upper = np.zeros(n), np.full(n, 2000.0)

        eta = np.zeros(n)
        distance = float(np.max(np.abs(eta - target)))
        iterations = 300
        adam = AdamDescent(step_size=20.0, decay="cosine", decay_final_frac=0.05)
        for t in range(iterations):
            grad = h * (eta - target)
            eta = np.clip(
                _step(adam, eta, grad, iteration=t, num_iterations=iterations), lower, upper
            )

        self.assertLess(float(np.max(np.abs(eta - target))), 0.01 * distance)

    def test_adam_under_noise(self):
        """Test 9: Adam converges at the per-row SNR measured on cell 5.

        Each row's gradient is corrupted by zero-mean Gaussian noise with
        ``sigma_j = |g_j| / 0.3``, i.e. a per-row SNR of 0.3 -- the middle of
        c5's measured 0.20-0.42, where the sign of the gradient is wrong on
        31-45 % of iterations.  Adam's ``sqrt(v_hat)`` is then approximately
        ``sigma``, so the step self-scales to about ``lr * SNR`` and ``beta1 =
        0.9`` averages ~10 gradients, which is exactly the window that recovers
        the deterministic value.

        The clipped rule at the same nominal step length has neither: its step
        is a fixed L2 length along a direction that is mostly noise, so it
        random-walks and its last-50 average is an order of magnitude further
        out.
        """
        rng = np.random.default_rng(0)
        n = 20
        h = rng.uniform(1.0, 100.0, n)
        target = rng.uniform(0.0, 1000.0, n)
        distance = float(np.max(np.abs(target)))
        snr, lr, iterations = 0.3, 20.0, 500

        def walk(algorithm, gd: bool):
            eta = np.zeros(n)
            noise_rng = np.random.default_rng(1)
            trail = []
            for t in range(iterations):
                true_grad = h * (eta - target)
                sigma = np.maximum(np.abs(true_grad) / snr, 1e-9)
                grad = true_grad + noise_rng.normal(0.0, sigma)
                kwargs = {} if gd else {"iteration": t, "num_iterations": iterations}
                eta = np.clip(_step(algorithm, eta, grad, **kwargs), 0.0, 2000.0)
                trail.append(eta.copy())
            return np.mean(trail[-50:], axis=0)

        adam = walk(AdamDescent(step_size=lr), gd=False)
        self.assertLess(float(np.max(np.abs(adam - target))), 0.05 * distance)

        # The same nominal per-iteration movement: `GradientDescent` moves
        # `step_size * clip` in L2, and Adam moves at most `lr` on each of n
        # coordinates, i.e. `lr * sqrt(n)`.
        clipped = walk(
            GradientDescent(step_size=1.0, clip=lr * np.sqrt(n)),
            gd=True,
        )
        self.assertGreater(float(np.max(np.abs(clipped - target))), 0.05 * distance)

    def test_adagrad_and_capex_scaled_move_in_mw(self):
        """The fallback and the ablation rule share Adam's units and cap."""
        moved = _step(AdagradDescent(step_size=50.0, eps=1e-12), np.zeros(3), [1e2, -1e6, 0.0])
        np.testing.assert_allclose(np.abs(moved[:2]), 50.0, rtol=1e-9)
        self.assertEqual(float(moved[2]), 0.0)

        # capex-scaled: alpha * g / gamma, capped at max_step_mw.  gamma = 0 is
        # what `floor` exists for -- it must not divide by zero.
        capex = CapexScaledDescent(
            step_size=1.0, capex={"x": np.array([100.0, 0.0])}, floor=10.0, max_step_mw=1e9
        )
        moved = _step(capex, np.zeros(2), np.array([200.0, 200.0]))
        np.testing.assert_allclose(moved, [-2.0, -20.0], rtol=1e-9)


# ===========================================================================
# 4.  GradientDescent is untouched
# ===========================================================================


class TestGradientDescentUnchanged(unittest.TestCase):
    """Test 4: the archived arithmetic of the reproducibility baseline.

    ``GradientDescent.step`` divides the *stacked* per-parameter 2-norms by
    ``clip`` and moves ``step_size`` along the clipped gradient.  The expected
    values below are written out longhand rather than recomputed from the class,
    so a change to the class fails this test instead of following it.
    """

    def test_gradient_descent_unchanged_clipping(self):
        grad = {
            "generator_capacity": np.array([3.0, -4.0]),  # 2-norm 5
            "storageunit_power": np.array([12.0]),  # 2-norm 12
        }
        state = {
            "generator_capacity": np.array([100.0, 200.0]),
            "storageunit_power": np.array([50.0]),
        }
        # total norm = sqrt(5^2 + 12^2) = 13 > clip = 6.5, so the scale is 0.5
        # and one step is step_size * 0.5 * g.
        algorithm = GradientDescent(step_size=2.0, clip=6.5)
        out = algorithm.step(
            {k: v.copy() for k, v in state.items()},
            {k: torch.as_tensor(v) for k, v in grad.items()},
        )
        np.testing.assert_allclose(out["generator_capacity"], [100.0 - 3.0, 200.0 + 4.0])
        np.testing.assert_allclose(out["storageunit_power"], [50.0 - 12.0])

    def test_gradient_descent_unchanged_unclipped(self):
        algorithm = GradientDescent(step_size=0.25, clip=1e9)
        out = algorithm.step(
            {"x": np.array([10.0, -10.0])},
            {"x": torch.as_tensor(np.array([4.0, 8.0]))},
        )
        np.testing.assert_allclose(out["x"], [9.0, -12.0])

    def test_gradient_descent_accepts_the_shared_keywords(self):
        """The loop passes the same keywords to every rule; this one ignores them."""
        out = GradientDescent(step_size=0.25, clip=1e9).step(
            {"x": np.array([10.0])},
            {"x": torch.as_tensor(np.array([4.0]))},
            iteration=7,
            num_iterations=100,
            actual_decrease=1.0,
            predicted_decrease=1.0,
        )
        np.testing.assert_allclose(out["x"], [9.0])


# ===========================================================================
# 5.  Active-set trackers
# ===========================================================================


class _StubProblem:
    """The two attributes the active-set trackers read off a problem."""

    def __init__(self, lower, upper):
        self.lower_bounds = lower
        self.upper_bounds = upper


class TestFreeGradTracker(unittest.TestCase):
    """Test 5: ``n_free`` / ``n_at_lower`` / ``free_grad_norm_l2``.

    The synthetic active set below is a miniature of ``ca2040_z4``: one frozen
    row (``lower == upper``), one row pinned on its floor, one on its ceiling
    and two interior, with the frozen row carrying almost all of the gradient
    norm -- which is exactly why the *free* norm is the one that describes the
    step.
    """

    def setUp(self):
        self.state = {"g": np.array([5.0, 0.1, 400.0, 50.0, 120.0])}
        self.problem = _StubProblem(
            {"g": np.array([5.0, 0.1, 0.0, 0.0, 0.0])},
            {"g": np.array([5.0, 900.0, 400.0, 900.0, 900.0])},
        )
        self.grad = {"g": torch.as_tensor(np.array([1.0e5, 7.0, -20.0, 3.0, 4.0]))}

    def test_counts_partition_the_rows(self):
        args = (None, self.grad, self.state, None, self.problem)
        self.assertEqual(tr.track_n_free(*args), 2)
        self.assertEqual(tr.track_n_at_lower(*args), 2)  # frozen row + the floor
        self.assertEqual(tr.track_n_at_upper(*args), 1)
        total = tr.track_n_free(*args) + tr.track_n_at_lower(*args) + tr.track_n_at_upper(*args)
        self.assertEqual(total, self.state["g"].size)

    def test_free_grad_norm_excludes_the_dead_rows(self):
        args = (None, self.grad, self.state, None, self.problem)
        self.assertAlmostEqual(
            tr.track_free_grad_norm_l2(*args), float(np.hypot(3.0, 4.0)), places=9
        )
        self.assertGreater(tr.track_grad_norm_l2(*args), 1.0e5)

    def test_step_norm_is_post_projection(self):
        last = {"g": np.array([5.0, 0.1, 400.0, 47.0, 116.0])}
        args = (None, self.grad, self.state, last, self.problem)
        self.assertAlmostEqual(tr.track_step_norm_mw_actual(*args), 5.0, places=9)
        self.assertAlmostEqual(tr.track_step_norm_free_mw(*args), 5.0, places=9)
        # No previous state: the step is zero, not the gradient norm.
        self.assertEqual(
            tr.track_step_norm_mw_actual(None, self.grad, self.state, None, self.problem), 0.0
        )


# ===========================================================================
# 8.  Trust region
# ===========================================================================


class TestTrustRegion(unittest.TestCase):
    def test_trust_region_expands_on_linear(self):
        """Test 8: ``rho ~ 1`` doubles the radius up to ``max_radius_mw``.

        F is an LP value function plus linear capex, so the first-order model is
        exact at these step lengths (c4's measured ``rho`` is 0.994-1.006 at
        every sampled iteration).  The radius must therefore grow geometrically
        and then sit at its ceiling.
        """
        region = TrustRegionDescent(
            initial_radius_mw=50.0, max_radius_mw=800.0, expand=2.0, eta_high=0.9
        )
        self.assertEqual(region.radius_mw, 50.0)

        radii = []
        for t in range(8):
            _step(
                region,
                np.zeros(3),
                np.array([1.0, 0.0, 0.0]),
                iteration=t,
                actual_decrease=1.0,
                predicted_decrease=1.0,  # rho = 1
            )
            radii.append(region.radius_mw)
        self.assertEqual(radii[:4], [100.0, 200.0, 400.0, 800.0])
        self.assertEqual(radii[-1], 800.0)  # capped at max_radius_mw
        self.assertAlmostEqual(region.diagnostics()["rho_actual_pred"], 1.0)

    def test_trust_region_shrinks_on_a_bad_model(self):
        region = TrustRegionDescent(initial_radius_mw=50.0, shrink=0.5, eta_low=0.1)
        _step(
            region,
            np.zeros(2),
            np.array([1.0, 1.0]),
            actual_decrease=-5.0,
            predicted_decrease=1.0,  # rho = -5
        )
        self.assertEqual(region.radius_mw, 25.0)

    def test_trust_region_step_length_is_the_radius(self):
        region = TrustRegionDescent(initial_radius_mw=30.0)
        moved = _step(region, np.zeros(2), np.array([3.0, 4.0]))
        np.testing.assert_allclose(moved, [-18.0, -24.0])  # -30 * g / ||g||
        self.assertAlmostEqual(float(np.linalg.norm(moved)), 30.0)

    def test_trust_region_normalises_over_the_projected_gradient(self):
        """Verifier finding D2: dead coordinates must not eat the radius.

        Four rows, only one of which can move: a frozen row carrying almost all
        of the gradient norm, a row on its floor whose descent direction points
        further down, a row on its ceiling pushed further up, and one interior
        row.  Normalising over all four would give the interior row a step of
        ``30 * 1 / 1e5``; over the projected gradient it gets the whole radius.
        """
        region = TrustRegionDescent(initial_radius_mw=30.0)
        eta = np.array([5.0, 0.0, 100.0, 50.0])
        state = {"x": eta.copy()}
        region.step(
            state,
            {"x": torch.as_tensor(np.array([1.0e5, 7.0, -7.0, 2.0]))},
            lower_bounds={"x": np.array([5.0, 0.0, 0.0, 0.0])},
            upper_bounds={"x": np.array([5.0, 900.0, 100.0, 900.0])},
        )
        moved = np.asarray(state["x"], dtype=float) - eta
        np.testing.assert_allclose(moved, [0.0, 0.0, 0.0, -30.0], atol=1e-9)
        self.assertAlmostEqual(float(np.linalg.norm(moved)), 30.0)

    def test_trust_region_without_bounds_is_unprojected(self):
        """A bare unit test of the rule (no box) keeps the plain behaviour."""
        region = TrustRegionDescent(initial_radius_mw=10.0)
        moved = _step(region, np.zeros(2), np.array([0.0, 1.0]))
        np.testing.assert_allclose(moved, [0.0, -10.0])

    def test_trust_region_without_a_model_keeps_its_radius(self):
        region = TrustRegionDescent(initial_radius_mw=50.0)
        _step(region, np.zeros(2), np.array([1.0, 1.0]), predicted_decrease=float("nan"))
        self.assertEqual(region.radius_mw, 50.0)


# ===========================================================================
# 6, 10 and the reproduction check, on the 48-hour fixture
# ===========================================================================


@unittest.skipUnless(PLANNING_AVAILABLE, TASK_A)
class TestStoppingCriteria(PlanningFixtureMixin, unittest.TestCase):
    """Test 6: the convergence stops on the 48 h, 2-block fixture."""

    def _plan(self, optimizer: dict):
        return self.plan(planning=deep_merge(GRADIENT, {"optimizer": optimizer}))

    def test_loose_objective_tolerance_stops_when_the_window_fills(self):
        """A tolerance nothing can beat stops the run -- but not before ``W``.

        The window does not shrink to fit (verifier finding D1): with
        ``tol_window = 4`` the first four recorded values are iterates 0..3, so
        the earliest the test can fire is iteration 3.
        """
        result = self._plan(
            {
                "rule": "adam",
                "step_size": 5.0,
                "num_iterations": 20,
                "stopping": {"tol_rel_objective": 1.0e9, "tol_window": 4},
            }
        )
        self.assertEqual(result.solver["stop_reason"], "objective_tolerance")
        self.assertEqual(result.solver["stopped_by"], "objective_tolerance")
        self.assertEqual(result.solver["num_iterations_completed"], 3)
        self.assertTrue(np.isfinite(result.objective["raw"]))
        self.assert_within_bounds(result)

    def test_the_window_must_fill_before_the_plateau_test_fires(self):
        """The same tolerance cannot stop a run shorter than its window.

        This is the regression for D1: the shrinking window made the test a
        single-iteration one for the first ``W`` iterations, which on c4's
        recorded series (4.7e-5 per iteration against a 20-iteration 9.4e-4)
        would have ended the run after one step at the campaign's 1e-4.
        """
        result = self._plan(
            {
                "rule": "adam",
                "step_size": 5.0,
                "num_iterations": 3,
                "stopping": {"tol_rel_objective": 1.0e9, "tol_window": 20},
            }
        )
        self.assertEqual(result.solver["stop_reason"], "num_iterations")
        self.assertEqual(result.solver["num_iterations_completed"], 3)

    def test_tight_objective_tolerance_runs_to_the_backstop(self):
        result = self._plan(
            {
                "rule": "adam",
                "step_size": 5.0,
                "num_iterations": 3,
                "stopping": {"tol_rel_objective": 1.0e-14, "tol_window": 20},
            }
        )
        self.assertEqual(result.solver["stop_reason"], "num_iterations")
        self.assertEqual(result.solver["stopped_by"], "iterations")
        self.assertEqual(result.solver["num_iterations_completed"], 3)

    def test_defaults_do_not_stop_early(self):
        """No tolerance set = the pre-2026-09-10 behaviour."""
        result = self._plan({"rule": "adam", "step_size": 5.0, "num_iterations": 3})
        self.assertEqual(result.solver["stop_reason"], "num_iterations")
        self.assertIsNone(result.solver["stopping"]["tol_rel_objective"])
        self.assertIsNone(result.solver["stopping"]["tol_stationarity"])
        self.assertEqual(result.solver["stopping"]["tol_window"], 20)
        self.assertEqual(result.solver["stopping"]["checkpoint_window"], 5)

    def test_loose_stationarity_tolerance_stops(self):
        result = self._plan(
            {
                "rule": "adam",
                "step_size": 5.0,
                "num_iterations": 20,
                "stopping": {"tol_stationarity": 1.0e9},
            }
        )
        self.assertEqual(result.solver["stop_reason"], "stationarity")
        self.assertEqual(result.solver["stopped_by"], "stationarity")
        self.assertLess(result.solver["num_iterations_completed"], 20)
        self.assertTrue(np.isfinite(float(result.solver["stationarity_max_final"])))

    def test_checkpoint_plateau_stops_a_minibatch_run(self):
        """A minibatch cell's plateau test runs on the checkpoint series."""
        result = self.plan(
            selection={"block_size": 8},
            planning=deep_merge(
                GRADIENT,
                {
                    "optimizer": {
                        "rule": "adam",
                        "step_size": 5.0,
                        "num_iterations": 30,
                        "batch_size": 2,
                        "batch_strategy": "random",
                        "design_selection": "best_checkpointed",
                        "checkpoint_every": 1,
                        # In CHECKPOINTS: 3 of them, so the earliest stop is
                        # the checkpoint at iteration 2.
                        "stopping": {
                            "tol_rel_objective": 1.0e9,
                            "tol_window": 20,
                            "checkpoint_window": 3,
                        },
                    }
                },
            ),
        )
        self.assertEqual(result.solver["stop_reason"], "checkpoint_tolerance")
        self.assertEqual(result.solver["stopped_by"], "checkpoint_tolerance")
        self.assertEqual(result.solver["num_iterations_completed"], 2)


@unittest.skipUnless(PLANNING_AVAILABLE, TASK_A)
class TestGradientRuleReproducesTheBaseline(PlanningFixtureMixin, unittest.TestCase):
    """Test 4, end to end: ``rule: gradient`` runs the historical arithmetic.

    Replays the recorded trajectory through ``GradientDescent.step`` and the
    problem's own projection: if the loop had picked up any of the new machinery
    -- a decay, a cap, a moment -- ``param[i + 1]`` would not be
    ``project(step(param[i], grad[i]))`` any more.  This is what guarantees that
    cells c4 and c5 reproduce bit-for-bit under the new config surface.
    """

    def test_rule_gradient_is_plain_gradient_descent(self):
        step_size, clip = 0.2, 5.0e3
        result = self.plan(
            planning=deep_merge(
                GRADIENT,
                {
                    "optimizer": {
                        "num_iterations": 4,
                        "rule": "gradient",
                        "step_size": step_size,
                        "clip": clip,
                        "grad_history_every": 1,
                    }
                },
            )
        )
        self.assertEqual(result.solver["rule"], "gradient")

        params = result.history["param"]
        grads = result.history["grad_sampled"]
        self.assertEqual(len(params), 5)

        algorithm = GradientDescent(step_size=step_size, clip=clip)
        lower = result.lower_bounds
        upper = result.upper_bounds
        for i in range(len(params) - 1):
            state = {k: np.asarray(v, dtype=float).copy() for k, v in params[i].items()}
            grad = {
                k: torch.as_tensor(np.asarray(v, dtype=float).reshape(state[k].shape))
                for k, v in grads[i].items()
            }
            moved = algorithm.step(state, grad)
            for param, value in moved.items():
                expected = np.clip(
                    value,
                    np.asarray(lower[param], dtype=float).reshape(value.shape),
                    np.asarray(upper[param], dtype=float).reshape(value.shape),
                )
                np.testing.assert_allclose(
                    np.asarray(params[i + 1][param], dtype=float).reshape(expected.shape),
                    expected,
                    rtol=0,
                    atol=0,
                    err_msg=f"{param} diverged at iteration {i + 1}",
                )

    def test_default_config_is_rule_gradient(self):
        """Nothing changes for a config written before the step-rule spec."""
        from experiments.ra.planning.base import PLANNING_DEFAULTS

        optimizer = PLANNING_DEFAULTS["optimizer"]
        self.assertEqual(optimizer["rule"], "gradient")
        self.assertEqual(optimizer["lr_decay"], "none")
        self.assertEqual(optimizer["grad_history_every"], 0)
        self.assertIsNone(optimizer["stopping"]["tol_rel_objective"])
        self.assertIsNone(optimizer["stopping"]["tol_stationarity"])


@unittest.skipUnless(PLANNING_AVAILABLE, TASK_A)
class TestIterationGradientTable(PlanningFixtureMixin, unittest.TestCase):
    """Test 10: ``iterations/<task>.iteration_gradient.parquet``."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.result = planning.plan(
            cls.loaded,
            plan_config(
                cls.dataset,
                planning=deep_merge(
                    GRADIENT,
                    {
                        "optimizer": {
                            "rule": "adam",
                            "step_size": 5.0,
                            "num_iterations": 4,
                            "grad_history_every": 2,
                        }
                    },
                ),
            ),
        )

    def test_schema_and_shape(self):
        tables = history_mod.build_iteration_tables(self.result)
        frame = tables["iteration_gradient"]
        self.assertEqual(list(frame.columns), list(history_mod.ITERATION_GRADIENT_COLUMNS))
        # grad_history_every = 2 over 5 recorded iterates (0..4) -> 0, 2, 4.
        recorded = sorted(frame["iteration"].unique().tolist())
        self.assertEqual(recorded, [0, 2, 4])

        n_rows = sum(np.asarray(v, dtype=float).size for v in self.result.parameters.values())
        for iteration in recorded:
            block = frame[frame["iteration"] == iteration]
            self.assertEqual(len(block), n_rows)
            self.assertTrue(np.all(np.isfinite(block["gradient"].to_numpy(dtype=float))))
            self.assertTrue(np.all(np.isfinite(block["capacity_mw"].to_numpy(dtype=float))))

        # The active-set flags partition the rows, and they agree with the
        # capacity and the bounds on the same row.
        flags = frame[["at_lower", "at_upper", "free"]].to_numpy(dtype=bool)
        np.testing.assert_array_equal(flags.sum(axis=1), np.ones(len(frame), dtype=int))
        at_lower = frame[frame["at_lower"]]
        self.assertTrue(
            np.all(
                at_lower["capacity_mw"].to_numpy(dtype=float)
                <= at_lower["lower_mw"].to_numpy(dtype=float) + 1e-6
            )
        )

        # Adam's moments are recorded, and the realised step never exceeds the cap.
        after_first = frame[frame["iteration"] > 0]
        self.assertTrue(np.all(np.isfinite(after_first["m_hat"].to_numpy(dtype=float))))
        self.assertTrue(np.all(np.isfinite(after_first["v_hat"].to_numpy(dtype=float))))
        self.assertTrue(
            np.all(np.abs(after_first["step_mw"].to_numpy(dtype=float)) <= 3 * 5.0 + 1e-6)
        )
        self.assertTrue(frame["name"].map(lambda s: bool(str(s))).all())

    def test_written_to_disk_and_read_back(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            written = history_mod.write_iteration_tables(self.result, run_dir)
            names = {path.name.split(".", 1)[1] for path in written}
            self.assertIn("iteration_gradient.parquet", names)
            frame = history_mod.read_table(run_dir, "iteration_gradient")
            self.assertEqual(list(frame.columns), list(history_mod.ITERATION_GRADIENT_COLUMNS))
            self.assertGreater(len(frame), 0)

    def test_off_by_default(self):
        result = self.plan(planning=deep_merge(GRADIENT, {"optimizer": {"num_iterations": 2}}))
        tables = history_mod.build_iteration_tables(result)
        self.assertTrue(tables["iteration_gradient"].empty)
        with tempfile.TemporaryDirectory() as tmp:
            written = history_mod.write_iteration_tables(result, Path(tmp))
            self.assertNotIn(
                "iteration_gradient.parquet",
                {path.name.split(".", 1)[1] for path in written},
            )


@unittest.skipUnless(PLANNING_AVAILABLE, TASK_A)
class TestHonestStepColumns(PlanningFixtureMixin, unittest.TestCase):
    """``step_norm_mw`` is the post-projection movement, not the asked-for step."""

    def test_step_norm_matches_the_parameter_history(self):
        result = self.plan(
            planning=deep_merge(
                GRADIENT,
                {"optimizer": {"rule": "adam", "step_size": 5.0, "num_iterations": 3}},
            )
        )
        frame = history_mod.build_iteration_tables(result)["iterations"]
        params = result.history["param"]
        for i in range(1, len(params)):
            realised = np.sqrt(
                sum(
                    float(
                        np.sum(
                            (
                                np.asarray(params[i][k], dtype=float).reshape(-1)
                                - np.asarray(params[i - 1][k], dtype=float).reshape(-1)
                            )
                            ** 2
                        )
                    )
                    for k in params[i]
                )
            )
            row = frame[frame["iteration"] == i].iloc[0]
            self.assertAlmostEqual(float(row["step_norm_mw"]), realised, places=8)
            self.assertAlmostEqual(float(row["step_norm_mw_actual"]), realised, places=8)
            self.assertLessEqual(float(row["step_norm_free_mw"]), realised + 1e-9)
            self.assertGreaterEqual(int(row["n_free"]), 0)
            self.assertEqual(
                int(row["n_free"]) + int(row["n_at_lower"]) + int(row["n_at_upper"]),
                int(sum(np.asarray(v).size for v in result.parameters.values())),
            )


# ===========================================================================
# 7.  No first-iteration spike
# ===========================================================================


UNBOUNDED_MAX_MW = 1.0e7


def write_unbounded_dataset(root: Path) -> Path:
    """The planning fixture with ``p_nom_max = inf`` on every extendable row.

    ``expansion.py`` is the one place allowed to invent an upper bound, and it
    only does so where PyPSA left ``p_nom_max = +inf``.  The checked-in fixture
    gives every extendable row a finite maximum, which would make the spike test
    vacuous, so this variant removes them.
    """
    root = write_planning_dataset(root)
    static = root / "static"
    for fname in ("generators.csv", "storage_units.csv"):
        frame = pd.read_csv(static / fname, index_col=0)
        extendable = frame["p_nom_extendable"].astype(bool)
        frame.loc[extendable, "p_nom_max"] = np.inf
        frame.to_csv(static / fname)
    # No `convert_dataset`: only the static tables changed, and `expansion.py`
    # reads those straight off disk -- `weather.zarr` is untouched.
    return root


@unittest.skipUnless(PLANNING_AVAILABLE, TASK_A)
class TestNoFirstIterationSpike(unittest.TestCase):
    """Test 7: from as-built, with the invented bound effectively removed.

    Deviation from the spec's wording, recorded here rather than hidden: the
    spec asks for ``upper_bounds = inf``, but ``parameters.setup_bounds``
    deliberately refuses a non-finite upper bound (inventing one is
    ``expansion.py``'s job and only its job).  ``max_capacity_mw: 1e7`` on a
    fixture whose largest row is 700 MW is unbounded for every practical
    purpose: it is four orders of magnitude above anything the LP builds, so a
    rule that spikes has room to spike.
    """

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.dataset = write_unbounded_dataset(Path(cls._tmp.name) / "tiny")
        # Demand scaled up so the as-built system is SHORT and shedding at
        # VOLL, which is the state the spec describes: that is where the
        # gradient on a firm row is -1e5...-1e7 $/MW-yr and where a rule whose
        # step is proportional to the gradient blows up.  `clip_scale_to_one`
        # has to be off, or the scaling clips at 1.0 (LESSONS 2026-09-08).
        cls.loaded = load_system(
            cls.dataset,
            LoadOptions(
                years=(2020,),
                window=HourWindow(0, N_HOURS),
                voll=VOLL,
                demand_scaling="fixed",
                scale_load=3.0,
                clip_scale_to_one=False,
            ),
        )
        # The reference design: the single-level LP over the same unbounded box.
        cls.lp = planning.plan(cls.loaded, cls._config(planning={"method": "monolithic"}))

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    @classmethod
    def _config(cls, **overrides):
        cfg = plan_config(
            cls.dataset,
            planning={
                "expansion": {"mode": "pypsa", "max_capacity_mw": UNBOUNDED_MAX_MW},
                "warm_start": {"enabled": False},
            },
        )
        for key, value in overrides.items():
            cfg[key] = deep_merge(cfg[key], value) if isinstance(cfg.get(key), dict) else value
        return cfg

    def _flat(self, result) -> np.ndarray:
        return np.concatenate(
            [
                np.asarray(result.parameters[k], dtype=float).reshape(-1)
                for k in sorted(result.parameters)
            ]
        )

    def _run(self, optimizer: dict):
        return planning.plan(
            self.loaded,
            self._config(planning=deep_merge(GRADIENT, {"optimizer": optimizer})),
        )

    def test_upper_bounds_are_effectively_unbounded(self):
        # Only the extendable rows: the fixture's links are frozen, so their
        # upper bound is (correctly) their as-built capacity.
        for param in ("generator_capacity", "storageunit_power"):
            values = np.asarray(self.lp.upper_bounds[param], dtype=float).reshape(-1)
            self.assertTrue(
                np.any(values >= UNBOUNDED_MAX_MW - 1e-6),
                f"{param}: the fixture kept a finite invented bound",
            )

    def _as_built(self, result) -> np.ndarray:
        return np.concatenate(
            [
                np.asarray(result.meta["initial_parameters"][k], dtype=float).reshape(-1)
                for k in sorted(result.parameters)
            ]
        )

    def test_one_adam_iteration_moves_at_most_max_step(self):
        lr = 5.0
        result = self._run({"rule": "adam", "step_size": lr, "num_iterations": 1})
        moved = self._flat(result) - self._as_built(result)
        self.assertLessEqual(float(np.max(np.abs(moved))), 3.0 * lr + 1e-6)
        # ... and it did move: the test would pass vacuously on a no-op rule.
        self.assertGreater(float(np.max(np.abs(moved))), 0.0)

    def test_twenty_adam_iterations_stay_near_the_lp_optimum(self):
        lr = 5.0
        result = self._run({"rule": "adam", "step_size": lr, "num_iterations": 20})
        reference = np.maximum(self._flat(self.lp), self._as_built(result))
        ratio = self._flat(result) / np.maximum(reference, 1e-9)
        self.assertLessEqual(float(np.max(ratio)), 3.0, f"ratios {ratio}")

    def test_unclipped_gradient_rule_does_spike(self):
        """The negative control: the defect this rule replaces.

        ``clip`` is set out of the way so the units defect is isolated -- with
        ``step_size`` in MW^2/$ the step is proportional to the gradient, and
        from a short system shedding at VOLL the gradient on a firm row is
        -1e5...-1e7 $/MW-yr.  (With the shipped ``clip = 5e3`` the cap does bind
        on this fixture and hides the spike; on ca2040_z4 it binds too, which is
        the other half of the same defect -- the step is then a constant that
        has nothing to do with the problem.)
        """
        result = self._run(
            {"rule": "gradient", "step_size": 0.2, "clip": 1.0e12, "num_iterations": 1}
        )
        reference = np.maximum(self._flat(self.lp), self._as_built(result))
        ratio = self._flat(result) / np.maximum(reference, 1e-9)
        self.assertGreater(float(np.max(ratio)), 10.0, f"ratios {ratio}")

        # ... and the shipped clip does not save it: at `clip = 5e3` the same
        # single iteration still overbuilds a row by 5.5x, which is past the 3x
        # criterion the Adam run has to satisfy over twenty iterations.
        clipped = self._run(
            {"rule": "gradient", "step_size": 0.2, "clip": 5.0e3, "num_iterations": 1}
        )
        clipped_ratio = self._flat(clipped) / np.maximum(reference, 1e-9)
        self.assertGreater(float(np.max(clipped_ratio)), 3.0, f"ratios {clipped_ratio}")


# ===========================================================================
# Config surface
# ===========================================================================


@unittest.skipUnless(PLANNING_AVAILABLE, TASK_A)
class TestStepRuleConfig(unittest.TestCase):
    """``config.validate`` on ``planning.optimizer``."""

    def _validate(self, optimizer: dict):
        from experiments.ra import config as config_mod

        cfg = config_mod.base_config()
        cfg["mode"] = "plan"
        cfg["planning"]["method"] = "gradient"
        cfg["planning"]["optimizer"] = deep_merge(cfg["planning"]["optimizer"], optimizer)
        config_mod.validate(cfg)
        return cfg

    def test_defaults_validate(self):
        self._validate({})

    def test_unknown_rule_is_rejected(self):
        with self.assertRaises(ConfigError):
            self._validate({"rule": "nesterov"})

    def test_unknown_decay_is_rejected(self):
        with self.assertRaises(ConfigError):
            self._validate({"lr_decay": "exponential"})

    def test_ml_default_eps_is_allowed_but_zero_is_not(self):
        self._validate({"rule": "adam", "step_size": 200.0, "adam": {"eps": 1e-8}})
        with self.assertRaises(ConfigError):
            self._validate({"rule": "adam", "step_size": 200.0, "adam": {"eps": 0.0}})

    def test_a_gradient_step_size_under_adam_is_rejected(self):
        """0.2 is fine for either rule; 5e5 is a `gradient` setting, not an lr."""
        self._validate({"rule": "adam", "step_size": 0.2})
        with self.assertRaises(ConfigError):
            self._validate({"rule": "adam", "step_size": 5.0e5})

    def test_trust_region_rejects_a_minibatch(self):
        with self.assertRaises(ConfigError):
            self._validate(
                {
                    "rule": "trust_region",
                    "batch_size": 4,
                    "design_selection": "best_checkpointed",
                    "checkpoint_every": 20,
                }
            )

    def test_minibatch_plateau_needs_checkpoints(self):
        with self.assertRaises(ConfigError):
            self._validate(
                {
                    "rule": "adam",
                    "step_size": 200.0,
                    "batch_size": 4,
                    "design_selection": "best_checkpointed",
                    "checkpoint_every": 0,
                    "stopping": {"tol_rel_objective": 1e-4},
                }
            )

    def test_tol_window_must_admit_two_halves(self):
        with self.assertRaises(ConfigError):
            self._validate({"stopping": {"tol_window": 1}})
        with self.assertRaises(ConfigError):
            self._validate({"stopping": {"checkpoint_window": 1}})

    def test_campaign_configs_validate(self):
        from experiments.ra import config as config_mod
        from experiments.ra import identity

        root = ZAP_REPO_ROOT / "experiments" / "ra" / "configs" / "experiments"
        names = [
            "plan_z4_2020_c4b_adam",
            "plan_z4_2020_c5b_adam",
            "plan_z4_2020_a1_adam_lr200",
            "plan_z4_2020_a2_adam_lr50",
            "plan_z4_2020_a3_trust_region",
            "plan_z4_2020_a4_capex_scaled",
        ]
        ids = set()
        for name in names:
            cfg = config_mod.load_config(root / f"{name}.yaml")
            config_mod.validate(cfg)
            self.assertIn(
                cfg["planning"]["optimizer"]["rule"], ("adam", "trust_region", "capex_scaled")
            )
            ids.add(identity.run_id(cfg))
        self.assertEqual(len(ids), len(names))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
