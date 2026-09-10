"""Verifier reproductions for the S1-S5 reporting/guardrail diff (2026-09-09).

Finding V-S3-1: the design-level opex split is silently ``None`` on the ADMM
gradient method.  ``objectives.dispatch_cost_credit`` converts the dispatch
state to numpy but re-evaluates ``operation_cost`` on a ``copy.copy`` of the
*torchified* device, whose ``min_power`` / ``nominal_capacity`` are tensors, so
``Injector.operation_cost`` raises ``TypeError`` (numpy - Tensor).  The
``try/except`` in ``GradientMethod.solve`` demotes that to a warning and
``opex_gross_raw`` / ``opex_credit_raw`` come out ``None`` for every
``method: admm`` design, i.e. for the ADMM half of the campaign matrix.

Both tests below fail on the working tree at the time of writing and pass once
the credit is evaluated with the device's own array module (or the clone's
tensors are converted alongside the state).
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

from experiments.ra.planning import base as planning_base
from experiments.ra.planning import objectives
from zap.devices import Generator
from zap.importers.wy_store import HourWindow, LoadOptions, load_system
from zap.tests.test_ra_planning_methods import VOLL, write_planning_dataset
from zap.tests.test_ra_reporting_fixes import CREDIT_COST, CREDIT_ROW, tiny_plan_config


class TestCreditOnTorchifiedDevices(unittest.TestCase):
    """Minimal unit-level repro: no solve, one torchified Generator."""

    def test_dispatch_cost_credit_on_torchified_generator(self):
        T = 3
        gen = Generator(
            num_nodes=1,
            name=np.array(["g"]),
            terminal=np.array([0]),
            nominal_capacity=np.array([100.0]),
            dynamic_capacity=np.ones((1, T)),
            linear_cost=np.array([[-5.0]]),
        )
        expected = objectives.dispatch_cost_credit(
            [gen], [[np.full((1, T), 60.0)]], [[None]], [None]
        )
        self.assertAlmostEqual(expected, -5.0 * 60.0 * T)

        gen_t = gen.torchify(machine="cpu", dtype=torch.float32)
        # Raises TypeError on the working tree (numpy power - torch min_power).
        credit = objectives.dispatch_cost_credit(
            [gen_t], [[torch.full((1, T), 60.0)]], [[None]], [None]
        )
        self.assertAlmostEqual(credit, expected, places=3)


class TestAdmmDesignCarriesTheSplit(unittest.TestCase):
    """End-to-end: an ADMM design on the negative-cost fixture must report the split."""

    N_HOURS = 48

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.dataset = write_planning_dataset(Path(cls._tmp.name) / "tiny", n_hours=cls.N_HOURS)
        gens = pd.read_csv(cls.dataset / "static" / "generators.csv", index_col=0)
        gens.loc[CREDIT_ROW, "marginal_cost"] = CREDIT_COST
        gens.to_csv(cls.dataset / "static" / "generators.csv")
        cls.loaded = load_system(
            cls.dataset,
            LoadOptions(years=(2020,), window=HourWindow(0, cls.N_HOURS), voll=VOLL),
        )

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_admm_opex_split_is_reported(self):
        cfg = tiny_plan_config(
            self.dataset,
            self.N_HOURS,
            selection={"strategy": "all", "block_size": 24},
            planning={
                "method": "admm",
                "admm": {
                    "machine": "cpu",
                    "dtype": "float32",
                    "solver_kwargs": {"num_iterations": 200, "rho_power": 1.0, "verbose": 0},
                },
                "warm_start": {"enabled": False},
                "optimizer": {"num_iterations": 1, "batch_size": 0},
            },
        )
        with self.assertNoLogs("experiments.ra.planning.methods.gradient", level="WARNING"):
            result = planning_base.plan(self.loaded, cfg)
        obj = result.objective
        self.assertIsNotNone(obj["opex_credit_raw"], "ADMM design lost the opex split")
        self.assertLess(obj["opex_credit_raw"], 0.0)
        self.assertAlmostEqual(
            obj["opex_gross_raw"] + obj["opex_credit_raw"],
            obj["opex_raw"],
            delta=abs(obj["opex_raw"]) * 1e-6 + 1e-3,
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
