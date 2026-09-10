"""Verifier reproductions for the WP-E0/E2/E3/E4 evaluator diff (2026-09-09).

Each test here documents a defect found in adversarial review; it is written
to FAIL against the diff as reviewed and to pass once the defect is fixed.
Fixtures and helpers are borrowed from ``test_ra_evaluation``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ZAP_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(ZAP_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(ZAP_REPO_ROOT))

from experiments.ra import evaluate  # noqa: E402
from zap.tests.test_ra_evaluation import (  # noqa: E402,F401  (fixtures)
    dataset,
    designs,
    run_evaluate,
    write_config,
)


# ---------------------------------------------------------------------------
# F1. `execution` is excluded from the run-id hash, so a run dir accepts both
# granularities -- and `read_task_records` then yields every block twice.
# ---------------------------------------------------------------------------


def test_switching_task_granularity_in_one_run_dir_double_counts(tmp_path, dataset, designs):
    """Block-granularity files and case files for the same blocks both survive
    in ``tasks/`` and both are flattened into ``metrics.csv``: 16 rows for 8
    blocks, ``coverage == 2`` and every ``total_cost_usd`` NaN.  Resume-skip
    cannot see the other granularity's files, so nothing stops the second run.
    """
    design_paths, _loaded, _built = designs
    path = write_config(tmp_path, dataset, "gran", {"execution": {"task_granularity": "block"}})
    runs_root = tmp_path / "runs"

    code, run_dir = run_evaluate(tmp_path, path, design_paths.values(), runs_root=runs_root)
    assert code == 0
    assert len(pd.read_csv(run_dir / "metrics.csv")) == 8

    code, run_dir2 = run_evaluate(
        tmp_path,
        path,
        design_paths.values(),
        runs_root=runs_root,
        extra_args=["--set", "execution.task_granularity=case"],
    )
    assert code == 0
    assert run_dir2 == run_dir  # same hash: `execution` is excluded

    frame = pd.read_csv(run_dir / "metrics.csv")
    assert frame["task_id"].is_unique, "every block row appears twice"
    assert len(frame) == 8
    rows = pd.read_parquet(run_dir / "eval.parquet")
    assert (rows["coverage"] == 1.0).all()
    assert rows["total_cost_usd"].notna().all()


# ---------------------------------------------------------------------------
# F2. A case that lost a block (coverage < 1) is kept out of the *score* but
# still enters every reliability average with its truncated EUE / LOLH.
# ---------------------------------------------------------------------------


def _row(design_id: str, draw: int, *, coverage: float, eue: float, lolh: float, total) -> dict:
    return {
        "design_id": design_id,
        "source_run_id": "r",
        "formulation": "monolithic",
        "heuristic": "none",
        "selection_strategy": "all",
        "emissions_mode": "none",
        "method": "lp",
        "block_size": "168",
        "year": 2020,
        "draw": draw,
        "n_blocks": 52 if coverage == 1.0 else 51,
        "hours": 8736.0 * coverage,
        "window_hours": 8736.0,
        "coverage": coverage,
        "is_holdout": False,
        "operational_cost": 1.0e9 * coverage,
        "generation_cost": 1.0e9 * coverage,
        "voll_cost": eue * 1e4,
        "unserved_energy_mwh": eue,
        "lost_load_hours": lolh,
        "co2_tonnes": 1.0,
        "curtailment_mwh": 0.0,
        "imports_mwh": 0.0,
        "exports_mwh": 0.0,
        "demand_mwh": 1e8 * coverage,
        "capex_annual_usd": 5.0e8,
        "total_cost_usd": total,
        "eue_mwh": eue,
        "neue": eue / (1e8 * coverage),
        "lolh_hours": lolh,
        "lolh_frac": lolh / (8736.0 * coverage),
        "lol_any": bool(lolh > 0),
        "storage_cycles": 1.0,
        "min_available_mw": 1.0,
        "p5_available_mw": float("nan"),
        "mean_price_usd_per_mwh": 40.0,
        "max_price_usd_per_mwh": 50.0,
        "build_wall_clock_s": 1.0,
        "solve_wall_clock_s": 1.0,
        "wall_clock_s": 2.0,
    }


def test_partial_case_is_excluded_from_reliability_averages():
    """The partial case (draw 1) lost the block that shed: its recorded EUE is
    0 and its ``total_cost_usd`` is NaN by the coverage guard.  The summary
    correctly drops it from the score (``n_cases_scored == 1``) but averages it
    into ``eue_mwh_mean`` / ``lolp`` / ``lolh_mean`` / ``neue_mean`` as if it
    were a fully scored, perfectly reliable year.
    """
    rows = pd.DataFrame(
        [
            _row("d", 0, coverage=1.0, eue=10.0, lolh=5.0, total=1.5e9),
            _row("d", 1, coverage=51 / 52, eue=0.0, lolh=0.0, total=float("nan")),
        ],
        columns=list(evaluate.EVAL_COLUMNS),
    )
    summary = evaluate.eval_summary(rows, {"train": [2020], "heldout_eval": []})
    headline = summary[summary["split"] == "all"].iloc[0]
    assert int(headline["n_cases_scored"]) == 1
    # Every reliability statistic must be over the same case set as the score.
    assert headline["eue_mwh_mean"] == pytest.approx(10.0)
    assert headline["lolh_mean"] == pytest.approx(5.0)
    assert headline["lolp"] == pytest.approx(1.0)
    assert headline["neue_mean"] == pytest.approx(10.0 / 1e8)


# ---------------------------------------------------------------------------
# F3. A design without `objective.capex_annual` scores as opex only.
# ---------------------------------------------------------------------------


def test_missing_capex_yields_nan_total_cost_not_opex_only():
    """``_eval_rows`` substitutes 0 for a NaN capex, so a design whose record
    carries no ``objective.capex_annual`` (a hand-written as-built comparator,
    or any record the attribute lookup missed) gets ``total_cost_usd ==
    operational_cost`` and out-ranks every design that paid for its capacity.
    Spec 4.1: ``total_cost_usd = capex_annual + operational_cost``.
    """
    ok = pd.DataFrame(
        {
            "task_id": ["x-lp-168-y2020-b00000-d0"],
            "design_id": ["x"],
            "method": ["lp"],
            "block_size": ["168"],
            "year": [2020],
            "draw": [0],
            "status": ["ok"],
            "hours": [48.0],
            "operational_cost": [1.0e6],
            "unserved_energy_mwh": [0.0],
            "lost_load_hours": [0],
            "demand_mwh": [1.0e5],
        }
    )
    rows = evaluate._eval_rows(ok, attrs={}, holdout=set(), window_hours=48.0)
    assert np.isnan(float(rows["capex_annual_usd"].iloc[0]))
    assert np.isnan(float(rows["total_cost_usd"].iloc[0])), (
        "a missing capex must not silently become 0 in the criterion"
    )
