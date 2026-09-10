"""Verifier findings for the A1-A3 diff (2026-09-09).

Both tests were confirmed defects, landed as strict xfails. Both are fixed --
the price joins now carry `year` and `draw` (`metrics._price_join_keys`) -- so
the markers are gone and these are ordinary regression tests.
"""

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from experiments.ra import metrics, persist


def _base():
    return {
        "task_id": "t",
        "design_id": "asbuilt",
        "method": "lp",
        "block_size": "24",
        "year": 2020,
        "block_index": 0,
        "draw": None,
        "hour": 0,
        "quantity": metrics.PRICE_QUANTITY,
        "carrier": "",
        "bus": "b0",
        "name": "",
        "value": 0.0,
        "unit": "$/MWh",
    }


def _run(rows):
    run = Path(tempfile.mkdtemp())
    persist.write_parquet_atomic(pd.DataFrame(rows), run / "hourly.parquet", persist.hourly_schema())
    df = pd.DataFrame([{"task_id": r["task_id"], "status": "ok"} for r in rows])
    return run, df


def test_price_error_join_is_per_weather_year():
    """`hour` is the hour *within* a weather year, so (bus, hour) collides across
    years: before the fix every year's ADMM block joined the FIRST year's
    reference and same-block LP rows, and the 2021 row reported a delta of 70."""
    b = _base()
    rows = [
        {**b, "task_id": "ref20", "block_size": "reference", "year": 2020, "value": 30.0},
        {**b, "task_id": "ref21", "block_size": "reference", "year": 2021, "value": 100.0},
        {**b, "task_id": "a20", "method": "admm", "year": 2020, "value": 30.0},
        {**b, "task_id": "a21", "method": "admm", "year": 2021, "value": 100.0},
        {**b, "task_id": "l20", "method": "lp", "year": 2020, "value": 30.0},
        {**b, "task_id": "l21", "method": "lp", "year": 2021, "value": 100.0},
    ]
    run, df = _run(rows)
    out = metrics.price_error_rows(run, df).set_index("task_id")
    # Every ADMM block reproduces its own year's prices exactly: zero error.
    assert out.loc["a21", "delta_price"] == pytest.approx(0.0)
    assert out.loc["a21", "delta_price_vs_block_lp"] == pytest.approx(0.0)
    assert out.loc["a20", "delta_price"] == pytest.approx(0.0)
    assert out.loc["a20", "delta_price_vs_block_lp"] == pytest.approx(0.0)

    # ... and the block-level table is per year too.
    table = metrics.price_error_vs_reference(run, df)
    admm = table[table["method"] == "admm"]
    assert set(admm["year"]) == {2020, 2021}
    assert admm["max_abs_price_error"].max() == pytest.approx(0.0)


def test_price_error_block_lp_join_is_per_draw():
    """The same-block LP control must be the LP of the *same outage draw*: before
    the fix every ADMM draw was compared against draw 0's LP prices (delta 10).

    The reference here carries no draw, which is the other half of the rule: a
    draw-independent reference still joins every draw's blocks."""
    b = _base()
    rows = [
        {**b, "task_id": "ref", "block_size": "reference", "value": 30.0},
        {**b, "task_id": "a0", "method": "admm", "draw": 0, "value": 30.0},
        {**b, "task_id": "a1", "method": "admm", "draw": 1, "value": 40.0},
        {**b, "task_id": "l0", "method": "lp", "draw": 0, "value": 30.0},
        {**b, "task_id": "l1", "method": "lp", "draw": 1, "value": 40.0},
    ]
    run, df = _run(rows)
    out = metrics.price_error_rows(run, df).set_index("task_id")
    assert out.loc["a1", "delta_price_vs_block_lp"] == pytest.approx(0.0)
    assert out.loc["a0", "delta_price_vs_block_lp"] == pytest.approx(0.0)
    # The draw-less reference is shared, so draw 1 is genuinely 10 $/MWh from it.
    assert out.loc["a1", "delta_price"] == pytest.approx(10.0)
    assert out.loc["a0", "delta_price"] == pytest.approx(0.0)


def test_reference_with_draws_is_joined_per_draw():
    """When the reference itself is solved per draw, the join keys on draw too."""
    b = _base()
    rows = [
        {**b, "task_id": "ref0", "block_size": "reference", "draw": 0, "value": 30.0},
        {**b, "task_id": "ref1", "block_size": "reference", "draw": 1, "value": 40.0},
        {**b, "task_id": "a0", "method": "admm", "draw": 0, "value": 30.0},
        {**b, "task_id": "a1", "method": "admm", "draw": 1, "value": 40.0},
    ]
    run, df = _run(rows)
    out = metrics.price_error_rows(run, df).set_index("task_id")
    assert out.loc["a0", "delta_price"] == pytest.approx(0.0)
    assert out.loc["a1", "delta_price"] == pytest.approx(0.0)
