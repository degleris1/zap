"""Evaluation infrastructure: WP-E0, WP-E2, WP-E3, WP-E4.

Spec: ``memory/plans/2026-09-09-evaluation-spec.md``.  Everything here runs on
the hermetic 48-hour tiny fixture with a hand-written outage store (borrowed
from ``test_wy_store_design``), with the ``STUB`` solver wherever no dispatch is
needed.  The one check against ``data/ca2040_z4`` uses a **two-week** window and
skips when the dataset is absent.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

ZAP_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(ZAP_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(ZAP_REPO_ROOT))

from experiments.ra import cli, config, dispatch, evaluate, identity, metrics, paths, persist
from experiments.ra import system as system_mod
from experiments.ra import tasks as tasks_mod
from experiments.ra.config import ConfigError
from zap.importers.wy_store import (
    clear_unit_slice_cache,
    convert_dataset,
    unit_slice_cache_info,
)
from zap.tests.fixtures.tiny_dataset import write_tiny_dataset
from zap.tests.test_wy_store_design import (
    DRAWS,
    STATIC_P_NOM,
    _write_outage_store,
)

WINDOW = (0, 48)
YEAR = 2020


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def dataset(tmp_path_factory) -> Path:
    """A 48 h tiny dataset with the hand-written 168 h outage store beside it."""
    root = write_tiny_dataset(tmp_path_factory.mktemp("tiny") / "tiny", n_hours=48, years=(YEAR,))
    gens = pd.read_csv(root / "static" / "generators.csv", index_col=0)
    for name, value in STATIC_P_NOM.items():
        gens.loc[name, "p_nom"] = value
    gens.to_csv(root / "static" / "generators.csv")
    convert_dataset(root)
    _write_outage_store(root)
    return root


def write_config(tmp_path: Path, dataset: Path, name: str, extra: dict | None = None) -> Path:
    cfg = {
        "includes": [str(paths.config_root() / "base.yaml")],
        "name": name,
        "dataset": {"dir": str(dataset), "years": [YEAR], "window": {"start": 0, "stop": 48}},
        "system": {"demand_scaling": "none", "export_mode": "drop"},
        "selection": {"blocks": [24], "reference": "none"},
        "heuristics": {"outage_draws": [0, 1]},
        "methods": {"lp": {"enabled": True, "solver": "STUB", "timeout_s": 60}},
        "execution": {"task_granularity": "case"},
    }
    if extra:
        cfg = config.deep_merge(cfg, extra)
    path = tmp_path / f"{name}.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


def as_built_capacities(cfg: dict) -> dict[str, np.ndarray]:
    loaded = system_mod.build_system(cfg, cache=False)
    out = {}
    for cls_name, attr in (("Generator", "nominal_capacity"), ("StorageUnit", "power_capacity")):
        device = loaded.devices[loaded.index.device_index[cls_name]]
        out[cls_name] = np.asarray(getattr(device, attr), dtype=float).reshape(-1)
    return out, loaded


def design_record(
    cfg: dict,
    loaded,
    capacities: dict[str, np.ndarray],
    *,
    design_id: str,
    run_id: str = "plan-fixture",
    capex_annual: float = 1.0e6,
    initial: dict[str, np.ndarray] | None = None,
    dataset_name: str | None = None,
) -> dict:
    """A minimal, valid ``design.json`` document for the tiny fixture."""
    attrs = {"Generator": "nominal_capacity", "StorageUnit": "power_capacity"}
    params = {"Generator": "generator_capacity", "StorageUnit": "storageunit_power"}
    initial = initial if initial is not None else {}
    return {
        "schema_version": 1,
        "design_id": design_id,
        "run_id": run_id,
        "zap_commit": "0" * 40,
        "method": "monolithic",
        "preset": "monolithic",
        "kind": "primal",
        "dataset": dataset_name or str(cfg["dataset"]["dir"]),
        "years": list(cfg["dataset"]["years"]),
        "window": dict(cfg["dataset"]["window"]),
        "heuristics": {"name": "none"},
        "selection": {"strategy": "all", "block_size": 24, "blocks": []},
        "capacities": {
            cls_name: {
                "names": [str(n) for n in loaded.index.names[cls_name]],
                attrs[cls_name]: [float(v) for v in values],
            }
            for cls_name, values in capacities.items()
        },
        "initial_parameters": {
            params[cls_name]: [float(v) for v in values] for cls_name, values in initial.items()
        },
        "objective": {"capex_annual": capex_annual, "annual": capex_annual},
        "emissions": {"mode": "none"},
        "solver": {"status": "optimal"},
        "timing": {},
    }


def write_design(path: Path, record: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2))
    return path


@pytest.fixture
def designs(tmp_path, dataset):
    """Two designs on the tiny fixture: as-built, and a bigger one."""
    cfg = config.load_config(write_config(tmp_path, dataset, "designs"))
    built, loaded = as_built_capacities(cfg)
    small = design_record(cfg, loaded, built, design_id="d_small", initial=built)
    bigger = {k: v.copy() for k, v in built.items()}
    bigger["Generator"] = bigger["Generator"] * 2.0
    big = design_record(
        cfg, loaded, bigger, design_id="d_big", initial=built, capex_annual=2.0e6
    )
    paths_ = {
        "d_small": write_design(tmp_path / "src" / "designs" / "d_small.json", small),
        "d_big": write_design(tmp_path / "src" / "designs" / "d_big.json", big),
    }
    return paths_, loaded, built


def run_evaluate(tmp_path, config_path, design_paths, *, runs_root=None, extra_args=()):
    runs_root = runs_root or (tmp_path / "runs")
    args = ["evaluate", "--config", str(config_path), "--runs-root", str(runs_root)]
    for path in design_paths:
        args += ["--design-file", str(path)]
    code = cli.main([*args, *extra_args])
    cfg = config.load_config(config_path)
    return code, paths.run_dir(identity.run_id(cfg), runs_root)


# ===========================================================================
# WP-E0 -- unit-slice cache, empty ENS skip, demand_mwh
# ===========================================================================


def test_unit_slice_cache_is_reused_within_a_build_and_across_designs(dataset, tmp_path):
    """Spec 5.8: two cases differing only in the design reuse the slice."""
    cfg = config.load_config(write_config(tmp_path, dataset, "cache"))
    built, loaded = as_built_capacities(cfg)
    clear_unit_slice_cache()

    system_mod.clear_system_cache()
    system_mod.build_system(cfg, draw=0, cache=False)
    info = unit_slice_cache_info()
    # One decompression for generators, a hit for storage.
    assert info["misses"] == 1
    assert info["hits"] == 1
    assert info["size"] == 1

    # Same (year, draw), different design: the slice is reused, the availability
    # arrays are not (they are re-weighted per design).
    design = system_mod.Design(design_id="bigger", capacities={"Generator": built["Generator"] * 3})
    designed = system_mod.build_system(cfg, draw=0, design=design, cache=False)
    assert unit_slice_cache_info()["misses"] == 1
    assert unit_slice_cache_info()["hits"] == 3

    base_avail = np.asarray(loaded.devices[loaded.index.device_index["Generator"]].dynamic_capacity)
    new_avail = np.asarray(
        designed.devices[designed.index.device_index["Generator"]].dynamic_capacity
    )
    assert not np.array_equal(base_avail, new_avail), "a new design must re-weight the units"

    # A different draw is a different key: a miss, and the cache still holds one.
    system_mod.build_system(cfg, draw=1, cache=False)
    assert unit_slice_cache_info()["misses"] == 2
    assert unit_slice_cache_info()["size"] == 1
    assert unit_slice_cache_info()["key"][2] == 1  # (path, year, draw, start, stop)


def test_unit_slice_is_read_only(dataset, tmp_path):
    cfg = config.load_config(write_config(tmp_path, dataset, "readonly"))
    clear_unit_slice_cache()
    system_mod.build_system(cfg, draw=0, cache=False)
    from zap.importers.wy_store import _UNIT_SLICE_CACHE

    (array,) = list(_UNIT_SLICE_CACHE.values())
    with pytest.raises(ValueError):
        array[0, 0] = 0


def test_common_random_numbers_across_designs(dataset, tmp_path):
    """Spec 5.5: the *unit* realisations are shared, not the row means."""
    from zap.importers.wy_store import _open_outage_store, _unit_pool_from_store

    cfg = config.load_config(write_config(tmp_path, dataset, "crn"))
    built, loaded = as_built_capacities(cfg)
    row_names = list(loaded.index.names["Generator"])

    base = system_mod.build_system(cfg, draw=0, cache=False)
    bigger = {k: v.copy() for k, v in built.items()}
    bigger["Generator"][row_names.index("z1 CCGT")] = 750.0
    other = system_mod.build_system(
        cfg,
        draw=0,
        design=system_mod.Design(design_id="crn", capacities=bigger),
        cache=False,
    )

    a = np.asarray(base.devices[base.index.device_index["Generator"]].dynamic_capacity)
    b = np.asarray(other.devices[other.index.device_index["Generator"]].dynamic_capacity)
    for i, name in enumerate(row_names):
        if built["Generator"][i] == bigger["Generator"][i]:
            np.testing.assert_array_equal(a[i], b[i], err_msg=f"row {name} moved")
    changed = row_names.index("z1 CCGT")
    assert not np.array_equal(a[changed], b[changed])

    # And the shared leading units really are the same realisations.
    _, root = _open_outage_store(dataset)
    pool = _unit_pool_from_store(root)
    up = np.asarray(root["available"][0, list(DRAWS).index(0), :48, :], dtype=np.uint8).T
    offset = pool.row_offset["z1 CCGT"]
    # as-built 250 MW = 1 unit; designed 750 MW = 3 units, sharing unit 0.
    np.testing.assert_array_equal(a[changed], up[offset, :48])
    assert b[changed].min() < 1.0


def test_empty_ens_profile_writes_no_file(dataset, tmp_path):
    """WP-E0 / spec 5.9: no shedding, no parquet -- and no change downstream."""
    path = write_config(
        tmp_path,
        dataset,
        "noens",
        {"methods": {"lp": {"solver": "HIGHS", "timeout_s": 300}}, "heuristics": {"outage_draws": []}},
    )
    cfg = config.load_config(path)
    runs_root = tmp_path / "runs"
    assert cli.main(["run", "--config", str(path), "--runs-root", str(runs_root)]) == 0
    run_dir = paths.run_dir(identity.run_id(cfg), runs_root)

    frame = pd.read_csv(run_dir / "metrics.csv")
    assert float(frame["unserved_energy_mwh"].sum()) == pytest.approx(0.0, abs=1e-6)
    assert not (run_dir / "ens_profile").exists(), "a block that shed nothing wrote a file"
    assert frame["ens_profile_path"].isna().all()

    evaluate.write_eval_tables(run_dir)
    profile = pd.read_parquet(run_dir / "eval_ens_profile.parquet")
    assert profile.empty


def test_write_ens_profile_returns_none_on_an_empty_frame(dataset, tmp_path, monkeypatch):
    cfg = config.load_config(
        write_config(tmp_path, dataset, "ensnone", {"heuristics": {"outage_draws": []}})
    )
    from experiments.ra import blocks as blocks_mod

    loaded = system_mod.build_system(cfg, cache=False)
    block = blocks_mod.make_blocks(cfg, 24)[0]
    devices = dispatch.slice_devices(loaded, cfg, block)
    task = tasks_mod.Task(task_id="t", method="lp", block=block, block_size=24)
    monkeypatch.setattr(
        persist,
        "build_ens_profile_frame",
        lambda *a, **k: pd.DataFrame({c: pd.Series(dtype="object") for c in persist.ENS_PROFILE_COLUMNS}),
    )
    out = persist.write_ens_profile(tmp_path / "nowhere", task, cfg, loaded, devices, None, block)
    assert out is None
    assert not (tmp_path / "nowhere" / "ens_profile").exists()


def test_demand_mwh_is_the_block_gross_demand(dataset, tmp_path):
    """WP-E0 / E6: `demand_mwh` sums to the fixture's known demand."""
    from experiments.ra import blocks as blocks_mod

    path = write_config(
        tmp_path,
        dataset,
        "demand",
        {"methods": {"lp": {"solver": "HIGHS", "timeout_s": 300}}, "heuristics": {"outage_draws": []}},
    )
    cfg = config.load_config(path)
    loaded = system_mod.build_system(cfg, cache=False)

    totals = []
    expected = 0.0
    for block in blocks_mod.make_blocks(cfg, 24):
        devices = dispatch.slice_devices(loaded, cfg, block)
        outcome = loaded.network.dispatch(devices, time_horizon=block.hours, solver="HIGHS")
        m = metrics.block_metrics(loaded, devices, outcome, block)
        totals.append(m["demand_mwh"])
        for device in devices:
            if type(device).__name__ == "Load":
                expected += float(
                    (
                        np.asarray(device.load, dtype=float)
                        * np.asarray(device.nominal_capacity, dtype=float)
                    ).sum()
                )
    assert sum(totals) == pytest.approx(expected, rel=1e-12)
    assert "demand_mwh" in metrics.ADDITIVE_METRICS
    # The whole 48 h window solved as one block gives the same total: additive.
    whole = blocks_mod.Block(index=0, year=YEAR, start=0, stop=48)
    devices = dispatch.slice_devices(loaded, cfg, whole)
    outcome = loaded.network.dispatch(devices, time_horizon=48, solver="HIGHS")
    assert metrics.block_metrics(loaded, devices, outcome, whole)["demand_mwh"] == pytest.approx(
        sum(totals), rel=1e-12
    )


# ===========================================================================
# WP-E3 -- case granularity
# ===========================================================================


VOLATILE_COLUMNS = (
    "wall_clock_s",
    "solve_wall_clock_s",
    "build_wall_clock_s",
    "started_utc",
    "finished_utc",
)


def test_case_enumeration_is_design_inner(tmp_path, dataset):
    cfg = config.load_config(write_config(tmp_path, dataset, "order"))
    cases = tasks_mod.enumerate_tasks(cfg, design_ids=("a", "b"))
    assert [c.task_id for c in cases] == [
        "a-lp-24-y2020-d0",
        "b-lp-24-y2020-d0",
        "a-lp-24-y2020-d1",
        "b-lp-24-y2020-d1",
    ]
    assert all(len(c.tasks) == 2 for c in cases)  # 48 h / 24 h
    assert cases[0].tasks[0].task_id == "a-lp-24-y2020-b00000-d0"


def test_case_and_block_granularity_are_equivalent(tmp_path, dataset, designs):
    """Spec 5.7: identical metrics.csv rows and identical eval.parquet."""
    design_paths, _loaded, _built = designs
    frames = {}
    for granularity in ("block", "case"):
        path = write_config(
            tmp_path,
            dataset,
            f"equiv_{granularity}",
            {"execution": {"task_granularity": granularity}},
        )
        code, run_dir = run_evaluate(
            tmp_path, path, design_paths.values(), runs_root=tmp_path / f"runs_{granularity}"
        )
        assert code == 0
        frames[granularity] = (
            pd.read_csv(run_dir / "metrics.csv"),
            pd.read_parquet(run_dir / "eval.parquet"),
            run_dir,
        )

    block_csv, block_eval, block_dir = frames["block"]
    case_csv, case_eval, case_dir = frames["case"]

    # One file per block vs one per case.
    assert len(list((block_dir / "tasks").glob("*.json"))) == 8
    assert len(list((case_dir / "tasks").glob("*.json"))) == 4

    def normalise(frame):
        frame = frame.drop(columns=[c for c in VOLATILE_COLUMNS if c in frame.columns])
        return (
            frame.sort_values("task_id").reset_index(drop=True)
            if "task_id" in frame.columns
            else frame
        )

    a, b = normalise(block_csv), normalise(case_csv)
    assert list(a.columns) == list(b.columns)
    pd.testing.assert_frame_equal(a, b, check_like=True)

    def normalise_eval(frame):
        keep = [c for c in frame.columns if c not in VOLATILE_COLUMNS]
        return (
            frame[keep]
            .sort_values(["design_id", "year", "draw"])
            .reset_index(drop=True)
        )

    pd.testing.assert_frame_equal(normalise_eval(block_eval), normalise_eval(case_eval))


def test_a_run_dir_keeps_the_granularity_it_already_has(tmp_path, dataset, designs):
    """Verifier F1: `execution` is not in the run-id hash, so one directory can be
    entered at either granularity; the files already there must win, or every
    block ends up counted twice.
    """
    design_paths, _loaded, _built = designs
    path = write_config(tmp_path, dataset, "adopt", {"execution": {"task_granularity": "block"}})
    runs_root = tmp_path / "runs"
    code, run_dir = run_evaluate(tmp_path, path, design_paths.values(), runs_root=runs_root)
    assert code == 0
    assert len(list((run_dir / "tasks").glob("*.json"))) == 8
    resolved = yaml.safe_load((run_dir / "config.resolved.yaml").read_text())
    assert resolved["execution"]["task_granularity"] == "block"

    code, run_dir2 = run_evaluate(
        tmp_path,
        path,
        design_paths.values(),
        runs_root=runs_root,
        extra_args=["--set", "execution.task_granularity=case"],
    )
    assert code == 0 and run_dir2 == run_dir
    # No case files were written, nothing was re-solved, nothing is doubled.
    assert len(list((run_dir / "tasks").glob("*.json"))) == 8
    frame = pd.read_csv(run_dir / "metrics.csv")
    assert frame["task_id"].is_unique
    assert len(frame) == 8
    rows = pd.read_parquet(run_dir / "eval.parquet")
    assert (rows["coverage"] == 1.0).all()

    # An empty directory takes what the config asks for.
    cfg = config.load_config(path)
    cfg["execution"]["task_granularity"] = "case"
    fresh = tmp_path / "fresh"
    (fresh / "tasks").mkdir(parents=True)
    assert cli.adopt_task_granularity(cfg, fresh) == "case"
    assert cfg["execution"]["task_granularity"] == "case"


def test_mixed_granularity_ledger_refuses_to_aggregate(tmp_path, dataset, designs):
    """The backstop under the guard: a directory that already holds both kinds
    of task file must not silently produce doubled sums."""
    design_paths, _loaded, _built = designs
    path = write_config(tmp_path, dataset, "mixed")
    code, run_dir = run_evaluate(tmp_path, path, design_paths.values())
    assert code == 0

    case_path = run_dir / "tasks" / "d_small-lp-24-y2020-d0.json"
    record = json.loads(case_path.read_text())
    for block in record["blocks"]:  # write the per-block files beside the case
        (run_dir / "tasks" / f"{block['task_id']}.json").write_text(json.dumps(block))

    with pytest.raises(ValueError, match="more than once"):
        metrics.read_task_records(run_dir)


def test_as_built_comparator_opts_in_with_an_explicit_zero_capex(tmp_path, dataset, designs):
    """Verifier F3: a missing capex is unknown (NaN score); an explicit 0.0 is a
    number and scores as opex alone. The section 5.1 regression uses the latter.
    """
    _design_paths, loaded, built = designs
    path = write_config(tmp_path, dataset, "capexopt")
    cfg = config.load_config(path)

    explicit = design_record(cfg, loaded, built, design_id="d_zero", initial=built,
                             capex_annual=0.0)
    silent = design_record(cfg, loaded, built, design_id="d_missing", initial=built)
    silent["objective"].pop("capex_annual")
    files = [
        write_design(tmp_path / "capex" / "designs" / "d_zero.json", explicit),
        write_design(tmp_path / "capex" / "designs" / "d_missing.json", silent),
    ]
    code, run_dir = run_evaluate(tmp_path, path, files)
    assert code == 0

    rows = pd.read_parquet(run_dir / "eval.parquet").set_index("design_id")
    zero = rows.loc["d_zero"].iloc[0] if isinstance(rows.loc["d_zero"], pd.DataFrame) else rows.loc["d_zero"]
    missing = (
        rows.loc["d_missing"].iloc[0]
        if isinstance(rows.loc["d_missing"], pd.DataFrame)
        else rows.loc["d_missing"]
    )
    assert float(zero["capex_annual_usd"]) == 0.0
    assert float(zero["total_cost_usd"]) == pytest.approx(float(zero["operational_cost"]))
    assert np.isnan(float(missing["capex_annual_usd"]))
    assert np.isnan(float(missing["total_cost_usd"])), "unknown capex must not score as 0"

    # ... and the unscorable design is not ranked ahead of the scored one.
    summary = pd.read_parquet(run_dir / "eval_summary.parquet")
    headline = summary[summary["split"] == "all"].set_index("design_id")
    assert int(headline.loc["d_missing", "n_cases_scored"]) == 0
    assert np.isnan(float(headline.loc["d_missing", "score_usd"]))
    assert float(headline.loc["d_zero", "rank_score"]) == 1.0


def test_a_failing_middle_block_marks_the_case_and_keeps_the_other_rows(
    tmp_path, dataset, designs, monkeypatch
):
    design_paths, _loaded, _built = designs
    path = write_config(
        tmp_path, dataset, "midfail", {"dataset": {"window": {"start": 0, "stop": 48}}}
    )
    real = dispatch.solve_block

    def flaky(task, cfg_, design=None, **kwargs):
        if task.block.index == 1 and task.draw == 0 and task.design_id == "d_small":
            raise RuntimeError("synthetic middle-block failure")
        return real(task, cfg_, design=design, **kwargs)

    monkeypatch.setattr(dispatch, "solve_block", flaky)
    code, run_dir = run_evaluate(tmp_path, path, design_paths.values())
    assert code == 1  # a required method failed

    record = json.loads((run_dir / "tasks" / "d_small-lp-24-y2020-d0.json").read_text())
    assert record["status"] == "failed"
    assert len(record["blocks"]) == 2
    assert [b["status"] for b in record["blocks"]] == ["ok", "failed"]

    frame = pd.read_csv(run_dir / "metrics.csv")
    assert len(frame) == 8  # every block row survives
    assert int((frame["status"] == "ok").sum()) == 7
    # The failed block is excluded from the case's numbers, and the case is
    # then short of the window: no total cost.
    rows = pd.read_parquet(run_dir / "eval.parquet")
    partial = rows[(rows["design_id"] == "d_small") & (rows["draw"] == 0)]
    assert float(partial["coverage"].iloc[0]) == pytest.approx(0.5)
    assert np.isnan(float(partial["total_cost_usd"].iloc[0]))
    excluded = pd.read_csv(run_dir / "eval_excluded.csv")
    assert len(excluded) == 1
    assert tuple(excluded.columns) == evaluate.EVAL_EXCLUDED_COLUMNS
    assert excluded["design_id"].iloc[0] == "d_small"
    assert "## Excluded cases" in (run_dir / "CARD.md").read_text()


def test_resume_skips_a_finished_case(tmp_path, dataset, designs, monkeypatch):
    design_paths, _loaded, _built = designs
    path = write_config(tmp_path, dataset, "resume")
    calls = {"n": 0}
    real = dispatch.solve_block

    def counting(task, cfg_, design=None, **kwargs):
        calls["n"] += 1
        return real(task, cfg_, design=design, **kwargs)

    monkeypatch.setattr(dispatch, "solve_block", counting)
    code, run_dir = run_evaluate(tmp_path, path, design_paths.values(), extra_args=["--shard", "1/2"])
    assert code == 0
    first = calls["n"]
    assert first == 4  # two cases x two blocks
    mtimes = {p.name: p.stat().st_mtime_ns for p in (run_dir / "tasks").glob("*.json")}

    code, run_dir = run_evaluate(tmp_path, path, design_paths.values())
    assert code == 0
    assert calls["n"] == 8  # only the missing cases were solved
    for name, mtime in mtimes.items():
        assert (run_dir / "tasks" / name).stat().st_mtime_ns == mtime


# ===========================================================================
# WP-E2 -- `ra evaluate`, preflight, SOURCES.json
# ===========================================================================


def test_evaluate_end_to_end_stub(tmp_path, dataset, designs):
    """2 designs x 2 draws x 2 blocks through the command."""
    design_paths, _loaded, _built = designs
    path = write_config(tmp_path, dataset, "e2e")
    code, run_dir = run_evaluate(tmp_path, path, design_paths.values())
    assert code == 0

    assert len(list((run_dir / "tasks").glob("*.json"))) == 4
    frame = pd.read_csv(run_dir / "metrics.csv")
    assert len(frame) == 8
    assert set(frame["design_id"]) == {"d_small", "d_big"}

    rows = pd.read_parquet(run_dir / "eval.parquet")
    assert tuple(rows.columns) == evaluate.EVAL_COLUMNS
    assert len(rows) == 4
    assert set(rows["draw"]) == {0, 1}
    assert (rows["coverage"] == 1.0).all()
    assert rows["total_cost_usd"].notna().all()
    assert (rows["neue"] >= 0).all()

    summary = pd.read_parquet(run_dir / "eval_summary.parquet")
    assert tuple(summary.columns) == evaluate.EVAL_SUMMARY_COLUMNS
    assert set(summary["split"]) <= set(evaluate.EVAL_SPLITS)
    headline = summary[summary["split"] == "all"].set_index("design_id")
    assert sorted(headline["rank_score"]) == [1.0, 2.0]
    # score = capex + mean(operational cost)
    for design_id, row in headline.iterrows():
        cases = rows[rows["design_id"] == design_id]
        assert row["score_usd"] == pytest.approx(
            float(cases["capex_annual_usd"].iloc[0]) + float(cases["operational_cost"].mean())
        )

    sources = json.loads((run_dir / "designs" / "SOURCES.json").read_text())
    assert {d["design_id"] for d in sources["designs"]} == {"d_small", "d_big"}
    assert all(len(d["design_json_sha256"]) == 64 for d in sources["designs"])
    assert (run_dir / "designs" / "d_small.json").exists()

    report = json.loads((run_dir / "preflight.json").read_text())
    assert report["ok"]
    assert {c["name"] for c in report["checks"]} >= {
        "mode_is_dispatch",
        "ucap_off",
        "draws_present",
        "outage_store",
        "pool_capacity",
        "as_built_match",
    }

    card = (run_dir / "CARD.md").read_text()
    for heading in (
        "## Designs scored",
        "## Evaluation set",
        "## Ranking (split: all)",
        "## Held-out",
        "## Excluded cases",
        "## Preflight",
        "## Evaluation compute",
    ):
        assert heading in card, heading
    assert "d_small" in card

    env = json.loads((run_dir / "env.json").read_text())
    assert env["splits"]["splits_path"].endswith("splits.yaml")
    assert env["execution"]["task_granularity"] == "case"


def test_design_set_is_additive(tmp_path, dataset, designs):
    """D4: adding a design re-runs only its cases and covers both afterwards."""
    design_paths, _loaded, _built = designs
    path = write_config(tmp_path, dataset, "additive")
    runs_root = tmp_path / "runs"
    code, run_dir = run_evaluate(tmp_path, path, [design_paths["d_small"]], runs_root=runs_root)
    assert code == 0
    assert len(list((run_dir / "tasks").glob("*.json"))) == 2
    mtimes = {p.name: p.stat().st_mtime_ns for p in (run_dir / "tasks").glob("*.json")}

    code, run_dir2 = run_evaluate(
        tmp_path, path, design_paths.values(), runs_root=runs_root
    )
    assert code == 0
    assert run_dir2 == run_dir  # the design set is not in the run id
    for name, mtime in mtimes.items():
        assert (run_dir / "tasks" / name).stat().st_mtime_ns == mtime
    rows = pd.read_parquet(run_dir / "eval.parquet")
    assert set(rows["design_id"]) == {"d_small", "d_big"}


def test_duplicate_design_id_with_a_different_file_raises(tmp_path, dataset, designs):
    design_paths, loaded, built = designs
    path = write_config(tmp_path, dataset, "dupe")
    code, _run_dir = run_evaluate(tmp_path, path, [design_paths["d_small"]])
    assert code == 0

    cfg = config.load_config(path)
    other = design_record(cfg, loaded, built, design_id="d_small", capex_annual=9.9e9, initial=built)
    other_path = write_design(tmp_path / "other" / "designs" / "d_small.json", other)
    with pytest.raises(ConfigError, match="already been scored"):
        run_evaluate(tmp_path, path, [other_path])

    # The identical file is a no-op, not an error.
    code, _ = run_evaluate(tmp_path, path, [design_paths["d_small"]])
    assert code == 0


def test_foreign_dataset_design_raises(tmp_path, dataset, designs):
    _design_paths, loaded, built = designs
    path = write_config(tmp_path, dataset, "foreign")
    cfg = config.load_config(path)
    record = design_record(
        cfg, loaded, built, design_id="d_foreign", initial=built, dataset_name="ca2040_county"
    )
    foreign = write_design(tmp_path / "foreign" / "designs" / "d_foreign.json", record)
    with pytest.raises(ConfigError, match="built on dataset"):
        run_evaluate(tmp_path, path, [foreign])


def test_missing_draw_aborts_before_the_first_task(tmp_path, dataset, designs):
    design_paths, _loaded, _built = designs
    path = write_config(tmp_path, dataset, "missdraw", {"heuristics": {"outage_draws": [0, 7]}})
    with pytest.raises(ConfigError, match="preflight failed"):
        run_evaluate(tmp_path, path, design_paths.values())
    cfg = config.load_config(path)
    run_dir = paths.run_dir(identity.run_id(cfg), tmp_path / "runs")
    assert not (run_dir / "tasks").exists() or not list((run_dir / "tasks").glob("*.json"))
    report = json.loads((run_dir / "preflight.json").read_text())
    store = next(c for c in report["checks"] if c["name"] == "outage_store")
    assert not store["ok"]
    assert store["missing"][0]["draw"] == 7


def test_plan_mode_config_raises(tmp_path, dataset, designs):
    design_paths, _loaded, _built = designs
    path = write_config(
        tmp_path,
        dataset,
        "planmode",
        {"mode": "plan", "planning": {"method": "monolithic"}, "heuristics": {"outage_draws": [0]}},
    )
    with pytest.raises(ConfigError, match="mode: dispatch"):
        run_evaluate(tmp_path, path, design_paths.values())


def test_ucap_derate_raises(tmp_path, dataset, designs):
    design_paths, _loaded, _built = designs
    path = write_config(
        tmp_path,
        dataset,
        "ucap",
        {"heuristics": {"name": "ucap_derate", "ucap_derate": True, "outage_draws": []}},
        )
    with pytest.raises(ConfigError, match="preflight failed"):
        run_evaluate(tmp_path, path, design_paths.values())


def test_no_draws_needs_the_opt_in(tmp_path, dataset, designs):
    design_paths, _loaded, _built = designs
    path = write_config(tmp_path, dataset, "nodraws", {"heuristics": {"outage_draws": []}})
    with pytest.raises(ConfigError, match="preflight failed"):
        run_evaluate(tmp_path, path, design_paths.values())

    allowed = write_config(
        tmp_path,
        dataset,
        "nodraws_ok",
        {"heuristics": {"outage_draws": []}, "evaluation": {"allow_no_draws": True}},
    )
    code, run_dir = run_evaluate(tmp_path, allowed, design_paths.values())
    assert code == 0
    assert len(pd.read_parquet(run_dir / "eval.parquet")) == 2


def test_pool_capacity_failure_is_caught_by_preflight(tmp_path, dataset, designs):
    """A design that outgrows its slice of the pool never reaches a task."""
    _design_paths, loaded, built = designs
    path = write_config(tmp_path, dataset, "poolcap")
    cfg = config.load_config(path)
    huge = {k: v.copy() for k, v in built.items()}
    names = list(loaded.index.names["Generator"])
    huge["Generator"][names.index("z1 CCGT")] = 1.0e6  # 4,000 units of 250 MW
    record = design_record(cfg, loaded, huge, design_id="d_huge", initial=built)
    huge_path = write_design(tmp_path / "huge" / "designs" / "d_huge.json", record)

    with pytest.raises(ConfigError, match="preflight failed"):
        run_evaluate(tmp_path, path, [huge_path])
    run_dir = paths.run_dir(identity.run_id(cfg), tmp_path / "runs")
    report = json.loads((run_dir / "preflight.json").read_text())
    pool = next(c for c in report["checks"] if c["name"] == "pool_capacity")
    assert not pool["ok"]
    assert pool["failures"][0]["row"] == "z1 CCGT"
    assert pool["worst_ratio"] > 1.0


def test_as_built_mismatch_is_caught_by_preflight(tmp_path, dataset, designs):
    """D2 / spec 5.4: initial_parameters must equal a fresh as-built load."""
    _design_paths, loaded, built = designs
    path = write_config(tmp_path, dataset, "asbuilt")
    cfg = config.load_config(path)
    wrong = {k: v.copy() for k, v in built.items()}
    wrong["Generator"] = wrong["Generator"] + 1.0
    record = design_record(cfg, loaded, built, design_id="d_wrong", initial=wrong)
    wrong_path = write_design(tmp_path / "wrong" / "designs" / "d_wrong.json", record)
    with pytest.raises(ConfigError, match="preflight failed"):
        run_evaluate(tmp_path, path, [wrong_path])
    run_dir = paths.run_dir(identity.run_id(cfg), tmp_path / "runs")
    report = json.loads((run_dir / "preflight.json").read_text())
    check = next(c for c in report["checks"] if c["name"] == "as_built_match")
    assert not check["ok"]
    assert check["mismatches"][0]["class"] == "Generator"


def test_design_run_picks_up_every_design(tmp_path, dataset, designs):
    design_paths, _loaded, _built = designs
    path = write_config(tmp_path, dataset, "designrun")
    source_run = design_paths["d_small"].parent.parent
    code = cli.main(
        [
            "evaluate",
            "--config",
            str(path),
            "--runs-root",
            str(tmp_path / "runs"),
            "--design-run",
            str(source_run),
        ]
    )
    assert code == 0
    cfg = config.load_config(path)
    run_dir = paths.run_dir(identity.run_id(cfg), tmp_path / "runs")
    assert set(pd.read_parquet(run_dir / "eval.parquet")["design_id"]) == {"d_small", "d_big"}


# ===========================================================================
# WP-E4 -- eval tables and the ranking
# ===========================================================================


def _summary_fixture(run_dir: Path, rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    return evaluate.eval_summary(frame, evaluate.read_splits(run_dir))


def test_score_se_and_quantiles_on_a_hand_computed_fixture(tmp_path):
    capex = 100.0
    ops = [10.0, 20.0, 30.0, 40.0]
    rows = [
        {
            "design_id": "d",
            "source_run_id": "r",
            "formulation": "monolithic",
            "heuristic": "none",
            "selection_strategy": "all",
            "method": "lp",
            "block_size": "168",
            "year": 2020,
            "draw": i,
            "is_holdout": False,
            "operational_cost": op,
            "generation_cost": op,
            "voll_cost": 0.0,
            "capex_annual_usd": capex,
            "total_cost_usd": capex + op,
            "eue_mwh": float(i),
            "neue": float(i) / 100.0,
            "lolh_hours": float(i),
            "lolh_frac": float(i) / 168.0,
            "lol_any": bool(i > 0),
            "co2_tonnes": 1.0,
            "curtailment_mwh": 0.0,
            "min_available_mw": 100.0 - i,
            "p5_available_mw": float("nan"),
        }
        for i, op in enumerate(ops)
    ]
    summary = _summary_fixture(tmp_path, rows)
    row = summary[summary["split"] == "all"].iloc[0]
    totals = np.array([capex + o for o in ops])
    assert row["score_usd"] == pytest.approx(totals.mean())
    assert row["score_se_usd"] == pytest.approx(totals.std(ddof=1) / np.sqrt(4))
    assert row["score_p50"] == pytest.approx(np.quantile(totals, 0.5))
    assert row["score_p95"] == pytest.approx(np.quantile(totals, 0.95))
    assert row["score_max"] == pytest.approx(totals.max())
    assert row["n_cases"] == 4
    assert row["n_cases_scored"] == 4
    assert row["eue_mwh_mean"] == pytest.approx(1.5)
    assert row["lolh_mean"] == pytest.approx(1.5)
    # LOLP is P(a case sheds at all) = 3/4, NOT lolh / hours.
    assert row["lolp"] == pytest.approx(0.75)
    assert "lolh_frac" in rows[0] and "lolh_frac" not in summary.columns
    assert row["min_available_mw_min"] == pytest.approx(97.0)
    assert np.isnan(row["p5_available_mw_mean"])


def test_summary_splits_follow_splits_yaml(tmp_path, dataset, designs):
    """Train / holdout come from the split file, with provenance in env.json."""
    splits = tmp_path / "split_file.yaml"
    splits.write_text(
        yaml.safe_dump(
            {
                "dataset_years": [2019, 2020],
                "heldout_eval": [2020],
                "train": [2019],
            }
        )
    )
    design_paths, _loaded, _built = designs
    path = write_config(
        tmp_path, dataset, "splitcfg", {"evaluation": {"splits_path": str(splits)}}
    )
    code, run_dir = run_evaluate(tmp_path, path, design_paths.values())
    assert code == 0

    env = json.loads((run_dir / "env.json").read_text())
    assert env["splits"]["heldout_eval"] == [2020]
    assert env["splits"]["splits_path"] == str(splits)
    assert len(env["splits"]["splits_sha256"]) == 64

    rows = pd.read_parquet(run_dir / "eval.parquet")
    assert rows["is_holdout"].all()  # every case is weather year 2020
    summary = pd.read_parquet(run_dir / "eval_summary.parquet")
    assert set(summary["split"]) == {"all", "holdout"}

    # E10: editing the split file afterwards does not reclassify the run.
    splits.write_text(yaml.safe_dump({"dataset_years": [2020], "heldout_eval": [], "train": [2020]}))
    evaluate.write_eval_tables(run_dir)
    assert pd.read_parquet(run_dir / "eval.parquet")["is_holdout"].all()


def test_coverage_guard_nans_a_partial_case(tmp_path, dataset, designs):
    design_paths, _loaded, _built = designs
    path = write_config(tmp_path, dataset, "coverage")
    code, run_dir = run_evaluate(
        tmp_path, path, design_paths.values(), extra_args=["--max-tasks", "1"]
    )
    assert code == 0
    # `--max-tasks 1` ran one case; every block of it is present, so it is
    # complete -- the guard fires only when a case is missing hours.
    rows = pd.read_parquet(run_dir / "eval.parquet")
    assert len(rows) == 1
    assert float(rows["window_hours"].iloc[0]) == 48.0
    assert float(rows["coverage"].iloc[0]) == pytest.approx(1.0)

    # Delete one block record from the case and re-aggregate.
    task_path = run_dir / "tasks" / "d_small-lp-24-y2020-d0.json"
    record = json.loads(task_path.read_text())
    record["blocks"] = record["blocks"][:1]
    task_path.write_text(json.dumps(record))
    metrics.aggregate(run_dir, config.load_config(path))
    evaluate.write_eval_tables(run_dir)
    rows = pd.read_parquet(run_dir / "eval.parquet")
    assert float(rows["coverage"].iloc[0]) == pytest.approx(0.5)
    assert np.isnan(float(rows["total_cost_usd"].iloc[0]))
    summary = pd.read_parquet(run_dir / "eval_summary.parquet")
    assert int(summary["n_cases"].iloc[0]) == 1
    assert int(summary["n_cases_scored"].iloc[0]) == 0


def test_mean_price_is_demand_weighted(tmp_path):
    """E6: the case price is weighted by each block's demand."""
    rows = pd.DataFrame(
        [
            {
                "task_id": "d-lp-24-y2020-b00000",
                "status": "ok",
                "design_id": "d",
                "method": "lp",
                "block_size": "24",
                "year": 2020,
                "draw": 0,
                "hours": 24,
                "operational_cost": 1.0,
                "demand_mwh": 100.0,
                "mean_price": 10.0,
                "max_price": 10.0,
            },
            {
                "task_id": "d-lp-24-y2020-b00001",
                "status": "ok",
                "design_id": "d",
                "method": "lp",
                "block_size": "24",
                "year": 2020,
                "draw": 0,
                "hours": 24,
                "operational_cost": 1.0,
                "demand_mwh": 300.0,
                "mean_price": 50.0,
                "max_price": 50.0,
            },
        ]
    )
    out = evaluate._eval_rows(rows, {}, set(), window_hours=48.0)
    assert len(out) == 1
    assert float(out["mean_price_usd_per_mwh"].iloc[0]) == pytest.approx(
        (10.0 * 100.0 + 50.0 * 300.0) / 400.0
    )
    assert float(out["demand_mwh"].iloc[0]) == pytest.approx(400.0)
    assert float(out["max_price_usd_per_mwh"].iloc[0]) == pytest.approx(50.0)


def test_neue_is_eue_over_demand(tmp_path, dataset, designs):
    design_paths, _loaded, _built = designs
    path = write_config(tmp_path, dataset, "neue")
    code, run_dir = run_evaluate(tmp_path, path, design_paths.values())
    assert code == 0
    rows = pd.read_parquet(run_dir / "eval.parquet")
    for _, row in rows.iterrows():
        assert row["neue"] == pytest.approx(row["eue_mwh"] / row["demand_mwh"])


# ===========================================================================
# Spec section 5 -- correctness checks that need a real solve
# ===========================================================================


def test_ens_is_monotone_in_capacity(tmp_path, dataset):
    """Spec 5.3: a component-wise larger design cannot shed more."""
    path = write_config(
        tmp_path,
        dataset,
        "monotone",
        {
            "methods": {"lp": {"solver": "HIGHS", "timeout_s": 300}},
            "system": {"demand_scaling": "fixed", "scale_load": 3.0},
            "heuristics": {"outage_draws": [0]},
        },
    )
    cfg = config.load_config(path)
    built, loaded = as_built_capacities(cfg)
    small = design_record(cfg, loaded, built, design_id="d_small", initial=built)
    bigger = {k: v.copy() for k, v in built.items()}
    bigger["Generator"] = bigger["Generator"] * 2.0
    bigger["StorageUnit"] = bigger["StorageUnit"] * 2.0
    big = design_record(cfg, loaded, bigger, design_id="d_big", initial=built)
    files = [
        write_design(tmp_path / "mono" / "designs" / "d_small.json", small),
        write_design(tmp_path / "mono" / "designs" / "d_big.json", big),
    ]
    code, run_dir = run_evaluate(tmp_path, path, files)
    assert code == 0

    rows = pd.read_parquet(run_dir / "eval.parquet").set_index("design_id")
    assert float(rows.loc["d_small", "eue_mwh"]) > 0.0, "the fixture must actually shed"
    # Spec 5.6: VOLL * ENS is already inside `operational_cost`; it is never
    # added again, and `voll_cost` reconciles with the shed energy.
    voll = float(cfg["system"]["voll"])
    for design_id in ("d_small", "d_big"):
        assert float(rows.loc[design_id, "voll_cost"]) == pytest.approx(
            voll * float(rows.loc[design_id, "eue_mwh"]), rel=1e-9
        )
        assert float(rows.loc[design_id, "total_cost_usd"]) == pytest.approx(
            float(rows.loc[design_id, "capex_annual_usd"])
            + float(rows.loc[design_id, "operational_cost"]),
            rel=1e-12,
        )
        assert float(rows.loc[design_id, "hours"]) == float(
            rows.loc[design_id, "window_hours"]
        )
    assert float(rows.loc["d_big", "eue_mwh"]) <= float(rows.loc["d_small", "eue_mwh"]) + 1e-6
    assert float(rows.loc["d_big", "operational_cost"]) <= float(
        rows.loc["d_small", "operational_cost"]
    ) + 1e-6


def test_asbuilt_design_reproduces_the_blocked_dispatch_on_z4(tmp_path):
    """Spec 5.1 (two weeks, not the year): the design path perturbs nothing.

    Scoring the **as-built** design through ``ra evaluate`` must reproduce the
    plain blocked-dispatch row of the same window, block for block, to solver
    precision.  A difference means the design path changed the system.
    """
    from experiments.ra.paths import dataset_dir

    z4 = dataset_dir("ca2040_z4")
    if not (z4 / "weather.zarr").exists():
        pytest.skip("data/ca2040_z4 is not present")

    common = {
        "dataset": {"dir": "ca2040_z4", "years": [2020], "window": {"start": 7, "stop": 343}},
        "system": {
            "demand_scaling": "none",
            "export_mode": "drop",
            "storage_soc_mode": "cyclic_free",
        },
        "selection": {"blocks": [168], "reference": "none"},
        "heuristics": {"outage_draws": []},
        "methods": {"lp": {"enabled": True, "solver": "HIGHS", "timeout_s": 900}},
        "evaluation": {"allow_no_draws": True},
        "execution": {"task_granularity": "case"},
    }
    runs_root = tmp_path / "runs"

    plain_cfg = config.deep_merge(
        {"includes": [str(paths.config_root() / "base.yaml")], "name": "z4_plain"}, common
    )
    plain_path = tmp_path / "z4_plain.yaml"
    plain_path.write_text(yaml.safe_dump(plain_cfg))
    assert cli.main(["run", "--config", str(plain_path), "--runs-root", str(runs_root)]) == 0
    plain_dir = paths.run_dir(identity.run_id(config.load_config(plain_path)), runs_root)
    plain = pd.read_csv(plain_dir / "metrics.csv").sort_values("block_index")

    cfg = config.load_config(plain_path)
    built, loaded = as_built_capacities(cfg)
    record = design_record(cfg, loaded, built, design_id="asbuilt_z4", initial=built,
                           capex_annual=0.0)
    design_path = write_design(tmp_path / "z4" / "designs" / "asbuilt_z4.json", record)

    eval_cfg = config.deep_merge(dict(plain_cfg), {"name": "z4_eval"})
    eval_path = tmp_path / "z4_eval.yaml"
    eval_path.write_text(yaml.safe_dump(eval_cfg))
    code, eval_dir = run_evaluate(tmp_path, eval_path, [design_path], runs_root=runs_root)
    assert code == 0
    scored = pd.read_csv(eval_dir / "metrics.csv").sort_values("block_index")

    assert len(plain) == len(scored) == 2
    for column in ("operational_cost", "generation_cost", "co2_tonnes", "demand_mwh"):
        np.testing.assert_allclose(
            scored[column].to_numpy(dtype=float),
            plain[column].to_numpy(dtype=float),
            rtol=1e-9,
            err_msg=f"{column} moved when the as-built design was applied",
        )
    assert float(scored["unserved_energy_mwh"].sum()) == pytest.approx(0.0, abs=1e-6)
    rows = pd.read_parquet(eval_dir / "eval.parquet")
    assert float(rows["coverage"].iloc[0]) == pytest.approx(1.0)
    assert float(rows["total_cost_usd"].iloc[0]) == pytest.approx(
        float(scored["operational_cost"].sum()), rel=1e-12
    )


def teardown_module(module):  # pragma: no cover - hygiene between test modules
    system_mod.clear_system_cache()
    clear_unit_slice_cache()
    shutil.rmtree(Path(__file__).parent / "__pycache__", ignore_errors=True)
