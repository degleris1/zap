"""``design.json`` — the evaluation seam (spec section 3.4).

A planning run writes ``runs/<run_id>/designs/<design_id>.json``; the evaluation
pipeline reads it back as an :class:`experiments.ra.system.Design` and applies it
to a freshly loaded system.  ``capacities`` is keyed by device *class* name so it
feeds ``Design.capacities`` unchanged, and ``names`` is the source-row order:
a design built on a different dataset must fail loudly, not silently mis-map.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ..identity import zap_commit, zap_dirty
from ..system import Design

SCHEMA_VERSION = 1

#: The device-class capacity attribute that ``Design.apply`` will write.
CAPACITY_ATTRS = ("nominal_capacity", "power_capacity")


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.reshape(-1).tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def serialize_history(history: dict) -> dict:
    """Serialize an optimization history for JSON output.

    Ported from ``experiments/multi_year/runner.py::_serialize_history``: scalar
    trackers (loss, grad_norm, ...) plus the ``"param"`` tracker whose entries
    are dicts of numpy arrays.

    ``trackers.track_loss`` returns ``J.cpu().detach().numpy()`` -- a 0-d
    ``np.ndarray``, not an ``np.floating`` -- so the ported coercion left it
    untouched and ``json.dumps`` raised on every gradient run.  0-d and
    single-element arrays become floats, larger ones become lists.
    """

    def _scalar(x):
        if isinstance(x, (int, float, np.floating, np.integer)):
            return float(x)
        if hasattr(x, "detach"):  # torch tensor
            x = x.detach().cpu().numpy()
        if isinstance(x, np.ndarray):
            return float(x.item()) if x.size == 1 else x.tolist()
        return x

    serialized: dict = {}
    for key, values in history.items():
        if key == "param":
            serialized[key] = [
                {
                    pname: pval.tolist() if hasattr(pval, "tolist") else pval
                    for pname, pval in snapshot.items()
                }
                for snapshot in values
            ]
        else:
            serialized[key] = [_scalar(x) for x in values]
    return serialized


def result_to_record(result) -> dict:
    """The ``design.json`` document for a :class:`PlanningResult`."""
    import datetime as _dt

    return {
        "schema_version": SCHEMA_VERSION,
        "design_id": result.design_id,
        "run_id": result.run_id,
        "zap_commit": zap_commit(),
        "zap_dirty": zap_dirty(),
        "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "method": result.method,
        "preset": result.preset,
        "kind": result.kind,
        "dataset": result.dataset,
        "years": _jsonable(result.years),
        "window": _jsonable(result.window),
        "heuristics": _jsonable(result.heuristics),
        "selection": _jsonable(result.selection),
        "annualization": _jsonable(result.annualization),
        "parameter_names": {k: list(v) for k, v in result.parameter_names.items()},
        "capacities": _jsonable(result.capacities),
        # The as-built capacities the design was expanded from, per parameter.
        # ``metrics.planning_metrics`` differences them against the design to
        # report ``capacity_added_mw``; keeping them on the record means a
        # design read back later can say the same thing.
        "initial_parameters": _jsonable((result.meta or {}).get("initial_parameters") or {}),
        "bounds": {
            param: {
                "lower": _jsonable(result.lower_bounds.get(param)),
                "upper": _jsonable(result.upper_bounds.get(param)),
            }
            for param in result.parameter_names
        },
        "objective": _jsonable(result.objective),
        "emissions": _jsonable(result.emissions),
        "solver": _jsonable(result.solver),
        "timing": _jsonable(result.timing),
        "compute": _jsonable(result.compute),
        "history_path": result.history_path,
    }


def write_result(result, run_dir) -> Path:
    """Write ``designs/<design_id>.json`` (and its history) under ``run_dir``."""
    run_dir = Path(run_dir)
    designs = run_dir / "designs"
    designs.mkdir(parents=True, exist_ok=True)

    if result.history:
        history_path = designs / f"{result.design_id}.history.json"
        history_path.write_text(json.dumps(serialize_history(result.history), indent=2))
        result.history_path = str(history_path.relative_to(run_dir))

    path = designs / f"{result.design_id}.json"
    path.write_text(json.dumps(result_to_record(result), indent=2))
    return path


def record_to_design(record: dict, system=None) -> Design:
    """Build a :class:`Design` from a ``design.json`` document.

    When ``system`` is given, the recorded row names are asserted equal to
    ``SystemIndex.names[cls]`` for every device class: a design built on a
    different dataset fails loudly instead of silently mis-mapping (3.4).
    """
    version = int(record.get("schema_version", 0))
    if version != SCHEMA_VERSION:
        raise ValueError(
            f"design.json schema_version {version} is not supported (expected {SCHEMA_VERSION})"
        )

    capacities: dict[str, Any] = {}
    for cls_name, entry in (record.get("capacities") or {}).items():
        values = None
        for attr in CAPACITY_ATTRS:
            if attr in entry:
                values = np.asarray(entry[attr], dtype=float)
                break
        if values is None:
            raise ValueError(f"design.json capacities['{cls_name}'] has none of {CAPACITY_ATTRS}")
        if system is not None:
            check_design_names(cls_name, entry.get("names"), system)
        capacities[cls_name] = values

    return Design(design_id=str(record.get("design_id", "design")), capacities=capacities)


def check_design_names(cls_name: str, names, system) -> None:
    """Assert a design's row names match the system's (3.4)."""
    index = getattr(system, "index", None)
    if index is None:
        raise ValueError("cannot validate design names: the system has no SystemIndex")
    if cls_name not in index.names:
        raise ValueError(
            f"design references device class {cls_name!r}, which this system does not have "
            f"({sorted(index.names)})"
        )
    if names is None:
        raise ValueError(f"design has no row names for {cls_name!r}; refusing to apply it")
    expected = [str(n) for n in index.names[cls_name]]
    got = [str(n) for n in names]
    if got != expected:
        raise ValueError(
            f"design row names for {cls_name!r} do not match the system's "
            f"({len(got)} vs {len(expected)} rows; first mismatch at index "
            f"{_first_mismatch(got, expected)}). This design was built on a different dataset."
        )


def _first_mismatch(a: list, b: list) -> int:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return min(len(a), len(b))


def result_to_design(result) -> Design:
    return record_to_design(result_to_record(result))


def read_design_record(path) -> dict:
    """The raw ``design.json`` document, for the run card."""
    return json.loads(Path(path).read_text())


def read_design(path, system=None) -> Design:
    """Read ``design.json`` into an :class:`experiments.ra.system.Design`."""
    return record_to_design(read_design_record(path), system=system)


def design_paths(run_dir) -> list[Path]:
    """``runs/<id>/designs/*.json``, excluding the history sidecars."""
    designs = Path(run_dir) / "designs"
    if not designs.is_dir():
        return []
    return sorted(p for p in designs.glob("*.json") if not p.name.endswith(".history.json"))


def read_history(run_dir, design_id: str) -> dict | None:
    path = Path(run_dir) / "designs" / f"{design_id}.history.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())
