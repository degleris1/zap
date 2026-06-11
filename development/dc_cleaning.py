"""Shared bad-bus manifest helpers for the corrected DC-placement studies."""

from __future__ import annotations

import json
import os
from typing import Iterable


DEFAULT_BAD_BUSES_JSON = "development/results/cleaning/bad_buses_490_load1p0_25.json"


def load_bad_buses(path: str) -> list[int]:
    """Load a bad-bus manifest stored either as a list or as {"bad_buses": [...]}."""
    with open(os.path.expanduser(path)) as f:
        data = json.load(f)
    bad = data.get("bad_buses", data) if isinstance(data, dict) else data
    if not isinstance(bad, list):
        raise ValueError(f"Bad-bus manifest {path} must contain a list of bus ids")
    return sorted(int(x) for x in bad)


def resolve_bad_buses(path: str | None, detected: Iterable[int] | None = None) -> list[int]:
    """Use a frozen manifest when supplied; otherwise use a freshly detected list."""
    if path:
        return load_bad_buses(path)
    if detected is None:
        raise ValueError("Either bad_buses_json or detected bad buses must be supplied")
    return sorted(int(x) for x in detected)


def cleaning_metadata(path: str | None, bad: Iterable[int]) -> dict:
    """Common JSON fields that make cited runs auditable."""
    bad = sorted(int(x) for x in bad)
    return {
        "bad_buses_json": path,
        "bad_buses": bad,
        "n_bad": len(bad),
    }
