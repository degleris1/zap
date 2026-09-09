"""Filesystem locations used by the RA harness.

No absolute path is ever hard-coded: everything hangs off the location of the
installed ``zap`` package, with environment-variable overrides for Sherlock.
"""

from __future__ import annotations

import os
from pathlib import Path

import zap

#: ``.../CH3_SGD_Planning/zap`` -- the zap repository (submodule) root.
ZAP_ROOT = Path(zap.__file__).resolve().parent.parent

#: ``.../CH3_SGD_Planning`` -- the brain repository root.
BRAIN_ROOT = ZAP_ROOT.parent

CONFIG_ROOT = Path(__file__).resolve().parent / "configs"


def zap_root() -> Path:
    """Root of the zap repository (the directory containing the ``zap`` package)."""
    return ZAP_ROOT


def brain_root() -> Path:
    """Root of the CH3 brain repository (the parent of the zap submodule)."""
    return BRAIN_ROOT


def data_root() -> Path:
    """Directory holding the datasets (``ca2040_z4``, ``ca2040_county``, ...).

    Override with the ``CH3_DATA_DIR`` environment variable.
    """
    env = os.environ.get("CH3_DATA_DIR")
    if env:
        return Path(env).expanduser().resolve()
    return BRAIN_ROOT / "data"


def dataset_dir(name: str) -> Path:
    """Resolve a dataset name (or an explicit path) to a directory."""
    p = Path(name).expanduser()
    if p.is_absolute():
        return p
    return data_root() / name


def runs_root() -> Path:
    """Directory holding run directories.

    Override with the ``CH3_RUNS_DIR`` environment variable.
    """
    env = os.environ.get("CH3_RUNS_DIR")
    if env:
        return Path(env).expanduser().resolve()
    return ZAP_ROOT / "experiments" / "runs"


def run_dir(run_id: str, root: str | Path | None = None) -> Path:
    """Directory for a single run."""
    base = Path(root).expanduser() if root is not None else runs_root()
    return base / run_id


def config_root() -> Path:
    """Directory holding the checked-in configs of this harness."""
    return CONFIG_ROOT
