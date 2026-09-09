"""Run identity: canonical config serialization, hashing, zap commit, run id.

``execution`` and ``output`` are excluded from the hash -- sharding and the
location of the runs directory do not change results -- while the zap commit is
injected under ``_zap_commit`` so that the same config at a different code
version is a different run.
"""

from __future__ import annotations

import functools
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
from typing import Any

from .paths import zap_root

EXCLUDED_FROM_HASH = ("execution", "output")


def canonical_json(cfg: dict) -> str:
    """Deterministic JSON: sorted keys, no whitespace, floats via ``repr``."""
    return json.dumps(cfg, sort_keys=True, separators=(",", ":"), default=_json_default)


def _json_default(obj: Any):  # pragma: no cover - defensive
    if isinstance(obj, (set, frozenset, tuple)):
        return list(obj)
    return str(obj)


@functools.lru_cache(maxsize=1)
def zap_commit() -> str:
    """``git rev-parse HEAD`` in the zap repository, or ``"unknown"``."""
    env = os.environ.get("CH3_ZAP_COMMIT")
    if env:
        return env
    return _git("rev-parse", "HEAD") or "unknown"


@functools.lru_cache(maxsize=1)
def zap_dirty() -> bool:
    """True if the zap working tree has uncommitted changes."""
    status = _git("status", "--porcelain")
    return bool(status)


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(zap_root()), *args],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - defensive
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip()


def hashable_config(cfg: dict) -> dict:
    """The part of the config that determines the result, plus the zap commit."""
    payload = {k: v for k, v in cfg.items() if k not in EXCLUDED_FROM_HASH}
    payload["_zap_commit"] = zap_commit()
    return payload


def config_hash(cfg: dict) -> str:
    blob = canonical_json(hashable_config(cfg)).encode("utf-8")
    return hashlib.blake2b(blob, digest_size=6).hexdigest()


def run_id(cfg: dict) -> str:
    return f"{cfg.get('name', 'unnamed')}-{config_hash(cfg)}"


def env_info(cfg: dict | None = None) -> dict:
    """Provenance of the machine and the software stack, written to ``env.json``."""
    info = {
        "zap_commit": zap_commit(),
        "zap_dirty": zap_dirty(),
        "hostname": socket.gethostname(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "slurm_cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
    }
    try:  # cvxpy is a hard dependency of zap, but keep env.json best-effort
        import cvxpy as cp

        info["cvxpy"] = cp.__version__
        info["cvxpy_installed_solvers"] = sorted(cp.installed_solvers())
    except Exception:  # noqa: BLE001 - env.json is best-effort provenance
        info["cvxpy"] = None
    try:
        import numpy as np
        import pandas as pd

        info["numpy"] = np.__version__
        info["pandas"] = pd.__version__
    except Exception:  # noqa: BLE001, S110 - env.json is best-effort provenance
        pass
    if cfg is not None:
        info["run_id"] = run_id(cfg)
        info["config_hash"] = config_hash(cfg)
        # `output` is excluded from the hash, so `config.resolved.yaml` stores it
        # at its defaults; the *effective* persistence flags are recorded here.
        info["output"] = dict(cfg.get("output") or {})
    return info
