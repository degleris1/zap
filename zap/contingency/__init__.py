"""Contingency analysis for ``zap``: linearized (DC) distribution factors.

PTDF / LODF built directly from zap's AC-line devices.  Post-contingency line
flows are an explicit linear map of nodal injections, mathematically equivalent
to the extensive scenario-replication dispatch in
``PowerNetwork.model_contingency_problem`` (preventive DC) but ~base-case size.
See ``zap/tests/test_contingency_lodf.py`` for the exact-equivalence test that
certifies this against zap's own contingency solver.
"""

from .ptdf import (
    find_ac_line,
    build_signed_incidence,
    branch_susceptance,
    thermal_limits,
    connected_components,
    build_ptdf,
)
from .lodf import build_phi, build_lodf, post_contingency_flows, sparsify_lodf

__all__ = [
    # primary
    "build_ptdf",
    "build_lodf",
    "post_contingency_flows",
    # adapters / helpers
    "find_ac_line",
    "build_signed_incidence",
    "branch_susceptance",
    "thermal_limits",
    "connected_components",
    "build_phi",
    "sparsify_lodf",
]
