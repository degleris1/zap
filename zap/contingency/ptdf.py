"""PTDF (power transfer distribution factors) for zap's linearized DC model.

Sign conventions (verified against ``zap/devices/transporter/ac_line.py`` and
``PowerNetwork.model_dispatch_problem``):

- AC-line flow ``f = power[1] = b * (theta_src - theta_sink)`` with effective
  branch susceptance ``b = susceptance * nominal_capacity`` (``ac_line.py:139``),
  i.e. ``f = diag(b) A^T theta`` for the *source-positive* signed incidence
  ``A = incidence[source] - incidence[sink]`` (shape ``N x L``).
- Nodal power balance ``sum_devices net_power == 0`` reduces, for the AC lines,
  to ``A f = p`` where ``p`` is the net nodal injection of *all other* devices
  (generators inject ``+g``, loads inject ``-load``, etc.).
- Hence ``B_bus theta = p`` with ``B_bus = A diag(b) A^T`` and
  ``f = PTDF p`` with ``PTDF = diag(b) A^T B_bus^{-1}`` (slack-reduced).

``PTDF p`` is independent of the slack/reference choice for any per-component
balanced injection (``1^T p = 0`` on each connected component) -- this is the
slack-invariance property exercised by the validation tests.
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components as _connected_components


def _as_vector(x) -> np.ndarray:
    """Collapse a zap ``make_dynamic`` (N, 1)/(N, T) array to a 1-D vector.

    Susceptance / nominal_capacity / max_power of an AC line are static, so any
    time axis is constant; we take the first column.
    """
    a = np.asarray(x, dtype=float)
    if a.ndim == 2:
        a = a[:, 0]
    return np.ascontiguousarray(a.ravel())


def find_ac_line(devices):
    """Return ``(index, device)`` of the (unique) AC-line device.

    The AC-line index differs by importer (2 in the toy networks, 3 in the
    PyPSA loader), so callers should detect it by type rather than hard-code it.
    """
    from zap.devices.transporter import ACLine

    matches = [(i, d) for i, d in enumerate(devices) if isinstance(d, ACLine)]
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one ACLine device, found {len(matches)} "
            f"(indices {[i for i, _ in matches]})"
        )
    return matches[0]


def build_signed_incidence(ac_line, num_nodes=None) -> sp.csc_matrix:
    """Source-positive signed node-branch incidence ``A = inc[src] - inc[sink]``.

    Shape ``(num_nodes, num_lines)``; column ``l`` has ``+1`` at the source bus
    and ``-1`` at the sink bus of line ``l``.
    """
    inc = ac_line.incidence_matrix  # list [source, sink], each (N, L) csc
    A = (inc[0] - inc[1]).tocsc()
    if num_nodes is not None and A.shape[0] != num_nodes:
        raise ValueError(f"incidence has {A.shape[0]} nodes, expected {num_nodes}")
    return A


def branch_susceptance(ac_line) -> np.ndarray:
    """Effective branch susceptance ``b = susceptance * nominal_capacity`` (L,).

    Matches ``b_pnom`` in ``ac_line.py`` (lines 139, 186, 240) -- the coefficient
    in ``f = b_pnom * (theta_src - theta_sink)``.
    """
    return _as_vector(ac_line.susceptance) * _as_vector(ac_line.nominal_capacity)


def thermal_limits(ac_line) -> np.ndarray:
    """Symmetric thermal limit ``F_bar = max_power * nominal_capacity`` (L,).

    ``max_power == capacity`` and ``min_power == -capacity`` for a PowerLine, so
    the binding constraint is ``|f_l| <= F_bar_l`` (slack is 0 in the PyPSA
    loader).
    """
    return _as_vector(ac_line.max_power) * _as_vector(ac_line.nominal_capacity)


def connected_components(A, b=None):
    """Connected components of the *electrical* AC graph.

    Returns ``(n_components, labels)`` with ``labels[node]`` the component id.
    Only lines with nonzero susceptance contribute connectivity: a branch with
    ``b == 0`` (e.g. an un-built expansion candidate carried as a ghost column)
    does not join its endpoints, and a bus with no live line is its own island.
    Pass ``b`` to restrict to the live subgraph; this is required for correct
    slack-reference counting (one reference per *electrical* component).
    """
    if b is not None:
        live = np.asarray(b, dtype=float) != 0.0
        A = A[:, live]
    absA = abs(A).tocsr()
    # node-node adjacency: two buses adjacent iff a live line joins them
    adj = absA @ absA.T
    n_comp, labels = _connected_components(adj, directed=False)
    return n_comp, np.asarray(labels)


def _reference_nodes(labels, n_comp):
    """One reference (slack) bus per connected component (the lowest index)."""
    refs = np.full(n_comp, -1, dtype=int)
    for node, comp in enumerate(labels):
        if refs[comp] < 0:
            refs[comp] = node
    return refs


def build_ptdf(A, b, ref=None, return_x=False):
    """Dense PTDF matrix ``(L, N)`` mapping nodal injection to line flow.

    ``f = PTDF @ p`` for any per-component balanced injection ``p``.

    Parameters
    ----------
    A : (N, L) sparse signed incidence (source - sink).
    b : (L,) effective branch susceptance ``susceptance * nominal_capacity``.
    ref : optional explicit reference bus. By default one slack per connected
        component is chosen automatically (required for multi-island networks).
    return_x : if True also return the embedded reactance-like matrix ``X``
        (N, N) with ``theta = X p``, zeroed on reference rows/cols.

    The network is small (hundreds of buses) so we use a dense solve for
    clarity and numerical robustness.
    """
    N, L = A.shape
    b = np.asarray(b, dtype=float).ravel()
    A = A.tocsc()

    B_bus = (A @ sp.diags(b) @ A.T).toarray()  # (N, N), symmetric PSD, singular

    n_comp, labels = connected_components(A, b)
    if ref is not None:
        refs = np.atleast_1d(ref).astype(int)
    else:
        refs = _reference_nodes(labels, n_comp)

    keep = np.setdiff1d(np.arange(N), refs)
    B_red = B_bus[np.ix_(keep, keep)]

    # X_red = B_red^{-1}; embed into full X with zero reference rows/cols.
    X = np.zeros((N, N), dtype=float)
    X_red = np.linalg.inv(B_red)
    X[np.ix_(keep, keep)] = X_red

    PTDF = (sp.diags(b) @ A.T).toarray() @ X  # (L, N)

    if return_x:
        return PTDF, X
    return PTDF
