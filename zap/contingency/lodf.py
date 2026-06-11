"""LODF (line outage distribution factors) for zap's linearized DC model.

For preventive single-line outages (injections unchanged, generation fixed),
the post-contingency flow on line ``l`` after outaging line ``k`` is

    f_l^{(k)} = f_l + LODF[l, k] * f_k

with

    Phi       = PTDF @ A                         (L x L line-to-line sensitivity)
    LODF[l,k] = Phi[l, k] / (1 - Phi[k, k])      (l != k),   LODF[k, k] = -1.

``Phi[l, k]`` is the sensitivity of flow on line ``l`` to a unit power transfer
across the endpoints of line ``k`` (``PTDF`` applied to line ``k``'s injection
pattern, which is exactly column ``k`` of the signed incidence ``A``).

``1 - Phi[k, k] -> 0`` iff outaging ``k`` disconnects its endpoints (a
radial/bridge line); such lines are flagged and excluded from the *outaged*
contingency set (their failure is an islanding event, out of scope for a
thermal hosting study).  Parallel double-circuits keep ``Phi[k, k] < 1`` because
the partner provides a return path, so they are correctly retained.
"""

import numpy as np
import scipy.sparse as sp


def build_phi(ptdf, A) -> np.ndarray:
    """Line-to-line sensitivity matrix ``Phi = PTDF @ A`` (L x L)."""
    A = A.tocsc() if sp.issparse(A) else A
    return np.asarray(ptdf @ A)


def build_lodf(phi, eps_radial: float = 1e-6):
    """Build the dense LODF matrix and the radial-line mask from ``Phi``.

    Returns
    -------
    lodf : (L, L) array, ``lodf[l, k]`` = sensitivity of f_l to outage of k.
        Diagonal is set to -1. Columns of radial lines are set to 0 (they must
        not be used as contingencies; see ``is_radial``).
    is_radial : (L,) bool, True where outaging line k islands its endpoints.
    """
    phi = np.asarray(phi, dtype=float)
    diag = np.diag(phi).copy()
    denom = 1.0 - diag

    is_radial = np.abs(denom) < eps_radial

    # Avoid divide-by-zero; radial columns are zeroed out below.
    safe_denom = np.where(is_radial, 1.0, denom)
    lodf = phi / safe_denom[np.newaxis, :]  # broadcast over rows: divide col k by (1 - Phi[k,k])
    np.fill_diagonal(lodf, -1.0)
    lodf[:, is_radial] = 0.0  # never use radial lines as outages

    return lodf, is_radial


def post_contingency_flows(f_base, lodf, outage_set=None) -> np.ndarray:
    """Post-contingency flows ``f^{(k)}_l = f_l + LODF[l, k] f_k``.

    Parameters
    ----------
    f_base : (L,) or (L, T) base-case line flows.
    lodf : (L, L) LODF matrix.
    outage_set : optional iterable of contingency line indices ``k``. Defaults
        to all lines.

    Returns
    -------
    (L, K) array if ``f_base`` is 1-D, else (L, K, T): post-contingency flow on
    every monitored line ``l`` for each contingency ``k`` in ``outage_set``.
    """
    f_base = np.asarray(f_base, dtype=float)
    lodf = np.asarray(lodf, dtype=float)
    K = np.arange(lodf.shape[0]) if outage_set is None else np.asarray(outage_set)

    if f_base.ndim == 1:
        # f^{(k)}_l = f_l + LODF[l,k] * f_k
        return f_base[:, None] + lodf[:, K] * f_base[K][None, :]
    else:
        # (L, K, T): broadcast over time
        fk = f_base[K, :]  # (K, T)
        return f_base[:, None, :] + lodf[:, K][:, :, None] * fk[None, :, :]


def sparsify_lodf(lodf, eps: float = 1e-3) -> sp.csr_matrix:
    """Drop near-zero LODF entries (``|LODF| <= eps``) to a sparse matrix.

    Most line pairs are electrically distant (LODF ~ 0); keeping only
    significant couplings makes the N-1 constraint set tractable.  A dropped
    pair ``(l, k)`` can change ``f_l^{(k)}`` by at most ``eps * |f_k| <=
    eps * F_bar_k``; choose ``eps`` accordingly and certify with a dense
    re-check of the solution (see the lazy-generation loop in
    ``hosting_capacity``).
    """
    lodf = np.asarray(lodf, dtype=float)
    mask = np.abs(lodf) > eps
    out = sp.csr_matrix(np.where(mask, lodf, 0.0))
    out.eliminate_zeros()
    return out
