"""Adequacy / evaluation plots R1-R7.

All of phase B (D8): the evaluation pipeline that fills ``eval.parquet`` over a
multi-year, multi-draw campaign is not built yet, and R6 additionally waits on
the PRAS reference run.  Their schemas are declared here so phase B is a body
change and the columns are reviewable today.
"""

from __future__ import annotations

from . import phase_b

phase_b(
    "R1",
    title="Cost vs reliability frontier",
    tier="report",
    needs=("eval",),
    columns=(
        "design_id",
        "label",
        "formulation",
        "heuristic",
        "selection_strategy",
        "emissions_mode",
        "capex_bn_usd",
        "opex_bn_usd",
        "voll_ens_bn_usd",
        "total_cost_bn_usd",
        "eue_mwh",
        "lolh_hours",
        "lolp",
        "is_holdout",
    ),
)

phase_b(
    "R2",
    title="Total system cost by formulation",
    tier="report",
    needs=("eval",),
    columns=(
        "design_id",
        "label",
        "formulation",
        "heuristic",
        "split",
        "n_year_draws",
        "mean_total_cost_bn_usd",
        "p05_total_cost_bn_usd",
        "p95_total_cost_bn_usd",
    ),
)

phase_b(
    "R3",
    title="Unserved energy distribution across weather years and draws",
    tier="report",
    needs=("eval",),
    columns=(
        "design_id",
        "label",
        "split",
        "quantile",
        "eue_mwh",
        "lolh_hours",
        "n_year_draws",
    ),
)

phase_b(
    "R4",
    title="Available capacity at the tightest hour",
    tier="report",
    needs=("eval",),
    columns=(
        "design_id",
        "label",
        "year",
        "draw",
        "min_available_gw",
        "p5_available_gw",
        "peak_net_load_gw",
        "headroom_gw",
    ),
)

phase_b(
    "R5",
    title="When unserved energy happens (day x hour heat map)",
    tier="report",
    needs=("eval",),
    columns=(
        "design_id",
        "label",
        "day_of_year",
        "hour_of_day",
        "mean_ens_mwh",
        "p95_ens_mwh",
        "n_draws",
        "n_year_draws_with_ens",
    ),
)

phase_b(
    "R6",
    title="Block-dispatch evaluation vs the PRAS reference",
    tier="report",
    needs=("eval",),
    columns=(
        "design_id",
        "label",
        "metric",
        "block_dispatch_value",
        "pras_value",
        "dev_abs",
        "dev_rel",
        "n_year_draws",
    ),
)

phase_b(
    "R7",
    title="Train vs held-out generalisation gap",
    tier="report",
    needs=("eval",),
    columns=(
        "design_id",
        "label",
        "formulation",
        "metric",
        "train_value",
        "holdout_value",
        "gap_abs",
        "gap_rel",
    ),
)
