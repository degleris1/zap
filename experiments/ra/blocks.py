"""Block construction: pure functions over hour indices.

A :class:`Block` is a contiguous half-open range of hours *within a weather
year*, ``[start, stop)``, indexed from 0 at the start of the year.  Blocks are
solved independently with cyclic storage (``initial_soc == final_soc``), which
``load_system`` sets from ``system.storage_init_soc``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

#: Attributes that would carry an annual energy budget if a dataset ever had
#: one. ``ca2040_z4`` has ``e_sum_min = -inf`` / ``e_sum_max = +inf`` on every
#: generator, so pro-rating is inert there (spec section 5.6).
ENERGY_BUDGET_ATTRS = (
    "energy_budget_min",
    "energy_budget_max",
    "e_sum_min",
    "e_sum_max",
    "annual_energy_min",
    "annual_energy_max",
)

REFERENCE = "reference"


@dataclass(frozen=True)
class Block:
    index: int
    year: int
    start: int
    stop: int

    @property
    def hours(self) -> int:
        return self.stop - self.start

    def to_dict(self) -> dict:
        return {"index": self.index, "year": self.year, "start": self.start, "stop": self.stop}


def window_bounds(cfg: dict) -> tuple[int, int]:
    win = cfg["dataset"]["window"]
    return int(win["start"]), int(win["stop"])


def reference_bounds(cfg: dict) -> tuple[int, int] | None:
    """Hour bounds of the reference solve, or None when there is no reference."""
    sel = cfg["selection"]
    start, stop = window_bounds(cfg)
    mode = sel["reference"]
    if mode == "none":
        return None
    if mode == "full_year":
        return start, stop
    ref = sel["reference_window"]
    return int(ref["start"]), int(ref["start"]) + int(ref["hours"])


def make_blocks(cfg: dict, block_size: int | str) -> list[Block]:
    """Blocks for one block size, in deterministic order (year, then index)."""
    start, stop = window_bounds(cfg)
    years = list(cfg["dataset"]["years"])

    if block_size == REFERENCE:
        bounds = reference_bounds(cfg)
        if bounds is None:
            return []
        return [Block(index=0, year=y, start=bounds[0], stop=bounds[1]) for y in years]

    size = int(block_size)
    if size <= 0:
        raise ValueError(f"block size must be positive, got {block_size!r}")

    n_hours = stop - start
    n_blocks = n_hours // size
    remainder = n_hours - n_blocks * size
    if remainder:
        logger.warning(
            "dropping the final %d hour(s) of the window: %d hours is not a multiple of %d",
            remainder,
            n_hours,
            size,
        )

    return [
        Block(index=i, year=year, start=start + i * size, stop=start + (i + 1) * size)
        for year in years
        for i in range(n_blocks)
    ]


def block_is_inside(block: Block, bounds: tuple[int, int]) -> bool:
    """True if the block lies entirely inside ``bounds`` (half-open)."""
    return bounds[0] <= block.start and block.stop <= bounds[1]


def block_overlaps(block: Block, bounds: tuple[int, int]) -> bool:
    return block.start < bounds[1] and bounds[0] < block.stop


def prorate_energy_budgets(devices: list, block_hours: int, year_hours: int) -> list:
    """Scale any annual energy budget to the length of a block.

    A no-op on ``ca2040_z4``: none of its devices carries an energy budget
    attribute (spec section 5.6 / section 8.2).  The pass exists so that a
    dataset that *does* carry one cannot slip through unscaled.
    """
    if year_hours <= 0:
        raise ValueError(f"year_hours must be positive, got {year_hours}")
    factor = block_hours / year_hours

    touched = []
    for device in devices:
        for attr in ENERGY_BUDGET_ATTRS:
            value = getattr(device, attr, None)
            if value is None:
                continue
            setattr(device, attr, value * factor)
            touched.append(f"{type(device).__name__}.{attr}")

    if not touched:
        logger.debug("no energy budgets to pro-rate")
    else:  # pragma: no cover - no dataset in phase 1 exercises this
        logger.info("pro-rated energy budgets by %.6g: %s", factor, ", ".join(touched))
    return devices
