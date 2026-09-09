"""Period selection: which operational blocks the planning problem sees (D-W3).

The ``selection`` axis and the gradient batch strategy are two different levers.
``selection`` decides which blocks *exist* in the problem;
``planning.optimizer.batch_strategy`` decides which of them a gradient step
*sees*.  Phase 1 keeps them separate.

Phase 1 registers ``all | uniform | random | stratified`` as thin wrappers over
the ported sampler, and registers ``kmedoids`` / ``gradient_stress`` as classes
whose ``select()`` raises: the config axis exists and validates today, and
adding k-medoids is one new class plus one registry line.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import ClassVar

from ..config import ConfigError
from .base import selection_options


@dataclass(frozen=True)
class SelectionSpec:
    strategy: str
    block_size: int | None  # None = the whole loaded horizon
    num_blocks: int | None
    seed: int
    avoid_year_boundaries: bool = False

    def to_dict(self) -> dict:
        return {
            "strategy": self.strategy,
            "block_size": self.block_size,
            "num_blocks": self.num_blocks,
            "seed": self.seed,
            "avoid_year_boundaries": self.avoid_year_boundaries,
        }


class PeriodSelector(abc.ABC):
    name: ClassVar[str] = "abstract"

    def __init__(self, spec: SelectionSpec) -> None:
        self.spec = spec

    @abc.abstractmethod
    def select(self, sampler) -> list[tuple[int, int]]:  # pragma: no cover - abstract
        ...

    def weights(self, blocks) -> list[float] | None:
        """Subproblem weights, or None for uniform (D-W8)."""
        return None


class SamplerSelector(PeriodSelector):
    """``all | uniform | random | stratified`` -> ``sampler.sample_blocks(...)``."""

    name = "sampler"

    def select(self, sampler) -> list[tuple[int, int]]:
        spec = self.spec
        return sampler.sample_blocks(
            block_size=spec.block_size,
            num_blocks=spec.num_blocks,
            strategy=spec.strategy,
            avoid_year_boundaries=spec.avoid_year_boundaries,
            seed=spec.seed,
        )


class AllSelector(SamplerSelector):
    name = "all"


class UniformSelector(SamplerSelector):
    name = "uniform"


class RandomSelector(SamplerSelector):
    name = "random"


class StratifiedSelector(SamplerSelector):
    name = "stratified"


class KMedoidsSelector(PeriodSelector):
    name = "kmedoids"

    def select(self, sampler):
        raise NotImplementedError(
            "k-medoids period selection is phase-2 work; see memory/PROJECT.md 2.2 "
            "('Representative days/weeks are chosen by k-medoids clustering')."
        )


class GradientStressSelector(PeriodSelector):
    name = "gradient_stress"

    def select(self, sampler):
        raise NotImplementedError(
            "stress-period identification via gradient information is phase-2 work; "
            "see memory/PROJECT.md 2.2."
        )


SELECTORS: dict[str, type[PeriodSelector]] = {
    cls.name: cls
    for cls in (
        AllSelector,
        UniformSelector,
        RandomSelector,
        StratifiedSelector,
        KMedoidsSelector,
        GradientStressSelector,
    )
}

#: Strategies that are implemented today; the rest raise on ``select()``.
PHASE1_STRATEGIES = ("all", "uniform", "random", "stratified")
PHASE2_STRATEGIES = ("kmedoids", "gradient_stress")


def selection_spec(cfg: dict) -> SelectionSpec:
    sel = selection_options(cfg)
    block_size = sel["block_size"]
    if block_size is not None:
        block_size = int(block_size)
        if block_size <= 0:
            raise ConfigError(f"selection.block_size must be positive or null, got {block_size}")
    num_blocks = sel["num_blocks"]
    num_blocks = None if num_blocks is None else int(num_blocks)
    return SelectionSpec(
        strategy=str(sel["strategy"]),
        block_size=block_size,
        num_blocks=num_blocks,
        seed=int(sel["seed"]),
        avoid_year_boundaries=bool(sel["avoid_year_boundaries"]),
    )


def make_selector(cfg: dict, *, total_hours: int) -> PeriodSelector:
    """Build the configured selector, validating it against the loaded horizon."""
    spec = selection_spec(cfg)
    if spec.strategy not in SELECTORS:
        raise ConfigError(
            f"unknown selection.strategy {spec.strategy!r}; known strategies: {sorted(SELECTORS)}"
        )
    if spec.block_size is not None and spec.block_size > total_hours:
        raise ConfigError(
            f"selection.block_size ({spec.block_size}) exceeds the loaded horizon "
            f"({total_hours} hours)"
        )
    if spec.strategy in ("random", "stratified") and spec.num_blocks is None:
        raise ConfigError(f"selection.num_blocks is required for strategy {spec.strategy!r}")
    return SELECTORS[spec.strategy](spec)
