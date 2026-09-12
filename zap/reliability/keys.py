"""Unit-key schemes for the forced-outage sampler (outage-pool spec, D1/D2).

This module is the *isolated, versioned* part of the outage machinery. It answers
one question -- "which RNG streams back this source row's virtual units?" -- and
nothing else. Everything that depends on how a unit is identified lives here, so
adding a future ``identity-v1`` scheme (keys ``(row_name, k)``, once the 1:1
re-export lands) is a registry entry and no change anywhere else.

The key
-------
``slot-v1`` keys a virtual unit by ``(carrier, bus, ordinal, k)``:

* ``ordinal`` -- the 0-based index of the source row inside its
  ``(component, carrier, bus)`` group, in static-table order. Required because
  the import buses carry several rows of one carrier at one bus; without it
  those rows would either collide or have to share a namespace whose allocation
  depends on the other rows' *designed* capacity.
* ``k`` -- the slot index inside the row, ``0 .. n-1`` for a row that needs
  ``n`` units at its designed capacity.

The two properties that matter downstream:

1. **Slot k's id does not depend on n.** Growing a row from 10 GW to 11 GW
   leaves the first 200 slots' ids -- and therefore their realisations --
   bit-identical, so perturbed designs are common-random-number paired on every
   slot they share (the accreditation work's paired standard errors).
2. **Rows are disjoint.** Two rows never share a slot id, whatever their
   capacities, because the ordinal separates rows inside a ``(carrier, bus)``
   group and the group separates the rest.

On a fixed dataset ``slot-v1`` is informationally equivalent to keying on row
identity; its value is locality (property 1), not cross-resolution pairing,
which is explicitly not attempted.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np
import pandas as pd

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids a circular import
    from zap.reliability.outages import OutageParams

#: The static tables the pool is derived from, in the order rows are enumerated.
#: ``(component name, key in the static mapping, csv file name)``.
COMPONENT_TABLES = (
    ("Generator", "generators", "generators.csv"),
    ("StorageUnit", "storage_units", "storage_units.csv"),
)

#: Default scheme; carried on every LoadOptions, config, run card and table.
DEFAULT_SCHEME = "slot-v1"


@dataclass(frozen=True)
class RowSpec:
    """One pooled source row: its name and everything a key scheme may read.

    ``ordinal`` is assigned by :func:`row_specs` and is a property of the static
    tables alone -- never of a design, a capacity or a draw.
    """

    name: str
    component: str
    carrier: str
    bus: str
    ordinal: int


@runtime_checkable
class KeyScheme(Protocol):
    """Maps a row and a slot count onto that row's ``uint64`` unit ids."""

    name: str

    def unit_ids(self, row: RowSpec, n_units: int) -> np.ndarray:  # pragma: no cover - protocol
        ...


def _uid(text: str) -> np.uint64:
    """``uint64`` digest of a key string (blake2b, 8 bytes, big-endian)."""
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    return np.uint64(int.from_bytes(digest, "big"))


@dataclass(frozen=True)
class SlotV1:
    """``(carrier, bus, ordinal, k)`` keys -- the spec's D1.1 scheme."""

    name: str = "slot-v1"

    def key_string(self, row: RowSpec, slot: int) -> str:
        return f"{self.name}|{row.carrier}|{row.bus}|{int(row.ordinal)}|{int(slot)}"

    def unit_ids(self, row: RowSpec, n_units: int) -> np.ndarray:
        n = int(n_units)
        if n < 0:
            raise ValueError(f"n_units must be >= 0, got {n_units}")
        return np.array(
            [_uid(self.key_string(row, k)) for k in range(n)],
            dtype=np.uint64,
        )


#: The registry. A new scheme is one entry here plus a config value.
SCHEMES: dict[str, KeyScheme] = {"slot-v1": SlotV1()}


def get_scheme(name: str) -> KeyScheme:
    """Look a scheme up by name; unknown names are a hard error, never a default."""
    try:
        return SCHEMES[str(name)]
    except KeyError:
        raise KeyError(
            f"unknown outage key scheme {name!r}; registered schemes are "
            f"{sorted(SCHEMES)}"
        ) from None


def _name_series(df: pd.DataFrame) -> pd.Series:
    """Row names, whether the table carries them as the index or a ``name`` column."""
    if "name" in df.columns:
        return df["name"].astype(str)
    return pd.Series([str(i) for i in df.index], index=df.index)


def row_specs(
    static: Mapping[str, pd.DataFrame],
    params: OutageParams,
) -> list[RowSpec]:
    """Every pooled source row of ``static``, with its ordinal assigned.

    ``static`` is keyed as :func:`zap.importers.wy_store.read_static` keys it
    (``"generators"``, ``"storage_units"``); missing tables are treated as empty.
    Only rows whose carrier has outage parameters are returned. A carrier that
    is in neither ``params.carriers`` nor ``params.excluded_carriers`` is a hard
    error, so a new dataset cannot silently skip units.

    The ordinal counts rows inside their own ``(component, carrier, bus)`` group,
    so adding or removing rows in a *different* group never renumbers this one.
    """
    out: list[RowSpec] = []
    seen: set[str] = set()
    for component, key, filename in COMPONENT_TABLES:
        df = static.get(key)
        if df is None or len(df) == 0:
            continue
        unknown = sorted(
            set(df["carrier"].astype(str)) - set(params.carriers) - set(params.excluded_carriers)
        )
        if unknown:
            raise KeyError(
                f"{filename}: carriers {unknown} are in neither `carriers` nor "
                "`excluded_carriers` of the outage parameters; add them explicitly"
            )

        names = _name_series(df)
        counters: dict[tuple[str, str, str], int] = {}
        for name, carrier, bus in zip(
            names, df["carrier"].astype(str), df["bus"].astype(str), strict=True
        ):
            if carrier not in params.carriers:
                continue
            if name in seen:
                raise ValueError(f"duplicate component name {name!r} across static tables")
            seen.add(name)
            group = (component, carrier, bus)
            ordinal = counters.get(group, 0)
            counters[group] = ordinal + 1
            out.append(
                RowSpec(
                    name=str(name),
                    component=component,
                    carrier=carrier,
                    bus=bus,
                    ordinal=ordinal,
                )
            )
    return out


def row_spec_index(rows: Sequence[RowSpec]) -> dict[str, RowSpec]:
    """``{row name: RowSpec}`` -- the lookup every consumer actually wants."""
    return {r.name: r for r in rows}
