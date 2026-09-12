"""Tests for ``zap.reliability.keys`` (outage-pool spec WP-O1).

Hermetic: everything here is built from in-memory DataFrames. The properties
under test are the two the accreditation work depends on -- slot ids are a
prefix chain in ``n``, and rows never collide -- plus the stability of the
per-group ordinal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from zap.reliability import keys as K
from zap.reliability.outages import OutageParams

GEN_COLUMNS = ["name", "bus", "p_nom", "carrier"]
SU_COLUMNS = ["name", "bus", "p_nom", "carrier", "max_hours"]


def _params(**overrides) -> OutageParams:
    raw = {
        "version": 2,
        "reviewed": False,
        "excluded_carriers": ["solar", "onwind", "unspecified_imports"],
        "carriers": {
            "CCGT": {
                "unit_size_mw": 250,
                "forced_outage_rate": 0.045,
                "mttr_h": 50,
                "source": "test",
            },
            "OCGT": {
                "unit_size_mw": 100,
                "forced_outage_rate": 0.040,
                "mttr_h": 40,
                "source": "test",
            },
            "battery": {
                "unit_size_mw": 50,
                "forced_outage_rate": 0.020,
                "mttr_h": 24,
                "source": "test",
            },
        },
    }
    raw.update(overrides)
    return OutageParams.from_dict(raw, sha256="test")


def _static(generators, storage=()) -> dict[str, pd.DataFrame]:
    return {
        "generators": pd.DataFrame(generators, columns=GEN_COLUMNS),
        "storage_units": pd.DataFrame(storage, columns=SU_COLUMNS),
    }


DEFAULT_GENERATORS = [
    ("z1 CCGT", "z1", 1000.0, "CCGT"),
    ("z2 CCGT", "z2", 250.0, "CCGT"),
    ("z1 OCGT", "z1", 0.0, "OCGT"),
    ("z1 solar", "z1", 5000.0, "solar"),
]


# ---------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------


def test_default_scheme_is_registered():
    scheme = K.get_scheme(K.DEFAULT_SCHEME)
    assert scheme.name == "slot-v1"
    assert K.DEFAULT_SCHEME in K.SCHEMES


def test_unknown_scheme_raises():
    with pytest.raises(KeyError) as exc:
        K.get_scheme("identity-v1")
    assert "slot-v1" in str(exc.value)


# ---------------------------------------------------------------------------
# Unit ids
# ---------------------------------------------------------------------------


def test_unit_ids_are_deterministic():
    scheme = K.SlotV1()
    row = K.RowSpec("z1 CCGT", "Generator", "CCGT", "z1", 0)
    a = scheme.unit_ids(row, 7)
    b = K.SlotV1().unit_ids(K.RowSpec("z1 CCGT", "Generator", "CCGT", "z1", 0), 7)
    assert a.dtype == np.uint64
    np.testing.assert_array_equal(a, b)


def test_unit_ids_of_3_are_a_prefix_of_300():
    """D1.3: slot k's id does not depend on n -- the CRN pairing property."""
    scheme = K.SlotV1()
    row = K.RowSpec("z1 CCGT", "Generator", "CCGT", "z1", 0)
    small = scheme.unit_ids(row, 3)
    big = scheme.unit_ids(row, 300)
    np.testing.assert_array_equal(small, big[:3])
    assert len(set(big.tolist())) == 300  # and no internal collisions


def test_rows_sharing_carrier_and_bus_get_disjoint_ids():
    """The import buses carry several rows of one carrier; the ordinal separates them."""
    scheme = K.SlotV1()
    a = scheme.unit_ids(K.RowSpec("a", "Generator", "CCGT", "z1", 0), 50)
    b = scheme.unit_ids(K.RowSpec("b", "Generator", "CCGT", "z1", 1), 50)
    assert not (set(a.tolist()) & set(b.tolist()))


def test_ids_differ_across_carrier_bus_and_ordinal():
    scheme = K.SlotV1()
    base = K.RowSpec("r", "Generator", "CCGT", "z1", 0)
    variants = [
        K.RowSpec("r", "Generator", "OCGT", "z1", 0),
        K.RowSpec("r", "Generator", "CCGT", "z2", 0),
        K.RowSpec("r", "Generator", "CCGT", "z1", 1),
    ]
    seen = {int(scheme.unit_ids(base, 1)[0])}
    for row in variants:
        uid = int(scheme.unit_ids(row, 1)[0])
        assert uid not in seen
        seen.add(uid)


def test_negative_n_units_raises():
    with pytest.raises(ValueError):
        K.SlotV1().unit_ids(K.RowSpec("r", "Generator", "CCGT", "z1", 0), -1)


def test_zero_units_is_an_empty_uint64_array():
    ids = K.SlotV1().unit_ids(K.RowSpec("r", "Generator", "CCGT", "z1", 0), 0)
    assert ids.shape == (0,)
    assert ids.dtype == np.uint64


# ---------------------------------------------------------------------------
# row_specs
# ---------------------------------------------------------------------------


def test_row_specs_skips_excluded_carriers_and_assigns_ordinals():
    rows = K.row_specs(_static(DEFAULT_GENERATORS), _params())
    names = [r.name for r in rows]
    assert names == ["z1 CCGT", "z2 CCGT", "z1 OCGT"]  # solar is excluded
    # One member per (component, carrier, bus) group here, so every ordinal is 0.
    assert [r.ordinal for r in rows] == [0, 0, 0]
    assert rows[0].component == "Generator"


def test_row_specs_numbers_a_shared_carrier_bus_group():
    generators = [
        ("imp a", "z1_imports", 100.0, "CCGT"),
        ("imp b", "z1_imports", 200.0, "CCGT"),
        ("imp c", "z1_imports", 300.0, "CCGT"),
    ]
    rows = K.row_specs(_static(generators), _params())
    assert [(r.name, r.ordinal) for r in rows] == [("imp a", 0), ("imp b", 1), ("imp c", 2)]


def test_ordinals_are_stable_when_another_group_changes():
    """Adding rows to a *different* group must not renumber this one."""
    before = [
        ("z1 CCGT a", "z1", 100.0, "CCGT"),
        ("z1 CCGT b", "z1", 100.0, "CCGT"),
        ("z2 OCGT", "z2", 100.0, "OCGT"),
    ]
    after = [
        ("z2 OCGT extra", "z2", 50.0, "OCGT"),  # new row, different group, first
        ("z1 CCGT a", "z1", 100.0, "CCGT"),
        ("z1 CCGT b", "z1", 100.0, "CCGT"),
        ("z2 OCGT", "z2", 100.0, "OCGT"),
    ]
    spec_before = {r.name: r.ordinal for r in K.row_specs(_static(before), _params())}
    spec_after = {r.name: r.ordinal for r in K.row_specs(_static(after), _params())}
    assert spec_before["z1 CCGT a"] == spec_after["z1 CCGT a"] == 0
    assert spec_before["z1 CCGT b"] == spec_after["z1 CCGT b"] == 1
    # The unit ids of the untouched group are therefore bit-identical.
    scheme = K.SlotV1()
    rows_before = {r.name: r for r in K.row_specs(_static(before), _params())}
    rows_after = {r.name: r for r in K.row_specs(_static(after), _params())}
    np.testing.assert_array_equal(
        scheme.unit_ids(rows_before["z1 CCGT b"], 4),
        scheme.unit_ids(rows_after["z1 CCGT b"], 4),
    )


def test_unknown_carrier_is_a_hard_error():
    generators = [("mystery", "z1", 10.0, "fusion")]
    with pytest.raises(KeyError) as exc:
        K.row_specs(_static(generators), _params())
    assert "fusion" in str(exc.value)


def test_row_specs_accepts_name_as_the_index():
    """``wy_store.read_static`` reads with ``index_col=0``; both forms must work."""
    frame = pd.DataFrame(DEFAULT_GENERATORS, columns=GEN_COLUMNS).set_index("name")
    static = {"generators": frame, "storage_units": pd.DataFrame(columns=SU_COLUMNS)}
    rows = K.row_specs(static, _params())
    assert [r.name for r in rows] == ["z1 CCGT", "z2 CCGT", "z1 OCGT"]


def test_duplicate_row_names_across_tables_raise():
    generators = [("dup", "z1", 10.0, "CCGT")]
    storage = [("dup", "z1", 10.0, "battery", 4.0)]
    with pytest.raises(ValueError):
        K.row_specs(_static(generators, storage), _params())


def test_storage_rows_follow_generator_rows():
    rows = K.row_specs(
        _static(DEFAULT_GENERATORS, [("z1 battery", "z1", 400.0, "battery", 4.0)]), _params()
    )
    assert rows[-1].name == "z1 battery"
    assert rows[-1].component == "StorageUnit"
