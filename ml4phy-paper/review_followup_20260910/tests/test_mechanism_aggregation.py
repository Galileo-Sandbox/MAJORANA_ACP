"""Unit checks for mechanism-control aggregation primitives."""

import importlib.util
from pathlib import Path

import numpy as np

PATH = Path(__file__).parents[1] / "aggregate_mechanism.py"
SPEC = importlib.util.spec_from_file_location("aggregate_mechanism", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_identity_hash_is_dtype_stable_for_integer_rows():
    assert MODULE.sha_rows(np.array([3, 1, 2], dtype=np.int32)) == MODULE.sha_rows(
        np.array([3, 1, 2], dtype=np.int64)
    )


def test_region_endpoint_conventions():
    values = np.array([1577.0, 1605.999, 1606.0, 2619.0, 2619.001])
    broad = MODULE.mask(values, MODULE.REGIONS["DEP_broad"])
    core = MODULE.mask(values, MODULE.REGIONS["FE_core"])
    assert broad.tolist() == [True, True, False, False, False]
    assert core.tolist() == [False, False, False, True, False]


def test_primary_endpoints_are_frozen_composites():
    assert MODULE.PRIMARY == (
        "peaks_equal_four_cores",
        "continuum_equal_two_windows",
    )
