"""Pinned results of neutron sources covering the beams, targets, libraries, and physics the
code handles (see regression_cases.py for the cases). A code change that moves any of them
fails here; if it is meant to, write the references again with
python tests/regression_cases.py."""

import numpy as np
import pytest
from regression_cases import CASES, REFERENCE_DIR, run_case


@pytest.fixture(scope="module")
def library_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("libraries")


@pytest.mark.parametrize("name", list(CASES))
def test_results_unchanged(name, library_dir):
    reference = np.load(REFERENCE_DIR / f"{name}.npz")
    results = run_case(name, library_dir)
    assert set(results) == set(reference.files)
    for key, value in results.items():
        expected = reference[key]
        assert value.shape == expected.shape, key
        scale = np.max(np.abs(expected)) if expected.size else 0.0
        np.testing.assert_allclose(value, expected, rtol=1e-6, atol=1e-9 * scale, err_msg=key)
