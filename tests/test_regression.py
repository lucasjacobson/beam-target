"""Pinned results of neutron sources covering the beams, targets, libraries, and physics the
code handles (see regression_cases.py for the cases). A code change that moves any of them
fails here; if it is meant to, write the references again with
python tests/regression_cases.py."""

import numpy as np
import pytest
from regression_cases import CASES, REFERENCE_DIR, run_case

# Slow: skipped by default; run with pytest -m slow (see pyproject.toml)
pytestmark = pytest.mark.slow


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


def test_threads_and_tokens_do_not_change_results(library_dir):
    """Sharing the beam energies of evaluated continua between threads, with or without
    tokens limiting how many run at once, gives the same results, bit for bit."""
    import threading

    from beamtarget import set_parallel

    name = "p_Be_18_endfb81"
    try:
        set_parallel(1)
        serial = run_case(name, library_dir)
        set_parallel(4)
        threaded = run_case(name, library_dir)
        set_parallel(4, threading.Semaphore(2))
        limited = run_case(name, library_dir)
    finally:
        set_parallel(1)
    for key, value in serial.items():
        np.testing.assert_array_equal(threaded[key], value, err_msg=key)
        np.testing.assert_array_equal(limited[key], value, err_msg=key)
