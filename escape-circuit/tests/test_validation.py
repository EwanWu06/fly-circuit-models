import importlib.util
from pathlib import Path

import pytest

# Load the experiment module dynamically since its name starts with a digit
_exp_path = Path(__file__).resolve().parent.parent / "experiments" / "00_validate_extraction.py"
_spec = importlib.util.spec_from_file_location("validate_extraction", _exp_path)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
compute_validation = _module.compute_validation


@pytest.fixture(scope="module")
def validation_data():
    """Compute validation metrics from live circuit extraction."""
    return compute_validation()


def test_lc4_right_matches_ache_2019_fafb(validation_data):
    """LC4 -> GF should match Ache et al. 2019's FAFB count of 55 cells."""
    assert validation_data["lc4_cells_right"] == 55
    assert abs(validation_data["lc4_synapses_right"] - 2442) / 2442 < 0.15
