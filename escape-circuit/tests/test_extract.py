import numpy as np
import pytest

from src import config, extract
from src.circuit import Circuit

GF_BODY_IDS = (10001, 10010)


@pytest.fixture(scope="module")
def c() -> Circuit:
    return extract.build_circuit()


def test_contains_both_giant_fibers(c):
    for body_id in GF_BODY_IDS:
        assert c.types[c.index_of(body_id)] == "DNp01"


def test_input_layer_counts_match_measurement(c):
    assert int(c.mask_by_type("LC4").sum()) == 126
    assert int(c.mask_by_type("LPLC2").sum()) == 185


def test_size_is_in_expected_range(c):
    assert 350 <= c.n_neurons <= 380


def test_has_both_excitatory_and_inhibitory_edges(c):
    assert (c.weight > 0).sum() > 0
    assert (c.weight < 0).sum() > 0


def test_all_edge_indices_are_valid(c):
    assert c.pre.min() >= 0 and c.pre.max() < c.n_neurons
    assert c.post.min() >= 0 and c.post.max() < c.n_neurons


def test_min_weight_filter_applied(c):
    assert np.abs(c.weight).min() >= config.MIN_SYNAPSE_WEIGHT


def test_roundtrip(tmp_path, c):
    p = tmp_path / "circuit.npz"
    c.save(p)
    loaded = Circuit.load(p)
    assert np.array_equal(loaded.weight, c.weight)
    assert loaded.n_neurons == c.n_neurons
