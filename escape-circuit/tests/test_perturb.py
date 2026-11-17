import numpy as np
import pytest

from src import extract, perturb


@pytest.fixture(scope="module")
def c():
    return extract.build_circuit()


def _degrees(circuit):
    n = circuit.n_neurons
    return (np.bincount(circuit.pre, minlength=n),
            np.bincount(circuit.post, minlength=n))


def test_swap_count_formula_pins_definition_of_f():
    """Verify the precise f → swap count mapping against known-good values.

    Each test value was independently computed on the 364-neuron / 7,784-edge
    circuit and verified. This guards against mutations that drop the /2,
    use *2, or change round to ceil—all of which pass the behavioural tests
    but mislabel the axis.
    """
    n_edges = 7784  # 364-neuron Drosophila escape circuit
    test_cases = [
        (0.00, 0),
        (0.05, 195),
        (0.10, 389),
        (0.25, 973),
        (0.50, 1946),
        (1.00, 3892),
    ]
    for fraction, expected_swaps in test_cases:
        actual_swaps = perturb.swap_count(n_edges, fraction)
        assert actual_swaps == expected_swaps, (
            f"swap_count({n_edges}, {fraction}) = {actual_swaps}, "
            f"expected {expected_swaps}"
        )


def test_rewiring_preserves_in_and_out_degree(c):
    rng = np.random.default_rng(0)
    out = perturb.DegreePreservingRewire().apply(c, 0.5, rng)
    assert _degrees(out)[0].tolist() == _degrees(c)[0].tolist()
    assert _degrees(out)[1].tolist() == _degrees(c)[1].tolist()


def test_fraction_zero_is_identity(c):
    rng = np.random.default_rng(0)
    out = perturb.DegreePreservingRewire().apply(c, 0.0, rng)
    assert perturb.edge_overlap(c, out) == 1.0


def test_more_rewiring_lowers_overlap(c):
    rng = np.random.default_rng(1)
    overlaps = [perturb.edge_overlap(
                    c, perturb.DegreePreservingRewire().apply(c, f, rng))
                for f in (0.05, 0.25, 1.0)]
    assert overlaps[0] > overlaps[1] > overlaps[2]


def test_no_self_loops_or_duplicate_edges(c):
    rng = np.random.default_rng(2)
    out = perturb.DegreePreservingRewire().apply(c, 1.0, rng)
    assert not np.any(out.pre == out.post)
    pairs = set(zip(out.pre.tolist(), out.post.tolist()))
    assert len(pairs) == out.n_edges


def test_same_seed_gives_same_result(c):
    a = perturb.DegreePreservingRewire().apply(c, 0.5, np.random.default_rng(7))
    b = perturb.DegreePreservingRewire().apply(c, 0.5, np.random.default_rng(7))
    assert np.array_equal(a.pre, b.pre) and np.array_equal(a.post, b.post)
