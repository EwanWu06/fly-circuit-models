"""The degree-preserving wirings (P4)."""
from minifly import circuit, config as cfg, rewire

BASE = circuit.edges_of(circuit.BASELINE)
WIRINGS = rewire.enumerate_wirings()


def test_there_are_964_wirings_including_the_original():
    assert len(WIRINGS) == 964
    rewire.index_of(WIRINGS, BASE)


def test_every_wiring_keeps_degrees_signs_and_is_simple():
    target = rewire.degrees(BASE)
    for w in WIRINGS:
        assert rewire.degrees(w) == target
        pairs = [(a, b) for a, b, _ in w]
        assert len(set(pairs)) == len(pairs)
        assert all(a != b for a, b in pairs)
        assert all(s == circuit.SIGN_BY_PRE[a] for a, _, s in w)


def test_mirror_is_an_involution_within_the_set():
    keys = {rewire._key(w) for w in WIRINGS}
    for w in WIRINGS:
        m = rewire.mirror(w)
        assert rewire._key(m) in keys
        assert rewire._key(rewire.mirror(m)) == rewire._key(w)


def test_mirror_of_the_designed_circuit_is_uncrossed():
    m = {(a, b) for a, b, _ in rewire.mirror(BASE)}
    assert (cfg.PN_L, cfg.MN_L) in m and (cfg.PN_R, cfg.MN_R) in m
