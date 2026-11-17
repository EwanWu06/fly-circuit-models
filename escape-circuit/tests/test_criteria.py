from src import criteria


def test_b1_rejects_a_silent_giant_fiber():
    assert not criteria.check_b1({10.0: 40.0, 20.0: None})


def test_b1_accepts_all_in_range():
    assert criteria.check_b1({10.0: 35.0, 20.0: 40.0, 40.0: 45.0, 80.0: 49.0})


def test_b2_requires_lc4_to_grow_as_rv_shrinks():
    lplc2 = {10.0: 1.0, 20.0: 1.0, 40.0: 1.0, 80.0: 1.0}
    assert criteria.check_b2({10.0: 4.0, 20.0: 3.0, 40.0: 2.0, 80.0: 1.0}, lplc2)
    assert not criteria.check_b2({10.0: 1.0, 20.0: 2.0, 40.0: 3.0, 80.0: 4.0},
                                 lplc2)


def test_b4_requires_sublinearity():
    assert criteria.check_b4(0.7)
    assert not criteria.check_b4(1.0)
