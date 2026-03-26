"""Test thrml-modus-ponens! grounded operation with beta approach."""

import pytest
from pln_thrml import STV, truth_modus_ponens
from conftest import STRENGTH_TOL, parse_stv


CASES = [
    (0.8, 0.9, 0.9, 0.85),
    (0.5, 0.8, 0.95, 0.9),
    (0.1, 0.7, 0.8, 0.75),
    (0.9, 0.95, 0.5, 0.8),
    (0.3, 0.6, 0.7, 0.6),
]


@pytest.mark.parametrize("s_A,c_A,s_AB,c_AB", CASES)
def test_modus_ponens(metta, s_A, c_A, s_AB, c_AB):
    results = metta.run(f"""
        (A (stv {s_A} {c_A}))
        ((Implication A B) (stv {s_AB} {c_AB}))
        !(thrml-modus-ponens! (A B (stv {s_A} {c_A}) (stv {s_AB} {c_AB})))
    """)

    sampled_s, sampled_c = parse_stv(results)
    expected = truth_modus_ponens(STV(s_A, c_A), STV(s_AB, c_AB))

    assert abs(sampled_s - expected.strength) < STRENGTH_TOL
    assert sampled_c > 0.0  # confidence from posterior should be positive
