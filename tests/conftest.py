"""Shared fixtures for MeTTa integration tests."""

import pytest
from hyperon import MeTTa
from metta import register_all


STRENGTH_TOL = 0.05   # beta K=16 has quantization noise
CONFIDENCE_TOL = 0.15  # confidence from posterior is approximate


@pytest.fixture
def metta():
    """Fresh MeTTa runner with all thrml ops registered."""
    m = MeTTa()
    register_all(m)
    return m


def parse_stv(results):
    """Parse (stv s c) from MeTTa run results."""
    atom = results[-1][0]
    children = atom.get_children()
    return float(str(children[1])), float(str(children[2]))
