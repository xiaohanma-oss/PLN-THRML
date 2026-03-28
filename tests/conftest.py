"""Shared fixtures for MeTTa integration tests."""

import os
import pytest

try:
    from hyperon import MeTTa
    from metta import register_all
    _HAS_HYPERON = True
except ImportError:
    _HAS_HYPERON = False


STRENGTH_TOL = 0.05   # beta K=16 has quantization noise
CONFIDENCE_TOL = 0.15  # confidence from posterior is approximate

_PLN_LIB_PATH = os.path.join(
    os.path.dirname(__file__), "..", "vendor", "PLN", "lib_pln.metta"
)


@pytest.fixture
def metta():
    """Fresh MeTTa runner with all thrml ops registered."""
    if not _HAS_HYPERON:
        pytest.skip("hyperon not installed")
    m = MeTTa()
    register_all(m)
    return m


@pytest.fixture(scope="module")
def pln_lib():
    """Module-scoped MeTTa runner with upstream lib_pln.metta loaded."""
    if not _HAS_HYPERON:
        pytest.skip("hyperon not installed")
    m = MeTTa()
    # min/max not built-in in this MeTTa version
    m.run("(= (min $a $b) (if (< $a $b) $a $b))")
    m.run("(= (max $a $b) (if (> $a $b) $a $b))")
    with open(_PLN_LIB_PATH) as f:
        m.run(f.read())
    return m


def upstream_truth(pln, rule_name, *stv_pairs):
    """Call an upstream Truth_* function and return (strength, confidence).

    Example: upstream_truth(pln, "Truth_ModusPonens", (1.0, 0.9), (0.6, 0.9))
    """
    args = " ".join(f"(stv {s} {c})" for s, c in stv_pairs)
    results = pln.run(f"!({rule_name} {args})")
    return parse_stv(results)


def parse_stv(results):
    """Parse (stv s c) from MeTTa run results."""
    for batch in results:
        for atom in batch:
            s = str(atom)
            if s.startswith("(stv "):
                children = atom.get_children()
                return float(str(children[1])), float(str(children[2]))
    raise ValueError(f"No (stv ...) found in results: {results}")
