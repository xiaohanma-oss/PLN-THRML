"""Shared fixtures for PLN-THRML tests."""

STRENGTH_TOL = 0.05   # beta K=16 has quantization noise
CONFIDENCE_TOL = 0.15  # confidence from posterior is approximate

STRENGTH_TOL_BY_K = {4: 0.12, 8: 0.08, 16: 0.05}
CONFIDENCE_TOL_BY_K = {4: 0.25, 8: 0.20, 16: 0.15}


def strength_tol(k=16):
    """Return strength tolerance for a given K."""
    return STRENGTH_TOL_BY_K.get(k, STRENGTH_TOL)


def confidence_tol(k=16):
    """Return confidence tolerance for a given K."""
    return CONFIDENCE_TOL_BY_K.get(k, CONFIDENCE_TOL)
