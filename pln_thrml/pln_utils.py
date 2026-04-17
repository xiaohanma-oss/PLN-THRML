"""
pln_thrml.pln_utils — PLN conversion utilities (no thrml dependency)
=====================================================================

Pure-math functions shared by all PLN-THRML modules:
  c2w / w2c   — confidence ↔ evidence weight
  stv_to_beta_params — (s, c) → Beta(α, β) parameterization
"""

EPS = 1e-7          # clamp for log-safety
DEFAULT_EPSILON = 0.02  # PLN modus-ponens background rate

__all__ = [
    "EPS", "DEFAULT_EPSILON",
    "c2w", "w2c",
    "stv_to_beta_params",
]


def c2w(c):
    """Confidence → evidence weight.  lib_pln.metta: Truth_c2w = c/(1-c)."""
    if c >= 1.0:
        return float('inf')
    if c <= 0.0:
        return 0.0
    return c / (1.0 - c)


def w2c(w):
    """Evidence weight → confidence.  lib_pln.metta: Truth_w2c = w/(w+1)."""
    if w < 0:
        return 0.0
    return w / (w + 1.0)


MAX_CONFIDENCE = 0.9999  # clamp to avoid inf in c2w


def stv_to_beta_params(strength, confidence):
    """Convert PLN (strength, confidence) to Beta(alpha, beta).

    Mean-preserving parameterization:
        n = w + 2  (where w = c/(1-c))
        alpha = s * n,  beta = (1-s) * n

    This guarantees Beta mean = s for any confidence.
    At c=0 → n=2, Beta(2s, 2(1-s)) which has mean=s; for s=0.5 → Beta(1,1)=uniform.
    """
    w = c2w(min(confidence, MAX_CONFIDENCE))
    n = w + 2.0  # total count (Beta(1,1) baseline = 2)
    alpha = max(strength * n, EPS)
    beta = max((1.0 - strength) * n, EPS)
    return alpha, beta
