"""
pln_thrml.qln_cpu — QLN n-layer: closed-form confidence propagation on CPU
===========================================================================

Implements the n-layer of the unified (ρ, n) separation architecture:
  - **ρ (strength s)** sampled on TSU via binary Ising / LBM
  - **n (evidence count)** propagated on CPU via deterministic algebra (this module)

Each function computes output confidence from input confidences using PLN
closed-form formulas.  Inversion and Revision are CPU-only (no TSU component).

Revision uses QLN formula (n_rev = n₁ + n₂) per Definition 3.9 of the QLN
paper, which differs from upstream PLN (w_rev = w₁ + w₂) by ~2 pseudocounts.
"""

from pln_thrml.beta import c2w, w2c, EPS

__all__ = [
    "c_modus_ponens",
    "c_deduction",
    "c_abduction",
    "inversion_pln",
    "inversion_bayes",
    "revision",
]


# ═══════════════════════════════════════════════════════════════════════════
#  Confidence formulas (extracted from hybrid.py private functions)
# ═════════��═════════════════════════════════════════════════════════════════

def c_modus_ponens(c_A, c_AB):
    """PLN Modus Ponens confidence: c_B = c_A × c_AB."""
    return float(c_A) * float(c_AB)


def c_deduction(s_AB, c_AB, s_BC, c_BC):
    """PLN Deduction confidence: c_AC = s_AB × s_BC × c_AB × c_BC."""
    return float(s_AB) * float(s_BC) * float(c_AB) * float(c_BC)


def c_abduction(s_AC, c_AC, s_BC, c_BC):
    """PLN Abduction confidence: c_AB = s_AC × s_BC × c_AC × c_BC."""
    return float(s_AC) * float(s_BC) * float(c_AC) * float(c_BC)


# ═════════════════════════════��══════════════════════════���══════════════════
#  Inversion (CPU-only, no TSU)
# ═══════════════════════════════════════════════════════════════════════════

def inversion_pln(s_A, c_A, s_B, c_B, s_AB, c_AB):
    """Upstream PLN heuristic inversion (lib_pln.metta:150).

    Truth_inversion(B, AB) = (s_AB, c_B × c_AB × 0.6)

    Note: strength is unchanged — the upstream PLN formula does NOT use
    s_A or s_B for the inverted strength.  The comment in lib_pln.metta
    says "not according to OpenCOG classic".

    Returns
    -------
    (s_BA, c_BA) : tuple[float, float]
    """
    s_BA = float(s_AB)
    c_BA = float(c_B) * float(c_AB) * 0.6
    return s_BA, c_BA


def inversion_bayes(s_A, c_A, s_B, c_B, s_AB, c_AB):
    """Bayesian inversion: P(A|B) = P(B|A) × P(A) / P(B).

    Uses exact Bayes formula for strength.  The QLN paper (Remark 3.6)
    confirms that the Petz recovery map reduces to Bayes' theorem in the
    classical limit, supporting this formula over the upstream heuristic.

    Confidence uses evidence-weight scaling: the inverted link can't have
    more evidence than the inputs provide.

    Returns
    -------
    (s_BA, c_BA) : tuple[float, float]
    """
    s_A_f = float(s_A)
    s_B_f = max(float(s_B), EPS)
    s_AB_f = float(s_AB)

    s_BA = min(s_AB_f * s_A_f / s_B_f, 1.0)

    # Confidence: evidence weight of the inverted link is bounded by inputs.
    # w_BA = w_AB × (s_AB × s_A / s_B), clamped.
    w_AB = c2w(min(float(c_AB), 0.9999))
    w_A = c2w(min(float(c_A), 0.9999))
    w_B = c2w(min(float(c_B), 0.9999))

    # Scale by min evidence and the Bayes ratio
    w_BA = min(w_AB, w_A, w_B) * min(s_AB_f * s_A_f / s_B_f, 1.0)
    c_BA = w2c(max(w_BA, 0.0))

    return s_BA, c_BA


# ═════════════��═════════════════════════════════════════���═══════════════════
#  Revision (CPU-only, QLN formula)
# ═════���════════════════════════════��════════════════════════════════════════

def revision(s1, c1, s2, c2):
    """QLN Revision: merge two independent evidence sources.

    Uses QLN Definition 3.9:
        n_rev = n₁ + n₂
        s_rev = (n₁ × s₁ + n₂ × s₂) / n_rev
        c_rev = w2c(n_rev - 2)

    This differs from upstream PLN (w_rev = w₁ + w₂) by ~2 pseudocounts
    because QLN's n includes the Beta(1,1) baseline per source.
    The choice is intentional: in the (ρ, n) architecture, n is a first-class
    citizen and n₁ + n₂ is the natural merge operation.

    Returns
    -------
    (s_rev, c_rev) : tuple[float, float]
    """
    s1_f, s2_f = float(s1), float(s2)
    c1_f = min(float(c1), 0.9999)
    c2_f = min(float(c2), 0.9999)

    n1 = c2w(c1_f) + 2.0
    n2 = c2w(c2_f) + 2.0

    n_rev = n1 + n2
    s_rev = (n1 * s1_f + n2 * s2_f) / n_rev

    w_rev = max(n_rev - 2.0, 0.0)
    c_rev = w2c(w_rev)

    return float(s_rev), float(c_rev)
