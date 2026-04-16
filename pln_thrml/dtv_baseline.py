"""
pln_thrml.dtv_baseline — Continuous Beta MC ground truth (zero discretization)
================================================================================

DTV (Distributional Truth Value) baseline: Monte Carlo integration over
continuous Beta distributions. No K-bin discretization — serves as ground
truth for measuring approximation error of discrete methods.

Approach follows Geisweiller's tv-toolbox: sample from Beta posteriors,
apply PLN closed-form formulas per sample, estimate output (s, c) from
the resulting distribution's moments.
"""

import numpy as np
from scipy.stats import beta as beta_dist

from pln_thrml.beta import stv_to_beta_params, w2c, EPS


__all__ = [
    "dtv_modus_ponens",
    "dtv_deduction",
    "dtv_abduction",
    "dtv_inversion",
    "dtv_revision",
]


def _fit_stv_from_samples(samples):
    """Estimate (strength, confidence) from MC samples via moment-matching.

    Same logic as posterior_to_stv but on continuous samples.
    """
    samples = np.asarray(samples)
    samples = samples[(samples > 0) & (samples < 1)]  # discard boundary
    if len(samples) < 10:
        return 0.5, 0.0

    mu = samples.mean()
    var = samples.var()
    if var < 1e-12:
        return float(mu), 0.9999

    # moment-match to Beta: n = mu*(1-mu)/var - 1
    n = mu * (1.0 - mu) / var - 1.0
    n = max(n, 0.0)
    w_eff = max(n - 2.0, 0.0)
    confidence = w2c(w_eff)
    return float(mu), float(confidence)


def dtv_modus_ponens(s_A, c_A, s_AB, c_AB, background=0.02, n_samples=200_000,
                     seed=42):
    """DTV ground truth for Modus Ponens: P(B) given P(A) and P(B|A).

    Samples a ~ Beta(α_A, β_A), then computes:
        b = s_AB * a + background * (1 - a)
    which is the PLN modus ponens formula applied per sample.

    Returns (strength, confidence) estimated from output distribution.
    """
    rng = np.random.default_rng(seed)
    alpha_A, beta_A = stv_to_beta_params(s_A, c_A)
    alpha_AB, beta_AB = stv_to_beta_params(s_AB, c_AB)

    a_samples = beta_dist.rvs(alpha_A, beta_A, size=n_samples,
                              random_state=rng)
    ab_samples = beta_dist.rvs(alpha_AB, beta_AB, size=n_samples,
                               random_state=rng)

    # PLN modus ponens: b = ab * a + background * (1 - a)
    b_samples = ab_samples * a_samples + background * (1.0 - a_samples)
    b_samples = np.clip(b_samples, EPS, 1.0 - EPS)

    return _fit_stv_from_samples(b_samples)


def dtv_deduction(s_A, c_A, s_B, c_B, s_C, c_C,
                  s_AB, c_AB, s_BC, c_BC,
                  n_samples=200_000, seed=42):
    """DTV ground truth for Deduction: P(A→C) given P(A→B) and P(B→C).

    Samples from ALL 5 Beta distributions simultaneously (Geisweiller
    fullDeduction approach — s_B and s_C have uncertainty too):
        a ~ Beta(α_A, β_A)
        b ~ Beta(α_B, β_B)
        c ~ Beta(α_C, β_C)
        ab ~ Beta(α_AB, β_AB)
        bc ~ Beta(α_BC, β_BC)

    PLN deduction formula per sample:
        s_AC = ab * bc + (1 - ab) * (c - b * bc) / (1 - b)

    Returns (strength, confidence) estimated from output distribution.
    """
    rng = np.random.default_rng(seed)

    alpha_A, beta_A = stv_to_beta_params(s_A, c_A)
    alpha_B, beta_B = stv_to_beta_params(s_B, c_B)
    alpha_C, beta_C = stv_to_beta_params(s_C, c_C)
    alpha_AB, beta_AB = stv_to_beta_params(s_AB, c_AB)
    alpha_BC, beta_BC = stv_to_beta_params(s_BC, c_BC)

    a = beta_dist.rvs(alpha_A, beta_A, size=n_samples, random_state=rng)
    b = beta_dist.rvs(alpha_B, beta_B, size=n_samples, random_state=rng)
    c = beta_dist.rvs(alpha_C, beta_C, size=n_samples, random_state=rng)
    ab = beta_dist.rvs(alpha_AB, beta_AB, size=n_samples, random_state=rng)
    bc = beta_dist.rvs(alpha_BC, beta_BC, size=n_samples, random_state=rng)

    # PLN deduction: s_AC = ab * bc + (1 - ab) * (c - b * bc) / (1 - b)
    # Clamp denominator to avoid division by zero
    denom = np.maximum(1.0 - b, EPS)
    numerator = c - b * bc
    s_AC = ab * bc + (1.0 - ab) * numerator / denom

    # Clamp to valid probability range
    s_AC = np.clip(s_AC, EPS, 1.0 - EPS)

    return _fit_stv_from_samples(s_AC)


def dtv_abduction(s_A, c_A, s_B, c_B, s_AC, c_AC, s_BC, c_BC,
                  n_samples=200_000, seed=42):
    """DTV ground truth for Abduction: P(A|B) given A←C→B structure.

    Explaining-away: observing B makes A less likely if both explain C.
    Samples from 4 Beta distributions:
        a ~ Beta(α_A, β_A)
        b ~ Beta(α_B, β_B)
        ac ~ Beta(α_AC, β_AC)
        bc ~ Beta(α_BC, β_BC)

    PLN abduction formula per sample:
        s_AB = ac * bc + (1 - ac) * (b - a * bc) / (1 - a)

    Returns (strength, confidence) estimated from output distribution.
    """
    rng = np.random.default_rng(seed)

    alpha_A, beta_A = stv_to_beta_params(s_A, c_A)
    alpha_B, beta_B = stv_to_beta_params(s_B, c_B)
    alpha_AC, beta_AC = stv_to_beta_params(s_AC, c_AC)
    alpha_BC, beta_BC = stv_to_beta_params(s_BC, c_BC)

    a = beta_dist.rvs(alpha_A, beta_A, size=n_samples, random_state=rng)
    b = beta_dist.rvs(alpha_B, beta_B, size=n_samples, random_state=rng)
    ac = beta_dist.rvs(alpha_AC, beta_AC, size=n_samples, random_state=rng)
    bc = beta_dist.rvs(alpha_BC, beta_BC, size=n_samples, random_state=rng)

    # PLN abduction: s_AB = ac * bc + (1 - ac) * (b - a * bc) / (1 - a)
    denom = np.maximum(1.0 - a, EPS)
    numerator = b - a * bc
    s_AB = ac * bc + (1.0 - ac) * numerator / denom

    s_AB = np.clip(s_AB, EPS, 1.0 - EPS)

    return _fit_stv_from_samples(s_AB)


def dtv_inversion(s_A, c_A, s_B, c_B, s_AB, c_AB,
                  n_samples=200_000, seed=42):
    """DTV ground truth for Inversion: P(B→A) given P(A→B), P(A), P(B).

    Per-sample Bayes theorem: s_BA = ab * a / b.

    Samples from 3 Beta distributions:
        a ~ Beta(α_A, β_A)
        b ~ Beta(α_B, β_B)
        ab ~ Beta(α_AB, β_AB)

    Returns (strength, confidence) estimated from output distribution.
    """
    rng = np.random.default_rng(seed)

    alpha_A, beta_A = stv_to_beta_params(s_A, c_A)
    alpha_B, beta_B = stv_to_beta_params(s_B, c_B)
    alpha_AB, beta_AB = stv_to_beta_params(s_AB, c_AB)

    a = beta_dist.rvs(alpha_A, beta_A, size=n_samples, random_state=rng)
    b = beta_dist.rvs(alpha_B, beta_B, size=n_samples, random_state=rng)
    ab = beta_dist.rvs(alpha_AB, beta_AB, size=n_samples, random_state=rng)

    # Bayes inversion: s_BA = ab * a / b
    denom = np.maximum(b, EPS)
    s_BA = ab * a / denom

    s_BA = np.clip(s_BA, EPS, 1.0 - EPS)

    return _fit_stv_from_samples(s_BA)


def dtv_revision(s1, c1, s2, c2, n_samples=200_000, seed=42):
    """DTV ground truth for Revision: merge two Beta evidence sources.

    Samples from 2 Beta distributions and computes the evidence-weighted
    combination (QLN formula: n_rev = n₁ + n₂).

    The DTV approach: sample from both posteriors, combine with weights
    proportional to their evidence counts, moment-match the result.

    Returns (strength, confidence) estimated from output distribution.
    """
    from pln_thrml.beta import c2w

    rng = np.random.default_rng(seed)

    alpha1, beta1 = stv_to_beta_params(s1, c1)
    alpha2, beta2 = stv_to_beta_params(s2, c2)

    x1 = beta_dist.rvs(alpha1, beta1, size=n_samples, random_state=rng)
    x2 = beta_dist.rvs(alpha2, beta2, size=n_samples, random_state=rng)

    # QLN revision: weighted average with n-weights
    n1 = c2w(min(float(c1), 0.9999)) + 2.0
    n2 = c2w(min(float(c2), 0.9999)) + 2.0
    w1 = n1 / (n1 + n2)
    w2 = n2 / (n1 + n2)

    combined = w1 * x1 + w2 * x2
    combined = np.clip(combined, EPS, 1.0 - EPS)

    return _fit_stv_from_samples(combined)
