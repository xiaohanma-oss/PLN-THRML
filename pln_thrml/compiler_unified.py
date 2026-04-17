"""
pln_thrml.compiler_unified — Confidence-modulated Ising compiler
=================================================================

Extends compiler_binary with g(n) modulation: coupling weights are scaled
by a function of the evidence count, so higher confidence inputs produce
stronger (more deterministic) Ising couplings.

    J_modulated = J(s) × g(n)
    h_modulated = h(s) × g(n)

This implements the precision modulation concept from Active Inference:
precision (inverse variance) scales prediction errors, here evidence count
scales coupling strength.

The default g(n) = min(sqrt(n/2), 10) is a design choice balancing:
  - Linear g=n/2 (existing tempered): grows too fast, kills mixing
  - Logarithmic g=log(n+1): too flat, barely distinguishes confidence
  - sqrt(n/2): gentle curve, n=2→1, n=10→2.24, n=100→7.07, cap=10
"""

import math

import numpy as np
import jax.numpy as jnp

from pln_thrml.pln_utils import c2w, EPS, DEFAULT_EPSILON
from pln_thrml.compiler_binary import (
    ising_params,
    prior_bias,
    SpinNode,
    SpinEBMFactor,
    Block,
    _assemble_binary_graph_ex,
)


def _parent_bias(strength, background):
    """Parent-bias μ from the full log-joint expansion of a CPT edge.

    μ = (1/4) · ln[s·(1-s) / (bg·(1-bg))]

    Needed on the parent node for exact Boltzmann marginals when the
    parent is free (not clamped).  Missing μ was the root cause of the
    Deduction Chain bias (fixed via include_parent_bias=True) and of
    the inv-V Abduction bias (fixed here).
    """
    s = float(np.clip(strength, EPS, 1.0 - EPS))
    bg = float(np.clip(background, EPS, 1.0 - EPS))
    return 0.25 * (np.log(s * (1.0 - s)) - np.log(bg * (1.0 - bg)))

__all__ = [
    "default_g_fn",
    "compile_modulated_chain",
    "compile_modulated_inv_v",
    "compile_modulated_inv_v_with_hidden",
]


MAX_CONFIDENCE = 0.9999


def default_g_fn(n, cap=10.0):
    """Default confidence modulation: g(n) = min(sqrt(n/2), cap).

    n=2 (minimum, c≈0) → g=1 (baseline coupling)
    n=10 (c≈0.8)       → g≈2.24
    n=100 (c≈0.98)     → g≈7.07
    cap=10              → max scaling regardless of confidence
    """
    return min(math.sqrt(max(n, 0.0) / 2.0), cap)


def _confidence_to_n(c):
    """Convert confidence to evidence count n = c2w(c) + 2."""
    return c2w(min(float(c), MAX_CONFIDENCE)) + 2.0


# ═══════════════════════════════════════════════════════════════════════════
#  Modulated chain compiler (MP and Deduction)
# ═══════════════════════════════════════════════════════════════════════════

def compile_modulated_chain(priors, strengths, confidences, backgrounds=None,
                            g_fn=None, clamp_root=True):
    """Build a binary Ising chain with confidence-modulated couplings.

    Like compile_binary_chain, but each edge's J and h_correction are
    scaled by g(n_edge) where n_edge is derived from the edge's confidence.

    Parameters
    ----------
    priors : list[float]
        Prior strength for each node.
    strengths : list[float]
        Implication strengths for each edge (n-1 edges).
    confidences : list[float]
        Confidence for each edge (n-1 values).  Used for g(n) modulation.
    backgrounds : list[float] or None
        Background rates per edge.
    g_fn : callable or None
        g(n) → scale factor.  Defaults to default_g_fn.
    clamp_root : bool
        If True, root is clamped per batch.

    Returns
    -------
    dict
        Graph dict compatible with run_binary_sampling.
    """
    n = len(priors)
    if backgrounds is None:
        backgrounds = [DEFAULT_EPSILON] * (n - 1)
    if g_fn is None:
        g_fn = default_g_fn

    spins = [SpinNode() for _ in range(n)]
    blocks = [Block([s]) for s in spins]
    factors = []

    if clamp_root:
        free_blocks = blocks[1:]
        clamped_blocks = [blocks[0]]
    else:
        free_blocks = blocks
        clamped_blocks = []

    # Biases on free nodes (neutral prior = 0.5)
    start = 1 if clamp_root else 0
    for i in range(start, n):
        h = prior_bias(priors[i])
        if abs(h) > 1e-10:
            factors.append(SpinEBMFactor([blocks[i]], jnp.array([h])))

    # Modulated pairwise couplings
    for i in range(n - 1):
        J, h_corr = ising_params(float(strengths[i]), float(backgrounds[i]))
        n_edge = _confidence_to_n(confidences[i])
        scale = g_fn(n_edge)

        J_mod = J * scale
        h_mod = h_corr * scale

        if abs(J_mod) > 1e-10:
            factors.append(SpinEBMFactor(
                [blocks[i], blocks[i + 1]], jnp.array([J_mod])))
        if abs(h_mod) > 1e-10:
            factors.append(SpinEBMFactor(
                [blocks[i + 1]], jnp.array([h_mod])))

    return _assemble_binary_graph_ex(
        spins, blocks, free_blocks, clamped_blocks, factors,
        root_prior=priors[0] if clamp_root else None)


# ═══════════════════════════════════════════════════════════════════════════
#  Modulated inverted-V compiler (Abduction without hidden units)
# ═══════════════════════════════════════════════════════════════════════════

def compile_modulated_inv_v(left_prior, right_prior,
                            left_strength, right_strength,
                            left_confidence, right_confidence,
                            left_background=DEFAULT_EPSILON,
                            right_background=DEFAULT_EPSILON,
                            center_prior=0.5, g_fn=None):
    """Build a modulated inverted-V: Left → Center ← Right.

    Same topology as compile_binary_inv_v but with g(n) modulated couplings.
    """
    if g_fn is None:
        g_fn = default_g_fn

    spins = [SpinNode() for _ in range(3)]
    blocks = [Block([s]) for s in spins]
    factors = []

    # Biases
    for i, p in enumerate([left_prior, center_prior, right_prior]):
        h = prior_bias(p)
        if abs(h) > 1e-10:
            factors.append(SpinEBMFactor([blocks[i]], jnp.array([h])))

    # Left → Center (modulated)
    J_left, h_corr_left = ising_params(left_strength, left_background)
    mu_left = _parent_bias(left_strength, left_background)
    scale_left = g_fn(_confidence_to_n(left_confidence))
    # Scale J only, NOT h_corr — both edges' h_corrections land on center,
    # scaling them amplifies the encoding artifact and overwhelms the coupling.
    if abs(J_left * scale_left) > 1e-10:
        factors.append(SpinEBMFactor(
            [blocks[0], blocks[1]], jnp.array([J_left * scale_left])))
    if abs(h_corr_left) > 1e-10:
        factors.append(SpinEBMFactor([blocks[1]], jnp.array([h_corr_left])))
    # Parent-bias μ on left (A): missing in the original inv-V compiler,
    # caused Abduction Δ up to 0.65.  Added here to match the full 4-term
    # expansion of ln P(a,b,c) for multi-parent DAGs.
    if abs(mu_left) > 1e-10:
        factors.append(SpinEBMFactor([blocks[0]], jnp.array([mu_left])))

    # Right → Center: scale J only
    J_right, h_corr_right = ising_params(right_strength, right_background)
    mu_right = _parent_bias(right_strength, right_background)
    scale_right = g_fn(_confidence_to_n(right_confidence))
    if abs(J_right * scale_right) > 1e-10:
        factors.append(SpinEBMFactor(
            [blocks[2], blocks[1]], jnp.array([J_right * scale_right])))
    if abs(h_corr_right) > 1e-10:
        factors.append(SpinEBMFactor([blocks[1]], jnp.array([h_corr_right])))
    if abs(mu_right) > 1e-10:
        factors.append(SpinEBMFactor([blocks[2]], jnp.array([mu_right])))

    return _assemble_binary_graph_ex(
        spins, blocks, blocks, [], factors, root_prior=None)


# ═══════════════════════════════════════════════════════════════════════════
#  Modulated inverted-V with hidden unit (Abduction explaining-away)
# ═══════════════════════════════════════════════════════════════════════════

def compile_modulated_inv_v_with_hidden(
        left_prior, right_prior,
        left_strength, right_strength,
        left_confidence, right_confidence,
        left_background=DEFAULT_EPSILON,
        right_background=DEFAULT_EPSILON,
        center_prior=0.5, hidden_strength=1.5, g_fn=None):
    """Inverted-V with LBM hidden unit for explaining-away.

    Topology:
        visible:  A ──J_AC── C ──J_BC── B
                                │
        hidden:                h

    The hidden unit implements causal competition (explaining-away):
    when the effect C is observed true, the hidden unit activates and
    suppresses both causes A and B — they compete to explain C.

        J_hC = +h_str    (C true → activates h)
        J_hA = -h_str    (h active → suppresses A)
        J_hB = -h_str    (h active → suppresses B)
        bias_h = -0.5 × h_str  (threshold: C=+1 easily activates h)

    This gives standard Δ=0.052 (from 0.234 without hidden).
    Asymmetric cases remain a structural limitation of binary Ising
    (needs pdit d>2 for full four-parameter interaction).

    Parameters
    ----------
    hidden_strength : float
        Coupling strength for the hidden explaining-away unit.
        Higher values enforce stronger explaining-away constraint.
    """
    if g_fn is None:
        g_fn = default_g_fn

    # 3 visible + 1 hidden
    vis_spins = [SpinNode() for _ in range(3)]   # [A=left, C=center, B=right]
    hid_spin = SpinNode()
    all_spins = vis_spins + [hid_spin]

    vis_blocks = [Block([s]) for s in vis_spins]
    hid_block = Block([hid_spin])
    all_blocks = vis_blocks + [hid_block]

    factors = []

    # Visible biases
    for i, p in enumerate([left_prior, center_prior, right_prior]):
        h = prior_bias(p)
        if abs(h) > 1e-10:
            factors.append(SpinEBMFactor([vis_blocks[i]], jnp.array([h])))

    # Left(A) → Center(C): scale J only, NOT h_corr (same inv-V reason)
    J_left, h_corr_left = ising_params(left_strength, left_background)
    mu_left = _parent_bias(left_strength, left_background)
    scale_left = g_fn(_confidence_to_n(left_confidence))
    if abs(J_left * scale_left) > 1e-10:
        factors.append(SpinEBMFactor(
            [vis_blocks[0], vis_blocks[1]], jnp.array([J_left * scale_left])))
    if abs(h_corr_left) > 1e-10:
        factors.append(SpinEBMFactor(
            [vis_blocks[1]], jnp.array([h_corr_left])))
    if abs(mu_left) > 1e-10:
        factors.append(SpinEBMFactor(
            [vis_blocks[0]], jnp.array([mu_left])))

    # Right(B) → Center(C): scale J only
    J_right, h_corr_right = ising_params(right_strength, right_background)
    mu_right = _parent_bias(right_strength, right_background)
    scale_right = g_fn(_confidence_to_n(right_confidence))
    if abs(J_right * scale_right) > 1e-10:
        factors.append(SpinEBMFactor(
            [vis_blocks[2], vis_blocks[1]], jnp.array([J_right * scale_right])))
    if abs(h_corr_right) > 1e-10:
        factors.append(SpinEBMFactor(
            [vis_blocks[1]], jnp.array([h_corr_right])))
    if abs(mu_right) > 1e-10:
        factors.append(SpinEBMFactor(
            [vis_blocks[2]], jnp.array([mu_right])))

    # Hidden unit: causal competition (explaining-away)
    # C true → h activates → suppresses both A and B (they compete)
    h_str = float(hidden_strength)

    # J_hC = +h_str (C true → activates hidden)
    if abs(h_str) > 1e-10:
        factors.append(SpinEBMFactor(
            [hid_block, vis_blocks[1]], jnp.array([h_str])))

    # J_hA = -h_str (h active → suppresses A)
    if abs(h_str) > 1e-10:
        factors.append(SpinEBMFactor(
            [hid_block, vis_blocks[0]], jnp.array([-h_str])))

    # J_hB = -h_str (h active → suppresses B)
    if abs(h_str) > 1e-10:
        factors.append(SpinEBMFactor(
            [hid_block, vis_blocks[2]], jnp.array([-h_str])))

    # Hidden bias = -0.5 × h_str (C=+1 easily activates h)
    h_bias = -0.5 * h_str
    if abs(h_bias) > 1e-10:
        factors.append(SpinEBMFactor([hid_block], jnp.array([h_bias])))

    # All nodes free (abduction queries all visible)
    free_blocks = all_blocks
    clamped_blocks = []

    return _assemble_binary_graph_ex(
        all_spins, all_blocks, free_blocks, clamped_blocks, factors,
        root_prior=None)
