"""
pln_thrml.unified — LBM(s) on TSU + QLN(n) on CPU
====================================================

Unified orchestrator: strength on TSU, evidence count on CPU.
  - **s layer (TSU)**: Binary Ising sampling with g=1 (no precision modulation)
  - **n layer (CPU)**: QLN-style closed-form confidence propagation

n is the precision itself (= Active Inference ζ). No g(n) cross-talk
between layers — each edge's 2×2 CPT is exactly encoded by Ising (J, h)
from strength alone. Confidence propagates independently via PLN formulas.

CPU-only rules (Inversion, Revision) delegate directly to qln_cpu.
"""

import math

from pln_thrml.pln_utils import DEFAULT_EPSILON
from pln_thrml.compiler_unified import (
    compile_modulated_chain,
    compile_modulated_inv_v,
    compile_modulated_inv_v_with_hidden,
)

# Pure separation: g=1 always. No precision modulation across layers.
_G_IDENTITY = lambda n: 1.0
from pln_thrml.compiler_binary import (
    compile_binary_joint_2node,
    run_binary_sampling,
    estimate_binary_marginal,
    _binary_conditional,
    DEFAULT_BINARY_N_BATCHES,
)
from pln_thrml.qln_cpu import (
    c_modus_ponens,
    c_deduction,
    c_abduction,
    inversion_bayes,
    inversion_pln,
    revision,
)


__all__ = [
    "unified_modus_ponens",
    "unified_deduction",
    "unified_abduction",
    "unified_inversion",
    "unified_revision",
]


DEFAULT_MAX_ROUNDS = 3
DEFAULT_TOL = 0.01


def _iterative_meta(history, converged):
    """Build metadata dict for iterative functions."""
    return {
        "rounds": len(history),
        "converged": converged,
        "history": history,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Modus Ponens (TSU + CPU)
# ═══════════════════════════════════════════════════════════════════════════

def unified_modus_ponens(s_A, c_A, s_AB, c_AB,
                         background=DEFAULT_EPSILON,
                         seed=42, n_batches=None,
                         max_rounds=1, tol=DEFAULT_TOL, g_fn=None):
    """Pure-separation Modus Ponens: binary Ising (g=1) + PLN formula.

    TSU: exact 2×2 CPT → Ising (J, h), single-round sampling.
    CPU: c_B = c_A × c_AB.

    Returns (s_B, c_B, meta).
    """
    if n_batches is None:
        n_batches = DEFAULT_BINARY_N_BATCHES

    c_B = c_modus_ponens(c_A, c_AB)

    graph = compile_modulated_chain(
        priors=[s_A, 0.5],
        strengths=[s_AB],
        confidences=[c_AB],
        backgrounds=[background],
        g_fn=_G_IDENTITY,
        clamp_root=True)
    samples = run_binary_sampling(graph, seed=seed, n_batches=n_batches)
    s_B = estimate_binary_marginal(samples, graph, 1)

    return s_B, c_B, _iterative_meta([(s_B, c_B)], converged=True)


# ═══════════════════════════════════════════════════════════════════════════
#  Deduction (TSU + CPU)
# ═══════════════════════════════════════════════════════════════════════════

def unified_deduction(s_A, c_A, s_B, c_B, s_C, c_C,
                      s_AB, c_AB, s_BC, c_BC,
                      background=DEFAULT_EPSILON,
                      seed=42, n_batches=None,
                      max_rounds=1, tol=DEFAULT_TOL, g_fn=None):
    """Pure-separation Deduction: binary Ising (g=1) 3-node chain + PLN formula.

    TSU: single 3-node chain A→B→C, clamp A, simultaneously sample B+C.
    CPU: c_AC = s_AB × s_BC × c_AB × c_BC.

    The dominant error is a Jensen gap: binary Ising encodes E[s] but
    the conditional is nonlinear in s, so f(E[s]) ≠ E[f(s)]. This is
    the true cost of mean-field separation between s and n.

    Returns (s_AC, c_AC, meta).
    """
    if n_batches is None:
        n_batches = DEFAULT_BINARY_N_BATCHES

    c_AC = c_deduction(s_AB, c_AB, s_BC, c_BC)

    # PLN base rates: P(child | parent=False) — structural parameters.
    bg_AB_raw = ((float(s_B) - float(s_A) * float(s_AB))
                 / max(1.0 - float(s_A), 1e-7))
    bg_BC_raw = ((float(s_C) - float(s_B) * float(s_BC))
                 / max(1.0 - float(s_B), 1e-7))
    bg_AB = max(min(bg_AB_raw, 0.98), 0.01) if 0.0 < bg_AB_raw < 1.0 else background
    bg_BC = max(min(bg_BC_raw, 0.98), 0.01) if 0.0 < bg_BC_raw < 1.0 else background

    graph = compile_modulated_chain(
        priors=[s_A, 0.5, 0.5],
        strengths=[s_AB, s_BC],
        confidences=[c_AB, c_BC],
        backgrounds=[bg_AB, bg_BC],
        g_fn=_G_IDENTITY,
        clamp_root=True)
    samples = run_binary_sampling(graph, seed=seed, n_batches=n_batches)
    s_C_inferred = estimate_binary_marginal(samples, graph, 2)

    return s_C_inferred, c_AC, _iterative_meta(
        [(s_C_inferred, c_AC)], converged=True)


# ═══════════════════════════════════════════════════════════════════════════
#  Abduction (TSU + CPU)
# ═══════════════════════════════════════════════════════════════════════════

def _pln_abduction_formula(s_A, s_B, s_AC, s_BC):
    """PLN point-estimate abduction strength (deterministic)."""
    denom = max(1.0 - float(s_A), 1e-7)
    return (float(s_AC) * float(s_BC)
            + (1.0 - float(s_AC)) * (float(s_B) - float(s_A) * float(s_BC))
            / denom)


def _bias_to_prior(bias_correction):
    """Convert MFC bias correction to prior strength for recompilation."""
    return 1.0 / (1.0 + math.exp(-2.0 * bias_correction))


def unified_abduction(s_A, c_A, s_B, c_B,
                      s_AC, c_AC, s_BC, c_BC,
                      s_C=None, c_C=None,
                      background=DEFAULT_EPSILON,
                      seed=42, n_batches=None,
                      max_rounds=DEFAULT_MAX_ROUNDS,
                      tol=DEFAULT_TOL, g_fn=None,
                      method="chain",
                      hidden_strength=1.5,
                      mfc_lambda=1.0, mfc_alpha=0.3):
    """Pure-separation Abduction.

    Default method "chain" implements PLN's formal definition
    (Ch 5.4): Abduction = Deduction ∘ Inversion.  Given edges A→C
    and B→C, Bayes-invert B→C to C→B, then run Deduction chain
    A→C→B to query P(B|A).

    Parameters
    ----------
    s_C, c_C : float or None
        Prior strength/confidence of the shared effect C.  PLN book
        Ch 5.4 abduction formally requires P(C) as input.  If None,
        estimated via Noisy-OR: s_C ≈ 1 − (1−s_A·s_AC)·(1−s_B·s_BC).
    method : str
        "chain"         — Deduction∘Inversion composition (default, PLN Ch 5.4)
        "mfc"           — Legacy: Binary Ising + MFC bias feedback
        "binary"        — Legacy: Binary Ising inv-V, no hidden (baseline)
        "binary_hidden" — Legacy: Binary Ising inv-V + hidden explaining-away

    Returns (s_AB, c_AB, meta).
    """
    # Estimate s_C via Noisy-OR if not provided (PLN book Ch 5.4 requires it)
    if s_C is None:
        s_C = 1.0 - (1.0 - float(s_A) * float(s_AC)) * (1.0 - float(s_B) * float(s_BC))
    if c_C is None:
        c_C = min(float(c_A), float(c_B), float(c_AC), float(c_BC))

    # ── Chain method: Abduction = Deduction ∘ Inversion (PLN Ch 5.4) ──
    if method == "chain":
        import numpy as np
        from pln_thrml.compiler_binary import compile_binary_chain
        # Step 1: Bayes-invert B→C into C→B
        s_CB, c_CB = inversion_bayes(
            s_A=s_B, c_A=c_B, s_B=s_C, c_B=c_C,
            s_AB=s_BC, c_AB=c_BC)
        # Step 2: Deduction chain A→C_shared→B with correct priors
        # Priors [s_A (clamped root), s_C (shared effect), s_B (target)]
        if n_batches is None:
            n_batches = DEFAULT_BINARY_N_BATCHES
        graph = compile_binary_chain(
            [float(s_A), float(s_C), float(s_B)],
            [float(s_AC), float(s_CB)],
            [background, background],
            clamp_root=True, include_parent_bias=True)
        samples = run_binary_sampling(graph, seed=seed, n_batches=n_batches)
        root_bits = np.asarray(samples["root_bits"]).flatten()
        A_pos = root_bits > 0
        if int(A_pos.sum()) == 0:
            s_AB = float("nan")
        else:
            B_all = np.asarray(samples["vmap_samples"][1][:, :, 0])
            B_sel = B_all[A_pos]
            s_AB = float((B_sel > 0).mean())
        # n-path confidence from PLN abduction formula
        c_AB = c_abduction(s_AC, c_AC, s_BC, c_BC)
        return s_AB, c_AB, _iterative_meta([(s_AB, c_AB)], converged=True)

    # n-path: c from PLN formula (same for all legacy methods)
    c_AB = c_abduction(s_AC, c_AC, s_BC, c_BC)

    # ── MFC: binary Ising + mean-field constraint bias feedback ──────
    if method == "mfc":
        if n_batches is None:
            n_batches = DEFAULT_BINARY_N_BATCHES
        bias_correction = 0.0
        history = []

        for rnd in range(max_rounds):
            center_prior = _bias_to_prior(bias_correction) if rnd > 0 else 0.5
            graph = compile_modulated_inv_v(
                left_prior=s_A, right_prior=s_B,
                left_strength=s_AC, right_strength=s_BC,
                left_confidence=c_AC, right_confidence=c_BC,
                left_background=background,
                right_background=background,
                center_prior=center_prior,
                g_fn=_G_IDENTITY)
            samples = run_binary_sampling(
                graph, seed=seed + rnd * 1000, n_batches=n_batches)
            # Query P(B|A) from joint samples: A=node0, B=node2
            s_AB_sampled = _binary_conditional(samples, graph, 2, 0)
            history.append((s_AB_sampled, c_AB))

            # MFC: CPU computes constraint error → low-pass filtered bias
            s_expected = _pln_abduction_formula(s_A, s_B, s_AC, s_BC)
            epsilon = s_expected - s_AB_sampled
            bias_correction = (mfc_alpha * bias_correction
                               + (1.0 - mfc_alpha) * mfc_lambda * epsilon)

            if rnd > 0 and abs(s_AB_sampled - history[-2][0]) < tol:
                return (s_AB_sampled, c_AB,
                        _iterative_meta(history, converged=True))

        return s_AB_sampled, c_AB, _iterative_meta(history, converged=False)

    # ── Binary Ising (with or without hidden unit) ───────────────────
    if n_batches is None:
        n_batches = DEFAULT_BINARY_N_BATCHES
    history = []
    s_prev = None

    for rnd in range(max_rounds):
        if method == "binary_hidden":
            graph = compile_modulated_inv_v_with_hidden(
                left_prior=s_A, right_prior=s_B,
                left_strength=s_AC, right_strength=s_BC,
                left_confidence=c_AC, right_confidence=c_BC,
                left_background=background,
                right_background=background,
                center_prior=0.5,
                hidden_strength=hidden_strength,
                g_fn=_G_IDENTITY)
        else:  # "binary"
            graph = compile_modulated_inv_v(
                left_prior=s_A, right_prior=s_B,
                left_strength=s_AC, right_strength=s_BC,
                left_confidence=c_AC, right_confidence=c_BC,
                left_background=background,
                right_background=background,
                center_prior=0.5,
                g_fn=_G_IDENTITY)

        samples = run_binary_sampling(
            graph, seed=seed + rnd * 1000, n_batches=n_batches)
        # Binary methods: query P(C) center marginal (node 1).
        # P(B|A) conditional estimation doesn't work for binary
        # (prior dominance → always ~0.99).
        s_center = estimate_binary_marginal(samples, graph, 1)
        history.append((s_center, c_AB))

        if s_prev is not None and abs(s_center - s_prev) < tol:
            return (s_center, c_AB,
                    _iterative_meta(history, converged=True))
        s_prev = s_center

    return s_center, c_AB, _iterative_meta(
        history, converged=(max_rounds <= 1))


# ═══════════════════════════════════════════════════════════════════════════
#  Inversion (CPU-only)
# ═══════════════════════════════════════════════════════════════════════════

def unified_inversion(s_A, c_A, s_B, c_B, s_AB, c_AB, method="bayes",
                      seed=42, n_batches=None, background=DEFAULT_EPSILON):
    """Unified Inversion: P(B→A) from P(A→B).

    Parameters
    ----------
    method : str
        "bayes"  — CPU Bayes formula (default, highest precision)
        "pln"    — upstream PLN heuristic (lib_pln.metta:150)
        "binary" — TSU: all-free 2-node graph + conditional estimation

    Returns
    -------
    (s_BA, c_BA) : tuple[float, float]
    """
    if method == "bayes":
        return inversion_bayes(s_A, c_A, s_B, c_B, s_AB, c_AB)
    elif method == "pln":
        return inversion_pln(s_A, c_A, s_B, c_B, s_AB, c_AB)
    elif method == "binary":
        if n_batches is None:
            n_batches = DEFAULT_BINARY_N_BATCHES
        # Use PLN base rate so P(B) in joint = s_B (independent evidence)
        bg_raw = (float(s_B) - float(s_A) * float(s_AB)) / max(1.0 - float(s_A), 1e-7)
        bg = max(min(bg_raw, 0.98), 0.01) if 0.0 < bg_raw < 1.0 else background
        graph = compile_binary_joint_2node(s_A, s_AB, bg)
        samples = run_binary_sampling(graph, seed=seed, n_batches=n_batches)
        # P(A|B): target=A(node0), condition=B(node1)
        s_BA = _binary_conditional(samples, graph, 0, 1)
        # n-path: CPU
        c_BA = inversion_bayes(s_A, c_A, s_B, c_B, s_AB, c_AB)[1]
        return s_BA, c_BA
    else:
        raise ValueError(f"Unknown inversion method: {method}")


# ═══════════════════════════════════════════════════════════════════════════
#  Revision (CPU-only, PLN book formula)
# ═══════════════════════════════════════════════════════════════════════════

def unified_revision(s1, c1, s2, c2):
    """Unified Revision: CPU-only, PLN book formula n_rev = n₁ + n₂ (raw counts).

    Returns
    -------
    (s_rev, c_rev) : tuple[float, float]
    """
    return revision(s1, c1, s2, c2)
