"""
pln_thrml.unified — LBM(s) on TSU + QLN(n) on CPU
====================================================

Pure (ρ, n) separation orchestrator:
  - **s layer (TSU)**: Binary Ising sampling with g=1 (no precision modulation)
  - **n layer (CPU)**: QLN-style closed-form confidence propagation

n is the precision itself (= Active Inference ζ). No g(n) cross-talk
between layers — each edge's 2×2 CPT is exactly encoded by Ising (J, h)
from strength alone. Confidence propagates independently via PLN formulas.

CPU-only rules (Inversion, Revision) delegate directly to qln_cpu.
"""

import math

from pln_thrml.beta import (
    DEFAULT_EPSILON, c2w, w2c,
    build_beta_inv_v_graph, run_beta_sampling,
    estimate_beta_marginal, estimate_beta_conditional,
    DEFAULT_BETA_N_BATCHES,
)
from pln_thrml.compiler_unified import (
    compile_modulated_chain,
    compile_modulated_inv_v,
    compile_modulated_inv_v_with_hidden,
)

# Pure separation: g=1 always. No precision modulation across layers.
_G_IDENTITY = lambda n: 1.0
from pln_thrml.compiler_binary import (
    compile_binary_chain_with_hidden,
    run_binary_sampling,
    estimate_binary_marginal,
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


def _joint_ising_params(s_parent, s_link, background=DEFAULT_EPSILON):
    """Compute BOTH Ising biases from the full joint distribution.

    Unlike prior_bias(s) which assumes the node is isolated, this
    computes h_parent that gives the correct marginal P(parent)
    even in the presence of coupling J.  This is what LBM does
    implicitly — encode the correct joint, not just the conditional.

    Returns (J, h_parent, h_child).
    """
    s = max(min(float(s_link), 1.0 - 1e-7), 1e-7)
    bg = max(min(float(background), 1.0 - 1e-7), 1e-7)
    sp = max(min(float(s_parent), 1.0 - 1e-7), 1e-7)

    # Full joint: P(A,B) = P(B|A) × P(A)
    p_tt = sp * s
    p_tf = sp * (1.0 - s)
    p_ft = (1.0 - sp) * bg
    p_ff = (1.0 - sp) * (1.0 - bg)

    J = 0.25 * math.log(p_tt * p_ff / max(p_tf * p_ft, 1e-30))
    h_parent = 0.25 * math.log(p_tt * p_tf / max(p_ft * p_ff, 1e-30))
    h_child = 0.25 * math.log(p_tt * p_ft / max(p_tf * p_ff, 1e-30))

    return J, h_parent, h_child


def _binary_conditional(samples_dict, graph, target_idx, condition_idx):
    """Estimate P(target=T | condition=T) from binary Ising samples.

    For inv-V abduction: P(B|A) from joint samples of all-free graph.
    """
    import jax.numpy as jnp

    vmap_samples = samples_dict["vmap_samples"]
    # All-free graph: node_idx maps directly to vmap_samples index
    target = vmap_samples[target_idx][:, :, 0]      # [n_batches, n_samples]
    condition = vmap_samples[condition_idx][:, :, 0]  # [n_batches, n_samples]

    # Spin values are 0.0 or 1.0 (mapped from {-1, +1})
    cond_true = condition > 0.5
    both_true = cond_true & (target > 0.5)

    n_cond = float(jnp.sum(cond_true))
    if n_cond < 1.0:
        return 0.5  # fallback if condition is never true
    return float(jnp.sum(both_true)) / n_cond


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

    The dominant error is the Jensen gap from (ρ,n) separation: binary
    Ising encodes E[s] but the conditional is nonlinear in s, so
    f(E[s]) ≠ E[f(s)].  This is the true cost of mean-field separation.

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
                      background=DEFAULT_EPSILON,
                      seed=42, n_batches=None,
                      max_rounds=DEFAULT_MAX_ROUNDS,
                      tol=DEFAULT_TOL, g_fn=None,
                      method="kbin",
                      k=16,
                      hidden_strength=1.5,
                      mfc_lambda=1.0, mfc_alpha=0.3):
    """Pure-separation Abduction with four method options.

    Parameters
    ----------
    method : str
        "kbin"          — K-bin CategoricalNode inv-V (pdit native, highest accuracy)
        "mfc"           — Binary Ising + Mean-Field Constraint bias feedback
        "binary"        — Binary Ising inv-V, no hidden (baseline)
        "binary_hidden" — Binary Ising inv-V + hidden explaining-away unit

    Returns (s_AB, c_AB, meta).
    """

    # n-path: c from PLN formula (same for all methods)
    c_AB = c_abduction(s_AC, c_AC, s_BC, c_BC)

    # ── K-bin CategoricalNode (pdit native) ──────────────────────────
    if method == "kbin":
        if n_batches is None:
            n_batches = DEFAULT_BETA_N_BATCHES
        graph = build_beta_inv_v_graph(
            left_prior=s_A, left_confidence=c_A,
            right_prior=s_B, right_confidence=c_B,
            left_strength=s_AC, right_strength=s_BC,
            left_impl_confidence=c_AC, right_impl_confidence=c_BC,
            left_background=background, right_background=background,
            k=k)
        samples = run_beta_sampling(graph, seed=seed, n_batches=n_batches)
        # Query P(B|A) — the abduction result — not P(C)
        _, s_AB_sampled, _ = estimate_beta_conditional(
            samples, graph, target=graph["right"], condition=graph["left"])
        return s_AB_sampled, c_AB, _iterative_meta(
            [(s_AB_sampled, c_AB)], converged=True)

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
        # Build all-free 2-node graph with CORRECT joint distribution.
        # Use _joint_ising_params to get biases that account for coupling,
        # unlike prior_bias() which assumes isolated nodes.
        from pln_thrml.compiler_binary import (
            SpinNode, SpinEBMFactor, Block, _assemble_binary_graph_ex,
        )
        import jax.numpy as jnp

        # Use PLN base rate so P(B) in joint = s_B (independent evidence)
        bg_raw = (float(s_B) - float(s_A) * float(s_AB)) / max(1.0 - float(s_A), 1e-7)
        bg = max(min(bg_raw, 0.98), 0.01) if 0.0 < bg_raw < 1.0 else background
        J, h_A, h_B = _joint_ising_params(s_A, s_AB, bg)
        spins = [SpinNode(), SpinNode()]
        blocks = [Block([s]) for s in spins]
        factors = []
        if abs(J) > 1e-10:
            factors.append(SpinEBMFactor(
                [blocks[0], blocks[1]], jnp.array([J])))
        if abs(h_A) > 1e-10:
            factors.append(SpinEBMFactor([blocks[0]], jnp.array([h_A])))
        if abs(h_B) > 1e-10:
            factors.append(SpinEBMFactor([blocks[1]], jnp.array([h_B])))
        graph = _assemble_binary_graph_ex(
            spins, blocks, blocks, [], factors, root_prior=None)

        samples = run_binary_sampling(graph, seed=seed, n_batches=n_batches)
        # P(A|B): target=A(node0), condition=B(node1)
        s_BA = _binary_conditional(samples, graph, 0, 1)
        # n-path: CPU
        c_BA = inversion_bayes(s_A, c_A, s_B, c_B, s_AB, c_AB)[1]
        return s_BA, c_BA
    else:
        raise ValueError(f"Unknown inversion method: {method}")


# ═══════════════════════════════════════════════════════════════════════════
#  Revision (CPU-only, QLN formula)
# ═══════════════════════════════════════════════════════════════════════════

def unified_revision(s1, c1, s2, c2):
    """Unified Revision: CPU-only, QLN formula n_rev = n₁ + n₂.

    Returns
    -------
    (s_rev, c_rev) : tuple[float, float]
    """
    return revision(s1, c1, s2, c2)
