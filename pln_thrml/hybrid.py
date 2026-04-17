"""
pln_thrml.hybrid — (ρ, n) separation: binary Ising s + PLN formula c
=====================================================================

Orchestrates the hybrid PLN architecture:
  - **s (strength)** from binary Ising sampling (1 pbit per proposition)
  - **c (confidence)** from PLN closed-form formulas (deterministic algebra)

These two computations have NO data dependency and can run in parallel on
different hardware (TSU for s, GPU/CPU for c).

Currently covers: Modus Ponens, Deduction, Abduction, Inversion.
"""

from pln_thrml.compiler_binary import (
    compile_binary_chain,
    compile_binary_inv_v,
    compile_binary_chain_with_hidden,
    compile_binary_joint_2node,
    compile_joint_categorical,
    run_binary_sampling,
    run_joint_sampling,
    estimate_binary_marginal,
    estimate_joint_marginal,
    estimate_joint_conditional,
    _binary_conditional,
    DEFAULT_BINARY_N_BATCHES,
)
from pln_thrml.pln_utils import DEFAULT_EPSILON, c2w, EPS
from pln_thrml.qln_cpu import inversion_bayes


__all__ = [
    "hybrid_modus_ponens",
    "hybrid_deduction",
    "hybrid_deduction_corrected",
    "hybrid_deduction_laplace",
    "hybrid_deduction_tempered",
    "hybrid_deduction_pmode",
    "hybrid_deduction_joint",
    "hybrid_deduction_pbit_joint",
    "hybrid_deduction_hidden",
    "hybrid_abduction",
    "hybrid_inversion",
]


# ═══════════════════════════════════════════════════════════════════════════
#  PLN confidence formulas (from vendor/PLN/lib_pln.metta)
# ═══════════════════════════════════════════════════════════════════════════

def _c_modus_ponens(c_A, c_AB):
    """PLN Modus Ponens confidence: c_B = c_A × c_AB."""
    return c_A * c_AB


def _c_deduction(s_AB, c_AB, s_BC, c_BC):
    """PLN Deduction confidence: c_AC = s_AB × s_BC × c_AB × c_BC."""
    return s_AB * s_BC * c_AB * c_BC


def _c_abduction(s_AC, c_AC, s_BC, c_BC):
    """PLN Abduction confidence (simplified): c_AB = s_AC × s_BC × c_AC × c_BC."""
    return s_AC * s_BC * c_AC * c_BC


def _var_from_stv(s, c):
    """Variance of a Beta distribution parameterised by (s, c).

    Var(X) = s(1-s) / (n+1), where n = c2w(c) + 2.
    """
    c_safe = min(float(c), 0.9999)
    n = c2w(c_safe) + 2.0
    return float(s) * (1.0 - float(s)) / (n + 1.0)


def _jensen_correction_deduction(s_B, c_B, s_C, s_BC):
    """Second-order Jensen correction for the PLN deduction base-rate term.

    The PLN deduction formula contains h(s_B) = (s_C - s_B·s_BC)/(1-s_B),
    which is nonlinear in s_B.  When s_B is uncertain (low c_B → high Var),
    E[h(s_B)] ≠ h(E[s_B]).

    The correction is:  ½ · h''(s_B) · Var(s_B)
    where h''(x) = 2(s_C - s_BC) / (1-x)³
    and   Var(s_B) = s_B(1-s_B) / (n_B+1).

    This uses n_B (from c_B) to correct ρ — the mechanism by which
    confidence count n influences the strength computation in the
    (ρ, n) separation architecture.  Partially answers QLN Open Problem #4.
    """
    s_B = float(s_B)
    denom = max(1.0 - s_B, EPS)
    h_pp = 2.0 * (float(s_C) - float(s_BC)) / (denom ** 3)
    var_B = _var_from_stv(s_B, c_B)
    return 0.5 * h_pp * var_B


def _laplace_deduction(s_A, c_A, s_B, c_B, s_C, c_C,
                       s_AB, c_AB, s_BC, c_BC, background=DEFAULT_EPSILON):
    """Laplace-approximated deduction: reconstruct Gaussians from (s, n),
    propagate analytically, extract output (s, c).

    Each input (s_i, c_i) → N(μ=s_i, σ²=s_i(1-s_i)/(n_i+1)).
    The deduction formula f is Taylor-expanded to second order at the
    operating point (mean field).  Output mean and variance give (s, c).

    This implements the Active Inference Laplace approximation for the
    (ρ, n) separation architecture: n information is injected into s
    via variance, and output n is derived from output variance.
    """
    s_A, s_B, s_C = float(s_A), float(s_B), float(s_C)
    s_AB, s_BC = float(s_AB), float(s_BC)

    # Reconstruct Gaussians: σ² from (s, c) for each input
    var_A = _var_from_stv(s_A, c_A)
    var_B = _var_from_stv(s_B, c_B)
    var_C = _var_from_stv(s_C, c_C)
    var_AB = _var_from_stv(s_AB, c_AB)
    var_BC = _var_from_stv(s_BC, c_BC)

    # PLN deduction: f(s_AB, s_BC, s_B, s_C) = s_AB*s_BC + (1-s_AB)*h(s_B)
    # where h(x) = (s_C - x*s_BC)/(1-x)
    denom_B = max(1.0 - s_B, EPS)
    h_val = (s_C - s_B * s_BC) / denom_B
    f_mu = s_AB * s_BC + (1.0 - s_AB) * h_val  # mean field point estimate

    # ── Second-order correction for s (E[f] ≈ f(μ) + ½ Σ f'' σ²) ──

    # Partial second derivatives at operating point:
    # ∂²f/∂s_B² : h''(s_B) = 2(s_C - s_BC)/(1-s_B)³, scaled by (1-s_AB)
    h_pp = 2.0 * (s_C - s_BC) / (denom_B ** 3)
    corr_B = 0.5 * (1.0 - s_AB) * h_pp * var_B

    # ∂²f/∂s_AB² = 0 (f is linear in s_AB)
    # ∂²f/∂s_BC² = 0 (f is linear in s_BC)
    # ∂²f/∂s_C²  = 0 (f is linear in s_C)

    # Cross term ∂²f/∂s_AB∂s_BC = 1 (from s_AB*s_BC term)
    # But inputs are independent → cross variance = 0, no contribution
    # Cross term ∂²f/∂s_AB∂h = -h(s_B), but this mixes with s_B correction

    s_out = f_mu + corr_B

    # ── Variance propagation for output n (Var[f] ≈ Σ (∂f/∂xᵢ)² σ²ᵢ) ──

    # ∂f/∂s_AB = s_BC - h(s_B) = s_BC - (s_C - s_B*s_BC)/(1-s_B)
    df_dAB = s_BC - h_val

    # ∂f/∂s_BC = s_AB + (1-s_AB)*(-s_B)/(1-s_B)... wait, let me redo
    # f = s_AB*s_BC + (1-s_AB)*(s_C - s_B*s_BC)/(1-s_B)
    # ∂f/∂s_BC = s_AB + (1-s_AB)*(-s_B)/(1-s_B)
    df_dBC = s_AB + (1.0 - s_AB) * (-s_B) / denom_B

    # ∂f/∂s_B = (1-s_AB) * h'(s_B)
    # h'(x) = [-s_BC(1-x) + (s_C - x*s_BC)] / (1-x)² = (s_C - s_BC)/(1-x)²
    h_p = (s_C - s_BC) / (denom_B ** 2)
    df_dB = (1.0 - s_AB) * h_p

    # ∂f/∂s_C = (1-s_AB) / (1-s_B)
    df_dC = (1.0 - s_AB) / denom_B

    # ∂f/∂s_A = 0 (s_A not in the deduction formula for s_AC)

    var_out = (df_dAB ** 2 * var_AB
               + df_dBC ** 2 * var_BC
               + df_dB ** 2 * var_B
               + df_dC ** 2 * var_C)

    # Convert output variance → output n → output c
    s_out_safe = max(min(s_out, 1.0 - EPS), EPS)
    if var_out > EPS:
        n_out = s_out_safe * (1.0 - s_out_safe) / var_out - 1.0
        n_out = max(n_out, 0.0)
    else:
        n_out = 1e4  # very high precision

    from pln_thrml.pln_utils import w2c
    c_out = w2c(max(n_out - 2.0, 0.0))

    return s_out, c_out


# ═══════════════════════════════════════════════════════════════════════════
#  Hybrid inference functions
# ═══════════════════════════════════════════════════════════════════════════

def hybrid_modus_ponens(s_A, c_A, s_AB, c_AB,
                        background=DEFAULT_EPSILON,
                        seed=42, n_batches=None):
    """Hybrid Modus Ponens: s from binary Ising, c from PLN formula.

    Given A=(s_A, c_A) and link A→B=(s_AB, c_AB), infer B=(s_B, c_B).

    Parameters
    ----------
    s_A, c_A : float
        Premise truth value.
    s_AB, c_AB : float
        Implication link truth value.
    background : float
        Background rate (PLN epsilon).
    seed : int
        Random seed for sampling.

    Returns
    -------
    (s_B, c_B) : tuple[float, float]
        Inferred truth value for B.
    """
    if n_batches is None:
        n_batches = DEFAULT_BINARY_N_BATCHES

    # ρ-path: binary Ising sampling for s
    graph = compile_binary_chain(
        [s_A, 0.5], [s_AB], [background], clamp_root=True)
    samples = run_binary_sampling(graph, seed=seed, n_batches=n_batches)
    s_B = estimate_binary_marginal(samples, graph, 1)

    # n-path: PLN closed-form formula for c
    c_B = _c_modus_ponens(c_A, c_AB)

    return s_B, c_B


def hybrid_deduction(s_A, c_A, s_B, c_B, s_C, c_C,
                     s_AB, c_AB, s_BC, c_BC,
                     background=DEFAULT_EPSILON,
                     seed=42, n_batches=None):
    """Hybrid Deduction: s from chained binary MP, c from PLN formula.

    Given A→B→C chain, infer the composed link A→C.
    Uses two chained clamped-root MP calls for s (compositional).

    Parameters
    ----------
    s_A, c_A : float
        Premise A truth value.
    s_B, c_B : float
        Intermediate B truth value (used as prior context).
    s_C, c_C : float
        Conclusion C truth value (prior, typically weak).
    s_AB, c_AB : float
        Link A→B truth value.
    s_BC, c_BC : float
        Link B→C truth value.

    Returns
    -------
    (s_AC, c_AC) : tuple[float, float]
        Inferred truth value for the composed link A→C.
    """
    if n_batches is None:
        n_batches = DEFAULT_BINARY_N_BATCHES

    # ρ-path: chain two clamped-root MP calls
    # Step 1: A → B  (clamp A, sample B)
    graph1 = compile_binary_chain(
        [s_A, 0.5], [s_AB], [background], clamp_root=True)
    samples1 = run_binary_sampling(graph1, seed=seed, n_batches=n_batches)
    s_B_inferred = estimate_binary_marginal(samples1, graph1, 1)

    # Step 2: B → C  (clamp B using inferred s_B, sample C)
    graph2 = compile_binary_chain(
        [s_B_inferred, 0.5], [s_BC], [background], clamp_root=True)
    samples2 = run_binary_sampling(graph2, seed=seed + 1000, n_batches=n_batches)
    s_C_inferred = estimate_binary_marginal(samples2, graph2, 1)

    # n-path: PLN closed-form formula for c
    c_AC = _c_deduction(s_AB, c_AB, s_BC, c_BC)

    return s_C_inferred, c_AC


def hybrid_deduction_corrected(s_A, c_A, s_B, c_B, s_C, c_C,
                               s_AB, c_AB, s_BC, c_BC,
                               background=DEFAULT_EPSILON,
                               seed=42, n_batches=None):
    """Hybrid Deduction with second-order Jensen correction.

    Same as hybrid_deduction, but applies a correction to s_AC that
    accounts for the nonlinear interaction between the base-rate term
    1/(1-s_B) and the uncertainty in s_B (encoded by n_B / c_B).

    The correction uses n to adjust ρ — partially answering QLN Open
    Problem #4 ("Full PLN recovery: deriving PLN's (s, n) formulas
    from the QLN classical limit").
    """
    # PLN deduction formula with second-order correction
    s_B_f = float(s_B)
    denom = max(1.0 - s_B_f, EPS)
    h_mu = (float(s_C) - s_B_f * float(s_BC)) / denom
    correction = _jensen_correction_deduction(s_B, c_B, s_C, s_BC)

    s_AC = float(s_AB) * float(s_BC) + (1.0 - float(s_AB)) * (h_mu + correction)

    # n-path: PLN formula for c (unchanged)
    c_AC = _c_deduction(s_AB, c_AB, s_BC, c_BC)

    return s_AC, c_AC


def hybrid_deduction_laplace(s_A, c_A, s_B, c_B, s_C, c_C,
                             s_AB, c_AB, s_BC, c_BC,
                             background=DEFAULT_EPSILON,
                             **kwargs):
    """Hybrid Deduction via Laplace approximation.

    Reconstructs Gaussian N(s, σ²) from each (s, c) input, then
    analytically propagates mean and variance through the PLN
    deduction formula using second-order Taylor expansion.

    Output (s, c) are BOTH derived from the same Laplace framework:
      - s from E[f(X)] ≈ f(μ) + ½ Σ f''_ii σ²_i
      - c from Var[f(X)] ≈ Σ (∂f/∂x_i)² σ²_i → n → c

    Pure CPU, O(1) scalar operations, no sampling.
    """
    return _laplace_deduction(s_A, c_A, s_B, c_B, s_C, c_C,
                              s_AB, c_AB, s_BC, c_BC, background)


def hybrid_abduction(s_A, c_A, s_B, c_B,
                     s_AC, c_AC, s_BC, c_BC,
                     background=DEFAULT_EPSILON,
                     seed=42, n_batches=None):
    """Hybrid Abduction: s from inv-V binary Ising, c from PLN formula.

    Given A→C and B→C, infer A→B (explaining away).
    Uses inverted-V topology: A → Center ← B, query Center.

    Parameters
    ----------
    s_A, c_A : float
        Left cause truth value.
    s_B, c_B : float
        Right cause truth value.
    s_AC, c_AC : float
        Left link A→C truth value.
    s_BC, c_BC : float
        Right link B→C truth value.

    Returns
    -------
    (s_AB, c_AB) : tuple[float, float]
        Inferred truth value for A→B.
    """
    if n_batches is None:
        n_batches = DEFAULT_BINARY_N_BATCHES

    # ρ-path: inv-V Ising graph, all free, query center
    graph = compile_binary_inv_v(
        left_prior=s_A, right_prior=s_B,
        left_strength=s_AC, right_strength=s_BC,
        left_background=background, right_background=background,
        center_prior=0.5)
    samples = run_binary_sampling(graph, seed=seed, n_batches=n_batches)
    # Center node is index 1 in inv-V [left, center, right]
    s_center = estimate_binary_marginal(samples, graph, 1)

    # n-path: PLN closed-form formula for c
    c_AB = _c_abduction(s_AC, c_AC, s_BC, c_BC)

    return s_center, c_AB


def hybrid_inversion(s_A, c_A, s_B, c_B, s_AB, c_AB,
                     background=DEFAULT_EPSILON,
                     seed=42, n_batches=None):
    """Hybrid Inversion: s from 2-node all-free Ising, c from Bayes formula.

    Given A=(s_A, c_A), B=(s_B, c_B), and link A→B=(s_AB, c_AB),
    infer the reversed link B→A=(s_BA, c_BA).

    Uses Bayes' theorem: P(A|B) = P(B|A) × P(A) / P(B).

    Parameters
    ----------
    s_A, c_A : float
        Truth value for A.
    s_B, c_B : float
        Truth value for B.
    s_AB, c_AB : float
        Forward implication link A→B truth value.
    background : float
        Background rate (PLN epsilon).
    seed : int
        Random seed for sampling.
    n_batches : int or None
        Number of sampling batches (None → default).

    Returns
    -------
    (s_BA, c_BA) : tuple[float, float]
        Inferred truth value for the reversed link B→A.
    """
    if n_batches is None:
        n_batches = DEFAULT_BINARY_N_BATCHES

    # Derive background that makes the joint's P(B) marginal correct:
    # P(B) = P(B|A)·P(A) + P(B|¬A)·P(¬A)  →  P(B|¬A) = (s_B - s_A·s_AB)/(1-s_A)
    bg_raw = (float(s_B) - float(s_A) * float(s_AB)) / max(1.0 - float(s_A), 1e-7)
    bg = max(min(bg_raw, 0.98), 0.01) if 0.0 < bg_raw < 1.0 else background

    # ρ-path: 2-node all-free Ising graph encoding joint P(A,B)
    graph = compile_binary_joint_2node(s_A, s_AB, bg)
    samples = run_binary_sampling(graph, seed=seed, n_batches=n_batches)
    # P(A|B): target=A (node 0), condition=B (node 1)
    s_BA = _binary_conditional(samples, graph, 0, 1)

    # n-path: Bayes confidence formula (CPU)
    c_BA = inversion_bayes(s_A, c_A, s_B, c_B, s_AB, c_AB)[1]

    return s_BA, c_BA


def hybrid_deduction_tempered(s_A, c_A, s_B, c_B, s_C, c_C,
                              s_AB, c_AB, s_BC, c_BC,
                              background=DEFAULT_EPSILON,
                              seed=42, n_batches=None):
    """Hybrid Deduction with confidence-tempered Ising coupling.

    J_eff = J(s_link) × (n_link / 2), where n_link = c2w(c_link) + 2.
    Encodes confidence into energy function (Penalty Logic / Active
    Inference precision modulation).

    Higher c → larger n → stronger J → more deterministic sampling.
    Lower c → smaller n → weaker J → more diffuse → captures uncertainty.
    """
    from pln_thrml.compiler_binary import (
        ising_params, prior_bias, SpinNode, SpinEBMFactor, Block,
        _assemble_binary_graph_ex,
    )
    import jax.numpy as jnp

    if n_batches is None:
        n_batches = DEFAULT_BINARY_N_BATCHES

    n_AB = c2w(min(float(c_AB), 0.9999)) + 2.0
    n_BC = c2w(min(float(c_BC), 0.9999)) + 2.0
    scale_AB = n_AB / 2.0   # normalise: n=2 (minimum) → scale=1
    scale_BC = n_BC / 2.0

    spins = [SpinNode() for _ in range(3)]
    blocks = [Block([s]) for s in spins]
    factors = []
    free_blocks = blocks[1:]
    clamped_blocks = [blocks[0]]

    # Neutral priors on free nodes
    for idx in [1, 2]:
        h = prior_bias(0.5)
        if abs(h) > 1e-10:
            factors.append(SpinEBMFactor([blocks[idx]], jnp.array([h])))

    # Tempered coupling A→B
    J_AB, h_corr_AB = ising_params(float(s_AB), background)
    if abs(J_AB * scale_AB) > 1e-10:
        factors.append(SpinEBMFactor(
            [blocks[0], blocks[1]], jnp.array([J_AB * scale_AB])))
    if abs(h_corr_AB * scale_AB) > 1e-10:
        factors.append(SpinEBMFactor(
            [blocks[1]], jnp.array([h_corr_AB * scale_AB])))

    # Tempered coupling B→C
    J_BC, h_corr_BC = ising_params(float(s_BC), background)
    if abs(J_BC * scale_BC) > 1e-10:
        factors.append(SpinEBMFactor(
            [blocks[1], blocks[2]], jnp.array([J_BC * scale_BC])))
    if abs(h_corr_BC * scale_BC) > 1e-10:
        factors.append(SpinEBMFactor(
            [blocks[2]], jnp.array([h_corr_BC * scale_BC])))

    graph = _assemble_binary_graph_ex(
        spins, blocks, free_blocks, clamped_blocks, factors,
        root_prior=float(s_A))

    samples = run_binary_sampling(graph, seed=seed, n_batches=n_batches)
    s_C_inferred = estimate_binary_marginal(samples, graph, 2)

    c_AC = _c_deduction(s_AB, c_AB, s_BC, c_BC)
    return s_C_inferred, c_AC


def hybrid_deduction_pmode(s_A, c_A, s_B, c_B, s_C, c_C,
                           s_AB, c_AB, s_BC, c_BC,
                           background=DEFAULT_EPSILON,
                           seed=42, n_samples=200_000,
                           mode="ideal"):
    """Hybrid Deduction simulating pmode (Gaussian) hardware.

    Each input truth value (s, c) is sampled from N(μ=s, σ²=Var(s,c)),
    where Var = s(1-s)/(n+1) encodes both s and c in one Gaussian.

    Two modes:
      "ideal":   Independent Gaussian per input → CPU applies PLN formula
                 per sample.  No hardware coupling constraint.
      "coupled": Gaussian conditional chain A→B→C.  Each node's value is
                 sampled conditional on its parent via linear regression
                 (the natural coupling for pmode hardware).
    """
    import numpy as np

    rng = np.random.RandomState(seed)

    if mode == "ideal":
        # Independent Gaussian per input, CPU formula per sample
        g_A = rng.normal(s_A, max(np.sqrt(_var_from_stv(s_A, c_A)), EPS), n_samples)
        g_AB = rng.normal(s_AB, max(np.sqrt(_var_from_stv(s_AB, c_AB)), EPS), n_samples)
        g_BC = rng.normal(s_BC, max(np.sqrt(_var_from_stv(s_BC, c_BC)), EPS), n_samples)
        g_B = rng.normal(s_B, max(np.sqrt(_var_from_stv(s_B, c_B)), EPS), n_samples)
        g_C = rng.normal(s_C, max(np.sqrt(_var_from_stv(s_C, c_C)), EPS), n_samples)

        g_A = np.clip(g_A, 0.001, 0.999)
        g_AB = np.clip(g_AB, 0.001, 0.999)
        g_BC = np.clip(g_BC, 0.001, 0.999)
        g_B = np.clip(g_B, 0.001, 0.999)
        g_C = np.clip(g_C, 0.001, 0.999)

        denom = np.maximum(1.0 - g_B, 1e-7)
        s_AC = g_AB * g_BC + (1.0 - g_AB) * (g_C - g_B * g_BC) / denom
        s_AC = np.clip(s_AC, 0.0, 1.0)
        s_out = float(np.mean(s_AC))

    elif mode == "coupled":
        # Gaussian conditional chain: simulate pmode hardware with linear coupling
        # A ~ N(s_A, Var_A)
        # B|A ~ N(s_AB * A + bg * (1-A), Var_residual_B)
        # C|B ~ N(s_BC * B + bg * (1-B), Var_residual_C)
        sigma_A = max(np.sqrt(_var_from_stv(s_A, c_A)), EPS)
        g_A = rng.normal(s_A, sigma_A, n_samples)
        g_A = np.clip(g_A, 0.001, 0.999)

        # B conditional on A: linear coupling (pmode natural structure)
        mu_B_given_A = float(s_AB) * g_A + background * (1.0 - g_A)
        sigma_B_residual = max(np.sqrt(_var_from_stv(s_AB, c_AB)), EPS)
        g_B = rng.normal(mu_B_given_A, sigma_B_residual)
        g_B = np.clip(g_B, 0.001, 0.999)

        # C conditional on B: linear coupling
        mu_C_given_B = float(s_BC) * g_B + background * (1.0 - g_B)
        sigma_C_residual = max(np.sqrt(_var_from_stv(s_BC, c_BC)), EPS)
        g_C = rng.normal(mu_C_given_B, sigma_C_residual)
        g_C = np.clip(g_C, 0.0, 1.0)

        s_out = float(np.mean(g_C))

    else:
        raise ValueError(f"Unknown mode: {mode}")

    c_AC = _c_deduction(s_AB, c_AB, s_BC, c_BC)
    return s_out, c_AC


def hybrid_deduction_joint(s_A, c_A, s_B, c_B, s_C, c_C,
                           s_AB, c_AB, s_BC, c_BC,
                           background=DEFAULT_EPSILON,
                           seed=42, n_batches=None):
    """Hybrid Deduction via QLN block: pdit K=8 joint categorical.

    Encodes the full joint P(A,B,C) in a single K=2³=8 categorical
    variable and marginalises to get s_C.
    """
    if n_batches is None:
        n_batches = DEFAULT_BINARY_N_BATCHES

    # ρ-path: joint categorical sampling
    graph = compile_joint_categorical(
        [s_A, s_B, s_C], [s_AB, s_BC], [background, background])
    samples = run_joint_sampling(graph, seed=seed, n_batches=n_batches)
    # PLN's s_AC = P(C=T | A=T), not the unconditional marginal P(C=T).
    # Condition on A=T (prop 0 = A) before measuring C (prop 2).
    s_C_inferred = estimate_joint_conditional(
        samples, graph, target_idx=2, condition_idx=0, condition_value=1)

    # n-path: PLN formula
    c_AC = _c_deduction(s_AB, c_AB, s_BC, c_BC)

    return s_C_inferred, c_AC


def hybrid_deduction_pbit_joint(s_A, c_A, s_B, c_B, s_C, c_C,
                                s_AB, c_AB, s_BC, c_BC,
                                background=DEFAULT_EPSILON,
                                seed=42, n_batches=None):
    """Hybrid Deduction via single 3-spin pbit chain with parent-bias fix.

    Encodes the full Markov-chain log joint P(A)·P(B|A)·P(C|B) into a
    3-spin pairwise Ising energy using the complete expansion
        ln P(a,b,c) = const + a·(λ_A + μ_AB) + b·(ν_AB + μ_BC) + c·ν_BC
                              + ab·J_AB + bc·J_BC
    where μ is the parent-bias contribution (missing from the default
    compile path).  Gibbs-samples (B, C) jointly under stochastic A-clamp
    and returns P(C=T | A=+1) by filtering batches on root_bits.

    Unlike hybrid_deduction (two chained 2-node graphs, which collapses
    B to a point estimate), this variant keeps B and C jointly uncertain
    in a single chain, matching the full Bayesian posterior.
    """
    import numpy as np
    if n_batches is None:
        n_batches = DEFAULT_BINARY_N_BATCHES

    graph = compile_binary_chain(
        [s_A, 0.5, 0.5], [s_AB, s_BC], [background, background],
        clamp_root=True, include_parent_bias=True)
    samples = run_binary_sampling(graph, seed=seed, n_batches=n_batches)

    root_bits = np.asarray(samples["root_bits"]).flatten()
    A_pos = root_bits > 0
    if int(A_pos.sum()) == 0:
        s_C_inferred = float("nan")
    else:
        C_all = np.asarray(samples["vmap_samples"][1][:, :, 0])
        C_sel = C_all[A_pos]
        s_C_inferred = float((C_sel > 0).mean())

    c_AC = _c_deduction(s_AB, c_AB, s_BC, c_BC)
    return s_C_inferred, c_AC


def hybrid_deduction_hidden(s_A, c_A, s_B, c_B, s_C, c_C,
                            s_AB, c_AB, s_BC, c_BC,
                            background=DEFAULT_EPSILON,
                            seed=42, n_batches=None):
    """Hybrid Deduction via LBM hidden units: base rate encoding.

    Adds hidden pbits to a 3-node chain to encode s_B and s_C base rates
    that the plain Ising pairwise couplings miss.
    """
    if n_batches is None:
        n_batches = DEFAULT_BINARY_N_BATCHES

    # ρ-path: binary chain with hidden units
    graph = compile_binary_chain_with_hidden(
        priors=[s_A, 0.5, 0.5],
        strengths=[s_AB, s_BC],
        base_rates=[s_B, s_C],
        backgrounds=[background, background],
        clamp_root=True)
    samples = run_binary_sampling(graph, seed=seed, n_batches=n_batches)
    # Visible node C is index 2
    s_C_inferred = estimate_binary_marginal(samples, graph, 2)

    # n-path: PLN formula
    c_AC = _c_deduction(s_AB, c_AB, s_BC, c_BC)

    return s_C_inferred, c_AC
