"""
pln_thrml.compiler_onehot — One-hot spin compiler for pbit-only hardware
==========================================================================

Compiles PLN factor graphs to SpinNode representations, simulating what
hardware would do on pbit-only TSU (no pdit).  Each K-state categorical
variable becomes K binary spins with a mutual exclusion constraint.

Two coupling modes:
  - full: K² SpinEBMFactor edges per categorical pair (exact)
  - potts: K SpinEBMFactor edges per categorical pair (approximate)

TSU hardware constraint: ~12 neighbors per pbit.
  K=4 full: 3 (excl) + 4 (coupling) = 7/neighbor → max 2 categorical neighbors
  K=4 potts: 3 (excl) + 1 (coupling) = 4/neighbor → max 9 categorical neighbors
"""

import numpy as np
import jax
import jax.numpy as jnp
from scipy.optimize import minimize_scalar

from thrml.pgm import SpinNode
from thrml.models.discrete_ebm import SpinEBMFactor, SpinGibbsConditional
from thrml.block_management import Block
from thrml.block_sampling import BlockGibbsSpec, SamplingSchedule, sample_states
from thrml.factor import FactorSamplingProgram

from pln_thrml.beta import (
    beta_prior_weights, beta_implication_weights,
    bin_centers, posterior_to_stv, w2c,
    DEFAULT_K, DEFAULT_BETA_N_BATCHES, DEFAULT_BETA_SCHEDULE, EPS,
)


__all__ = [
    "compile_onehot_full",
    "compile_onehot_potts",
    "fit_potts_weight",
    "count_neighbors_per_spin",
    "run_onehot_sampling",
    "estimate_onehot_marginal",
]

# Ising one-hot constraint strength.  The penalty quadratic is
#   E = -alpha * (sum_si - (2-K))^2
# which decomposes into pairwise coupling J = -2*alpha and
# per-spin bias h = 2*alpha*(2-K).
DEFAULT_ONEHOT_ALPHA = 2.0


# ═══════════════════════════════════════════════════════════════════════════
#  Potts weight fitting
# ═══════════════════════════════════════════════════════════════════════════

def fit_potts_weight(W, method="kl"):
    """Fit scalar Potts weight from a K×K categorical weight table.

    Parameters
    ----------
    W : ndarray, shape [K, K]
        Categorical implication weight table.
    method : str
        "diag" — diagonal mean (zero optimization cost).
        "kl"   — minimize mean-row KL(softmax(W_row) || softmax(W_potts_row)).

    Returns
    -------
    w_opt : float
        Scalar coupling weight for same-index Potts connections.
    """
    W = np.asarray(W)
    K = W.shape[0]

    if method == "diag":
        return float(np.diag(W).mean())

    if method == "kl":
        def kl_loss(w_scalar):
            W_potts = np.eye(K) * w_scalar
            total = 0.0
            for i in range(K):
                p = _softmax(W[i])
                q = _softmax(W_potts[i])
                total += float(np.sum(p * np.log(p / np.clip(q, 1e-12, None))))
            return total / K

        result = minimize_scalar(kl_loss, bounds=(-50.0, 50.0), method="bounded")
        return float(result.x)

    raise ValueError(f"Unknown method: {method}")


def _softmax(x):
    x = np.asarray(x, dtype=np.float64)
    e = np.exp(x - x.max())
    return e / e.sum()


# ═══════════════════════════════════════════════════════════════════════════
#  Neighbor counting
# ═══════════════════════════════════════════════════════════════════════════

def count_neighbors_per_spin(k, n_categorical_neighbors=1):
    """Compute neighbor consumption per spin in one-hot encoding.

    Returns dict with neighbor counts and feasibility at budget=12.
    """
    exclusion = k - 1
    full_per_neighbor = k
    potts_per_neighbor = 1

    full_total = exclusion + full_per_neighbor * n_categorical_neighbors
    potts_total = exclusion + potts_per_neighbor * n_categorical_neighbors

    budget = 12
    max_full = max(0, (budget - exclusion) // full_per_neighbor) if exclusion < budget else 0
    max_potts = max(0, (budget - exclusion) // potts_per_neighbor) if exclusion < budget else 0

    return {
        "k": k,
        "exclusion": exclusion,
        "full_per_neighbor": full_per_neighbor,
        "potts_per_neighbor": potts_per_neighbor,
        "full_total": full_total,
        "potts_total": potts_total,
        "max_full_neighbors": max_full,
        "max_potts_neighbors": max_potts,
        "full_feasible": full_total <= budget,
        "potts_feasible": potts_total <= budget,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  One-hot graph builders
# ═══════════════════════════════════════════════════════════════════════════

def _build_onehot_group(k, alpha=DEFAULT_ONEHOT_ALPHA):
    """Create K SpinNodes for one categorical variable with exclusion factors.

    One-hot in Ising {-1,+1}: exactly 1 spin is +1, rest -1.
    Quadratic penalty: E = -alpha * (sum_si - (2-K))^2
    Decomposes to: coupling J = -2*alpha, bias h = 2*alpha*(2-K).
    """
    spins = [SpinNode() for _ in range(k)]
    blocks = [Block([s]) for s in spins]
    factors = []

    J = -2.0 * alpha
    h = 2.0 * alpha * (2 - k)

    # Pairwise anti-ferromagnetic coupling (exclusion)
    for i in range(k):
        for j in range(i + 1, k):
            factors.append(SpinEBMFactor([Block([spins[i]]), Block([spins[j]])],
                                         jnp.array([J])))

    # Per-spin bias (negative for K > 2, enforces mostly -1)
    for i in range(k):
        factors.append(SpinEBMFactor([Block([spins[i]])], jnp.array([h])))

    return spins, blocks, factors


def _add_prior_factors(spins, prior_weights):
    """Add unary bias factors encoding the Beta prior over K bins.

    prior_weights: shape [K], from beta_prior_weights().
    Each weight w[m] becomes a bias on spin_m.

    Scaling: In Ising one-hot, the effective energy for valid state m is
    2*bias[m] + const (the factor of 2 comes from +1 vs -1 encoding).
    To match categorical P(m) ∝ exp(w[m]), we scale by 1/2.
    """
    factors = []
    for m, w in enumerate(prior_weights):
        factors.append(SpinEBMFactor([Block([spins[m]])],
                                     jnp.array([float(w) / 2.0])))
    return factors


def _add_full_coupling(parent_spins, child_spins, W):
    """Add K² SpinEBMFactor edges between two one-hot groups.

    W: shape [K, K], categorical implication weight table (row-centered).

    In Ising one-hot, valid state (parent=m, child=n) has effective energy:
        E_eff = 4*W_s[m,n] - 2*C_n_s   (where C_n = column sum, rows are 0)
    To match categorical P ∝ exp(W[m,n]), we scale coupling by 1/4 and
    add column-sum correction bias on child spins.
    """
    W_np = np.asarray(W)
    K = len(parent_spins)
    factors = []

    # Pairwise couplings (scaled 1/4)
    for m in range(K):
        for n in range(K):
            w_val = float(W_np[m, n]) / 4.0
            if abs(w_val) < 1e-10:
                continue
            factors.append(SpinEBMFactor(
                [Block([parent_spins[m]]), Block([child_spins[n]])],
                jnp.array([w_val])))

    # Column-sum correction: add C_n/4 bias to child spin n
    # (effective energy contribution: 2 * C_n/4 = C_n/2, cancels the -C_n/2 artifact)
    col_sums = W_np.sum(axis=0)
    for n in range(K):
        correction = float(col_sums[n]) / 4.0
        if abs(correction) < 1e-10:
            continue
        factors.append(SpinEBMFactor([Block([child_spins[n]])],
                                     jnp.array([correction])))

    return factors


def _add_potts_coupling(parent_spins, child_spins, W, method="kl"):
    """Add K SpinEBMFactor edges (same-index only) between two one-hot groups.

    Potts approximation: only diagonal couplings with fitted weight.
    Same 1/4 scaling as full coupling for Ising one-hot equivalence.
    Column-sum correction: Potts has col_sums = [w_opt]*K (uniform), so
    correction is w_opt/4 per child spin (constant, doesn't affect relative probs).
    We still add it for energy consistency.
    """
    w_opt = fit_potts_weight(np.asarray(W), method=method)
    factors = []
    K = len(parent_spins)
    for m in range(K):
        factors.append(SpinEBMFactor(
            [Block([parent_spins[m]]), Block([child_spins[m]])],
            jnp.array([w_opt / 4.0])))

    # Potts column sum = w_opt (uniform), correction is constant → no effect
    # But add for consistency
    correction = w_opt / 4.0
    if abs(correction) > 1e-10:
        for n in range(K):
            factors.append(SpinEBMFactor([Block([child_spins[n]])],
                                         jnp.array([correction])))

    return factors, w_opt


def compile_onehot_full(k, priors, confidences, strengths, impl_confidences,
                        backgrounds, alpha=DEFAULT_ONEHOT_ALPHA):
    """Build a one-hot full factor graph for a directed chain.

    Same interface as build_beta_chain(clamp_root=True) but using SpinNodes.
    priors[0] is the root (clamped via strong prior), rest are free with weak prior.

    Returns a dict compatible with run_onehot_sampling.
    """
    n = len(priors)
    all_spins = []     # list of lists: all_spins[i] = [K spins for node i]
    all_blocks = []    # list of lists: all_blocks[i] = [K blocks for node i]
    all_factors = []

    # Create one-hot groups
    for i in range(n):
        spins, blocks, excl_factors = _build_onehot_group(k, alpha)
        all_spins.append(spins)
        all_blocks.append(blocks)
        all_factors.extend(excl_factors)

    # Prior factors
    for i in range(n):
        pw = beta_prior_weights(priors[i], confidences[i], k)
        all_factors.extend(_add_prior_factors(all_spins[i], pw))

    # Implication factors (full K² coupling)
    for i in range(n - 1):
        W = beta_implication_weights(strengths[i], impl_confidences[i],
                                     backgrounds[i], k)
        all_factors.extend(_add_full_coupling(all_spins[i], all_spins[i + 1], W))

    return _assemble_onehot_graph(all_spins, all_blocks, all_factors, k, n)


def compile_onehot_potts(k, priors, confidences, strengths, impl_confidences,
                         backgrounds, alpha=DEFAULT_ONEHOT_ALPHA,
                         potts_method="kl"):
    """Build a one-hot Potts factor graph for a directed chain.

    Same as compile_onehot_full but with diagonal-only coupling.
    """
    n = len(priors)
    all_spins = []
    all_blocks = []
    all_factors = []
    potts_weights = []

    for i in range(n):
        spins, blocks, excl_factors = _build_onehot_group(k, alpha)
        all_spins.append(spins)
        all_blocks.append(blocks)
        all_factors.extend(excl_factors)

    for i in range(n):
        pw = beta_prior_weights(priors[i], confidences[i], k)
        all_factors.extend(_add_prior_factors(all_spins[i], pw))

    for i in range(n - 1):
        W = beta_implication_weights(strengths[i], impl_confidences[i],
                                     backgrounds[i], k)
        potts_factors, w_opt = _add_potts_coupling(
            all_spins[i], all_spins[i + 1], W, method=potts_method)
        all_factors.extend(potts_factors)
        potts_weights.append(w_opt)

    graph = _assemble_onehot_graph(all_spins, all_blocks, all_factors, k, n)
    graph["potts_weights"] = potts_weights
    graph["potts_method"] = potts_method
    return graph


def compile_onehot_full_inv_v(k, left_prior, left_confidence,
                              right_prior, right_confidence,
                              left_strength, right_strength,
                              left_impl_confidence, right_impl_confidence,
                              left_background, right_background,
                              center_prior=0.5, center_confidence=0.01,
                              alpha=DEFAULT_ONEHOT_ALPHA):
    """One-hot full inverted-V: Left → Center ← Right (abduction).

    Nodes order: [left, center, right].  Same as build_beta_inv_v_graph.
    """
    priors_list = [left_prior, center_prior, right_prior]
    confs_list = [left_confidence, center_confidence, right_confidence]
    all_spins, all_blocks, all_factors = [], [], []

    for i in range(3):
        spins, blocks, excl = _build_onehot_group(k, alpha)
        all_spins.append(spins)
        all_blocks.append(blocks)
        all_factors.extend(excl)

    for i in range(3):
        pw = beta_prior_weights(priors_list[i], confs_list[i], k)
        all_factors.extend(_add_prior_factors(all_spins[i], pw))

    # Edge: left(0) → center(1)
    W_left = beta_implication_weights(left_strength, left_impl_confidence,
                                      left_background, k)
    all_factors.extend(_add_full_coupling(all_spins[0], all_spins[1], W_left))

    # Edge: right(2) → center(1)
    W_right = beta_implication_weights(right_strength, right_impl_confidence,
                                       right_background, k)
    all_factors.extend(_add_full_coupling(all_spins[2], all_spins[1], W_right))

    return _assemble_onehot_graph(all_spins, all_blocks, all_factors, k, 3)


def compile_onehot_potts_inv_v(k, left_prior, left_confidence,
                               right_prior, right_confidence,
                               left_strength, right_strength,
                               left_impl_confidence, right_impl_confidence,
                               left_background, right_background,
                               center_prior=0.5, center_confidence=0.01,
                               alpha=DEFAULT_ONEHOT_ALPHA, potts_method="kl"):
    """One-hot Potts inverted-V: Left → Center ← Right (abduction)."""
    priors_list = [left_prior, center_prior, right_prior]
    confs_list = [left_confidence, center_confidence, right_confidence]
    all_spins, all_blocks, all_factors = [], [], []
    potts_weights = []

    for i in range(3):
        spins, blocks, excl = _build_onehot_group(k, alpha)
        all_spins.append(spins)
        all_blocks.append(blocks)
        all_factors.extend(excl)

    for i in range(3):
        pw = beta_prior_weights(priors_list[i], confs_list[i], k)
        all_factors.extend(_add_prior_factors(all_spins[i], pw))

    W_left = beta_implication_weights(left_strength, left_impl_confidence,
                                      left_background, k)
    pf, w = _add_potts_coupling(all_spins[0], all_spins[1], W_left, potts_method)
    all_factors.extend(pf)
    potts_weights.append(w)

    W_right = beta_implication_weights(right_strength, right_impl_confidence,
                                       right_background, k)
    pf, w = _add_potts_coupling(all_spins[2], all_spins[1], W_right, potts_method)
    all_factors.extend(pf)
    potts_weights.append(w)

    graph = _assemble_onehot_graph(all_spins, all_blocks, all_factors, k, 3)
    graph["potts_weights"] = potts_weights
    return graph


def _assemble_onehot_graph(all_spins, all_blocks, factors, k, n):
    """Assemble one-hot spin graph into a sampling program."""
    # Flatten all blocks for the spec
    flat_blocks = [b for group in all_blocks for b in group]

    conditional = SpinGibbsConditional()
    # Each spin is its own super-block (sequential Gibbs)
    super_blocks = [(b,) for b in flat_blocks]
    spec = BlockGibbsSpec(free_super_blocks=super_blocks, clamped_blocks=[])
    program = FactorSamplingProgram(
        gibbs_spec=spec,
        samplers=[conditional] * len(flat_blocks),
        factors=factors,
        other_interaction_groups=[])

    return {
        "all_spins": all_spins,
        "all_blocks": all_blocks,
        "flat_blocks": flat_blocks,
        "factors": factors,
        "spec": spec,
        "program": program,
        "k": k,
        "n": n,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Sampling and measurement
# ═══════════════════════════════════════════════════════════════════════════

def run_onehot_sampling(graph, seed=42, n_batches=None, schedule=None):
    """Run block Gibbs sampling on a one-hot spin graph.

    Returns raw spin samples: list of arrays per block.
    """
    if n_batches is None:
        n_batches = DEFAULT_BETA_N_BATCHES
    if schedule is None:
        # Single-spin Gibbs needs many steps per sample for mixing.
        # steps_per_sample=4 with adequate warmup works well empirically.
        schedule = SamplingSchedule(
            n_warmup=1000,
            n_samples=3000,
            steps_per_sample=4,
        )

    k = graph["k"]
    n = graph["n"]
    flat_blocks = graph["flat_blocks"]
    program = graph["program"]
    n_spins = len(flat_blocks)

    all_samples = []
    for batch in range(n_batches):
        key = jax.random.PRNGKey(seed + batch)
        # Initialize: for each node, set one random spin to True
        init_state = []
        subkey = jax.random.PRNGKey(seed + batch + 10000)
        for node_idx in range(n):
            active = int(jax.random.randint(subkey, (), 0, k))
            subkey = jax.random.fold_in(subkey, node_idx)
            for spin_idx in range(k):
                init_state.append(jnp.array([spin_idx == active]))

        results = sample_states(key, program, schedule, init_state, [],
                                flat_blocks)
        # results: list of n_spins arrays, each shape [n_samples, 1]
        all_samples.append(results)

    return all_samples


def decode_onehot_samples(all_samples, graph):
    """Decode spin samples to categorical indices.

    Returns dict: node_idx → array of shape [n_batches, n_samples].
    """
    k = graph["k"]
    n = graph["n"]
    n_batches = len(all_samples)

    decoded = {}
    for node_idx in range(n):
        node_samples = []
        for batch_samples in all_samples:
            # Extract K spins for this node
            spin_offset = node_idx * k
            spin_arrays = []
            for spin_idx in range(k):
                s = batch_samples[spin_offset + spin_idx][:, 0]  # [n_samples]
                spin_arrays.append(s)
            # Stack: [K, n_samples], argmax → [n_samples]
            stacked = jnp.stack(spin_arrays, axis=0)  # [K, n_samples]
            indices = jnp.argmax(stacked, axis=0)      # [n_samples]
            node_samples.append(indices)
        decoded[node_idx] = jnp.stack(node_samples, axis=0)  # [n_batches, n_samples]

    return decoded


def estimate_onehot_marginal(all_samples, graph, node_idx):
    """Estimate posterior histogram and (s, c) for a node from one-hot samples.

    Returns (posterior, strength, confidence) — same format as
    estimate_beta_marginal.
    """
    decoded = decode_onehot_samples(all_samples, graph)
    flat = np.asarray(decoded[node_idx]).flatten()

    k = graph["k"]
    histogram = np.bincount(flat, minlength=k).astype(float)
    posterior = histogram / histogram.sum()

    strength, confidence = posterior_to_stv(jnp.array(posterior), k)
    return posterior, strength, confidence


def sample_and_measure_onehot(graph, target_node_idx, seed=42, n_batches=30):
    """Convenience: run sampling + estimate marginal in one call."""
    all_samples = run_onehot_sampling(graph, seed=seed, n_batches=n_batches)
    _, s, c = estimate_onehot_marginal(all_samples, graph, target_node_idx)
    return s, c
