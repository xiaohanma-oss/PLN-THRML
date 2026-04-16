"""
pln_thrml.beta — Beta-discretized factor graphs for PLN inference
==================================================================

Instead of binary CategoricalNodes (True/False), each proposition's strength
is modeled as a K-bin discrete random variable over [0,1].  The prior for
each node encodes a Beta(α,β) distribution derived from PLN's (stv s c).

After Gibbs sampling, **both** strength and confidence emerge from the
posterior distribution — strength as the mean, confidence from the
concentration (inverse variance) via moment-matching back to a Beta.

This is the primary inference engine.  All PLN rules compile to beta factor
graphs via the graph builders below.
"""

import jax
import jax.numpy as jnp

from thrml.block_management import Block
from thrml.block_sampling import BlockGibbsSpec, SamplingSchedule, sample_states
from thrml.pgm import CategoricalNode
from thrml.models.discrete_ebm import (CategoricalEBMFactor,
                                       SquareCategoricalEBMFactor,
                                       CategoricalGibbsConditional)
from thrml.factor import FactorSamplingProgram


EPS = 1e-7          # clamp for log-safety
DEFAULT_EPSILON = 0.02  # PLN modus-ponens background rate

__all__ = [
    # Constants
    "EPS", "DEFAULT_EPSILON", "DEFAULT_K",
    "DEFAULT_BETA_N_BATCHES", "DEFAULT_BETA_SCHEDULE",
    # Conversion
    "c2w", "w2c", "bin_centers", "bin_width",
    "stv_to_beta_params", "posterior_to_stv",
    # Weights & factors
    "beta_prior_weights", "beta_implication_weights",
    "make_beta_prior_factor", "make_beta_implication_factor",
    # Graph builders
    "build_beta_chain", "build_beta_inv_v_graph",
    # Sampling & measurement
    "run_beta_sampling", "sample_and_measure",
    "estimate_beta_marginal", "estimate_beta_conditional",
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


DEFAULT_K = 16

DEFAULT_BETA_N_BATCHES = 50
DEFAULT_BETA_SCHEDULE = SamplingSchedule(n_warmup=500, n_samples=2000,
                                         steps_per_sample=3)


# ═══════════════════════════════════════════════════════════════════════════
#  Bin setup
# ═══════════════════════════════════════════════════════════════════════════

def bin_centers(k=DEFAULT_K):
    """K evenly-spaced bin centers in (0, 1)."""
    return jnp.linspace(0.5 / k, 1.0 - 0.5 / k, k)


def bin_width(k=DEFAULT_K):
    return 1.0 / k


# ═══════════════════════════════════════════════════════════════════════════
#  STV ↔ Beta conversion
# ═══════════════════════════════════════════════════════════════════════════

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


def posterior_to_stv(posterior, k=DEFAULT_K):
    """Convert K-bin posterior histogram to (strength, confidence).

    Uses moment-matching on the discretized posterior to recover the
    effective Beta(α, β) parameters, then converts to (strength, confidence).

    The bin-center discretization introduces systematic bias for distributions
    near 0 or 1 (the outermost bin centers are 0.5/K and 1-0.5/K, not 0 and 1).
    We correct for this by using the raw mean for strength (which is unbiased
    for symmetric distributions) and the variance-based concentration estimate.
    """
    centers = bin_centers(k)
    posterior = posterior / jnp.sum(posterior)  # normalize

    mu = float(jnp.sum(posterior * centers))
    var = float(jnp.sum(posterior * (centers - mu) ** 2))

    if var < 1e-12:
        return mu, w2c(1e6)

    # Moment-matching: Var(Beta) = mu*(1-mu)/(n+1), solve for n
    n = mu * (1.0 - mu) / var - 1.0
    n = max(n, 0.01)

    # Recover strength from the fitted Beta mean (alpha / n)
    # Use mu directly — it is the posterior mean, which IS the strength estimate
    strength = mu

    # Evidence weight: subtract Beta(1,1) baseline (n=2)
    w_eff = max(n - 2.0, 0.0)
    return strength, w2c(w_eff)




# ═══════════════════════════════════════════════════════════════════════════
#  Weight computation
# ═══════════════════════════════════════════════════════════════════════════

def beta_prior_weights(strength, confidence, k=DEFAULT_K):
    """Log Beta PDF over K bins, as a weight vector.

    W[j] = (alpha-1)*log(c_j) + (beta-1)*log(1-c_j)
    Centered (subtract mean) for numerical stability.

    Returns shape [K].
    """
    alpha, beta = stv_to_beta_params(strength, confidence)
    centers = bin_centers(k)
    log_c = jnp.log(jnp.clip(centers, EPS, 1.0 - EPS))
    log_1mc = jnp.log(jnp.clip(1.0 - centers, EPS, 1.0 - EPS))

    w = (alpha - 1.0) * log_c + (beta - 1.0) * log_1mc
    w = w - jnp.mean(w)  # center for stability
    return w


def beta_implication_weights(strength, confidence, background=DEFAULT_EPSILON,
                              k=DEFAULT_K):
    """K×K conditional weight table for a directed implication.

    For parent bin i (center s_i):
        mu_i = s_i * s_AB + eps * (1 - s_i)
        conditional is Beta(mu_i * w_cond, (1-mu_i) * w_cond)

    Returns shape [K, K]  (parent_bins × child_bins).
    """
    centers = bin_centers(k)
    log_c = jnp.log(jnp.clip(centers, EPS, 1.0 - EPS))
    log_1mc = jnp.log(jnp.clip(1.0 - centers, EPS, 1.0 - EPS))

    w_cond = c2w(confidence)
    n_cond = w_cond + 2.0  # mean-preserving: n = w + 2

    # mu_i for each parent bin
    mu = centers * strength + background * (1.0 - centers)  # shape [K]

    # Per-row Beta parameters (mean-preserving)
    alpha = mu * n_cond          # shape [K]
    beta = (1.0 - mu) * n_cond   # shape [K]

    # W[i, j] = (alpha_i - 1)*log(c_j) + (beta_i - 1)*log(1-c_j)
    w = ((alpha - 1.0)[:, None] * log_c[None, :]
         + (beta - 1.0)[:, None] * log_1mc[None, :])

    # Center each row for stability
    w = w - jnp.mean(w, axis=1, keepdims=True)
    return w


# ═══════════════════════════════════════════════════════════════════════════
#  Factor builders
# ═══════════════════════════════════════════════════════════════════════════

def make_beta_prior_factor(node, strength, confidence, k=DEFAULT_K):
    """Unary factor encoding Beta(alpha, beta) prior over K bins.

    Weight shape: [1, K].
    """
    w = beta_prior_weights(strength, confidence, k)
    return CategoricalEBMFactor([Block([node])], w[None, :])


def make_beta_implication_factor(parent, child, strength, confidence,
                                  background=DEFAULT_EPSILON, k=DEFAULT_K):
    """Pairwise factor: K×K conditional table for directed implication.

    Weight shape: [1, K, K].
    """
    w = beta_implication_weights(strength, confidence, background, k)
    return SquareCategoricalEBMFactor([Block([parent]), Block([child])], w[None, :, :])


# ═══════════════════════════════════════════════════════════════════════════
#  Graph builders
# ═══════════════════════════════════════════════════════════════════════════

def _assemble_free_graph(nodes, factors, free_blocks, k=DEFAULT_K, **extra):
    """Assemble a fully free (no clamp) factor graph into a sampling program."""
    spec = BlockGibbsSpec(free_blocks, [])
    sampler = CategoricalGibbsConditional(n_categories=k)
    prog = FactorSamplingProgram(
        gibbs_spec=spec,
        samplers=[sampler] * len(free_blocks),
        factors=factors, other_interaction_groups=[])
    result = dict(nodes=nodes, factors=factors, free_blocks=free_blocks,
                  clamped_blocks=[], spec=spec, program=prog,
                  k=k, single_node=False)
    result.update(extra)
    return result


def _beta_prior_logits(strength, confidence, k=DEFAULT_K):
    """Log-probabilities for sampling a node's clamped state from its Beta prior."""
    w = beta_prior_weights(strength, confidence, k)
    return w - jax.scipy.special.logsumexp(w)


def build_beta_chain(priors, confidences, strengths, impl_confidences,
                      backgrounds, k=DEFAULT_K, clamp_root=True):
    """Build a directed chain X_0 → X_1 → ... → X_{n-1} with K-bin nodes.

    clamp_root=True: root sampled from its prior, rest are free (marginal queries).
    clamp_root=False: all nodes free (conditional queries).
    """
    n = len(priors)
    nodes = [CategoricalNode() for _ in range(n)]

    # Prior factors: skip root when clamped
    start = 1 if clamp_root else 0
    factors = [make_beta_prior_factor(nodes[i], priors[i], confidences[i], k)
               for i in range(start, n)]

    # Implication factors for each edge
    for i in range(n - 1):
        factors.append(make_beta_implication_factor(
            nodes[i], nodes[i + 1],
            strengths[i], impl_confidences[i], backgrounds[i], k))

    if not clamp_root:
        even = [nodes[i] for i in range(0, n, 2)]
        odd = [nodes[i] for i in range(1, n, 2)]
        free_blocks = [Block(even), Block(odd)] if odd else [Block(even)]
        return _assemble_free_graph(nodes, factors, free_blocks, k, n=n)

    # Clamped root path
    clamped_blocks = [Block([nodes[0]])]
    root_logits = _beta_prior_logits(priors[0], confidences[0], k)
    free_nodes = nodes[1:]

    if not free_nodes:
        spec = BlockGibbsSpec([], clamped_blocks)
        prog = FactorSamplingProgram(
            gibbs_spec=spec, samplers=[], factors=factors,
            other_interaction_groups=[])
        return dict(nodes=nodes, factors=factors, free_blocks=[],
                    clamped_blocks=clamped_blocks, spec=spec, program=prog,
                    n=n, k=k, root_logits=root_logits, single_node=True)

    even = [free_nodes[i] for i in range(0, len(free_nodes), 2)]
    odd = [free_nodes[i] for i in range(1, len(free_nodes), 2)]
    free_blocks = [Block(even), Block(odd)] if odd else [Block(even)]

    spec = BlockGibbsSpec(free_blocks, clamped_blocks)
    sampler = CategoricalGibbsConditional(n_categories=k)
    prog = FactorSamplingProgram(
        gibbs_spec=spec, samplers=[sampler] * len(free_blocks),
        factors=factors, other_interaction_groups=[])

    return dict(nodes=nodes, factors=factors, free_blocks=free_blocks,
                clamped_blocks=clamped_blocks, spec=spec, program=prog,
                n=n, k=k, root_logits=root_logits, single_node=False)


def _build_three_node_graph(node_priors, edges, k=DEFAULT_K, **labels):
    """3-node free graph with 2-coloring: nodes[0],nodes[2] in one block, nodes[1] in another.

    node_priors: [(s, c), (s, c), (s, c)]
    edges: [(parent_idx, child_idx, strength, confidence, background), ...]
    """
    nodes = [CategoricalNode() for _ in range(3)]
    factors = [make_beta_prior_factor(nodes[i], *node_priors[i], k) for i in range(3)]
    for pi, ci, s, c, bg in edges:
        factors.append(make_beta_implication_factor(nodes[pi], nodes[ci], s, c, bg, k))
    free_blocks = [Block([nodes[0], nodes[2]]), Block([nodes[1]])]
    return _assemble_free_graph(nodes, factors, free_blocks, k, **labels)


def build_beta_inv_v_graph(left_prior, left_confidence,
                           right_prior, right_confidence,
                           left_strength, right_strength,
                           left_impl_confidence, right_impl_confidence,
                           left_background, right_background,
                           center_prior=0.5, center_confidence=0.01,
                           k=DEFAULT_K):
    """Inverted-V: Left → Center ← Right (abduction). Nodes order: [left, center, right]."""
    g = _build_three_node_graph(
        [(left_prior, left_confidence), (center_prior, center_confidence),
         (right_prior, right_confidence)],
        [(0, 1, left_strength, left_impl_confidence, left_background),
         (2, 1, right_strength, right_impl_confidence, right_background)],
        k, left=None, center=None, right=None)
    g["left"], g["center"], g["right"] = g["nodes"][0], g["nodes"][1], g["nodes"][2]
    return g


# ═══════════════════════════════════════════════════════════════════════════
#  Sampling
# ═══════════════════════════════════════════════════════════════════════════

def run_beta_sampling(graph, seed=42, n_batches=None, schedule=None):
    """Run block Gibbs sampling on a Beta-discretized factor graph.

    Supports two modes:
    - Clamped root: each batch draws a different root state from the Beta
      prior, providing marginalization over the premise.
    - All free: standard Gibbs sampling with no clamped nodes.
    """
    if n_batches is None:
        n_batches = DEFAULT_BETA_N_BATCHES
    if schedule is None:
        schedule = DEFAULT_BETA_SCHEDULE

    k = graph["k"]
    spec = graph["spec"]
    prog = graph["program"]
    key = jax.random.key(seed)

    # Single-node graph: just sample from the prior directly
    if graph.get("single_node", False):
        key, subkey = jax.random.split(key)
        root_samples = jax.random.categorical(
            subkey, graph["root_logits"],
            shape=(n_batches, schedule.n_samples))
        return {"root_samples": root_samples, "k": k}

    has_clamped = bool(graph.get("clamped_blocks"))

    if has_clamped:
        # Sample root's clamped state from its Beta prior (one value per batch)
        key, subkey = jax.random.split(key)
        root_state = jax.random.categorical(
            subkey, graph["root_logits"], shape=(n_batches, 1)
        ).astype(jnp.uint8)
        state_clamp = [root_state]
    else:
        state_clamp = []
        root_state = None

    # Initialize free blocks randomly
    init_state = []
    for block in spec.free_blocks:
        key, subkey = jax.random.split(key)
        init_state.append(
            jax.random.randint(subkey, (n_batches, len(block.nodes)),
                               minval=0, maxval=k, dtype=jnp.uint8)
        )

    keys = jax.random.split(key, n_batches)
    observe_blocks = list(spec.free_blocks)

    if has_clamped:
        samples = jax.jit(jax.vmap(
            lambda s, c, k_: sample_states(k_, prog, schedule, s, c, observe_blocks)
        ))(init_state, state_clamp, keys)
    else:
        samples = jax.jit(jax.vmap(
            lambda s, k_: sample_states(k_, prog, schedule, s, [], observe_blocks)
        ))(init_state, keys)

    if root_state is not None:
        return {"block_samples": samples, "root_clamped_states": root_state, "k": k}
    return samples


def sample_and_measure(graph, target_node, seed=42):
    """Run sampling and return (strength, confidence) for a target node.

    Convenience wrapper combining run_beta_sampling + estimate_beta_marginal.
    """
    samples = run_beta_sampling(graph, seed=seed)
    _, strength, confidence = estimate_beta_marginal(samples, graph, target_node)
    return strength, confidence


# ═══════════════════════════════════════════════════════════════════════════
#  Measurement
# ═══════════════════════════════════════════════════════════════════════════

def _node_location(graph, node):
    """Find (block_idx, position_within_block) for a node in free blocks."""
    for bi, block in enumerate(graph["free_blocks"]):
        for ni, n in enumerate(block.nodes):
            if n is node:
                return bi, ni
    raise ValueError("Node not found in any free block")


def _flatten_node(samples, block_idx, node_within_block):
    """Extract a single node's samples as a flat uint8 array."""
    return samples[block_idx][:, :, node_within_block].flatten()


def _build_histogram(flat_samples, k):
    """Build normalized histogram from categorical samples."""
    counts = jnp.bincount(flat_samples.astype(jnp.int32), length=k)
    return counts / jnp.sum(counts)


def _unpack_samples(samples):
    """Extract block samples and optional root_clamped_states from samples.

    Handles both formats:
    - list of arrays (all-free graphs): returns (samples, None)
    - dict with "block_samples" (clamped root): returns (block_samples, root_states)
    - dict with "root_samples" (single-node): returns (None, None) — caller handles
    """
    if isinstance(samples, dict):
        if "root_samples" in samples:
            return None, None
        return samples["block_samples"], samples.get("root_clamped_states")
    return samples, None


def estimate_beta_marginal(samples, graph, node, k=None):
    """Posterior histogram and (strength, confidence) for a single node.

    Returns (posterior, strength, confidence).
    """
    if k is None:
        k = graph.get("k", DEFAULT_K)

    # Single-node graph: root was sampled directly
    if isinstance(samples, dict) and "root_samples" in samples:
        flat = samples["root_samples"].flatten()
        posterior = _build_histogram(flat, k)
        strength, confidence = posterior_to_stv(posterior, k)
        return posterior, strength, confidence

    block_samples, _ = _unpack_samples(samples)
    bi, ni = _node_location(graph, node)
    flat = _flatten_node(block_samples, bi, ni)
    posterior = _build_histogram(flat, k)
    strength, confidence = posterior_to_stv(posterior, k)
    return posterior, strength, confidence


def _is_clamped_root(graph, node):
    """Check if a node is the clamped root (nodes[0] in a clamped graph)."""
    if graph.get("single_node", False):
        return False
    if not graph.get("clamped_blocks"):
        return False
    return graph["nodes"][0] is node


def _weighted_histogram(target_bins, condition_bins, k):
    """Unnormalized weighted histogram — weight = bin_center[condition_bin]."""
    centers = bin_centers(k)
    weights = centers[condition_bins]
    one_hot = jax.nn.one_hot(target_bins, k)       # [N, K]
    return one_hot.T @ weights                      # [K]


def _finalize_posterior(posterior, k):
    """Normalize posterior and convert to (posterior, strength, confidence)."""
    total = jnp.sum(posterior)
    if total < 1e-10:
        return jnp.ones(k) / k, 0.5, 0.0
    posterior = posterior / total
    strength, confidence = posterior_to_stv(posterior, k)
    return posterior, strength, confidence


def estimate_beta_conditional(samples, graph, target, condition, k=None):
    """P(target | condition=True) — weighted conditional posterior.

    Uses bin-center weighting: each sample is weighted by the "truthiness"
    of the condition variable (bin_center[condition_state]).  This gives
    the proper Bayesian conditional E[target | condition is True].

    Returns (posterior, strength, confidence).
    """
    if k is None:
        k = graph["k"]

    cond_is_clamped = _is_clamped_root(graph, condition)
    target_is_clamped = _is_clamped_root(graph, target)

    block_samples, root_clamped = _unpack_samples(samples)

    def _get_root_states():
        if root_clamped is None:
            raise ValueError("Root clamped states not available. "
                             "Use run_beta_sampling with a clamped-root graph.")
        return root_clamped[:, 0]  # [n_batches]

    if target_is_clamped and cond_is_clamped:
        raise ValueError("Cannot condition clamped root on itself")

    centers = bin_centers(k)

    if cond_is_clamped:
        tbi, tni = _node_location(graph, target)
        t_raw = block_samples[tbi][:, :, tni]
        root_flat = _get_root_states()
        root_weights = centers[root_flat]

        one_hot = jax.nn.one_hot(t_raw.astype(jnp.int32), k)  # [batches, samples, K]
        batch_hists = one_hot.sum(axis=1)                        # [batches, K]
        posterior = (root_weights[:, None] * batch_hists).sum(axis=0)  # [K]
        return _finalize_posterior(posterior, k)

    if target_is_clamped:
        cbi, cni = _node_location(graph, condition)
        c_raw = block_samples[cbi][:, :, cni]
        root_flat = _get_root_states()

        cond_truthiness = jnp.mean(centers[c_raw], axis=1)        # [batches]
        root_one_hot = jax.nn.one_hot(root_flat.astype(jnp.int32), k)  # [batches, K]
        posterior = (cond_truthiness[:, None] * root_one_hot).sum(axis=0)  # [K]
        return _finalize_posterior(posterior, k)

    # Both target and condition are in free blocks
    tbi, tni = _node_location(graph, target)
    cbi, cni = _node_location(graph, condition)
    t_flat = _flatten_node(block_samples, tbi, tni)
    c_flat = _flatten_node(block_samples, cbi, cni)

    posterior = _weighted_histogram(
        t_flat.astype(jnp.int32), c_flat.astype(jnp.int32), k)
    return _finalize_posterior(posterior, k)
