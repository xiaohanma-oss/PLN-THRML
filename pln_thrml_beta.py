"""
pln_thrml_beta.py — Beta-discretized factor graphs for PLN inference
=====================================================================

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
from thrml.models.discrete_ebm import CategoricalEBMFactor, CategoricalGibbsConditional
from thrml.factor import FactorSamplingProgram

# ── PLN truth-value utilities ────────────────────
EPS = 1e-7          # clamp for log-safety
DEFAULT_EPSILON = 0.02  # PLN modus-ponens background rate


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

# ── constants ────────────────────────────────────────────────────────────
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


def effective_k(confidence):
    """Choose bin count based on confidence level.

    Higher confidence means sharper Beta distribution, needing more bins
    to capture the peak accurately.
    """
    if confidence >= 0.9:
        return 64
    if confidence >= 0.7:
        return 32
    return DEFAULT_K


# ═══════════════════════════════════════════════════════════════════════════
#  Weight computation (pure JAX, no thrml dependency)
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
    return CategoricalEBMFactor([Block([parent]), Block([child])], w[None, :, :])


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

    Parameters
    ----------
    priors : list[float]            strength of each node's prior
    confidences : list[float]       confidence of each node's prior
    strengths : list[float]         implication strength for each edge
    impl_confidences : list[float]  implication confidence for each edge
    backgrounds : list[float]       background rate for each edge
    clamp_root : bool               If True, clamp root to its prior (good for
                                    marginal queries).  If False, all nodes are
                                    free (needed for conditional queries).
    """
    n = len(priors)
    nodes = [CategoricalNode() for _ in range(n)]

    factors = []

    if clamp_root:
        # Prior factors for non-root nodes only (root is clamped)
        for i in range(1, n):
            factors.append(make_beta_prior_factor(nodes[i], priors[i], confidences[i], k))
    else:
        # All nodes get priors
        for i in range(n):
            factors.append(make_beta_prior_factor(nodes[i], priors[i], confidences[i], k))

    # Implication factors for each edge
    for i in range(n - 1):
        factors.append(make_beta_implication_factor(
            nodes[i], nodes[i + 1],
            strengths[i], impl_confidences[i], backgrounds[i], k))

    if clamp_root:
        # Root is clamped; rest are free with 2-coloring
        clamped_blocks = [Block([nodes[0]])]
        free_nodes = nodes[1:]

        if not free_nodes:
            free_blocks = []
            spec = BlockGibbsSpec([], clamped_blocks)
            prog = FactorSamplingProgram(
                gibbs_spec=spec, samplers=[], factors=factors,
                other_interaction_groups=[])
            root_logits = _beta_prior_logits(priors[0], confidences[0], k)
            return dict(nodes=nodes, factors=factors, free_blocks=free_blocks,
                        clamped_blocks=clamped_blocks, spec=spec, program=prog,
                        n=n, k=k, root_logits=root_logits, single_node=True)

        even = [free_nodes[i] for i in range(0, len(free_nodes), 2)]
        odd = [free_nodes[i] for i in range(1, len(free_nodes), 2)]
        free_blocks = [Block(even), Block(odd)] if odd else [Block(even)]

        spec = BlockGibbsSpec(free_blocks, clamped_blocks)
        sampler = CategoricalGibbsConditional(n_categories=k)
        prog = FactorSamplingProgram(
            gibbs_spec=spec,
            samplers=[sampler] * len(free_blocks),
            factors=factors,
            other_interaction_groups=[],
        )
        root_logits = _beta_prior_logits(priors[0], confidences[0], k)

        return dict(nodes=nodes, factors=factors, free_blocks=free_blocks,
                    clamped_blocks=clamped_blocks, spec=spec, program=prog,
                    n=n, k=k, root_logits=root_logits, single_node=False)
    else:
        # All nodes free, 2-coloring
        even = [nodes[i] for i in range(0, n, 2)]
        odd = [nodes[i] for i in range(1, n, 2)]
        free_blocks = [Block(even), Block(odd)] if odd else [Block(even)]
        return _assemble_free_graph(nodes, factors, free_blocks, k, n=n)


def build_beta_v_graph(root_prior, root_confidence,
                       left_strength, right_strength,
                       left_impl_confidence, right_impl_confidence,
                       left_background, right_background,
                       left_prior=0.5, left_confidence=0.01,
                       right_prior=0.5, right_confidence=0.01,
                       k=DEFAULT_K):
    """V-shape:  Left ← Root → Right   (induction topology) with K-bin nodes.

    All nodes are free.

    Returns dict with keys: root, left, right, nodes, factors, free_blocks,
                            spec, program, k
    """
    root = CategoricalNode()
    left = CategoricalNode()
    right = CategoricalNode()

    factors = [
        make_beta_prior_factor(root, root_prior, root_confidence, k),
        make_beta_prior_factor(left, left_prior, left_confidence, k),
        make_beta_prior_factor(right, right_prior, right_confidence, k),
        make_beta_implication_factor(root, left, left_strength,
                                     left_impl_confidence, left_background, k),
        make_beta_implication_factor(root, right, right_strength,
                                     right_impl_confidence, right_background, k),
    ]

    # Coloring: {left, right} and {root}
    free_blocks = [Block([left, right]), Block([root])]
    return _assemble_free_graph(
        [root, left, right], factors, free_blocks, k,
        root=root, left=left, right=right)


def build_beta_inv_v_graph(left_prior, left_confidence,
                           right_prior, right_confidence,
                           left_strength, right_strength,
                           left_impl_confidence, right_impl_confidence,
                           left_background, right_background,
                           center_prior=0.5, center_confidence=0.01,
                           k=DEFAULT_K):
    """Inverted-V:  Left → Center ← Right   (abduction topology) with K-bin nodes.

    All nodes are free (no clamping).

    Returns dict with keys: left, center, right, nodes, factors, free_blocks,
                            spec, program, k
    """
    left = CategoricalNode()
    center = CategoricalNode()
    right = CategoricalNode()

    factors = [
        make_beta_prior_factor(left, left_prior, left_confidence, k),
        make_beta_prior_factor(right, right_prior, right_confidence, k),
        make_beta_prior_factor(center, center_prior, center_confidence, k),
        make_beta_implication_factor(left, center, left_strength,
                                     left_impl_confidence, left_background, k),
        make_beta_implication_factor(right, center, right_strength,
                                     right_impl_confidence, right_background, k),
    ]

    # Coloring: {left, right} and {center}
    free_blocks = [Block([left, right]), Block([center])]
    return _assemble_free_graph(
        [left, center, right], factors, free_blocks, k,
        left=left, center=center, right=right)


def build_beta_symmetric_pair(prior_a, confidence_a,
                              prior_b, confidence_b,
                              strength, impl_confidence,
                              background=DEFAULT_EPSILON,
                              k=DEFAULT_K):
    """Symmetric 2-node graph:  A ↔ B  (Equivalence/Similarity) with K-bin nodes.

    Both nodes are free (no clamping).

    Returns dict with keys: a, b, nodes, factors, free_blocks, spec, program, k
    """
    a = CategoricalNode()
    b = CategoricalNode()

    factors = [
        make_beta_prior_factor(a, prior_a, confidence_a, k),
        make_beta_prior_factor(b, prior_b, confidence_b, k),
        make_beta_implication_factor(a, b, strength, impl_confidence, background, k),
        make_beta_implication_factor(b, a, strength, impl_confidence, background, k),
    ]

    free_blocks = [Block([a]), Block([b])]
    return _assemble_free_graph([a, b], factors, free_blocks, k, a=a, b=b)


def build_beta_symmetric_chain(priors, confidences, strengths, impl_confidences,
                               backgrounds, k=DEFAULT_K):
    """Symmetric chain  X_0 ↔ X_1 ↔ ... ↔ X_{n-1}  with K-bin nodes.

    Each edge has bidirectional coupling.  All nodes get Beta priors.
    All nodes are free (no clamping).

    Returns dict with keys: nodes, factors, free_blocks, spec, program, n, k
    """
    n = len(priors)
    nodes = [CategoricalNode() for _ in range(n)]

    factors = []
    for i in range(n):
        factors.append(make_beta_prior_factor(nodes[i], priors[i], confidences[i], k))

    for i in range(n - 1):
        factors.append(make_beta_implication_factor(
            nodes[i], nodes[i + 1],
            strengths[i], impl_confidences[i], backgrounds[i], k))
        factors.append(make_beta_implication_factor(
            nodes[i + 1], nodes[i],
            strengths[i], impl_confidences[i], backgrounds[i], k))

    # 2-coloring: even indices / odd indices
    even = [nodes[i] for i in range(0, n, 2)]
    odd = [nodes[i] for i in range(1, n, 2)]
    free_blocks = [Block(even), Block(odd)] if odd else [Block(even)]
    return _assemble_free_graph(nodes, factors, free_blocks, k, n=n)


def _greedy_color(names, adjacency):
    """Greedy graph coloring — nodes of the same color share no factors and can be sampled in parallel.

    Parameters
    ----------
    names : list[str]
        Sorted node names (deterministic ordering).
    adjacency : dict[str, set[str]]
        For each node, the set of nodes it shares a factor edge with.

    Returns
    -------
    list[list[str]]
        Groups of node names, one list per color.
    """
    color_of = {}
    for name in names:
        neighbor_colors = {color_of[nb] for nb in adjacency.get(name, set())
                          if nb in color_of}
        c = 0
        while c in neighbor_colors:
            c += 1
        color_of[name] = c
    n_colors = max(color_of.values()) + 1 if color_of else 1
    groups = [[] for _ in range(n_colors)]
    for name in names:
        groups[color_of[name]].append(name)
    return [g for g in groups if g]


def build_beta_full_graph(priors, implications, similarities=None,
                          equivalences=None, backgrounds=None, k=DEFAULT_K):
    """Compile an entire knowledge base into one beta factor graph.

    Unlike the per-rule builders, this accepts *all* knowledge at once and
    produces a single factor graph.  One round of Gibbs sampling then yields
    samples from the full joint distribution.

    All nodes are free (no clamping).  Graph-coloring block assignment for
    efficient mixing — nodes that share no factor edge are sampled together.

    Parameters
    ----------
    priors : dict[str, {"strength": float, "confidence": float}]
    implications : list[{"src", "dst", "strength", "confidence"}]
    similarities : list[{"src", "dst", "strength", "confidence"}] | None
    equivalences : list[{"src", "dst", "strength", "confidence"}] | None
    backgrounds : dict[(str, str), float] | None
    """
    similarities = similarities or []
    equivalences = equivalences or []
    backgrounds = backgrounds or {}

    # Collect all node names
    names: set[str] = set(priors.keys())
    for link in implications:
        names.add(link["src"])
        names.add(link["dst"])
    for link in similarities + equivalences:
        names.add(link["src"])
        names.add(link["dst"])

    name_to_node = {name: CategoricalNode() for name in sorted(names)}

    # Build factors
    factors = []

    # Priors
    for name, node in name_to_node.items():
        p = priors.get(name)
        s = p["strength"] if p else 0.5
        c = p["confidence"] if p else 0.01
        factors.append(make_beta_prior_factor(node, s, c, k))

    # Directed links
    for link in implications:
        parent = name_to_node[link["src"]]
        child = name_to_node[link["dst"]]
        bg = backgrounds.get((link["src"], link["dst"]), DEFAULT_EPSILON)
        factors.append(make_beta_implication_factor(
            parent, child, link["strength"], link["confidence"], bg, k))

    # Symmetric links → bidirectional
    for link in similarities + equivalences:
        a = name_to_node[link["src"]]
        b = name_to_node[link["dst"]]
        bg = backgrounds.get((link["src"], link["dst"]), DEFAULT_EPSILON)
        factors.append(make_beta_implication_factor(
            a, b, link["strength"], link["confidence"], bg, k))
        factors.append(make_beta_implication_factor(
            b, a, link["strength"], link["confidence"], bg, k))

    # Build adjacency from factor links
    sorted_names = sorted(names)
    adjacency = {name: set() for name in sorted_names}
    for link in implications:
        adjacency[link["src"]].add(link["dst"])
        adjacency[link["dst"]].add(link["src"])
    for link in similarities + equivalences:
        adjacency[link["src"]].add(link["dst"])
        adjacency[link["dst"]].add(link["src"])

    # Graph coloring → one Block per color group
    color_groups = _greedy_color(sorted_names, adjacency)
    free_blocks = [Block([name_to_node[n] for n in group]) for group in color_groups]
    spec = BlockGibbsSpec(free_blocks, [])
    samplers = [CategoricalGibbsConditional(n_categories=k) for _ in free_blocks]

    prog = FactorSamplingProgram(
        gibbs_spec=spec,
        samplers=samplers,
        factors=factors,
        other_interaction_groups=[],
    )
    return dict(nodes=name_to_node, factors=factors, free_blocks=free_blocks,
                clamped_blocks=[], spec=spec, program=prog, k=k, single_node=False)


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
        # Store for later use by estimate_beta_conditional
        graph["_root_clamped_states"] = root_state
    else:
        state_clamp = []

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

    return samples


def sample_and_measure(graph, target_node, seed=42):
    """Run sampling and return (strength, confidence) for a target node.

    Convenience wrapper combining run_beta_sampling + estimate_beta_marginal.
    """
    samples = run_beta_sampling(graph, seed=seed)
    _, strength, confidence = estimate_beta_marginal(samples, graph, target_node)
    return strength, confidence


# ═══════════════════════════════════════════════════════════════════════════
#  Convergence diagnostics
# ═══════════════════════════════════════════════════════════════════════════

def diagnose_convergence(samples, graph, node):
    """Compute convergence diagnostics for a node's samples.

    Uses batches as independent chains to compute split-R-hat and ESS.

    Parameters
    ----------
    samples : raw samples from run_beta_sampling
    graph : factor graph dict
    node : target CategoricalNode

    Returns
    -------
    dict with keys:
        r_hat : float — split-R-hat statistic (< 1.05 is good)
        ess : int — effective sample size
        converged : bool — True if R-hat < 1.05 and ESS > 400
    """
    if isinstance(samples, dict) and "root_samples" in samples:
        # Single-node graph: samples drawn directly from prior, always converged
        n_total = samples["root_samples"].size
        return {"r_hat": 1.0, "ess": int(n_total), "converged": True}

    bi, ni = _node_location(graph, node)
    k = graph.get("k", DEFAULT_K)
    centers = bin_centers(k)

    # raw shape: [n_batches, n_samples, n_nodes_in_block]
    raw = samples[bi][:, :, ni]  # [n_batches, n_samples]
    n_batches, n_samples = raw.shape

    # Convert categorical bins to continuous values for R-hat
    values = centers[raw]  # [n_batches, n_samples]

    # --- Split-R-hat ---
    # Split each chain (batch) in half
    half = n_samples // 2
    first_half = values[:, :half]
    second_half = values[:, half:2*half]
    # Stack as 2*n_batches chains
    chains = jnp.concatenate([first_half, second_half], axis=0)  # [2*n_batches, half]
    m = chains.shape[0]  # number of split chains
    n = chains.shape[1]  # length of each split chain

    chain_means = jnp.mean(chains, axis=1)  # [m]
    grand_mean = jnp.mean(chain_means)

    # Between-chain variance
    B = n * jnp.var(chain_means, ddof=1) if m > 1 else 0.0
    # Within-chain variance
    chain_vars = jnp.var(chains, axis=1, ddof=1)  # [m]
    W = jnp.mean(chain_vars)

    if W < 1e-12:
        r_hat = 1.0
    else:
        var_hat = (n - 1) / n * W + B / n
        r_hat = float(jnp.sqrt(var_hat / W))

    # --- ESS (via autocorrelation) ---
    # Use the grand chain (all batches concatenated)
    grand_chain = values.flatten()
    n_total = len(grand_chain)
    grand_mean_val = float(jnp.mean(grand_chain))
    grand_var = float(jnp.var(grand_chain))

    if grand_var < 1e-12:
        ess = n_total
    else:
        # Compute autocorrelation up to lag 100
        max_lag = min(100, n_total // 4)
        centered = grand_chain - grand_mean_val
        acf_sum = 0.0
        for lag in range(1, max_lag + 1):
            acf = float(jnp.mean(centered[:-lag] * centered[lag:])) / grand_var
            if acf < 0.05:
                break
            acf_sum += acf
        ess = max(1, int(n_total / (1 + 2 * acf_sum)))

    converged = (r_hat < 1.05) and (ess > 400)
    return {"r_hat": r_hat, "ess": ess, "converged": converged}


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

    bi, ni = _node_location(graph, node)
    flat = _flatten_node(samples, bi, ni)
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
    """Build weighted histogram of target bins, weighted by condition bin centers.

    Weight = bin_center[condition_bin] — how "true" the condition is.
    This gives a proper Bayesian conditional E[target | condition=True].
    """
    centers = bin_centers(k)
    weights = centers[condition_bins]  # "truthiness" of each condition sample

    # Weighted histogram: posterior[j] = sum of weights where target == j
    posterior = jnp.zeros(k)
    for j in range(k):
        mask = (target_bins == j)
        posterior = posterior.at[j].set(jnp.sum(weights * mask))

    total = jnp.sum(posterior)
    if total < 1e-10:
        return jnp.ones(k) / k
    return posterior / total


def estimate_beta_conditional(samples, graph, target, condition, k=None):
    """P(target | condition=True) — weighted conditional posterior.

    Uses bin-center weighting: each sample is weighted by the "truthiness"
    of the condition variable (bin_center[condition_state]).  This gives
    the proper Bayesian conditional E[target | condition is True].

    Handles clamped root as either condition or target node.

    Returns (posterior, strength, confidence).
    """
    if k is None:
        k = graph["k"]

    cond_is_clamped = _is_clamped_root(graph, condition)
    target_is_clamped = _is_clamped_root(graph, target)

    def _get_root_states():
        root_states = graph.get("_root_clamped_states")
        if root_states is None:
            raise ValueError("Root clamped states not available. "
                             "Use run_beta_sampling which stores them.")
        return root_states[:, 0]  # [n_batches]

    if target_is_clamped and cond_is_clamped:
        raise ValueError("Cannot condition clamped root on itself")

    centers = bin_centers(k)

    if cond_is_clamped:
        # Condition is clamped root, target is free
        tbi, tni = _node_location(graph, target)
        t_raw = samples[tbi][:, :, tni]  # [n_batches, n_samples]
        root_flat = _get_root_states()   # [n_batches]

        # Weight each batch by root's "truthiness"
        root_weights = centers[root_flat]  # [n_batches]

        # Weighted histogram of target across all batches
        posterior = jnp.zeros(k)
        for b in range(t_raw.shape[0]):
            batch_hist = jnp.bincount(
                t_raw[b].astype(jnp.int32), length=k).astype(jnp.float32)
            posterior = posterior + root_weights[b] * batch_hist

        total = jnp.sum(posterior)
        if total < 1e-10:
            return jnp.ones(k) / k, 0.5, 0.0
        posterior = posterior / total
        strength, confidence = posterior_to_stv(posterior, k)
        return posterior, strength, confidence

    if target_is_clamped:
        # Target is clamped root, condition is free — inversion case
        cbi, cni = _node_location(graph, condition)
        c_raw = samples[cbi][:, :, cni]  # [n_batches, n_samples]
        root_flat = _get_root_states()    # [n_batches]

        # For each batch, compute mean condition "truthiness"
        # Then weight root's bin by that truthiness
        posterior = jnp.zeros(k)
        for b in range(c_raw.shape[0]):
            cond_truthiness = float(jnp.mean(centers[c_raw[b]]))
            root_bin = int(root_flat[b])
            posterior = posterior.at[root_bin].add(cond_truthiness)

        total = jnp.sum(posterior)
        if total < 1e-10:
            return jnp.ones(k) / k, 0.5, 0.0
        posterior = posterior / total
        strength, confidence = posterior_to_stv(posterior, k)
        return posterior, strength, confidence

    # Standard case: both target and condition are in free blocks
    tbi, tni = _node_location(graph, target)
    cbi, cni = _node_location(graph, condition)
    t_flat = _flatten_node(samples, tbi, tni)
    c_flat = _flatten_node(samples, cbi, cni)

    posterior = _weighted_histogram(
        t_flat.astype(jnp.int32), c_flat.astype(jnp.int32), k)
    strength, confidence = posterior_to_stv(posterior, k)
    return posterior, strength, confidence
