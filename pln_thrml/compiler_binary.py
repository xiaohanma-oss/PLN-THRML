"""
pln_thrml.compiler_binary — Binary Ising compiler for (ρ, n) separation
=========================================================================

Compiles PLN factor graphs to 1-pbit-per-proposition Ising representations.
Each proposition is a single SpinNode (True/False), and each implication
edge is a single Ising J coupling.  This is the simplest possible hardware
target — no K-bin discretisation, no one-hot encoding, no Potts approximation.

The 2×2 conditional probability table:

              child=F         child=T
    parent=F   1 - ε            ε           (background)
    parent=T   1 - s_link       s_link      (implication strength)

maps exactly to Ising parameters (J coupling + h bias), with zero
approximation error.  Strength s is recovered from sampling frequency;
confidence c is computed separately via PLN closed-form formulas.

TSU hardware mapping:
  - 1 pbit per proposition
  - 1 Ising coupling (J) per implication edge
  - 1 bias (h) per proposition
  - ~12 neighbour budget → up to 12 implication edges per proposition
"""

import numpy as np
import jax
import jax.numpy as jnp

from thrml.pgm import SpinNode
from thrml.models.discrete_ebm import SpinEBMFactor, SpinGibbsConditional
from thrml.block_management import Block
from thrml.block_sampling import BlockGibbsSpec, SamplingSchedule, sample_states
from thrml.factor import FactorSamplingProgram

from pln_thrml.beta import EPS, DEFAULT_EPSILON


__all__ = [
    "ising_params",
    "prior_bias",
    "compile_binary_chain",
    "compile_binary_inv_v",
    "compile_binary_v",
    "run_binary_sampling",
    "estimate_binary_marginal",
    "compile_joint_categorical",
    "run_joint_sampling",
    "estimate_joint_marginal",
    "compile_binary_chain_with_hidden",
]

# Default sampling schedule — lighter than one-hot (no exclusion constraints,
# faster mixing for binary spins).
DEFAULT_BINARY_N_BATCHES = 50
DEFAULT_BINARY_SCHEDULE = SamplingSchedule(
    n_warmup=500, n_samples=2000, steps_per_sample=2,
)


# ═══════════════════════════════════════════════════════════════════════════
#  Ising parameter calculation
# ═══════════════════════════════════════════════════════════════════════════

def ising_params(s_link, background=DEFAULT_EPSILON):
    """Compute Ising coupling J from a 2×2 conditional probability table.

    The CPT is:
        P(child=T | parent=F) = background  (ε)
        P(child=T | parent=T) = s_link

    In {-1, +1} Ising notation, the energy contribution of the pairwise
    factor is  E = -J · s_parent · s_child.  We derive J by matching the
    Boltzmann ratio to the CPT ratio.

    Parameters
    ----------
    s_link : float
        Implication strength P(child=T | parent=T).
    background : float
        Background rate P(child=T | parent=F).

    Returns
    -------
    J : float
        Ising coupling weight.
    h_child_correction : float
        Additional bias on the child spin to account for marginal asymmetry.
    """
    s = float(np.clip(s_link, EPS, 1.0 - EPS))
    bg = float(np.clip(background, EPS, 1.0 - EPS))

    # 2×2 table entries:
    #   p00 = P(child=F | parent=F) = 1 - bg
    #   p01 = P(child=T | parent=F) = bg
    #   p10 = P(child=F | parent=T) = 1 - s
    #   p11 = P(child=T | parent=T) = s
    #
    # Ising energy for valid configs (parent, child ∈ {-1, +1}):
    #   E(+1, +1) = -J - h_p - h_c
    #   E(+1, -1) = +J - h_p + h_c
    #   E(-1, +1) = +J + h_p - h_c
    #   E(-1, -1) = -J + h_p + h_c
    #
    # J = (1/4) ln[ p00 * p11 / (p01 * p10) ]
    J = 0.25 * np.log((1.0 - bg) * s / (bg * (1.0 - s)))

    # Child correction: accounts for the fact that the marginal
    # P(child=T) differs between parent=T and parent=F rows.
    # h_correction = (1/4) ln[ p01 * p11 / (p00 * p10) ]
    h_child_correction = 0.25 * np.log(bg * s / ((1.0 - bg) * (1.0 - s)))

    return float(J), float(h_child_correction)


def prior_bias(s):
    """Compute unary bias h encoding a prior strength.

    In Ising {-1, +1}: P(spin=+1) = σ(2h) = s, so h = 0.5 * ln(s/(1-s)).

    Parameters
    ----------
    s : float
        Prior strength (probability of True).

    Returns
    -------
    h : float
        Unary bias.  Positive favours +1 (True), negative favours -1 (False).
    """
    s = float(np.clip(s, EPS, 1.0 - EPS))
    return 0.5 * np.log(s / (1.0 - s))


# ═══════════════════════════════════════════════════════════════════════════
#  Graph builders
# ═══════════════════════════════════════════════════════════════════════════

def _assemble_binary_graph_ex(spins, blocks, free_blocks, clamped_blocks,
                              factors, root_prior=None):
    """Assemble binary Ising graph into a sampling program."""
    conditional = SpinGibbsConditional()
    super_blocks = [(b,) for b in free_blocks]
    spec = BlockGibbsSpec(free_super_blocks=super_blocks,
                          clamped_blocks=clamped_blocks)
    program = FactorSamplingProgram(
        gibbs_spec=spec,
        samplers=[conditional] * len(free_blocks),
        factors=factors,
        other_interaction_groups=[],
    )
    return {
        "spins": spins,
        "blocks": blocks,
        "free_blocks": free_blocks,
        "clamped_blocks": clamped_blocks,
        "factors": factors,
        "spec": spec,
        "program": program,
        "n": len(spins),
        "root_prior": root_prior,
    }


def compile_binary_chain(priors, strengths, backgrounds=None,
                         clamp_root=True):
    """Build a binary Ising factor graph for a directed chain.

    Topology: X₀ → X₁ → ... → Xₙ₋₁

    Parameters
    ----------
    priors : list[float]
        Prior strength for each node.  priors[0] is the root.
    strengths : list[float]
        Implication strengths.  strengths[i] = P(X_{i+1}=T | X_i=T).
    backgrounds : list[float] or None
        Background rates per edge.  Defaults to DEFAULT_EPSILON.
    clamp_root : bool
        If True (default), root is clamped per batch by sampling from
        Bernoulli(priors[0]).  This gives correct modus-ponens marginals.
        If False, all nodes are free (for deduction / conditioning).

    Returns
    -------
    dict
        Graph dict with spins, blocks, factors, program, n.
    """
    n = len(priors)
    if backgrounds is None:
        backgrounds = [DEFAULT_EPSILON] * (n - 1)

    spins = [SpinNode() for _ in range(n)]
    blocks = [Block([s]) for s in spins]
    factors = []

    # Determine free vs clamped blocks
    if clamp_root:
        free_blocks = blocks[1:]  # root is clamped
        clamped_blocks = [blocks[0]]
    else:
        free_blocks = blocks
        clamped_blocks = []

    # Unary biases (priors) — skip root if clamped (prior handled by sampling)
    start = 1 if clamp_root else 0
    for i in range(start, n):
        h = prior_bias(priors[i])
        if abs(h) > 1e-10:
            factors.append(SpinEBMFactor([blocks[i]], jnp.array([h])))

    # Pairwise couplings (implications)
    for i in range(n - 1):
        J, h_corr = ising_params(strengths[i], backgrounds[i])
        if abs(J) > 1e-10:
            factors.append(SpinEBMFactor(
                [blocks[i], blocks[i + 1]], jnp.array([J])))
        # Apply child correction as additional bias on child
        if abs(h_corr) > 1e-10:
            factors.append(SpinEBMFactor(
                [blocks[i + 1]], jnp.array([h_corr])))

    return _assemble_binary_graph_ex(
        spins, blocks, free_blocks, clamped_blocks, factors,
        root_prior=priors[0] if clamp_root else None)


def compile_binary_inv_v(left_prior, right_prior,
                         left_strength, right_strength,
                         left_background=DEFAULT_EPSILON,
                         right_background=DEFAULT_EPSILON,
                         center_prior=0.5):
    """Build a binary Ising inverted-V: Left → Center ← Right.

    Used for abduction: two causes independently imply a common effect.

    Parameters
    ----------
    left_prior, right_prior : float
        Prior strengths for left and right nodes.
    left_strength, right_strength : float
        Implication strengths for Left→Center and Right→Center.
    center_prior : float
        Prior strength for center node (typically 0.5 = uninformative).

    Returns
    -------
    dict
        Graph dict.  Node order: [left, center, right].
    """
    spins = [SpinNode() for _ in range(3)]
    blocks = [Block([s]) for s in spins]
    factors = []

    # Biases
    for i, p in enumerate([left_prior, center_prior, right_prior]):
        h = prior_bias(p)
        if abs(h) > 1e-10:
            factors.append(SpinEBMFactor([blocks[i]], jnp.array([h])))

    # Left → Center
    J_left, h_corr_left = ising_params(left_strength, left_background)
    if abs(J_left) > 1e-10:
        factors.append(SpinEBMFactor(
            [blocks[0], blocks[1]], jnp.array([J_left])))
    if abs(h_corr_left) > 1e-10:
        factors.append(SpinEBMFactor([blocks[1]], jnp.array([h_corr_left])))

    # Right → Center
    J_right, h_corr_right = ising_params(right_strength, right_background)
    if abs(J_right) > 1e-10:
        factors.append(SpinEBMFactor(
            [blocks[2], blocks[1]], jnp.array([J_right])))
    if abs(h_corr_right) > 1e-10:
        factors.append(SpinEBMFactor([blocks[1]], jnp.array([h_corr_right])))

    return _assemble_binary_graph_ex(
        spins, blocks, blocks, [], factors, root_prior=None)


def compile_binary_v(root_prior, left_strength, right_strength,
                     left_background=DEFAULT_EPSILON,
                     right_background=DEFAULT_EPSILON,
                     left_prior=0.5, right_prior=0.5):
    """Build a binary Ising V-shape: Left ← Root → Right.

    Used for induction: a common cause implies two effects.

    Parameters
    ----------
    root_prior : float
        Prior strength for root node.
    left_strength, right_strength : float
        Implication strengths for Root→Left and Root→Right.
    left_prior, right_prior : float
        Prior strengths for left and right nodes.

    Returns
    -------
    dict
        Graph dict.  Node order: [left, root, right].
    """
    spins = [SpinNode() for _ in range(3)]
    blocks = [Block([s]) for s in spins]
    factors = []

    # Biases
    for i, p in enumerate([left_prior, root_prior, right_prior]):
        h = prior_bias(p)
        if abs(h) > 1e-10:
            factors.append(SpinEBMFactor([blocks[i]], jnp.array([h])))

    # Root → Left
    J_left, h_corr_left = ising_params(left_strength, left_background)
    if abs(J_left) > 1e-10:
        factors.append(SpinEBMFactor(
            [blocks[1], blocks[0]], jnp.array([J_left])))
    if abs(h_corr_left) > 1e-10:
        factors.append(SpinEBMFactor([blocks[0]], jnp.array([h_corr_left])))

    # Root → Right
    J_right, h_corr_right = ising_params(right_strength, right_background)
    if abs(J_right) > 1e-10:
        factors.append(SpinEBMFactor(
            [blocks[1], blocks[2]], jnp.array([J_right])))
    if abs(h_corr_right) > 1e-10:
        factors.append(SpinEBMFactor([blocks[2]], jnp.array([h_corr_right])))

    return _assemble_binary_graph_ex(
        spins, blocks, blocks, [], factors, root_prior=None)


# ═══════════════════════════════════════════════════════════════════════════
#  Sampling and measurement
# ═══════════════════════════════════════════════════════════════════════════

def run_binary_sampling(graph, seed=42, n_batches=None, schedule=None):
    """Run block Gibbs sampling on a binary Ising graph.

    Supports two modes:
    - Clamped root (root_prior is set): each batch draws the root from
      Bernoulli(root_prior), then runs Gibbs on free nodes only.
    - All free (root_prior is None): standard Gibbs on all nodes.

    Uses jax.vmap over batches for efficient parallel execution (same
    pattern as beta.py's run_beta_sampling).
    """
    if n_batches is None:
        n_batches = DEFAULT_BINARY_N_BATCHES
    if schedule is None:
        schedule = DEFAULT_BINARY_SCHEDULE

    free_blocks = graph["free_blocks"]
    program = graph["program"]
    root_prior = graph.get("root_prior")
    has_clamped = root_prior is not None
    n_free = len(free_blocks)

    key = jax.random.key(seed)

    # Initialise free blocks: random bool per spin, shape [n_batches, 1] per block
    init_state = []
    for i in range(n_free):
        key, subkey = jax.random.split(key)
        init_state.append(
            jax.random.bernoulli(subkey, 0.5, (n_batches, 1)))

    if has_clamped:
        # Sample root per batch from Bernoulli(root_prior)
        key, subkey = jax.random.split(key)
        root_bits = jax.random.bernoulli(subkey, root_prior, (n_batches, 1))
        state_clamp = [root_bits]
    else:
        state_clamp = []
        root_bits = None

    keys = jax.random.split(key, n_batches)
    observe_blocks = list(free_blocks)

    if has_clamped:
        samples = jax.jit(jax.vmap(
            lambda s, c, k_: sample_states(k_, program, schedule, s, c,
                                           observe_blocks)
        ))(init_state, state_clamp, keys)
    else:
        samples = jax.jit(jax.vmap(
            lambda s, k_: sample_states(k_, program, schedule, s, [],
                                       observe_blocks)
        ))(init_state, keys)

    if has_clamped:
        return {"vmap_samples": samples, "root_bits": root_bits}
    return {"vmap_samples": samples, "root_bits": None}


def estimate_binary_marginal(samples_dict, graph, node_idx):
    """Estimate strength s from binary spin sampling frequency.

    Parameters
    ----------
    samples_dict : dict
        Output of run_binary_sampling.
    graph : dict
        Graph dict from compile_binary_*.
    node_idx : int
        Index into graph["spins"] (0-based).

    Returns
    -------
    s : float
        Estimated strength = P(proposition = True).
    """
    root_bits = samples_dict["root_bits"]
    has_clamped = root_bits is not None

    # If node is the clamped root (node_idx == 0 and has_clamped),
    # return the empirical frequency from the clamped draws.
    if has_clamped and node_idx == 0:
        return float(jnp.mean(root_bits))

    # For free nodes: determine which index in free_blocks corresponds
    # to this node.
    if has_clamped:
        free_idx = node_idx - 1
    else:
        free_idx = node_idx

    # vmap_samples: list of arrays, each shape [n_batches, n_samples, 1]
    vmap_samples = samples_dict["vmap_samples"]
    spin_data = vmap_samples[free_idx][:, :, 0]  # [n_batches, n_samples]
    return float(jnp.mean(spin_data))


# ═══════════════════════════════════════════════════════════════════════════
#  QLN block-diagonal: joint categorical (pdit K=2^n)
# ═══════════════════════════════════════════════════════════════════════════

def compile_joint_categorical(priors, strengths, backgrounds=None):
    """QLN block: 1 CategoricalNode(K=2^n) encoding the joint distribution.

    Maps n propositions in a directed chain to a single K=2^n categorical
    variable whose states represent all joint True/False assignments.

    State k encodes the assignment (a₁,...,aₙ) where aᵢ = bit i of k.
    Weight w[k] = log P(a₁,...,aₙ) = log P(a₁) + Σ log P(aᵢ₊₁|aᵢ).

    Parameters
    ----------
    priors : list[float]
        Prior strength for each node.
    strengths : list[float]
        Implication strengths for each edge.
    backgrounds : list[float] or None
        Background rates per edge.

    Returns
    -------
    dict
        Graph dict with node, block, program, n_props, K.
    """
    from thrml.pgm import CategoricalNode
    from thrml.models.discrete_ebm import (CategoricalEBMFactor,
                                           CategoricalGibbsConditional)

    n = len(priors)
    if backgrounds is None:
        backgrounds = [DEFAULT_EPSILON] * (n - 1)

    K = 2 ** n  # joint state count

    # Build joint probability table
    log_probs = np.zeros(K)
    for k in range(K):
        bits = [(k >> i) & 1 for i in range(n)]  # bit i = proposition i
        # P(a₁)
        s0 = priors[0]
        p = s0 if bits[0] else (1.0 - s0)
        log_p = np.log(max(p, EPS))
        # Π P(aᵢ₊₁ | aᵢ)
        for i in range(n - 1):
            s_link = float(np.clip(strengths[i], EPS, 1.0 - EPS))
            bg = float(np.clip(backgrounds[i], EPS, 1.0 - EPS))
            if bits[i]:   # parent = True
                cond = s_link if bits[i + 1] else (1.0 - s_link)
            else:         # parent = False
                cond = bg if bits[i + 1] else (1.0 - bg)
            log_p += np.log(max(cond, EPS))
        log_probs[k] = log_p

    # Centre for numerical stability
    log_probs -= log_probs.max()

    # Build CategoricalNode graph
    node = CategoricalNode()
    block = Block([node])
    factor = CategoricalEBMFactor([block], jnp.array(log_probs)[None, :])

    conditional = CategoricalGibbsConditional(n_categories=K)
    spec = BlockGibbsSpec(free_super_blocks=[(block,)], clamped_blocks=[])
    program = FactorSamplingProgram(
        gibbs_spec=spec,
        samplers=[conditional],
        factors=[factor],
        other_interaction_groups=[],
    )

    return {
        "node": node,
        "block": block,
        "factors": [factor],
        "spec": spec,
        "program": program,
        "n_props": n,
        "K": K,
        "log_probs": log_probs,
    }


def run_joint_sampling(graph, seed=42, n_batches=None, schedule=None):
    """Sample from a joint categorical graph. Returns raw categorical samples."""
    if n_batches is None:
        n_batches = DEFAULT_BINARY_N_BATCHES
    if schedule is None:
        schedule = DEFAULT_BINARY_SCHEDULE

    block = graph["block"]
    program = graph["program"]
    K = graph["K"]

    all_samples = []
    for batch in range(n_batches):
        key = jax.random.PRNGKey(seed + batch)
        init_state = [jax.random.randint(
            jax.random.PRNGKey(seed + batch + 10000),
            (1,), 0, K, dtype=jnp.uint8)]
        results = sample_states(key, program, schedule, init_state, [], [block])
        all_samples.append(results)

    return all_samples


def estimate_joint_marginal(all_samples, graph, prop_idx):
    """Estimate P(proposition_i = True) from joint categorical samples.

    Decodes the categorical state k into bits and counts how often bit
    prop_idx is 1.
    """
    counts_true = 0
    counts_total = 0

    for batch_samples in all_samples:
        cat_samples = np.asarray(batch_samples[0][:, 0])  # [n_samples]
        bits = (cat_samples >> prop_idx) & 1
        counts_true += int(bits.sum())
        counts_total += len(bits)

    return float(counts_true) / float(counts_total)


# ═══════════════════════════════════════════════════════════════════════════
#  LBM-inspired: hidden units for base rate encoding
# ═══════════════════════════════════════════════════════════════════════════

def compile_binary_chain_with_hidden(priors, strengths, base_rates,
                                     backgrounds=None,
                                     clamp_root=True):
    """LBM-inspired: visible pbit chain + hidden pbits encoding base rates.

    Adds hidden SpinNodes that encode the base rate information missing
    from pairwise Ising couplings.  Each intermediate/tail node gets a
    hidden unit connected to it, whose coupling strength is derived from
    the node's base rate.

    Topology (n=3, deduction A→B→C):

        visible:  A ──J_AB── B ──J_BC── C
                              │          │
        hidden:              h_B        h_C

    Parameters
    ----------
    priors : list[float]
        Prior strength for each visible node.
    strengths : list[float]
        Implication strengths for each edge.
    base_rates : list[float]
        Base rate (prior strength) for nodes 1..n-1.
        Used to compute hidden unit couplings.
        Length: n-1 (one per non-root node).
    backgrounds : list[float] or None
        Background rates per edge.
    clamp_root : bool
        If True, root is clamped per batch (for MP-style inference).

    Returns
    -------
    dict
        Graph dict.  Visible nodes are indices 0..n-1,
        hidden nodes are indices n..2n-2.
    """
    n = len(priors)
    if backgrounds is None:
        backgrounds = [DEFAULT_EPSILON] * (n - 1)

    n_hidden = n - 1  # one hidden per non-root node

    # Create visible + hidden spins
    visible_spins = [SpinNode() for _ in range(n)]
    hidden_spins = [SpinNode() for _ in range(n_hidden)]
    all_spins = visible_spins + hidden_spins

    visible_blocks = [Block([s]) for s in visible_spins]
    hidden_blocks = [Block([s]) for s in hidden_spins]
    all_blocks = visible_blocks + hidden_blocks

    factors = []

    # Determine free vs clamped
    if clamp_root:
        free_blocks = visible_blocks[1:] + hidden_blocks
        clamped_blocks = [visible_blocks[0]]
    else:
        free_blocks = all_blocks
        clamped_blocks = []

    # Visible biases (skip root if clamped)
    start = 1 if clamp_root else 0
    for i in range(start, n):
        h = prior_bias(priors[i])
        if abs(h) > 1e-10:
            factors.append(SpinEBMFactor([visible_blocks[i]], jnp.array([h])))

    # Visible pairwise couplings (same as compile_binary_chain)
    for i in range(n - 1):
        J, h_corr = ising_params(strengths[i], backgrounds[i])
        if abs(J) > 1e-10:
            factors.append(SpinEBMFactor(
                [visible_blocks[i], visible_blocks[i + 1]], jnp.array([J])))
        if abs(h_corr) > 1e-10:
            factors.append(SpinEBMFactor(
                [visible_blocks[i + 1]], jnp.array([h_corr])))

    # Hidden unit couplings: encode base rates
    for i in range(n_hidden):
        br = float(np.clip(base_rates[i], EPS, 1.0 - EPS))
        vis_idx = i + 1  # visible node this hidden connects to

        # Hidden bias: encode the base rate as a field on the hidden unit
        h_hidden = prior_bias(br)
        if abs(h_hidden) > 1e-10:
            factors.append(SpinEBMFactor(
                [hidden_blocks[i]], jnp.array([h_hidden])))

        # Hidden-visible coupling: strength proportional to how much
        # the base rate deviates from 0.5 (uniform).
        # When br ≈ 0.5, coupling ≈ 0 (no correction needed).
        # When br is far from 0.5, coupling pulls the visible node
        # toward the base rate.
        J_hv = 0.25 * np.log(br / max(1.0 - br, EPS))
        if abs(J_hv) > 1e-10:
            factors.append(SpinEBMFactor(
                [hidden_blocks[i], visible_blocks[vis_idx]],
                jnp.array([J_hv])))

    return _assemble_binary_graph_ex(
        all_spins, all_blocks, free_blocks, clamped_blocks, factors,
        root_prior=priors[0] if clamp_root else None)
