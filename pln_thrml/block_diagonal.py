"""
block_diagonal.py — Block-diagonal inference for TSU hardware deployment
========================================================================

Partitions large factor graphs into small blocks (2-4 propositions each)
that fit within the TSU p-bit connection budget (~12 connections per p-bit).

Block internal: K×K full coupling via Gibbs sampling (exact).
Block boundary: marginal messages as soft-clamp prior factors (BP-style).

Tree-structured block graphs converge exactly (Pearl 1988).
Cyclic block graphs use damped loopy BP with bounded error.
"""

from collections import defaultdict, deque
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp

from thrml.block_management import Block
from thrml.pgm import CategoricalNode
from thrml.models.discrete_ebm import CategoricalEBMFactor

from pln_thrml.beta import (
    DEFAULT_K, DEFAULT_EPSILON, EPS,
    DEFAULT_BETA_N_BATCHES, DEFAULT_BETA_SCHEDULE,
    make_beta_prior_factor, make_beta_implication_factor,
    beta_implication_weights,
    _assemble_free_graph,
    run_beta_sampling, estimate_beta_marginal, posterior_to_stv,
)


__all__ = [
    "BlockPartition", "BlockDiagonalResult",
    "partition_into_blocks", "run_block_diagonal_sampling",
    "sample_and_measure_block_diagonal",
]


# ═══════════════════════════════════════════════════════════════════════════
#  Data structures
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class BlockPartition:
    """A partition of an inference graph into sub-blocks.

    Attributes
    ----------
    blocks : list[list[str]]
        Node names per block.
    boundary_nodes : dict[str, list[int]]
        Node name → list of block indices it appears in.
    cut_edges : list[tuple[str, str, dict]]
        Edges removed during partitioning (src, dst, link_dict).
    has_cycle : bool
        True if the block-level adjacency graph contains a cycle.
    """
    blocks: list[list[str]]
    boundary_nodes: dict[str, list[int]]
    cut_edges: list[tuple[str, str, dict]]
    has_cycle: bool


@dataclass
class BlockDiagonalResult:
    """Result of block-diagonal inference.

    Attributes
    ----------
    marginals : dict[str, jnp.ndarray]
        Node name → posterior histogram of shape [K].
    strengths : dict[str, float]
        Node name → recovered strength.
    confidences : dict[str, float]
        Node name → recovered confidence.
    n_iterations : int
        Number of message-passing iterations performed.
    converged : bool
        True if KL threshold was met before max_iterations.
    """
    marginals: dict
    strengths: dict
    confidences: dict
    n_iterations: int
    converged: bool


# ═══════════════════════════════════════════════════════════════════════════
#  Graph partitioning
# ═══════════════════════════════════════════════════════════════════════════

def _build_adjacency(priors, implications, similarities=None,
                     equivalences=None, negated_implications=None):
    """Build undirected adjacency and collect all edge metadata."""
    similarities = similarities or []
    equivalences = equivalences or []
    negated_implications = negated_implications or []

    names = set(priors.keys())
    edges = []  # (src, dst, link_dict, confidence)

    for link in implications:
        names.add(link["src"])
        names.add(link["dst"])
        edges.append((link["src"], link["dst"], link, link["confidence"]))

    for link in negated_implications:
        names.add(link["src"])
        names.add(link["dst"])
        edges.append((link["src"], link["dst"], link, link["confidence"]))

    for link in similarities + equivalences:
        names.add(link["src"])
        names.add(link["dst"])
        edges.append((link["src"], link["dst"], link, link["confidence"]))

    adjacency = {n: set() for n in names}
    for src, dst, _, _ in edges:
        adjacency[src].add(dst)
        adjacency[dst].add(src)

    return sorted(names), edges, adjacency


def _connected_components(names, adjacency):
    """Find connected components via BFS."""
    visited = set()
    components = []
    for name in names:
        if name in visited:
            continue
        component = []
        queue = deque([name])
        while queue:
            node = queue.popleft()
            if node in visited:
                continue
            visited.add(node)
            component.append(node)
            for nb in adjacency.get(node, set()):
                if nb not in visited:
                    queue.append(nb)
        components.append(sorted(component))
    return components


def _has_cycle_in_block_graph(blocks, cut_edges):
    """Detect if the block-level graph (blocks as nodes, cut_edges as edges) has a cycle."""
    if not cut_edges:
        return False

    # Build block-level adjacency
    node_to_block = {}
    for bi, block in enumerate(blocks):
        for name in block:
            node_to_block[name] = bi

    block_adj = defaultdict(set)
    for src, dst, _, _ in cut_edges:
        b_src = node_to_block.get(src)
        b_dst = node_to_block.get(dst)
        if b_src is not None and b_dst is not None and b_src != b_dst:
            block_adj[b_src].add(b_dst)
            block_adj[b_dst].add(b_src)

    # BFS cycle detection on undirected graph
    visited = set()
    for start in block_adj:
        if start in visited:
            continue
        queue = deque([(start, -1)])
        while queue:
            node, parent = queue.popleft()
            if node in visited:
                return True  # cycle found
            visited.add(node)
            for nb in block_adj[node]:
                if nb != parent:
                    queue.append((nb, node))

    return False


def partition_into_blocks(priors, implications, similarities=None,
                          equivalences=None, negated_implications=None,
                          max_block_size=4):
    """Partition an inference graph into blocks of at most max_block_size nodes.

    Algorithm: sort edges by confidence (ascending), iteratively remove
    the lowest-confidence edge until all connected components fit within
    max_block_size. Boundary nodes (incident to removed edges) appear in
    multiple blocks to receive inter-block messages.

    Parameters
    ----------
    priors : dict[str, {"strength": float, "confidence": float}]
    implications : list[{"src", "dst", "strength", "confidence"}]
    similarities, equivalences, negated_implications : optional link lists
    max_block_size : int
        Maximum number of propositions per block.

    Returns
    -------
    BlockPartition
    """
    names, edges, adjacency = _build_adjacency(
        priors, implications, similarities, equivalences, negated_implications)

    # If already small enough, return single block
    components = _connected_components(names, adjacency)
    if all(len(comp) <= max_block_size for comp in components):
        boundary = {}
        for name in names:
            block_indices = [i for i, comp in enumerate(components)
                            if name in comp]
            if len(block_indices) > 1:
                boundary[name] = block_indices
        return BlockPartition(
            blocks=components,
            boundary_nodes=boundary,
            cut_edges=[],
            has_cycle=False,
        )

    # Sort edges by confidence ascending — cut weakest first
    sorted_edges = sorted(edges, key=lambda e: e[3])
    cut_edges = []
    remaining_adj = {n: set(adj) for n, adj in adjacency.items()}

    for src, dst, link, conf in sorted_edges:
        # Check if all components are already small enough
        comps = _connected_components(names, remaining_adj)
        if all(len(comp) <= max_block_size for comp in comps):
            break

        # Remove this edge
        remaining_adj[src].discard(dst)
        remaining_adj[dst].discard(src)
        cut_edges.append((src, dst, link, conf))

    blocks = _connected_components(names, remaining_adj)

    # Merge pass: combine adjacent small blocks via cut edges (strongest first)
    merged = True
    while merged:
        merged = False
        for ce in sorted(cut_edges, key=lambda e: e[3], reverse=True):
            src, dst, link, conf = ce
            bi_src = next((i for i, bl in enumerate(blocks) if src in bl), None)
            bi_dst = next((i for i, bl in enumerate(blocks) if dst in bl), None)
            if bi_src is None or bi_dst is None or bi_src == bi_dst:
                continue
            if len(blocks[bi_src]) + len(blocks[bi_dst]) <= max_block_size:
                blocks[bi_src] = sorted(blocks[bi_src] + blocks[bi_dst])
                del blocks[bi_dst]
                cut_edges.remove(ce)
                merged = True
                break  # restart after merge

    # Identify boundary nodes: endpoints of cut edges
    node_to_blocks = defaultdict(list)
    for bi, block in enumerate(blocks):
        for name in block:
            node_to_blocks[name].append(bi)

    boundary_nodes = {}
    for src, dst, _, _ in cut_edges:
        # Both src and dst are boundary nodes — they need messages
        for name in [src, dst]:
            if name not in boundary_nodes:
                boundary_nodes[name] = node_to_blocks[name]

    has_cycle = _has_cycle_in_block_graph(blocks, cut_edges)

    return BlockPartition(
        blocks=blocks,
        boundary_nodes=boundary_nodes,
        cut_edges=cut_edges,
        has_cycle=has_cycle,
    )


# ═══════════════════════════════════════════════════════════════════════════
#  BP messages through implication factors
# ═══════════════════════════════════════════════════════════════════════════

def make_message_factor(node, log_weights):
    """Create a unary factor from inter-block BP message log-weights.

    Parameters
    ----------
    node : CategoricalNode
    log_weights : jnp.ndarray of shape [K]
        Log-message from BP through a cut edge's implication factor.
    """
    w = log_weights - jnp.mean(log_weights)
    return CategoricalEBMFactor([Block([node])], w[None, :])


def _compute_bp_message(source_marginal, impl_weights, direction="forward"):
    """Compute BP message through an implication factor.

    For a cut edge src→dst with weight table W[i,j] (parent×child bins):
    - Forward (src→dst): msg[j] = log(∑_i exp(W[i,j]) · marginal_src[i])
    - Backward (dst→src): msg[i] = log(∑_j exp(W[i,j]) · marginal_dst[j])

    Parameters
    ----------
    source_marginal : jnp.ndarray [K]
        Probability vector of the source node.
    impl_weights : jnp.ndarray [K, K]
        Implication weight table (parent_bins × child_bins).
    direction : "forward" or "backward"

    Returns
    -------
    jnp.ndarray [K] — log-message (centered).
    """
    # Convert weights to conditional probabilities (softmax per row)
    # W[i,j] are log-energies; exp(W[i,j]) ∝ P(child=j | parent=i)
    if direction == "forward":
        # msg[j] = ∑_i P(j|i) * marginal[i]
        # P(j|i) ∝ exp(W[i,j]), normalize per row i
        log_cond = impl_weights - jax.scipy.special.logsumexp(
            impl_weights, axis=1, keepdims=True)
        cond = jnp.exp(log_cond)  # [K, K], rows sum to 1
        msg_prob = jnp.dot(source_marginal, cond)  # [K]
    else:
        # msg[i] = ∑_j P(j|i) * marginal_dst[j]
        log_cond = impl_weights - jax.scipy.special.logsumexp(
            impl_weights, axis=1, keepdims=True)
        cond = jnp.exp(log_cond)  # [K, K]
        msg_prob = jnp.dot(cond, source_marginal)  # [K]

    msg_prob = jnp.clip(msg_prob, EPS, None)
    msg_log = jnp.log(msg_prob)
    return msg_log - jnp.mean(msg_log)


# ═══════════════════════════════════════════════════════════════════════════
#  Sub-graph construction
# ═══════════════════════════════════════════════════════════════════════════

def _build_block_subgraph(block_names, priors, implications,
                          similarities=None, equivalences=None,
                          negated_implications=None, backgrounds=None,
                          messages=None, k=4):
    """Build a thrml factor graph for one block, including message factors.

    Parameters
    ----------
    block_names : list[str]
        Node names in this block.
    priors : dict[str, {"strength": float, "confidence": float}]
    implications, similarities, equivalences, negated_implications : link lists
        Only links with both endpoints in block_names are included.
    backgrounds : dict or None
    messages : dict[str, jnp.ndarray] or None
        node_name → log-weights [K] from neighboring blocks.
    k : int
        Number of bins.

    Returns
    -------
    graph : dict
        Compatible with run_beta_sampling.
    name_to_node : dict[str, CategoricalNode]
        Mapping for extracting marginals later.
    """
    similarities = similarities or []
    equivalences = equivalences or []
    negated_implications = negated_implications or []
    backgrounds = backgrounds or {}
    messages = messages or {}

    block_set = set(block_names)
    name_to_node = {name: CategoricalNode() for name in sorted(block_names)}

    factors = []

    # Priors
    for name in sorted(block_names):
        p = priors.get(name)
        s = p["strength"] if p else 0.5
        c = p["confidence"] if p else 0.01
        factors.append(make_beta_prior_factor(name_to_node[name], s, c, k))

    # Directed implications (both endpoints must be in block)
    for link in implications:
        if link["src"] in block_set and link["dst"] in block_set:
            bg = backgrounds.get((link["src"], link["dst"]), DEFAULT_EPSILON)
            factors.append(make_beta_implication_factor(
                name_to_node[link["src"]], name_to_node[link["dst"]],
                link["strength"], link["confidence"], bg, k))

    # Negated implications
    for link in negated_implications:
        if link["src"] in block_set and link["dst"] in block_set:
            bg = backgrounds.get((link["src"], link["dst"]), DEFAULT_EPSILON)
            factors.append(make_beta_implication_factor(
                name_to_node[link["src"]], name_to_node[link["dst"]],
                1.0 - link["strength"], link["confidence"], bg, k))

    # Symmetric links → bidirectional
    for link in similarities + equivalences:
        if link["src"] in block_set and link["dst"] in block_set:
            bg = backgrounds.get((link["src"], link["dst"]), DEFAULT_EPSILON)
            factors.append(make_beta_implication_factor(
                name_to_node[link["src"]], name_to_node[link["dst"]],
                link["strength"], link["confidence"], bg, k))
            factors.append(make_beta_implication_factor(
                name_to_node[link["dst"]], name_to_node[link["src"]],
                link["strength"], link["confidence"], bg, k))

    # Message factors from neighboring blocks
    for name, log_w in messages.items():
        if name in name_to_node:
            factors.append(make_message_factor(name_to_node[name], log_w))

    # Assemble free graph (all nodes free, no clamping)
    nodes_list = [name_to_node[n] for n in sorted(block_names)]
    free_blocks = [Block([node]) for node in nodes_list]
    graph = _assemble_free_graph(nodes_list, factors, free_blocks, k=k)

    return graph, name_to_node


# ═══════════════════════════════════════════════════════════════════════════
#  KL divergence for convergence monitoring
# ═══════════════════════════════════════════════════════════════════════════

def _kl_divergence(p, q):
    """KL(p || q) for discrete distributions. Returns scalar."""
    p_safe = jnp.clip(p, EPS, 1.0)
    q_safe = jnp.clip(q, EPS, 1.0)
    return float(jnp.sum(p_safe * jnp.log(p_safe / q_safe)))


# ═══════════════════════════════════════════════════════════════════════════
#  Block-diagonal sampling engine
# ═══════════════════════════════════════════════════════════════════════════

def run_block_diagonal_sampling(priors, implications, similarities=None,
                                equivalences=None, negated_implications=None,
                                backgrounds=None, k=4, max_block_size=4,
                                max_iterations=20, kl_threshold=0.01,
                                damping=0.5, seed=42,
                                n_batches=None, schedule=None):
    """Block-diagonal inference with iterative message passing.

    Algorithm
    ---------
    1. Partition graph into blocks of ≤ max_block_size propositions.
    2. Initialize inter-block messages as uniform log(1/K).
    3. Iterate:
       a. For each block, build sub-graph with current messages as prior factors.
       b. Run Gibbs sampling within each block.
       c. Extract boundary node marginals → update messages.
       d. Check convergence: max KL(new, old) < kl_threshold.
    4. Extract final marginals for all nodes.

    For cyclic block graphs, messages are damped:
        new_msg = damping * new + (1 - damping) * old

    Parameters
    ----------
    priors : dict[str, {"strength": float, "confidence": float}]
    implications : list[{"src", "dst", "strength", "confidence"}]
    similarities, equivalences, negated_implications : optional link lists
    backgrounds : dict or None
    k : int
        Bins per node (default 4 for TSU deployment).
    max_block_size : int
        Maximum propositions per block.
    max_iterations : int
        Maximum message-passing rounds.
    kl_threshold : float
        Convergence criterion on message KL divergence.
    damping : float
        Message damping factor for cyclic graphs (0-1).
    seed : int
    n_batches, schedule : sampling parameters (defaults from pln_thrml.beta)

    Returns
    -------
    BlockDiagonalResult
    """
    backgrounds = backgrounds or {}

    partition = partition_into_blocks(
        priors, implications, similarities, equivalences,
        negated_implications, max_block_size)

    node_to_block = {}
    for bi, block in enumerate(partition.blocks):
        for name in block:
            node_to_block[name] = bi

    # Precompute implication weight tables for cut edges and build message routes.
    # Each cut edge has a directed implication src→dst. We compute:
    #   - forward message: src marginal → through W[i,j] → message on dst
    #   - backward message: dst marginal → through W[i,j]^T → message on src
    # route = (source_name, source_block, target_name, target_block,
    #          impl_weights, direction)
    message_routes = []
    for src, dst, link, conf in partition.cut_edges:
        b_src = node_to_block[src]
        b_dst = node_to_block[dst]
        bg = backgrounds.get((src, dst), DEFAULT_EPSILON)

        # Check if this is a negated implication
        is_negated = link in (negated_implications or [])
        s = 1.0 - link["strength"] if is_negated else link["strength"]
        w_table = beta_implication_weights(s, link["confidence"], bg, k)

        # Forward: src's marginal → BP message → dst
        message_routes.append((src, b_src, dst, b_dst, w_table, "forward"))
        # Backward: dst's marginal → BP message → src
        message_routes.append((dst, b_dst, src, b_src, w_table, "backward"))

    # Initialize messages as uniform (centered log)
    uniform_msg = jnp.zeros(k)  # log(1/K) centered = 0
    # messages[(target_name, target_block)] = log_weights [K]
    messages = {}
    for _, _, target, target_block, _, _ in message_routes:
        messages[(target, target_block)] = uniform_msg

    use_damping = partition.has_cycle

    converged = False
    n_iter = 0

    # Store per-block marginals across iterations
    block_marginals = {}  # (node_name, block_idx) → posterior [K]

    for iteration in range(max_iterations):
        n_iter = iteration + 1
        max_kl = 0.0

        for bi, block_names in enumerate(partition.blocks):
            # Collect incoming messages for this block
            block_msgs = {}
            for target, target_block in messages:
                if target_block == bi and target in block_names:
                    block_msgs[target] = messages[(target, target_block)]

            # Build and sample sub-graph
            key = seed + iteration * len(partition.blocks) + bi
            graph, name_to_node = _build_block_subgraph(
                block_names, priors, implications,
                similarities, equivalences, negated_implications,
                backgrounds, block_msgs, k)

            samples = run_beta_sampling(graph, seed=key,
                                       n_batches=n_batches,
                                       schedule=schedule)

            # Extract marginals for all nodes in this block
            for name in block_names:
                node = name_to_node[name]
                posterior, _, _ = estimate_beta_marginal(samples, graph, node)
                block_marginals[(name, bi)] = posterior

        # Update messages: accumulate all incoming messages per target node,
        # then apply damping. Multiple cut edges pointing to the same target
        # (e.g. B→D and C→D in a diamond) must be summed in log-space
        # (= multiplied in probability space), not overwritten.
        new_log_ws: dict = {}
        for source, source_block, target, target_block, w_table, direction in message_routes:
            source_marginal = block_marginals.get((source, source_block))
            if source_marginal is None:
                continue
            msg = _compute_bp_message(source_marginal, w_table, direction)
            key = (target, target_block)
            new_log_ws[key] = new_log_ws[key] + msg if key in new_log_ws else msg

        for key, new_log_w in new_log_ws.items():
            old_log_w = messages[key]
            if use_damping:
                new_log_w = damping * new_log_w + (1 - damping) * old_log_w
            # Compute KL for convergence check
            old_prob = jnp.exp(old_log_w - jnp.max(old_log_w))
            old_prob = old_prob / jnp.sum(old_prob)
            new_prob = jnp.exp(new_log_w - jnp.max(new_log_w))
            new_prob = new_prob / jnp.sum(new_prob)
            max_kl = max(max_kl, _kl_divergence(new_prob, old_prob))
            messages[key] = new_log_w

        if max_kl < kl_threshold:
            converged = True
            break

    # Final pass: extract marginals for all nodes
    # Use the last iteration's block marginals
    result_marginals = {}
    result_strengths = {}
    result_confidences = {}

    for name in priors:
        bi = node_to_block[name]
        posterior = block_marginals.get((name, bi))
        if posterior is not None:
            result_marginals[name] = posterior
            s, c = posterior_to_stv(posterior, k)
            result_strengths[name] = s
            result_confidences[name] = c

    return BlockDiagonalResult(
        marginals=result_marginals,
        strengths=result_strengths,
        confidences=result_confidences,
        n_iterations=n_iter,
        converged=converged,
    )


def sample_and_measure_block_diagonal(priors, implications, target_name,
                                      similarities=None, equivalences=None,
                                      negated_implications=None,
                                      backgrounds=None, k=4,
                                      max_block_size=4, seed=42,
                                      **kwargs):
    """High-level API: partition → message passing → return (strength, confidence).

    Parameters
    ----------
    target_name : str
        Name of the target proposition to measure.
    **kwargs : passed to run_block_diagonal_sampling

    Returns
    -------
    (strength, confidence) : tuple[float, float]
    """
    result = run_block_diagonal_sampling(
        priors, implications, similarities, equivalences,
        negated_implications, backgrounds, k=k,
        max_block_size=max_block_size, seed=seed, **kwargs)

    if target_name not in result.strengths:
        raise ValueError(f"Target '{target_name}' not found in results. "
                        f"Available: {list(result.strengths.keys())}")

    return result.strengths[target_name], result.confidences[target_name]
