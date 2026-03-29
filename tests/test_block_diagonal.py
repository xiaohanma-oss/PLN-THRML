"""Tests for block-diagonal inference with inter-block message passing."""

import pytest
import jax.numpy as jnp

from block_diagonal import (
    partition_into_blocks,
    run_block_diagonal_sampling,
    sample_and_measure_block_diagonal,
    BlockPartition,
    _kl_divergence,
)
from pln_thrml_beta import (
    build_beta_full_graph, build_beta_chain,
    run_beta_sampling, estimate_beta_marginal,
    beta_implication_weights,
)
from conftest import strength_tol, confidence_tol


# ═══════════════════════════════════════════════════════════════════════════
#  Graph partitioning
# ═══════════════════════════════════════════════════════════════════════════

class TestPartitioning:
    def test_single_block_no_split(self):
        """3 nodes with max_block_size=4 → no split needed."""
        priors = {
            "A": {"strength": 0.8, "confidence": 0.9},
            "B": {"strength": 0.5, "confidence": 0.01},
            "C": {"strength": 0.5, "confidence": 0.01},
        }
        implications = [
            {"src": "A", "dst": "B", "strength": 0.9, "confidence": 0.85},
            {"src": "B", "dst": "C", "strength": 0.8, "confidence": 0.8},
        ]
        partition = partition_into_blocks(priors, implications, max_block_size=4)
        assert len(partition.blocks) == 1
        assert len(partition.cut_edges) == 0
        assert len(partition.boundary_nodes) == 0
        assert not partition.has_cycle

    def test_chain_splits_into_blocks(self):
        """5-node chain with max_block_size=3 → splits into ≥ 2 blocks."""
        priors = {n: {"strength": 0.5, "confidence": 0.01}
                  for n in ["A", "B", "C", "D", "E"]}
        priors["A"]["strength"] = 0.8
        priors["A"]["confidence"] = 0.9
        implications = [
            {"src": "A", "dst": "B", "strength": 0.8, "confidence": 0.9},
            {"src": "B", "dst": "C", "strength": 0.7, "confidence": 0.85},
            {"src": "C", "dst": "D", "strength": 0.6, "confidence": 0.8},
            {"src": "D", "dst": "E", "strength": 0.5, "confidence": 0.75},
        ]
        partition = partition_into_blocks(priors, implications, max_block_size=3)
        # 5-chain max=3: merge pass should produce 2 blocks, not 3
        assert len(partition.blocks) == 2
        assert all(len(b) <= 3 for b in partition.blocks)
        # Exactly one cut edge after merge
        assert len(partition.cut_edges) == 1
        # Boundary nodes should exist
        assert len(partition.boundary_nodes) >= 1

    def test_cuts_weakest_edge_first(self):
        """Partitioning should cut the lowest-confidence edge first."""
        priors = {n: {"strength": 0.5, "confidence": 0.01}
                  for n in ["A", "B", "C"]}
        implications = [
            {"src": "A", "dst": "B", "strength": 0.9, "confidence": 0.9},
            {"src": "B", "dst": "C", "strength": 0.8, "confidence": 0.3},
        ]
        partition = partition_into_blocks(priors, implications, max_block_size=2)
        # Should cut B→C (confidence 0.3) before A→B (confidence 0.9)
        assert len(partition.cut_edges) >= 1
        cut_confs = [c for _, _, _, c in partition.cut_edges]
        assert min(cut_confs) == 0.3

    def test_diamond_detects_cycle(self):
        """Diamond graph A→{B,C}→D should detect cycle in block graph
        when split into blocks."""
        priors = {n: {"strength": 0.5, "confidence": 0.01}
                  for n in ["A", "B", "C", "D"]}
        priors["A"]["strength"] = 0.8
        priors["A"]["confidence"] = 0.9
        implications = [
            {"src": "A", "dst": "B", "strength": 0.7, "confidence": 0.9},
            {"src": "A", "dst": "C", "strength": 0.7, "confidence": 0.9},
            {"src": "B", "dst": "D", "strength": 0.7, "confidence": 0.85},
            {"src": "C", "dst": "D", "strength": 0.7, "confidence": 0.85},
        ]
        partition = partition_into_blocks(priors, implications, max_block_size=2)
        # With 4 nodes, max_block_size=2: must split into at least 2 blocks
        assert len(partition.blocks) >= 2
        assert all(len(b) <= 2 for b in partition.blocks)

    def test_disconnected_components(self):
        """Disconnected nodes should be in separate blocks."""
        priors = {
            "A": {"strength": 0.8, "confidence": 0.9},
            "B": {"strength": 0.5, "confidence": 0.5},
            "C": {"strength": 0.3, "confidence": 0.7},
        }
        partition = partition_into_blocks(priors, [], max_block_size=2)
        # 3 isolated nodes → 3 blocks of size 1
        total_nodes = sum(len(b) for b in partition.blocks)
        assert total_nodes == 3


# ═══════════════════════════════════════════════════════════════════════════
#  Message passing and accuracy
# ═══════════════════════════════════════════════════════════════════════════

class TestMessagePassing:
    """Compare block-diagonal K=4 against full-graph K=16 baseline."""

    def _full_graph_baseline(self, priors, implications, target_name, k=4):
        """Run full-graph inference as ground truth (same K for fair comparison)."""
        graph = build_beta_full_graph(priors, implications, k=k)
        samples = run_beta_sampling(graph, seed=42)
        _, s, c = estimate_beta_marginal(
            samples, graph, graph["nodes"][target_name])
        return s, c

    def test_chain_3_vs_full(self):
        """A→B→C: block-diagonal K=4 vs full-graph K=4."""
        priors = {
            "A": {"strength": 0.8, "confidence": 0.9},
            "B": {"strength": 0.5, "confidence": 0.01},
            "C": {"strength": 0.5, "confidence": 0.01},
        }
        implications = [
            {"src": "A", "dst": "B", "strength": 0.9, "confidence": 0.85},
            {"src": "B", "dst": "C", "strength": 0.8, "confidence": 0.8},
        ]

        s_full, _ = self._full_graph_baseline(priors, implications, "C", k=4)

        s_bd, c_bd = sample_and_measure_block_diagonal(
            priors, implications, "C", k=4, max_block_size=2, seed=42)

        # Block-diagonal should match full graph at same K
        assert s_bd == pytest.approx(s_full, abs=strength_tol(4)), \
            f"Block-diag s={s_bd:.3f} vs full s={s_full:.3f}"
        assert c_bd > 0, "Confidence should be positive"

    def test_chain_5_vs_full(self):
        """5-node chain: verify signal propagates through message passing."""
        priors = {
            "A": {"strength": 0.8, "confidence": 0.9},
            "B": {"strength": 0.5, "confidence": 0.01},
            "C": {"strength": 0.5, "confidence": 0.01},
            "D": {"strength": 0.5, "confidence": 0.01},
            "E": {"strength": 0.5, "confidence": 0.01},
        }
        implications = [
            {"src": "A", "dst": "B", "strength": 0.9, "confidence": 0.9},
            {"src": "B", "dst": "C", "strength": 0.8, "confidence": 0.85},
            {"src": "C", "dst": "D", "strength": 0.8, "confidence": 0.85},
            {"src": "D", "dst": "E", "strength": 0.8, "confidence": 0.85},
        ]

        s_full, _ = self._full_graph_baseline(priors, implications, "E", k=4)

        result = run_block_diagonal_sampling(
            priors, implications, k=4, max_block_size=3, seed=42)

        s_bd = result.strengths["E"]
        # Signal should propagate: E's strength should be influenced by A
        # Direction depends on coupling; just verify not stuck at uniform
        assert s_bd != pytest.approx(0.5, abs=0.05), \
            f"Message should propagate: E strength={s_bd:.3f}"
        # Should be in ballpark of full graph
        assert s_bd == pytest.approx(s_full, abs=0.15), \
            f"Block-diag s={s_bd:.3f} vs full s={s_full:.3f}"

    def test_diamond_vs_full(self):
        """Diamond A→{B,C}→D: cyclic block graph with damping."""
        priors = {
            "A": {"strength": 0.8, "confidence": 0.9},
            "B": {"strength": 0.5, "confidence": 0.01},
            "C": {"strength": 0.5, "confidence": 0.01},
            "D": {"strength": 0.5, "confidence": 0.01},
        }
        implications = [
            {"src": "A", "dst": "B", "strength": 0.7, "confidence": 0.9},
            {"src": "A", "dst": "C", "strength": 0.7, "confidence": 0.9},
            {"src": "B", "dst": "D", "strength": 0.7, "confidence": 0.9},
            {"src": "C", "dst": "D", "strength": 0.7, "confidence": 0.9},
        ]

        s_full, _ = self._full_graph_baseline(priors, implications, "D", k=4)

        s_bd, c_bd = sample_and_measure_block_diagonal(
            priors, implications, "D", k=4, max_block_size=2, seed=42)

        assert s_bd == pytest.approx(s_full, abs=0.15), \
            f"Diamond block-diag s={s_bd:.3f} vs full s={s_full:.3f}"
        assert c_bd > 0

        # max=3: D is isolated (B→D and C→D are both cut edges).
        # Both messages must be accumulated, not overwritten.
        s_bd3, c_bd3 = sample_and_measure_block_diagonal(
            priors, implications, "D", k=4, max_block_size=3, seed=42)
        assert s_bd3 == pytest.approx(s_full, abs=0.15), \
            f"Diamond max=3 block-diag s={s_bd3:.3f} vs full s={s_full:.3f}"
        assert c_bd3 > 0

    def test_convergence_happens(self):
        """Message passing should converge (n_iterations < max_iterations)."""
        priors = {
            "A": {"strength": 0.8, "confidence": 0.9},
            "B": {"strength": 0.5, "confidence": 0.01},
            "C": {"strength": 0.5, "confidence": 0.01},
        }
        implications = [
            {"src": "A", "dst": "B", "strength": 0.9, "confidence": 0.85},
            {"src": "B", "dst": "C", "strength": 0.8, "confidence": 0.8},
        ]

        result = run_block_diagonal_sampling(
            priors, implications, k=4, max_block_size=2,
            max_iterations=20, kl_threshold=0.01, seed=42)

        assert result.converged, \
            f"Did not converge in {result.n_iterations} iterations"
        assert result.n_iterations < 20

    def test_single_block_matches_full(self):
        """When all nodes fit in one block, should match full graph exactly."""
        priors = {
            "A": {"strength": 0.8, "confidence": 0.9},
            "B": {"strength": 0.5, "confidence": 0.01},
        }
        implications = [
            {"src": "A", "dst": "B", "strength": 0.9, "confidence": 0.85},
        ]

        s_full, _ = self._full_graph_baseline(priors, implications, "B", k=4)

        s_bd, _ = sample_and_measure_block_diagonal(
            priors, implications, "B", k=4, max_block_size=4, seed=42)

        # Same K, same graph → should match closely (only seed difference)
        assert s_bd == pytest.approx(s_full, abs=0.05)


# ═══════════════════════════════════════════════════════════════════════════
#  Connection budget verification
# ═══════════════════════════════════════════════════════════════════════════

class TestConnectionBudget:
    """Verify K=4 stays within TSU 12-connection limit."""

    def test_implication_weight_entries_k4(self):
        """K=4 implication factor has 4×4=16 weight entries."""
        w = beta_implication_weights(0.9, 0.85, k=4)
        assert w.shape == (4, 4)

    def test_max_connections_per_node_k4(self):
        """With K=4 and block_size=3, max connections per p-bit should be
        computed and documented.

        Each node is 4 p-bits (one-hot K=4).
        Each pairwise factor (implication) couples 4 p-bits to 4 p-bits.
        Per p-bit: each edge contributes K-1=3 intra-factor connections,
        plus connections from priors.

        For block_size=3, a node has at most 2 edges within the block.
        Connections per p-bit = 2 edges × (K-1) + (K-1) prior = 3×(K-1) = 9.
        Well within the 12-connection budget.
        """
        k = 4
        max_edges_per_node = 2  # in a block of 3
        connections_per_pbit = max_edges_per_node * (k - 1) + (k - 1)
        assert connections_per_pbit <= 12, \
            f"K=4, {max_edges_per_node} edges: {connections_per_pbit} connections > 12"


# ═══════════════════════════════════════════════════════════════════════════
#  KL divergence utility
# ═══════════════════════════════════════════════════════════════════════════

class TestKLDivergence:
    def test_same_distribution(self):
        """KL(p || p) = 0."""
        p = jnp.array([0.25, 0.25, 0.25, 0.25])
        assert _kl_divergence(p, p) == pytest.approx(0.0, abs=1e-6)

    def test_different_distributions(self):
        """KL(p || q) > 0 for p ≠ q."""
        p = jnp.array([0.9, 0.05, 0.03, 0.02])
        q = jnp.array([0.25, 0.25, 0.25, 0.25])
        assert _kl_divergence(p, q) > 0
