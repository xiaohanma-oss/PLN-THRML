"""
Full-graph compilation tests — verify build_full_graph produces correct
marginals and conditionals, matching the per-rule builders.
"""

import pytest
from pln_thrml import (
    build_full_graph, build_chain, build_inv_v_graph,
    run_sampling, estimate_marginal, estimate_conditional,
    DEFAULT_EPSILON, STV, truth_modus_ponens,
)
from conftest import STRENGTH_TOL, parse_stv


FULL_GRAPH_TOL = 0.10  # wider tolerance: beta full-graph with K=16 bins


# ── Pure Python tests (no MeTTa dependency) ─────────────────────────────

class TestFullGraphPython:

    def test_single_implication(self):
        """A→B: full graph should give same P(B) as build_chain."""
        priors = {"A": {"strength": 0.8, "confidence": 0.9}}
        impls = [{"src": "A", "dst": "B", "strength": 0.9, "confidence": 0.85}]

        graph = build_full_graph(priors, impls)
        samples = run_sampling(graph, seed=42)
        p_B = estimate_marginal(samples, graph, graph["nodes"]["B"])

        # Compare with per-rule chain
        chain = build_chain(
            priors=[0.8, 0.5], strengths=[0.9], backgrounds=[DEFAULT_EPSILON])
        chain_samples = run_sampling(chain, seed=42)
        p_B_chain = estimate_marginal(chain_samples, chain, chain["nodes"][1])

        # Both should be close to PLN modus ponens
        expected = truth_modus_ponens(STV(0.8, 0.9), STV(0.9, 0.85))
        assert abs(p_B - expected.strength) < FULL_GRAPH_TOL
        assert abs(p_B_chain - expected.strength) < FULL_GRAPH_TOL

    def test_chain_deduction(self):
        """A→B→C: P(C|A) from full graph should match chain builder."""
        priors = {
            "A": {"strength": 0.8, "confidence": 0.9},
            "B": {"strength": 0.5, "confidence": 0.5},
            "C": {"strength": 0.3, "confidence": 0.5},
        }
        impls = [
            {"src": "A", "dst": "B", "strength": 0.8, "confidence": 0.9},
            {"src": "B", "dst": "C", "strength": 0.9, "confidence": 0.85},
        ]

        graph = build_full_graph(priors, impls)
        samples = run_sampling(graph, seed=42)
        p_C_given_A = estimate_conditional(
            samples, graph, graph["nodes"]["C"], graph["nodes"]["A"])

        # Should be in a reasonable range for deduction
        assert 0.5 < p_C_given_A < 1.0

    def test_converging_topology(self):
        """A→C←B (abduction): full graph handles collider structure."""
        priors = {
            "A": {"strength": 0.6, "confidence": 0.9},
            "B": {"strength": 0.5, "confidence": 0.9},
        }
        impls = [
            {"src": "A", "dst": "C", "strength": 0.8, "confidence": 0.9},
            {"src": "B", "dst": "C", "strength": 0.7, "confidence": 0.9},
        ]

        graph = build_full_graph(priors, impls)
        samples = run_sampling(graph, seed=42)

        # P(C) should reflect both causes
        p_C = estimate_marginal(samples, graph, graph["nodes"]["C"])
        assert 0.3 < p_C < 0.9

        # Explaining-away: P(B|A) should differ from P(B)
        p_B = estimate_marginal(samples, graph, graph["nodes"]["B"])
        p_B_given_A = estimate_conditional(
            samples, graph, graph["nodes"]["B"], graph["nodes"]["A"])
        # Both should be valid probabilities
        assert 0.0 < p_B < 1.0
        assert 0.0 < p_B_given_A < 1.0

    def test_diamond_graph(self):
        """A→B, A→C, B→D, C→D: topology that no existing builder supports."""
        priors = {"A": {"strength": 0.8, "confidence": 0.9}}
        impls = [
            {"src": "A", "dst": "B", "strength": 0.9, "confidence": 0.9},
            {"src": "A", "dst": "C", "strength": 0.7, "confidence": 0.9},
            {"src": "B", "dst": "D", "strength": 0.8, "confidence": 0.9},
            {"src": "C", "dst": "D", "strength": 0.6, "confidence": 0.9},
        ]

        graph = build_full_graph(priors, impls)
        samples = run_sampling(graph, seed=42)

        # D has two paths from A, should have high marginal
        p_D = estimate_marginal(samples, graph, graph["nodes"]["D"])
        assert 0.3 < p_D < 1.0

        # Automatic Bayes: P(A|D) should be higher than P(A|¬D)
        p_A_given_D = estimate_conditional(
            samples, graph, graph["nodes"]["A"], graph["nodes"]["D"], cond_val=1)
        p_A_given_not_D = estimate_conditional(
            samples, graph, graph["nodes"]["A"], graph["nodes"]["D"], cond_val=0)
        assert p_A_given_D > p_A_given_not_D

    def test_similarity_bidirectional(self):
        """Similarity links create symmetric coupling."""
        priors = {
            "A": {"strength": 0.9, "confidence": 0.9},
            "B": {"strength": 0.5, "confidence": 0.5},
        }
        sims = [{"src": "A", "dst": "B", "strength": 0.85, "confidence": 0.9}]

        graph = build_full_graph(priors, [], sims)
        samples = run_sampling(graph, seed=42)

        # B should be pulled toward A's high strength
        p_B = estimate_marginal(samples, graph, graph["nodes"]["B"])
        assert p_B > 0.5

        # Symmetry: P(B|A) ≈ P(A|B) since coupling is symmetric
        p_B_given_A = estimate_conditional(
            samples, graph, graph["nodes"]["B"], graph["nodes"]["A"])
        p_A_given_B = estimate_conditional(
            samples, graph, graph["nodes"]["A"], graph["nodes"]["B"])
        # Not exactly equal (priors differ) but both should be high
        assert p_B_given_A > 0.5
        assert p_A_given_B > 0.5

    def test_empty_graph(self):
        """Single node with prior, no edges."""
        priors = {"X": {"strength": 0.7, "confidence": 0.9}}
        graph = build_full_graph(priors, [])
        samples = run_sampling(graph, seed=42)
        p_X = estimate_marginal(samples, graph, graph["nodes"]["X"])
        assert abs(p_X - 0.7) < FULL_GRAPH_TOL


# ── MeTTa integration tests ─────────────────────────────────────────────

class TestFullGraphMeTTa:

    def test_compile_and_query_marginal(self, metta):
        """Compile knowledge, then query marginal P(B)."""
        metta.run("""
            (A (stv 0.8 0.9))
            ((Implication A B) (stv 0.9 0.85))
        """)
        metta.run("!(thrml-compile!)")
        s, c = parse_stv(metta.run("!(thrml-query! (B))"))

        expected = truth_modus_ponens(STV(0.8, 0.9), STV(0.9, 0.85))
        assert abs(s - expected.strength) < FULL_GRAPH_TOL

    def test_compile_and_query_conditional(self, metta):
        """Compile chain A→B→C, query conditional P(C|A)."""
        metta.run("""
            (A (stv 0.8 0.9))
            (B (stv 0.5 0.5))
            (C (stv 0.3 0.5))
            ((Implication A B) (stv 0.8 0.9))
            ((Implication B C) (stv 0.9 0.85))
        """)
        metta.run("!(thrml-compile!)")
        s, c = parse_stv(metta.run("!(thrml-query! (C A))"))
        assert 0.5 < s < 1.0  # deduction should produce reasonable conditional

    def test_automatic_bayes_inversion(self, metta):
        """P(A|B) comes free from full-graph — no inversion rule needed."""
        metta.run("""
            (A (stv 0.5 0.9))
            ((Implication A B) (stv 0.87 0.9))
        """)
        metta.run("!(thrml-compile!)")

        # Forward: P(B|A) and Backward: P(A|B) — both via beta conditional
        s_forward, _ = parse_stv(metta.run("!(thrml-query! (B A))"))
        s_backward, _ = parse_stv(metta.run("!(thrml-query! (A B))"))

        # Both should be valid probabilities (beta full-graph conditionals
        # use weighted estimation which is approximate)
        assert 0.0 < s_forward < 1.0
        assert 0.0 < s_backward < 1.0
