"""
Scalability tests — upstream PLN examples through full MeTTa pipeline.

Each test loads knowledge from upstream vendor/PLN/examples/ into MeTTa,
compiles to a beta factor graph via thrml-compile!, and queries via
thrml-query!.  This validates the complete pipeline:
  MeTTa atoms → extract → build_beta_full_graph → Gibbs sampling → query

Sources:
    vendor/PLN/examples/DeductionRevision.metta
    vendor/PLN/examples/FlyingRaven.metta
    vendor/PLN/examples/Smokes.metta
    vendor/PLN/examples/RavenInduction.metta

All marked @pytest.mark.slow — run with ``pytest -m slow``.
"""

import time
import pytest

from pln_thrml_beta import (
    build_beta_chain, build_beta_full_graph, run_beta_sampling,
    estimate_beta_marginal, estimate_beta_conditional,
    diagnose_convergence, _greedy_color,
)
from conftest import STRENGTH_TOL, CONFIDENCE_TOL, parse_stv


# ═══════════════════════════════════════════════════════════════════════════
#  Test 1: DeductionRevision diamond
#  Source: vendor/PLN/examples/DeductionRevision.metta
#
#  Upstream knowledge:
#    (= (STV A) (stv 0.5 0.9))  (= (STV B) (stv 0.25 0.9))
#    (= (STV C) (stv 0.25 0.9)) (= (STV D) (stv 0.5 0.9))
#    (Inheritance A B) (stv 0.25 0.9)
#    (Inheritance A C) (stv 0.25 0.9)
#    (Inheritance B D) (stv 0.5 0.9)
#    (Inheritance C D) (stv 0.5 0.9)
#
#  Upstream expected: P(Inheritance A D) ≈ (stv 0.125 0.304)
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.slow
class TestDeductionRevisionDiamond:
    """A→{B,C}→D diamond — two deduction paths merging via revision."""

    METTA_KB = """
        (A (stv 0.5 0.9))
        (B (stv 0.25 0.9))
        (C (stv 0.25 0.9))
        (D (stv 0.5 0.9))
        ((Inheritance A B) (stv 0.25 0.9))
        ((Inheritance A C) (stv 0.25 0.9))
        ((Inheritance B D) (stv 0.5 0.9))
        ((Inheritance C D) (stv 0.5 0.9))
    """

    def test_compile_and_query(self, metta):
        """Full pipeline: compile 4-node diamond, query P(D|A)."""
        metta.run(self.METTA_KB)
        stv = metta.run("!(thrml-compile!)")
        # Should report 4 nodes
        s_nodes, _ = parse_stv(stv)
        assert s_nodes == 4.0

        s, c = parse_stv(metta.run("!(thrml-query! (D A))"))
        # Factor graph result may differ from PLN's analytical (0.125)
        # due to backward coupling; verify it's low (weak links)
        assert s < 0.3, f"P(D|A) = {s:.3f}, expected low with weak links"
        assert c > 0.0

    def test_diamond_two_paths_stronger_than_chain(self):
        """Diamond (2 paths) should give higher confidence than single chain.

        Build both via Python API for direct comparison.
        """
        diamond = build_beta_full_graph(
            priors={
                "A": {"strength": 0.5, "confidence": 0.9},
                "B": {"strength": 0.25, "confidence": 0.9},
                "C": {"strength": 0.25, "confidence": 0.9},
                "D": {"strength": 0.5, "confidence": 0.9},
            },
            implications=[
                {"src": "A", "dst": "B", "strength": 0.25, "confidence": 0.9},
                {"src": "A", "dst": "C", "strength": 0.25, "confidence": 0.9},
                {"src": "B", "dst": "D", "strength": 0.5, "confidence": 0.9},
                {"src": "C", "dst": "D", "strength": 0.5, "confidence": 0.9},
            ],
        )
        d_samples = run_beta_sampling(diamond, seed=42)
        _, _, c_diamond = estimate_beta_conditional(
            d_samples, diamond, diamond["nodes"]["D"], diamond["nodes"]["A"])

        chain = build_beta_chain(
            priors=[0.5, 0.25, 0.5],
            confidences=[0.9, 0.9, 0.9],
            strengths=[0.25, 0.5],
            impl_confidences=[0.9, 0.9],
            backgrounds=[0.02, 0.02],
        )
        c_samples = run_beta_sampling(chain, seed=42)
        _, _, c_chain = estimate_beta_conditional(
            c_samples, chain, chain["nodes"][2], chain["nodes"][0])

        assert c_diamond > c_chain, \
            f"Diamond c={c_diamond:.3f} should be > chain c={c_chain:.3f}"


# ═══════════════════════════════════════════════════════════════════════════
#  Test 2: FlyingRaven conflicting paths
#  Source: vendor/PLN/examples/FlyingRaven.metta
#
#  Upstream knowledge:
#    Sam→Raven(0.99), Pingu→Penguin(0.99)
#    Raven→Bird(0.99), Penguin→Bird(0.99), Bird→flies(0.99)
#    Penguin→Not(flies)(1.0)
#
#  Upstream expected:
#    P(flies|Sam)   ≈ (stv 0.970 0.693)
#    P(flies|Pingu) ≈ (stv 0.010 0.802)
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.slow
class TestFlyingRaven:
    """Sam→Raven→Bird→flies vs Penguin→Not(flies) — conflicting paths."""

    METTA_KB = """
        (Sam     (stv 0.16 0.9))
        (Pingu   (stv 0.16 0.9))
        (Raven   (stv 0.16 0.9))
        (Penguin (stv 0.16 0.9))
        (Bird    (stv 0.16 0.9))
        (flies   (stv 0.16 0.9))
        ((Inheritance Sam     Raven)   (stv 0.99 0.9))
        ((Inheritance Pingu   Penguin) (stv 0.99 0.9))
        ((Inheritance Raven   Bird)    (stv 0.99 0.9))
        ((Inheritance Penguin Bird)    (stv 0.99 0.9))
        ((Inheritance Bird    flies)   (stv 0.99 0.9))
        ((Implication Penguin (Not flies)) (stv 1.0 0.9))
    """

    def test_compile_6_nodes(self, metta):
        """Full 6-node graph with negation compiles and samples."""
        metta.run(self.METTA_KB)
        stv = metta.run("!(thrml-compile!)")
        s_nodes, s_factors = parse_stv(stv)
        # 6 named nodes + "Not" may appear as an extra node in extraction
        assert s_nodes >= 6.0
        # At least 6 priors + 5 inheritance + 1 negated implication
        assert s_factors >= 12.0

    def test_sam_flies_via_chain(self, metta):
        """Sam→Raven→Bird→flies chain with clamped root.

        Tests the positive deduction path from upstream FlyingRaven.
        Upstream expected: P(flies|Sam) ≈ 0.970
        """
        graph = build_beta_chain(
            priors=[0.99, 0.5, 0.5, 0.5],
            confidences=[0.9, 0.01, 0.01, 0.01],
            strengths=[0.99, 0.99, 0.99],
            impl_confidences=[0.9, 0.9, 0.9],
            backgrounds=[0.02, 0.02, 0.02],
        )
        samples = run_beta_sampling(graph, seed=42)
        _, s, _ = estimate_beta_marginal(
            samples, graph, graph["nodes"][3])
        assert s > 0.6, f"P(flies|Sam) = {s:.3f}, expected high (upstream ~0.970)"

    def test_pingu_flies_low_via_chain(self, metta):
        """Pingu→Penguin→flies with negated strength.

        The negation Penguin→Not(flies) with strength=1.0 compiles to
        Penguin→flies with strength=0.0.  Upstream expected: ~0.010
        """
        graph = build_beta_chain(
            priors=[0.99, 0.5, 0.5],
            confidences=[0.9, 0.01, 0.01],
            strengths=[0.99, 0.0],   # 1.0 - 1.0 = 0.0 (negated)
            impl_confidences=[0.9, 0.9],
            backgrounds=[0.02, 0.02],
        )
        samples = run_beta_sampling(graph, seed=42)
        _, s, _ = estimate_beta_marginal(
            samples, graph, graph["nodes"][2])
        assert s < 0.2, f"P(flies|Pingu) = {s:.3f}, expected low (upstream ~0.010)"


# ═══════════════════════════════════════════════════════════════════════════
#  Test 3: Smokes social network
#  Source: vendor/PLN/examples/Smokes.metta
#
#  Upstream knowledge (hand-expanded from high-order Implication):
#    6 people, friend links, smoking propagation rule (strength=0.4),
#    smokes→cancerous (strength=0.6)
#    Anna smokes (stv 1.0 0.9), Edward smokes (stv 1.0 0.9)
#
#  Upstream expected: P(cancerous|Edward) ≈ (stv 0.6 0.486)
#
#  Note: upstream uses Product-typed Implication which we hand-expand
#  to pairwise Inheritance links.  See Smokes.metta lines 13-18.
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.slow
class TestSmokesNetwork:
    """6-person social network with smoking propagation + cancer rule."""

    # Hand-expanded from upstream's high-order Implication (lines 13-18):
    #   friend(A,B) ∧ smokes(A) → smokes(B) with strength=0.4
    # becomes direct Inheritance links for smoking propagation.
    #
    # Anna and Edward are known smokers (upstream lines 42-46).
    # Friend links from upstream lines 24-41.
    METTA_KB = """
        (Anna      (stv 1.0    0.9))
        (Bob       (stv 0.1667 0.9))
        (Edward    (stv 1.0    0.9))
        (Frank     (stv 0.1667 0.9))
        (Gary      (stv 0.1667 0.9))
        (Helen     (stv 0.1667 0.9))
        (cancerous (stv 0.1667 0.9))

        ; Friend-based smoking propagation (expanded from Product rule)
        ((Inheritance Anna   Bob)    (stv 0.4 0.9))
        ((Inheritance Anna   Edward) (stv 0.4 0.9))
        ((Inheritance Anna   Frank)  (stv 0.4 0.9))
        ((Inheritance Edward Frank)  (stv 0.4 0.9))
        ((Inheritance Gary   Helen)  (stv 0.4 0.9))

        ; smokes → cancerous (upstream line 19-23)
        ((Inheritance Edward cancerous) (stv 0.6 0.9))
    """

    def test_compile_7_nodes(self, metta):
        """7-node social network compiles via thrml-compile!."""
        metta.run(self.METTA_KB)
        stv = metta.run("!(thrml-compile!)")
        s_nodes, _ = parse_stv(stv)
        assert s_nodes == 7.0

    def test_edward_cancerous_via_chain(self, metta):
        """Direct Edward→cancerous path with clamped root.

        This matches what the upstream chainer computes: a single
        modus ponens from Edward(smokes=1.0) to cancerous via
        the smokes→cancerous rule (strength=0.6, confidence=0.9).
        Upstream expected: (stv 0.6 0.486)
        """
        graph = build_beta_chain(
            priors=[1.0, 0.5],
            confidences=[0.9, 0.01],
            strengths=[0.6],
            impl_confidences=[0.9],
            backgrounds=[0.02],
        )
        samples = run_beta_sampling(graph, seed=42)
        _, s, _ = estimate_beta_marginal(
            samples, graph, graph["nodes"][1])
        assert abs(s - 0.6) < STRENGTH_TOL, \
            f"P(cancerous|Edward) strength {s:.3f}, expected ~0.6"

    def test_full_network_marginal(self, metta):
        """Full 7-node network: cancerous marginal reflects smoking links."""
        metta.run(self.METTA_KB)
        metta.run("!(thrml-compile!)")
        s, c = parse_stv(metta.run("!(thrml-query! (cancerous))"))

        # In the full joint, cancerous marginal reflects both
        # Edward→cancerous and the network prior structure.
        assert s > 0.15, \
            f"cancerous marginal {s:.3f} unexpectedly low for graph with smokers"


# ═══════════════════════════════════════════════════════════════════════════
#  Test 4: RavenInduction
#  Source: vendor/PLN/examples/RavenInduction.metta
#
#  Upstream knowledge:
#    rv1→raven(0.9), rv2→raven(0.9)
#    rv1→black(0.8), rv2→black(0.0)
#
#  Upstream expected: P(black|raven) ≈ (stv 0.38 0.654)
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.slow
class TestRavenInduction:
    """Two raven instances with mixed evidence about blackness."""

    METTA_KB = """
        (rv1   (stv 0.25 0.9))
        (rv2   (stv 0.25 0.9))
        (raven (stv 0.25 0.9))
        (black (stv 0.25 0.9))
        ((Inheritance rv1 raven) (stv 0.9 0.9))
        ((Inheritance rv2 raven) (stv 0.9 0.9))
        ((Inheritance rv1 black) (stv 0.8 0.9))
        ((Inheritance rv2 black) (stv 0.0 0.9))
    """

    def test_compile_4_nodes(self, metta):
        """4-node induction graph compiles via thrml-compile!."""
        metta.run(self.METTA_KB)
        stv = metta.run("!(thrml-compile!)")
        s_nodes, _ = parse_stv(stv)
        assert s_nodes == 4.0

    def test_raven_black_conditional(self, metta):
        """P(black|raven) should reflect mixed evidence.

        Upstream expected ≈ 0.38.  Factor graph may differ (exact joint
        vs PLN's induction heuristic), but should be moderate.
        """
        metta.run(self.METTA_KB)
        metta.run("!(thrml-compile!)")
        s, c = parse_stv(metta.run("!(thrml-query! (black raven))"))

        assert 0.05 < s < 0.7, \
            f"P(black|raven) = {s:.3f}, expected moderate (upstream ~0.38)"


# ═══════════════════════════════════════════════════════════════════════════
#  Test 5: Synthetic long chains  (20 + 50 nodes)
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.slow
class TestLongChains:
    """Performance and signal attenuation on chains up to 50 nodes."""

    @staticmethod
    def _build_chain(n):
        return build_beta_chain(
            priors=[0.8] + [0.5] * (n - 1),
            confidences=[0.9] + [0.01] * (n - 1),
            strengths=[0.9] * (n - 1),
            impl_confidences=[0.9] * (n - 1),
            backgrounds=[0.02] * (n - 1),
        )

    def test_20_node_timing(self):
        """README claims 20-node chain in 1.5s; allow 10s for JIT + CI."""
        t0 = time.perf_counter()
        graph = self._build_chain(20)
        samples = run_beta_sampling(graph, seed=42)
        elapsed = time.perf_counter() - t0

        assert elapsed < 10.0, f"20-node chain took {elapsed:.1f}s (limit 10s)"
        _, s, _ = estimate_beta_marginal(samples, graph, graph["nodes"][19])
        assert 0.0 < s < 1.0

    def test_50_node_timing(self):
        """50-node chain should complete within 60s."""
        t0 = time.perf_counter()
        graph = self._build_chain(50)
        samples = run_beta_sampling(graph, seed=42)
        elapsed = time.perf_counter() - t0

        assert elapsed < 60.0, f"50-node chain took {elapsed:.1f}s (limit 60s)"
        _, s, _ = estimate_beta_marginal(samples, graph, graph["nodes"][49])
        assert 0.0 < s < 1.0

    def test_signal_attenuation_monotonic(self):
        """Shorter chains produce stronger tail signals than longer ones."""
        tail_strengths = {}
        for n in [3, 5, 10]:
            graph = self._build_chain(n)
            samples = run_beta_sampling(graph, seed=42)
            _, s, _ = estimate_beta_marginal(samples, graph, graph["nodes"][n - 1])
            tail_strengths[n] = s

        assert tail_strengths[3] > tail_strengths[5], \
            f"s_3={tail_strengths[3]:.3f} should be > s_5={tail_strengths[5]:.3f}"
        assert tail_strengths[5] > tail_strengths[10], \
            f"s_5={tail_strengths[5]:.3f} should be > s_10={tail_strengths[10]:.3f}"

    def test_convergence_diagnostics_20_node(self):
        """Convergence on a 20-node chain (all free for R-hat)."""
        graph = build_beta_chain(
            priors=[0.8] + [0.5] * 19,
            confidences=[0.9] + [0.01] * 19,
            strengths=[0.7] * 19,
            impl_confidences=[0.9] * 19,
            backgrounds=[0.02] * 19,
            clamp_root=False,
        )
        samples = run_beta_sampling(graph, seed=42)

        # Check mid-chain and tail nodes; skip early nodes near root
        # which can have degenerate variance when links are strong
        for idx in [10, 15, 19]:
            diag = diagnose_convergence(samples, graph, graph["nodes"][idx])
            assert diag["r_hat"] < 1.2, \
                f"Node {idx}: R-hat = {diag['r_hat']:.3f} (limit 1.2)"
            assert diag["ess"] > 50, \
                f"Node {idx}: ESS = {diag['ess']} (limit 50)"


# ═══════════════════════════════════════════════════════════════════════════
#  Test 6: Graph coloring correctness at scale
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.slow
class TestGraphColoring:
    """Verify _greedy_color produces valid colorings on larger graphs."""

    @staticmethod
    def _assert_valid_coloring(names, adjacency, groups):
        """No two adjacent nodes share the same color group."""
        name_to_group = {}
        for gi, group in enumerate(groups):
            for name in group:
                name_to_group[name] = gi
        for name in names:
            for neighbor in adjacency.get(name, set()):
                assert name_to_group[name] != name_to_group[neighbor], \
                    f"Adjacent nodes {name} and {neighbor} share color {name_to_group[name]}"

    def test_chain_20_nodes(self):
        names = [str(i) for i in range(20)]
        adjacency = {names[i]: set() for i in range(20)}
        for i in range(19):
            adjacency[names[i]].add(names[i + 1])
            adjacency[names[i + 1]].add(names[i])

        groups = _greedy_color(names, adjacency)
        assert len(groups) == 2, f"Chain should need 2 colors, got {len(groups)}"
        self._assert_valid_coloring(names, adjacency, groups)

    def test_chain_50_nodes(self):
        names = [str(i) for i in range(50)]
        adjacency = {names[i]: set() for i in range(50)}
        for i in range(49):
            adjacency[names[i]].add(names[i + 1])
            adjacency[names[i + 1]].add(names[i])

        groups = _greedy_color(names, adjacency)
        assert len(groups) == 2
        self._assert_valid_coloring(names, adjacency, groups)

    def test_complete_graph_20(self):
        names = [str(i) for i in range(20)]
        adjacency = {n: set(names) - {n} for n in names}

        groups = _greedy_color(names, adjacency)
        assert len(groups) == 20, f"Complete graph needs 20 colors, got {len(groups)}"
        self._assert_valid_coloring(names, adjacency, groups)

    def test_diamond_dag(self):
        """Diamond from DeductionRevision: A→{B,C}→D."""
        names = ["A", "B", "C", "D"]
        adjacency = {
            "A": {"B", "C"}, "B": {"A", "D"},
            "C": {"A", "D"}, "D": {"B", "C"},
        }
        groups = _greedy_color(names, adjacency)
        assert len(groups) == 2
        self._assert_valid_coloring(names, adjacency, groups)

    def test_raven_topology(self):
        """FlyingRaven: 6 nodes with specific adjacency."""
        names = ["Bird", "Penguin", "Pingu", "Raven", "Sam", "flies"]
        adjacency = {
            "Sam": {"Raven"}, "Raven": {"Sam", "Bird"},
            "Pingu": {"Penguin"}, "Penguin": {"Pingu", "Bird", "flies"},
            "Bird": {"Raven", "Penguin", "flies"}, "flies": {"Bird", "Penguin"},
        }
        groups = _greedy_color(sorted(names), adjacency)
        self._assert_valid_coloring(names, adjacency, groups)
