"""
Golden tests — verify thrml inference through MeTTa end-to-end.

With the beta approach, both strength and confidence emerge from the
posterior distribution.  Strength is compared against upstream PLN
truth functions (lib_pln.metta).  Confidence is checked for positivity
(evidence exists) rather than exact match against PLN's analytical
formulas (which are heuristic for some rules).

Sources:
  ruletests/inversion.metta
  ruletests/equivalenceToImplication.metta
  ruletests/transitiveSimilarity.metta
  ruletests/evaluationImplicationRuleA.metta
  ruletests/RuleTester.metta
  examples/Smokes.metta
  examples/RavenInduction.metta
"""

from conftest import STRENGTH_TOL, CONFIDENCE_TOL, parse_conclusion, upstream_truth


class TestThrmlInfer:
    """Test the unified !(thrml ...) operator that dispatches by premise structure."""

    # ── Modus Ponens ──────────────────────────────────────────────────────

    def test_modus_ponens_strong(self, metta, pln_lib):
        concl, s, c = parse_conclusion(metta.run(
            '!(thrml (A (stv 0.8 0.9)) ((Implication A B) (stv 0.9 0.85)))'
        ))
        exp_s, _ = upstream_truth(pln_lib, "Truth_ModusPonens",
                                  (0.8, 0.9), (0.9, 0.85))
        assert concl == "B"
        assert abs(s - exp_s) < STRENGTH_TOL
        assert c > 0.0

    def test_modus_ponens_upstream_smokes(self, metta, pln_lib):
        """From examples/Smokes.metta: Edward smokes → cancerous."""
        concl, s, c = parse_conclusion(metta.run(
            '!(thrml (Edward (stv 1.0 0.9)) ((Implication Edward Cancerous) (stv 0.6 0.9)))'
        ))
        exp_s, _ = upstream_truth(pln_lib, "Truth_ModusPonens",
                                  (1.0, 0.9), (0.6, 0.9))
        assert concl == "Cancerous"
        assert abs(s - exp_s) < STRENGTH_TOL
        assert c > 0.0

    def test_modus_ponens_medium(self, metta, pln_lib):
        concl, s, c = parse_conclusion(metta.run(
            '!(thrml (A (stv 0.5 0.8)) ((Implication A B) (stv 0.95 0.9)))'
        ))
        exp_s, _ = upstream_truth(pln_lib, "Truth_ModusPonens",
                                  (0.5, 0.8), (0.95, 0.9))
        assert concl == "B"
        assert abs(s - exp_s) < STRENGTH_TOL
        assert c > 0.0

    def test_modus_ponens_rare(self, metta, pln_lib):
        concl, s, c = parse_conclusion(metta.run(
            '!(thrml (A (stv 0.1 0.7)) ((Implication A B) (stv 0.8 0.75)))'
        ))
        exp_s, _ = upstream_truth(pln_lib, "Truth_ModusPonens",
                                  (0.1, 0.7), (0.8, 0.75))
        assert concl == "B"
        assert abs(s - exp_s) < STRENGTH_TOL
        assert c > 0.0

    def test_modus_ponens_high_premise(self, metta, pln_lib):
        """High-confidence premise with moderate implication."""
        concl, s, c = parse_conclusion(metta.run(
            '!(thrml (A (stv 0.9 0.95)) ((Implication A B) (stv 0.5 0.8)))'
        ))
        exp_s, _ = upstream_truth(pln_lib, "Truth_ModusPonens",
                                  (0.9, 0.95), (0.5, 0.8))
        assert concl == "B"
        assert abs(s - exp_s) < STRENGTH_TOL
        assert c > 0.0

    def test_modus_ponens_confidence_ordering(self, metta):
        """Higher input confidence → higher output confidence."""
        results = []
        for c_val in ["0.3", "0.6", "0.9"]:
            _, _, c_out = parse_conclusion(metta.run(
                f'!(thrml (A (stv 0.8 {c_val})) ((Implication A B) (stv 0.9 {c_val})))'
            ))
            results.append(c_out)
        assert results[0] < results[1] < results[2]

    # ── Deduction ─────────────────────────────────────────────────────────

    def test_deduction(self, metta):
        concl, s, c = parse_conclusion(metta.run(
            '!(thrml ((Implication A B) (stv 0.8 0.9)) ((Implication B C) (stv 0.9 0.9)))'
        ))
        assert concl == "(Implication A C)"
        assert s > 0.5
        assert c > 0.0

    def test_inheritance_deduction(self, metta):
        """Deduction also works with Inheritance links."""
        concl, s, c = parse_conclusion(metta.run(
            '!(thrml ((Inheritance A B) (stv 0.8 0.9)) ((Inheritance B C) (stv 0.9 0.9)))'
        ))
        assert concl == "(Inheritance A C)"
        assert s > 0.5
        assert c > 0.0

    def test_deduction_stronger_links_higher_result(self, metta):
        """Stronger implication links → higher deduction strength."""
        results = []
        for s_val in ["0.5", "0.7", "0.9"]:
            _, s_out, _ = parse_conclusion(metta.run(
                f'!(thrml ((Implication A B) (stv {s_val} 0.9)) ((Implication B C) (stv {s_val} 0.9)))'
            ))
            results.append(s_out)
        assert results[0] < results[1] < results[2]

    # ── Inversion ─────────────────────────────────────────────────────────

    def test_inversion(self, metta):
        """From ruletests/inversion.metta: equal priors, Bayes gives same strength."""
        concl, s, c = parse_conclusion(metta.run(
            '!(thrml ((Implication B A) (stv 0.87 0.81)))'
        ))
        assert concl == "(Implication A B)"
        assert 0.5 < s < 1.0
        assert c > 0.0

    def test_inversion_ruletester(self, metta):
        """From RuleTester.metta: Sam/Raven with asymmetric priors."""
        concl, s, c = parse_conclusion(metta.run(
            '!(thrml ((Implication Sam Raven) (stv 0.99 0.9)))'
        ))
        assert concl == "(Implication Raven Sam)"
        assert 0.0 < s < 1.0
        assert c > 0.0

    # ── Negation ──────────────────────────────────────────────────────────

    def test_negation(self, metta, pln_lib):
        concl, s, c = parse_conclusion(metta.run(
            '!(thrml ((Not Penguin) (stv 0.99 0.9)))'
        ))
        exp_s, exp_c = upstream_truth(pln_lib, "Truth_Negation", (0.99, 0.9))
        assert concl == "Penguin"
        assert abs(s - exp_s) < 0.001
        assert abs(c - exp_c) < 0.001

    def test_negation_half(self, metta):
        """Negation of s=0.5 should give s=0.5."""
        concl, s, c = parse_conclusion(metta.run(
            '!(thrml ((Not X) (stv 0.5 0.7)))'
        ))
        assert concl == "X"
        assert abs(s - 0.5) < 0.001
        assert abs(c - 0.7) < 0.001

    # ── Revision ──────────────────────────────────────────────────────────

    def test_revision(self, metta):
        concl, s, c = parse_conclusion(metta.run(
            '!(thrml (A (stv 0.8 0.9)) (A (stv 0.3 0.7)))'
        ))
        assert concl == "A"
        assert 0.3 < s < 0.8
        assert c > 0.0

    # ── Symmetric Modus Ponens ────────────────────────────────────────────

    def test_symmetric_mp(self, metta, pln_lib):
        concl, s, c = parse_conclusion(metta.run(
            '!(thrml (A (stv 0.8 0.9)) ((Similarity A B) (stv 0.85 0.9)))'
        ))
        exp_s, _ = upstream_truth(pln_lib, "Truth_SymmetricModusPonens",
                                  (0.8, 0.9), (0.85, 0.9))
        assert concl == "B"
        assert abs(s - exp_s) < STRENGTH_TOL
        assert c > 0.0

    # ── Equivalence → Implication ─────────────────────────────────────────

    def test_equiv_to_impl(self, metta):
        """From ruletests/equivalenceToImplication.metta."""
        concl, s, c = parse_conclusion(metta.run(
            '!(thrml ((Equivalence Anna Frank) (stv 0.98 0.87)))'
        ))
        assert concl == "(Implication Anna Frank)"
        assert s > 0.8
        assert c > 0.0

    # ── Transitive Similarity ─────────────────────────────────────────────

    def test_transitive_similarity(self, metta):
        """From ruletests/transitiveSimilarity.metta."""
        concl, s, c = parse_conclusion(metta.run(
            '!(thrml ((Similarity A B) (stv 1.0 0.89)) ((Similarity B C) (stv 1.0 0.5)))'
        ))
        assert concl == "(Similarity A C)"
        assert s > 0.7
        assert c > 0.0

    # ── Induction ─────────────────────────────────────────────────────────

    def test_induction(self, metta):
        """From examples/RavenInduction.metta: rv1→raven, rv1→black ⊢ raven→black."""
        concl, s, c = parse_conclusion(metta.run(
            '!(thrml ((Implication Rv1 Raven) (stv 0.9 0.9)) ((Implication Rv1 Black) (stv 0.8 0.9)))'
        ))
        assert concl == "(Implication Raven Black)"
        assert s > 0.5
        assert c > 0.0

    # ── Abduction ─────────────────────────────────────────────────────────

    def test_abduction(self, metta):
        """From ruletests/RuleTester.metta: A→C, B→C ⊢ A→B."""
        concl, s, c = parse_conclusion(metta.run(
            '!(thrml ((Implication A C) (stv 0.3 0.9)) ((Implication B C) (stv 0.6 0.9)))'
        ))
        assert concl == "(Implication A B)"
        assert 0.0 < s < 1.0
        assert c > 0.0

    # ── Evaluation Implication ────────────────────────────────────────────

    def test_eval_implication(self, metta):
        """From ruletests/evaluationImplicationRuleA.metta."""
        concl, s, c = parse_conclusion(metta.run(
            '!(thrml ((Evaluation IsReallyFat Cat) (stv 1.0 0.9)) ((Implication IsReallyFat IsFat) (stv 1.0 0.9)))'
        ))
        assert concl == "(Evaluation IsFat Cat)"
        assert s > 0.8
        assert c > 0.0


class TestQuantaleAlgebra:
    """Verify Q_tv ⊗ (revision / energy addition) algebraic properties via MeTTa."""

    def test_revision_commutativity(self, metta):
        """rev(a, b) ≈ rev(b, a) — Q_tv ⊗ is commutative."""
        _, s1, c1 = parse_conclusion(metta.run(
            '!(thrml (A (stv 0.8 0.7)) (A (stv 0.3 0.5)))'
        ))
        _, s2, c2 = parse_conclusion(metta.run(
            '!(thrml (A (stv 0.3 0.5)) (A (stv 0.8 0.7)))'
        ))
        assert abs(s1 - s2) < STRENGTH_TOL
        assert abs(c1 - c2) < CONFIDENCE_TOL

    def test_revision_three_sources_stronger_than_two(self, metta):
        """Adding a third agreeing evidence source → higher confidence."""
        # Two sources
        _, _, c_two = parse_conclusion(metta.run(
            '!(thrml (A (stv 0.7 0.6)) (A (stv 0.7 0.5)))'
        ))
        # Three sources: feed revision result back with a third observation
        _, s_two, c_two_val = parse_conclusion(metta.run(
            '!(thrml (A (stv 0.7 0.6)) (A (stv 0.7 0.5)))'
        ))
        _, _, c_three = parse_conclusion(metta.run(
            f'!(thrml (A (stv {s_two} {c_two_val})) (A (stv 0.7 0.7)))'
        ))
        assert c_three > c_two, "More evidence should yield higher confidence"

    def test_revision_identity_element(self, metta):
        """Revision with uniform prior (c≈0) should not change the result.

        The Q_tv identity is Beta(1,1) = uniform = (s=0.5, c≈0).
        Near-uniform has a small but non-zero effect, so we use relaxed tolerance.
        """
        _, s1, c1 = parse_conclusion(metta.run(
            '!(thrml (A (stv 0.8 0.7)) (A (stv 0.5 0.001)))'
        ))
        assert abs(s1 - 0.8) < 2 * STRENGTH_TOL
        assert c1 > 0.5

    def test_revision_conflicting_evidence(self, metta):
        """Conflicting evidence → strength between the two inputs."""
        _, s, c = parse_conclusion(metta.run(
            '!(thrml (A (stv 0.9 0.8)) (A (stv 0.1 0.8)))'
        ))
        assert 0.2 < s < 0.8, f"Conflicting revision strength {s} should be between inputs"
        assert c > 0.0


class TestTopologies:
    """Multi-step inference topologies via MeTTa — validates Q_tv ⊕ (marginalization)."""

    def test_diamond_two_paths_stronger(self, metta):
        """Diamond graph (2 deduction paths + revision) → higher confidence than single chain.

        Uses a simple atom name for revision to avoid ambiguous dispatch
        with compound atoms like (Inheritance A D).
        """
        # Path 1: A→B→D deduction
        _, s_p1, c_p1 = parse_conclusion(metta.run(
            '!(thrml ((Inheritance A B) (stv 0.7 0.9)) ((Inheritance B D) (stv 0.7 0.9)))'
        ))
        # Path 2: A→C→D deduction
        _, s_p2, c_p2 = parse_conclusion(metta.run(
            '!(thrml ((Inheritance A C) (stv 0.7 0.9)) ((Inheritance C D) (stv 0.7 0.9)))'
        ))
        # Merge via revision using simple atom to avoid spurious rule matches
        _, _, c_rev = parse_conclusion(metta.run(
            f'!(thrml (AD_result (stv {s_p1} {c_p1})) (AD_result (stv {s_p2} {c_p2})))'
        ))
        # Single chain confidence (same as path 1)
        assert c_rev > c_p1, \
            f"Diamond c={c_rev:.3f} should be > single chain c={c_p1:.3f}"

    def test_longer_chain_weaker(self, metta):
        """Longer chain → lower confidence at the tail."""
        # Short: A→B (1-step MP)
        _, _, c_short = parse_conclusion(metta.run(
            '!(thrml (A (stv 0.8 0.9)) ((Implication A B) (stv 0.8 0.9)))'
        ))
        # Long: A→B→C (2-step MP)
        _, s_ab, c_ab = parse_conclusion(metta.run(
            '!(thrml (A (stv 0.8 0.9)) ((Implication A B) (stv 0.8 0.9)))'
        ))
        _, _, c_long = parse_conclusion(metta.run(
            f'!(thrml (B (stv {s_ab} {c_ab})) ((Implication B C) (stv 0.8 0.9)))'
        ))
        assert c_short > c_long, \
            f"Short chain c={c_short:.3f} should be > long chain c={c_long:.3f}"

    def test_sam_flies_chain(self, metta):
        """Sam→Raven→Bird→flies positive deduction chain.

        From vendor/PLN/examples/FlyingRaven.metta.
        Upstream expected: P(flies|Sam) ≈ 0.970
        """
        # Step 1: Sam→Raven
        _, s1, c1 = parse_conclusion(metta.run(
            '!(thrml (Sam (stv 0.99 0.9)) ((Implication Sam Raven) (stv 0.99 0.9)))'
        ))
        # Step 2: Raven→Bird
        _, s2, c2 = parse_conclusion(metta.run(
            f'!(thrml (Raven (stv {s1} {c1})) ((Implication Raven Bird) (stv 0.99 0.9)))'
        ))
        # Step 3: Bird→flies
        _, s3, _ = parse_conclusion(metta.run(
            f'!(thrml (Bird (stv {s2} {c2})) ((Implication Bird Flies) (stv 0.99 0.9)))'
        ))
        assert s3 > 0.6, f"P(flies|Sam) = {s3:.3f}, expected high"

    def test_pingu_doesnt_fly(self, metta):
        """Pingu→Penguin→¬flies negative path.

        From vendor/PLN/examples/FlyingRaven.metta.
        Negation: Penguin→Not(flies) s=1.0 compiles to Penguin→flies s=0.0.
        Upstream expected: P(flies|Pingu) ≈ 0.010
        """
        # Step 1: Pingu→Penguin
        _, s1, c1 = parse_conclusion(metta.run(
            '!(thrml (Pingu (stv 0.99 0.9)) ((Implication Pingu Penguin) (stv 0.99 0.9)))'
        ))
        # Step 2: Penguin→flies with negated strength (0.0)
        _, s2, _ = parse_conclusion(metta.run(
            f'!(thrml (Penguin (stv {s1} {c1})) ((Implication Penguin Flies) (stv 0.01 0.9)))'
        ))
        assert s2 < 0.3, f"P(flies|Pingu) = {s2:.3f}, expected low"

    def test_edward_cancerous(self, metta, pln_lib):
        """Edward smokes → cancerous, direct MP.

        From vendor/PLN/examples/Smokes.metta.
        Upstream expected: (stv 0.6 0.486)
        """
        concl, s, c = parse_conclusion(metta.run(
            '!(thrml (Edward (stv 1.0 0.9)) ((Implication Edward Cancerous) (stv 0.6 0.9)))'
        ))
        exp_s, _ = upstream_truth(pln_lib, "Truth_ModusPonens",
                                  (1.0, 0.9), (0.6, 0.9))
        assert concl == "Cancerous"
        assert abs(s - exp_s) < STRENGTH_TOL


class TestExtremeInputs:
    """Verify MeTTa inference handles extreme (s, c) values without crashing."""

    def test_extreme_mp_no_crash(self, metta):
        """Very confident premise + strong implication → valid output."""
        _, s, c = parse_conclusion(metta.run(
            '!(thrml (A (stv 0.05 0.95)) ((Implication A B) (stv 0.95 0.95)))'
        ))
        assert 0.0 < s < 1.0, f"Extreme MP strength {s} out of [0,1]"
        assert 0.0 <= c <= 1.0, f"Extreme MP confidence {c} out of [0,1]"

    def test_near_zero_confidence(self, metta):
        """Near-zero confidence inputs → valid output."""
        _, s, c = parse_conclusion(metta.run(
            '!(thrml (A (stv 0.5 0.01)) ((Implication A B) (stv 0.9 0.01)))'
        ))
        assert 0.0 < s < 1.0
        assert 0.0 <= c <= 1.0

    def test_near_one_both(self, metta):
        """Near-maximum strength and confidence → valid output."""
        _, s, c = parse_conclusion(metta.run(
            '!(thrml (A (stv 0.99 0.99)) ((Implication A B) (stv 0.99 0.99)))'
        ))
        assert 0.0 < s < 1.0
        assert 0.0 <= c <= 1.0
