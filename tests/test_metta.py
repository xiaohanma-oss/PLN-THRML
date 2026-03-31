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

from conftest import STRENGTH_TOL, parse_conclusion, upstream_truth


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
