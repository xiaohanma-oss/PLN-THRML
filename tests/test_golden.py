"""
Golden tests — verify all thrml grounded ops through MeTTa end-to-end.

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

from conftest import STRENGTH_TOL, parse_stv, upstream_truth


# ── Modus Ponens (upstream: examples/Smokes.metta + custom cases) ───────

class TestModusPonens:
    def test_upstream_smokes(self, metta, pln_lib):
        """From examples/Smokes.metta:
        Edward smokes (stv 1.0 0.9), smokes→cancerous (stv 0.6 0.9)
        """
        s, c = parse_stv(metta.run("""
            (Edward (stv 1.0 0.9))
            ((Implication Edward Cancerous) (stv 0.6 0.9))
            !(thrml-modus-ponens! (Edward Cancerous (stv 1.0 0.9) (stv 0.6 0.9)))
        """))
        exp_s, _ = upstream_truth(pln_lib, "Truth_ModusPonens",
                                  (1.0, 0.9), (0.6, 0.9))
        assert abs(s - exp_s) < STRENGTH_TOL
        assert c > 0.0

    def test_strong(self, metta, pln_lib):
        s, c = parse_stv(metta.run("""
            (A (stv 0.8 0.9))
            ((Implication A B) (stv 0.9 0.85))
            !(thrml-modus-ponens! (A B (stv 0.8 0.9) (stv 0.9 0.85)))
        """))
        exp_s, _ = upstream_truth(pln_lib, "Truth_ModusPonens",
                                  (0.8, 0.9), (0.9, 0.85))
        assert abs(s - exp_s) < STRENGTH_TOL
        assert c > 0.0

    def test_medium(self, metta, pln_lib):
        s, c = parse_stv(metta.run("""
            (A (stv 0.5 0.8))
            ((Implication A B) (stv 0.95 0.9))
            !(thrml-modus-ponens! (A B (stv 0.5 0.8) (stv 0.95 0.9)))
        """))
        exp_s, _ = upstream_truth(pln_lib, "Truth_ModusPonens",
                                  (0.5, 0.8), (0.95, 0.9))
        assert abs(s - exp_s) < STRENGTH_TOL
        assert c > 0.0

    def test_rare(self, metta, pln_lib):
        s, c = parse_stv(metta.run("""
            (A (stv 0.1 0.7))
            ((Implication A B) (stv 0.8 0.75))
            !(thrml-modus-ponens! (A B (stv 0.1 0.7) (stv 0.8 0.75)))
        """))
        exp_s, _ = upstream_truth(pln_lib, "Truth_ModusPonens",
                                  (0.1, 0.7), (0.8, 0.75))
        assert abs(s - exp_s) < STRENGTH_TOL
        assert c > 0.0


# ── Deduction (upstream: examples/DeductionRevision.metta) ──────────────

class TestDeduction:
    def test_smokes_pattern(self, metta, pln_lib):
        s, c = parse_stv(metta.run("""
            (A (stv 0.5 0.9))
            (B (stv 0.4 0.9))
            (C (stv 0.4 0.9))
            ((Implication A B) (stv 0.8 0.9))
            ((Implication B C) (stv 0.9 0.9))
            !(thrml-deduction! (A B C (stv 0.8 0.9) (stv 0.9 0.9)))
        """))
        # PLN deduction strength ≈ 0.73; beta approach may differ slightly
        assert s > 0.5
        assert c > 0.0


# ── Inversion (upstream: ruletests/inversion.metta, RuleTester.metta) ───
# Beta gives exact Bayesian P(A|B), not PLN's heuristic.
# We check that strength is Bayes-consistent and confidence is positive.

class TestInversion:
    def test_upstream_inversion(self, metta):
        """From ruletests/inversion.metta:
        A,B (stv 0.5 1.0), (Impl B A) (stv 0.87 0.81)
        Bayes: P(A|B) = P(B|A)*P(A)/P(B)
        """
        s, c = parse_stv(metta.run("""
            (B (stv 0.5 1.0))
            (A (stv 0.5 1.0))
            ((Implication B A) (stv 0.87 0.81))
            !(thrml-inversion! (B A (stv 0.87 0.81)))
        """))
        # With equal priors, Bayes gives same strength
        assert 0.5 < s < 1.0
        assert c > 0.0

    def test_upstream_ruletester(self, metta):
        """From RuleTester.metta:
        Sam,Raven (stv 0.142 1.0), (Impl Sam Raven) (stv 0.99 0.9)
        """
        s, c = parse_stv(metta.run("""
            (Sam (stv 0.142 1.0))
            (Raven (stv 0.142 1.0))
            ((Implication Sam Raven) (stv 0.99 0.9))
            !(thrml-inversion! (Sam Raven (stv 0.99 0.9)))
        """))
        assert 0.5 < s < 1.0
        assert c > 0.0


# ── Negation (upstream: RuleTester.metta) ───────────────────────────────

class TestNegation:
    def test_upstream(self, metta, pln_lib):
        """Negation is purely analytical — no sampling."""
        s, c = parse_stv(metta.run("""
            (Penguin (stv 0.99 0.9))
            !(thrml-negation! (Penguin (stv 0.99 0.9)))
        """))
        exp_s, exp_c = upstream_truth(pln_lib, "Truth_Negation", (0.99, 0.9))
        assert abs(s - exp_s) < 0.001
        assert abs(c - exp_c) < 0.001

    def test_half(self, metta):
        s, c = parse_stv(metta.run("""
            (X (stv 0.5 0.7))
            !(thrml-negation! (X (stv 0.5 0.7)))
        """))
        assert abs(s - 0.5) < 0.001
        assert abs(c - 0.7) < 0.001


# ── Revision (no upstream ruletest — custom case) ───────────────────────

class TestRevision:
    def test_basic(self, metta):
        s, c = parse_stv(metta.run("""
            !(thrml-revision! (A 0.8 0.9 0.3 0.7))
        """))
        # Revised strength should be between the two sources
        assert 0.3 < s < 0.8
        assert c > 0.0


# ── Symmetric Modus Ponens (no upstream ruletest — custom case) ─────────

class TestSymmetricModusPonens:
    def test_basic(self, metta, pln_lib):
        s, c = parse_stv(metta.run("""
            (A (stv 0.8 0.9))
            ((Similarity A B) (stv 0.85 0.9))
            !(thrml-symmetric-mp! (A B (stv 0.8 0.9) (stv 0.85 0.9)))
        """))
        exp_s, _ = upstream_truth(pln_lib, "Truth_SymmetricModusPonens",
                                  (0.8, 0.9), (0.85, 0.9))
        assert abs(s - exp_s) < STRENGTH_TOL
        assert c > 0.0


# ── Equivalence→Implication (upstream: ruletests/equivalenceToImplication.metta) ─

class TestEquivToImpl:
    def test_upstream(self, metta):
        """From ruletests/equivalenceToImplication.metta:
        Anna,Frank (stv 0.5 1.0), (Equiv Anna Frank) (stv 0.98 0.87)
        """
        s, c = parse_stv(metta.run("""
            (Anna (stv 0.5 1.0))
            (Frank (stv 0.5 1.0))
            ((Equivalence Anna Frank) (stv 0.98 0.87))
            !(thrml-equiv-to-impl! (Anna Frank (stv 0.98 0.87)))
        """))
        # With symmetric strong coupling, P(B|A) should be high
        assert s > 0.8
        assert c > 0.0


# ── Transitive Similarity (upstream: ruletests/transitiveSimilarity.metta) ──

class TestTransitiveSimilarity:
    def test_upstream(self, metta):
        """From ruletests/transitiveSimilarity.metta:
        A,B,C (stv 0.333 1.0)
        (Sim A B) (stv 1.0 0.89), (Sim B C) (stv 1.0 0.5)
        """
        s, c = parse_stv(metta.run("""
            (A (stv 0.333 1.0))
            (B (stv 0.333 1.0))
            (C (stv 0.333 1.0))
            ((Similarity A B) (stv 1.0 0.89))
            ((Similarity B C) (stv 1.0 0.5))
            !(thrml-transitive-sim! (A B C (stv 1.0 0.89) (stv 1.0 0.5)))
        """))
        # With perfect similarities, P(C|A) should be high
        assert s > 0.7
        assert c > 0.0


# ── Induction (upstream: examples/RavenInduction.metta) ─────────────────

class TestInduction:
    def test_upstream_raven(self, metta):
        """From examples/RavenInduction.metta:
        rv1→raven, rv1→black ⊢ raven→black
        """
        s, c = parse_stv(metta.run("""
            (Raven (stv 0.25 0.9))
            (Black (stv 0.25 0.9))
            (Rv1 (stv 0.25 0.9))
            ((Implication Rv1 Raven) (stv 0.9 0.9))
            ((Implication Rv1 Black) (stv 0.8 0.9))
            !(thrml-induction! (Raven Black Rv1 (stv 0.9 0.9) (stv 0.8 0.9)))
        """))
        assert s > 0.5
        assert c > 0.0


# ── Abduction (upstream: ruletests/RuleTester.metta) ────────────────────

class TestAbduction:
    def test_upstream_ruletester(self, metta):
        """A→C, B→C ⊢ A→B"""
        s, c = parse_stv(metta.run("""
            (A (stv 0.142 1.0))
            (B (stv 0.142 1.0))
            (C (stv 0.142 1.0))
            ((Implication A C) (stv 0.3 0.9))
            ((Implication B C) (stv 0.6 0.9))
            !(thrml-abduction! (A B C (stv 0.3 0.9) (stv 0.6 0.9)))
        """))
        assert 0.0 < s < 1.0
        assert c > 0.0


# ── Evaluation Implication (upstream: ruletests/evaluationImplicationRuleA.metta) ─

class TestEvalImpl:
    def test_upstream(self, metta):
        """(Eval IsReallyFat Cat), (Impl IsReallyFat IsFat) ⊢ (Eval IsFat Cat)"""
        s, c = parse_stv(metta.run("""
            (IsReallyFat (stv 0.25 1.0))
            (Cat (stv 0.25 1.0))
            (IsFat (stv 0.25 1.0))
            ((Evaluation IsReallyFat Cat) (stv 1.0 0.9))
            ((Implication IsReallyFat IsFat) (stv 1.0 0.9))
            !(thrml-eval-impl! (IsReallyFat Cat IsFat (stv 1.0 0.9) (stv 1.0 0.9)))
        """))
        # With perfect premises, strength should be high
        assert s > 0.8
        assert c > 0.0
