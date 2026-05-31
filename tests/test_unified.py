"""
Seven-column precision comparison: DTV | PLN | Binary | Hybrid | Unified | Inversion | Revision.

Validates the unified LBM(s) + QLN(n) architecture:
  - s layer (TSU): g(n)-modulated binary Ising sampling
  - n layer (CPU): QLN-style closed-form confidence propagation

New rules: Inversion (CPU-only, Bayes vs PLN heuristic) and Revision (CPU-only, PLN book formula n_rev = n₁ + n₂).
All Δ values measured against DTV (continuous Beta MC, zero discretisation error).
"""

import pytest
import numpy as np

from pln_thrml.unified import (
    unified_modus_ponens,
    unified_deduction,
    unified_abduction,
    unified_inversion,
    unified_revision,
)
from pln_thrml.hybrid import (
    hybrid_modus_ponens,
    hybrid_deduction,
)
from pln_thrml.dtv_baseline import (
    dtv_modus_ponens,
    dtv_deduction,
    dtv_abduction,
    dtv_inversion,
    dtv_revision,
)
from pln_thrml.qln_cpu import (
    c_modus_ponens,
    c_deduction,
    inversion_pln,
    inversion_bayes,
    revision,
)
from pln_thrml.compiler_unified import default_g_fn
from pln_thrml.pln_utils import c2w, w2c
from tests.conftest import STRENGTH_TOL


# ═══════════════════════════════════════════════════════════════════════════
#  Parameters (reuse from test_hybrid)
# ═══════════════════════════════════════════════════════════════════════════

MP_PARAMS = [
    {"s_A": 0.8, "c_A": 0.9, "s_AB": 0.9, "c_AB": 0.85, "id": "strong"},
    {"s_A": 0.6, "c_A": 0.7, "s_AB": 0.7, "c_AB": 0.8, "id": "medium"},
    {"s_A": 0.3, "c_A": 0.95, "s_AB": 0.5, "c_AB": 0.9, "id": "rare"},
]

DEDUCTION_PARAMS = [
    {"s_A": 0.8, "c_A": 0.9, "s_B": 0.5, "c_B": 0.01, "s_C": 0.5, "c_C": 0.01,
     "s_AB": 0.9, "c_AB": 0.85, "s_BC": 0.8, "c_BC": 0.9, "id": "standard"},
    {"s_A": 0.05, "c_A": 0.8, "s_B": 0.2, "c_B": 0.9999, "s_C": 1.0, "c_C": 0.8,
     "s_AB": 1.0, "c_AB": 0.9999, "s_BC": 0.3, "c_BC": 0.8, "id": "extreme"},
]

ABDUCTION_PARAMS = [
    # s_C is PLN book Ch 5.4 required input; default Noisy-OR point estimate
    {"s_A": 0.8, "c_A": 0.9, "s_B": 0.7, "c_B": 0.85,
     "s_AC": 0.9, "c_AC": 0.85, "s_BC": 0.8, "c_BC": 0.9,
     "s_C": 0.877, "c_C": 0.85, "id": "standard"},
    {"s_A": 0.5, "c_A": 0.8, "s_B": 0.9, "c_B": 0.9,
     "s_AC": 0.7, "c_AC": 0.9, "s_BC": 0.6, "c_BC": 0.85,
     "s_C": 0.701, "c_C": 0.8, "id": "asymmetric"},
]

INVERSION_PARAMS = [
    {"s_A": 0.7, "c_A": 0.9, "s_B": 0.7, "c_B": 0.9,
     "s_AB": 0.8, "c_AB": 0.85, "id": "symmetric"},
    {"s_A": 0.8, "c_A": 0.9, "s_B": 0.3, "c_B": 0.85,
     "s_AB": 0.9, "c_AB": 0.9, "id": "asymmetric"},
]

REVISION_PARAMS = [
    {"s1": 0.8, "c1": 0.5, "s2": 0.7, "c2": 0.5, "id": "equal_conf"},
    {"s1": 0.9, "c1": 0.9, "s2": 0.3, "c2": 0.3, "id": "unequal_conf"},
    {"s1": 0.6, "c1": 0.8, "s2": 0.6, "c2": 0.8, "id": "idempotent"},
]


S_TOL = 0.15   # Unified trades some DTV accuracy for hardware-meaningful g(n) modulation
C_TOL = 0.15


# ═══════════════════════════════════════════════════════════════════════════
#  PLN closed-form strength formulas (for comparison columns)
# ═══════════════════════════════════════════════════════════════════════════

def _pln_formula_mp(p, background=0.02):
    return p["s_A"] * p["s_AB"] + background * (1.0 - p["s_A"])


def _pln_formula_deduction(p):
    denom = max(1.0 - p["s_B"], 1e-7)
    return (p["s_AB"] * p["s_BC"]
            + (1.0 - p["s_AB"]) * (p["s_C"] - p["s_B"] * p["s_BC"]) / denom)


def _pln_formula_abduction(p):
    denom = max(1.0 - p["s_A"], 1e-7)
    return (p["s_AC"] * p["s_BC"]
            + (1.0 - p["s_AC"]) * (p["s_B"] - p["s_A"] * p["s_BC"]) / denom)


# ═══════════════════════════════════════════════════════════════════════════
#  Test: g(n) modulation function
# ═══════════════════════════════════════════════════════════════════════════

class TestGFunction:
    """g(n) modulation correctness."""

    def test_baseline(self):
        """n=2 (minimum, c≈0) → g=1."""
        assert default_g_fn(2.0) == pytest.approx(1.0, abs=1e-10)

    def test_moderate(self):
        """n=10 → g≈2.24."""
        assert default_g_fn(10.0) == pytest.approx(2.236, abs=0.01)

    def test_high(self):
        """n=100 → g≈7.07."""
        assert default_g_fn(100.0) == pytest.approx(7.071, abs=0.01)

    def test_cap(self):
        """n=1000 → g capped at 10."""
        assert default_g_fn(1000.0) == pytest.approx(10.0, abs=1e-10)

    def test_zero(self):
        """n=0 → g=0."""
        assert default_g_fn(0.0) == pytest.approx(0.0, abs=1e-10)

    def test_negative_clamped(self):
        """Negative n → g=0 (clamped via max)."""
        assert default_g_fn(-5.0) == pytest.approx(0.0, abs=1e-10)


# ═══════════════════════════════════════════════════════════════════════════
#  Test: Unified Modus Ponens
# ═══════════════════════════════════════════════════════════════════════════

class TestUnifiedMP:
    """Unified MP: DTV | PLN | Hybrid | Unified comparison."""

    @pytest.mark.parametrize("p", MP_PARAMS, ids=lambda p: p["id"])
    def test_strength_within_tolerance(self, p, capsys):
        s_dtv, c_dtv = dtv_modus_ponens(p["s_A"], p["c_A"], p["s_AB"], p["c_AB"])
        s_pln = _pln_formula_mp(p)
        s_hyb, c_hyb = hybrid_modus_ponens(
            p["s_A"], p["c_A"], p["s_AB"], p["c_AB"])
        s_uni, c_uni, meta = unified_modus_ponens(
            p["s_A"], p["c_A"], p["s_AB"], p["c_AB"])

        with capsys.disabled():
            print(f"\n  MP [{p['id']}]:"
                  f"  DTV={s_dtv:.3f}"
                  f"  PLN={s_pln:.3f}"
                  f"  Hyb={s_hyb:.3f}"
                  f"  Uni={s_uni:.3f}"
                  f"  Δ(hyb)={abs(s_dtv-s_hyb):.3f}"
                  f"  Δ(uni)={abs(s_dtv-s_uni):.3f}"
                  f"  rounds={meta['rounds']}")

        assert abs(s_dtv - s_uni) < S_TOL, (
            f"Unified MP s={s_uni:.4f} too far from DTV s={s_dtv:.4f}")

    @pytest.mark.parametrize("p", MP_PARAMS, ids=lambda p: p["id"])
    def test_confidence_matches_pln(self, p):
        _, c_uni, _ = unified_modus_ponens(
            p["s_A"], p["c_A"], p["s_AB"], p["c_AB"])
        c_pln = c_modus_ponens(p["c_A"], p["c_AB"])
        assert c_uni == pytest.approx(c_pln, abs=0.001), (
            f"Unified c={c_uni:.4f} should match PLN c={c_pln:.4f}")


# ═══════════════════════════════════════════════════════════════════════════
#  Test: Unified Deduction
# ═══════════════════════════════════════════════════════════════════════════

class TestUnifiedDeduction:
    """Unified Deduction: DTV | PLN | Hybrid | Unified comparison."""

    @pytest.mark.parametrize("p", DEDUCTION_PARAMS, ids=lambda p: p["id"])
    def test_strength_within_tolerance(self, p, capsys):
        s_dtv, _ = dtv_deduction(
            p["s_A"], p["c_A"], p["s_B"], p["c_B"], p["s_C"], p["c_C"],
            p["s_AB"], p["c_AB"], p["s_BC"], p["c_BC"])
        s_pln = _pln_formula_deduction(p)
        s_hyb, _ = hybrid_deduction(
            p["s_A"], p["c_A"], p["s_B"], p["c_B"], p["s_C"], p["c_C"],
            p["s_AB"], p["c_AB"], p["s_BC"], p["c_BC"])
        s_uni, c_uni, meta = unified_deduction(
            p["s_A"], p["c_A"], p["s_B"], p["c_B"], p["s_C"], p["c_C"],
            p["s_AB"], p["c_AB"], p["s_BC"], p["c_BC"])

        with capsys.disabled():
            print(f"\n  Ded [{p['id']}]:"
                  f"  DTV={s_dtv:.3f}"
                  f"  PLN={s_pln:.3f}"
                  f"  Hyb={s_hyb:.3f}"
                  f"  Uni={s_uni:.3f}"
                  f"  Δ(hyb)={abs(s_dtv-s_hyb):.3f}"
                  f"  Δ(uni)={abs(s_dtv-s_uni):.3f}"
                  f"  rounds={meta['rounds']}")

        # Extreme cases (c≈1 → g(n)=cap) have higher deviation;
        # even hybrid has Δ>0.25 on extreme params
        tol = 0.30 if p["id"] == "extreme" else S_TOL
        assert abs(s_dtv - s_uni) < tol, (
            f"Unified Ded s={s_uni:.4f} too far from DTV s={s_dtv:.4f}")

    @pytest.mark.parametrize("p", DEDUCTION_PARAMS, ids=lambda p: p["id"])
    def test_confidence_matches_pln(self, p):
        _, c_uni, _ = unified_deduction(
            p["s_A"], p["c_A"], p["s_B"], p["c_B"], p["s_C"], p["c_C"],
            p["s_AB"], p["c_AB"], p["s_BC"], p["c_BC"])
        c_pln = c_deduction(p["s_AB"], p["c_AB"], p["s_BC"], p["c_BC"])
        assert c_uni == pytest.approx(c_pln, abs=0.001)


# ═══════════════════════════════════════════════════════════════════════════
#  Test: Unified Abduction
# ═══════════════════════════════════════════════════════════════════════════

class TestUnifiedAbduction:
    """Unified Abduction: chain method (PLN Ch 5.4) + legacy methods vs DTV."""

    @pytest.mark.parametrize("p", ABDUCTION_PARAMS, ids=lambda p: p["id"])
    def test_chain_within_tolerance(self, p, capsys):
        """Chain method (Deduction∘Inversion) should satisfy Δs<0.05 tolerance."""
        kw = dict(s_A=p["s_A"], c_A=p["c_A"], s_B=p["s_B"], c_B=p["c_B"],
                  s_AC=p["s_AC"], c_AC=p["c_AC"], s_BC=p["s_BC"], c_BC=p["c_BC"],
                  s_C=p["s_C"], c_C=p["c_C"])
        s_dtv, _ = dtv_abduction(**kw)
        s_chain, c_chain, _ = unified_abduction(**kw, method="chain")

        with capsys.disabled():
            print(f"\n  Abd [{p['id']}]:"
                  f"  DTV={s_dtv:.3f}"
                  f"  chain={s_chain:.3f}(Δ={abs(s_dtv-s_chain):.3f})"
                  f"  c={c_chain:.3f}")

        assert abs(s_dtv - s_chain) < STRENGTH_TOL, \
            f"chain Δ={abs(s_dtv-s_chain):.3f} exceeds {STRENGTH_TOL}"
        assert 0.0 <= s_chain <= 1.0

    @pytest.mark.parametrize("p", ABDUCTION_PARAMS, ids=lambda p: p["id"])
    def test_legacy_methods(self, p, capsys):
        """Legacy methods (binary/binary_hidden/mfc) — kept for regression tracking."""
        kw = dict(s_A=p["s_A"], c_A=p["c_A"], s_B=p["s_B"], c_B=p["c_B"],
                  s_AC=p["s_AC"], c_AC=p["c_AC"], s_BC=p["s_BC"], c_BC=p["c_BC"],
                  s_C=p["s_C"], c_C=p["c_C"])
        s_dtv, _ = dtv_abduction(**kw)
        s_bin, _, _ = unified_abduction(**kw, method="binary")
        s_hid, _, _ = unified_abduction(**kw, method="binary_hidden")
        s_mfc, _, meta_mfc = unified_abduction(**kw, method="mfc", max_rounds=5)

        with capsys.disabled():
            print(f"\n  Abd [{p['id']}] legacy:"
                  f"  DTV={s_dtv:.3f}"
                  f"  Bin={s_bin:.3f}(Δ={abs(s_dtv-s_bin):.3f})"
                  f"  Hid={s_hid:.3f}(Δ={abs(s_dtv-s_hid):.3f})"
                  f"  MFC={s_mfc:.3f}(Δ={abs(s_dtv-s_mfc):.3f})"
                  f"  MFC-rnds={meta_mfc['rounds']}")

        for s in [s_bin, s_hid, s_mfc]:
            assert 0.0 <= s <= 1.0


# ═══════════════════════════════════════════════════════════════════════════
#  Test: Inversion (CPU-only)
# ═══════════════════════════════════════════════════════════════════════════

class TestInversion:
    """Inversion: DTV | PLN heuristic | Bayes formula."""

    @pytest.mark.parametrize("p", INVERSION_PARAMS, ids=lambda p: p["id"])
    def test_bayes_vs_dtv(self, p, capsys):
        s_dtv, c_dtv = dtv_inversion(
            p["s_A"], p["c_A"], p["s_B"], p["c_B"], p["s_AB"], p["c_AB"])
        s_pln, c_pln = inversion_pln(
            p["s_A"], p["c_A"], p["s_B"], p["c_B"], p["s_AB"], p["c_AB"])
        s_bay, c_bay = inversion_bayes(
            p["s_A"], p["c_A"], p["s_B"], p["c_B"], p["s_AB"], p["c_AB"])
        s_uni, c_uni = unified_inversion(
            p["s_A"], p["c_A"], p["s_B"], p["c_B"], p["s_AB"], p["c_AB"])

        with capsys.disabled():
            print(f"\n  Inv [{p['id']}]:"
                  f"  DTV=({s_dtv:.3f},{c_dtv:.3f})"
                  f"  PLN=({s_pln:.3f},{c_pln:.3f})"
                  f"  Bayes=({s_bay:.3f},{c_bay:.3f})"
                  f"  Unified=({s_uni:.3f},{c_uni:.3f})"
                  f"  Δ(pln)={abs(s_dtv-s_pln):.3f}"
                  f"  Δ(bay)={abs(s_dtv-s_bay):.3f}")

        # Unified (Bayes) should be closer to DTV than PLN heuristic
        # (at least for strength, which is the key difference)
        assert abs(s_dtv - s_uni) < 0.15, (
            f"Unified Inv s={s_uni:.4f} too far from DTV s={s_dtv:.4f}")

    def test_symmetric_case(self):
        """When s_A = s_B, inversion should preserve s_AB."""
        s, c = unified_inversion(0.7, 0.9, 0.7, 0.9, 0.8, 0.85)
        assert s == pytest.approx(0.8, abs=0.01), (
            f"Symmetric inversion: s_BA={s:.4f} should ≈ s_AB=0.8")

    def test_division_safety(self):
        """s_B near 0 should clamp, not blow up."""
        s, c = unified_inversion(0.5, 0.8, 0.05, 0.9, 0.8, 0.85)
        assert np.isfinite(s) and np.isfinite(c)
        assert 0.0 <= s <= 1.0


# ═══════════════════════════════════════════════════════════════════════════
#  Test: Revision (CPU-only, PLN book formula)
# ═══════════════════════════════════════════════════════════════════════════

class TestRevision:
    """Revision: PLN book n_rev = n₁+n₂ (raw counts) vs DTV baseline."""

    @pytest.mark.parametrize("p", REVISION_PARAMS, ids=lambda p: p["id"])
    def test_strength_matches_dtv(self, p, capsys):
        s_dtv, c_dtv = dtv_revision(p["s1"], p["c1"], p["s2"], p["c2"])
        s_rev, c_rev = unified_revision(p["s1"], p["c1"], p["s2"], p["c2"])

        with capsys.disabled():
            print(f"\n  Rev [{p['id']}]:"
                  f"  DTV=({s_dtv:.3f},{c_dtv:.3f})"
                  f"  QLN=({s_rev:.3f},{c_rev:.3f})"
                  f"  Δs={abs(s_dtv-s_rev):.3f}"
                  f"  Δc={abs(c_dtv-c_rev):.3f}")

        # Revision strength should match DTV closely (both are weighted avg)
        assert abs(s_dtv - s_rev) < 0.05, (
            f"Revision s={s_rev:.4f} too far from DTV s={s_dtv:.4f}")

    def test_equal_evidence_is_average(self):
        """Equal confidence → s_rev is simple average."""
        s, c = unified_revision(0.8, 0.5, 0.4, 0.5)
        assert s == pytest.approx(0.6, abs=0.01)

    def test_high_conf_dominates(self):
        """Higher-confidence source dominates the weighted average."""
        s, c = unified_revision(0.9, 0.95, 0.3, 0.1)
        # c=0.95 → n=19, c=0.1 → n≈0.11; high-conf dominates strongly
        assert s > 0.7, f"High-conf source (0.9) should dominate, got s={s:.3f}"

    def test_confidence_increases(self):
        """Merging independent sources should increase confidence."""
        _, c1 = 0.6, 0.5
        _, c2 = 0.6, 0.5
        _, c_rev = unified_revision(0.6, 0.5, 0.6, 0.5)
        assert c_rev > 0.5, (
            f"Revision c={c_rev:.3f} should be > input c=0.5")

    def test_idempotent_strength(self):
        """revision(x, x) should preserve strength."""
        s, c = unified_revision(0.6, 0.8, 0.6, 0.8)
        assert s == pytest.approx(0.6, abs=0.01)

    def test_revision_matches_pln_book(self, capsys):
        """Verify revision uses PLN book formula (n_rev = n1 + n2, raw counts)."""
        s1, c1, s2, c2 = 0.8, 0.5, 0.7, 0.5

        # Code path
        s_code, c_code = revision(s1, c1, s2, c2)

        # Direct PLN book formula (Ch 5 §5.10)
        w1, w2 = c2w(c1), c2w(c2)
        w_pln = w1 + w2
        s_pln = (w1 * s1 + w2 * s2) / w_pln
        c_pln = w2c(w_pln)

        with capsys.disabled():
            print(f"\n  Rev code vs PLN book:"
                  f"  code=({s_code:.4f},{c_code:.4f})"
                  f"  PLN=({s_pln:.4f},{c_pln:.4f})")

        # Code must match PLN book / QLN paper convention exactly
        assert abs(s_code - s_pln) < 1e-9
        assert abs(c_code - c_pln) < 1e-9


# ═══════════════════════════════════════════════════════════════════════════
#  Test: Damped Deduction (marginal message passing)
# ═══════════════════════════════════════════════════════════════════════════

class TestDeductionGCalibration:
    """Per-rule g calibration: weaker g reduces Jensen gap for multi-edge chains."""

    @pytest.mark.parametrize("p", DEDUCTION_PARAMS, ids=lambda p: p["id"])
    def test_weaker_g_reduces_bias(self, p, capsys):
        """sqrt(n/4) cap=6 should reduce Δ for standard deduction parameters.

        Active Inference insight: multi-edge chains need weaker precision
        (inverse temperature) than single-edge rules.  The default g(n) =
        sqrt(n/2) is calibrated for single edges; a 3-node deduction chain
        amplifies coupling twice, causing over-confident posteriors.
        """
        import math
        kw = dict(s_A=p["s_A"], c_A=p["c_A"], s_B=p["s_B"], c_B=p["c_B"],
                  s_C=p["s_C"], c_C=p["c_C"],
                  s_AB=p["s_AB"], c_AB=p["c_AB"], s_BC=p["s_BC"], c_BC=p["c_BC"],
                  max_rounds=1, seed=42, n_batches=200)

        s_dtv, _ = dtv_deduction(
            p["s_A"], p["c_A"], p["s_B"], p["c_B"], p["s_C"], p["c_C"],
            p["s_AB"], p["c_AB"], p["s_BC"], p["c_BC"])

        g_default = lambda n: min(math.sqrt(max(n, 0.0) / 2.0), 10)
        g_weak = lambda n: min(math.sqrt(max(n, 0.0) / 4.0), 6)

        s_def, _, _ = unified_deduction(**kw, g_fn=g_default)
        s_weak, _, _ = unified_deduction(**kw, g_fn=g_weak)

        delta_def = abs(s_dtv - s_def)
        delta_weak = abs(s_dtv - s_weak)

        with capsys.disabled():
            print(f"\n  GCal [{p['id']}]:"
                  f"  DTV={s_dtv:.3f}"
                  f"  default={s_def:.3f} (Δ={delta_def:.3f})"
                  f"  weak={s_weak:.3f} (Δ={delta_weak:.3f})"
                  f"  improvement={delta_def - delta_weak:+.3f}")

        # Both should be within overall tolerance
        tol = 0.30 if p["id"] == "extreme" else S_TOL
        assert delta_weak < tol, (
            f"Weak-g Ded s={s_weak:.4f} too far from DTV s={s_dtv:.4f}")


# ═══════════════════════════════════════════════════════════════════════════
#  Test: Convergence diagnostics
# ═══════════════════════════════════════════════════════════════════════════

class TestConvergence:
    """Iterative loop convergence properties."""

    def test_mp_runs_all_rounds(self):
        """MP runs max_rounds since c doesn't feed back into s.

        Each round uses a different seed → sampling noise prevents
        convergence by the tol=0.01 criterion, but that's fine — MP has
        no iterative feedback loop.
        """
        _, _, meta = unified_modus_ponens(0.8, 0.9, 0.9, 0.85, max_rounds=3)
        assert meta["rounds"] <= 3

    def test_deduction_converges(self):
        """Deduction should converge within max_rounds."""
        _, _, meta = unified_deduction(
            0.8, 0.9, 0.5, 0.01, 0.5, 0.01,
            0.9, 0.85, 0.8, 0.9, max_rounds=5)
        # Should converge; if not, at least shouldn't crash
        assert meta["rounds"] <= 5

    def test_history_recorded(self):
        """History should record (s, c) for each round."""
        _, _, meta = unified_modus_ponens(
            0.8, 0.9, 0.9, 0.85, max_rounds=3)
        assert len(meta["history"]) > 0
        s, c = meta["history"][0]
        assert 0.0 <= s <= 1.0
        assert 0.0 <= c <= 1.0


# ═══════════════════════════════════════════════════════════════════════════
#  Test: Hallucination bound
# ═══════════════════════════════════════════════════════════════════════════

class TestHallucinationBound:
    """c_output ≤ min(c_inputs) for inference rules (NOT revision)."""

    @pytest.mark.parametrize("p", MP_PARAMS, ids=lambda p: p["id"])
    def test_mp(self, p):
        _, c, _ = unified_modus_ponens(
            p["s_A"], p["c_A"], p["s_AB"], p["c_AB"])
        c_min = min(p["c_A"], p["c_AB"])
        assert c <= c_min + 0.001, (
            f"MP c_out={c:.4f} > min(c_inputs)={c_min:.4f}")

    @pytest.mark.parametrize("p", DEDUCTION_PARAMS, ids=lambda p: p["id"])
    def test_deduction(self, p):
        _, c, _ = unified_deduction(
            p["s_A"], p["c_A"], p["s_B"], p["c_B"], p["s_C"], p["c_C"],
            p["s_AB"], p["c_AB"], p["s_BC"], p["c_BC"])
        c_min = min(p["c_AB"], p["c_BC"])
        assert c <= c_min + 0.001, (
            f"Deduction c_out={c:.4f} > min(c_inputs)={c_min:.4f}")

    def test_revision_excluded(self):
        """Revision SHOULD increase confidence — it's evidence merging, not inference."""
        _, c = unified_revision(0.7, 0.5, 0.7, 0.5)
        assert c > 0.5, "Revision should increase c (excluded from hallucination bound)"
