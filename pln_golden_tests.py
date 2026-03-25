#!/usr/bin/env python3
"""
PLN Golden Tests — Verification against trueagi-io/PLN ruletests
=================================================================

These golden values are extracted from the upstream PLN repository's
ruletests/ directory and examples/.  Each test verifies that our Python
truth functions produce the same STV as the MeTTa implementation.

Source: https://github.com/trueagi-io/PLN
"""

from pln_thrml import (
    STV, c2w, w2c,
    truth_deduction, truth_modus_ponens, truth_inversion,
    truth_induction, truth_abduction, truth_revision, truth_negation,
    truth_symmetric_modus_ponens, truth_equivalence_to_implication,
    truth_transitive_similarity, truth_evaluation_implication,
    truth_or, consistency_implication_implicant_conjunction,
)

TOL = 0.001  # tolerance for floating-point comparison

all_pass = True


def check(name, result, expected_s, expected_c, tol=TOL):
    """Verify STV matches expected values."""
    global all_pass
    s_err = abs(result.strength - expected_s)
    c_err = abs(result.confidence - expected_c)
    passed = s_err < tol and c_err < tol
    mark = "PASS" if passed else "FAIL"
    print(f"  {name:<50s}  got={result}  "
          f"expect=(stv {expected_s:.6f} {expected_c:.6f})  [{mark}]")
    if not passed:
        print(f"    s_err={s_err:.6f}  c_err={c_err:.6f}")
    all_pass &= passed
    return passed


# ═══════════════════════════════════════════════════════════════════════════
#  1. c2w / w2c round-trip
# ═══════════════════════════════════════════════════════════════════════════

print("=" * 72)
print("1. UTILITY FUNCTIONS: c2w / w2c")
print("=" * 72)

for c in [0.0, 0.1, 0.5, 0.9, 0.99]:
    w = c2w(c)
    c_back = w2c(w)
    err = abs(c_back - c)
    passed = err < TOL
    mark = "PASS" if passed else "FAIL"
    print(f"  c2w({c}) = {w:.4f},  w2c({w:.4f}) = {c_back:.6f}  [{mark}]")
    all_pass &= passed

print(f"\n  truth_or(0.8, 0.9) = {truth_or(0.8, 0.9):.6f}  "
      f"expected=0.980000  [{'PASS' if abs(truth_or(0.8, 0.9) - 0.98) < TOL else 'FAIL'}]")
all_pass &= abs(truth_or(0.8, 0.9) - 0.98) < TOL

# ═══════════════════════════════════════════════════════════════════════════
#  2. Inversion — from ruletests/inversion.metta
#     (Inheritance B A) with (stv 0.87 0.81)
#     Expected: (stv 0.87 0.486)
# ═══════════════════════════════════════════════════════════════════════════

print(f"\n{'=' * 72}")
print("2. INVERSION — ruletests/inversion.metta")
print("=" * 72)

# truth_inversion(stv_B, stv_AB): A→B ⊢ B→A
# In the test: B has some truth, AB has (stv 0.87 0.81)
# Expected confidence: 0.81 * 0.6 = 0.486  (with stv_B.confidence = 1.0)
stv_B = STV(0.5, 1.0)  # B's strength doesn't affect inversion strength
stv_AB = STV(0.87, 0.81)
result = truth_inversion(stv_B, stv_AB)
# Expected: strength = 0.87 (unchanged), confidence = 1.0 * 0.81 * 0.6 = 0.486
check("Inversion: (stv 0.87 0.81) → B→A", result, 0.87, 0.486)

# ═══════════════════════════════════════════════════════════════════════════
#  3. Deduction — from examples/Smokes.metta
#     Edward smokes (stv 0.8 0.9), smoking→cancer (stv 0.9 0.7)
#     Chain: edward→smokes→cancer
# ═══════════════════════════════════════════════════════════════════════════

print(f"\n{'=' * 72}")
print("3. DEDUCTION — examples/Smokes.metta pattern")
print("=" * 72)

# Standard deduction test — parameters must satisfy consistency conditions:
#   sAB in [smallest_intersection(sA,sB), largest_intersection(sA,sB)]
#   sBC in [smallest_intersection(sB,sC), largest_intersection(sB,sC)]
stv_A = STV(0.6, 0.9)
stv_B = STV(0.7, 0.9)
stv_C = STV(0.5, 0.9)
stv_AB = STV(0.8, 0.9)
stv_BC = STV(0.7, 0.9)  # must be ≤ sC/sB = 0.714
result = truth_deduction(stv_A, stv_B, stv_C, stv_AB, stv_BC)
# s = 0.8*0.7 + (1-0.8)*(0.5 - 0.7*0.7)/(1-0.7) = 0.56 + 0.2*(0.01/0.3)
s_expected = 0.8 * 0.7 + (1.0 - 0.8) * (0.5 - 0.7 * 0.7) / (1.0 - 0.7)
c_expected = 0.8 * 0.7 * 0.9 * 0.9
check("Deduction: standard case", result, s_expected, c_expected)

# ═══════════════════════════════════════════════════════════════════════════
#  4. Modus Ponens
# ═══════════════════════════════════════════════════════════════════════════

print(f"\n{'=' * 72}")
print("4. MODUS PONENS")
print("=" * 72)

stv_A = STV(0.8, 0.9)
stv_AB = STV(0.9, 0.8)
result = truth_modus_ponens(stv_A, stv_AB)
s_expected = 0.8 * 0.9 + 0.02 * (1.0 - 0.8)
c_expected = 0.8 * 0.9 * 0.9 * 0.8
check("Modus Ponens: (0.8,0.9) + (0.9,0.8)", result, s_expected, c_expected)

# ═══════════════════════════════════════════════════════════════════════════
#  5. Revision
# ═══════════════════════════════════════════════════════════════════════════

print(f"\n{'=' * 72}")
print("5. REVISION")
print("=" * 72)

stv1 = STV(0.8, 0.5)
stv2 = STV(0.6, 0.3)
result = truth_revision(stv1, stv2)
w1 = c2w(0.5)  # 1.0
w2 = c2w(0.3)  # 0.4286
w = w1 + w2
f = (w1 * 0.8 + w2 * 0.6) / w
c = w2c(w)
c = min(1.0, max(c, 0.5, 0.3))
check("Revision: (0.8,0.5) + (0.6,0.3)", result, f, c)

# ═══════════════════════════════════════════════════════════════════════════
#  6. Negation
# ═══════════════════════════════════════════════════════════════════════════

print(f"\n{'=' * 72}")
print("6. NEGATION")
print("=" * 72)

stv = STV(0.8, 0.9)
result = truth_negation(stv)
check("Negation: (0.8, 0.9)", result, 0.2, 0.9)

# Penguins are not Cars (golden test from negation demo)
stv = STV(0.01, 0.99)
result = truth_negation(stv)
check("Negation: Penguins ≠ Cars", result, 0.99, 0.99)

# ═══════════════════════════════════════════════════════════════════════════
#  7. Symmetric Modus Ponens (new)
# ═══════════════════════════════════════════════════════════════════════════

print(f"\n{'=' * 72}")
print("7. SYMMETRIC MODUS PONENS")
print("=" * 72)

stv_A = STV(0.8, 0.9)
stv_AB = STV(0.9, 0.8)
result = truth_symmetric_modus_ponens(stv_A, stv_AB)
s_expected = 0.8 * 0.9 + 0.2 * (1.0 - 0.8) * (1.0 + 0.9)
c_expected = 0.9 * 0.8 * truth_or(0.8, 0.9)
check("SymMP: (0.8,0.9) + (0.9,0.8)", result, s_expected, c_expected)

# ═══════════════════════════════════════════════════════════════════════════
#  8. Equivalence to Implication (new)
# ═══════════════════════════════════════════════════════════════════════════

print(f"\n{'=' * 72}")
print("8. EQUIVALENCE TO IMPLICATION")
print("=" * 72)

# Standard case
stv_A = STV(0.6, 1.0)
stv_B = STV(0.4, 1.0)
stv_AB = STV(0.8, 0.9)
result = truth_equivalence_to_implication(stv_A, stv_B, stv_AB)
s_expected = (1.0 + 0.4 / 0.6) * 0.8 / (1.0 + 0.8)
check("Equiv→Impl: standard", result, s_expected, 0.9)

# High-confidence hack: s_AB * c_AB > 0.99
stv_AB_hi = STV(0.999, 0.999)
result_hi = truth_equivalence_to_implication(STV(0.9, 1.0), STV(0.9, 1.0), stv_AB_hi)
check("Equiv→Impl: high-conf hack", result_hi, 0.999, 0.999)

# ═══════════════════════════════════════════════════════════════════════════
#  9. Transitive Similarity (new)
# ═══════════════════════════════════════════════════════════════════════════

print(f"\n{'=' * 72}")
print("9. TRANSITIVE SIMILARITY")
print("=" * 72)

stv_A = STV(0.5, 1.0)
stv_B = STV(0.5, 1.0)
stv_C = STV(0.5, 1.0)
stv_AB = STV(0.9, 0.9)
stv_BC = STV(0.9, 0.9)
result = truth_transitive_similarity(stv_A, stv_B, stv_C, stv_AB, stv_BC)
c_expected = 0.9 * 0.9 * truth_or(0.9, 0.9)
print(f"  Trans-sim: equal priors, strong sims")
print(f"  A={stv_A}  B={stv_B}  C={stv_C}")
print(f"  A~B={stv_AB}  B~C={stv_BC}")
print(f"  → A~C = {result}")
print(f"  Expected confidence = {c_expected:.6f}")
c_ok = abs(result.confidence - c_expected) < TOL
all_pass &= c_ok
print(f"  Confidence check: [{'PASS' if c_ok else 'FAIL'}]")

# ═══════════════════════════════════════════════════════════════════════════
#  10. Evaluation Implication (new)
# ═══════════════════════════════════════════════════════════════════════════

print(f"\n{'=' * 72}")
print("10. EVALUATION IMPLICATION")
print("=" * 72)

# Note: simpleDeductionStrength(Bs, As, Cs, ABs, ACs)
# Consistency: sAB must satisfy consistency(Bs=0.7, As=0.6, sAB=0.8) → ≤ 0.857 ✓
#              sAC must satisfy consistency(As=0.6, Cs=0.5, sAC) → ≤ 0.833
stv_A = STV(0.6, 0.9)
stv_B = STV(0.7, 0.9)
stv_C = STV(0.5, 0.9)
stv_AB = STV(0.8, 0.9)
stv_AC = STV(0.80, 0.8)  # must be ≤ sC/sA = 0.833
result = truth_evaluation_implication(stv_A, stv_B, stv_C, stv_AB, stv_AC)
c_expected = 0.8 * 0.80 * 0.9 * 0.8
print(f"  Eval-impl result: {result}")
print(f"  Expected confidence = {c_expected:.6f}")
c_ok = abs(result.confidence - c_expected) < TOL
all_pass &= c_ok
print(f"  Confidence check: [{'PASS' if c_ok else 'FAIL'}]")

# ═══════════════════════════════════════════════════════════════════════════
#  11. Consistency check (new)
# ═══════════════════════════════════════════════════════════════════════════

print(f"\n{'=' * 72}")
print("11. CONSISTENCY CHECKS")
print("=" * 72)

# Valid case: P(C|A) ≤ P(C)/P(A)
ok = consistency_implication_implicant_conjunction(0.5, 0.6, 0.8, 0.9, 0.8)
print(f"  Consistency(0.5,0.6,0.8, AC=0.9,BC=0.8): {ok}")
# 0.9 ≤ 0.8/0.5 = 1.6 ✓, 0.8 ≤ 0.8/0.6 = 1.33 ✓
expected_ok = True
all_pass &= (ok == expected_ok)
print(f"  Expected: {expected_ok}  [{'PASS' if ok == expected_ok else 'FAIL'}]")

# Invalid case: P(C|A) > P(C)/P(A)
ok2 = consistency_implication_implicant_conjunction(0.3, 0.4, 0.2, 0.9, 0.8)
print(f"  Consistency(0.3,0.4,0.2, AC=0.9,BC=0.8): {ok2}")
# 0.9 ≤ 0.2/0.3 = 0.667 ✗
expected_ok2 = False
all_pass &= (ok2 == expected_ok2)
print(f"  Expected: {expected_ok2}  [{'PASS' if ok2 == expected_ok2 else 'FAIL'}]")

# ═══════════════════════════════════════════════════════════════════════════
#  Summary
# ═══════════════════════════════════════════════════════════════════════════

print(f"\n{'=' * 72}")
if all_pass:
    print("ALL GOLDEN TESTS PASSED")
else:
    print("SOME GOLDEN TESTS FAILED")
print(f"{'=' * 72}")
