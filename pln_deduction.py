#!/usr/bin/env python3
"""
PLN Deduction on a Thermodynamic Factor Graph
==============================================

Rule:  A→B, B→C  ⊢  A→C
       From two chained implications, derive the transitive implication.

PLN deduction formula (strength):
    s_AC = s_AB · s_BC  +  (1 − s_AB) · s_C0

    where s_C0 = P(C=1|B=0) is the background rate.

    This is equivalent to the full PLN formula:
        s_AC = s_AB · s_BC + (1 − s_AB) · (s_C − s_B · s_BC) / (1 − s_B)
    since (s_C − s_B · s_BC)/(1 − s_B) = s_C0 by total probability.

Factor graph (3-node chain):
    A ──── B ──── C
    bias_A + coupling_AB + coupling_BC

Block Gibbs coloring:  {A, C} and {B}  (non-adjacent → parallel updates)
"""

from pln_thrml import (
    build_chain, run_sampling, estimate_marginal, estimate_conditional,
    pln_deduction_strength, compare,
)

# ═══════════════════════════════════════════════════════════════════════════
#  Test cases
# ═══════════════════════════════════════════════════════════════════════════

test_cases = [
    # (s_A, s_AB, s_B0, s_BC, s_C0, name)
    (0.80, 0.90, 0.30, 0.85, 0.20, "Standard chain"),
    (0.50, 0.95, 0.10, 0.90, 0.15, "High-confidence chain"),
    (0.70, 0.60, 0.40, 0.70, 0.30, "Moderate chain"),
    (0.90, 0.50, 0.50, 0.50, 0.50, "No-information chain"),
    (0.30, 0.80, 0.20, 0.75, 0.10, "Low-prior, strong rules"),
]

print("=" * 72)
print("PLN DEDUCTION — THERMODYNAMIC VERIFICATION")
print("=" * 72)
print(f"\nRule:   A→B, B→C  ⊢  A→C")
print(f"Formula: s_AC = s_AB · s_BC + (1 − s_AB) · s_C0")
print(f"Factor graph: 3-node chain, block Gibbs with coloring {{A,C}}, {{B}}")

all_pass = True

for s_A, s_AB, s_B0, s_BC, s_C0, name in test_cases:
    print(f"\n{'─' * 72}")
    print(f"  {name}")
    print(f"  P(A)={s_A}  s_AB={s_AB}  s_B0={s_B0}  s_BC={s_BC}  s_C0={s_C0}")

    # Analytical
    s_AC_pln = pln_deduction_strength(s_AB, s_BC, s_C0)
    print(f"  PLN predicts: s_AC = {s_AC_pln:.4f}")

    # Build chain
    graph = build_chain(
        priors=[s_A, 0.5, 0.5],
        strengths=[s_AB, s_BC],
        backgrounds=[s_B0, s_C0],
    )
    A, B, C = graph["nodes"]

    # Sample
    samples = run_sampling(graph, seed=hash(name) % (2**31))

    # Measure
    p_AC = estimate_conditional(samples, graph, C, A, cond_val=1)
    p_AB = estimate_conditional(samples, graph, B, A, cond_val=1)
    p_BC = estimate_conditional(samples, graph, C, B, cond_val=1)

    all_pass &= compare(f"P(C=1|A=1) [deduction]", s_AC_pln, p_AC)
    all_pass &= compare(f"P(B=1|A=1) = s_AB", s_AB, p_AB)
    all_pass &= compare(f"P(C=1|B=1) = s_BC", s_BC, p_BC)

print(f"\n{'=' * 72}")
if all_pass:
    print("ALL TESTS PASSED")
else:
    print("SOME TESTS FAILED (try increasing n_samples)")
print(f"{'=' * 72}")
