#!/usr/bin/env python3
"""
PLN Inversion on a Thermodynamic Factor Graph
==============================================

Rule:  A→B  ⊢  B→A
       From an implication, derive its reverse via Bayes' rule.

Analytical (Bayes' rule):
    P(A=1|B=1) = P(B=1|A=1) · P(A=1) / P(B=1)

    where P(B=1) = s_A · s_AB + (1 − s_A) · s_B0

Key insight: the thrml factor graph encodes the *joint* distribution
P(A,B) = P(A) · P(B|A).  Gibbs sampling explores this joint, so
measuring P(A|B) requires no extra work — just condition on B=1 in
the samples.  The hardware automatically performs Bayesian inversion.
"""

from pln_thrml import (
    build_chain, run_sampling, estimate_conditional,
    pln_inversion_strength, compare,
)

# ═══════════════════════════════════════════════════════════════════════════
#  Test cases
# ═══════════════════════════════════════════════════════════════════════════

test_cases = [
    # (s_A, s_AB, s_B0, name)
    (0.80, 0.90, 0.30, "Strong A, strong rule"),
    (0.50, 0.80, 0.20, "Coin-flip prior"),
    (0.30, 0.95, 0.10, "Rare A, almost-certain rule"),
    (0.70, 0.60, 0.40, "Moderate everything"),
    (0.90, 0.70, 0.50, "Common A, moderate rule"),
    (0.20, 0.99, 0.05, "Rare A, near-deterministic rule"),
]

print("=" * 72)
print("PLN INVERSION — THERMODYNAMIC VERIFICATION")
print("=" * 72)
print(f"\nRule:   A→B  ⊢  B→A  (via Bayes' rule)")
print(f"Formula: P(A|B) = P(B|A)·P(A) / P(B)")
print(f"Key insight: Gibbs sampling on the joint automatically inverts.")

all_pass = True

for s_A, s_AB, s_B0, name in test_cases:
    print(f"\n{'─' * 72}")
    print(f"  {name}:  s_A={s_A}, s_AB={s_AB}, s_B0={s_B0}")

    # Analytical via Bayes
    s_BA_bayes = pln_inversion_strength(s_AB, s_A, s_B0)
    print(f"  Bayes predicts: P(A=1|B=1) = {s_BA_bayes:.4f}")

    # Build 2-node chain: A → B
    graph = build_chain(
        priors=[s_A, 0.5],
        strengths=[s_AB],
        backgrounds=[s_B0],
    )
    A, B = graph["nodes"]

    # Sample
    samples = run_sampling(graph, seed=hash(name) % (2**31))

    # Measure reverse conditional
    p_A_given_B1 = estimate_conditional(samples, graph, A, B, cond_val=1)

    # Also verify forward direction
    p_B_given_A1 = estimate_conditional(samples, graph, B, A, cond_val=1)

    all_pass &= compare(f"P(A=1|B=1) [inversion]", s_BA_bayes, p_A_given_B1)
    all_pass &= compare(f"P(B=1|A=1) [forward]", s_AB, p_B_given_A1)

print(f"\n{'=' * 72}")
if all_pass:
    print("ALL TESTS PASSED — Gibbs sampling automatically inverts conditionals")
else:
    print("SOME TESTS FAILED (try increasing n_samples)")
print(f"{'=' * 72}")
