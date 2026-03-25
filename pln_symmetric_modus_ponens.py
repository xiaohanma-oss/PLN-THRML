#!/usr/bin/env python3
"""
PLN Symmetric Modus Ponens on a Thermodynamic Factor Graph
===========================================================

Rule:  A, A~B  ⊢  B   (via Similarity link)
       Given the truth of A and the similarity A~B, derive the truth of B.

PLN formula (lib_pln.metta Truth_SymmetricModusPonens):
    snotAB = 0.2
    s_B = s_A · s_AB + snotAB · (1 − s_A) · (1 + s_AB)
    c_B = c_A · c_AB · truth_or(s_A, s_AB)

    where truth_or(a, b) = 1 - (1-a)·(1-b)

    The background rate when A is false is  snotAB · (1 + s_AB),
    which is larger than the epsilon=0.02 used in directed modus ponens,
    reflecting the symmetric nature of Similarity links.

Factor graph (2 nodes):
    A → CategoricalNode (p-bit)
    B → CategoricalNode (p-bit)
    bias_A:        W = [log(1−s_A), log(s_A)]
    coupling_AB:   W[A=0] = [log(1−bg), log(bg)]    bg = snotAB·(1+s_AB)
                   W[A=1] = [log(1−s_AB), log(s_AB)]

Gibbs sampling recovers P(B=1) which should match the PLN formula.
"""

from pln_thrml import (
    build_chain, run_sampling, estimate_marginal, estimate_conditional,
    compare,
)

# ═══════════════════════════════════════════════════════════════════════════
#  Analytical formula (matches factor graph semantics)
# ═══════════════════════════════════════════════════════════════════════════

SNOTAB = 0.2  # PLN constant for symmetric link background


def symmetric_mp_strength(s_A, s_AB):
    """P(B=1) = s_A · s_AB + snotAB · (1 − s_A) · (1 + s_AB)"""
    bg = SNOTAB * (1.0 + s_AB)
    return s_A * s_AB + bg * (1.0 - s_A)


def symmetric_mp_background(s_AB):
    """P(B=1|A=0) = snotAB · (1 + s_AB)"""
    return SNOTAB * (1.0 + s_AB)


# ═══════════════════════════════════════════════════════════════════════════
#  Test cases
# ═══════════════════════════════════════════════════════════════════════════

test_cases = [
    # (s_A, s_AB, name)
    (0.80, 0.90, "Strong prior, strong similarity"),
    (0.50, 0.80, "Coin-flip prior, high similarity"),
    (0.30, 0.70, "Low prior, moderate similarity"),
    (0.90, 0.50, "High prior, weak similarity"),
    (0.95, 0.95, "Near-certain both"),
    (0.10, 0.80, "Rare antecedent"),
]

print("=" * 72)
print("PLN SYMMETRIC MODUS PONENS — THERMODYNAMIC VERIFICATION")
print("=" * 72)
print(f"\nRule:   A, A~B ⊢ B   (Similarity link)")
print(f"Formula: s_B = s_A · s_AB + 0.2 · (1 − s_A) · (1 + s_AB)")
print(f"Factor graph: 2 nodes (A, B), bias on A, coupling A→B")
print(f"Background rate P(B=1|A=0) = snotAB · (1 + s_AB)")

all_pass = True

for s_A, s_AB, name in test_cases:
    bg = symmetric_mp_background(s_AB)
    print(f"\n{'─' * 72}")
    print(f"  {name}:  s_A={s_A}, s_AB={s_AB}, bg={bg:.4f}")

    # Analytical
    s_B_pln = symmetric_mp_strength(s_A, s_AB)

    # Build 2-node chain: A → B with symmetric background
    graph = build_chain(
        priors=[s_A, 0.5],
        strengths=[s_AB],
        backgrounds=[bg],
    )
    A, B = graph["nodes"]

    # Sample
    samples = run_sampling(graph, seed=hash(name) % (2**31))

    # Measure
    p_B = estimate_marginal(samples, graph, B)
    p_B_given_A1 = estimate_conditional(samples, graph, B, A, cond_val=1)
    p_B_given_A0 = estimate_conditional(samples, graph, B, A, cond_val=0)

    all_pass &= compare(f"P(B=1) [symmetric MP]", s_B_pln, p_B)
    all_pass &= compare(f"P(B=1|A=1) = s_AB", s_AB, p_B_given_A1)
    all_pass &= compare(f"P(B=1|A=0) = bg", bg, p_B_given_A0)

print(f"\n{'=' * 72}")
if all_pass:
    print("ALL TESTS PASSED")
else:
    print("SOME TESTS FAILED (try increasing n_samples)")
print(f"{'=' * 72}")
