#!/usr/bin/env python3
"""
PLN Modus Ponens on a Thermodynamic Factor Graph
=================================================

Rule:  A, A→B  ⊢  B
       Given the truth of A and the implication A→B, derive the truth of B.

PLN formula (strength):
    s_B = s_A · s_AB + ε · (1 − s_A)

    where ε = 0.02 is the "leak" probability — even when A is false,
    B can still be true with this small background rate.

Factor graph (2 nodes):
    VariableAtom A  →  CategoricalNode (p-bit)
    VariableAtom B  →  CategoricalNode (p-bit)
    bias_A:        W = [log(1−s_A), log(s_A)]
    coupling_AB:   W[A=0] = [log(1−ε), log(ε)]
                   W[A=1] = [log(1−s_AB), log(s_AB)]

Gibbs sampling recovers P(B=1) which should match the PLN formula.
"""

from pln_thrml import (
    build_chain, run_sampling, estimate_marginal, estimate_conditional,
    pln_modus_ponens_strength, compare, DEFAULT_EPSILON,
)

# ═══════════════════════════════════════════════════════════════════════════
#  Test cases
# ═══════════════════════════════════════════════════════════════════════════

test_cases = [
    # (s_A, s_AB, name)
    (0.80, 0.90, "Strong prior, strong rule"),
    (0.50, 0.95, "Coin-flip prior, strong rule"),
    (0.30, 0.70, "Low prior, moderate rule"),
    (0.90, 0.50, "High prior, weak rule"),
    (0.95, 0.99, "Near-certain both"),
    (0.10, 0.80, "Rare antecedent"),
]

eps = DEFAULT_EPSILON  # 0.02

print("=" * 72)
print("PLN MODUS PONENS — THERMODYNAMIC VERIFICATION")
print("=" * 72)
print(f"\nRule:   A, A→B ⊢ B")
print(f"Formula: s_B = s_A · s_AB + {eps} · (1 − s_A)")
print(f"Factor graph: 2 nodes (A, B), bias on A, coupling A→B")
print(f"Background rate P(B=1|A=0) = {eps}")

all_pass = True

for s_A, s_AB, name in test_cases:
    print(f"\n{'─' * 72}")
    print(f"  {name}:  s_A={s_A}, s_AB={s_AB}")

    # Analytical
    s_B_pln = pln_modus_ponens_strength(s_A, s_AB, eps)

    # Build 2-node chain: A → B
    graph = build_chain(
        priors=[s_A, 0.5],          # B's prior doesn't matter (overridden by coupling)
        strengths=[s_AB],
        backgrounds=[eps],
    )
    A, B = graph["nodes"]

    # Sample
    samples = run_sampling(graph, seed=hash(name) % (2**31))

    # Measure
    p_B = estimate_marginal(samples, graph, B)
    p_B_given_A1 = estimate_conditional(samples, graph, B, A, cond_val=1)
    p_B_given_A0 = estimate_conditional(samples, graph, B, A, cond_val=0)

    all_pass &= compare(f"P(B=1) [modus ponens]", s_B_pln, p_B)
    all_pass &= compare(f"P(B=1|A=1) = s_AB", s_AB, p_B_given_A1)
    all_pass &= compare(f"P(B=1|A=0) = eps", eps, p_B_given_A0)

print(f"\n{'=' * 72}")
if all_pass:
    print("ALL TESTS PASSED")
else:
    print("SOME TESTS FAILED (try increasing n_samples)")
print(f"{'=' * 72}")
