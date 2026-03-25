#!/usr/bin/env python3
"""
PLN Abduction on a Thermodynamic Factor Graph
==============================================

Rule:  A→B, C→B  ⊢  A→C
       From two implications sharing the same consequent,
       infer a relationship between the antecedents.

Topology (inverted-V):
      A       C
       \\     /
         B

This is the "explaining away" topology: A and C are independent causes
of B.  Marginally, A and C are independent.  But given information about
B (or about each other through B), they become dependent.

The factor graph has two pairwise factors (A→B and C→B) whose energies
add at node B.  The Gibbs sampler correctly captures explaining away —
if A=1 already explains B=1, then P(C=1|A=1) may differ from P(C=1).

Analytical: exact joint enumeration over (A, B, C).
"""

from pln_thrml import (
    build_inv_v_graph, run_sampling, estimate_conditional, estimate_marginal,
    bayesnet_conditional_inv_v, inv_v_marginal, compare,
)

# ═══════════════════════════════════════════════════════════════════════════
#  Test cases
# ═══════════════════════════════════════════════════════════════════════════

test_cases = [
    # (s_A, s_C, s_AB, s_CB, bg_AB, bg_CB, name)
    (0.60, 0.50, 0.90, 0.85, 0.10, 0.10, "Two strong causes"),
    (0.50, 0.50, 0.80, 0.80, 0.20, 0.20, "Symmetric causes"),
    (0.70, 0.30, 0.90, 0.70, 0.15, 0.15, "Asymmetric priors"),
    (0.40, 0.60, 0.95, 0.60, 0.05, 0.20, "Asymmetric strengths"),
    (0.50, 0.50, 0.70, 0.70, 0.30, 0.30, "Moderate explaining away"),
]

print("=" * 72)
print("PLN ABDUCTION — THERMODYNAMIC VERIFICATION")
print("=" * 72)
print(f"\nRule:   A→B, C→B  ⊢  A→C")
print(f"Topology: inverted-V  A → B ← C  (explaining away)")
print(f"Analytical: exact joint enumeration")

all_pass = True

for s_A, s_C, s_AB, s_CB, bg_AB, bg_CB, name in test_cases:
    print(f"\n{'─' * 72}")
    print(f"  {name}")
    print(f"  P(A)={s_A}  P(C)={s_C}  s_AB={s_AB}  s_CB={s_CB}  "
          f"bg_AB={bg_AB}  bg_CB={bg_CB}")

    # Analytical
    p_C_given_A1 = bayesnet_conditional_inv_v(
        left_prior=s_A, right_prior=s_C,
        left_s=s_AB, right_s=s_CB,
        left_bg=bg_AB, right_bg=bg_CB,
    )
    print(f"  Analytical: P(C=1|A=1) = {p_C_given_A1:.4f}")

    # Build inverted-V
    graph = build_inv_v_graph(
        left_prior=s_A, right_prior=s_C,
        left_strength=s_AB, right_strength=s_CB,
        left_background=bg_AB, right_background=bg_CB,
    )

    # Sample
    samples = run_sampling(graph, seed=hash(name) % (2**31))

    left_node = graph["left"]     # A
    center_node = graph["center"] # B
    right_node = graph["right"]   # C

    # Measure
    gibbs_C_given_A1 = estimate_conditional(
        samples, graph, right_node, left_node, cond_val=1)
    gibbs_pA = estimate_marginal(samples, graph, left_node)
    gibbs_pC = estimate_marginal(samples, graph, right_node)

    # Expected marginal under factored model (shifted from prior by coupling)
    expected_pA = inv_v_marginal(
        left_prior=s_A, right_prior=s_C,
        left_s=s_AB, right_s=s_CB,
        left_bg=bg_AB, right_bg=bg_CB, target="left")

    all_pass &= compare(f"P(C=1|A=1) [abduction]",
                         p_C_given_A1, gibbs_C_given_A1)
    all_pass &= compare(f"P(A=1) [factored marginal]", expected_pA, gibbs_pA)

    # Show explaining-away effect
    p_C_marginal = gibbs_pC
    print(f"  Note: P(C=1)={p_C_marginal:.4f}, "
          f"P(C=1|A=1)={gibbs_C_given_A1:.4f}  "
          f"({'explaining away' if gibbs_C_given_A1 < p_C_marginal else 'shared cause boost'})")

print(f"\n{'=' * 72}")
if all_pass:
    print("ALL TESTS PASSED — inverted-V abduction verified")
else:
    print("SOME TESTS FAILED (try increasing n_samples)")
print(f"{'=' * 72}")
