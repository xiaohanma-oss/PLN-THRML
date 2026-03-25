#!/usr/bin/env python3
"""
PLN Induction on a Thermodynamic Factor Graph
==============================================

Rule:  A→B, A→C  ⊢  B→C
       From two implications sharing the same antecedent,
       infer a relationship between the consequents.

Topology (V-shape):
        A
       / \\
      B   C

Analytical (exact Bayesian network marginalization):
    P(C=1|B=1) = Σ_a  P(C=1|A=a) · P(A=a|B=1)

    where P(A=a|B=1) comes from Bayes on the A→B link.

B and C are conditionally independent given A, but marginally
correlated — information about B tells us about A, which tells
us about C.  The factor graph captures this automatically.
"""

from pln_thrml import (
    build_v_graph, run_sampling, estimate_conditional,
    bayesnet_conditional_v, compare,
)

# ═══════════════════════════════════════════════════════════════════════════
#  Test cases
# ═══════════════════════════════════════════════════════════════════════════

test_cases = [
    # (s_A, s_AB, s_AB_bg, s_AC, s_AC_bg, name)
    (0.60, 0.90, 0.20, 0.85, 0.15, "Strong shared cause"),
    (0.50, 0.80, 0.10, 0.70, 0.20, "Moderate links"),
    (0.80, 0.95, 0.30, 0.60, 0.10, "Asymmetric strengths"),
    (0.40, 0.70, 0.30, 0.90, 0.25, "Weaker left, stronger right"),
    (0.70, 0.85, 0.15, 0.85, 0.15, "Symmetric links"),
]

print("=" * 72)
print("PLN INDUCTION — THERMODYNAMIC VERIFICATION")
print("=" * 72)
print(f"\nRule:   A→B, A→C  ⊢  B→C")
print(f"Topology: V-shape  B ← A → C")
print(f"Analytical: exact Bayesian network marginalization")

all_pass = True

for s_A, s_AB, s_AB_bg, s_AC, s_AC_bg, name in test_cases:
    print(f"\n{'─' * 72}")
    print(f"  {name}")
    print(f"  P(A)={s_A}  s_AB={s_AB}  bg_AB={s_AB_bg}  "
          f"s_AC={s_AC}  bg_AC={s_AC_bg}")

    # Analytical
    p_C_given_B1 = bayesnet_conditional_v(
        root_prior=s_A,
        left_s=s_AB, right_s=s_AC,
        left_bg=s_AB_bg, right_bg=s_AC_bg,
    )
    print(f"  Analytical: P(C=1|B=1) = {p_C_given_B1:.4f}")

    # Build V-shape
    graph = build_v_graph(
        root_prior=s_A,
        left_strength=s_AB, right_strength=s_AC,
        left_background=s_AB_bg, right_background=s_AC_bg,
    )

    # Sample
    samples = run_sampling(graph, seed=hash(name) % (2**31))

    # Measure P(C=1|B=1) — the "induced" conditional
    left_node = graph["left"]    # B
    right_node = graph["right"]  # C
    root_node = graph["root"]    # A

    gibbs_C_given_B1 = estimate_conditional(
        samples, graph, right_node, left_node, cond_val=1)

    # Also verify the direct conditionals
    gibbs_B_given_A1 = estimate_conditional(
        samples, graph, left_node, root_node, cond_val=1)
    gibbs_C_given_A1 = estimate_conditional(
        samples, graph, right_node, root_node, cond_val=1)

    all_pass &= compare(f"P(C=1|B=1) [induction]",
                         p_C_given_B1, gibbs_C_given_B1)
    all_pass &= compare(f"P(B=1|A=1) = s_AB", s_AB, gibbs_B_given_A1)
    all_pass &= compare(f"P(C=1|A=1) = s_AC", s_AC, gibbs_C_given_A1)

print(f"\n{'=' * 72}")
if all_pass:
    print("ALL TESTS PASSED — V-shape induction verified")
else:
    print("SOME TESTS FAILED (try increasing n_samples)")
print(f"{'=' * 72}")
