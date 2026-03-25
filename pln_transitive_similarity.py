#!/usr/bin/env python3
"""
PLN Transitive Similarity on a Thermodynamic Factor Graph
==========================================================

Rule:  A~B, B~C  ⊢  A~C
       From two similarity links sharing a middle term,
       infer the similarity between the endpoints.

PLN formula (lib_pln.metta Truth_transitiveSimilarity):
    Strength via TransitiveSimilarityStrength — a complex formula that:
    1. Converts similarities to directed implications via sim2inh:
         T1 = (1 + s_B/s_A) · s_AB / (1 + s_AB)    (A→B direction)
         T2 = (1 + s_C/s_B) · s_BC / (1 + s_BC)    (B→C direction)
         T3 = (1 + s_B/s_C) · s_BC / (1 + s_BC)    (C→B direction)
         T4 = (1 + s_A/s_B) · s_AB / (1 + s_AB)    (B→A direction)
    2. Applies deduction in both directions (A→C and C→A)
    3. Combines via: sim = 1 / (1/fwd + 1/bwd - 1)

    Confidence: c_AB · c_BC · truth_or(s_AB, s_BC)

Topology (symmetric chain):
    A ↔ B ↔ C

    Each edge is a bidirectional coupling with the same strength in both
    directions.  The factor graph samples the full joint P(A,B,C) and
    we extract P(C=1|A=1) and P(A=1|C=1).

    KEY INSIGHT: For a symmetric graph, P(C|A) should equal P(A|C)
    (Bayesian symmetry from the symmetric topology).  The PLN heuristic
    may differ from the exact Bayesian result.
"""

from pln_thrml import (
    build_symmetric_chain, run_sampling, estimate_conditional,
    estimate_marginal, compare, transitive_similarity_strength,
)

# ═══════════════════════════════════════════════════════════════════════════
#  Exact Bayesian conditional for symmetric 3-node chain
# ═══════════════════════════════════════════════════════════════════════════

def exact_conditional_sym_chain(prior_a, s_AB, bg_AB, s_BC, bg_BC):
    """Exact P(C=1|A=1) for symmetric chain A ↔ B ↔ C.

    Joint: P(A,B,C) ∝ P(A) · CPT_AB[A,B] · CPT_BA[B,A] · CPT_BC[B,C] · CPT_CB[C,B]

    Enumerates all 8 states.
    """
    total = 0.0
    p_a1 = 0.0
    p_a1_c1 = 0.0

    for a in (0, 1):
        for b in (0, 1):
            for c in (0, 1):
                # Prior on A only
                pa = prior_a if a == 1 else (1.0 - prior_a)

                # CPT A→B
                if a == 1:
                    f_ab = s_AB if b == 1 else (1.0 - s_AB)
                else:
                    f_ab = bg_AB if b == 1 else (1.0 - bg_AB)
                # CPT B→A (symmetric)
                if b == 1:
                    f_ba = s_AB if a == 1 else (1.0 - s_AB)
                else:
                    f_ba = bg_AB if a == 1 else (1.0 - bg_AB)
                # CPT B→C
                if b == 1:
                    f_bc = s_BC if c == 1 else (1.0 - s_BC)
                else:
                    f_bc = bg_BC if c == 1 else (1.0 - bg_BC)
                # CPT C→B (symmetric)
                if c == 1:
                    f_cb = s_BC if b == 1 else (1.0 - s_BC)
                else:
                    f_cb = bg_BC if b == 1 else (1.0 - bg_BC)

                joint = pa * f_ab * f_ba * f_bc * f_cb
                total += joint
                if a == 1:
                    p_a1 += joint
                    if c == 1:
                        p_a1_c1 += joint

    if p_a1 < 1e-12:
        return 0.0
    return p_a1_c1 / p_a1


# ═══════════════════════════════════════════════════════════════════════════
#  Test cases
# ═══════════════════════════════════════════════════════════════════════════

# (prior_A, s_AB, bg_AB, s_BC, bg_BC, name)
test_cases = [
    (0.50, 0.90, 0.10, 0.85, 0.15, "Strong similarities"),
    (0.60, 0.80, 0.20, 0.80, 0.20, "Symmetric moderate"),
    (0.70, 0.95, 0.05, 0.70, 0.25, "Asymmetric strengths"),
    (0.50, 0.75, 0.25, 0.90, 0.10, "Weak AB, strong BC"),
    (0.40, 0.85, 0.15, 0.85, 0.15, "Equal edges, low prior"),
]

print("=" * 72)
print("PLN TRANSITIVE SIMILARITY — THERMODYNAMIC VERIFICATION")
print("=" * 72)
print(f"\nRule:   A~B, B~C  ⊢  A~C")
print(f"Topology: symmetric chain  A ↔ B ↔ C")
print(f"Analytical: exact joint enumeration for symmetric factor graph")

all_pass = True

for prior_A, s_AB, bg_AB, s_BC, bg_BC, name in test_cases:
    print(f"\n{'─' * 72}")
    print(f"  {name}:  s_A={prior_A}, s_AB={s_AB}, bg_AB={bg_AB}, "
          f"s_BC={s_BC}, bg_BC={bg_BC}")

    # Exact Bayesian
    exact_CA = exact_conditional_sym_chain(prior_A, s_AB, bg_AB, s_BC, bg_BC)

    # Build symmetric chain
    graph = build_symmetric_chain(
        priors=[prior_A, 0.5, 0.5],
        strengths=[s_AB, s_BC],
        backgrounds=[bg_AB, bg_BC],
    )
    A, B, C = graph["nodes"]

    # Sample
    samples = run_sampling(graph, seed=hash(name) % (2**31))

    # Measure
    p_C_given_A = estimate_conditional(samples, graph, C, A, cond_val=1)
    p_A_given_C = estimate_conditional(samples, graph, A, C, cond_val=1)
    p_A = estimate_marginal(samples, graph, A)
    p_B = estimate_marginal(samples, graph, B)
    p_C = estimate_marginal(samples, graph, C)

    print(f"  Marginals: P(A)={p_A:.4f}  P(B)={p_B:.4f}  P(C)={p_C:.4f}")

    all_pass &= compare(f"P(C|A) [Bayesian exact]", exact_CA, p_C_given_A)

    # Symmetry check: P(C|A) ≈ P(A|C) is NOT guaranteed when priors differ
    print(f"  P(C|A)={p_C_given_A:.4f}  P(A|C)={p_A_given_C:.4f}  "
          f"(asymmetry from prior)")

    # PLN transitive similarity formula comparison
    pln_s = transitive_similarity_strength(prior_A, p_B, p_C, s_AB, s_BC)
    pln_diff = abs(pln_s - exact_CA)
    print(f"  PLN trans-sim strength={pln_s:.4f}  "
          f"Bayesian P(C|A)={exact_CA:.4f}  diff={pln_diff:.4f}")

print(f"\n{'=' * 72}")
if all_pass:
    print("ALL BAYESIAN TESTS PASSED")
else:
    print("SOME TESTS FAILED")
print(f"{'=' * 72}")
