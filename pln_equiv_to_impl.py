#!/usr/bin/env python3
"""
PLN Equivalence-to-Implication on a Thermodynamic Factor Graph
===============================================================

Rule:  A≡B  ⊢  A→B
       Given the equivalence of A and B, derive the implication A→B.

PLN formula (lib_pln.metta Truth_equivalenceToImplication):
    If s_AB · c_AB > 0.99:
        s_impl = s_AB      (high-confidence hack)
    Else:
        s_impl = (1 + s_B/s_A) · s_AB / (1 + s_AB)
    c_impl = c_AB           (unchanged)

    This converts a symmetric relationship (Equivalence) into a directed
    one (Implication).  The formula is derived from the PLN book's
    sim2inh conversion.

Factor graph:
    Equivalence is modeled as a symmetric 2-node coupling:
        A ↔ B  (same strength in both directions)
    The asymmetric conditional P(B=1|A=1) recovered by Gibbs sampling
    should match the PLN implication strength.

    NOTE: For a symmetric graph with equal priors, P(B|A) = P(A|B) = strength.
    The PLN formula accounts for unequal priors (s_A ≠ s_B), which is why
    the conversion formula depends on the ratio s_B/s_A.
"""

from pln_thrml import (
    build_symmetric_pair, run_sampling, estimate_marginal,
    estimate_conditional, compare, STV, truth_equivalence_to_implication,
)

# ═══════════════════════════════════════════════════════════════════════════
#  Analytical: exact Bayesian conditional for symmetric pair
# ═══════════════════════════════════════════════════════════════════════════

def exact_conditional_symmetric(prior_a, prior_b, strength, background):
    """Exact P(B=1|A=1) for symmetric 2-node graph.

    Joint P(A,B) ∝ P_bias(A) · P_bias(B) · P(B|A)_fwd · P(A|B)_bwd
    Since both directions have the same CPT, the factor is:
        f(a,b) = P(a) · P(b) · CPT[a→b] · CPT[b→a]
    """
    # Enumerate all 4 joint states
    total = 0.0
    p_a1 = 0.0
    p_a1_b1 = 0.0

    for a in (0, 1):
        for b in (0, 1):
            # Priors
            pa = prior_a if a == 1 else (1.0 - prior_a)
            pb = prior_b if b == 1 else (1.0 - prior_b)
            # Forward CPT: P(B=b|A=a)
            if a == 1:
                fwd = strength if b == 1 else (1.0 - strength)
            else:
                fwd = background if b == 1 else (1.0 - background)
            # Backward CPT: P(A=a|B=b)
            if b == 1:
                bwd = strength if a == 1 else (1.0 - strength)
            else:
                bwd = background if a == 1 else (1.0 - background)

            joint = pa * pb * fwd * bwd
            total += joint
            if a == 1:
                p_a1 += joint
                if b == 1:
                    p_a1_b1 += joint

    if p_a1 < 1e-12:
        return 0.0
    return p_a1_b1 / p_a1


# ═══════════════════════════════════════════════════════════════════════════
#  Test cases
# ═══════════════════════════════════════════════════════════════════════════

# (prior_a, prior_b, equiv_strength, background, name)
test_cases = [
    (0.70, 0.70, 0.90, 0.10, "Equal priors, strong equiv"),
    (0.80, 0.60, 0.85, 0.15, "Unequal priors, high equiv"),
    (0.50, 0.50, 0.80, 0.20, "Coin-flip, moderate equiv"),
    (0.90, 0.30, 0.70, 0.20, "Very unequal priors"),
    (0.60, 0.60, 0.95, 0.05, "Equal priors, very strong equiv"),
    (0.40, 0.80, 0.75, 0.25, "Inverted priors"),
]

print("=" * 72)
print("PLN EQUIVALENCE → IMPLICATION — THERMODYNAMIC VERIFICATION")
print("=" * 72)
print(f"\nRule:   A≡B ⊢ A→B")
print(f"PLN formula: s_impl = (1 + s_B/s_A) · s_AB / (1 + s_AB)")
print(f"Factor graph: 2 nodes (A ↔ B), symmetric coupling")
print(f"Gibbs sampling recovers P(B=1|A=1) = directed implication strength")

all_pass = True

for prior_a, prior_b, equiv_s, bg, name in test_cases:
    print(f"\n{'─' * 72}")
    print(f"  {name}:  s_A={prior_a}, s_B={prior_b}, s_AB={equiv_s}, bg={bg}")

    # Exact Bayesian conditional
    exact = exact_conditional_symmetric(prior_a, prior_b, equiv_s, bg)

    # PLN formula
    stv_result = truth_equivalence_to_implication(
        STV(prior_a), STV(prior_b), STV(equiv_s, 0.9))
    pln_s = stv_result.strength

    # Build symmetric graph
    graph = build_symmetric_pair(prior_a, prior_b, equiv_s, bg)
    a_node, b_node = graph["a"], graph["b"]

    # Sample
    samples = run_sampling(graph, seed=hash(name) % (2**31))

    # Measure
    p_B_given_A = estimate_conditional(samples, graph, b_node, a_node, cond_val=1)
    p_A_given_B = estimate_conditional(samples, graph, a_node, b_node, cond_val=1)
    p_A = estimate_marginal(samples, graph, a_node)
    p_B = estimate_marginal(samples, graph, b_node)

    print(f"  Sampled: P(A)={p_A:.4f}  P(B)={p_B:.4f}")
    all_pass &= compare(f"P(B|A) [Bayesian exact]", exact, p_B_given_A)
    all_pass &= compare(f"P(A|B) [Bayesian exact]",
                        exact_conditional_symmetric(prior_b, prior_a, equiv_s, bg),
                        p_A_given_B)

    # Compare PLN formula vs Bayesian
    pln_err = abs(pln_s - exact)
    print(f"  PLN equiv→impl:  s={pln_s:.4f}  "
          f"Bayesian P(B|A)={exact:.4f}  diff={pln_err:.4f}")

print(f"\n{'=' * 72}")
if all_pass:
    print("ALL BAYESIAN TESTS PASSED")
else:
    print("SOME TESTS FAILED")
print(f"{'=' * 72}")
