#!/usr/bin/env python3
"""
PLN Negation on a Thermodynamic Factor Graph
=============================================

Rule:  ¬A
       Negate a proposition's truth value.

lib_pln.metta Truth_Negation:
    strength:   1 - s
    confidence: c  (unchanged)

Factor graph: single node with bias W = [log(s), log(1-s)]
(swapped from positive case).
"""

from pln_thrml import (
    STV, truth_negation, make_prior_factor,
    run_sampling, estimate_marginal, N_CATS,
)
from thrml.block_management import Block
from thrml.block_sampling import BlockGibbsSpec
from thrml.pgm import CategoricalNode
from thrml.models.discrete_ebm import CategoricalGibbsConditional
from thrml.factor import FactorSamplingProgram

# ═══════════════════════════════════════════════════════════════════════════
#  Test cases
# ═══════════════════════════════════════════════════════════════════════════

test_cases = [
    (STV(0.90, 0.9), "High strength"),
    (STV(0.50, 0.9), "Coin-flip"),
    (STV(0.10, 0.9), "Low strength"),
    (STV(0.99, 0.9), "Golden: Penguins are not Cars"),
]

print("=" * 72)
print("PLN NEGATION — THERMODYNAMIC VERIFICATION")
print("=" * 72)
print(f"\nRule:   ¬A")
print(f"Strength: 1 - s   Confidence: c (unchanged)")

all_pass = True

for stv_A, name in test_cases:
    stv_not_A = truth_negation(stv_A)

    # Build factor graph (full-confidence weights for strength verification)
    node = CategoricalNode()
    factor = make_prior_factor(node, stv_not_A.strength)
    free_blocks = [Block([node])]
    spec = BlockGibbsSpec(free_blocks, [])
    sampler = CategoricalGibbsConditional(N_CATS)
    prog = FactorSamplingProgram(
        gibbs_spec=spec, samplers=[sampler], factors=[factor],
        other_interaction_groups=[],
    )
    graph = dict(factors=[factor], free_blocks=free_blocks,
                 spec=spec, program=prog)
    samples = run_sampling(graph, seed=hash(name) % (2**31))
    p = estimate_marginal(samples, graph, node)

    err = abs(p - stv_not_A.strength)
    passed = err < 0.02
    all_pass &= passed
    mark = "PASS" if passed else "FAIL"

    print(f"\n  {name}")
    print(f"  A={stv_A}  →  ¬A={stv_not_A}")
    print(f"  Gibbs P(¬A=1)={p:.4f}  err={err:.4f}  [{mark}]")

    if "Golden" in name:
        ok = abs(stv_not_A.strength - 0.01) < 0.001 and abs(stv_not_A.confidence - 0.9) < 0.001
        print(f"  Golden: expected (stv 0.01 0.9)  [{'PASS' if ok else 'FAIL'}]")
        all_pass &= ok

print(f"\n{'=' * 72}")
if all_pass:
    print("ALL TESTS PASSED")
else:
    print("SOME TESTS FAILED")
print(f"{'=' * 72}")
