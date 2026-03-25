#!/usr/bin/env python3
"""
PLN Revision on a Thermodynamic Factor Graph
=============================================

Rule:  (A <s1,c1>), (A <s2,c2>)  ⊢  (A <s_rev, c_rev>)
       Combine two independent evidence sources for the same proposition.

lib_pln.metta Truth_Revision:
    w1 = c1/(1-c1),  w2 = c2/(1-c2)
    strength:   (w1·s1 + w2·s2) / (w1+w2)    (arithmetic weighted average)
    confidence: min(1.0, max(w2c(w1+w2), c1, c2))

Thermodynamic note:
    Adding two energy factors on the same node gives GEOMETRIC combination:
        P(A=1) ∝ s1^c1 · s2^c2
    This differs from PLN's arithmetic revision.
    Both are shown for comparison.
"""

import math

from pln_thrml import (
    STV, truth_revision, c2w, w2c,
    make_prior_factor, _safe_log, run_sampling, estimate_marginal,
    N_CATS, DEFAULT_N_BATCHES, DEFAULT_SCHEDULE,
)
from thrml.block_management import Block
from thrml.block_sampling import BlockGibbsSpec
from thrml.pgm import CategoricalNode
from thrml.models.discrete_ebm import CategoricalEBMFactor, CategoricalGibbsConditional
from thrml.factor import FactorSamplingProgram
import jax.numpy as jnp


def _make_stv_prior(node, stv):
    """Prior factor with confidence-scaled weights."""
    c = stv.confidence
    s = stv.strength
    w = jnp.array([[c * _safe_log(1.0 - s), c * _safe_log(s)]])
    return CategoricalEBMFactor([Block([node])], w)


def geometric_revision(stv1, stv2):
    """What the factor graph gives: geometric pooling."""
    s1, c1 = stv1.strength, stv1.confidence
    s2, c2 = stv2.strength, stv2.confidence
    log_p1 = c1 * math.log(max(s1, 1e-10)) + c2 * math.log(max(s2, 1e-10))
    log_p0 = c1 * math.log(max(1 - s1, 1e-10)) + c2 * math.log(max(1 - s2, 1e-10))
    mx = max(log_p0, log_p1)
    p1 = math.exp(log_p1 - mx)
    p0 = math.exp(log_p0 - mx)
    return p1 / (p0 + p1)


# ═══════════════════════════════════════════════════════════════════════════
#  Test cases
# ═══════════════════════════════════════════════════════════════════════════

test_cases = [
    (STV(0.80, 0.9), STV(0.80, 0.9), "Agreement"),
    (STV(0.90, 0.5), STV(0.30, 0.5), "Disagreement"),
    (STV(0.70, 0.9), STV(0.80, 0.3), "First dominates"),
    (STV(0.70, 0.3), STV(0.80, 0.9), "Second dominates"),
    (STV(0.60, 0.1), STV(0.90, 0.1), "Low confidence both"),
]

print("=" * 72)
print("PLN REVISION — FORMULA + THERMODYNAMIC COMPARISON")
print("=" * 72)
print(f"\nPLN:  arithmetic weighted average  f = (w1·f1 + w2·f2) / (w1+w2)")
print(f"TSU:  geometric combination        P ∝ s1^c1 · s2^c2")

all_pass = True

for stv1, stv2, name in test_cases:
    print(f"\n{'─' * 72}")
    print(f"  {name}")
    print(f"  Evidence 1: {stv1}  Evidence 2: {stv2}")

    stv_rev = truth_revision(stv1, stv2)
    geo_s = geometric_revision(stv1, stv2)
    print(f"  PLN Revision:  {stv_rev}")
    print(f"  Geometric:     s={geo_s:.6f}")

    # Factor graph: single node with two confidence-scaled priors
    node = CategoricalNode()
    factors = [_make_stv_prior(node, stv1), _make_stv_prior(node, stv2)]
    free_blocks = [Block([node])]
    spec = BlockGibbsSpec(free_blocks, [])
    sampler = CategoricalGibbsConditional(N_CATS)
    prog = FactorSamplingProgram(
        gibbs_spec=spec, samplers=[sampler], factors=factors,
        other_interaction_groups=[],
    )
    graph = dict(factors=factors, free_blocks=free_blocks,
                 spec=spec, program=prog)
    samples = run_sampling(graph, seed=hash(name) % (2**31))
    p = estimate_marginal(samples, graph, node)

    err_geo = abs(p - geo_s)
    passed = err_geo < 0.02
    all_pass &= passed
    mark = "PASS" if passed else "FAIL"
    print(f"  Gibbs P(A=1)={p:.4f}  err_vs_geo={err_geo:.4f}  "
          f"err_vs_pln={abs(p - stv_rev.strength):.4f}  [{mark}]")

print(f"\n{'=' * 72}")
if all_pass:
    print("ALL TESTS PASSED (Gibbs matches geometric combination)")
else:
    print("SOME TESTS FAILED")
print(f"\nNote: PLN revision (arithmetic) ≠ Boltzmann (geometric)")
print(f"{'=' * 72}")
