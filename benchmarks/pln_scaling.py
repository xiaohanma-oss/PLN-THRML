#!/usr/bin/env python3
"""
PLN Deduction Chain — Scaling Test
===================================

Builds deduction chains of increasing length (3 to 20 nodes) and reports
how Gibbs-sampling accuracy and wall-clock time scale.

All links use uniform parameters:  strength = 0.9, background = 0.2.

Analytical baseline: iterative matrix multiply through the chain's
transition matrix to get exact P(X_last=1 | X_0=1).

Expected behavior:
  - Error stays small if enough warmup is provided (scales with chain length).
  - Block Gibbs with 2-coloring (even/odd) updates all nodes in 2 steps
    per sweep regardless of chain length — good scaling.
"""

import time
from pln_thrml import (
    build_chain, run_sampling, estimate_conditional, chain_conditional,
    compare, SamplingSchedule,
)

# ═══════════════════════════════════════════════════════════════════════════
#  Parameters
# ═══════════════════════════════════════════════════════════════════════════

STRENGTH = 0.9
BACKGROUND = 0.2
ROOT_PRIOR = 0.5
CHAIN_LENGTHS = [3, 5, 8, 10, 15, 20]

print("=" * 72)
print("PLN DEDUCTION CHAIN — SCALING TEST")
print("=" * 72)
print(f"\nUniform parameters: strength={STRENGTH}, background={BACKGROUND}")
print(f"Root prior: P(X_0=1) = {ROOT_PRIOR}")
print(f"Chain lengths: {CHAIN_LENGTHS}")
print(f"\nMeasure: P(X_last=1 | X_0=1) — transitive deduction through the chain")

results = []

for n in CHAIN_LENGTHS:
    print(f"\n{'─' * 72}")
    print(f"  Chain length: {n} nodes ({n-1} edges)")

    # Scale warmup with chain length
    warmup = 500 * max(1, n // 3)
    schedule = SamplingSchedule(n_warmup=warmup, n_samples=5000,
                                 steps_per_sample=3)

    # Analytical
    strengths = [STRENGTH] * (n - 1)
    backgrounds = [BACKGROUND] * (n - 1)
    analytical = chain_conditional(n, strengths, backgrounds)
    print(f"  Analytical P(X_{n-1}=1 | X_0=1) = {analytical:.6f}")

    # Build chain
    priors = [ROOT_PRIOR] + [0.5] * (n - 1)
    graph = build_chain(priors, strengths, backgrounds)

    # Sample (with timing)
    t0 = time.time()
    samples = run_sampling(graph, seed=n * 1000, schedule=schedule)
    elapsed = time.time() - t0

    # Measure
    X0 = graph["nodes"][0]
    Xn = graph["nodes"][-1]
    gibbs = estimate_conditional(samples, graph, Xn, X0, cond_val=1)

    err = abs(gibbs - analytical)
    passed = err < 0.02
    results.append((n, analytical, gibbs, err, elapsed, warmup, passed))

    compare(f"P(X_{n-1}=1 | X_0=1)", analytical, gibbs)
    print(f"  Time: {elapsed:.1f}s  |  Warmup: {warmup}  |  "
          f"Total samples: {200 * 5000:,}")

# ═══════════════════════════════════════════════════════════════════════════
#  Summary table
# ═══════════════════════════════════════════════════════════════════════════

print(f"\n{'=' * 72}")
print(f"  SCALING SUMMARY")
print(f"{'=' * 72}")
print(f"\n  {'N':>3s}  {'Analytical':>10s}  {'Gibbs':>10s}  "
      f"{'Error':>8s}  {'Time(s)':>8s}  {'Warmup':>7s}  {'Status':>6s}")
print(f"  {'─'*3}  {'─'*10}  {'─'*10}  {'─'*8}  {'─'*8}  {'─'*7}  {'─'*6}")

all_pass = True
for n, ana, gib, err, t, wu, ok in results:
    mark = "PASS" if ok else "FAIL"
    all_pass &= ok
    print(f"  {n:3d}  {ana:10.6f}  {gib:10.6f}  {err:8.4f}  "
          f"{t:8.1f}  {wu:7d}  {mark:>6s}")

print()
if all_pass:
    print("ALL CHAIN LENGTHS PASSED — thermodynamic deduction scales well")
else:
    print("SOME LENGTHS EXCEEDED 2% TOLERANCE (try more warmup/samples)")

# ── Observations ──
print(f"\nObservations:")
if len(results) >= 2:
    t_short = results[0][4]
    t_long = results[-1][4]
    n_short = results[0][0]
    n_long = results[-1][0]
    print(f"  - {n_short}-node chain: {t_short:.1f}s  →  "
          f"{n_long}-node chain: {t_long:.1f}s")
    print(f"  - Block Gibbs 2-coloring: even/odd nodes update in parallel")
    print(f"  - Each Gibbs sweep touches all nodes in 2 steps (constant)")
    print(f"  - Warmup scales linearly with chain length for good mixing")
print(f"{'=' * 72}")
