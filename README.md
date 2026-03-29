# PLN-THRML

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Version 0.2.0](https://img.shields.io/badge/version-0.2.0-green.svg)](pyproject.toml)

> Compile PLN inference rules to thermodynamic factor graphs and run them
> via Gibbs sampling — bridging [Hyperon/PLN](https://github.com/trueagi-io/PLN)
> and [Extropic/thrml](https://github.com/extropic-ai/thrml).

## Table of Contents

- [Overview](#overview)
- [How it works](#how-it-works)
- [Architecture mapping](#architecture-mapping)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Results](#results)
- [What this means for Hyperon](#what-this-means-for-hyperon)
- [Project structure](#project-structure)
- [Not yet covered](#not-yet-covered)
- [Contributing](#contributing)
- [References](#references)

## Overview

PLN treats inference as probability propagation; the Thermodynamic Sampling
Unit (TSU) treats computation as energy minimisation. The Boltzmann
distribution `P(x) ∝ e^{−ℰ(x)}` connects the two: the log of each PLN
conditional probability is directly a factor-graph weight.

The central question is whether this compilation preserves PLN's full
semantics — not just strength, but also confidence. This repo shows that it
does: all 11 PLN inference rules are compiled via Beta discretization and
verified against PLN's analytical formulas.

## How it works

1. **Parameterize** — PLN `(stv s c)` → Beta(α, β) with mean = s
2. **Build graph** — discretize each Beta distribution into K=16 bins (`CategoricalNode`); each implication becomes a K×K weight table (log of the discretized conditional Beta)
3. **Sample** — Block Gibbs sampling (50 batches × 2,000 samples)
4. **Recover** — moment-match the posterior histogram → `(strength, confidence)`

See [docs/beta-discretization.md](docs/beta-discretization.md) for details.

## Architecture mapping

| PLN concept | thrml construct | Extropic hardware |
|---|---|---|
| Proposition (VariableAtom) | `CategoricalNode` (K=16 bins) | p-bit cluster (multi-bit) |
| Prior P(A) | Unary `CategoricalEBMFactor` (Beta prior) | Bias field on p-bits |
| Implication A→B (strength s) | Pairwise `SquareCategoricalEBMFactor` (K×K) | Coupling matrix between clusters |
| TruthValue (strength, confidence) | Beta posterior → moment-matching | Posterior distribution readout |
| Inference rule | Block Gibbs sampling | Thermal equilibration |
| Deduction chain | Chain factor graph | Pipeline of coupled clusters |
| Induction (V-shape) | Star factor graph | Hub-and-spoke topology |
| Abduction (inverted-V) | Collider factor graph | Explaining-away circuit |
| Large knowledge graph | Block-diagonal decomposition | Block-local Gibbs + BP messages |

## Installation

```bash
git clone --recurse-submodules https://github.com/mafeifei666666/PLN-THRML.git
cd PLN-THRML
pip install -e .                          # core only (thrml + jax)
pip install -e ".[metta]"                 # + MeTTa bridge (requires hyperon)
pip install -e ".[dev]"                   # + pytest for running tests
```

> The [trueagi-io/PLN](https://github.com/trueagi-io/PLN) submodule provides
> test baselines.  If you cloned without `--recurse-submodules`, run
> `git submodule update --init`.

## Quick start

### Python API (no MeTTa dependency)

```python
from pln_thrml_beta import build_beta_chain, sample_and_measure

# Modus ponens: A → B, P(A)=0.8, s(A→B)=0.9
graph = build_beta_chain(
    priors=[0.8, 0.5], confidences=[0.9, 0.01],
    strengths=[0.9], impl_confidences=[0.85],
    backgrounds=[0.02],
)
s, c = sample_and_measure(graph, target_node=graph["nodes"][1])
print(f"P(B) = (stv {s:.4f} {c:.4f})")   # ≈ (stv 0.73 0.76)
```

### MeTTa bridge (requires `hyperon`)

```python
from hyperon import MeTTa
from metta import register_all

metta = MeTTa()
register_all(metta)

results = metta.run('''
    (A (stv 0.8 0.9))
    ((Implication A B) (stv 0.9 0.85))
    !(thrml-modus-ponens! (A B (stv 0.8 0.9) (stv 0.9 0.85)))
''')
# => (stv 0.7242 0.5508)
```

### Run tests

```bash
pytest tests/ -v                          # all 11 rules covered
pytest -m slow -v                        # scalability tests (trueagi-io/PLN examples)
```

## Results

All 11 PLN rules compiled and verified.  Summary of maximum strength errors:

| Rule | Max Error | Rule | Max Error |
|------|-----------|------|-----------|
| Modus Ponens | 0.044 | Negation | 0.000 |
| Deduction | 0.073 | Revision | 0.015 |
| Inversion | 0.038 | Sym. Modus Ponens | 0.005 |
| Induction | 0.046 | Equiv→Impl | 0.025 |
| Abduction | 0.130 | Trans. Similarity | 0.050 |
| Eval. Implication | 0.061 | | |

Most rules match within 5%.  Abduction/Induction diverge because PLN uses
closed-form approximations while the factor graph computes the exact joint
posterior.  Inversion gives exact Bayesian P(A|B) vs PLN's heuristic.

Full per-rule tables and divergence analysis: [docs/results.md](docs/results.md)

### Effect of K on accuracy

Reducing K from 16 to 4 (for TSU hardware deployment) introduces additional
discretization error.  Measured on Modus Ponens (s_A=0.8, s_AB=0.9, PLN
analytical strength = 0.724) and single-node prior recovery:

| K | MP Δs vs PLN | Prior Δs (worst) | Prior Δc (worst) | Couplings/impl |
|---|---|---|---|---|
| 16 | 0.008 | 0.001 | 0.013 | 256 |
| 8 | 0.011 | 0.008 | 0.035 | 64 |
| 4 | 0.024 | 0.019 | 0.067 | 16 |

K=4 strength error is 0.024 — well within the 0.05 tolerance used for K=16
tests.  Confidence recovery degrades more (Δc up to 0.067) because
moment-matching from 4 bins has less information.

### Block-diagonal vs full-graph accuracy

Comparing block-diagonal inference (K=4, inter-block BP messages) against
full-graph inference (K=4, all nodes in one graph):

| Topology | Full s | Block-diag s | Δs | Iterations | Converged |
|---|---|---|---|---|---|
| 3-chain A→B→C | 0.660 | 0.642 | 0.018 | 3 | ✓ |
| 5-chain A→…→E | 0.252 | 0.183 | 0.069 | 7 | ✓ |
| Diamond A→{B,C}→D | 0.129 | 0.222 | 0.094 | 20 | ✗ |

Tree-structured graphs converge quickly with small error.  The diamond
(cyclic block graph) shows the known limitation of loopy BP — damped
iterations bound but do not eliminate the error.

## What this means for Hyperon

1. **Complete Q_tv on THRML**: The Hyperon whitepaper (2025) defines PLN
   inference as message-passing on a product quantale Q_logic × Q_tv.
   This project demonstrates that the entire Q_tv component — both the
   quantale product ⊗ (evidence combination at factors) and the quantale
   sum ⊕ (marginalization at variables) — can be compiled to thrml factor
   graphs (log-probability weights, via `P(x) ∝ e^{−ℰ(x)}`) and executed through
   Gibbs sampling.  All 11 PLN
   rules verified end-to-end across multiple parameter sets constitute the
   evidence.  Q_logic (rule selection, structure discovery) remains with
   CPU/GPU; all truth-value computation has a complete path to
   thermodynamic hardware.

2. **Direct compilation path**: MeTTa PLN rules can be compiled to thrml
   factor graphs (log-probability weights), then executed on Extropic's
   TSU.  No approximation is introduced — the factor graph encodes the
   exact same joint distribution that PLN reasons over.

3. **Automatic inversion**: The factor graph encodes the joint P(A,B),
   so Gibbs sampling recovers *both* P(B|A) and P(A|B) without separate
   inversion rules.  Hardware naturally performs Bayes' rule.

4. **Composability**: Rules combine by adding factors to the graph.
   Deduction chains, V-shapes, and collider topologies all work with the
   same compilation transform.

5. **Scaling**: Block Gibbs with graph coloring enables parallel updates.
   A 20-node deduction chain runs in 1.5s on CPU; a 50-node chain
   completes within 60s.  On a TSU, thermal equilibration would be
   near-instantaneous.  Verified by scalability tests (`pytest -m slow`)
   covering [trueagi-io/PLN](https://github.com/trueagi-io/PLN) examples — DeductionRevision (diamond DAG),
   FlyingRaven (conflicting paths with negation), Smokes (social network
   propagation), and RavenInduction (instance-to-class generalization).

6. **Hardware outlook**: Extropic's Z1 chip (early 2026) provides 250,000
   p-bits.  Each K=16 proposition requires 4 p-bits (62,500 nodes upper
   bound), but each pairwise implication expands to 16×16 = 256 binary
   couplings under sum-of-spins embedding, and each p-bit has only ~12
   physical connections — so connectivity, not p-bit count, is the
   binding constraint.  Native pdit support (validated on X0) would
   treat each K-state variable as one hardware unit and substantially
   relax this limit.

7. **Block-diagonal architecture**: `block_diagonal.py` addresses the TSU
   connectivity bottleneck by partitioning large graphs into blocks of 2–4
   propositions with K=4 (16 couplings per implication, fits 12-connection
   budget).  Block-internal inference uses exact Gibbs sampling; block-boundary
   influence propagates via BP-style messages through implication factors.
   Tree-structured block graphs converge exactly (Pearl 1988); cyclic graphs
   use damped loopy BP.  Z1 capacity estimate: ~1,225 simultaneous propositions
   (4,900 p-bits ÷ 4 p-bits/prop), ~60,000 with time-division multiplexing.

## Project structure

```
pln_thrml_beta.py          Beta factor graph engine (builds, samples, recovers stv)
block_diagonal.py          Block-diagonal decomposition + loopy BP for large/cyclic graphs
vendor/PLN/                trueagi-io/PLN (git submodule) — test baselines
metta/                     MeTTa integration layer (optional, requires hyperon)
  __init__.py              Entry point — exports register_all()
  atoms.py                 Atom extraction from MeTTa space
  ops/
    rules.py               All 11 rules (declarative table + generic factory, including revision & negation)
  declarations/
    pln_types.metta        Type declarations (stv, Implication, Similarity, etc.)
tests/
  conftest.py              Shared fixtures and tolerance constants (strength ±0.05, confidence ±0.15)
  test_factor_graph.py     Factor graph engine unit tests (K=4/8/16 parametrized)
  test_metta.py            All rules verified end-to-end via MeTTa
  test_block_diagonal.py   Block-diagonal decomposition tests (partitioning, BP, convergence)
  test_scale.py            Scalability tests from trueagi-io/PLN examples (pytest -m slow)
docs/
  results.md               Full per-rule results tables
  pln-formulas.md          PLN truth-value confidence formulas
  beta-discretization.md   Beta-discretized approach details
  interactive_overview.html  Interactive visualisation (4 tabs: pipeline, 11 rules, energy, block-diagonal)
```

## Not yet covered

- **Cyclic inference (full-graph)**: `block_diagonal.py` handles cyclic *block*
  graphs via damped loopy BP; full-graph cyclic Gibbs (without block
  decomposition) is not yet implemented.
- **EvidenceID / StampDisjoint**: Evidence tracking to prevent double-counting
  during revision (PLN uses `StampDisjoint` and `StampConcat`).
- **PLN.Derive**: Priority-queue based iterative inference engine with
  bounded belief buffer.
- **ECAN attention**: Attention allocation for prioritizing which
  subgraphs to sample first.
- **Continuous-valued nodes**: Extending beyond discrete propositions.

See [docs/future-work/future-work.md](docs/future-work/future-work.md) for
detailed analysis including native pdit support, low-rank factorization,
geodesic controller scheduling, and other block-diagonal extensions.

## Contributing

See [Installation](#installation) for setup, then:

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

## References

1. Goertzel, Ikle, Goertzel, Heljakka.
   *Probabilistic Logic Networks* (Springer, 2008).

2. Extropic AI. *thrml: Thermodynamic Hypergraphical Model Library*.
   https://github.com/extropic-ai/thrml

3. TrueAGI. *PLN (Probabilistic Logic Networks)*.
   https://github.com/trueagi-io/PLN

4. TrueAGI. *PLN Experimental (mathematical foundations)*.
   https://github.com/trueagi-io/pln-experimental

5. Goertzel, B. *Hyperon for AGI ⇒ ASI, Whitepaper 2025 (Deepish-Dive Version)*.
   October 15, 2025.

6. Jelincic, Lockwood, Garlapati, Schillinger, Chuang, Verdon, McCourt.
   *An efficient probabilistic hardware architecture for diffusion-like models*.
   arXiv:2510.23972, 2025.

7. Goertzel, B. *Evidence Is to Logic What Energy Is to Physics*.
   Substack, March 2026.

8. Tran, Mota, Garcez.
   *Reasoning in Neurosymbolic AI (Logical Boltzmann Machines)*.
   arXiv:2505.20313, 2025.

9. Goertzel, B. *Quantum Logic Networks: PLN-Style Inference in the
   Operator Evidence Algebra*. March 2026.
   §9.2 block-diagonal hardware regime (d=4–16) directly motivates
   this project's block-diagonal architecture.

## License

[MIT](LICENSE) — Copyright (c) 2025 Xiaohan Ma
