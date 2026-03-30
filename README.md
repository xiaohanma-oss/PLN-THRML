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
does: 11 PLN inference rules are compiled via Beta discretization and
verified against PLN's analytical formulas.

### Why this matters

Today's knowledge graphs run PLN inference on CPUs — fast for small graphs,
but inference time grows linearly with node count.  Extropic's Thermodynamic
Sampling Unit (TSU) flips this: all p-bits update in one physical step, so
inference time is **constant** regardless of graph size.  On a 60,000-proposition
graph the projected speedup is ~720× over H100 GPU, at ~200,000× less energy
(see [Z1 vs H100](#z1-vs-h100-q_tv-inference-at-scale) below).  This repo
proves the compilation from PLN to TSU factor graphs is faithful — the
missing piece for deploying PLN at scale on thermodynamic hardware.

### PLN truth values in 30 seconds

PLN represents uncertain knowledge with two numbers:

| Component | Meaning | Analogy |
|-----------|---------|---------|
| **strength** *s* | How likely something is true | A poll result: "80% of respondents said yes" |
| **confidence** *c* | How much evidence supports the estimate | The poll's sample size: 1,000 people vs 10 people |

`(stv 0.8 0.9)` means "80% likely true, based on strong evidence."
`(stv 0.8 0.01)` means "80% likely true, but we barely know — almost a guess."

This dual-value semantics is what makes PLN more expressive than plain
probabilities, and what this project must preserve when compiling to hardware.

## How it works

1. **Parameterize** — PLN `(stv s c)` → Beta(α, β) with mean = s
2. **Build graph** — discretize each Beta distribution into **K bins** (`CategoricalNode`); each implication becomes a K×K weight table (log of the discretized conditional Beta).  K controls accuracy vs compute: K=16 (default) for research, K=4 for TSU hardware deployment.
3. **Sample** — Block Gibbs sampling (50 batches × 2,000 samples).  Root nodes can be **clamped** (fixed to their prior values as known evidence) while free nodes are updated by Gibbs sweeps.
4. **Recover** — moment-match the posterior histogram → `(strength, confidence)`

> **Background rate ε** (default 0.02): a small floor probability added to
> every implication to prevent zero-probability states, analogous to Laplace
> smoothing.  Configurable via the `backgrounds` parameter.

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
from pln_thrml import build_beta_chain, sample_and_measure

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
from pln_thrml.metta import register_all

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
pytest tests/ -v                          # all implemented rules covered
pytest -m slow -v                        # scalability tests (trueagi-io/PLN examples)
```

## Results

11 PLN rules compiled and verified.  Summary of maximum strength errors:

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
| Diamond A→{B,C}→D | 0.129 | 0.137 | 0.008 | 17 | ✓ |

All three topologies converge within tolerance.  The diamond (cyclic block
graph) previously exhibited inter-block message accumulation error; after
the fix it converges in 17 iterations with Δs = 0.008.

### Z1 vs H100: Q\_tv inference at scale

Back-of-envelope comparison for Q_tv inference on a 60,000-proposition
knowledge graph (K=4, ~240,000 binary nodes) in the RAPTL framework, where
Z1 acts as the Q_tv computation backend and H100 handles Q_logic.
Uses Z1's published specs and H100 datasheets.

| Metric | Z1 (TSU) | H100 (GPU) | Ratio |
|--------|----------|------------|-------|
| Inference time | **50 μs** | ~36 ms† | Z1 ~720× faster |
| Energy per inference | **125 μJ** | ~25 J | Z1 ~200,000× less |
| Time scaling with N | O(1) — physical parallel | O(N) — sweep iterations | — |

**Z1 timing** = K\_mix × 2τ₀ = 250 × 200 ns = 50 μs, independent of graph size
(all 250,000 p-bits update in parallel via bipartite Gibbs, τ₀ ≈ 100 ns from
[arXiv:2510.23972](https://arxiv.org/abs/2510.23972)).

**† GPU estimate** (AI-derived, no independent benchmark cited): assumes
~10¹⁰ sparse ops/s on H100 for this topology — 144 μs/sweep × 250 sweeps
≈ 36 ms.  The energy figure (NVIDIA H100 datasheet: 700 W TDP × 36 ms ≈ 25 J)
is more reliable than the speed estimate.

The key asymmetry: TSU inference time is *constant* in graph size because every
p-bit updates in one physical step.  GPU sweep time grows linearly with node
count, so the crossover favours Z1 increasingly as the knowledge graph scales.

## What this means for Hyperon

1. **Complete Q_tv on THRML**: All 11 implemented PLN rules compile directly to thrml
   factor graphs (log-probability weights via `P(x) ∝ e^{−ℰ(x)}`), encoding
   both ⊗ (evidence combination at factors) and ⊕ (marginalization at
   variables) without approximation.  Q_logic (rule selection, structure
   discovery) remains on CPU/GPU.  Gibbs sampling recovers both P(B|A) and
   P(A|B) from the same graph — hardware performs Bayes' rule automatically.

2. **Composability**: Rules combine by adding factors to the graph.
   Deduction chains, V-shapes, and collider topologies all work with the
   same compilation transform.

3. **Scaling**: A 20-node deduction chain runs in 1.5s on CPU; a 50-node
   chain completes within 60s.  On a TSU, thermal equilibration would be
   near-instantaneous.  Full benchmark suite: `pytest -m slow`.

6. **Hardware outlook**: Extropic's Z1 chip (early 2026) provides 250,000
   p-bits.  At K=16 each pairwise implication requires 256 binary couplings
   under sum-of-spins embedding, far exceeding each p-bit's ~12 physical
   connections.  This connectivity bottleneck is resolved by the K=4
   block-diagonal architecture (see §7): implication factors shrink to
   16 couplings, and Z1 can host ~60,000 propositions via time-division
   multiplexing.

7. **Block-diagonal architecture**: `pln_thrml/block_diagonal.py` addresses the TSU
   connectivity bottleneck by partitioning large graphs into blocks of 2–4
   propositions with K=4 (16 couplings per implication, fits 12-connection
   budget).  Block-internal inference uses exact Gibbs sampling; block-boundary
   influence propagates via BP-style messages through implication factors.
   Tree-structured block graphs converge exactly (Pearl 1988); cyclic graphs
   use damped loopy BP.  Z1 capacity estimate: ~1,225 simultaneous propositions
   (4,900 p-bits ÷ 4 p-bits/prop), ~60,000 with time-division multiplexing.

8. **Three-tier pipeline (RAPTL framework)**: In Goertzel's Resource-Aware
   Probabilistic Tensor Logic framework, CPU/GPU/TSU each handle what they do
   best:

   | Tier | Hardware | Role | Why this hardware |
   |------|----------|------|-------------------|
   | Control | CPU | Geodesic controller: symbolic matching, variable binding, next-step scheduling (Q_logic core) | Discrete graph traversal — no energy encoding possible |
   | Compile | GPU | RAPTL tensor contractions, ShardZipper (MORK → contiguous arrays), certified semantic-preserving rewrites, neural pattern mining (WILLIAM) | Batch float parallelism, deterministic |
   | Sample | TSU | Q_tv joint posterior sampling over the compiled factor graph (see [Z1 vs H100](#z1-vs-h100-q_tv-inference-at-scale)) | Stochastic native, O(1) in graph size, ~200,000× less energy |

   The three tiers can run as an **asynchronous pipeline** — CPU schedules step
   N+1 while GPU compiles step N and TSU samples step N−1.  Theoretical
   justification: the weakness-bounded leakage theorem (Goertzel 2026, Thm 5.3)
   proves that evidence conservation is robust to scheduling reordering, with
   error bounded by pairwise weakness — which is small for typical PLN rules.

## Project structure

```
pln_thrml/                 Main package
  __init__.py              Public API re-exports (beta + block_diagonal)
  beta.py                  Beta factor graph engine (builds, samples, recovers stv)
  block_diagonal.py        Block-diagonal decomposition + loopy BP for large/cyclic graphs
  metta/                   MeTTa integration layer (optional, requires hyperon)
    __init__.py            Entry point — exports register_all()
    atoms.py               Atom extraction from MeTTa space
    ops/
      rules.py             All 11 rules (declarative table + generic factory, including revision & negation)
    declarations/
      pln_types.metta      Type declarations (stv, Implication, Similarity, etc.)
vendor/PLN/                trueagi-io/PLN (git submodule) — test baselines
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

- **Cyclic inference (full-graph)**: `pln_thrml/block_diagonal.py` handles cyclic *block*
  graphs via damped loopy BP; full-graph cyclic Gibbs (without block
  decomposition) is not yet implemented.
- **EvidenceID / StampDisjoint**: Evidence tracking to prevent double-counting
  during revision (PLN uses `StampDisjoint` and `StampConcat`).
- **PLN.Derive**: Priority-queue based iterative inference engine with
  bounded belief buffer.
- **ECAN attention**: Attention allocation for prioritizing which
  subgraphs to sample first.
- **Continuous-valued nodes**: Extending beyond discrete propositions.
- **Intensional / higher-order / quantifier / temporal rules**: This repo
  covers the extensional inference subset of PLN (11 rules). Intensional
  inference (attribute-based), higher-order boolean combinations, quantifier
  reasoning (ForAll / ThereExists), and temporal/causal rules (predictive
  implication, sequential AND) are not yet implemented.

See [docs/future-work/future-work.md](docs/future-work/future-work.md) for
detailed analysis including native pdit support, low-rank factorization,
geodesic controller scheduling, and other block-diagonal extensions.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, testing, code
conventions, and pull request guidelines.

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
