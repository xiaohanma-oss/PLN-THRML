# PLN-THRML

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Version 0.2.0](https://img.shields.io/badge/version-0.2.0-green.svg)](pyproject.toml)

> Compile PLN inference rules to thermodynamic factor graphs and run them
> via Gibbs sampling — bridging [Hyperon/PLN](https://github.com/trueagi-io/PLN)
> and [Extropic/thrml](https://github.com/extropic-ai/thrml).

## Table of Contents

- [Overview](#overview)
- [Why this matters](#why-this-matters)
- [Installation](#installation)
- [Quick start](#quick-start)
- [How it works](#how-it-works)
- [API reference](#api-reference)
- [Results](#results)
- [Hyperon integration outlook](#hyperon-integration-outlook)
- [Project structure](#project-structure)
- [Contributing](#contributing)
- [Acknowledgements](#acknowledgements)

## Overview

PLN-THRML compiles probabilistic logic inference rules into factor graphs
that run on thermodynamic hardware. You give it PLN premises with truth
values; it builds a Boltzmann energy model, runs Gibbs sampling, and returns
inferred truth values. 11 PLN rules are implemented and verified end-to-end.

<details>
<summary><strong>New to PLN? (30-second primer)</strong></summary>

PLN represents uncertain knowledge with two numbers:

| Component          | Meaning                                 | Analogy                                           |
| ------------------ | --------------------------------------- | ------------------------------------------------- |
| **strength** _s_   | How likely something is true            | A poll result: "80% of respondents said yes"      |
| **confidence** _c_ | How much evidence supports the estimate | The poll's sample size: 1,000 people vs 10 people |

`(stv 0.8 0.9)` means "80% likely true, based on strong evidence."
`(stv 0.8 0.01)` means "80% likely true, but we barely know — almost a guess."

Where do _s_ and _c_ come from? Given _n_ observations, _n⁺_ of which are
positive, and a prior weight _k_ (default 1):

    s = n⁺ / n          c = n / (n + k)

More observations push _c_ → 1; fewer keep it near 0. The strength _s_ is
just the observed frequency. This dual-value semantics is what makes PLN
more expressive than plain probabilities, and what this project must preserve
when compiling to hardware.

</details>

<details>
<summary><strong>New to thrml / factor graphs? (30-second primer)</strong></summary>

A **factor graph** is a bipartite graph of variable nodes and factor nodes.
Each variable holds a discrete distribution over K states; each factor
encodes how likely certain state combinations are via a weight table.

In thrml:

| thrml construct                  | What it does                                                       |
| -------------------------------- | ------------------------------------------------------------------ |
| `CategoricalNode(K)`            | A discrete random variable with K possible states                  |
| `CategoricalEBMFactor` (unary)  | Weight table encoding a prior — how likely each state is on its own |
| `SquareCategoricalEBMFactor` (pairwise) | K×K weight table encoding a conditional relationship between two nodes |

**Gibbs sampling** iteratively resamples each variable conditioned on its
neighbors until the joint distribution converges. The Boltzmann connection
`P(x) ∝ e^{−ℰ(x)}` means lower energy = higher probability — so
log-probabilities become energy weights directly.

</details>

**Technical summary**: PLN truth values compile to Boltzmann energy
weights via `P(x) ∝ e^{−ℰ(x)}` (take the negative log of each
probability to get the energy), and Gibbs sampling over the
resulting factor graph recovers correct inference results — including
both P(B|A) and P(A|B) from the same graph (hardware performs Bayes'
rule automatically). 11 rules × multiple parameter sets constitute the
end-to-end evidence. Rule selection and structure discovery remain on
CPU/GPU.

## Why this matters

### Parallel inference

The core task is propagating PLN truth values across a knowledge graph —
given known premises, infer the strength and confidence of every reachable
conclusion. Each architecture parallelizes this differently and hits
different bottlenecks:

|               | CPU                  | GPU (estimated)        | TSU                              |
| ------------- | -------------------- | ---------------------- | -------------------------------- |
| Parallelism   | Sequential per-rule  | Batched tensor contraction | All nodes sample simultaneously |
| Bottleneck    | Inference chain length | Communication + memory bandwidth | Mixing time (energy barrier height) |
| Best fit      | Small graphs, exact reasoning | Large sparse graphs  | Graph topology fits on-chip¹    |

On CPU, PLN runs rule-based chaining — each step applies one rule's
formula in dependency order. On GPU, Goertzel's RAPTL-ShardZipper
framework encodes logical relations as sparse tensors for batch
contraction. On a TSU, the knowledge graph compiles into a factor graph
where all sampling cells update in parallel via block Gibbs sampling —
wall-clock per iteration is independent of node count for bipartite
graphs that fit on-chip, but total time depends on mixing time, graph
depth, and lattice size.

¹ Graphs exceeding a single TSU chip require multi-chip partitioning
with communication overhead. Mixing time depends on graph structure and
can grow sharply for landscapes with tall energy barriers.

### Inference accuracy

PLN's rule formulas operate on point estimates — they plug strength
values into algebraic expressions that assume independence or use
heuristic simplifications. PLN-THRML instead discretizes each truth
value into a K-bin Beta distribution and lets the factor graph compute
the exact joint posterior via Gibbs sampling, preserving the full
distributional shape. The [Results](#results) table shows this
concretely — abduction and induction diverge from PLN's formulas not
because the factor graph is inaccurate, but because the point-estimate
formulas lose information. Inversion gives exact Bayesian P(A|B) where
PLN uses a heuristic with a fixed 0.6 confidence discount.

### Energy efficiency

The TSU architecture paper ([arXiv:2510.23972](https://arxiv.org/abs/2510.23972))
reports ~10,000× lower energy per sample vs GPU baselines on DTM sampling
of binarized Fashion-MNIST (E_cell ≈ 2 femtojoules). PLN factor graph
inference is expected to benefit similarly but has not been independently
benchmarked.

## Installation

```bash
git clone --recurse-submodules https://github.com/xiaohanma-oss/PLN-THRML.git
cd PLN-THRML
pip install -e .                          # core only (thrml + jax)
pip install -e ".[dev]"                   # + pytest for running tests
```

> The [trueagi-io/PLN](https://github.com/trueagi-io/PLN) submodule provides
> test baselines. If you cloned without `--recurse-submodules`, run
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

### Unified API (separation architecture)

```python
from pln_thrml import unified_lbm_strength

# Modus ponens — LBM(s) on TSU + QLN(n) on CPU
result = unified_lbm_strength(
    rule="mp",
    premises=[(0.8, 0.9)],        # P(A) = (s=0.8, c=0.9)
    links=[(0.9, 0.85)],          # s(A→B)=0.9, c=0.85
)
print(f"s={result['s']:.3f}  c={result['c']:.3f}")
# s and c computed independently: s from Gibbs sampling, c from PLN formula
```

### Run tests

```bash
# Run per-file to avoid OOM (JAX compilation cache grows large)
python -m pytest tests/test_unified.py -v
python -m pytest tests/test_hybrid.py -v
python -m pytest tests/test_hardware_approximation.py -v
```

## How it works

1. **Parameterize** — PLN `(stv s c)` → Beta(α, β) with mean = s
2. **Discretize** — discretize each Beta distribution into **K bins** (`CategoricalNode`). Higher K → better accuracy; lower K → faster sampling.
3. **Build graph** — each implication becomes a K×K weight table (log of the discretized conditional Beta); assemble all nodes and factors into a factor graph.
4. **Sample** — Block Gibbs sampling (50 batches × 2,000 samples). Root nodes can be **clamped** (fixed to their prior values as known evidence) while free nodes are updated by Gibbs sweeps.
5. **Recover** — moment-match the posterior histogram → `(strength, confidence)`

The Beta-to-energy compilation preserves both strength (bin position) and
confidence (bin sharpness) through discretization.

| PLN concept                       | thrml construct                             | Extropic hardware               |
| --------------------------------- | ------------------------------------------- | ------------------------------- |
| Proposition (VariableAtom)        | `CategoricalNode` (K bins)                  | pdit (K-category sampling cell) |
| Prior P(A)                        | Unary `CategoricalEBMFactor` (Beta prior)   | Bias field on pdit              |
| Implication A→B (strength s)      | Pairwise `SquareCategoricalEBMFactor` (K×K) | Coupling between pdits          |
| TruthValue (strength, confidence) | Beta posterior → moment-matching            | Posterior distribution readout  |
| Inference rule                    | Block Gibbs sampling                        | Thermal equilibration           |
| Deduction chain                   | Chain factor graph                          | Pipeline of coupled clusters    |
| Induction (V-shape)               | Star factor graph                           | Hub-and-spoke topology          |
| Abduction (inverted-V)            | Collider factor graph                       | Explaining-away circuit         |
| Revision (evidence merge)         | Multiple unary factors on one node          | Competing bias fields converge  |

## API reference

### Unified API (`pln_thrml.unified`)

| Function | Purpose |
| -------- | ------- |
| `unified_lbm_strength(rule, premises, links, ...)` | LBM(s) on TSU + QLN(n) on CPU — the primary inference entry point |

Supports: `mp` (modus ponens), `deduction`, `abduction`, `inversion`, `revision`.

### Low-level engine (`pln_thrml.beta`)

**Graph builders** — each returns a `dict` with `"nodes"`, `"factors"`, and metadata:

| Function | Topology |
| -------- | -------- |
| `build_beta_chain(priors, confidences, strengths, ...)` | Directed chain X₀→X₁→...→Xₙ (modus ponens, deduction) |
| `build_beta_inv_v_graph(...)` | Inverted-V: Left→Center←Right (abduction) |

**Sampling & measurement**:

| Function | Returns |
| -------- | ------- |
| `sample_and_measure(graph, target_node)` | `(strength, confidence)` — one-step sampling + recovery |
| `run_beta_sampling(graph)` | Raw sample array for further analysis |
| `estimate_beta_marginal(samples, graph, node)` | Posterior histogram + `(s, c)` for a node |
| `estimate_beta_conditional(samples, graph, target, condition)` | P(target \| condition=True) via weighted posterior |

**Conversion utilities**: `stv_to_beta_params(s, c)`, `posterior_to_stv(histogram, k)`, `c2w(c)` / `w2c(w)`, `bin_centers(k)`.

### Confidence layer (`pln_thrml.qln_cpu`)

| Function | Purpose |
| -------- | ------- |
| `qln_confidence_*` | Closed-form confidence propagation for each rule (Inversion Bayes/PLN, Revision QLN) |

## Results

11 PLN rules compiled and verified (K=16, 100,000 samples). Strength errors
for representative parameter sets
(representative parameter sets):

| Rule              | Max Error | Rule              | Max Error |
| ----------------- | --------- | ----------------- | --------- |
| Modus Ponens      | 0.008     | Negation          | 0.000     |
| Deduction         | 0.073     | Revision          | 0.015     |
| Inversion         | 0.038     | Sym. Modus Ponens | 0.005     |
| Induction         | 0.046     | Equiv→Impl        | 0.025     |
| Abduction         | 0.130     | Trans. Similarity | 0.050     |
| Eval. Implication | 0.061     |                   |           |

Most rules match within 5%. Abduction/Induction diverge because PLN uses
closed-form approximations while the factor graph computes the exact joint
posterior. Inversion gives exact Bayesian P(A|B) vs PLN's heuristic.

### Effect of K on accuracy

Reducing K from 16 to 4 (for TSU hardware deployment) introduces additional
discretization error. Measured on Modus Ponens (s_A=0.8, s_AB=0.9, PLN
analytical strength = 0.724) and single-node prior recovery:

| K   | MP Δs vs PLN | Prior Δs (worst) | Prior Δc (worst) | Couplings/impl |
| --- | ------------ | ---------------- | ---------------- | -------------- |
| 16  | 0.008        | 0.001            | 0.013            | 256            |
| 8   | 0.011        | 0.008            | 0.035            | 64             |
| 4   | 0.024        | 0.019            | 0.067            | 16             |

K=4 strength error is 0.024 — well within the 0.05 tolerance used for K=16
tests. Confidence recovery degrades more (Δc up to 0.067) because
moment-matching from 4 bins has less information.

## Hyperon integration outlook

RAPTL (Resource-Aware Probabilistic Tensor Logic) bundles inference into
a triple product quantale Q = Q_logic × Q_uncertainty × Q_resource.
These three components travel together through every operation — RAPTL's
joint optimization depends on co-locating all three so that, e.g.,
uncertainty tolerance can inform sparse-to-dense approximation decisions
alongside resource constraints.

This project takes a different approach: extract Q_tv and compile it to
TSU hardware, keeping Q_logic on CPU/GPU. This trades RAPTL's joint
optimization for hardware-native sampling — 11 PLN rules validate that
Q_tv executes faithfully this way.

A possible heterogeneous pipeline extending this idea:

| Tier    | Hardware | Role                                                       |
| ------- | -------- | ---------------------------------------------------------- |
| Control | CPU      | Variable binding, rule scheduling                          |
| Compile | CPU+GPU  | Sparse tensor contraction, graph sharding (ShardZipper)    |
| Sample  | TSU      | Boltzmann sampling over compiled factor graphs              |

The Sample tier is more general than PLN alone — any algorithm reducible
to sampling from P(x) ∝ e^{−ℰ(x)} is a candidate (see [Sister Projects](#sister-projects)).

The three-tier pipeline is our projection, not described in the
references. Whether the gain from hardware-native sampling outweighs the
loss of RAPTL's joint optimization is an open question.

### (ρ, n) separation: hybrid architecture

Inspired by QLN's quantum truth value QTV = (ρ, n), `hybrid.py` separates
PLN's (s, c) computation:

- **ρ → s**: 1 binary pbit per proposition with Ising J coupling.
  The 2×2 conditional probability table is exactly encoded as Ising
  parameters — zero approximation error, no K-bin discretisation.
- **n → c**: PLN closed-form formulas on CPU/GPU (e.g. c_B = c_A × c_AB).
  No sampling needed — c is deterministic algebra.

These two paths have **no data dependency** and can run in parallel.
The binary Ising compiler (`compiler_binary.py`) produces factor graphs
with 1 pbit/proposition and 1 Ising coupling/edge — the simplest
possible TSU target.  Within the 12-neighbour pbit budget, each
proposition can participate in up to 12 implication edges.

## Project structure

```
pln_thrml/                 Main package
  __init__.py              Public API re-exports (beta + unified + qln_cpu)
  beta.py                  Beta factor graph engine (builds, samples, recovers stv) + PLN utilities
  unified.py               Unified LBM(s) on TSU + QLN(n) on CPU: per-rule g(n) calibration
  qln_cpu.py               QLN n-layer: closed-form confidence propagation (Inversion, Revision)
  hybrid.py                (ρ, n) separation: binary Ising sampling s + PLN formula c
  compiler_binary.py       Binary Ising compiler (1 pbit/proposition, exact 2×2 encoding)
  compiler_onehot.py       One-hot spin compiler (pbit-only hardware simulation)
  compiler_unified.py      Confidence-modulated Ising compiler (g(n) precision scaling)
  dtv_baseline.py          DTV continuous Monte Carlo baseline (zero discretization error)
vendor/PLN/                trueagi-io/PLN (git submodule) — test baselines
tests/
  conftest.py              Shared tolerance constants (strength ±0.05, confidence ±0.15)
  test_unified.py          Unified architecture validation (DTV / PLN / Hybrid / Unified / Inversion / Revision)
  test_hybrid.py           (ρ, n) separation validation (DTV / PLN / Cat / OH / Binary / Hybrid)
  test_hardware_approximation.py  Pbit-only hardware cost analysis (DTV / Cat / OneHot / Potts comparison)
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, testing, code
conventions, and pull request guidelines.

## Sister Projects

Five projects compiling Hyperon's cognitive architecture to thermodynamic hardware:

| Project | What it compiles |
|---------|-----------------|
| **[PLN-THRML](https://github.com/xiaohanma-oss/PLN-THRML)** | **Probabilistic inference → Boltzmann energy tables** |
| [ECAN-THRML](https://github.com/xiaohanma-oss/ECAN-THRML) | Attention diffusion → Lattice Boltzmann simulation |
| [MOSES-THRML](https://github.com/xiaohanma-oss/MOSES-THRML) | Program evolution → Boltzmann sampling |
| [QuantiMORK-THRML](https://github.com/xiaohanma-oss/QuantiMORK-THRML) | Predictive coding → wavelet-sparse factor graphs |
| [Geodesic-THRML](https://github.com/xiaohanma-oss/Geodesic-THRML) | Unified geodesic scheduler for all above |

## Acknowledgements

- [Hyperon/PLN](https://github.com/trueagi-io/PLN) — TrueAGI
- [thrml](https://github.com/extropic-ai/thrml) — Extropic AI factor graph library

## License

[MIT](LICENSE) — Copyright (c) 2026 Xiaohan Ma
