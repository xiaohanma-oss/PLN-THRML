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
that run on thermodynamic hardware. The **(ρ, n) separation architecture**
splits each inference into two independent paths: **strength (s)** via
Ising Gibbs sampling on TSU, **confidence (c)** via closed-form algebra
on CPU. 5 PLN rules (MP, Deduction, Abduction, Inversion, Revision) are
implemented and verified end-to-end with 88 tests.

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
Each variable holds a discrete distribution; each factor encodes how likely
certain state combinations are via a weight table.

In thrml (two levels):

| thrml construct                  | What it does                                                       |
| -------------------------------- | ------------------------------------------------------------------ |
| `SpinNode`                       | Binary (±1) variable — 1 pbit per proposition                     |
| `SpinEBMFactor`                  | Ising bias (_h_) + coupling (_J_) between spins                   |
| `CategoricalNode(K)`            | K-state discrete variable (K-bin Beta prior)                       |
| `CategoricalEBMFactor` (unary)  | Weight vector encoding a prior over K bins                         |
| `SquareCategoricalEBMFactor` (pairwise) | K×K weight table encoding a conditional relationship between two nodes |

The separation architecture's main path uses `SpinNode` (1 pbit = 1
proposition, exact 2×2 encoding). K-bin `CategoricalNode` is used for
baseline comparison and rules (like abduction) that need higher resolution.

**Gibbs sampling** iteratively resamples each variable conditioned on its
neighbors until the joint distribution converges. The Boltzmann connection
`P(x) ∝ e^{−ℰ(x)}` means lower energy = higher probability — so
log-probabilities become energy weights directly.

</details>

**Technical summary**: PLN's 2×2 conditional probability table encodes
exactly as Ising parameters (bias _h_, coupling _J_) — 1 pbit per
proposition, zero discretisation error for strength. Confidence
propagates via closed-form PLN/QLN algebra on CPU, running in parallel
with the TSU sampler. For rules needing higher resolution (abduction),
a K-bin Beta fallback path provides full distributional inference.
5 rules × multiple parameter sets × 88 tests constitute the evidence.
Rule selection and structure discovery remain on CPU/GPU.

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

The separation architecture computes strength via Ising sampling and
confidence via PLN closed-form formulas. For most rules (MP, Deduction),
the binary Ising path recovers strength with zero discretisation error —
the 2×2 conditional table maps exactly to Ising (h, J). For rules where
binary encoding is insufficient (Abduction), a K-bin Beta fallback or
hidden-unit topology provides higher resolution. Inversion uses exact
Bayesian P(A|B) = P(B|A)·P(A)/P(B), bypassing PLN's heuristic 0.6
confidence discount.

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

### Unified API (separation architecture)

```python
from pln_thrml import unified_modus_ponens

# Modus ponens: P(A)=(0.8, 0.9), s(A→B)=0.9, c(A→B)=0.85
s, c, meta = unified_modus_ponens(s_A=0.8, c_A=0.9, s_AB=0.9, c_AB=0.85)
print(f"P(B) = (stv {s:.3f} {c:.3f})")   # ≈ (stv 0.724 0.765)
# s from Ising Gibbs sampling, c from PLN closed-form formula
```

### Low-level Beta engine

```python
from pln_thrml import build_beta_chain, sample_and_measure

# Same inference via K-bin factor graph (baseline/comparison)
graph = build_beta_chain(
    priors=[0.8, 0.5], confidences=[0.9, 0.01],
    strengths=[0.9], impl_confidences=[0.85],
    backgrounds=[0.02],
)
s, c = sample_and_measure(graph, target_node=graph["nodes"][1])
print(f"P(B) = (stv {s:.4f} {c:.4f})")   # ≈ (stv 0.73 0.76)
```

### Run tests

```bash
# Run per-file to avoid OOM (JAX compilation cache grows large)
python -m pytest tests/test_unified.py -v
python -m pytest tests/test_hybrid.py -v
python -m pytest tests/test_hardware_approximation.py -v
```

## How it works

### (ρ, n) separation — the main path

1. **Compile s** — PLN's 2×2 conditional table `[ε, s_AB; 1−ε, 1−s_AB]` maps exactly to Ising parameters: bias _h_ encodes the prior, coupling _J_ encodes the implication strength. 1 `SpinNode` per proposition.
2. **Sample s (TSU)** — Block Gibbs sampling over Ising spins. Binary encoding means zero discretisation error for strength.
3. **Compute c (CPU)** — PLN closed-form: e.g. `c_B = c_A × c_AB` for modus ponens. No sampling needed — deterministic algebra.
4. **Iterate** — g(n)-modulated coupling strength converges over 2–3 rounds. Each round recompiles with updated confidence.
5. **Return** — `(strength, confidence, metadata)` with convergence history.

### K-bin Beta fallback

For rules where binary encoding is insufficient (abduction's explaining-away
needs full joint), a K-bin `CategoricalNode` path provides higher resolution:
PLN `(stv s c)` → Beta(α,β) → K-bin discretisation → K×K factor weights →
Gibbs sampling → moment-match posterior → `(s, c)`.

### PLN → thrml mapping

| PLN concept                       | Separation path (main)                       | K-bin path (fallback)                       | Hardware          |
| --------------------------------- | -------------------------------------------- | ------------------------------------------- | ----------------- |
| Proposition                       | `SpinNode` (binary ±1)                       | `CategoricalNode` (K bins)                  | pbit / pdit       |
| Prior P(A)                        | Ising bias _h_                               | Unary `CategoricalEBMFactor`                | Bias field        |
| Implication A→B                   | Ising coupling _J_                           | `SquareCategoricalEBMFactor` (K×K)          | Coupling          |
| Confidence c                      | PLN/QLN formula on CPU                       | Moment-matching from posterior              | CPU               |
| Deduction chain                   | Chain of Ising spins                         | Chain of K-bin nodes                        | Spin pipeline     |
| Abduction (inverted-V)            | Hidden-unit topology                         | Collider factor graph                       | Explaining-away   |
| Revision                          | QLN formula: n_rev = n₁ + n₂                | —                                           | CPU only          |

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

5 PLN rules verified across three architecture levels (88 tests total).
Representative strength errors (Δ vs DTV continuous baseline):

| Rule         | Unified (Ising+QLN) | Hybrid (Binary+PLN) | Categorical (K=4) |
| ------------ | ------------------- | -------------------- | ------------------ |
| Modus Ponens | 0.053               | 0.018                | 0.061              |
| Deduction    | 0.050               | 0.085                | 0.049              |
| Abduction    | 0.116 (K-bin path)  | 0.269                | 0.036              |
| Inversion    | 0.021 (Bayes exact) | —                    | —                  |
| Revision     | 0.001 (QLN formula) | —                    | —                  |

MP and Deduction use the binary Ising main path. Abduction falls back to
K-bin for full joint resolution. Inversion and Revision are CPU-only
(Bayes formula and QLN n₁+n₂).

### Hardware cost comparison

Five-column comparison at K=4 (test_hardware_approximation):

| Method              | Pbits/prop | Couplings/edge | MP Δs (strong) |
| ------------------- | ---------- | -------------- | -------------- |
| Categorical (full)  | K          | K²             | 0.061          |
| One-hot (full)      | K          | K²             | 0.055          |
| Potts (diagonal)    | K          | K              | 0.019          |
| Binary Ising        | 1          | 1              | 0.018          |

Binary Ising achieves comparable accuracy to K=4 categorical at 1/16th
the hardware cost (1 pbit vs 16 spin nodes, 1 coupling vs 16).

## Hyperon integration outlook

RAPTL (Resource-Aware Probabilistic Tensor Logic) bundles inference into
a triple product quantale Q = Q_logic × Q_uncertainty × Q_resource.
These three components travel together through every operation — RAPTL's
joint optimization depends on co-locating all three so that, e.g.,
uncertainty tolerance can inform sparse-to-dense approximation decisions
alongside resource constraints.

This project takes a different approach: extract Q_tv and compile it to
TSU hardware via (ρ, n) separation, keeping Q_logic on CPU/GPU. This
trades RAPTL's joint optimization for hardware-native sampling — 5 PLN
rules validate that Q_tv executes faithfully this way.

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
