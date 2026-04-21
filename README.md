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
that run on thermodynamic hardware. Each inference splits into two
independent paths: **strength (s)** via Ising Gibbs sampling on TSU,
**confidence (c)** via closed-form algebra on CPU. 5 PLN rules (MP,
Deduction, Abduction, Inversion, Revision) are implemented and verified
end-to-end with 88 tests.

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

In thrml, two primitives cover everything PLN-THRML needs:

| thrml construct                  | What it does                                                       |
| -------------------------------- | ------------------------------------------------------------------ |
| `SpinNode`                       | Binary (±1) variable — 1 pbit per proposition                     |
| `SpinEBMFactor`                  | Ising bias (_h_) + coupling (_J_) between spins                   |

Every proposition is one `SpinNode` — 1 pbit, exact 2×2 encoding, zero
discretisation error for strength. Rules differ only in how the spins
connect: a chain of `SpinEBMFactor`s for deduction, a joint 2-node graph
for inversion, a hidden-unit topology for abduction's explaining-away.

**Gibbs sampling** iteratively resamples each variable conditioned on its
neighbors until the joint distribution converges. The Boltzmann connection
`P(x) ∝ e^{−ℰ(x)}` means lower energy = higher probability — so
log-probabilities become energy weights directly.

</details>

**Technical summary**: PLN's 2×2 conditional probability table encodes
exactly as Ising parameters (bias _h_, coupling _J_) — 1 pbit per
proposition, zero discretisation error for strength. Confidence
propagates via closed-form PLN/QLN algebra on CPU, running in parallel
with the TSU sampler. 5 rules × multiple parameter sets × 88 tests
constitute the evidence.
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

### Unified API

```python
from pln_thrml import unified_modus_ponens

# Modus ponens: P(A)=(0.8, 0.9), s(A→B)=0.9, c(A→B)=0.85
s, c, meta = unified_modus_ponens(s_A=0.8, c_A=0.9, s_AB=0.9, c_AB=0.85)
print(f"P(B) = (stv {s:.3f} {c:.3f})")   # ≈ (stv 0.777 0.765)
# s from Ising Gibbs sampling, c from PLN closed-form formula
```

### Hybrid API (lightweight)

```python
from pln_thrml.hybrid import hybrid_modus_ponens

# Same inference — binary Ising s + PLN formula c
s, c = hybrid_modus_ponens(s_A=0.8, c_A=0.9, s_AB=0.9, c_AB=0.85)
print(f"P(B) = (stv {s:.3f} {c:.3f})")   # ≈ (stv 0.777 0.765)
```

### Run tests

```bash
# Run per-file to avoid OOM (JAX compilation cache grows large)
python -m pytest tests/test_unified.py -v
python -m pytest tests/test_hybrid.py -v
```

## How it works

### The two paths: strength on TSU, confidence on CPU

1. **Compile s** — PLN's 2×2 conditional table `[1−ε, ε; 1−s_AB, s_AB]` (rows = parent F/T, cols = child F/T, ε = background rate ≈ 0.02 = P(child=T | parent=F)) maps exactly to Ising parameters: bias _h_ encodes the prior, coupling _J_ encodes the implication strength. 1 `SpinNode` per proposition.
2. **Sample s (TSU)** — Block Gibbs sampling over Ising spins. Binary encoding means zero discretisation error for strength.
3. **Compute c (CPU)** — PLN closed-form: e.g. `c_B = c_A × c_AB` for modus ponens. No sampling needed — deterministic algebra.
4. **Return** — `(strength, confidence)` per rule.

### PLN → thrml mapping

| PLN concept                       | Hardware path                                | Device            |
| --------------------------------- | -------------------------------------------- | ----------------- |
| Proposition                       | `SpinNode` (binary ±1)                       | pbit              |
| Prior P(A)                        | Ising bias _h_                               | Bias field        |
| Implication A→B                   | Ising coupling _J_                           | Coupling          |
| Confidence c                      | PLN/QLN formula on CPU                       | CPU               |
| Deduction chain                   | Chain of Ising spins                         | Spin pipeline     |
| Abduction (inverted-V)            | Hidden-unit topology                         | Explaining-away   |
| Inversion (Bayes)                 | 2-node all-free joint graph                  | pbit pair         |
| Revision                          | PLN book formula: n_rev = n₁ + n₂ (raw)     | CPU only          |

## API reference

### Unified API (`pln_thrml.unified`)

| Function | Rule |
| -------- | ---- |
| `unified_modus_ponens(s_A, c_A, s_AB, c_AB)` | A, A→B ⊢ B |
| `unified_deduction(s_A, c_A, s_B, c_B, s_C, c_C, s_AB, c_AB, s_BC, c_BC)` | A→B, B→C ⊢ A→C |
| `unified_abduction(s_A, c_A, s_B, c_B, s_AC, c_AC, s_BC, c_BC)` | A→C, B→C ⊢ A→B |
| `unified_inversion(s_A, c_A, s_B, c_B, s_AB, c_AB)` | A→B ⊢ B→A |
| `unified_revision(s1, c1, s2, c2)` | merge two evidence streams |

TSU rules (MP, Deduction, Abduction) return `(strength, confidence, metadata)` and iterate 2–3 rounds. CPU-only rules (Inversion, Revision) return `(strength, confidence)` immediately.

### Hybrid API (`pln_thrml.hybrid`)

| Function | Rule |
| -------- | ---- |
| `hybrid_modus_ponens(s_A, c_A, s_AB, c_AB)` | A, A→B ⊢ B |
| `hybrid_deduction(s_A, c_A, ..., s_AB, c_AB, s_BC, c_BC)` | A→B, B→C ⊢ A→C |
| `hybrid_abduction(s_A, c_A, ..., s_AC, c_AC, s_BC, c_BC)` | A→C, B→C ⊢ A→B |
| `hybrid_inversion(s_A, c_A, s_B, c_B, s_AB, c_AB)` | A→B ⊢ B→A |

All return `(strength, confidence)`.

### Confidence layer (`pln_thrml.qln_cpu`)

| Function | Purpose |
| -------- | ------- |
| `c_modus_ponens(c_A, c_AB)` | c_B = c_A × c_AB |
| `c_deduction(s_AB, c_AB, s_BC, c_BC)` | c_AC from chain |
| `c_abduction(s_AC, c_AC, s_BC, c_BC)` | c_AB from shared consequent |
| `inversion_bayes(s_A, c_A, s_B, c_B, s_AB, c_AB)` | Exact Bayesian P(B→A) |
| `inversion_pln(s_A, c_A, s_B, c_B, s_AB, c_AB)` | PLN heuristic |
| `revision(s1, c1, s2, c2)` | Evidence merge: n_rev = n₁ + n₂ |

## Results

5 PLN rules verified across three architecture levels (88 tests total).
Representative strength errors (Δ vs DTV continuous baseline):

| Rule         | Unified (Ising+QLN) | Hybrid (Binary+PLN) |
| ------------ | ------------------- | -------------------- |
| Modus Ponens | 0.053               | 0.018                |
| Deduction    | 0.050               | 0.019 (3-spin joint chain) |
| Abduction    | 0.116               | 0.002 (Ded∘Inv reduction, Bayes-correct bg) |
| Inversion    | 0.021 (Bayes exact) | Bayes exact          |
| Revision     | 0.001 (PLN book)    | —                    |

All rules use the binary Ising path (1 pbit per proposition, zero
discretisation error for strength). Inversion and Revision are CPU-only
(Bayes formula and PLN book n_rev = n₁ + n₂ raw counts).

## Hyperon integration outlook

RAPTL (Resource-Aware Probabilistic Tensor Logic) bundles inference into
a triple product quantale Q = Q_logic × Q_uncertainty × Q_resource.
These three components travel together through every operation — RAPTL's
joint optimization depends on co-locating all three so that, e.g.,
uncertainty tolerance can inform sparse-to-dense approximation decisions
alongside resource constraints.

This project takes a different approach: extract Q_tv and compile it to
TSU hardware by splitting strength (TSU Ising sampling) from confidence
(CPU closed-form), keeping Q_logic on CPU/GPU. This
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
  __init__.py              Public API re-exports (pln_utils + unified + qln_cpu)
  pln_utils.py             PLN conversion utilities (c2w, w2c, stv_to_beta_params — no thrml dependency)
  hybrid.py                Binary Ising s + PLN formula c (MP / 3-spin-chain Deduction / Ded∘Inv Abduction / 2-node Inversion)
  compiler_binary.py       Binary Ising compiler (1 pbit/proposition, exact 2×2 encoding, joint 2-node graph)
  unified.py               Unified LBM(s) on TSU + QLN(n) on CPU: per-rule g(n) calibration
  qln_cpu.py               QLN n-layer: closed-form confidence propagation (Inversion, Revision)
  compiler_unified.py      Confidence-modulated Ising compiler (g(n) precision scaling)
  dtv_baseline.py          DTV continuous Monte Carlo baseline (zero discretization error)
vendor/PLN/                trueagi-io/PLN (git submodule) — test baselines
tests/
  conftest.py              Shared strength tolerance (±0.05); confidence is closed-form (≤1e-3)
  test_unified.py          Unified architecture validation (DTV / PLN / Unified / Inversion / Revision)
  test_hybrid.py           Binary-Ising-s + PLN-formula-c validation (DTV / PLN / Binary / Hybrid)
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
