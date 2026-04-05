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

> **Interactive overview**: Open [`docs/interactive_overview.html`](docs/interactive_overview.html) in your browser for a visual walkthrough — pipeline, all 11 rule topologies, and the energy mapping.

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

**Technical summary**: All Q_tv operations in the PLN product quantale
(⊗ evidence combination at factors, ⊕ marginalization at variables) compile
faithfully to Boltzmann energy weights and recover correct inference results
from thermodynamic sampling. 11 rules × multiple parameter sets constitute
the end-to-end evidence. Gibbs sampling recovers both P(B|A) and P(A|B)
from the same graph — hardware performs Bayes' rule automatically. Q_logic
(rule selection, structure discovery) remains on CPU/GPU.

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

¹ The TSU uses an L×L grid of sampling cells with sparse local
connectivity. Graphs exceeding a single chip require multi-chip
partitioning with communication overhead. Mixing time (K_mix) depends on
graph structure and spectral gap — it is not a fixed constant and can
grow sharply for energy landscapes with tall barriers.

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
pip install -e ".[metta]"                 # + MeTTa bridge (requires hyperon)
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

### MeTTa bridge (requires `hyperon`)

```python
from hyperon import MeTTa
from pln_thrml.metta import register_all

metta = MeTTa()
register_all(metta)

# Unified operator — input structure determines the rule automatically
results = metta.run('''
    !(thrml (A (stv 0.8 0.9)) ((Implication A B) (stv 0.9 0.85)))
''')
# => (B (stv 0.7242 0.5508))
#      B       — inferred conclusion atom
#      0.7242  — strength (inferred probability)
#      0.5508  — confidence (evidence weight behind the estimate)

# More examples:
# Deduction:  !(thrml ((Implication A B) (stv 0.8 0.9)) ((Implication B C) (stv 0.9 0.9)))
# Inversion:  !(thrml ((Implication B A) (stv 0.87 0.81)))
# Revision:   !(thrml (A (stv 0.8 0.9)) (A (stv 0.3 0.7)))
```

> The input structure determines the rule automatically — no need to
> remember individual rule names or argument orders.

### Run tests

```bash
pytest tests/ -v                          # all implemented rules covered
pytest -m slow -v                        # scalability tests (trueagi-io/PLN examples)
```

## How it works

1. **Parameterize** — PLN `(stv s c)` → Beta(α, β) with mean = s
2. **Discretize** — discretize each Beta distribution into **K bins** (`CategoricalNode`). K controls accuracy vs compute: K=16 (default) for research, K=4 for TSU hardware deployment.
3. **Build graph** — each implication becomes a K×K weight table (log of the discretized conditional Beta); assemble all nodes and factors into a factor graph.
4. **Sample** — Block Gibbs sampling (50 batches × 2,000 samples). Root nodes can be **clamped** (fixed to their prior values as known evidence) while free nodes are updated by Gibbs sweeps.
5. **Recover** — moment-match the posterior histogram → `(strength, confidence)`

See [docs/beta-discretization.md](docs/beta-discretization.md) for details.

| PLN concept                       | thrml construct                             | Extropic hardware               |
| --------------------------------- | ------------------------------------------- | ------------------------------- |
| Proposition (VariableAtom)        | `CategoricalNode` (K=16 bins)               | pdit (K-category sampling cell) |
| Prior P(A)                        | Unary `CategoricalEBMFactor` (Beta prior)   | Bias field on pdit              |
| Implication A→B (strength s)      | Pairwise `SquareCategoricalEBMFactor` (K×K) | Coupling between pdits          |
| TruthValue (strength, confidence) | Beta posterior → moment-matching            | Posterior distribution readout  |
| Inference rule                    | Block Gibbs sampling                        | Thermal equilibration           |
| Deduction chain                   | Chain factor graph                          | Pipeline of coupled clusters    |
| Induction (V-shape)               | Star factor graph                           | Hub-and-spoke topology          |
| Abduction (inverted-V)            | Collider factor graph                       | Explaining-away circuit         |
| Revision (evidence merge)         | Multiple unary factors on one node          | Competing bias fields converge  |

## API reference

### Core engine (`pln_thrml`)

**Graph builders** — each returns a `dict` with `"nodes"`, `"factors"`, and metadata:

| Function | Topology |
| -------- | -------- |
| `build_beta_chain(priors, confidences, strengths, ...)` | Directed chain X₀→X₁→...→Xₙ (modus ponens, deduction) |
| `build_beta_v_graph(...)` | V-shape: Left←Root→Right (induction) |
| `build_beta_inv_v_graph(...)` | Inverted-V: Left→Center←Right (abduction) |
| `build_beta_symmetric_chain(...)` | Bidirectional chain (Similarity, Equivalence) |
| `build_beta_full_graph(priors, implications, ...)` | Arbitrary topology from a knowledge-base dict |

**Sampling & measurement**:

| Function | Returns |
| -------- | ------- |
| `sample_and_measure(graph, target_node)` | `(strength, confidence)` — one-step sampling + recovery |
| `run_beta_sampling(graph)` | Raw sample array for further analysis |
| `estimate_beta_marginal(samples, graph, node)` | Posterior histogram + `(s, c)` for a node |
| `estimate_beta_conditional(samples, graph, target, condition)` | P(target \| condition=True) via weighted posterior |
| `diagnose_convergence(samples, graph, node)` | R-hat, ESS, and convergence diagnostics |

**Conversion utilities**: `stv_to_beta_params(s, c)`, `posterior_to_stv(histogram, k)`, `c2w(c)` / `w2c(w)`, `bin_centers(k)`, `effective_k(c)`.

### MeTTa bridge (`pln_thrml.metta`)

| Function | Purpose |
| -------- | ------- |
| `register_all(metta)` | Register the `thrml` operator and type declarations with a MeTTa runner |

The `thrml` operator dispatches automatically based on premise structure —
see [Quick start](#quick-start) for examples of all supported input forms.

## Results

11 PLN rules compiled and verified (K=16, 100,000 samples). Strength errors
for representative parameter sets
(full per-parameter breakdown in [docs/results.md](docs/results.md)):

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

Full per-rule tables and divergence analysis: [docs/results.md](docs/results.md)

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

This project explores a different architectural choice: extracting Q_tv
(the uncertainty component) and compiling it to TSU hardware, while
Q_logic remains on CPU/GPU. This is not what RAPTL prescribes — RAPTL
keeps the triple bundled on GPU via ShardZipper. Our approach trades
RAPTL's joint optimization for hardware-native sampling, and 11 PLN
rules validate that Q_tv can be faithfully executed this way.

A possible heterogeneous pipeline extending this idea:

| Tier    | Hardware | Role                                                       |
| ------- | -------- | ---------------------------------------------------------- |
| Control | CPU      | Variable binding, rule scheduling                          |
| Compile | CPU+GPU  | Sparse tensor contraction, graph sharding (ShardZipper)    |
| Sample  | TSU      | Boltzmann sampling over compiled factor graphs              |

The Sample tier is more general than PLN alone — any algorithm reducible
to sampling from P(x) ∝ e^{−ℰ(x)} is a candidate. Current status:

| Algorithm                  | TSU status                                        |
| -------------------------- | ------------------------------------------------- |
| PLN truth-value inference  | **Validated** — 11 rules, this project            |
| Factor-graph BP (general)  | Native — Gibbs sampling implements message passing |
| MOSES / EDA program search | Plausible — EDA sampling step fits, not yet tested |

The three-tier pipeline is our projection, not described in the
references. Whether the gain from hardware-native sampling outweighs the
loss of RAPTL's joint optimization is an open question.

## Project structure

```
pln_thrml/                 Main package
  __init__.py              Public API re-exports (beta)
  beta.py                  Beta factor graph engine (builds, samples, recovers stv)
  metta/                   MeTTa integration layer (optional, requires hyperon)
    __init__.py            Entry point — exports register_all()
    dispatch.metta         MeTTa dispatch rules — pattern-matching rule selection
    rules.py               Grounded ops (10 rule builders) + atom helpers
    declarations/
      pln_types.metta      Type declarations (stv, Implication, Similarity, etc.)
vendor/PLN/                trueagi-io/PLN (git submodule) — test baselines
tests/
  conftest.py              Shared fixtures and tolerance constants (strength ±0.05, confidence ±0.15)
  test_factor_graph.py     Factor graph engine unit tests (K=4/8/16 parametrized)
  test_metta.py            MeTTa end-to-end: 11 rules + quantale algebra + topologies + extreme inputs
  test_scale.py            Scalability tests from trueagi-io/PLN examples (pytest -m slow)
docs/
  results.md               Full per-rule results tables
  pln-formulas.md          PLN truth-value confidence formulas
  beta-discretization.md   Beta-discretized approach details
  interactive_overview.html  Interactive visualisation (3 tabs: pipeline, 11 rules, energy)
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, testing, code
conventions, and pull request guidelines.

## Acknowledgements

- [Hyperon/PLN](https://github.com/trueagi-io/PLN) — TrueAGI team
- [thrml](https://github.com/extropic-ai/thrml) — Extropic AI

## License

[MIT](LICENSE) — Copyright (c) 2026 Xiaohan Ma
