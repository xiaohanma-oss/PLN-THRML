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
Unit (TSU) treats computation as energy minimisation.  The transform
`W = log P(child|parent)` bridges the two: every PLN conditional probability
table compiles directly into a Boltzmann energy table.

The question is whether this compilation preserves PLN's semantics — not just
strength, but also confidence.  This repo validates that it does: all 11 PLN
inference rules are compiled via Beta discretization and verified against
PLN's analytical formulas.

## How it works

1. **Parameterize** — PLN `(stv s c)` → Beta(α, β) with mean = s
2. **Build graph** — `W = log P`: each node becomes a K=16 bin `CategoricalNode`; each implication becomes a K×K energy factor
3. **Sample** — Block Gibbs sampling (50 batches × 2,000 samples)
4. **Recover** — moment-match the posterior histogram → `(strength, confidence)`

See [docs/beta-discretization.md](docs/beta-discretization.md) for details.

## Architecture mapping

| PLN concept | thrml construct | Extropic hardware |
|---|---|---|
| Proposition (VariableAtom) | `CategoricalNode` (K=16 bins) | p-bit cluster (multi-bit) |
| Prior P(A) | Unary `CategoricalEBMFactor` (Beta prior) | Bias field on p-bits |
| Implication A→B (strength s) | Pairwise `CategoricalEBMFactor` (K×K) | Coupling matrix between clusters |
| TruthValue (strength, confidence) | Beta posterior → moment-matching | Posterior distribution readout |
| Inference rule | Block Gibbs sampling | Thermal equilibration |
| Deduction chain | Chain factor graph | Pipeline of coupled clusters |
| Induction (V-shape) | Star factor graph | Hub-and-spoke topology |
| Abduction (inverted-V) | Collider factor graph | Explaining-away circuit |

## Installation

**From PyPI:**

```bash
pip install pln-thrml                     # core only (thrml + jax)
pip install pln-thrml[metta]              # + MeTTa bridge (requires hyperon)
```

**From source:**

```bash
git clone https://github.com/mafeifei666666/PLN-THRML.git
cd PLN-THRML
pip install -e .                          # editable install
```

> To run tests, also fetch the [trueagi-io/PLN](https://github.com/trueagi-io/PLN)
> test baselines: `git submodule update --init && pip install -e ".[dev]"`

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

## What this means for Hyperon

1. **Complete Q_tv on THRML**: The Hyperon whitepaper (2025) defines PLN
   inference as message-passing on a product quantale Q_logic × Q_tv.
   This project demonstrates that the entire Q_tv component — both the
   quantale product ⊗ (evidence combination at factors) and the quantale
   sum ⊕ (marginalization at variables) — can be compiled to thrml factor
   graphs via `W = log P` and executed through Gibbs sampling.  All 11 PLN
   rules verified end-to-end across multiple parameter sets constitute the
   evidence.  Q_logic (rule selection, structure discovery) remains with
   CPU/GPU; all truth-value computation has a complete path to
   thermodynamic hardware.

2. **Direct compilation path**: MeTTa PLN rules can be compiled to thrml
   factor graphs via the `W = log P` transform, then executed on Extropic's
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

## Project structure

```
pln_thrml_beta.py          Beta factor graph engine (builds, samples, recovers stv)
vendor/PLN/                trueagi-io/PLN (git submodule) — test baselines
metta/                     MeTTa integration layer (optional, requires hyperon)
  atoms.py                 Atom extraction from MeTTa space
  ops/
    rules.py               All 11 rules (declarative table + generic factory, including revision & negation)
  declarations/
    pln_types.metta        Type declarations (stv, Implication, Similarity, etc.)
tests/
  test_metta.py            All rules verified end-to-end via MeTTa
  test_factor_graph.py     Factor graph engine unit tests
  test_scale.py            Scalability tests from trueagi-io/PLN examples (pytest -m slow)
docs/
  results.md               Full per-rule results tables
  pln-formulas.md          PLN truth-value confidence formulas
  beta-discretization.md   Beta-discretized approach details
```

## Not yet covered

- **Cyclic inference**: Loopy graphs (A→B→C→A) — highest-leverage extension
- **EvidenceID / StampDisjoint**: Evidence tracking to prevent double-counting
  during revision (PLN uses `StampDisjoint` and `StampConcat`).
- **PLN.Derive**: Priority-queue based iterative inference engine with
  bounded belief buffer.
- **ECAN attention**: Attention allocation for prioritizing which
  subgraphs to sample first.
- **Continuous-valued nodes**: Extending beyond discrete propositions.

## Contributing

```bash
git clone --recurse-submodules https://github.com/mafeifei666666/PLN-THRML.git
cd PLN-THRML
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

## License

[MIT](LICENSE) — Copyright (c) 2025 Xiaohan Ma
