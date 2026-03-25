# PLN on Thermodynamic Hardware: A Working Demo

## The idea in 30 seconds

PLN treats inference as probability propagation; the Thermodynamic Sampling
Unit (TSU) treats computation as energy minimisation.  The transform
`W = log P(child|parent)` bridges the two: every PLN conditional probability
table compiles directly into a Boltzmann energy table.  Thermal fluctuations
in the hardware perform inference: the equilibrium distribution over p-bit
states **is** the PLN truth value.

This repo demonstrates the compilation for PLN inference rules and verifies
each by comparing Gibbs-sampled conditionals against PLN's analytical
formulas.  Together, this validates that PLN inference rules compile
faithfully to undirected factor graphs and produce results consistent
with PLN's analytical formulas under Gibbs sampling.

## Architecture mapping

| PLN concept | thrml construct | Extropic hardware |
|---|---|---|
| Proposition (VariableAtom) | `CategoricalNode` (2 states) | p-bit cluster |
| Prior P(A) | Unary `CategoricalEBMFactor` | Bias field on p-bits |
| Implication A→B (strength s) | Pairwise `CategoricalEBMFactor` | Coupling matrix between clusters |
| TruthValue.strength | `W = log P` (energy weight) | Physical coupling strength |
| TruthValue.confidence | `W_scaled = c · W` (temperature) | Effective temperature |
| Inference rule | Block Gibbs sampling | Thermal equilibration |
| Deduction chain | Chain factor graph | Pipeline of coupled clusters |
| Induction (V-shape) | Star factor graph | Hub-and-spoke topology |
| Abduction (inverted-V) | Collider factor graph | Explaining-away circuit |

## Quick start

```bash
pip install thrml          # requires Python 3.10+
python pln_deduction.py    # run any script
```

## Results

Every script runs multiple parameter settings and verifies against analytical
formulas with <2% tolerance (20,000 total samples: 4 batches × 5,000).
Representative results from a single run:

### Modus Ponens — `A, A→B ⊢ B`

| Setting | PLN | Gibbs | Error |
|---|---|---|---|
| Strong prior, strong rule (s_A=0.8, s_AB=0.9) | 0.7240 | 0.7255 | 0.0015 |
| Coin-flip prior, strong rule (s_A=0.5, s_AB=0.95) | 0.4850 | 0.4895 | 0.0045 |
| Rare antecedent (s_A=0.1, s_AB=0.8) | 0.0980 | 0.0978 | 0.0002 |

### Deduction — `A→B, B→C ⊢ A→C`

| Setting | PLN | Gibbs | Error |
|---|---|---|---|
| Standard chain (s_AB=0.9, s_BC=0.85) | 0.7850 | 0.7842 | 0.0008 |
| High-confidence (s_AB=0.95, s_BC=0.9) | 0.8625 | 0.8557 | 0.0068 |
| No-information (all 0.5) | 0.5000 | 0.4953 | 0.0047 |

### Inversion — `A→B ⊢ B→A`

| Setting | Bayes | Gibbs | Error |
|---|---|---|---|
| Rare A, near-deterministic (s_A=0.2, s_AB=0.99) | 0.8319 | 0.8202 | 0.0117 |
| Coin-flip prior (s_A=0.5, s_AB=0.8) | 0.8000 | 0.8000 | 0.0000 |

### Induction — `A→B, A→C ⊢ B→C`

| Setting | Analytical | Gibbs | Error |
|---|---|---|---|
| Strong shared cause | 0.7597 | 0.7615 | 0.0018 |
| Symmetric links | 0.8008 | 0.7963 | 0.0045 |

### Abduction — `A→B, C→B ⊢ A→C`

| Setting | Analytical | Gibbs | Error |
|---|---|---|---|
| Two strong causes | 0.8125 | 0.8179 | 0.0054 |
| Symmetric causes | 0.6800 | 0.6664 | 0.0136 |

### Scaling (deduction chain, strength=0.9, background=0.2)

| Chain length | Analytical | Gibbs | Error | Time |
|---|---|---|---|---|
| 3 nodes | 0.8300 | 0.8215 | 0.0085 | 1.1s |
| 10 nodes | 0.6801 | 0.6929 | 0.0128 | 0.9s |
| 20 nodes | 0.6670 | 0.6660 | 0.0010 | 1.4s |

Block Gibbs with 2-coloring scales well: each sweep updates all nodes in
2 parallel steps regardless of chain length.

## What this means for Hyperon

1. **Direct compilation path**: MeTTa PLN rules can be compiled to thrml
   factor graphs via the `W = log P` transform, then executed on Extropic's
   TSU.  No approximation is introduced — the factor graph encodes the
   exact same joint distribution that PLN reasons over.

2. **Automatic inversion**: The factor graph encodes the joint P(A,B),
   so Gibbs sampling recovers *both* P(B|A) and P(A|B) without separate
   inversion rules.  Hardware naturally performs Bayes' rule.

3. **Composability**: Rules combine by adding factors to the graph.
   Deduction chains, V-shapes, and collider topologies all work with the
   same compilation transform.

4. **Scaling**: Block Gibbs with graph coloring enables parallel updates.
   A 20-node deduction chain runs in 1.5s on CPU; on a TSU, thermal
   equilibration would be near-instantaneous.

## File overview

| File | Rule | Topology | Nodes |
|---|---|---|---|
| `pln_thrml.py` | Core library | — | — |
| `pln_modus_ponens.py` | A, A→B ⊢ B | A → B | 2 |
| `pln_deduction.py` | A→B, B→C ⊢ A→C | A → B → C | 3 |
| `pln_inversion.py` | A→B ⊢ B→A | A → B | 2 |
| `pln_induction.py` | A→B, A→C ⊢ B→C | B ← A → C | 3 |
| `pln_abduction.py` | A→B, C→B ⊢ A→C | A → B ← C | 3 |
| `pln_revision.py` | Merge evidence | Single node | 1 |
| `pln_negation.py` | ¬A | Single node | 1 |
| `pln_symmetric_modus_ponens.py` | A, A~B ⊢ B (Similarity) | A → B | 2 |
| `pln_equiv_to_impl.py` | A≡B ⊢ A→B | A ↔ B | 2 |
| `pln_transitive_similarity.py` | A~B, B~C ⊢ A~C | A ↔ B ↔ C | 3 |
| `pln_golden_tests.py` | All truth functions vs upstream | — | — |
| `benchmarks/pln_scaling.py` | Deduction chain scaling | X₀→X₁→...→X_n | 3–20 |
| `experiments/pln_evidence_is_energy.py` | Evidence = Energy proof (5 experiments) | — | — |

## Production PLN truth-value formulas (trueagi-io/PLN)

Production PLN truth functions (exact match to `lib_pln.metta`) are
implemented in `pln_thrml.py` and validated against upstream golden
tests in `pln_golden_tests.py`:

| Rule | Confidence formula |
|---|---|
| Deduction | `s_AB · s_BC · c_AB · c_BC` |
| Modus Ponens | `s_A · s_AB · c_A · c_AB` |
| Inversion | `c_B · c_AB · 0.6` (heuristic) |
| Induction | `w2c(s_CB · c_CB · c_CA)` |
| Abduction | `w2c(s_AB · c_AB · c_CB)` |
| Revision | `min(1.0, max(w2c(w1+w2), c1, c2))` |
| Negation | `c` (unchanged) |
| Symmetric Modus Ponens | `c_A · c_AB · truth_or(s_A, s_AB)` |
| Equivalence→Implication | `c_AB` (unchanged) |
| Transitive Similarity | `c_AB · c_BC · truth_or(s_AB, s_BC)` |
| Evaluation Implication | `s_AB · s_AC · c_AB · c_AC` |

Where `c2w(c) = c/(1-c)` and `w2c(w) = w/(w+1)`.

**Modus Ponens strength formula** includes a background (leak) term:
`s_B = s_A · s_AB + 0.02 · (1 − s_A)`.  Even when A is false, B can still
be true with probability ε = 0.02.  The result tables above reflect this
complete formula (see `pln_modus_ponens.py` and `pln_thrml.py`).

**Key findings:**
- **Inversion**: Production PLN uses a heuristic (strength unchanged,
  confidence penalized), while the factor graph gives the exact Bayesian
  answer. Both are shown in `pln_inversion.py`.
- **Revision**: PLN uses arithmetic weighted average; Boltzmann energy
  addition gives geometric combination. Both are shown in `pln_revision.py`.

## Hardware constraints (TSU)

The Thermodynamic Sampling Unit (arXiv:2510.23972) imposes hard constraints
that affect how PLN graphs map to silicon:

- **Sparse connectivity (~12 neighbors per variable)**: Each p-bit couples
  to roughly 12 others.  If a PLN node participates in more than 12
  implication relationships, the subgraph cannot be mapped directly to a
  single TSU tile and must be partitioned or virtualized.

- **Ising (binary) variables**: The TSU natively supports two-state p-bits.
  PLN truth values live in [0, 1], so encoding a continuous strength
  requires multi-bit thermometer or binary encoding — each logical
  proposition then occupies several physical p-bits.

These constraints do not affect the correctness of the factor-graph
compilation demonstrated here (which runs on CPU via `thrml`), but they
are load-bearing for any physical deployment.

## Architectural differences from upstream PLN

The upstream [trueagi-io/PLN](https://github.com/trueagi-io/PLN) is a
symbolic forward-chaining reasoner in MeTTa.  This project takes a
fundamentally different approach — compiling PLN rules into factor graphs
for thermodynamic sampling.  Several upstream features do not have direct
analogs:

| Upstream feature | Why not needed here |
|---|---|
| `PLN.Derive` / `PLN.Query` (forward chaining) | Factor graph sampling explores the entire joint distribution simultaneously — no sequential rule application needed |
| `StampDisjoint` / `StampConcat` (evidence tracking) | Independence is enforced by graph structure, not bookkeeping stamps |
| `BestCandidate` / `LimitSize` / `ConfidenceRank` (priority queue) | Serve the forward chainer's resource allocation; sampling has no priority queue |
| Rule guards (`SyllogisticRuleGuard`, etc.) | Structural constraints are enforced at graph construction time |
| `Member Deduction` / `Evaluation Inheritance` | MeTTa-specific higher-order constructs (MemberLink, EvaluationLink) with no direct binary-proposition analog |

The key insight is that the factor graph approach replaces PLN's
sequential rule application with parallel thermodynamic equilibration.
Where PLN applies one rule at a time (controlled by AIKR resource
management), the factor graph encodes *all* relationships simultaneously
and sampling recovers *all* conditionals at once.

## Evidence = Energy: the theoretical bridge

Goertzel (2026) argues that "evidence conservation is to logic what energy
conservation is to physics."  This repo proves the correspondence is not
merely an analogy — the TSU **physically realizes** evidence conservation.

The bridge is `W = log P`:

| PLN (evidence) | W = log P | TSU (energy) |
|---|---|---|
| Strength s | W = log(s) | Boltzmann weight |
| Evidence combination s₁ × s₂ | log(s₁) + log(s₂) | Energy addition |
| Best inference path (max) | max of energy sums | Lowest free energy |
| Confidence c | W_scaled = c · W | Inverse temperature (1/kT) |
| No hallucination | Chain converges to steady-state | Thermodynamic bound |
| Evidence conservation | Factor graph topology | Energy conservation |

`experiments/pln_evidence_is_energy.py` verifies this with five experiments:

1. **Correspondence table** — `_safe_log` maps every PLN operation to its
   thermodynamic counterpart, verified numerically.
2. **Noether's theorem** — reinforcement ρ = f × g is near-constant along
   inference chains at thermal equilibrium (Theorem 3.1).
3. **Hallucination bound** — chains converge to the transition matrix
   steady-state; the factor graph cannot amplify evidence (Theorem 4.2).
4. **Confidence = 1/kT** — sweeping confidence from 0→1 smoothly
   interpolates between maximum entropy (flat) and full evidence (sharp).
5. **Multi-path equilibrium** — a diamond graph (two paths to the same
   conclusion) reaches a single equilibrium that correctly combines evidence
   from both paths without double-counting.

```bash
python experiments/pln_evidence_is_energy.py
```

## Not yet covered

- **ECAN attention**: Attention allocation for prioritizing which
  subgraphs to sample first.

- **Continuous-valued nodes**: Extending beyond binary propositions.

## References

1. Goertzel, Ikle, Goertzel, Heljakka.
   *Probabilistic Logic Networks* (Springer, 2008).

2. Extropic AI. *thrml: Thermodynamic Hypergraphical Model Library*.
   https://github.com/extropic-ai/thrml

3. TrueAGI. *PLN (Probabilistic Logic Networks)*.
   https://github.com/trueagi-io/PLN

4. TrueAGI. *PLN Experimental (mathematical foundations)*.
   https://github.com/trueagi-io/pln-experimental

5. Jelinčič, Lockwood, Garlapati, Schillinger, Chuang, Verdon, McCourt.
   *An efficient probabilistic hardware architecture for diffusion-like models*.
   arXiv:2510.23972, 2025.

6. Goertzel. *Genenergy for Logic: Quantale Action, Evidence Conservation,
   and a Logical Analogue of 'Freeman Transformers'* (2026).

7. Goertzel. *Five Theorems on Evidence Conservation in Quantale-Based
   Inference Control* (2026).
