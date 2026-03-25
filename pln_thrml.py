"""
pln_thrml.py — Core library: compile PLN inference rules into thrml factor graphs
=================================================================================

The transform:  W = log P(child|parent)
converts every PLN conditional-probability table into a Boltzmann energy table
that thrml can sample via block Gibbs.

Propositions  →  CategoricalNode (2 states: 0=False, 1=True)
Implications  →  CategoricalEBMFactor (weights = log CPT)
Inference     →  Gibbs sampling recovers P(target|evidence)

Production PLN truth-value formulas (trueagi-io/PLN lib_pln.metta) are
included at the bottom of this file: STV dataclass, c2w/w2c, confidence
formulas for all rules, and consistency conditions.
"""

from dataclasses import dataclass
import jax
import jax.numpy as jnp
import numpy as np

from thrml.block_management import Block
from thrml.block_sampling import BlockGibbsSpec, sample_states, SamplingSchedule
from thrml.pgm import CategoricalNode
from thrml.models.discrete_ebm import CategoricalEBMFactor, CategoricalGibbsConditional
from thrml.factor import FactorSamplingProgram

# ── constants ────────────────────────────────────────────────────────────
N_CATS = 2          # binary propositions
EPS = 1e-7          # clamp for log-safety
DEFAULT_EPSILON = 0.02  # PLN modus-ponens background rate

# ── sampling defaults ────────────────────────────────────────────────────
DEFAULT_N_BATCHES = 4
DEFAULT_SCHEDULE = SamplingSchedule(n_warmup=500, n_samples=5000,
                                    steps_per_sample=3)


# ═══════════════════════════════════════════════════════════════════════════
#  Factor builders
# ═══════════════════════════════════════════════════════════════════════════

def _safe_log(p):
    """log(p) with clamping to avoid -inf."""
    return float(jnp.log(jnp.clip(p, EPS, 1.0 - EPS)))


def make_prior_factor(node, strength):
    """Unary bias: P(node=1) = strength.

    Weight shape (1, 2):  [[log(1-s), log(s)]]
    """
    w = jnp.array([[_safe_log(1.0 - strength), _safe_log(strength)]])
    return CategoricalEBMFactor([Block([node])], w)


def make_implication_factor(parent, child, strength, background):
    """Pairwise coupling: P(child=1|parent=1) = strength,
                          P(child=1|parent=0) = background.

    Weight shape (1, 2, 2):
        [[[log(1-bg), log(bg)],      ← parent=0
          [log(1-s),  log(s) ]]]     ← parent=1
    """
    w = jnp.array([
        [[_safe_log(1.0 - background), _safe_log(background)],
         [_safe_log(1.0 - strength),   _safe_log(strength)]]
    ])
    return CategoricalEBMFactor([Block([parent]), Block([child])], w)


def make_confidence_prior(node, strength, confidence):
    """Unary bias with confidence-as-temperature: W = c · log(P).

    At confidence=1.0: full-strength prior.
    At confidence→0:  flat prior (maximum entropy).
    """
    w = jnp.array([[confidence * _safe_log(1.0 - strength),
                     confidence * _safe_log(strength)]])
    return CategoricalEBMFactor([Block([node])], w)


def make_confidence_implication(parent, child, strength, background, confidence):
    """Pairwise coupling with confidence scaling: W = c · log(CPT).

    At confidence=1.0: full coupling.
    At confidence→0:  no coupling (child ignores parent).
    """
    w = jnp.array([
        [[confidence * _safe_log(1.0 - background),
          confidence * _safe_log(background)],
         [confidence * _safe_log(1.0 - strength),
          confidence * _safe_log(strength)]]
    ])
    return CategoricalEBMFactor([Block([parent]), Block([child])], w)


# ═══════════════════════════════════════════════════════════════════════════
#  Graph builders
# ═══════════════════════════════════════════════════════════════════════════

def build_chain(priors, strengths, backgrounds):
    """Build a directed chain  X_0 → X_1 → ... → X_{n-1}.

    Parameters
    ----------
    priors : list[float]        P(X_i=1) for each node (only X_0 used as bias)
    strengths : list[float]     P(X_{i+1}=1 | X_i=1) for each edge
    backgrounds : list[float]   P(X_{i+1}=1 | X_i=0) for each edge

    Returns
    -------
    dict with keys: nodes, factors, free_blocks, spec, program
    """
    n = len(priors)
    nodes = [CategoricalNode() for _ in range(n)]

    factors = [make_prior_factor(nodes[0], priors[0])]
    for i in range(n - 1):
        factors.append(make_implication_factor(
            nodes[i], nodes[i + 1], strengths[i], backgrounds[i]))

    # 2-coloring: even indices / odd indices
    even = [nodes[i] for i in range(0, n, 2)]
    odd  = [nodes[i] for i in range(1, n, 2)]
    free_blocks = [Block(even), Block(odd)] if odd else [Block(even)]
    spec = BlockGibbsSpec(free_blocks, [])
    sampler = CategoricalGibbsConditional(N_CATS)

    prog = FactorSamplingProgram(
        gibbs_spec=spec,
        samplers=[sampler] * len(free_blocks),
        factors=factors,
        other_interaction_groups=[],
    )
    return dict(nodes=nodes, factors=factors, free_blocks=free_blocks,
                spec=spec, program=prog, n=n)


def build_v_graph(root_prior, left_strength, right_strength,
                  left_background, right_background):
    """V-shape:  Left ← Root → Right   (induction topology).

    Returns dict with keys: root, left, right, factors, free_blocks, spec, program
    """
    root = CategoricalNode()
    left = CategoricalNode()
    right = CategoricalNode()

    factors = [
        make_prior_factor(root, root_prior),
        make_implication_factor(root, left, left_strength, left_background),
        make_implication_factor(root, right, right_strength, right_background),
    ]

    # Coloring: {left, right} and {root}
    free_blocks = [Block([left, right]), Block([root])]
    spec = BlockGibbsSpec(free_blocks, [])
    sampler = CategoricalGibbsConditional(N_CATS)

    prog = FactorSamplingProgram(
        gibbs_spec=spec,
        samplers=[sampler, sampler],
        factors=factors,
        other_interaction_groups=[],
    )
    return dict(root=root, left=left, right=right,
                factors=factors, free_blocks=free_blocks,
                spec=spec, program=prog)


def build_inv_v_graph(left_prior, right_prior,
                      left_strength, right_strength,
                      left_background, right_background):
    """Inverted-V:  Left → Center ← Right   (abduction topology).

    Returns dict with keys: left, center, right, factors, free_blocks, spec, program
    """
    left = CategoricalNode()
    center = CategoricalNode()
    right = CategoricalNode()

    factors = [
        make_prior_factor(left, left_prior),
        make_prior_factor(right, right_prior),
        make_implication_factor(left, center, left_strength, left_background),
        make_implication_factor(right, center, right_strength, right_background),
    ]

    # Coloring: {left, right} and {center}
    free_blocks = [Block([left, right]), Block([center])]
    spec = BlockGibbsSpec(free_blocks, [])
    sampler = CategoricalGibbsConditional(N_CATS)

    prog = FactorSamplingProgram(
        gibbs_spec=spec,
        samplers=[sampler, sampler],
        factors=factors,
        other_interaction_groups=[],
    )
    return dict(left=left, center=center, right=right,
                factors=factors, free_blocks=free_blocks,
                spec=spec, program=prog)


def build_symmetric_pair(prior_a, prior_b, strength, background):
    """Symmetric 2-node graph:  A ↔ B  (Equivalence / Similarity topology).

    Both A→B and B→A have the same strength and background, creating
    a symmetric coupling.  Each node also gets its own prior bias.

    Returns dict with keys: a, b, factors, free_blocks, spec, program
    """
    a = CategoricalNode()
    b = CategoricalNode()

    factors = [
        make_prior_factor(a, prior_a),
        make_prior_factor(b, prior_b),
        make_implication_factor(a, b, strength, background),
        make_implication_factor(b, a, strength, background),
    ]

    free_blocks = [Block([a]), Block([b])]
    spec = BlockGibbsSpec(free_blocks, [])
    sampler = CategoricalGibbsConditional(N_CATS)

    prog = FactorSamplingProgram(
        gibbs_spec=spec,
        samplers=[sampler, sampler],
        factors=factors,
        other_interaction_groups=[],
    )
    return dict(a=a, b=b, factors=factors, free_blocks=free_blocks,
                spec=spec, program=prog)


def build_symmetric_chain(priors, strengths, backgrounds):
    """Symmetric chain  X_0 ↔ X_1 ↔ ... ↔ X_{n-1}  (Similarity topology).

    Each edge has bidirectional coupling with the same strength/background.
    Only X_0 gets a prior bias (matching the directed chain convention).

    Parameters
    ----------
    priors : list[float]        P(X_i=1) for each node (only X_0 used as bias)
    strengths : list[float]     coupling strength for each edge (both directions)
    backgrounds : list[float]   background rate for each edge (both directions)

    Returns
    -------
    dict with keys: nodes, factors, free_blocks, spec, program, n
    """
    n = len(priors)
    nodes = [CategoricalNode() for _ in range(n)]

    factors = [make_prior_factor(nodes[0], priors[0])]
    for i in range(n - 1):
        # Bidirectional coupling
        factors.append(make_implication_factor(
            nodes[i], nodes[i + 1], strengths[i], backgrounds[i]))
        factors.append(make_implication_factor(
            nodes[i + 1], nodes[i], strengths[i], backgrounds[i]))

    # 2-coloring: even indices / odd indices
    even = [nodes[i] for i in range(0, n, 2)]
    odd  = [nodes[i] for i in range(1, n, 2)]
    free_blocks = [Block(even), Block(odd)] if odd else [Block(even)]
    spec = BlockGibbsSpec(free_blocks, [])
    sampler = CategoricalGibbsConditional(N_CATS)

    prog = FactorSamplingProgram(
        gibbs_spec=spec,
        samplers=[sampler] * len(free_blocks),
        factors=factors,
        other_interaction_groups=[],
    )
    return dict(nodes=nodes, factors=factors, free_blocks=free_blocks,
                spec=spec, program=prog, n=n)


# ═══════════════════════════════════════════════════════════════════════════
#  Sampling
# ═══════════════════════════════════════════════════════════════════════════

def run_sampling(graph, seed=42, n_batches=None, schedule=None):
    """Run block Gibbs sampling on a compiled factor graph.

    Returns raw samples: list of arrays, one per observed block.
    """
    if n_batches is None:
        n_batches = DEFAULT_N_BATCHES
    if schedule is None:
        schedule = DEFAULT_SCHEDULE

    spec = graph["spec"]
    prog = graph["program"]
    key = jax.random.key(seed)

    init_state = []
    for block in spec.free_blocks:
        key, subkey = jax.random.split(key)
        init_state.append(
            jax.random.randint(subkey, (n_batches, len(block.nodes)),
                               minval=0, maxval=N_CATS, dtype=jnp.uint8)
        )

    keys = jax.random.split(key, n_batches)
    observe_blocks = list(spec.free_blocks)

    samples = jax.jit(jax.vmap(
        lambda s, k: sample_states(k, prog, schedule, s, [], observe_blocks)
    ))(init_state, keys)

    return samples


# ═══════════════════════════════════════════════════════════════════════════
#  Measurement helpers
# ═══════════════════════════════════════════════════════════════════════════

def _flatten_node(samples, block_idx, node_within_block):
    """Extract a single node's samples as a flat float32 array."""
    return samples[block_idx][:, :, node_within_block].flatten().astype(jnp.float32)


def _node_location(graph, node):
    """Find (block_idx, position_within_block) for a node.

    Works for both chain graphs (nodes list) and named graphs.
    """
    for bi, block in enumerate(graph["free_blocks"]):
        for ni, n in enumerate(block.nodes):
            if n is node:
                return bi, ni
    raise ValueError("Node not found in any free block")


def estimate_marginal(samples, graph, node):
    """P(node=1) from samples."""
    bi, ni = _node_location(graph, node)
    flat = _flatten_node(samples, bi, ni)
    return float(jnp.mean(flat))


def estimate_conditional(samples, graph, target, condition, cond_val=1):
    """P(target=1 | condition=cond_val) from samples."""
    tbi, tni = _node_location(graph, target)
    cbi, cni = _node_location(graph, condition)

    t_flat = _flatten_node(samples, tbi, tni)
    c_flat = _flatten_node(samples, cbi, cni)

    mask = (c_flat == cond_val)
    count = float(jnp.sum(mask))
    if count < 1:
        return float("nan")
    return float(jnp.sum(t_flat * mask) / count)


# ═══════════════════════════════════════════════════════════════════════════
#  PLN analytical formulas  (pure Python, no JAX)
# ═══════════════════════════════════════════════════════════════════════════

def pln_modus_ponens_strength(s_A, s_AB, epsilon=DEFAULT_EPSILON):
    """P(B=1) = s_A * s_AB + epsilon * (1 - s_A)"""
    return s_A * s_AB + epsilon * (1.0 - s_A)


def pln_deduction_strength(s_AB, s_BC, s_C0):
    """P(C=1|A=1) = s_AB * s_BC + (1 - s_AB) * s_C0

    Equivalent to full PLN:  s_AB*s_BC + (1-s_AB)*(s_C - s_B*s_BC)/(1-s_B)
    since (s_C - s_B*s_BC)/(1-s_B) = s_C0 by total probability.
    """
    return s_AB * s_BC + (1.0 - s_AB) * s_C0


def pln_inversion_strength(s_AB, s_A, s_B0):
    """P(A=1|B=1) via Bayes' rule.

    P(B) = s_A*s_AB + (1-s_A)*s_B0
    P(A|B) = s_AB * s_A / P(B)
    """
    p_B = s_A * s_AB + (1.0 - s_A) * s_B0
    if p_B < 1e-12:
        return 0.0
    return s_AB * s_A / p_B


def bayesnet_conditional_v(root_prior, left_s, right_s, left_bg, right_bg):
    """Exact P(Right=1|Left=1) for V-shape  Left ← Root → Right.

    P(Right=1|Left=1) = Σ_r P(Right=1|Root=r) * P(Root=r|Left=1)

    where P(Root=r|Left=1) comes from Bayes on Root→Left.
    """
    # P(Left=1|Root=r)
    p_L_given_R1 = left_s
    p_L_given_R0 = left_bg

    # P(Left=1) = P(L|R=1)*P(R=1) + P(L|R=0)*P(R=0)
    p_L = p_L_given_R1 * root_prior + p_L_given_R0 * (1.0 - root_prior)

    if p_L < 1e-12:
        return 0.0

    # P(Root=1|Left=1) via Bayes
    p_R1_given_L1 = p_L_given_R1 * root_prior / p_L

    # P(Right=1|Left=1) = P(Ri=1|Ro=1)*P(Ro=1|L=1) + P(Ri=1|Ro=0)*P(Ro=0|L=1)
    return right_s * p_R1_given_L1 + right_bg * (1.0 - p_R1_given_L1)


def bayesnet_conditional_inv_v(left_prior, right_prior,
                               left_s, right_s,
                               left_bg, right_bg):
    """Exact P(Right=1|Left=1) for inverted-V  Left → Center ← Right.

    Uses full joint enumeration over (Left, Center, Right).
    """
    total_with_L1 = 0.0
    target_with_L1 = 0.0

    for l in (0, 1):
        for c in (0, 1):
            for r in (0, 1):
                # P(Left=l)
                p_l = left_prior if l == 1 else (1.0 - left_prior)
                # P(Right=r)
                p_r = right_prior if r == 1 else (1.0 - right_prior)
                # P(Center=c | Left=l)
                if l == 1:
                    p_c_l = left_s if c == 1 else (1.0 - left_s)
                else:
                    p_c_l = left_bg if c == 1 else (1.0 - left_bg)
                # P(Center=c | Right=r)
                if r == 1:
                    p_c_r = right_s if c == 1 else (1.0 - right_s)
                else:
                    p_c_r = right_bg if c == 1 else (1.0 - right_bg)

                # Joint under factored model:
                # P(L,C,R) ∝ P(L) * P(R) * P(C|L) * P(C|R)
                joint = p_l * p_r * p_c_l * p_c_r

                if l == 1:
                    total_with_L1 += joint
                    if r == 1:
                        target_with_L1 += joint

    if total_with_L1 < 1e-12:
        return 0.0
    return target_with_L1 / total_with_L1


def inv_v_marginal(left_prior, right_prior,
                   left_s, right_s,
                   left_bg, right_bg, target="left"):
    """Exact marginal P(target=1) under the factored inverted-V model.

    The factored model P(L,C,R) ∝ P(L)*P(R)*P(C|L)*P(C|R) shifts
    marginals from the priors because both parents couple through C.
    """
    total = 0.0
    target_sum = 0.0

    for l in (0, 1):
        for c in (0, 1):
            for r in (0, 1):
                p_l = left_prior if l == 1 else (1.0 - left_prior)
                p_r = right_prior if r == 1 else (1.0 - right_prior)
                p_c_l = (left_s if c == 1 else (1.0 - left_s)) if l == 1 \
                    else (left_bg if c == 1 else (1.0 - left_bg))
                p_c_r = (right_s if c == 1 else (1.0 - right_s)) if r == 1 \
                    else (right_bg if c == 1 else (1.0 - right_bg))
                joint = p_l * p_r * p_c_l * p_c_r
                total += joint
                t_val = l if target == "left" else r
                if t_val == 1:
                    target_sum += joint

    return target_sum / total if total > 1e-12 else 0.0


def chain_conditional(n, strengths, backgrounds):
    """Exact P(X_{n-1}=1 | X_0=1) for a chain via iterative matrix multiply.

    Transition matrix T[parent, child]:
        T[0,:] = [1-bg, bg]
        T[1,:] = [1-s,  s ]
    """
    # Start with P(X_0=0|X_0=1) = 0,  P(X_0=1|X_0=1) = 1
    state = np.array([0.0, 1.0])  # [P(X=0|evidence), P(X=1|evidence)]

    for i in range(n - 1):
        s, bg = strengths[i], backgrounds[i]
        T = np.array([[1.0 - bg, bg],
                       [1.0 - s,  s]])
        state = state @ T

    return state[1]  # P(X_{n-1}=1 | X_0=1)


# ═══════════════════════════════════════════════════════════════════════════
#  Reporting
# ═══════════════════════════════════════════════════════════════════════════

def compare(name, analytical, sampled, tol=0.02):
    """Print one comparison line, return whether it passed."""
    err = abs(sampled - analytical)
    passed = err < tol
    mark = "PASS" if passed else "FAIL"
    print(f"  {name:<32s}  analytical={analytical:.4f}  "
          f"gibbs={sampled:.4f}  err={err:.4f}  [{mark}]")
    return passed


# ═══════════════════════════════════════════════════════════════════════════
#  STV (Simple Truth Value) — production PLN (trueagi-io/PLN lib_pln.metta)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class STV:
    """Simple Truth Value: (strength, confidence).

    strength:   first-order probability estimate
    confidence: certainty of the estimate, encodes evidence weight
    """
    strength: float
    confidence: float = 1.0

    def __repr__(self):
        return f"(stv {self.strength:.6f} {self.confidence:.6f})"


def c2w(c):
    """Confidence → evidence weight.  lib_pln.metta: Truth_c2w = c/(1-c)."""
    if c >= 1.0:
        return float('inf')
    if c <= 0.0:
        return 0.0
    return c / (1.0 - c)


def w2c(w):
    """Evidence weight → confidence.  lib_pln.metta: Truth_w2c = w/(w+1)."""
    if w < 0:
        return 0.0
    return w / (w + 1.0)


# ═══════════════════════════════════════════════════════════════════════════
#  Consistency conditions (PLN book 5.2.2.2)
# ═══════════════════════════════════════════════════════════════════════════

def _clamp(v, lo, hi):
    return min(hi, max(v, lo))


def smallest_intersection_probability(sA, sB):
    if sA <= 0:
        return 0.0
    return _clamp((sA + sB - 1.0) / sA, 0.0, 1.0)


def largest_intersection_probability(sA, sB):
    if sA <= 0:
        return 1.0
    return _clamp(sB / sA, 0.0, 1.0)


def conditional_probability_consistency(sA, sB, sAB):
    """Check if P(B|A)=sAB is consistent with marginals P(A)=sA, P(B)=sB."""
    if sA <= 0:
        return False
    lo = smallest_intersection_probability(sA, sB)
    hi = largest_intersection_probability(sA, sB)
    return lo <= sAB <= hi


# ═══════════════════════════════════════════════════════════════════════════
#  Production PLN truth functions (exact match to lib_pln.metta)
# ═══════════════════════════════════════════════════════════════════════════

def truth_deduction(stv_A, stv_B, stv_C, stv_AB, stv_BC):
    """A→B, B→C ⊢ A→C.  lib_pln.metta: Truth_Deduction.

    Args:
        stv_A, stv_B, stv_C: node STVs
        stv_AB: STV of A→B premise
        stv_BC: STV of B→C premise
    Returns: STV of A→C conclusion
    """
    sA, cA = stv_A.strength, stv_A.confidence
    sB, cB = stv_B.strength, stv_B.confidence
    sC, cC = stv_C.strength, stv_C.confidence
    sAB, cAB = stv_AB.strength, stv_AB.confidence
    sBC, cBC = stv_BC.strength, stv_BC.confidence

    if not (conditional_probability_consistency(sA, sB, sAB) and
            conditional_probability_consistency(sB, sC, sBC)):
        return STV(1.0, 0.0)

    if sB > 0.9999:
        s = sC
    else:
        s = sAB * sBC + (1.0 - sAB) * (sC - sB * sBC) / (1.0 - sB)

    c = sAB * sBC * cAB * cBC
    return STV(s, c)


def truth_modus_ponens(stv_A, stv_AB):
    """A, A→B ⊢ B.  lib_pln.metta: Truth_ModusPonens.

    Args:
        stv_A:  STV of antecedent A
        stv_AB: STV of implication A→B
    Returns: STV of conclusion B
    """
    sA, cA = stv_A.strength, stv_A.confidence
    sAB, cAB = stv_AB.strength, stv_AB.confidence

    s = sA * sAB + 0.02 * (1.0 - sA)
    c = sA * sAB * cA * cAB
    return STV(s, c)


def truth_inversion(stv_B, stv_AB):
    """A→B ⊢ B→A.  lib_pln.metta: Truth_inversion.

    NOTE: Production PLN heuristic — strength unchanged, confidence penalized.
    The factor graph gives the Bayesian answer instead (use pln_inversion_strength).

    Args:
        stv_B:  STV of node B (target of original implication)
        stv_AB: STV of A→B premise
    Returns: STV of B→A conclusion
    """
    s = stv_AB.strength
    c = stv_B.confidence * stv_AB.confidence * 0.6
    return STV(s, c)


def truth_induction(stv_A, stv_B, stv_C, stv_CA, stv_CB):
    """C→A, C→B ⊢ A→B.  lib_pln.metta: Truth_Induction.

    Note: MeTTa names the premise params ($sBA, $sBC) but they correspond
    to C→A and C→B in the inference rule.
    """
    sA = stv_A.strength
    sB = stv_B.strength
    sCA, cCA = stv_CA.strength, stv_CA.confidence
    sCB, cCB = stv_CB.strength, stv_CB.confidence

    if sA <= 1e-10 or abs(1.0 - sB) < 1e-10:
        return STV(0.5, 0.0)

    term1 = (sCA * sCB * sB) / sA
    term2 = (1.0 - (sCA * sB) / sA) * (stv_C.strength - sB * sCB) / (1.0 - sB)
    s = term1 + term2

    c = w2c(sCB * cCB * cCA)
    return STV(s, c)


def truth_abduction(stv_A, stv_B, stv_C, stv_AC, stv_BC):
    """A→C, B→C ⊢ A→B.  lib_pln.metta: Truth_Abduction.

    Note: MeTTa names the premise params ($sAB, $sCB) but they correspond
    to A→C and B→C in the inference rule.
    """
    sB = stv_B.strength
    sC = stv_C.strength
    sAC, cAC = stv_AC.strength, stv_AC.confidence
    sBC, cBC = stv_BC.strength, stv_BC.confidence

    if sB <= 1e-10 or abs(1.0 - sB) < 1e-10:
        return STV(0.5, 0.0)

    term1 = (sAC * sBC * sC) / sB
    term2 = sC * (1.0 - sAC) * (1.0 - sBC) / (1.0 - sB)
    s = term1 + term2

    c = w2c(sAC * cAC * cBC)
    return STV(s, c)


def truth_revision(stv1, stv2):
    """Combine two independent evidence sources.  lib_pln.metta: Truth_Revision."""
    f1, c1 = stv1.strength, stv1.confidence
    f2, c2 = stv2.strength, stv2.confidence

    w1 = c2w(c1)
    w2 = c2w(c2)
    w = w1 + w2

    if w <= 0:
        return STV(0.5, 0.0)

    f = (w1 * f1 + w2 * f2) / w
    c = w2c(w)

    f = min(1.0, f)
    c = min(1.0, max(c, c1, c2))
    return STV(f, c)


def truth_negation(stv):
    """¬A.  lib_pln.metta: Truth_Negation."""
    return STV(1.0 - stv.strength, stv.confidence)


def truth_or(a, b):
    """Fuzzy OR.  lib_pln.metta: Truth_or = 1 - (1-a)*(1-b)."""
    return 1.0 - (1.0 - a) * (1.0 - b)


def truth_symmetric_modus_ponens(stv_A, stv_AB):
    """A, A~B ⊢ B (via Similarity).  lib_pln.metta: Truth_SymmetricModusPonens.

    Used for Similarity / IntentionalSimilarity / ExtensionalSimilarity links.

    Args:
        stv_A:  STV of node A
        stv_AB: STV of Similarity A~B
    Returns: STV of conclusion B
    """
    sA, cA = stv_A.strength, stv_A.confidence
    sAB, cAB = stv_AB.strength, stv_AB.confidence

    snotAB = 0.2
    s = sA * sAB + snotAB * (1.0 - sA) * (1.0 + sAB)
    c = cA * cAB * truth_or(sA, sAB)
    return STV(s, c)


def truth_equivalence_to_implication(stv_A, stv_B, stv_AB):
    """A≡B ⊢ A→B.  lib_pln.metta: Truth_equivalenceToImplication.

    Args:
        stv_A:  STV of node A
        stv_B:  STV of node B
        stv_AB: STV of Equivalence A≡B
    Returns: STV of Implication A→B
    """
    sA = stv_A.strength
    sB = stv_B.strength
    sAB, cAB = stv_AB.strength, stv_AB.confidence

    # Hack: if ABs*ABc > 0.99, strength is just ABs (distributional TV workaround)
    if sAB * cAB > 0.99:
        s = sAB
    else:
        s = (1.0 + sB / sA) * sAB / (1.0 + sAB) if sA > 1e-10 else 0.0
    return STV(s, cAB)


def _safe_div(a, b):
    """Safe division matching MeTTa /safe: returns 0 if denominator ≤ 0."""
    if b <= 0:
        return 0.0
    return a / b


def _negate(x):
    """1 - x.  Matches MeTTa negate."""
    return 1.0 - x


def _invert(x):
    """1 / x.  Matches MeTTa invert."""
    if abs(x) < 1e-12:
        return float('inf')
    return 1.0 / x


def transitive_similarity_strength(sA, sB, sC, sAB, sBC):
    """Strength of A~C given A~B, B~C.  lib_pln.metta: TransitiveSimilarityStrength.

    Complex formula using equivalence-to-implication conversions (T1–T4)
    followed by a deduction-like combination in both directions.
    """
    T1 = (1.0 + sB / sA) * sAB / (1.0 + sAB) if sA > 1e-10 else 0.0
    T2 = (1.0 + sC / sB) * sBC / (1.0 + sBC) if sB > 1e-10 else 0.0
    T3 = (1.0 + sB / sC) * sBC / (1.0 + sBC) if sC > 1e-10 else 0.0
    T4 = (1.0 + sA / sB) * sAB / (1.0 + sAB) if sB > 1e-10 else 0.0

    # Forward: deduction A→B→C via T1, T2
    fwd = T1 * T2 + _negate(T1) * _safe_div(sC - sB * T2, _negate(sB))
    # Backward: deduction C→B→A via T3, T4
    bwd = T3 * T4 + _negate(T3) * _safe_div(sC - sB * T4, _negate(sB))

    # Combine: similarity = invert(invert(fwd) + invert(bwd) - 1)
    return _invert(_invert(fwd) + _invert(bwd) - 1.0)


def truth_transitive_similarity(stv_A, stv_B, stv_C, stv_AB, stv_BC):
    """A~B, B~C ⊢ A~C.  lib_pln.metta: Truth_transitiveSimilarity.

    Args:
        stv_A, stv_B, stv_C: node STVs
        stv_AB: STV of Similarity A~B
        stv_BC: STV of Similarity B~C
    Returns: STV of Similarity A~C
    """
    s = transitive_similarity_strength(
        stv_A.strength, stv_B.strength, stv_C.strength,
        stv_AB.strength, stv_BC.strength)

    c = stv_AB.confidence * stv_BC.confidence * truth_or(
        stv_AB.strength, stv_BC.strength)
    return STV(s, c)


def simple_deduction_strength(sA, sB, sC, sAB, sBC):
    """Deduction strength helper.  lib_pln.metta: simpleDeductionStrength.

    Like truth_deduction strength but returns None if consistency fails.
    """
    if not (conditional_probability_consistency(sA, sB, sAB) and
            conditional_probability_consistency(sB, sC, sBC)):
        return None

    if sB > 0.99:
        return sC
    return sAB * sBC + (1.0 - sAB) * (sC - sB * sBC) / (1.0 - sB)


def truth_evaluation_implication(stv_A, stv_B, stv_C, stv_AB, stv_AC):
    """(Eval A B), (Impl A C) ⊢ (Eval C B).  lib_pln.metta: Truth_evaluationImplication.

    NOTE: Parameter order follows upstream — deduction is applied as
    simpleDeductionStrength(Bs, As, Cs, ABs, ACs).

    Args:
        stv_A, stv_B, stv_C: node STVs
        stv_AB: STV of (Evaluation A B) premise
        stv_AC: STV of (Implication A C) premise
    Returns: STV of conclusion
    """
    sAB, cAB = stv_AB.strength, stv_AB.confidence
    sAC, cAC = stv_AC.strength, stv_AC.confidence

    s = simple_deduction_strength(
        stv_B.strength, stv_A.strength, stv_C.strength, sAB, sAC)
    if s is None:
        return STV(1.0, 0.0)

    c = sAB * sAC * cAB * cAC
    return STV(s, c)


def consistency_implication_implicant_conjunction(sA, sB, sC, sAC, sBC):
    """Check conjunction consistency.  lib_pln.metta: Consistency_ImplicationImplicantConjunction.

    Verifies P(C|A) ≤ P(C)/P(A) and P(C|B) ≤ P(C)/P(B).
    """
    if sA <= 0 or sB <= 0 or sC <= 0:
        return False
    return sAC <= sC / sA and sBC <= sC / sB


def compare_stv(name, expected, sampled_strength, tol=0.02):
    """Compare expected STV against sampled strength.

    Confidence is analytical (not sampled), so only strength is checked.
    """
    err = abs(sampled_strength - expected.strength)
    passed = err < tol
    mark = "PASS" if passed else "FAIL"
    print(f"  {name:<36s}  PLN={expected}  "
          f"gibbs_s={sampled_strength:.4f}  err={err:.4f}  [{mark}]")
    return passed
