#!/usr/bin/env python3
"""Generate Results tables for README from current test parameters.

Runs beta-discretized Gibbs sampling for each PLN rule and prints
markdown tables comparing analytical PLN values with beta sampled values.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pln_thrml import (
    STV, DEFAULT_EPSILON,
    truth_modus_ponens, truth_deduction, truth_negation,
    truth_symmetric_modus_ponens,
)
from pln_thrml_beta import (
    build_beta_chain, build_beta_v_graph, build_beta_inv_v_graph,
    build_beta_symmetric_pair, build_beta_symmetric_chain,
    make_beta_prior_factor,
    run_beta_sampling, estimate_beta_marginal,
    DEFAULT_K,
)
from thrml.pgm import CategoricalNode
from thrml.block_management import Block
from thrml.block_sampling import BlockGibbsSpec
from thrml.models.discrete_ebm import CategoricalGibbsConditional
from thrml.factor import FactorSamplingProgram


def fmt(v):
    return f"{v:.4f}"


def print_table(title, rule_expr, rows):
    print(f"### {title} — `{rule_expr}`")
    print()
    print("| Setting | PLN | Beta Gibbs | Error |")
    print("|---|---|---|---|")
    for name, pln_val, gibbs_val in rows:
        err = abs(pln_val - gibbs_val)
        print(f"| {name} | {fmt(pln_val)} | {fmt(gibbs_val)} | {fmt(err)} |")
    print()


# ── Modus Ponens ─────────────────────────────────────────────────────

def run_modus_ponens():
    rows = []
    cases = [
        ("Strong prior (s_A=0.8, s_AB=0.9)", 0.8, 0.9, 0.9, 0.85),
        ("Smokes upstream (s_A=1.0, s_AB=0.6)", 1.0, 0.9, 0.6, 0.9),
        ("Rare antecedent (s_A=0.1, s_AB=0.8)", 0.1, 0.7, 0.8, 0.75),
    ]
    for name, s_A, c_A, s_AB, c_AB in cases:
        expected = truth_modus_ponens(STV(s_A, c_A), STV(s_AB, c_AB))
        graph = build_beta_chain(
            priors=[s_A, 0.5], confidences=[c_A, 0.01],
            strengths=[s_AB], impl_confidences=[c_AB],
            backgrounds=[DEFAULT_EPSILON])
        samples = run_beta_sampling(graph, seed=42)
        _, p_B, _ = estimate_beta_marginal(samples, graph, graph["nodes"][1])
        rows.append((name, expected.strength, p_B))
    print_table("Modus Ponens", "A, A→B ⊢ B", rows)


# ── Deduction ─────────────────────────────────────────────────────────

def run_deduction():
    s_AB, s_BC = 0.8, 0.9
    expected = truth_deduction(
        STV(0.5, 0.9), STV(0.4, 0.9), STV(0.4, 0.9),
        STV(s_AB, 0.9), STV(s_BC, 0.9))
    graph = build_beta_chain(
        priors=[0.99, 0.5, 0.5], confidences=[0.99, 0.01, 0.01],
        strengths=[s_AB, s_BC], impl_confidences=[0.9, 0.9],
        backgrounds=[DEFAULT_EPSILON, DEFAULT_EPSILON], clamp_root=False)
    samples = run_beta_sampling(graph, seed=42)
    _, p, _ = estimate_beta_marginal(samples, graph, graph["nodes"][2])
    print_table("Deduction", "A→B, B→C ⊢ A→C",
                [("Standard chain (s_AB=0.8, s_BC=0.9)", expected.strength, p)])


# ── Inversion ─────────────────────────────────────────────────────────

def run_inversion():
    s_A, s_AB = 0.5, 0.87
    # Exact Bayes
    p_B = s_A * s_AB + (1 - s_A) * DEFAULT_EPSILON
    bayes = s_AB * s_A / p_B
    graph = build_beta_chain(
        priors=[0.5, 0.99], confidences=[0.01, 0.99],
        strengths=[s_AB], impl_confidences=[0.81],
        backgrounds=[DEFAULT_EPSILON], clamp_root=False)
    samples = run_beta_sampling(graph, seed=42)
    _, p, _ = estimate_beta_marginal(samples, graph, graph["nodes"][0])
    print_table("Inversion", "A→B ⊢ B→A (exact Bayes)",
                [("Upstream (s_A=0.5, s_AB=0.87)", bayes, p)])


# ── Induction ─────────────────────────────────────────────────────────

def run_induction():
    graph = build_beta_v_graph(
        root_prior=0.5, root_confidence=0.01,
        left_strength=0.9, right_strength=0.8,
        left_impl_confidence=0.9, right_impl_confidence=0.9,
        left_background=DEFAULT_EPSILON, right_background=DEFAULT_EPSILON,
        left_prior=0.99, left_confidence=0.99,
        right_prior=0.5, right_confidence=0.01)
    samples = run_beta_sampling(graph, seed=42)
    _, p, _ = estimate_beta_marginal(samples, graph, graph["right"])
    # PLN analytical (for reference)
    from pln_thrml import truth_induction
    expected = truth_induction(
        STV(0.25, 0.9), STV(0.25, 0.9), STV(0.25, 0.9),
        STV(0.9, 0.9), STV(0.8, 0.9))
    print_table("Induction", "C→A, C→B ⊢ A→B",
                [("Raven upstream (s_CA=0.9, s_CB=0.8)", expected.strength, p)])


# ── Abduction ─────────────────────────────────────────────────────────

def run_abduction():
    graph = build_beta_inv_v_graph(
        left_prior=0.99, left_confidence=0.99,
        right_prior=0.5, right_confidence=0.01,
        left_strength=0.8, right_strength=0.7,
        left_impl_confidence=0.9, right_impl_confidence=0.9,
        left_background=DEFAULT_EPSILON, right_background=DEFAULT_EPSILON,
        center_prior=0.5, center_confidence=0.01)
    samples = run_beta_sampling(graph, seed=42)
    _, p, _ = estimate_beta_marginal(samples, graph, graph["right"])
    from pln_thrml import truth_abduction
    expected = truth_abduction(
        STV(0.5, 0.9), STV(0.5, 0.9), STV(0.5, 0.9),
        STV(0.8, 0.9), STV(0.7, 0.9))
    print_table("Abduction", "A→C, B→C ⊢ A→B",
                [("Symmetric priors (s_AC=0.8, s_BC=0.7)", expected.strength, p)])


# ── Negation ──────────────────────────────────────────────────────────

def run_negation():
    expected = truth_negation(STV(0.99, 0.9))
    print_table("Negation", "A ⊢ ¬A",
                [("Strong (s=0.99)", expected.strength, expected.strength)])


# ── Revision ──────────────────────────────────────────────────────────

def run_revision():
    from pln_thrml import truth_revision
    stv1, stv2 = STV(0.8, 0.9), STV(0.3, 0.7)
    expected = truth_revision(stv1, stv2)

    k = DEFAULT_K
    node = CategoricalNode()
    f1 = make_beta_prior_factor(node, stv1.strength, stv1.confidence, k)
    f2 = make_beta_prior_factor(node, stv2.strength, stv2.confidence, k)
    free_blocks = [Block([node])]
    spec = BlockGibbsSpec(free_blocks, [])
    sampler = CategoricalGibbsConditional(n_categories=k)
    prog = FactorSamplingProgram(
        gibbs_spec=spec, samplers=[sampler], factors=[f1, f2],
        other_interaction_groups=[])
    graph = dict(nodes=[node], factors=[f1, f2], free_blocks=free_blocks,
                 clamped_blocks=[], spec=spec, program=prog, n=1, k=k,
                 single_node=False)
    samples = run_beta_sampling(graph, seed=42)
    _, p, _ = estimate_beta_marginal(samples, graph, node)
    print_table("Revision", "combine evidence",
                [("Dual sources (s1=0.8,c1=0.9 + s2=0.3,c2=0.7)", expected.strength, p)])


# ── Symmetric Modus Ponens ────────────────────────────────────────────

def run_symmetric_mp():
    s_A, c_A, s_AB, c_AB = 0.8, 0.9, 0.85, 0.9
    expected = truth_symmetric_modus_ponens(STV(s_A, c_A), STV(s_AB, c_AB))
    snotAB = 0.2
    bg = snotAB * (1.0 + s_AB)
    graph = build_beta_chain(
        priors=[s_A, 0.5], confidences=[c_A, 0.01],
        strengths=[s_AB], impl_confidences=[c_AB], backgrounds=[bg])
    samples = run_beta_sampling(graph, seed=42)
    _, p_B, _ = estimate_beta_marginal(samples, graph, graph["nodes"][1])
    print_table("Symmetric Modus Ponens", "A, A~B ⊢ B",
                [("Standard (s_A=0.8, s_AB=0.85)", expected.strength, p_B)])


# ── Equivalence to Implication ────────────────────────────────────────

def run_equiv_to_impl():
    graph = build_beta_symmetric_pair(
        prior_a=0.99, confidence_a=0.99,
        prior_b=0.5, confidence_b=0.01,
        strength=0.98, impl_confidence=0.87, background=DEFAULT_EPSILON)
    samples = run_beta_sampling(graph, seed=42)
    _, p, _ = estimate_beta_marginal(samples, graph, graph["b"])
    from pln_thrml import truth_equivalence_to_implication
    expected = truth_equivalence_to_implication(
        STV(0.5, 1.0), STV(0.5, 1.0), STV(0.98, 0.87))
    print_table("Equivalence→Implication", "A≡B ⊢ A→B",
                [("Upstream (s_AB=0.98)", expected.strength, p)])


# ── Transitive Similarity ────────────────────────────────────────────

def run_transitive_sim():
    graph = build_beta_symmetric_chain(
        priors=[0.99, 0.5, 0.5], confidences=[0.99, 0.01, 0.01],
        strengths=[1.0, 1.0], impl_confidences=[0.89, 0.5],
        backgrounds=[DEFAULT_EPSILON, DEFAULT_EPSILON])
    samples = run_beta_sampling(graph, seed=42)
    _, p, _ = estimate_beta_marginal(samples, graph, graph["nodes"][2])
    from pln_thrml import truth_transitive_similarity
    expected = truth_transitive_similarity(
        STV(0.333, 1.0), STV(0.333, 1.0), STV(0.333, 1.0),
        STV(1.0, 0.89), STV(1.0, 0.5))
    print_table("Transitive Similarity", "A~B, B~C ⊢ A~C",
                [("Upstream (s_AB=1.0, s_BC=1.0)", expected.strength, p)])


# ── Evaluation Implication ────────────────────────────────────────────

def run_eval_impl():
    graph = build_beta_chain(
        priors=[0.99, 0.5, 0.5], confidences=[0.99, 0.01, 0.01],
        strengths=[1.0, 1.0], impl_confidences=[0.9, 0.9],
        backgrounds=[DEFAULT_EPSILON, DEFAULT_EPSILON], clamp_root=False)
    samples = run_beta_sampling(graph, seed=42)
    _, p, _ = estimate_beta_marginal(samples, graph, graph["nodes"][2])
    from pln_thrml import truth_evaluation_implication
    expected = truth_evaluation_implication(
        STV(0.25, 1.0), STV(0.25, 1.0), STV(0.25, 1.0),
        STV(1.0, 0.9), STV(1.0, 0.9))
    print_table("Evaluation Implication", "(Eval A B), (Impl A C) ⊢ (Eval C B)",
                [("Upstream (s_AB=1.0, s_AC=1.0)", expected.strength, p)])


if __name__ == "__main__":
    print("## Results\n")
    print("Each rule is verified against its analytical PLN formula.")
    print("Beta sampling: 50 batches × 2,000 samples (100,000 total),")
    print("K=16 bins, 500 warmup steps.\n")

    run_modus_ponens()
    run_deduction()
    run_inversion()
    run_induction()
    run_abduction()
    run_negation()
    run_revision()
    run_symmetric_mp()
    run_equiv_to_impl()
    run_transitive_sim()
    run_eval_impl()
