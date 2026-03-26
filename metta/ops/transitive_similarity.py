"""
thrml-transitive-sim! — A~B, B~C ⊢ A~C via thermodynamic sampling.

Upstream: lib_pln.metta Truth_transitiveSimilarity
Topology: 3-node symmetric chain  A ↔ B ↔ C

Strategy: Force A=True (0.99, 0.99), weak priors on B and C.
C's marginal gives P(C|A=True).
"""

from pln_thrml_beta import (
    build_beta_symmetric_chain, run_beta_sampling, estimate_beta_marginal,
    DEFAULT_EPSILON,
)
from metta.atoms import (
    extract_priors, extract_backgrounds,
    find_prior, parse_stv_param, make_stv, make_error,
)


def make_op(metta_ref):
    def thrml_transitive_sim(*atoms):
        if len(atoms) < 1:
            return [make_error("expected (thrml-transitive-sim! (A B C T1 T2))")]

        children = atoms[0].get_children()
        if len(children) < 5:
            return [make_error("expected (A B C (stv ..) (stv ..))")]

        name_A, name_B, name_C = str(children[0]), str(children[1]), str(children[2])
        s_AB, c_AB = parse_stv_param(children[3])
        s_BC, c_BC = parse_stv_param(children[4])

        priors = extract_priors(metta_ref)
        bgs = extract_backgrounds(metta_ref)

        s_A, c_A = find_prior(priors, name_A)
        s_B, c_B = find_prior(priors, name_B)
        s_C, c_C = find_prior(priors, name_C)

        bg_AB = bgs.get((name_A, name_B), DEFAULT_EPSILON)
        bg_BC = bgs.get((name_B, name_C), DEFAULT_EPSILON)

        graph = build_beta_symmetric_chain(
            priors=[0.99, 0.5, 0.5], confidences=[0.99, 0.01, 0.01],
            strengths=[s_AB, s_BC], impl_confidences=[c_AB, c_BC],
            backgrounds=[bg_AB, bg_BC])
        samples = run_beta_sampling(graph, seed=42)
        _, strength, confidence = estimate_beta_marginal(
            samples, graph, graph["nodes"][2])

        return [make_stv(strength, confidence)]

    return thrml_transitive_sim
