"""
thrml-induction! — C→A, C→B ⊢ A→B via thermodynamic sampling.

Upstream: lib_pln.metta Truth_Induction
Topology: V-shape  A ← C → B

Strategy: Force A=True (0.99, 0.99), weak priors on C and B.
B's marginal gives P(B|A=True).
"""

from pln_thrml_beta import (
    build_beta_v_graph, run_beta_sampling, estimate_beta_marginal,
    DEFAULT_EPSILON,
)
from metta.atoms import (
    extract_priors, extract_backgrounds,
    find_prior, parse_stv_param, make_stv, make_error,
)


def make_op(metta_ref):
    def thrml_induction(*atoms):
        if len(atoms) < 1:
            return [make_error("expected (thrml-induction! (A B C T1 T2))")]

        children = atoms[0].get_children()
        if len(children) < 5:
            return [make_error("expected (A B C (stv ..) (stv ..))")]

        name_A, name_B, name_C = str(children[0]), str(children[1]), str(children[2])
        s_CA, c_CA = parse_stv_param(children[3])
        s_CB, c_CB = parse_stv_param(children[4])

        priors = extract_priors(metta_ref)
        bgs = extract_backgrounds(metta_ref)

        s_A, c_A = find_prior(priors, name_A)
        s_B, c_B = find_prior(priors, name_B)
        s_C, c_C = find_prior(priors, name_C)

        bg_CA = bgs.get((name_C, name_A), DEFAULT_EPSILON)
        bg_CB = bgs.get((name_C, name_B), DEFAULT_EPSILON)

        graph = build_beta_v_graph(
            root_prior=0.5, root_confidence=0.01,
            left_strength=s_CA, right_strength=s_CB,
            left_impl_confidence=c_CA, right_impl_confidence=c_CB,
            left_background=bg_CA, right_background=bg_CB,
            left_prior=0.99, left_confidence=0.99,
            right_prior=0.5, right_confidence=0.01)
        samples = run_beta_sampling(graph, seed=42)
        _, strength, confidence = estimate_beta_marginal(
            samples, graph, graph["right"])

        return [make_stv(strength, confidence)]

    return thrml_induction
