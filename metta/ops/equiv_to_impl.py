"""
thrml-equiv-to-impl! — A≡B ⊢ A→B via thermodynamic sampling.

Upstream: lib_pln.metta Truth_equivalenceToImplication
Topology: symmetric pair  A ↔ B

Strategy: Force A=True (0.99, 0.99), weak prior on B.
B's marginal gives P(B|A=True).
"""

from pln_thrml_beta import (
    build_beta_symmetric_pair, run_beta_sampling, estimate_beta_marginal,
    DEFAULT_EPSILON,
)
from metta.atoms import (
    extract_priors, extract_backgrounds,
    find_prior, parse_stv_param, make_stv, make_error,
)


def make_op(metta_ref):
    def thrml_equiv_to_impl(*atoms):
        if len(atoms) < 1:
            return [make_error("expected (thrml-equiv-to-impl! (A B T))")]

        children = atoms[0].get_children()
        if len(children) < 3:
            return [make_error("expected (A B (stv ..))")]

        name_A, name_B = str(children[0]), str(children[1])
        s_AB, c_AB = parse_stv_param(children[2])

        priors = extract_priors(metta_ref)
        bgs = extract_backgrounds(metta_ref)

        s_A, c_A = find_prior(priors, name_A)
        s_B, c_B = find_prior(priors, name_B)
        bg = bgs.get((name_A, name_B), DEFAULT_EPSILON)

        graph = build_beta_symmetric_pair(
            prior_a=0.99, confidence_a=0.99,
            prior_b=0.5, confidence_b=0.01,
            strength=s_AB, impl_confidence=c_AB, background=bg)
        samples = run_beta_sampling(graph, seed=42)
        _, strength, confidence = estimate_beta_marginal(
            samples, graph, graph["b"])

        return [make_stv(strength, confidence)]

    return thrml_equiv_to_impl
