"""
thrml-inversion! — A→B ⊢ B→A via thermodynamic sampling.

Upstream: lib_pln.metta Truth_inversion
Topology: 2-node chain  A → B

Strategy: Force B=True (0.99, 0.99), weak prior on A.
A's marginal gives P(A|B=True) — exact Bayesian inversion.
"""

from pln_thrml_beta import (
    build_beta_chain, run_beta_sampling, estimate_beta_marginal,
    DEFAULT_EPSILON,
)
from metta.atoms import (
    extract_priors, extract_backgrounds,
    find_prior, parse_stv_param, make_stv, make_error,
)


def make_op(metta_ref):
    def thrml_inversion(*atoms):
        if len(atoms) < 1:
            return [make_error("expected (thrml-inversion! (A B T))")]

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

        graph = build_beta_chain(
            priors=[0.5, 0.99], confidences=[0.01, 0.99],
            strengths=[s_AB], impl_confidences=[c_AB], backgrounds=[bg],
            clamp_root=False)
        samples = run_beta_sampling(graph, seed=42)
        _, strength, confidence = estimate_beta_marginal(
            samples, graph, graph["nodes"][0])

        return [make_stv(strength, confidence)]

    return thrml_inversion
