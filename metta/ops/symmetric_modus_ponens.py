"""
thrml-symmetric-mp! — A, A~B ⊢ B via thermodynamic sampling.

Upstream: lib_pln.metta Truth_SymmetricModusPonens
Topology: 2-node chain with computed background rate
"""

from pln_thrml_beta import (
    build_beta_chain, run_beta_sampling, estimate_beta_marginal,
)
from metta.atoms import parse_stv_param, make_stv, make_error


def make_op(metta_ref):
    def thrml_symmetric_mp(*atoms):
        if len(atoms) < 1:
            return [make_error("expected (thrml-symmetric-mp! (A B T1 T2))")]

        children = atoms[0].get_children()
        if len(children) < 4:
            return [make_error("expected (A B (stv ..) (stv ..))")]

        name_A, name_B = str(children[0]), str(children[1])
        s_A, c_A = parse_stv_param(children[2])
        s_AB, c_AB = parse_stv_param(children[3])

        snotAB = 0.2
        bg = snotAB * (1.0 + s_AB)

        graph = build_beta_chain(
            priors=[s_A, 0.5], confidences=[c_A, 0.01],
            strengths=[s_AB], impl_confidences=[c_AB], backgrounds=[bg])
        samples = run_beta_sampling(graph, seed=42)
        _, strength, confidence = estimate_beta_marginal(
            samples, graph, graph["nodes"][1])

        return [make_stv(strength, confidence)]

    return thrml_symmetric_mp
