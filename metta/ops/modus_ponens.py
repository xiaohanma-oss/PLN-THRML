"""
thrml-modus-ponens! — A, A→B ⊢ B via thermodynamic sampling.

Upstream: lib_pln.metta Truth_ModusPonens
Topology: 2-node chain  A → B
"""

from pln_thrml_beta import (
    build_beta_chain, run_beta_sampling, estimate_beta_marginal,
    DEFAULT_EPSILON,
)
from metta.atoms import (
    extract_backgrounds, parse_stv_param,
    make_stv, make_error,
)


def make_op(metta_ref):
    def thrml_modus_ponens(*atoms):
        if len(atoms) < 1:
            return [make_error("expected (thrml-modus-ponens! (A B T1 T2))")]

        children = atoms[0].get_children()
        if len(children) < 4:
            return [make_error("expected (name_A name_B (stv ..) (stv ..))")]

        src_name, dst_name = str(children[0]), str(children[1])
        s_A, c_A = parse_stv_param(children[2])
        s_AB, c_AB = parse_stv_param(children[3])

        bgs = extract_backgrounds(metta_ref)
        eps = bgs.get((src_name, dst_name), DEFAULT_EPSILON)

        graph = build_beta_chain(
            priors=[s_A, 0.5], confidences=[c_A, 0.01],
            strengths=[s_AB], impl_confidences=[c_AB], backgrounds=[eps])
        samples = run_beta_sampling(graph, seed=42)
        _, strength, confidence = estimate_beta_marginal(
            samples, graph, graph["nodes"][1])

        return [make_stv(strength, confidence)]

    return thrml_modus_ponens
