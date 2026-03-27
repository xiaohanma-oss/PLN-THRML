"""
thrml-modus-ponens! — A, A→B ⊢ B via thermodynamic sampling.

Upstream: lib_pln.metta Truth_ModusPonens
Topology: 2-node chain  A → B
"""

from pln_thrml_beta import (
    build_beta_chain, sample_and_measure, DEFAULT_EPSILON,
)
from metta.atoms import (
    extract_backgrounds, parse_stv_param,
    make_stv, validate_op_args,
)


def make_op(metta_ref):
    def thrml_modus_ponens(*atoms):
        children, err = validate_op_args(atoms, 4, "thrml-modus-ponens!", "A B (stv ..) (stv ..)")
        if err:
            return err

        src_name, dst_name = str(children[0]), str(children[1])
        s_A, c_A = parse_stv_param(children[2])
        s_AB, c_AB = parse_stv_param(children[3])

        bgs = extract_backgrounds(metta_ref)
        eps = bgs.get((src_name, dst_name), DEFAULT_EPSILON)

        graph = build_beta_chain(
            priors=[s_A, 0.5], confidences=[c_A, 0.01],
            strengths=[s_AB], impl_confidences=[c_AB], backgrounds=[eps])
        strength, confidence = sample_and_measure(graph, graph["nodes"][1])
        return [make_stv(strength, confidence)]

    return thrml_modus_ponens
