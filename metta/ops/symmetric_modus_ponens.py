"""
thrml-symmetric-mp! — A, A~B ⊢ B via thermodynamic sampling.

Upstream: lib_pln.metta Truth_SymmetricModusPonens
Topology: 2-node chain with computed background rate
"""

from pln_thrml_beta import build_beta_chain, sample_and_measure
from metta.atoms import parse_stv_param, make_stv, validate_op_args


def make_op(metta_ref):
    def thrml_symmetric_mp(*atoms):
        children, err = validate_op_args(atoms, 4, "thrml-symmetric-mp!", "A B (stv ..) (stv ..)")
        if err:
            return err

        name_A, name_B = str(children[0]), str(children[1])
        s_A, c_A = parse_stv_param(children[2])
        s_AB, c_AB = parse_stv_param(children[3])

        snotAB = 0.2
        bg = snotAB * (1.0 + s_AB)

        graph = build_beta_chain(
            priors=[s_A, 0.5], confidences=[c_A, 0.01],
            strengths=[s_AB], impl_confidences=[c_AB], backgrounds=[bg])
        strength, confidence = sample_and_measure(graph, graph["nodes"][1])
        return [make_stv(strength, confidence)]

    return thrml_symmetric_mp
