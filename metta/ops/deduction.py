"""
thrml-deduction! — A→B, B→C ⊢ A→C via thermodynamic sampling.

Upstream: lib_pln.metta Truth_Deduction
Topology: 3-node chain  A → B → C

Strategy: Force A=True (0.99, 0.99), weak priors on B and C.
C's marginal gives P(C|A=True).
"""

from pln_thrml_beta import (
    build_beta_chain, sample_and_measure, DEFAULT_EPSILON,
)
from metta.atoms import (
    extract_backgrounds, parse_stv_param, make_stv, validate_op_args,
)


def make_op(metta_ref):
    def thrml_deduction(*atoms):
        children, err = validate_op_args(atoms, 5, "thrml-deduction!", "A B C (stv ..) (stv ..)")
        if err:
            return err

        name_A, name_B, name_C = str(children[0]), str(children[1]), str(children[2])
        s_AB, c_AB = parse_stv_param(children[3])
        s_BC, c_BC = parse_stv_param(children[4])

        bgs = extract_backgrounds(metta_ref)

        bg_AB = bgs.get((name_A, name_B), DEFAULT_EPSILON)
        bg_BC = bgs.get((name_B, name_C), DEFAULT_EPSILON)

        graph = build_beta_chain(
            priors=[0.99, 0.5, 0.5], confidences=[0.99, 0.01, 0.01],
            strengths=[s_AB, s_BC], impl_confidences=[c_AB, c_BC],
            backgrounds=[bg_AB, bg_BC], clamp_root=False)
        strength, confidence = sample_and_measure(graph, graph["nodes"][2])
        return [make_stv(strength, confidence)]

    return thrml_deduction
