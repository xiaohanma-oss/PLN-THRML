"""
thrml-eval-impl! — (Eval A B), (Impl A C) ⊢ (Eval C B) via thermodynamic sampling.

Upstream: lib_pln.metta Truth_evaluationImplication
Topology: 3-node chain B → A → C

Strategy: Force B=True (0.99, 0.99), weak priors on A and C.
C's marginal gives P(C|B=True).
"""

from pln_thrml_beta import (
    build_beta_chain, sample_and_measure, DEFAULT_EPSILON,
)
from metta.atoms import (
    extract_backgrounds, parse_stv_param, make_stv, validate_op_args,
)


def make_op(metta_ref):
    def thrml_eval_impl(*atoms):
        children, err = validate_op_args(atoms, 5, "thrml-eval-impl!", "A B C (stv ..) (stv ..)")
        if err:
            return err

        name_A, name_B, name_C = str(children[0]), str(children[1]), str(children[2])
        s_AB, c_AB = parse_stv_param(children[3])
        s_AC, c_AC = parse_stv_param(children[4])

        bgs = extract_backgrounds(metta_ref)

        bg_BA = bgs.get((name_B, name_A), DEFAULT_EPSILON)
        bg_AC = bgs.get((name_A, name_C), DEFAULT_EPSILON)

        graph = build_beta_chain(
            priors=[0.99, 0.5, 0.5], confidences=[0.99, 0.01, 0.01],
            strengths=[s_AB, s_AC], impl_confidences=[c_AB, c_AC],
            backgrounds=[bg_BA, bg_AC], clamp_root=False)
        strength, confidence = sample_and_measure(graph, graph["nodes"][2])
        return [make_stv(strength, confidence)]

    return thrml_eval_impl
