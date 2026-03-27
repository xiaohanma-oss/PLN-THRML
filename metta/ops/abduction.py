"""
thrml-abduction! — A→C, B→C ⊢ A→B via thermodynamic sampling.

Upstream: lib_pln.metta Truth_Abduction
Topology: Inverted-V  A → C ← B

Strategy: Force A=True (0.99, 0.99), weak priors on B and C.
B's marginal gives P(B|A=True).
"""

from pln_thrml_beta import (
    build_beta_inv_v_graph, sample_and_measure, DEFAULT_EPSILON,
)
from metta.atoms import (
    extract_backgrounds, parse_stv_param, make_stv, validate_op_args,
)


def make_op(metta_ref):
    def thrml_abduction(*atoms):
        children, err = validate_op_args(atoms, 5, "thrml-abduction!", "A B C (stv ..) (stv ..)")
        if err:
            return err

        name_A, name_B, name_C = str(children[0]), str(children[1]), str(children[2])
        s_AC, c_AC = parse_stv_param(children[3])
        s_BC, c_BC = parse_stv_param(children[4])

        bgs = extract_backgrounds(metta_ref)

        bg_AC = bgs.get((name_A, name_C), DEFAULT_EPSILON)
        bg_BC = bgs.get((name_B, name_C), DEFAULT_EPSILON)

        graph = build_beta_inv_v_graph(
            left_prior=0.99, left_confidence=0.99,
            right_prior=0.5, right_confidence=0.01,
            left_strength=s_AC, right_strength=s_BC,
            left_impl_confidence=c_AC, right_impl_confidence=c_BC,
            left_background=bg_AC, right_background=bg_BC,
            center_prior=0.5, center_confidence=0.01)
        strength, confidence = sample_and_measure(graph, graph["right"])
        return [make_stv(strength, confidence)]

    return thrml_abduction
