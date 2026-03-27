"""
thrml-induction! — C→A, C→B ⊢ A→B via thermodynamic sampling.

Upstream: lib_pln.metta Truth_Induction
Topology: V-shape  A ← C → B

Strategy: Force A=True (0.99, 0.99), weak priors on C and B.
B's marginal gives P(B|A=True).
"""

from pln_thrml_beta import (
    build_beta_v_graph, sample_and_measure, DEFAULT_EPSILON,
)
from metta.atoms import (
    extract_backgrounds, parse_stv_param, make_stv, validate_op_args,
)


def make_op(metta_ref):
    def thrml_induction(*atoms):
        children, err = validate_op_args(atoms, 5, "thrml-induction!", "A B C (stv ..) (stv ..)")
        if err:
            return err

        name_A, name_B, name_C = str(children[0]), str(children[1]), str(children[2])
        s_CA, c_CA = parse_stv_param(children[3])
        s_CB, c_CB = parse_stv_param(children[4])

        bgs = extract_backgrounds(metta_ref)

        bg_CA = bgs.get((name_C, name_A), DEFAULT_EPSILON)
        bg_CB = bgs.get((name_C, name_B), DEFAULT_EPSILON)

        graph = build_beta_v_graph(
            root_prior=0.5, root_confidence=0.01,
            left_strength=s_CA, right_strength=s_CB,
            left_impl_confidence=c_CA, right_impl_confidence=c_CB,
            left_background=bg_CA, right_background=bg_CB,
            left_prior=0.99, left_confidence=0.99,
            right_prior=0.5, right_confidence=0.01)
        strength, confidence = sample_and_measure(graph, graph["right"])
        return [make_stv(strength, confidence)]

    return thrml_induction
