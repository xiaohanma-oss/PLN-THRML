"""
thrml-equiv-to-impl! — A≡B ⊢ A→B via thermodynamic sampling.

Upstream: lib_pln.metta Truth_equivalenceToImplication
Topology: symmetric pair  A ↔ B

Strategy: Force A=True (0.99, 0.99), weak prior on B.
B's marginal gives P(B|A=True).
"""

from pln_thrml_beta import (
    build_beta_symmetric_pair, sample_and_measure, DEFAULT_EPSILON,
)
from metta.atoms import (
    extract_backgrounds, parse_stv_param, make_stv, validate_op_args,
)


def make_op(metta_ref):
    def thrml_equiv_to_impl(*atoms):
        children, err = validate_op_args(atoms, 3, "thrml-equiv-to-impl!", "A B (stv ..)")
        if err:
            return err

        name_A, name_B = str(children[0]), str(children[1])
        s_AB, c_AB = parse_stv_param(children[2])

        bgs = extract_backgrounds(metta_ref)
        bg = bgs.get((name_A, name_B), DEFAULT_EPSILON)

        graph = build_beta_symmetric_pair(
            prior_a=0.99, confidence_a=0.99,
            prior_b=0.5, confidence_b=0.01,
            strength=s_AB, impl_confidence=c_AB, background=bg)
        strength, confidence = sample_and_measure(graph, graph["b"])
        return [make_stv(strength, confidence)]

    return thrml_equiv_to_impl
