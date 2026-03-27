"""
thrml-inversion! — A→B ⊢ B→A via thermodynamic sampling.

Upstream: lib_pln.metta Truth_inversion
Topology: 2-node chain  A → B

Strategy: Force B=True (0.99, 0.99), weak prior on A.
A's marginal gives P(A|B=True) — exact Bayesian inversion.
"""

from pln_thrml_beta import (
    build_beta_chain, sample_and_measure, DEFAULT_EPSILON,
)
from metta.atoms import (
    extract_backgrounds, parse_stv_param, make_stv, validate_op_args,
)


def make_op(metta_ref):
    def thrml_inversion(*atoms):
        children, err = validate_op_args(atoms, 3, "thrml-inversion!", "A B (stv ..)")
        if err:
            return err

        name_A, name_B = str(children[0]), str(children[1])
        s_AB, c_AB = parse_stv_param(children[2])

        bgs = extract_backgrounds(metta_ref)
        bg = bgs.get((name_A, name_B), DEFAULT_EPSILON)

        graph = build_beta_chain(
            priors=[0.5, 0.99], confidences=[0.01, 0.99],
            strengths=[s_AB], impl_confidences=[c_AB], backgrounds=[bg],
            clamp_root=False)
        strength, confidence = sample_and_measure(graph, graph["nodes"][0])
        return [make_stv(strength, confidence)]

    return thrml_inversion
