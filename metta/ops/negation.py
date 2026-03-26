"""
thrml-negation! — ¬A: purely analytical, no sampling needed.

Upstream: lib_pln.metta Truth_Negation
"""

from pln_thrml import STV, truth_negation
from metta.atoms import parse_stv_param, make_stv, make_error


def make_op(metta_ref):
    def thrml_negation(*atoms):
        if len(atoms) < 1:
            return [make_error("expected (thrml-negation! (A T))")]

        children = atoms[0].get_children()
        if len(children) < 2:
            return [make_error("expected (A (stv ..))")]

        s, c = parse_stv_param(children[1])

        result = truth_negation(STV(s, c))
        return [make_stv(result.strength, result.confidence)]

    return thrml_negation
