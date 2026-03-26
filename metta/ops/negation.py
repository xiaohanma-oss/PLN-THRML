"""
thrml-negation! — ¬A: purely analytical, no sampling needed.

Upstream: lib_pln.metta Truth_Negation
"""

from metta.atoms import parse_stv_param, make_stv, make_error


def make_op(metta_ref):
    def thrml_negation(*atoms):
        if len(atoms) < 1:
            return [make_error("expected (thrml-negation! (A T))")]

        children = atoms[0].get_children()
        if len(children) < 2:
            return [make_error("expected (A (stv ..))")]

        s, c = parse_stv_param(children[1])
        return [make_stv(1.0 - s, c)]

    return thrml_negation
