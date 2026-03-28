"""ops — Registry of all thrml grounded operations for PLN inference rules."""

from hyperon import OperationAtom

from metta.ops.rules import RULE_SPECS, make_rule_op
from metta.ops.revision import make_op as make_revision
from metta.ops.negation import make_op as make_negation
from metta.ops.compile import make_compile_op, make_query_op


def register_all(metta):
    """Register all thrml grounded operations with a MeTTa runner."""
    # Sampling-based rules from declarative table
    for name in RULE_SPECS:
        op = make_rule_op(name, metta)
        metta.register_atom(name, OperationAtom(name, op, unwrap=False))

    # Special ops: revision (dual calling convention), negation (analytical)
    for name, factory in [("thrml-revision!", make_revision),
                          ("thrml-negation!", make_negation)]:
        op = factory(metta)
        metta.register_atom(name, OperationAtom(name, op, unwrap=False))

    # Full-graph compile/query pair (shared cache)
    compile_op, cache = make_compile_op(metta)
    query_op = make_query_op(metta, cache)
    metta.register_atom(
        "thrml-compile!", OperationAtom("thrml-compile!", compile_op, unwrap=False))
    metta.register_atom(
        "thrml-query!", OperationAtom("thrml-query!", query_op, unwrap=False))
