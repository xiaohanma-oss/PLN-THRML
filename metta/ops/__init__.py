"""ops — Registry of all thrml grounded operations for PLN inference rules."""

from hyperon import OperationAtom

from metta.ops.rules import RULE_SPECS, make_rule_op, make_revision_op


def register_all(metta):
    """Register all thrml grounded operations with a MeTTa runner."""
    # Sampling-based rules from declarative table
    for name in RULE_SPECS:
        op = make_rule_op(name, metta)
        metta.register_atom(name, OperationAtom(name, op, unwrap=False))

    # Revision: dual calling convention, not in RULE_SPECS
    rev_op = make_revision_op(metta)
    metta.register_atom("thrml-revision!", OperationAtom("thrml-revision!", rev_op, unwrap=False))

