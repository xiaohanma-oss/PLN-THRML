"""
ops — Registry of all thrml grounded operations for PLN inference rules.
"""

from hyperon import OperationAtom

from metta.ops.modus_ponens import make_op as make_modus_ponens
from metta.ops.deduction import make_op as make_deduction
from metta.ops.inversion import make_op as make_inversion
from metta.ops.induction import make_op as make_induction
from metta.ops.abduction import make_op as make_abduction
from metta.ops.revision import make_op as make_revision
from metta.ops.negation import make_op as make_negation
from metta.ops.symmetric_modus_ponens import make_op as make_symmetric_mp
from metta.ops.equiv_to_impl import make_op as make_equiv_to_impl
from metta.ops.transitive_similarity import make_op as make_transitive_sim
from metta.ops.evaluation_implication import make_op as make_eval_impl
from metta.ops.compile import make_compile_op, make_query_op


ALL_OPS = {
    "thrml-modus-ponens!":    make_modus_ponens,
    "thrml-deduction!":       make_deduction,
    "thrml-inversion!":       make_inversion,
    "thrml-induction!":       make_induction,
    "thrml-abduction!":       make_abduction,
    "thrml-revision!":        make_revision,
    "thrml-negation!":        make_negation,
    "thrml-symmetric-mp!":    make_symmetric_mp,
    "thrml-equiv-to-impl!":   make_equiv_to_impl,
    "thrml-transitive-sim!":  make_transitive_sim,
    "thrml-eval-impl!":       make_eval_impl,
}


def register_all(metta):
    """Register all thrml grounded operations with a MeTTa runner."""
    for name, factory in ALL_OPS.items():
        op = factory(metta)
        metta.register_atom(
            name,
            OperationAtom(name, op, unwrap=False)
        )

    # Full-graph compile/query pair (shared cache)
    compile_op, cache = make_compile_op(metta)
    query_op = make_query_op(metta, cache)
    metta.register_atom(
        "thrml-compile!",
        OperationAtom("thrml-compile!", compile_op, unwrap=False)
    )
    metta.register_atom(
        "thrml-query!",
        OperationAtom("thrml-query!", query_op, unwrap=False)
    )
