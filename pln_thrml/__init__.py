"""
pln_thrml — PLN inference rules compiled to thermodynamic factor graphs
=======================================================================

PLN utilities: pln_thrml.pln_utils (c2w, w2c, stv_to_beta_params)
Unified arch:  pln_thrml.unified (LBM(s) on TSU + QLN(n) on CPU)
QLN layer:     pln_thrml.qln_cpu (closed-form confidence propagation)
"""

from .pln_utils import (
    EPS, DEFAULT_EPSILON, c2w, w2c, stv_to_beta_params,
)
from .unified import (
    unified_modus_ponens, unified_deduction, unified_abduction,
    unified_inversion, unified_revision,
)
from .qln_cpu import (
    c_modus_ponens, c_deduction, c_abduction,
    inversion_pln, inversion_bayes, revision,
)

__all__ = [
    # pln_utils
    "EPS", "DEFAULT_EPSILON", "c2w", "w2c", "stv_to_beta_params",
    # unified
    "unified_modus_ponens", "unified_deduction", "unified_abduction",
    "unified_inversion", "unified_revision",
    # qln_cpu
    "c_modus_ponens", "c_deduction", "c_abduction",
    "inversion_pln", "inversion_bayes", "revision",
]
