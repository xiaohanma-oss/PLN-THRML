"""
pln_thrml — PLN inference rules compiled to thermodynamic factor graphs
=======================================================================

Core engine: pln_thrml.beta (Beta-discretized factor graphs)
Hardware deployment: pln_thrml.block_diagonal (block-diagonal partitioning)
MeTTa bridge: pln_thrml.metta (optional, requires hyperon)
"""

from pln_thrml.beta import *  # noqa: F401,F403
from pln_thrml.beta import __all__ as _beta_all

from pln_thrml.block_diagonal import *  # noqa: F401,F403
from pln_thrml.block_diagonal import __all__ as _bd_all

__all__ = list(_beta_all) + list(_bd_all)
