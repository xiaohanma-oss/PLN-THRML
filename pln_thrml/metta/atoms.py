"""
atoms.py — MeTTa atom parsing and construction helpers for PLN grounded ops.

Provides: stv parsing, background extraction, argument validation,
and atom constructors (make_stv, make_error).
"""

from hyperon import E, S, ValueAtom
from pln_thrml.beta import MAX_CONFIDENCE


# ── Atom parsing helpers ──────────────────────────────────────────────────

def _float_from_atom(atom):
    """Extract a float from a MeTTa atom (ValueAtom or symbol)."""
    return float(str(atom))


def parse_stv_param(atom):
    """Parse (stv s c) atom into (strength, confidence) tuple.

    Used by |-thrml rules that forward $T1/$T2 truth values from the
    inference chain into grounded operations (matching upstream lib_pln.metta).
    Clamps strength to [0, 1] and confidence to [0, 1).
    """
    children = atom.get_children()
    s = max(0.0, min(1.0, _float_from_atom(children[1])))
    c = max(0.0, min(MAX_CONFIDENCE, _float_from_atom(children[2])))
    return s, c



def extract_backgrounds(metta):
    """Query space for ((Background src dst) rate).

    Returns dict: {(src, dst): rate}
    """
    results = metta.run("!(match &self ((Background $x $y) $r) ((Background $x $y) $r))")
    bgs = {}
    for atom in results[0]:
        children = atom.get_children()
        link_atom = children[0]
        link_children = link_atom.get_children()
        bgs[(str(link_children[1]), str(link_children[2]))] = _float_from_atom(children[1])
    return bgs




def validate_op_args(atoms, min_children, op_name, expected_format):
    """验证 grounded op 参数并提取 children。成功返回 (children, None)，失败返回 (None, error_list)。"""
    if len(atoms) < 1:
        return None, [make_error(f"expected ({op_name} ({expected_format}))")]
    children = atoms[0].get_children()
    if len(children) < min_children:
        return None, [make_error(f"expected ({expected_format})")]
    return children, None


def make_stv(strength, confidence):
    return E(S("stv"), ValueAtom(strength), ValueAtom(confidence))


def make_error(msg):
    return E(S("Error"), S(msg))
