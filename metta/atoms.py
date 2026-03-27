"""
atoms.py — Extract PLN knowledge from MeTTa space (upstream Atomese style)
===========================================================================

Knowledge is stored in the MeTTa space using upstream lib_pln.metta conventions:

    Node priors:      (A (stv 0.8 0.9))
    Implications:     ((Implication A B) (stv 0.9 0.85))
    Inheritance:      ((Inheritance A B) (stv 0.9 0.85))
    Similarity:       ((Similarity A B) (stv 0.85 0.9))
    Equivalence:      ((Equivalence A B) (stv 0.8 0.9))
    Evaluations:      ((Evaluation P A) (stv 0.8 0.9))
    Backgrounds:      ((Background A B) 0.05)
"""

from hyperon import E, S, ValueAtom
from pln_thrml_beta import MAX_CONFIDENCE


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


# ── Space queries ─────────────────────────────────────────────────────────

def extract_priors(metta):
    """Query space for node priors: (name (stv s c)).

    Returns dict: {name: {"strength": s, "confidence": c}}
    """
    results = metta.run("!(match &self ($x (stv $s $c)) ($x (stv $s $c)))")
    priors = {}
    for atom in results[0]:
        children = atom.get_children()
        name_atom = children[0]
        try:
            name_atom.get_children()
            continue  # It's an expression, not a plain symbol
        except Exception:
            pass
        name = str(name_atom)
        s, c = parse_stv_param(children[1])
        priors[name] = {"strength": s, "confidence": c}
    return priors


def extract_links(metta, link_type):
    """Query space for link atoms: ((LinkType src dst) (stv s c)).

    Returns list of dicts: [{src, dst, strength, confidence}]
    """
    query = f"!(match &self (({link_type} $x $y) (stv $s $c)) (({link_type} $x $y) (stv $s $c)))"
    results = metta.run(query)
    links = []
    for atom in results[0]:
        children = atom.get_children()
        link_atom = children[0]
        link_children = link_atom.get_children()
        links.append({
            "src": str(link_children[1]),
            "dst": str(link_children[2]),
            "strength": _float_from_atom(children[1].get_children()[1]),
            "confidence": _float_from_atom(children[1].get_children()[2]),
        })
    return links


def extract_implications(metta):
    return extract_links(metta, "Implication")

def extract_inheritances(metta):
    return extract_links(metta, "Inheritance")

def extract_similarities(metta):
    return extract_links(metta, "Similarity")

def extract_equivalences(metta):
    return extract_links(metta, "Equivalence")

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


# ── Lookup helpers ────────────────────────────────────────────────────────

def find_prior(priors, name, default_s=0.5, default_c=0.5):
    p = priors.get(name)
    if p is None:
        return default_s, default_c
    return p["strength"], p["confidence"]

# ── Result constructors ──────────────────────────────────────────────────

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

def make_result(conclusion_atom, strength, confidence):
    return E(conclusion_atom, make_stv(strength, confidence))

def make_error(msg):
    return E(S("Error"), S(msg))
