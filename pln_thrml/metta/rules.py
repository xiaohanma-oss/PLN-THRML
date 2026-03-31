"""
PLN inference rules compiled to thermodynamic factor graphs.

Each rule is a small build function that extracts parameters from MeTTa atoms,
constructs a factor graph, runs sampling, and returns (strength, confidence).
The unified `thrml` operator dispatches to the correct rule based on the
premise atom structure.
"""

from hyperon import E, S, ValueAtom, OperationAtom

from pln_thrml.beta import (
    build_beta_chain, build_beta_v_graph, build_beta_inv_v_graph,
    build_beta_symmetric_chain, sample_and_measure, DEFAULT_EPSILON,
    _assemble_free_graph, make_beta_prior_factor, DEFAULT_K,
    MAX_CONFIDENCE,
)
from thrml.pgm import CategoricalNode
from thrml.block_management import Block

__all__ = ["register_all"]


# ═══════════════════════════════════════════════════════════════════════════
#  Atom helpers
# ═══════════════════════════════════════════════════════════════════════════

def _float_from_atom(atom):
    """Extract a float from a MeTTa atom (ValueAtom or symbol)."""
    return float(str(atom))


def parse_stv_param(atom):
    """Parse (stv s c) atom into (strength, confidence) tuple.

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


def make_stv(strength, confidence):
    return E(S("stv"), ValueAtom(strength), ValueAtom(confidence))


def make_error(msg):
    return E(S("Error"), S(msg))


# ═══════════════════════════════════════════════════════════════════════════
#  Rule builders
# ═══════════════════════════════════════════════════════════════════════════

def _bg(metta_ref, src, dst):
    return extract_backgrounds(metta_ref).get((src, dst), DEFAULT_EPSILON)


def _build_modus_ponens(children, metta_ref):
    src, dst = str(children[0]), str(children[1])
    s_AB, c_AB = parse_stv_param(children[2])
    s_A, c_A = parse_stv_param(children[3])
    graph = build_beta_chain(
        priors=[s_A, 0.5], confidences=[c_A, 0.01],
        strengths=[s_AB], impl_confidences=[c_AB],
        backgrounds=[_bg(metta_ref, src, dst)])
    return sample_and_measure(graph, graph["nodes"][1])


def _build_deduction(children, metta_ref):
    A, B, C = str(children[0]), str(children[1]), str(children[2])
    s_AB, c_AB = parse_stv_param(children[3])
    s_BC, c_BC = parse_stv_param(children[4])
    bgs = extract_backgrounds(metta_ref)
    graph = build_beta_chain(
        priors=[0.99, 0.5, 0.5], confidences=[0.99, 0.01, 0.01],
        strengths=[s_AB, s_BC], impl_confidences=[c_AB, c_BC],
        backgrounds=[bgs.get((A, B), DEFAULT_EPSILON),
                     bgs.get((B, C), DEFAULT_EPSILON)],
        clamp_root=False)
    return sample_and_measure(graph, graph["nodes"][2])


def _build_inversion(children, metta_ref):
    A, B = str(children[0]), str(children[1])
    s_AB, c_AB = parse_stv_param(children[2])
    graph = build_beta_chain(
        priors=[0.5, 0.99], confidences=[0.01, 0.99],
        strengths=[s_AB], impl_confidences=[c_AB],
        backgrounds=[_bg(metta_ref, A, B)], clamp_root=False)
    return sample_and_measure(graph, graph["nodes"][0])


def _build_induction(children, metta_ref):
    A, B, C = str(children[0]), str(children[1]), str(children[2])
    s_CA, c_CA = parse_stv_param(children[3])
    s_CB, c_CB = parse_stv_param(children[4])
    bgs = extract_backgrounds(metta_ref)
    graph = build_beta_v_graph(
        root_prior=0.5, root_confidence=0.01,
        left_strength=s_CA, right_strength=s_CB,
        left_impl_confidence=c_CA, right_impl_confidence=c_CB,
        left_background=bgs.get((C, A), DEFAULT_EPSILON),
        right_background=bgs.get((C, B), DEFAULT_EPSILON),
        left_prior=0.99, left_confidence=0.99,
        right_prior=0.5, right_confidence=0.01)
    return sample_and_measure(graph, graph["right"])


def _build_abduction(children, metta_ref):
    A, B, C = str(children[0]), str(children[1]), str(children[2])
    s_AC, c_AC = parse_stv_param(children[3])
    s_BC, c_BC = parse_stv_param(children[4])
    bgs = extract_backgrounds(metta_ref)
    graph = build_beta_inv_v_graph(
        left_prior=0.99, left_confidence=0.99,
        right_prior=0.5, right_confidence=0.01,
        left_strength=s_AC, right_strength=s_BC,
        left_impl_confidence=c_AC, right_impl_confidence=c_BC,
        left_background=bgs.get((A, C), DEFAULT_EPSILON),
        right_background=bgs.get((B, C), DEFAULT_EPSILON),
        center_prior=0.5, center_confidence=0.01)
    return sample_and_measure(graph, graph["right"])


def _build_equiv_to_impl(children, metta_ref):
    A, B = str(children[0]), str(children[1])
    s_AB, c_AB = parse_stv_param(children[2])
    graph = build_beta_symmetric_chain(
        priors=[0.99, 0.5], confidences=[0.99, 0.01],
        strengths=[s_AB], impl_confidences=[c_AB],
        backgrounds=[_bg(metta_ref, A, B)])
    return sample_and_measure(graph, graph["nodes"][1])


def _build_transitive_sim(children, metta_ref):
    A, B, C = str(children[0]), str(children[1]), str(children[2])
    s_AB, c_AB = parse_stv_param(children[3])
    s_BC, c_BC = parse_stv_param(children[4])
    bgs = extract_backgrounds(metta_ref)
    graph = build_beta_symmetric_chain(
        priors=[0.99, 0.5, 0.5], confidences=[0.99, 0.01, 0.01],
        strengths=[s_AB, s_BC], impl_confidences=[c_AB, c_BC],
        backgrounds=[bgs.get((A, B), DEFAULT_EPSILON),
                     bgs.get((B, C), DEFAULT_EPSILON)])
    return sample_and_measure(graph, graph["nodes"][2])


def _build_eval_impl(children, metta_ref):
    A, B, C = str(children[0]), str(children[1]), str(children[2])
    s_AB, c_AB = parse_stv_param(children[3])
    s_AC, c_AC = parse_stv_param(children[4])
    bgs = extract_backgrounds(metta_ref)
    graph = build_beta_chain(
        priors=[0.99, 0.5, 0.5], confidences=[0.99, 0.01, 0.01],
        strengths=[s_AB, s_AC], impl_confidences=[c_AB, c_AC],
        backgrounds=[bgs.get((B, A), DEFAULT_EPSILON),
                     bgs.get((A, C), DEFAULT_EPSILON)],
        clamp_root=False)
    return sample_and_measure(graph, graph["nodes"][2])


def _build_symmetric_mp(children, metta_ref):
    s_A, c_A = parse_stv_param(children[2])
    s_AB, c_AB = parse_stv_param(children[3])
    bg = 0.2 * (1.0 + s_AB)
    graph = build_beta_chain(
        priors=[s_A, 0.5], confidences=[c_A, 0.01],
        strengths=[s_AB], impl_confidences=[c_AB], backgrounds=[bg])
    return sample_and_measure(graph, graph["nodes"][1])


def _build_negation(children, metta_ref):
    s, c = parse_stv_param(children[1])
    return (1.0 - s, c)


def _build_beta_revision_graph(s1, c1, s2, c2, k=DEFAULT_K):
    node = CategoricalNode()
    factors = [
        make_beta_prior_factor(node, s1, c1, k),
        make_beta_prior_factor(node, s2, c2, k),
    ]
    return _assemble_free_graph([node], factors, [Block([node])], k)


# ═══════════════════════════════════════════════════════════════════════════
#  Unified thrml inference operator
# ═══════════════════════════════════════════════════════════════════════════

SYLLOGISTIC_LINKS = {"Implication", "Inheritance"}
SYMMETRIC_LINKS = {"Similarity", "IntentionalSimilarity", "ExtensionalSimilarity"}


def _parse_premise(atom):
    """Parse a premise atom like (A (stv 0.8 0.9)) or ((Implication A B) (stv 0.9 0.85)).

    Returns (content, stv, link_type, args) where:
    - content: the subject atom (A or (Implication A B))
    - stv: the (stv s c) atom
    - link_type: str name of link ("Implication", "Not", etc.) or None for bare symbols
    - args: list of link arguments ([A, B] for (Implication A B)) or []
    """
    children = atom.get_children()
    content = children[0]
    stv = children[1]
    try:
        inner = content.get_children()
        if len(inner) >= 1:
            link_type = str(inner[0])
            args = list(inner[1:])
            return content, stv, link_type, args
    except Exception:
        pass
    return content, stv, None, []


def _make_conclusion(content, strength, confidence):
    """Wrap a conclusion atom with its truth value: (content (stv s c))."""
    return [E(content, make_stv(strength, confidence))]


def _dispatch_single(premise_atom, metta_ref):
    """Dispatch a single-premise rule based on atom structure."""
    content, stv, link_type, args = _parse_premise(premise_atom)

    if link_type == "Not" and len(args) >= 1:
        s, c = _build_negation([args[0], stv], metta_ref)
        return _make_conclusion(args[0], s, c)

    if link_type == "Equivalence" and len(args) >= 2:
        s, c = _build_equiv_to_impl([args[0], args[1], stv], metta_ref)
        return _make_conclusion(E(S("Implication"), args[0], args[1]), s, c)

    if link_type in SYLLOGISTIC_LINKS and len(args) >= 2:
        s, c = _build_inversion([args[0], args[1], stv], metta_ref)
        return _make_conclusion(E(S(link_type), args[1], args[0]), s, c)

    return [make_error("thrml: unrecognized single-premise structure")]


def _dispatch_pair(atom1, atom2, metta_ref):
    """Dispatch a two-premise rule based on atom structure."""
    c1, stv1, lt1, a1 = _parse_premise(atom1)
    c2, stv2, lt2, a2 = _parse_premise(atom2)

    # Revision: same subject, different stvs
    if str(c1) == str(c2):
        s1, conf1 = parse_stv_param(stv1)
        s2, conf2 = parse_stv_param(stv2)
        graph = _build_beta_revision_graph(s1, conf1, s2, conf2)
        s, c = sample_and_measure(graph, graph["nodes"][0])
        return _make_conclusion(c1, s, c)

    # p1 bare atom, p2 is a link
    if lt1 is None and lt2 is not None and len(a2) >= 2:
        if lt2 in SYLLOGISTIC_LINKS and str(c1) == str(a2[0]):
            children = [c1, a2[1], stv2, stv1]
            s, c = _build_modus_ponens(children, metta_ref)
            return _make_conclusion(a2[1], s, c)

        if lt2 in SYMMETRIC_LINKS and str(c1) == str(a2[0]):
            children = [c1, a2[1], stv1, stv2]
            s, c = _build_symmetric_mp(children, metta_ref)
            return _make_conclusion(a2[1], s, c)

    # Both premises are links
    if lt1 is not None and lt2 is not None and len(a1) >= 2 and len(a2) >= 2:
        if lt1 == lt2 and lt1 in SYLLOGISTIC_LINKS and str(a1[1]) == str(a2[0]):
            children = [a1[0], a1[1], a2[1], stv1, stv2]
            s, c = _build_deduction(children, metta_ref)
            return _make_conclusion(E(S(lt1), a1[0], a2[1]), s, c)

        if lt1 == lt2 and lt1 in SYMMETRIC_LINKS and str(a1[1]) == str(a2[0]):
            children = [a1[0], a1[1], a2[1], stv1, stv2]
            s, c = _build_transitive_sim(children, metta_ref)
            return _make_conclusion(E(S(lt1), a1[0], a2[1]), s, c)

        if lt1 == lt2 and lt1 in SYLLOGISTIC_LINKS and str(a1[0]) == str(a2[0]):
            children = [a1[1], a2[1], a1[0], stv1, stv2]
            s, c = _build_induction(children, metta_ref)
            return _make_conclusion(E(S(lt1), a1[1], a2[1]), s, c)

        if lt1 == lt2 and lt1 in SYLLOGISTIC_LINKS and str(a1[1]) == str(a2[1]):
            children = [a1[0], a2[0], a1[1], stv1, stv2]
            s, c = _build_abduction(children, metta_ref)
            return _make_conclusion(E(S(lt1), a1[0], a2[0]), s, c)

        if lt1 == "Evaluation" and lt2 in SYLLOGISTIC_LINKS and str(a1[0]) == str(a2[0]):
            children = [a2[0], a1[1], a2[1], stv1, stv2]
            s, c = _build_eval_impl(children, metta_ref)
            return _make_conclusion(E(S("Evaluation"), a2[1], a1[1]), s, c)

    return [make_error("thrml: no matching rule for the given premises")]


def _make_thrml_infer_op(metta_ref):
    """Create the unified thrml inference operator.

    Accepts 1 or 2 premises in upstream PLN |- format and dispatches to the
    appropriate rule builder based on atom structure.

    Single premise:
        !(thrml ((Implication A B) (stv 0.87 0.81)))  -> Inversion

    Two premises:
        !(thrml (A (stv 0.8 0.9)) ((Implication A B) (stv 0.9 0.85)))  -> Modus Ponens
    """
    def thrml_infer(*atoms):
        if len(atoms) == 1:
            return _dispatch_single(atoms[0], metta_ref)
        if len(atoms) == 2:
            return _dispatch_pair(atoms[0], atoms[1], metta_ref)
        return [make_error("thrml expects 1 or 2 premises")]
    return thrml_infer


# ═══════════════════════════════════════════════════════════════════════════
#  Registration
# ═══════════════════════════════════════════════════════════════════════════

def register_all(metta):
    """Register the thrml inference operator with a MeTTa runner."""
    infer_op = _make_thrml_infer_op(metta)
    metta.register_atom("thrml", OperationAtom("thrml", infer_op, unwrap=False))
