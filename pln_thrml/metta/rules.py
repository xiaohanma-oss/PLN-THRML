"""
PLN inference rules compiled to thermodynamic factor graphs.

Each rule is a grounded operation that receives raw (s, c) numbers from
MeTTa dispatch rules, constructs a factor graph, runs sampling, and
returns (stv strength confidence).  STV destructuring happens in MeTTa
(dispatch.metta), not here.
"""

import os

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


def _clamp_sc(s_atom, c_atom):
    """Convert MeTTa number atoms to clamped (strength, confidence) floats."""
    s = max(0.0, min(1.0, _float_from_atom(s_atom)))
    c = max(0.0, min(MAX_CONFIDENCE, _float_from_atom(c_atom)))
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


# ═══════════════════════════════════════════════════════════════════════════
#  Background rate helper
# ═══════════════════════════════════════════════════════════════════════════

def _bg(metta_ref, src, dst):
    return extract_backgrounds(metta_ref).get((src, dst), DEFAULT_EPSILON)


# ═══════════════════════════════════════════════════════════════════════════
#  Grounded operations — each wraps a factor graph builder + sampler
# ═══════════════════════════════════════════════════════════════════════════

def _make_modus_ponens_op(metta_ref):
    def op(src_atom, dst_atom, s1, c1, s2, c2):
        src, dst = str(src_atom), str(dst_atom)
        s_A, c_A = _clamp_sc(s1, c1)
        s_AB, c_AB = _clamp_sc(s2, c2)
        graph = build_beta_chain(
            priors=[s_A, 0.5], confidences=[c_A, 0.01],
            strengths=[s_AB], impl_confidences=[c_AB],
            backgrounds=[_bg(metta_ref, src, dst)])
        s, c = sample_and_measure(graph, graph["nodes"][1])
        return [make_stv(s, c)]
    return op


def _make_deduction_op(metta_ref):
    def op(a_atom, b_atom, c_atom, s1, c1, s2, c2):
        A, B, C = str(a_atom), str(b_atom), str(c_atom)
        s_AB, c_AB = _clamp_sc(s1, c1)
        s_BC, c_BC = _clamp_sc(s2, c2)
        bgs = extract_backgrounds(metta_ref)
        graph = build_beta_chain(
            priors=[0.99, 0.5, 0.5], confidences=[0.99, 0.01, 0.01],
            strengths=[s_AB, s_BC], impl_confidences=[c_AB, c_BC],
            backgrounds=[bgs.get((A, B), DEFAULT_EPSILON),
                         bgs.get((B, C), DEFAULT_EPSILON)],
            clamp_root=False)
        s, c = sample_and_measure(graph, graph["nodes"][2])
        return [make_stv(s, c)]
    return op


def _make_inversion_op(metta_ref):
    def op(a_atom, b_atom, s1, c1):
        A, B = str(a_atom), str(b_atom)
        s_AB, c_AB = _clamp_sc(s1, c1)
        graph = build_beta_chain(
            priors=[0.5, 0.99], confidences=[0.01, 0.99],
            strengths=[s_AB], impl_confidences=[c_AB],
            backgrounds=[_bg(metta_ref, A, B)], clamp_root=False)
        s, c = sample_and_measure(graph, graph["nodes"][0])
        return [make_stv(s, c)]
    return op


def _make_induction_op(metta_ref):
    def op(a_atom, b_atom, c_atom, s1, c1, s2, c2):
        A, B, C = str(a_atom), str(b_atom), str(c_atom)
        s_CA, c_CA = _clamp_sc(s1, c1)
        s_CB, c_CB = _clamp_sc(s2, c2)
        bgs = extract_backgrounds(metta_ref)
        graph = build_beta_v_graph(
            root_prior=0.5, root_confidence=0.01,
            left_strength=s_CA, right_strength=s_CB,
            left_impl_confidence=c_CA, right_impl_confidence=c_CB,
            left_background=bgs.get((C, A), DEFAULT_EPSILON),
            right_background=bgs.get((C, B), DEFAULT_EPSILON),
            left_prior=0.99, left_confidence=0.99,
            right_prior=0.5, right_confidence=0.01)
        s, c = sample_and_measure(graph, graph["right"])
        return [make_stv(s, c)]
    return op


def _make_abduction_op(metta_ref):
    def op(a_atom, b_atom, c_atom, s1, c1, s2, c2):
        A, B, C = str(a_atom), str(b_atom), str(c_atom)
        s_AC, c_AC = _clamp_sc(s1, c1)
        s_BC, c_BC = _clamp_sc(s2, c2)
        bgs = extract_backgrounds(metta_ref)
        graph = build_beta_inv_v_graph(
            left_prior=0.99, left_confidence=0.99,
            right_prior=0.5, right_confidence=0.01,
            left_strength=s_AC, right_strength=s_BC,
            left_impl_confidence=c_AC, right_impl_confidence=c_BC,
            left_background=bgs.get((A, C), DEFAULT_EPSILON),
            right_background=bgs.get((B, C), DEFAULT_EPSILON),
            center_prior=0.5, center_confidence=0.01)
        s, c = sample_and_measure(graph, graph["right"])
        return [make_stv(s, c)]
    return op


def _make_equiv_to_impl_op(metta_ref):
    def op(a_atom, b_atom, s1, c1):
        A, B = str(a_atom), str(b_atom)
        s_AB, c_AB = _clamp_sc(s1, c1)
        graph = build_beta_symmetric_chain(
            priors=[0.99, 0.5], confidences=[0.99, 0.01],
            strengths=[s_AB], impl_confidences=[c_AB],
            backgrounds=[_bg(metta_ref, A, B)])
        s, c = sample_and_measure(graph, graph["nodes"][1])
        return [make_stv(s, c)]
    return op


def _make_transitive_sim_op(metta_ref):
    def op(a_atom, b_atom, c_atom, s1, c1, s2, c2):
        A, B, C = str(a_atom), str(b_atom), str(c_atom)
        s_AB, c_AB = _clamp_sc(s1, c1)
        s_BC, c_BC = _clamp_sc(s2, c2)
        bgs = extract_backgrounds(metta_ref)
        graph = build_beta_symmetric_chain(
            priors=[0.99, 0.5, 0.5], confidences=[0.99, 0.01, 0.01],
            strengths=[s_AB, s_BC], impl_confidences=[c_AB, c_BC],
            backgrounds=[bgs.get((A, B), DEFAULT_EPSILON),
                         bgs.get((B, C), DEFAULT_EPSILON)])
        s, c = sample_and_measure(graph, graph["nodes"][2])
        return [make_stv(s, c)]
    return op


def _make_eval_impl_op(metta_ref):
    def op(a_atom, b_atom, c_atom, s1, c1, s2, c2):
        A, B, C = str(a_atom), str(b_atom), str(c_atom)
        s_AB, c_AB = _clamp_sc(s1, c1)
        s_AC, c_AC = _clamp_sc(s2, c2)
        bgs = extract_backgrounds(metta_ref)
        graph = build_beta_chain(
            priors=[0.99, 0.5, 0.5], confidences=[0.99, 0.01, 0.01],
            strengths=[s_AB, s_AC], impl_confidences=[c_AB, c_AC],
            backgrounds=[bgs.get((B, A), DEFAULT_EPSILON),
                         bgs.get((A, C), DEFAULT_EPSILON)],
            clamp_root=False)
        s, c = sample_and_measure(graph, graph["nodes"][2])
        return [make_stv(s, c)]
    return op


def _make_symmetric_mp_op(metta_ref):
    def op(src_atom, dst_atom, s1, c1, s2, c2):
        s_A, c_A = _clamp_sc(s1, c1)
        s_AB, c_AB = _clamp_sc(s2, c2)
        bg = 0.2 * (1.0 + s_AB)
        graph = build_beta_chain(
            priors=[s_A, 0.5], confidences=[c_A, 0.01],
            strengths=[s_AB], impl_confidences=[c_AB], backgrounds=[bg])
        s, c = sample_and_measure(graph, graph["nodes"][1])
        return [make_stv(s, c)]
    return op


def _make_revision_op(metta_ref):
    def op(s1, c1, s2, c2):
        s1v, c1v = _clamp_sc(s1, c1)
        s2v, c2v = _clamp_sc(s2, c2)
        node = CategoricalNode()
        factors = [
            make_beta_prior_factor(node, s1v, c1v, DEFAULT_K),
            make_beta_prior_factor(node, s2v, c2v, DEFAULT_K),
        ]
        graph = _assemble_free_graph([node], factors, [Block([node])], DEFAULT_K)
        s, c = sample_and_measure(graph, graph["nodes"][0])
        return [make_stv(s, c)]
    return op


# ═══════════════════════════════════════════════════════════════════════════
#  Registration
# ═══════════════════════════════════════════════════════════════════════════

def _load_metta_file(metta, filename):
    """Load a .metta file relative to this package directory."""
    path = os.path.join(os.path.dirname(__file__), filename)
    with open(path) as f:
        metta.run(f.read())


def register_all(metta):
    """Register grounded ops and load MeTTa dispatch rules."""
    # Grounded operations (factor graph builders + samplers)
    ops = {
        "thrml-modus-ponens": _make_modus_ponens_op(metta),
        "thrml-deduction": _make_deduction_op(metta),
        "thrml-inversion": _make_inversion_op(metta),
        "thrml-induction": _make_induction_op(metta),
        "thrml-abduction": _make_abduction_op(metta),
        "thrml-equiv-to-impl": _make_equiv_to_impl_op(metta),
        "thrml-transitive-sim": _make_transitive_sim_op(metta),
        "thrml-eval-impl": _make_eval_impl_op(metta),
        "thrml-symmetric-mp": _make_symmetric_mp_op(metta),
        "thrml-revision": _make_revision_op(metta),
    }
    for name, fn in ops.items():
        metta.register_atom(name, OperationAtom(name, fn, unwrap=False))

    # Load type declarations and dispatch rules
    _load_metta_file(metta, "declarations/pln_types.metta")
    _load_metta_file(metta, "dispatch.metta")
