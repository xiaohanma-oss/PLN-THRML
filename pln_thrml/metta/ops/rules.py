"""
Declarative rule table — all sampling-based PLN inference ops in one place.

Each rule is a small build function that extracts parameters from MeTTa atoms,
constructs a factor graph, runs sampling, and returns (strength, confidence).
The generic factory `make_rule_op` wraps each into a grounded MeTTa operation.
"""

from pln_thrml.beta import (
    build_beta_chain, build_beta_v_graph, build_beta_inv_v_graph,
    build_beta_symmetric_chain, sample_and_measure, DEFAULT_EPSILON,
    _assemble_free_graph, make_beta_prior_factor, DEFAULT_K,
)
from thrml.pgm import CategoricalNode
from thrml.block_management import Block
from pln_thrml.metta.atoms import (
    extract_backgrounds, parse_stv_param, make_stv, make_error,
    validate_op_args,
)

__all__ = ["RULE_SPECS", "make_rule_op", "make_revision_op"]


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


# ── Rule table ───────────────────────────────────────────────────────────

RULE_SPECS = {
    "thrml-modus-ponens!":   (4, "A B (stv ..) (stv ..)", _build_modus_ponens),
    "thrml-deduction!":      (5, "A B C (stv ..) (stv ..)", _build_deduction),
    "thrml-inversion!":      (3, "A B (stv ..)", _build_inversion),
    "thrml-induction!":      (5, "A B C (stv ..) (stv ..)", _build_induction),
    "thrml-abduction!":      (5, "A B C (stv ..) (stv ..)", _build_abduction),
    "thrml-equiv-to-impl!":  (3, "A B (stv ..)", _build_equiv_to_impl),
    "thrml-transitive-sim!": (5, "A B C (stv ..) (stv ..)", _build_transitive_sim),
    "thrml-eval-impl!":      (5, "A B C (stv ..) (stv ..)", _build_eval_impl),
    "thrml-symmetric-mp!":   (4, "A B (stv ..) (stv ..)", _build_symmetric_mp),
    "thrml-negation!":       (2, "A (stv ..)", _build_negation),
}


def make_rule_op(name, metta_ref):
    """Generic factory: validate → build graph → sample → make_stv."""
    min_children, fmt, build_fn = RULE_SPECS[name]

    def op(*atoms):
        children, err = validate_op_args(atoms, min_children, name, fmt)
        if err:
            return err
        strength, confidence = build_fn(children, metta_ref)
        return [make_stv(strength, confidence)]

    return op


# ── Revision (dual calling convention — not in RULE_SPECS) ──────────────


def _build_beta_revision_graph(s1, c1, s2, c2, k=DEFAULT_K):
    node = CategoricalNode()
    factors = [
        make_beta_prior_factor(node, s1, c1, k),
        make_beta_prior_factor(node, s2, c2, k),
    ]
    return _assemble_free_graph([node], factors, [Block([node])], k)


def make_revision_op(metta_ref):
    """Create thrml-revision! operation with dual calling convention.

    Supports two input formats:
    - MeTTa rule format: ``(thrml-revision! (T (stv s1 c1) (stv s2 c2)))``
    - Direct call format: ``(thrml-revision! (T s1 c1 s2 c2))``

    Both produce a revised truth value by combining two independent Beta
    priors on a single node and sampling the posterior.
    """

    def thrml_revision(*atoms):
        if len(atoms) < 1:
            return [make_error("expected (thrml-revision! (A s1 c1 s2 c2)) or (thrml-revision! (T (stv s1 c1) (stv s2 c2)))")]

        children = atoms[0].get_children()

        # Support two calling conventions:
        # 1. |-thrml rule: (term (stv s1 c1) (stv s2 c2)) — 3 children
        # 2. Direct call: (name s1 c1 s2 c2) — 5 children
        is_stv_format = False
        if len(children) >= 3:
            try:
                sub = children[1].get_children()
                if str(sub[0]) == "stv":
                    is_stv_format = True
            except Exception:
                pass

        if is_stv_format:
            s1, c1 = parse_stv_param(children[1])
            s2, c2 = parse_stv_param(children[2])
        elif len(children) >= 5:
            s1 = float(str(children[1]))
            c1 = float(str(children[2]))
            s2 = float(str(children[3]))
            c2 = float(str(children[4]))
        else:
            return [make_error("expected (thrml-revision! (A s1 c1 s2 c2)) or (thrml-revision! (T (stv s1 c1) (stv s2 c2)))")]

        graph = _build_beta_revision_graph(s1, c1, s2, c2)
        strength, confidence = sample_and_measure(graph, graph["nodes"][0])
        return [make_stv(strength, confidence)]

    return thrml_revision
