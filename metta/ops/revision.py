"""
thrml-revision! — Combine two independent evidence sources via thermodynamic sampling.

Upstream: lib_pln.metta Truth_Revision
Topology: 1-node, dual Beta priors (energy addition = Beta multiplication = Bayesian revision)
"""

from pln_thrml_beta import (
    make_beta_prior_factor, sample_and_measure, DEFAULT_K,
)
from thrml.pgm import CategoricalNode
from thrml.block_management import Block
from thrml.block_sampling import BlockGibbsSpec
from thrml.models.discrete_ebm import CategoricalGibbsConditional
from thrml.factor import FactorSamplingProgram

from metta.atoms import make_stv, make_error, parse_stv_param


def _build_beta_revision_graph(s1, c1, s2, c2, k=DEFAULT_K):
    node = CategoricalNode()
    factors = [
        make_beta_prior_factor(node, s1, c1, k),
        make_beta_prior_factor(node, s2, c2, k),
    ]
    free_blocks = [Block([node])]
    spec = BlockGibbsSpec(free_blocks, [])
    sampler = CategoricalGibbsConditional(n_categories=k)
    prog = FactorSamplingProgram(
        gibbs_spec=spec, samplers=[sampler],
        factors=factors, other_interaction_groups=[])
    return dict(nodes=[node], factors=factors, free_blocks=free_blocks,
                clamped_blocks=[], spec=spec, program=prog,
                n=1, k=k, single_node=False)


def make_op(metta_ref):
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
            # Direct call convention
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
