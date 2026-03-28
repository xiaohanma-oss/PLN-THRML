"""
thrml-compile! / thrml-query! — Full-graph compilation and query.

Compile the entire MeTTa knowledge space into one beta factor graph,
run Gibbs sampling once, then answer arbitrary marginal/conditional
queries from the cached samples.

Usage in MeTTa:
    !(thrml-compile!)              ; compile & sample (returns node/factor counts)
    !(thrml-query! (B))            ; P(B=1)        — marginal
    !(thrml-query! (B A))          ; P(B=1|A=1)    — conditional
"""

from pln_thrml_beta import (
    build_beta_full_graph, run_beta_sampling,
    estimate_beta_marginal, estimate_beta_conditional,
)
from metta.atoms import (
    extract_priors, extract_links, extract_backgrounds,
    extract_negated_implications, make_stv, make_error,
)


def make_compile_op(metta_ref):
    """Create thrml-compile! operation and its shared cache."""
    cache = {}

    def thrml_compile(*atoms):
        priors = extract_priors(metta_ref)
        impls = extract_links(metta_ref, "Implication") + extract_links(metta_ref, "Inheritance")
        sims = extract_links(metta_ref, "Similarity")
        equivs = extract_links(metta_ref, "Equivalence")
        bgs = extract_backgrounds(metta_ref)
        neg_impls = extract_negated_implications(metta_ref)

        graph = build_beta_full_graph(priors, impls, sims, equivs, bgs,
                                      neg_impls)
        samples = run_beta_sampling(graph, seed=42)

        cache["graph"] = graph
        cache["samples"] = samples

        n_nodes = len(graph["nodes"])
        n_factors = len(graph["factors"])
        return [make_stv(float(n_nodes), float(n_factors))]

    return thrml_compile, cache


def make_query_op(metta_ref, cache):
    """Create thrml-query! operation that reads from the shared cache."""

    def thrml_query(*atoms):
        graph = cache.get("graph")
        samples = cache.get("samples")
        if graph is None:
            return [make_error("call thrml-compile! first")]

        if len(atoms) < 1:
            return [make_error("expected (thrml-query! (X)) or (thrml-query! (Y X))")]

        children = atoms[0].get_children()

        if len(children) == 1:
            # Marginal: P(X)
            name = str(children[0])
            node = graph["nodes"].get(name)
            if node is None:
                return [make_error(f"unknown-node {name}")]
            _, strength, confidence = estimate_beta_marginal(
                samples, graph, node)
            return [make_stv(strength, confidence)]

        elif len(children) == 2:
            # Conditional: P(target | condition=high)
            target_name = str(children[0])
            cond_name = str(children[1])
            target = graph["nodes"].get(target_name)
            cond = graph["nodes"].get(cond_name)
            if target is None:
                return [make_error(f"unknown-node {target_name}")]
            if cond is None:
                return [make_error(f"unknown-node {cond_name}")]
            _, strength, confidence = estimate_beta_conditional(
                samples, graph, target, cond)
            return [make_stv(strength, confidence)]

        return [make_error("expected (thrml-query! (X)) or (thrml-query! (Y X))")]

    return thrml_query
