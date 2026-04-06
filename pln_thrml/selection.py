"""
PLN rule selection via Geodesic-THRML's geodesic controller.

When multiple PLN rules apply to the same premises, this module uses
Geodesic-THRML's THRML Boltzmann step selection to pick the best one
— instead of MeTTa's default "fire all matching rules."

Usage (Python):
    from pln_thrml.metta.selection import select_and_apply
    result = select_and_apply(premises, goal_stv=(0.9, 0.9))

Usage (MeTTa, via register_all):
    !(select-thrml (A (stv 0.8 0.9)) ((Implication A B) (stv 0.9 0.85)))

Requires: pip install geodesic-thrml  (or pip install pln-thrml[geodesic])

References:
    - genenergy-logic §5: geodesic controller (Pareto on ρ = f·g / cost)
    - hyperon-whitepaper §5.2: bidirectional inference scheduling
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from pln_thrml.beta import (
    build_beta_chain,
    build_beta_v_graph,
    build_beta_inv_v_graph,
    build_beta_symmetric_chain,
    run_beta_sampling,
    estimate_beta_marginal,
    DEFAULT_EPSILON,
)


def select_and_apply(
    premises: list[dict],
    goal_stv: tuple[float, float] | None = None,
    temperature: float = 1.0,
    seed: int = 42,
    backgrounds: dict[tuple[str, str], float] | None = None,
) -> dict | None:
    """Select the best PLN rule via geodesic control, then apply it.

    1. Identify all rules applicable to the given premises
    2. For each: build factor graph → THRML Gibbs sampling → (posterior, s, c)
    3. Package as RuleSpec via geodesic_thrml.bridges.pln
    4. Call select_step_thrml() to pick the best one
    5. Return the selected result with diagnostics

    Args:
        premises: list of premise dicts, each with:
            - "atom": identifier string (e.g., "A", "B")
            - "strength": float
            - "confidence": float
            - "link_type": optional, e.g., "Implication", "Similarity"
            - "source": optional, source atom for links
            - "target": optional, target atom for links
        goal_stv: target (strength, confidence) for backward scoring.
            If None, all rules scored equally on backward factor.
        temperature: T > 0; lower = more greedy selection
        seed: random seed
        backgrounds: dict of {(src, dst): background_rate}.
            If None, uses DEFAULT_EPSILON for all.

    Returns:
        Dict with:
            name: selected rule name
            conclusion: conclusion atom identifier
            strength: conclusion strength
            confidence: conclusion confidence
            posterior: K-bin posterior histogram
            premise_confidences: input confidences
            selection: SelectionResult with diagnostics (rule_probs, energy, etc.)
        Or None if no rules apply.
    """
    from geodesic_thrml.controller_thrml import select_step_thrml
    from geodesic_thrml.bridges.pln import pln_result_to_rule_spec

    bgs = backgrounds or {}

    candidates = _enumerate_candidates(premises, bgs, seed)

    if not candidates:
        return None

    if len(candidates) == 1:
        return candidates[0]

    # Build RuleSpec for each candidate
    specs = [
        pln_result_to_rule_spec(
            name=c["name"],
            posterior=c["posterior"],
            strength=c["strength"],
            confidence=c["confidence"],
            premise_confidences=c["premise_confidences"],
        )
        for c in candidates
    ]

    # Geodesic step selection via THRML Boltzmann sampling
    result = select_step_thrml(
        specs, goal_stv=goal_stv, temperature=temperature, seed=seed)

    selected = dict(candidates[result.selected_idx])
    selected["selection"] = result
    return selected


# ═══════════════════════════════════════════════════════════════════════════
#  Rule enumeration — try each rule pattern against premises
# ═══════════════════════════════════════════════════════════════════════════

def _enumerate_candidates(
    premises: list[dict],
    backgrounds: dict,
    seed: int,
) -> list[dict]:
    """Try all applicable rules and collect results."""
    candidates = []

    # Classify premises
    atoms = [p for p in premises if "link_type" not in p or p["link_type"] is None]
    links = [p for p in premises if "link_type" in p and p["link_type"] is not None]

    syllogistic = [l for l in links if l["link_type"] in ("Implication", "Inheritance")]
    symmetric = [l for l in links if l["link_type"] in ("Similarity", "IntentionalSimilarity", "ExtensionalSimilarity")]

    def bg(src, dst):
        return backgrounds.get((src, dst), DEFAULT_EPSILON)

    # --- Single-premise rules ---

    # Inversion: (LinkType A B) → (LinkType B A)
    for link in syllogistic:
        src, dst = link["source"], link["target"]
        c = _try_rule(
            "inversion",
            lambda: build_beta_chain(
                priors=[0.5, 0.99], confidences=[0.01, 0.99],
                strengths=[link["strength"]], impl_confidences=[link["confidence"]],
                backgrounds=[bg(src, dst)], clamp_root=False),
            target_idx=0,
            premise_confidences=[link["confidence"]],
            conclusion=f"({link['link_type']} {dst} {src})",
            seed=seed,
        )
        if c:
            candidates.append(c)

    # --- Two-premise rules: atom + link ---

    for atom in atoms:
        for link in syllogistic:
            if link["source"] == atom["atom"]:
                src, dst = link["source"], link["target"]

                # Modus Ponens: A, A→B ⊢ B
                c = _try_rule(
                    "modus-ponens",
                    lambda: build_beta_chain(
                        priors=[atom["strength"], 0.5],
                        confidences=[atom["confidence"], 0.01],
                        strengths=[link["strength"]],
                        impl_confidences=[link["confidence"]],
                        backgrounds=[bg(src, dst)]),
                    target_idx=1,
                    premise_confidences=[atom["confidence"], link["confidence"]],
                    conclusion=dst,
                    seed=seed,
                )
                if c:
                    candidates.append(c)

        for link in symmetric:
            if link["source"] == atom["atom"]:
                src, dst = link["source"], link["target"]

                # Symmetric Modus Ponens: A, A~B ⊢ B
                bg_val = 0.2 * (1.0 + link["strength"])
                c = _try_rule(
                    "symmetric-mp",
                    lambda: build_beta_chain(
                        priors=[atom["strength"], 0.5],
                        confidences=[atom["confidence"], 0.01],
                        strengths=[link["strength"]],
                        impl_confidences=[link["confidence"]],
                        backgrounds=[bg_val]),
                    target_idx=1,
                    premise_confidences=[atom["confidence"], link["confidence"]],
                    conclusion=dst,
                    seed=seed,
                )
                if c:
                    candidates.append(c)

    # --- Two-premise rules: link + link ---

    for i, l1 in enumerate(syllogistic):
        for l2 in syllogistic[i + 1:]:
            # Deduction: A→B, B→C ⊢ A→C
            if l1["target"] == l2["source"]:
                A, B, C = l1["source"], l1["target"], l2["target"]
                c = _try_rule(
                    "deduction",
                    lambda: build_beta_chain(
                        priors=[0.99, 0.5, 0.5],
                        confidences=[0.99, 0.01, 0.01],
                        strengths=[l1["strength"], l2["strength"]],
                        impl_confidences=[l1["confidence"], l2["confidence"]],
                        backgrounds=[bg(A, B), bg(B, C)],
                        clamp_root=False),
                    target_idx=2,
                    premise_confidences=[l1["confidence"], l2["confidence"]],
                    conclusion=f"({l1['link_type']} {A} {C})",
                    seed=seed,
                )
                if c:
                    candidates.append(c)

            # Induction: C→A, C→B ⊢ A→B
            if l1["source"] == l2["source"] and l1["target"] != l2["target"]:
                C, A, B = l1["source"], l1["target"], l2["target"]
                c = _try_rule(
                    "induction",
                    lambda: build_beta_v_graph(
                        root_prior=0.5, root_confidence=0.01,
                        left_strength=l1["strength"],
                        right_strength=l2["strength"],
                        left_impl_confidence=l1["confidence"],
                        right_impl_confidence=l2["confidence"],
                        left_background=bg(C, A),
                        right_background=bg(C, B),
                        left_prior=0.99, left_confidence=0.99,
                        right_prior=0.5, right_confidence=0.01),
                    target_key="right",
                    premise_confidences=[l1["confidence"], l2["confidence"]],
                    conclusion=f"({l1['link_type']} {A} {B})",
                    seed=seed,
                )
                if c:
                    candidates.append(c)

            # Abduction: A→C, B→C ⊢ A→B
            if l1["target"] == l2["target"] and l1["source"] != l2["source"]:
                A, B, C = l1["source"], l2["source"], l1["target"]
                c = _try_rule(
                    "abduction",
                    lambda: build_beta_inv_v_graph(
                        left_prior=0.99, left_confidence=0.99,
                        right_prior=0.5, right_confidence=0.01,
                        left_strength=l1["strength"],
                        right_strength=l2["strength"],
                        left_impl_confidence=l1["confidence"],
                        right_impl_confidence=l2["confidence"],
                        left_background=bg(A, C),
                        right_background=bg(B, C),
                        center_prior=0.5, center_confidence=0.01),
                    target_key="right",
                    premise_confidences=[l1["confidence"], l2["confidence"]],
                    conclusion=f"({l1['link_type']} {A} {B})",
                    seed=seed,
                )
                if c:
                    candidates.append(c)

    # --- Revision: same atom, two different truth values ---
    if len(atoms) == 2 and atoms[0]["atom"] == atoms[1]["atom"]:
        from pln_thrml.beta import (
            _assemble_free_graph, make_beta_prior_factor, DEFAULT_K,
        )
        from thrml.pgm import CategoricalNode
        from thrml.block_management import Block

        a1, a2 = atoms[0], atoms[1]
        node = CategoricalNode()
        factors = [
            make_beta_prior_factor(node, a1["strength"], a1["confidence"], DEFAULT_K),
            make_beta_prior_factor(node, a2["strength"], a2["confidence"], DEFAULT_K),
        ]
        graph = _assemble_free_graph([node], factors, [Block([node])], DEFAULT_K)
        c = _try_rule_from_graph(
            "revision", graph, target_idx=0,
            premise_confidences=[a1["confidence"], a2["confidence"]],
            conclusion=a1["atom"],
            seed=seed,
        )
        if c:
            candidates.append(c)

    return candidates


# ═══════════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _try_rule(
    name: str,
    build_fn,
    premise_confidences: list[float],
    conclusion: str,
    seed: int,
    target_idx: int | None = None,
    target_key: str | None = None,
) -> dict | None:
    """Build graph, sample, return result dict or None on failure."""
    try:
        graph = build_fn()
        return _try_rule_from_graph(
            name, graph, premise_confidences=premise_confidences,
            conclusion=conclusion, seed=seed,
            target_idx=target_idx, target_key=target_key,
        )
    except Exception:
        return None


def _try_rule_from_graph(
    name: str,
    graph: dict,
    premise_confidences: list[float],
    conclusion: str,
    seed: int,
    target_idx: int | None = None,
    target_key: str | None = None,
) -> dict | None:
    """Sample from a pre-built graph and return result dict."""
    try:
        if target_key is not None:
            target = graph[target_key]
        elif target_idx is not None:
            target = graph["nodes"][target_idx]
        else:
            target = graph["nodes"][-1]

        samples = run_beta_sampling(graph, seed=seed)
        posterior, s, c = estimate_beta_marginal(samples, graph, target)

        return {
            "name": name,
            "conclusion": conclusion,
            "strength": float(s),
            "confidence": float(c),
            "posterior": np.array(posterior),
            "premise_confidences": list(premise_confidences),
        }
    except Exception:
        return None
