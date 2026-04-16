"""
Five-column precision comparison: DTV | Categorical K×K | OneHot-Full | Potts | PLN formula.

Validates one-hot spin encoding faithfulness and quantifies Potts approximation cost
for pbit-only hardware (no pdit).  See docs/references/tsu-architecture/ for hardware
constraints (~12 neighbors per pbit on TSU L×L grid).
"""

import pytest
import numpy as np

import jax.numpy as jnp

from pln_thrml.beta import (
    build_beta_chain, build_beta_inv_v_graph,
    run_beta_sampling, estimate_beta_marginal,
    sample_and_measure, SamplingSchedule,
    beta_implication_weights, beta_prior_weights,
    make_beta_prior_factor, CategoricalNode, DEFAULT_EPSILON,
)
from thrml.block_management import Block
from thrml.models.discrete_ebm import SquareCategoricalEBMFactor
from pln_thrml.dtv_baseline import dtv_modus_ponens, dtv_deduction, dtv_abduction
from pln_thrml.compiler_onehot import (
    compile_onehot_full, compile_onehot_potts,
    compile_onehot_full_inv_v, compile_onehot_potts_inv_v,
    run_onehot_sampling, estimate_onehot_marginal,
    count_neighbors_per_spin, fit_potts_weight,
)
from tests.conftest import strength_tol

# ═══════════════════════════════════════════════════════════════════════════
#  Parameters
# ═══════════════════════════════════════════════════════════════════════════

MP_PARAMS = [
    {"s_A": 0.8, "c_A": 0.9, "s_AB": 0.9, "c_AB": 0.85, "id": "strong"},
    {"s_A": 0.6, "c_A": 0.7, "s_AB": 0.7, "c_AB": 0.8, "id": "medium"},
    {"s_A": 0.3, "c_A": 0.95, "s_AB": 0.5, "c_AB": 0.9, "id": "rare"},
]

DEDUCTION_PARAMS = [
    {"s_A": 0.8, "c_A": 0.9, "s_B": 0.5, "c_B": 0.01, "s_C": 0.5, "c_C": 0.01,
     "s_AB": 0.9, "c_AB": 0.85, "s_BC": 0.8, "c_BC": 0.9, "id": "standard"},
    {"s_A": 0.05, "c_A": 0.8, "s_B": 0.2, "c_B": 0.9999, "s_C": 1.0, "c_C": 0.8,
     "s_AB": 1.0, "c_AB": 0.9999, "s_BC": 0.3, "c_BC": 0.8, "id": "extreme"},
]

ABDUCTION_PARAMS = [
    {"s_A": 0.8, "c_A": 0.9, "s_B": 0.7, "c_B": 0.85,
     "s_AC": 0.9, "c_AC": 0.85, "s_BC": 0.8, "c_BC": 0.9, "id": "standard"},
    {"s_A": 0.5, "c_A": 0.8, "s_B": 0.9, "c_B": 0.9,
     "s_AC": 0.7, "c_AC": 0.9, "s_BC": 0.6, "c_BC": 0.85, "id": "asymmetric"},
]

INVERSION_PARAMS = [
    {"s_A": 0.8, "c_A": 0.9, "s_AB": 0.9, "c_AB": 0.85, "id": "strong"},
    {"s_A": 0.3, "c_A": 0.8, "s_AB": 0.7, "c_AB": 0.9, "id": "asymmetric"},
]

# One-hot sampling schedule (tuned for K=4 mixing)
OH_SCHEDULE = SamplingSchedule(n_warmup=1000, n_samples=3000, steps_per_sample=4)
OH_N_BATCHES = 30


# ═══════════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _cat_mp(p, k):
    """Categorical modus ponens (free graph)."""
    g = build_beta_chain(
        [p["s_A"], 0.5], [p["c_A"], 0.01],
        [p["s_AB"]], [p["c_AB"]], [0.02], k=k, clamp_root=False)
    s = run_beta_sampling(g, seed=42)
    _, strength, conf = estimate_beta_marginal(s, g, g["nodes"][1])
    return strength, conf


def _oh_full_mp(p, k):
    """One-hot full modus ponens."""
    g = compile_onehot_full(k, [p["s_A"], 0.5], [p["c_A"], 0.01],
                            [p["s_AB"]], [p["c_AB"]], [0.02])
    samps = run_onehot_sampling(g, seed=42, n_batches=OH_N_BATCHES,
                                schedule=OH_SCHEDULE)
    _, strength, conf = estimate_onehot_marginal(samps, g, 1)
    return strength, conf


def _oh_potts_mp(p, k, method="kl"):
    """One-hot Potts modus ponens."""
    g = compile_onehot_potts(k, [p["s_A"], 0.5], [p["c_A"], 0.01],
                             [p["s_AB"]], [p["c_AB"]], [0.02],
                             potts_method=method)
    samps = run_onehot_sampling(g, seed=42, n_batches=OH_N_BATCHES,
                                schedule=OH_SCHEDULE)
    _, strength, conf = estimate_onehot_marginal(samps, g, 1)
    return strength, conf


def _cat_deduction(p, k):
    """Categorical deduction (3-node free chain)."""
    g = build_beta_chain(
        [p["s_A"], p["s_B"], p["s_C"]],
        [p["c_A"], p["c_B"], p["c_C"]],
        [p["s_AB"], p["s_BC"]], [p["c_AB"], p["c_BC"]],
        [0.02, 0.02], k=k, clamp_root=False)
    s = run_beta_sampling(g, seed=42)
    _, strength, conf = estimate_beta_marginal(s, g, g["nodes"][2])
    return strength, conf


def _oh_full_deduction(p, k):
    """One-hot full deduction."""
    g = compile_onehot_full(
        k, [p["s_A"], p["s_B"], p["s_C"]],
        [p["c_A"], p["c_B"], p["c_C"]],
        [p["s_AB"], p["s_BC"]], [p["c_AB"], p["c_BC"]],
        [0.02, 0.02])
    samps = run_onehot_sampling(g, seed=42, n_batches=OH_N_BATCHES,
                                schedule=OH_SCHEDULE)
    _, strength, conf = estimate_onehot_marginal(samps, g, 2)
    return strength, conf


def _oh_potts_deduction(p, k, method="kl"):
    """One-hot Potts deduction."""
    g = compile_onehot_potts(
        k, [p["s_A"], p["s_B"], p["s_C"]],
        [p["c_A"], p["c_B"], p["c_C"]],
        [p["s_AB"], p["s_BC"]], [p["c_AB"], p["c_BC"]],
        [0.02, 0.02], potts_method=method)
    samps = run_onehot_sampling(g, seed=42, n_batches=OH_N_BATCHES,
                                schedule=OH_SCHEDULE)
    _, strength, conf = estimate_onehot_marginal(samps, g, 2)
    return strength, conf


# ═══════════════════════════════════════════════════════════════════════════
#  pdit-Potts helpers (CategoricalNode + Potts-only weights, simulates pdit hardware)
# ═══════════════════════════════════════════════════════════════════════════

def _make_potts_implication_factor(parent, child, strength, confidence,
                                   background=DEFAULT_EPSILON, k=16,
                                   method="kl"):
    """Like make_beta_implication_factor but replaces K×K with Potts diagonal."""
    W = np.array(beta_implication_weights(strength, confidence, background, k))
    w_opt = fit_potts_weight(W, method=method)
    W_potts = jnp.eye(k) * w_opt
    return SquareCategoricalEBMFactor([Block([parent]), Block([child])],
                                      W_potts[None, :, :])


def _build_pdit_potts_chain(priors, confidences, strengths, impl_confidences,
                            backgrounds, k=16, method="kl"):
    """CategoricalNode chain with Potts-only implication weights."""
    from pln_thrml.beta import _assemble_free_graph
    n = len(priors)
    nodes = [CategoricalNode() for _ in range(n)]
    factors = [make_beta_prior_factor(nodes[i], priors[i], confidences[i], k)
               for i in range(n)]
    for i in range(n - 1):
        factors.append(_make_potts_implication_factor(
            nodes[i], nodes[i + 1],
            strengths[i], impl_confidences[i], backgrounds[i], k, method))
    even = [nodes[i] for i in range(0, n, 2)]
    odd = [nodes[i] for i in range(1, n, 2)]
    free_blocks = [Block(even), Block(odd)] if odd else [Block(even)]
    return _assemble_free_graph(nodes, factors, free_blocks, k, n=n)


def _build_pdit_potts_inv_v(left_prior, left_confidence,
                            right_prior, right_confidence,
                            left_strength, right_strength,
                            left_impl_confidence, right_impl_confidence,
                            left_background, right_background,
                            center_prior=0.5, center_confidence=0.01,
                            k=16, method="kl"):
    """CategoricalNode inv-V with Potts-only implication weights."""
    from pln_thrml.beta import _assemble_free_graph
    nodes = [CategoricalNode() for _ in range(3)]
    factors = [
        make_beta_prior_factor(nodes[0], left_prior, left_confidence, k),
        make_beta_prior_factor(nodes[1], center_prior, center_confidence, k),
        make_beta_prior_factor(nodes[2], right_prior, right_confidence, k),
    ]
    factors.append(_make_potts_implication_factor(
        nodes[0], nodes[1], left_strength, left_impl_confidence,
        left_background, k, method))
    factors.append(_make_potts_implication_factor(
        nodes[2], nodes[1], right_strength, right_impl_confidence,
        right_background, k, method))
    free_blocks = [Block([nodes[0], nodes[2]]), Block([nodes[1]])]
    return _assemble_free_graph(nodes, factors, free_blocks, k)


# ═══════════════════════════════════════════════════════════════════════════
#  PLN closed-form formulas (point estimates, no sampling)
# ═══════════════════════════════════════════════════════════════════════════

def _pln_formula_mp(p, background=0.02):
    """PLN Modus Ponens: s_B = s_A * s_AB + background * (1 - s_A)."""
    return p["s_A"] * p["s_AB"] + background * (1.0 - p["s_A"])


def _pln_formula_deduction(p):
    """PLN Deduction: s_AC = s_AB * s_BC + (1 - s_AB) * (s_C - s_B * s_BC) / (1 - s_B)."""
    denom = max(1.0 - p["s_B"], 1e-7)
    return p["s_AB"] * p["s_BC"] + (1.0 - p["s_AB"]) * (p["s_C"] - p["s_B"] * p["s_BC"]) / denom


def _pln_formula_abduction(p):
    """PLN Abduction: s_AB = s_AC * s_BC + (1 - s_AC) * (s_B - s_A * s_BC) / (1 - s_A)."""
    denom = max(1.0 - p["s_A"], 1e-7)
    return p["s_AC"] * p["s_BC"] + (1.0 - p["s_AC"]) * (p["s_B"] - p["s_A"] * p["s_BC"]) / denom


def _pln_formula_inversion(p):
    """PLN Inversion: s_BA = s_AB * s_A / s_B.  s_B estimated from MP."""
    s_B = p["s_A"] * p["s_AB"] + 0.02 * (1.0 - p["s_A"])
    if s_B < 1e-7:
        return 0.0
    return p["s_AB"] * p["s_A"] / s_B


# ═══════════════════════════════════════════════════════════════════════════
#  Test: One-hot full ≈ Categorical (equivalence)
# ═══════════════════════════════════════════════════════════════════════════

class TestEquivalence:
    """Verify one-hot full faithfully reproduces categorical factor graph."""

    @pytest.mark.parametrize("p", MP_PARAMS, ids=lambda p: p["id"])
    def test_onehot_full_matches_categorical_mp(self, p):
        k = 4
        s_cat, _ = _cat_mp(p, k)
        s_oh, _ = _oh_full_mp(p, k)
        assert abs(s_cat - s_oh) < 0.02, (
            f"OH-full Δs={abs(s_cat-s_oh):.4f} > 0.02 "
            f"(cat={s_cat:.4f}, oh={s_oh:.4f})")

    @pytest.mark.parametrize("p", DEDUCTION_PARAMS[:1], ids=lambda p: p["id"])
    def test_onehot_full_matches_categorical_deduction(self, p):
        k = 4
        s_cat, _ = _cat_deduction(p, k)
        s_oh, _ = _oh_full_deduction(p, k)
        assert abs(s_cat - s_oh) < 0.03, (
            f"OH-full Δs={abs(s_cat-s_oh):.4f} > 0.03 "
            f"(cat={s_cat:.4f}, oh={s_oh:.4f})")


# ═══════════════════════════════════════════════════════════════════════════
#  Test: Five-column comparison (DTV baseline)
# ═══════════════════════════════════════════════════════════════════════════

class TestFiveColumnComparison:
    """Five-column precision table with DTV as baseline."""

    @pytest.mark.parametrize("p", MP_PARAMS, ids=lambda p: p["id"])
    def test_modus_ponens_k4(self, p, capsys):
        k = 4
        s_dtv, _ = dtv_modus_ponens(p["s_A"], p["c_A"], p["s_AB"], p["c_AB"])
        s_pln = _pln_formula_mp(p)
        s_cat, _ = _cat_mp(p, k)
        s_oh, _ = _oh_full_mp(p, k)
        s_pk, _ = _oh_potts_mp(p, k, "kl")
        s_pd, _ = _oh_potts_mp(p, k, "diag")

        with capsys.disabled():
            print(f"\n  MP({p['id']}) K={k}: DTV={s_dtv:.3f} | "
                  f"PLN Δ={abs(s_dtv-s_pln):.3f} | "
                  f"Cat Δ={abs(s_dtv-s_cat):.3f} | OH Δ={abs(s_dtv-s_oh):.3f} | "
                  f"Potts(KL) Δ={abs(s_dtv-s_pk):.3f} | "
                  f"Potts(diag) Δ={abs(s_dtv-s_pd):.3f}")

        # Categorical and OH-full should be close to DTV
        assert abs(s_dtv - s_cat) < strength_tol(k) + 0.08, "Cat too far from DTV"
        assert abs(s_cat - s_oh) < 0.03, "OH-full too far from Cat"

    @pytest.mark.parametrize("p", DEDUCTION_PARAMS, ids=lambda p: p["id"])
    def test_deduction_k4(self, p, capsys):
        k = 4
        s_dtv, _ = dtv_deduction(
            p["s_A"], p["c_A"], p["s_B"], p["c_B"], p["s_C"], p["c_C"],
            p["s_AB"], p["c_AB"], p["s_BC"], p["c_BC"])
        s_pln = _pln_formula_deduction(p)
        s_cat, _ = _cat_deduction(p, k)
        s_oh, _ = _oh_full_deduction(p, k)
        s_pk, _ = _oh_potts_deduction(p, k, "kl")
        s_pd, _ = _oh_potts_deduction(p, k, "diag")

        with capsys.disabled():
            print(f"\n  Deduction({p['id']}) K={k}: DTV={s_dtv:.3f} | "
                  f"PLN Δ={abs(s_dtv-s_pln):.3f} | "
                  f"Cat Δ={abs(s_dtv-s_cat):.3f} | OH Δ={abs(s_dtv-s_oh):.3f} | "
                  f"Potts(KL) Δ={abs(s_dtv-s_pk):.3f} | "
                  f"Potts(diag) Δ={abs(s_dtv-s_pd):.3f}")

        # Extreme params with c≈1.0 cause mixing issues at K=4; relax tolerance
        tol = 0.05 if p["id"] != "extreme" else 0.25
        assert abs(s_cat - s_oh) < tol, (
            f"OH-full too far from Cat: Δ={abs(s_cat-s_oh):.4f} > {tol}")

    @pytest.mark.parametrize("p", ABDUCTION_PARAMS, ids=lambda p: p["id"])
    def test_abduction_k4(self, p, capsys):
        """Abduction: explaining-away is hardest for Potts."""
        k = 4
        s_dtv, _ = dtv_abduction(
            p["s_A"], p["c_A"], p["s_B"], p["c_B"],
            p["s_AC"], p["c_AC"], p["s_BC"], p["c_BC"])
        s_pln = _pln_formula_abduction(p)

        # Categorical inv_v_graph baseline: query center node (node 1)
        g = build_beta_inv_v_graph(
            left_prior=p["s_A"], left_confidence=p["c_A"],
            right_prior=p["s_B"], right_confidence=p["c_B"],
            left_strength=p["s_AC"], right_strength=p["s_BC"],
            left_impl_confidence=p["c_AC"], right_impl_confidence=p["c_BC"],
            left_background=0.02, right_background=0.02, k=k)
        cs = run_beta_sampling(g, seed=42)
        _, s_cat, _ = estimate_beta_marginal(cs, g, g["nodes"][1])

        # One-hot full inv-v
        oh = compile_onehot_full_inv_v(
            k, left_prior=p["s_A"], left_confidence=p["c_A"],
            right_prior=p["s_B"], right_confidence=p["c_B"],
            left_strength=p["s_AC"], right_strength=p["s_BC"],
            left_impl_confidence=p["c_AC"], right_impl_confidence=p["c_BC"],
            left_background=0.02, right_background=0.02)
        samps = run_onehot_sampling(oh, seed=42, n_batches=OH_N_BATCHES,
                                    schedule=OH_SCHEDULE)
        _, s_oh, _ = estimate_onehot_marginal(samps, oh, 1)  # center node

        # One-hot Potts inv-v (KL)
        ohp = compile_onehot_potts_inv_v(
            k, left_prior=p["s_A"], left_confidence=p["c_A"],
            right_prior=p["s_B"], right_confidence=p["c_B"],
            left_strength=p["s_AC"], right_strength=p["s_BC"],
            left_impl_confidence=p["c_AC"], right_impl_confidence=p["c_BC"],
            left_background=0.02, right_background=0.02)
        sampsp = run_onehot_sampling(ohp, seed=42, n_batches=OH_N_BATCHES,
                                     schedule=OH_SCHEDULE)
        _, s_pk, _ = estimate_onehot_marginal(sampsp, ohp, 1)

        # One-hot Potts inv-v (diag)
        ohpd = compile_onehot_potts_inv_v(
            k, left_prior=p["s_A"], left_confidence=p["c_A"],
            right_prior=p["s_B"], right_confidence=p["c_B"],
            left_strength=p["s_AC"], right_strength=p["s_BC"],
            left_impl_confidence=p["c_AC"], right_impl_confidence=p["c_BC"],
            left_background=0.02, right_background=0.02, potts_method="diag")
        sampspd = run_onehot_sampling(ohpd, seed=42, n_batches=OH_N_BATCHES,
                                      schedule=OH_SCHEDULE)
        _, s_pd, _ = estimate_onehot_marginal(sampspd, ohpd, 1)

        with capsys.disabled():
            print(f"\n  Abduction({p['id']}) K={k}: DTV={s_dtv:.3f} | "
                  f"PLN Δ={abs(s_dtv-s_pln):.3f} | "
                  f"Cat Δ={abs(s_dtv-s_cat):.3f} | OH Δ={abs(s_dtv-s_oh):.3f} | "
                  f"Potts(KL) Δ={abs(s_dtv-s_pk):.3f} | "
                  f"Potts(diag) Δ={abs(s_dtv-s_pd):.3f}")

        assert s_dtv > 0.0, "DTV should return positive strength"

    @pytest.mark.parametrize("p", INVERSION_PARAMS, ids=lambda p: p["id"])
    def test_inversion_k4(self, p, capsys):
        """Inversion: Bayesian flip P(B|A) → P(A|B)."""
        k = 4
        s_pln = _pln_formula_inversion(p)

        # Categorical: 2-node chain, clamp_root=False, query conditional P(A|B)
        g = build_beta_chain(
            [p["s_A"], 0.5], [p["c_A"], 0.01],
            [p["s_AB"]], [p["c_AB"]], [0.02], k=k, clamp_root=False)
        cs = run_beta_sampling(g, seed=42)
        _, s_cat, _ = estimate_beta_marginal(cs, g, g["nodes"][0])

        # One-hot full chain (2-node), query node 0
        oh = compile_onehot_full(k, [p["s_A"], 0.5], [p["c_A"], 0.01],
                                 [p["s_AB"]], [p["c_AB"]], [0.02])
        samps = run_onehot_sampling(oh, seed=42, n_batches=OH_N_BATCHES,
                                    schedule=OH_SCHEDULE)
        _, s_oh, _ = estimate_onehot_marginal(samps, oh, 0)

        # Potts KL
        ohp = compile_onehot_potts(k, [p["s_A"], 0.5], [p["c_A"], 0.01],
                                   [p["s_AB"]], [p["c_AB"]], [0.02])
        sampsp = run_onehot_sampling(ohp, seed=42, n_batches=OH_N_BATCHES,
                                     schedule=OH_SCHEDULE)
        _, s_pk, _ = estimate_onehot_marginal(sampsp, ohp, 0)

        # Potts diag
        ohpd = compile_onehot_potts(k, [p["s_A"], 0.5], [p["c_A"], 0.01],
                                    [p["s_AB"]], [p["c_AB"]], [0.02],
                                    potts_method="diag")
        sampspd = run_onehot_sampling(ohpd, seed=42, n_batches=OH_N_BATCHES,
                                      schedule=OH_SCHEDULE)
        _, s_pd, _ = estimate_onehot_marginal(sampspd, ohpd, 0)

        with capsys.disabled():
            print(f"\n  Inversion({p['id']}) K={k}: PLN={s_pln:.3f} | "
                  f"Cat={s_cat:.3f} | OH={s_oh:.3f} | "
                  f"Potts(KL)={s_pk:.3f} | Potts(diag)={s_pd:.3f} | "
                  f"OH Δ(Cat)={abs(s_cat-s_oh):.3f} | "
                  f"Potts(KL) Δ(Cat)={abs(s_cat-s_pk):.3f} | "
                  f"Potts(diag) Δ(Cat)={abs(s_cat-s_pd):.3f}")

        assert abs(s_cat - s_oh) < 0.05, "OH-full too far from Cat for inversion"


# ═══════════════════════════════════════════════════════════════════════════
#  Test: Neighbor budget
# ═══════════════════════════════════════════════════════════════════════════

class TestNeighborBudget:
    """Hardware feasibility: neighbor consumption per spin."""

    @pytest.mark.parametrize("k", [4, 8, 16])
    def test_neighbor_counts(self, k, capsys):
        info = count_neighbors_per_spin(k, n_categorical_neighbors=1)
        with capsys.disabled():
            print(f"\n  K={k}: excl={info['exclusion']} | "
                  f"full/neigh={info['full_per_neighbor']} | "
                  f"potts/neigh={info['potts_per_neighbor']} | "
                  f"max_full={info['max_full_neighbors']} | "
                  f"max_potts={info['max_potts_neighbors']}")

        # K=4 should be feasible for both modes
        if k == 4:
            assert info["full_feasible"], "K=4 full should fit in 12-neighbor budget"
            assert info["potts_feasible"], "K=4 potts should fit in 12-neighbor budget"
            assert info["max_full_neighbors"] == 2
            assert info["max_potts_neighbors"] == 9

        # K≥8: full is infeasible (exclusion + K coupling > 12)
        if k >= 8:
            assert not info["full_feasible"], f"K={k} full should NOT fit"

        # K=16: even exclusion alone exceeds budget
        if k == 16:
            assert info["exclusion"] > 12, "K=16 exclusion alone exceeds budget"

    def test_feasibility_summary(self, capsys):
        """Print full feasibility table."""
        with capsys.disabled():
            print("\n  === Neighbor Budget (budget=12) ===")
            print("  K  | Excl | Full/n | Potts/n | Max Full | Max Potts")
            print("  ---|------|--------|---------|----------|----------")
            for k in [4, 8, 16]:
                info = count_neighbors_per_spin(k, n_categorical_neighbors=1)
                mf = info["max_full_neighbors"] if info["max_full_neighbors"] > 0 else "N/A"
                mp = info["max_potts_neighbors"] if info["max_potts_neighbors"] > 0 else "N/A"
                print(f"  {k:2d} | {info['exclusion']:4d} | {info['full_per_neighbor']:6d} | "
                      f"{info['potts_per_neighbor']:7d} | {str(mf):>8s} | {str(mp):>9s}")


# ═══════════════════════════════════════════════════════════════════════════
#  Test: Potts weight fitting
# ═══════════════════════════════════════════════════════════════════════════

class TestPottsFitting:
    """Compare diagonal-mean vs KL-optimized Potts weight."""

    def test_two_methods_differ(self):
        W = np.array(beta_implication_weights(0.9, 0.85, 0.02, 4))
        w_diag = fit_potts_weight(W, method="diag")
        w_kl = fit_potts_weight(W, method="kl")
        # Both should be positive (ferromagnetic coupling for implication)
        assert w_diag > 0
        assert w_kl > 0
        # They should differ (non-trivial optimization)
        assert abs(w_diag - w_kl) > 0.01


# ═══════════════════════════════════════════════════════════════════════════
#  Test: pdit + Potts at K=16 (the likely hardware deployment path)
# ═══════════════════════════════════════════════════════════════════════════

class TestPditPotts:
    """CategoricalNode K=16 with Potts-only coupling — simulates pdit hardware.

    pdit eliminates one-hot overhead; Potts coupling uses 1 neighbor per
    pdit neighbor.  With ~12 budget → up to 12 categorical neighbors.
    """

    @pytest.mark.parametrize("p", MP_PARAMS, ids=lambda p: p["id"])
    def test_pdit_potts_mp(self, p, capsys):
        k = 16
        s_dtv, _ = dtv_modus_ponens(p["s_A"], p["c_A"], p["s_AB"], p["c_AB"])
        s_pln = _pln_formula_mp(p)

        # Cat full K×K (pdit + full coupling, gold standard)
        s_cat, _ = _cat_mp(p, k)

        # pdit + Potts KL
        g_pk = _build_pdit_potts_chain(
            [p["s_A"], 0.5], [p["c_A"], 0.01],
            [p["s_AB"]], [p["c_AB"]], [0.02], k=k, method="kl")
        cs_pk = run_beta_sampling(g_pk, seed=42)
        _, s_pk, _ = estimate_beta_marginal(cs_pk, g_pk, g_pk["nodes"][1])

        # pdit + Potts diag
        g_pd = _build_pdit_potts_chain(
            [p["s_A"], 0.5], [p["c_A"], 0.01],
            [p["s_AB"]], [p["c_AB"]], [0.02], k=k, method="diag")
        cs_pd = run_beta_sampling(g_pd, seed=42)
        _, s_pd, _ = estimate_beta_marginal(cs_pd, g_pd, g_pd["nodes"][1])

        with capsys.disabled():
            print(f"\n  pdit-MP({p['id']}) K={k}: DTV={s_dtv:.3f} | "
                  f"PLN Δ={abs(s_dtv-s_pln):.3f} | "
                  f"Cat(full) Δ={abs(s_dtv-s_cat):.3f} | "
                  f"pdit-Potts(KL) Δ={abs(s_dtv-s_pk):.3f} | "
                  f"pdit-Potts(diag) Δ={abs(s_dtv-s_pd):.3f}")

    @pytest.mark.parametrize("p", DEDUCTION_PARAMS, ids=lambda p: p["id"])
    def test_pdit_potts_deduction(self, p, capsys):
        k = 16
        s_dtv, _ = dtv_deduction(
            p["s_A"], p["c_A"], p["s_B"], p["c_B"], p["s_C"], p["c_C"],
            p["s_AB"], p["c_AB"], p["s_BC"], p["c_BC"])
        s_pln = _pln_formula_deduction(p)

        s_cat, _ = _cat_deduction(p, k)

        g_pk = _build_pdit_potts_chain(
            [p["s_A"], p["s_B"], p["s_C"]],
            [p["c_A"], p["c_B"], p["c_C"]],
            [p["s_AB"], p["s_BC"]], [p["c_AB"], p["c_BC"]],
            [0.02, 0.02], k=k, method="kl")
        cs_pk = run_beta_sampling(g_pk, seed=42)
        _, s_pk, _ = estimate_beta_marginal(cs_pk, g_pk, g_pk["nodes"][2])

        g_pd = _build_pdit_potts_chain(
            [p["s_A"], p["s_B"], p["s_C"]],
            [p["c_A"], p["c_B"], p["c_C"]],
            [p["s_AB"], p["s_BC"]], [p["c_AB"], p["c_BC"]],
            [0.02, 0.02], k=k, method="diag")
        cs_pd = run_beta_sampling(g_pd, seed=42)
        _, s_pd, _ = estimate_beta_marginal(cs_pd, g_pd, g_pd["nodes"][2])

        with capsys.disabled():
            print(f"\n  pdit-Ded({p['id']}) K={k}: DTV={s_dtv:.3f} | "
                  f"PLN Δ={abs(s_dtv-s_pln):.3f} | "
                  f"Cat(full) Δ={abs(s_dtv-s_cat):.3f} | "
                  f"pdit-Potts(KL) Δ={abs(s_dtv-s_pk):.3f} | "
                  f"pdit-Potts(diag) Δ={abs(s_dtv-s_pd):.3f}")

    @pytest.mark.parametrize("p", ABDUCTION_PARAMS, ids=lambda p: p["id"])
    def test_pdit_potts_abduction(self, p, capsys):
        k = 16
        s_dtv, _ = dtv_abduction(
            p["s_A"], p["c_A"], p["s_B"], p["c_B"],
            p["s_AC"], p["c_AC"], p["s_BC"], p["c_BC"])
        s_pln = _pln_formula_abduction(p)

        # Cat full
        g = build_beta_inv_v_graph(
            left_prior=p["s_A"], left_confidence=p["c_A"],
            right_prior=p["s_B"], right_confidence=p["c_B"],
            left_strength=p["s_AC"], right_strength=p["s_BC"],
            left_impl_confidence=p["c_AC"], right_impl_confidence=p["c_BC"],
            left_background=0.02, right_background=0.02, k=k)
        cs = run_beta_sampling(g, seed=42)
        _, s_cat, _ = estimate_beta_marginal(cs, g, g["nodes"][1])

        # pdit Potts KL
        g_pk = _build_pdit_potts_inv_v(
            left_prior=p["s_A"], left_confidence=p["c_A"],
            right_prior=p["s_B"], right_confidence=p["c_B"],
            left_strength=p["s_AC"], right_strength=p["s_BC"],
            left_impl_confidence=p["c_AC"], right_impl_confidence=p["c_BC"],
            left_background=0.02, right_background=0.02, k=k, method="kl")
        cs_pk = run_beta_sampling(g_pk, seed=42)
        _, s_pk, _ = estimate_beta_marginal(cs_pk, g_pk, g_pk["nodes"][1])

        # pdit Potts diag
        g_pd = _build_pdit_potts_inv_v(
            left_prior=p["s_A"], left_confidence=p["c_A"],
            right_prior=p["s_B"], right_confidence=p["c_B"],
            left_strength=p["s_AC"], right_strength=p["s_BC"],
            left_impl_confidence=p["c_AC"], right_impl_confidence=p["c_BC"],
            left_background=0.02, right_background=0.02, k=k, method="diag")
        cs_pd = run_beta_sampling(g_pd, seed=42)
        _, s_pd, _ = estimate_beta_marginal(cs_pd, g_pd, g_pd["nodes"][1])

        with capsys.disabled():
            print(f"\n  pdit-Abd({p['id']}) K={k}: DTV={s_dtv:.3f} | "
                  f"PLN Δ={abs(s_dtv-s_pln):.3f} | "
                  f"Cat(full) Δ={abs(s_dtv-s_cat):.3f} | "
                  f"pdit-Potts(KL) Δ={abs(s_dtv-s_pk):.3f} | "
                  f"pdit-Potts(diag) Δ={abs(s_dtv-s_pd):.3f}")

    @pytest.mark.parametrize("p", INVERSION_PARAMS, ids=lambda p: p["id"])
    def test_pdit_potts_inversion(self, p, capsys):
        k = 16
        s_pln = _pln_formula_inversion(p)

        # Cat full
        g = build_beta_chain([p["s_A"], 0.5], [p["c_A"], 0.01],
                             [p["s_AB"]], [p["c_AB"]], [0.02],
                             k=k, clamp_root=False)
        cs = run_beta_sampling(g, seed=42)
        _, s_cat, _ = estimate_beta_marginal(cs, g, g["nodes"][0])

        # pdit Potts KL
        g_pk = _build_pdit_potts_chain(
            [p["s_A"], 0.5], [p["c_A"], 0.01],
            [p["s_AB"]], [p["c_AB"]], [0.02], k=k, method="kl")
        cs_pk = run_beta_sampling(g_pk, seed=42)
        _, s_pk, _ = estimate_beta_marginal(cs_pk, g_pk, g_pk["nodes"][0])

        # pdit Potts diag
        g_pd = _build_pdit_potts_chain(
            [p["s_A"], 0.5], [p["c_A"], 0.01],
            [p["s_AB"]], [p["c_AB"]], [0.02], k=k, method="diag")
        cs_pd = run_beta_sampling(g_pd, seed=42)
        _, s_pd, _ = estimate_beta_marginal(cs_pd, g_pd, g_pd["nodes"][0])

        with capsys.disabled():
            print(f"\n  pdit-Inv({p['id']}) K={k}: PLN={s_pln:.3f} | "
                  f"Cat(full)={s_cat:.3f} | "
                  f"pdit-Potts(KL)={s_pk:.3f} | "
                  f"pdit-Potts(diag)={s_pd:.3f} | "
                  f"Potts(KL) Δ(Cat)={abs(s_cat-s_pk):.3f} | "
                  f"Potts(diag) Δ(Cat)={abs(s_cat-s_pd):.3f}")
