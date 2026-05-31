"""Reproduce Table 2 of paper-a (AGI-26 #92, "Compiling PLN to
Thermodynamic Hardware").

Runs the four hardware-bound PLN rules (Modus Ponens, Deduction,
Abduction, Inversion) on the parameter sets that produced Table 2,
prints a CSV-style table of (PLN s | ours s | abs delta), and exits
non-zero if any row exceeds the paper's tolerance (|Δs| > 0.05).

Anchor: tag `v0.2.0-agi26` reproduces the paper's exact numbers
(deterministic given thrml==0.1.3 + jax==0.5.3 with seed=42).

Parameter sets are the same ones used by tests/test_hybrid.py.
Run with: python scripts/reproduce_paper_tables.py
"""

from __future__ import annotations
import sys

from pln_thrml.hybrid import (
    hybrid_modus_ponens, hybrid_deduction,
    hybrid_abduction, hybrid_inversion,
)

# ───────────────────── Parameter sets (Table 2 inputs) ─────────────────────

MP_PARAMS = [
    {"s_A": 0.8, "c_A": 0.9,  "s_AB": 0.9, "c_AB": 0.85, "id": "strong"},
    {"s_A": 0.6, "c_A": 0.7,  "s_AB": 0.7, "c_AB": 0.80, "id": "medium"},
    {"s_A": 0.3, "c_A": 0.95, "s_AB": 0.5, "c_AB": 0.90, "id": "rare"},
]
DEDUCTION_PARAMS = [
    {"s_A": 0.8,  "c_A": 0.9,    "s_B": 0.5, "c_B": 0.01,  "s_C": 0.5, "c_C": 0.01,
     "s_AB": 0.9, "c_AB": 0.85,  "s_BC": 0.8, "c_BC": 0.9, "id": "standard"},
    {"s_A": 0.05, "c_A": 0.8,    "s_B": 0.2, "c_B": 0.9999, "s_C": 1.0, "c_C": 0.8,
     "s_AB": 1.0, "c_AB": 0.9999, "s_BC": 0.3, "c_BC": 0.8, "id": "extreme"},
]
ABDUCTION_PARAMS = [
    {"s_A": 0.8, "c_A": 0.9, "s_B": 0.7, "c_B": 0.85,
     "s_AC": 0.9, "c_AC": 0.85, "s_BC": 0.8, "c_BC": 0.9, "id": "standard"},
    {"s_A": 0.5, "c_A": 0.8, "s_B": 0.9, "c_B": 0.9,
     "s_AC": 0.7, "c_AC": 0.9, "s_BC": 0.6, "c_BC": 0.85, "id": "asymmetric"},
]
INVERSION_PARAMS = [
    {"s_A": 0.7, "c_A": 0.9, "s_B": 0.7, "c_B": 0.9,
     "s_AB": 0.8, "c_AB": 0.85, "id": "symmetric"},
    {"s_A": 0.8, "c_A": 0.9, "s_B": 0.3, "c_B": 0.85,
     "s_AB": 0.9, "c_AB": 0.9, "id": "asymmetric"},
]

# ───────────────────── PLN closed-form ground truth ─────────────────────

def pln_mp(p, background=0.02):
    return p["s_A"] * p["s_AB"] + background * (1.0 - p["s_A"])

def pln_deduction(p):
    denom = max(1.0 - p["s_B"], 1e-7)
    return (p["s_AB"] * p["s_BC"]
            + (1.0 - p["s_AB"]) * (p["s_C"] - p["s_B"] * p["s_BC"]) / denom)

def pln_abduction(p, background=0.02):
    s_C = p["s_BC"] * p["s_B"] + background * (1.0 - p["s_B"])
    s_C = max(min(s_C, 1.0 - 1e-7), 1e-7)
    return (p["s_AC"] * p["s_BC"] * p["s_B"] / s_C
            + (1.0 - p["s_AC"]) * (1.0 - p["s_BC"]) * p["s_B"] / (1.0 - s_C))

def pln_inversion_bayes(p):
    """PLN Inversion (Bayes-form): s_BA = s_AB × s_A / s_B."""
    s_B = max(p["s_B"], 1e-7)
    return min(p["s_AB"] * p["s_A"] / s_B, 1.0)

def pln_inversion_heuristic(p):
    """PLN Inversion (PLN-form heuristic per paper §4): s_inv = s_AB."""
    return p["s_AB"]

# ───────────────────── Compiler runs (ours) ─────────────────────

def ours_mp(p):
    s, _ = hybrid_modus_ponens(p["s_A"], p["c_A"], p["s_AB"], p["c_AB"])
    return s

def ours_deduction(p):
    s, _ = hybrid_deduction(
        p["s_A"], p["c_A"], p["s_B"], p["c_B"], p["s_C"], p["c_C"],
        p["s_AB"], p["c_AB"], p["s_BC"], p["c_BC"])
    return s

def ours_abduction(p):
    s, _ = hybrid_abduction(
        p["s_A"], p["c_A"], p["s_B"], p["c_B"],
        p["s_AC"], p["c_AC"], p["s_BC"], p["c_BC"])
    return s

def ours_inversion(p):
    s, _ = hybrid_inversion(
        p["s_A"], p["c_A"], p["s_B"], p["c_B"],
        p["s_AB"], p["c_AB"])
    return s

# ───────────────────── Driver ─────────────────────

TOL = 0.05
ROWS = []

def add(rule, pln_fn, ours_fn, p):
    pln_s = pln_fn(p)
    ours_s = ours_fn(p)
    delta = abs(pln_s - ours_s)
    ROWS.append((rule, p["id"], pln_s, ours_s, delta))

def main():
    print(f"Reproducing paper-a Table 2 (tolerance |Δs| ≤ {TOL})\n")
    print("Rule,id,PLN_s,ours_s,abs_delta,within_tol")
    for p in MP_PARAMS:
        add("Modus Ponens", pln_mp, ours_mp, p)
    for p in DEDUCTION_PARAMS:
        add("Deduction", pln_deduction, ours_deduction, p)
    for p in ABDUCTION_PARAMS:
        add("Abduction", pln_abduction, ours_abduction, p)
    # Paper Table 2 reports two Inversion rows: PLN-form (heuristic
    # s_inv=s_AB) on the symmetric params, Bayes-form (s_AB·s_A/s_B
    # clamped) on the asymmetric params.
    add("Inversion (PLN-form)",   pln_inversion_heuristic, ours_inversion, INVERSION_PARAMS[0])
    add("Inversion (Bayes-form)", pln_inversion_bayes,     ours_inversion, INVERSION_PARAMS[1])

    failed = 0
    for rule, rid, pln_s, ours_s, d in ROWS:
        ok = d <= TOL
        if not ok: failed += 1
        print(f"{rule},{rid},{pln_s:.3f},{ours_s:.3f},{d:.3f},{ok}")

    print(f"\nSummary: {len(ROWS) - failed}/{len(ROWS)} rows within tolerance.")
    if failed:
        print(f"FAIL: {failed} row(s) exceeded |Δs|={TOL}", file=sys.stderr)
        sys.exit(1)
    print("PASS: all rows within paper tolerance.")
    sys.exit(0)

if __name__ == "__main__":
    main()
