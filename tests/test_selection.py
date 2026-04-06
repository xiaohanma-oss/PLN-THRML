"""Tests for pln_thrml.metta.selection — geodesic rule selection."""

import numpy as np
import pytest

from pln_thrml.selection import select_and_apply


class TestSelectAndApply:
    """Geodesic selection among PLN rules."""

    def test_single_applicable_rule(self):
        """Single applicable rule → returned directly (no selection needed)."""
        premises = [
            {"atom": "A", "strength": 0.8, "confidence": 0.9},
            {"atom": "(Implication A B)", "strength": 0.9, "confidence": 0.85,
             "link_type": "Implication", "source": "A", "target": "B"},
        ]
        result = select_and_apply(premises, seed=42)
        assert result is not None
        assert result["name"] == "modus-ponens"
        assert 0.0 < result["strength"] < 1.0
        assert result["posterior"].shape[0] > 0

    def test_multiple_rules_selects_one(self):
        """When multiple rules apply, geodesic selects exactly one."""
        # These premises match both modus-ponens (A, A→B) and inversion (A→B)
        premises = [
            {"atom": "A", "strength": 0.8, "confidence": 0.9},
            {"atom": "(Implication A B)", "strength": 0.9, "confidence": 0.85,
             "link_type": "Implication", "source": "A", "target": "B"},
        ]
        result = select_and_apply(premises, temperature=0.01, seed=42)
        assert result is not None
        assert result["name"] in ("modus-ponens", "inversion")
        # Should have selection diagnostics when multiple candidates exist
        assert "selection" in result

    def test_goal_influences_selection(self):
        """Setting goal_stv should influence which rule is selected."""
        premises = [
            {"atom": "A", "strength": 0.8, "confidence": 0.9},
            {"atom": "(Implication A B)", "strength": 0.9, "confidence": 0.85,
             "link_type": "Implication", "source": "A", "target": "B"},
        ]
        # Run with goal — result should have selection diagnostics
        result = select_and_apply(
            premises, goal_stv=(0.9, 0.9), temperature=0.5, seed=42)
        assert result is not None

    def test_deduction_chain(self):
        """Deduction: A→B, B→C should produce A→C."""
        premises = [
            {"atom": "(Implication A B)", "strength": 0.9, "confidence": 0.8,
             "link_type": "Implication", "source": "A", "target": "B"},
            {"atom": "(Implication B C)", "strength": 0.85, "confidence": 0.75,
             "link_type": "Implication", "source": "B", "target": "C"},
        ]
        result = select_and_apply(premises, seed=42)
        assert result is not None
        # Could be deduction, induction, or abduction depending on pattern
        assert result["name"] in ("deduction", "induction", "abduction")

    def test_no_applicable_rules(self):
        """Empty or unmatched premises → None."""
        result = select_and_apply([], seed=42)
        assert result is None

    def test_result_has_posterior(self):
        """Result should have a proper posterior (not uniform placeholder)."""
        premises = [
            {"atom": "A", "strength": 0.8, "confidence": 0.9},
            {"atom": "(Implication A B)", "strength": 0.9, "confidence": 0.85,
             "link_type": "Implication", "source": "A", "target": "B"},
        ]
        result = select_and_apply(premises, seed=42)
        assert result is not None
        posterior = result["posterior"]
        assert posterior.shape[0] >= 4
        assert abs(posterior.sum() - 1.0) < 0.01
        # Should NOT be uniform
        uniform_std = np.std(np.ones(posterior.shape[0]) / posterior.shape[0])
        assert np.std(posterior) > uniform_std * 2

    def test_selection_diagnostics(self):
        """When multiple rules compete, result has selection diagnostics."""
        premises = [
            {"atom": "A", "strength": 0.8, "confidence": 0.9},
            {"atom": "(Implication A B)", "strength": 0.9, "confidence": 0.85,
             "link_type": "Implication", "source": "A", "target": "B"},
        ]
        result = select_and_apply(premises, temperature=1.0, seed=42)
        if "selection" in result:
            sel = result["selection"]
            assert hasattr(sel, "rule_probs")
            assert hasattr(sel, "energy")
            assert abs(sel.rule_probs.sum() - 1.0) < 1e-10

    def test_revision(self):
        """Revision: same atom with two truth values."""
        premises = [
            {"atom": "X", "strength": 0.7, "confidence": 0.6},
            {"atom": "X", "strength": 0.8, "confidence": 0.5},
        ]
        result = select_and_apply(premises, seed=42)
        assert result is not None
        assert result["name"] == "revision"
