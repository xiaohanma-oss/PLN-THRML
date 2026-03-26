"""
metta — MeTTa thin layer for PLN thermodynamic inference
=========================================================

Registers grounded operations that bridge MeTTa PLN atoms to
pln_thrml_beta's thrml factor graph engine.

Usage:
    from hyperon import MeTTa
    from metta import register_all

    metta = MeTTa()
    register_all(metta)

    metta.run('''
        (A (stv 0.8 0.9))
        ((Implication A B) (stv 0.9 0.85))
        !(thrml-modus-ponens! (A B))
    ''')

Install: pip install pln-thrml[metta]
"""

from metta.ops import register_all

__all__ = ["register_all"]
