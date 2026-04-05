; dispatch.metta — MeTTa dispatch rules for thrml inference operator
; ==================================================================
; Pattern-matching dispatch aligned with upstream PLN |- operator
; (vendor/PLN/lib_pln.metta:210-315).
;
; Each rule destructures (stv $s $c) in the pattern and passes raw
; numbers to grounded Python ops. The conclusion structure is
; assembled here in MeTTa.

; ── Guards ────────────────────────────────────────────────────────

(= (SyllogisticRuleGuard Implication) True)
(= (SyllogisticRuleGuard Inheritance) True)

(= (SymmetricLinkGuard Similarity) True)
(= (SymmetricLinkGuard IntentionalSimilarity) True)
(= (SymmetricLinkGuard ExtensionalSimilarity) True)

; ── Single-premise rules ──────────────────────────────────────────

; Negation: (Not A) → A with (1-s, c) — pure arithmetic, no sampling
(= (thrml ((Not $A) (stv $s $c)))
   ($A (stv (- 1.0 $s) $c)))

; Equivalence → Implication
(= (thrml ((Equivalence $A $B) (stv $s $c)))
   ((Implication $A $B) (thrml-equiv-to-impl $A $B $s $c)))

; Inversion (syllogistic links only)
(= (thrml (($LinkType $A $B) (stv $s $c)))
   (if (SyllogisticRuleGuard $LinkType)
       (($LinkType $B $A) (thrml-inversion $A $B $s $c))
       (empty)))

; ── Two-premise rules ─────────────────────────────────────────────

; Revision: same subject, different truth values
(= (thrml ($T (stv $s1 $c1)) ($T (stv $s2 $c2)))
   ($T (thrml-revision $s1 $c1 $s2 $c2)))

; Modus Ponens (syllogistic links)
(= (thrml ($A (stv $s1 $c1)) (($LinkType $A $B) (stv $s2 $c2)))
   (if (SyllogisticRuleGuard $LinkType)
       ($B (thrml-modus-ponens $A $B $s1 $c1 $s2 $c2))
       (empty)))

; Symmetric Modus Ponens
(= (thrml ($A (stv $s1 $c1)) (($LinkType $A $B) (stv $s2 $c2)))
   (if (SymmetricLinkGuard $LinkType)
       ($B (thrml-symmetric-mp $A $B $s1 $c1 $s2 $c2))
       (empty)))

; Deduction: A→B, B→C ⊢ A→C
(= (thrml (($LinkType $A $B) (stv $s1 $c1)) (($LinkType $B $C) (stv $s2 $c2)))
   (if (SyllogisticRuleGuard $LinkType)
       (($LinkType $A $C) (thrml-deduction $A $B $C $s1 $c1 $s2 $c2))
       (empty)))

; Transitive Similarity: A~B, B~C ⊢ A~C
(= (thrml (($LinkType $A $B) (stv $s1 $c1)) (($LinkType $B $C) (stv $s2 $c2)))
   (if (SymmetricLinkGuard $LinkType)
       (($LinkType $A $C) (thrml-transitive-sim $A $B $C $s1 $c1 $s2 $c2))
       (empty)))

; Induction: C→A, C→B ⊢ A→B
(= (thrml (($LinkType $C $A) (stv $s1 $c1)) (($LinkType $C $B) (stv $s2 $c2)))
   (if (SyllogisticRuleGuard $LinkType)
       (($LinkType $A $B) (thrml-induction $A $B $C $s1 $c1 $s2 $c2))
       (empty)))

; Abduction: A→C, B→C ⊢ A→B
(= (thrml (($LinkType $A $C) (stv $s1 $c1)) (($LinkType $B $C) (stv $s2 $c2)))
   (if (SyllogisticRuleGuard $LinkType)
       (($LinkType $A $B) (thrml-abduction $A $B $C $s1 $c1 $s2 $c2))
       (empty)))

; Evaluation Implication: Eval(A,B), A→C ⊢ Eval(C,B)
(= (thrml ((Evaluation $A $B) (stv $s1 $c1)) (($LinkType $A $C) (stv $s2 $c2)))
   (if (SyllogisticRuleGuard $LinkType)
       ((Evaluation $C $B) (thrml-eval-impl $A $B $C $s1 $c1 $s2 $c2))
       (empty)))
