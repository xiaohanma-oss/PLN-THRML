; pln_rules.metta — PLN inference rules via thrml thermodynamic sampling
; ======================================================================
; Each rule wraps a grounded operation (thrml-*!) that compiles the PLN
; inference into a thrml factor graph and returns (stv s c).
;
; Aligned with upstream lib_pln.metta |- operator conventions.
; Truth values $T1/$T2 from the inference chain are forwarded to grounded
; ops so that intermediate results propagate through multi-step reasoning.

; Guards are defined in pln_types.metta (SyllogisticRuleGuard, SymmetricModusPonensRuleGuard)

; ── Revision (upstream: |- same term, two STVs) ─────────────────────
; Two evidence sources for same proposition
(= (|-thrml ($T $T1)
            ($T $T2))
   ($T (thrml-revision! ($T $T1 $T2))))

; ── Modus Ponens ─────────────────────────────────────────────────────
; A, A→B ⊢ B (upstream: lib_pln.metta Truth_ModusPonens)
(= (|-thrml ($A $T1)
            ((Implication $A $B) $T2))
   ($B (thrml-modus-ponens! ($A $B $T1 $T2))))

; ── Symmetric Modus Ponens ───────────────────────────────────────────
; A, A~B ⊢ B (upstream: lib_pln.metta Truth_SymmetricModusPonens)
; Uses SymmetricModusPonensRuleGuard like upstream
(= (|-thrml ($A $T1)
            (($LinkType $A $B) $T2))
   (if (SymmetricModusPonensRuleGuard $LinkType)
       ($B (thrml-symmetric-mp! ($A $B $T1 $T2))) (empty)))

; ── Deduction ────────────────────────────────────────────────────────
; A→B, B→C ⊢ A→C (upstream: lib_pln.metta Truth_Deduction)
; Uses SyllogisticRuleGuard like upstream
(= (|-thrml (($LinkType $A $B) $T1)
            (($LinkType $B $C) $T2))
   (if (SyllogisticRuleGuard $LinkType)
       (($LinkType $A $C) (thrml-deduction! ($A $B $C $T1 $T2))) (empty)))

; ── Induction ────────────────────────────────────────────────────────
; C→A, C→B ⊢ A→B (upstream: lib_pln.metta Truth_Induction)
; Uses SyllogisticRuleGuard like upstream
(= (|-thrml (($LinkType $C $A) $T1)
            (($LinkType $C $B) $T2))
   (if (SyllogisticRuleGuard $LinkType)
       (($LinkType $A $B) (thrml-induction! ($A $B $C $T1 $T2))) (empty)))

; ── Abduction ────────────────────────────────────────────────────────
; A→C, B→C ⊢ A→B (upstream: lib_pln.metta Truth_Abduction)
; Uses SyllogisticRuleGuard like upstream
(= (|-thrml (($LinkType $A $C) $T1)
            (($LinkType $B $C) $T2))
   (if (SyllogisticRuleGuard $LinkType)
       (($LinkType $A $B) (thrml-abduction! ($A $B $C $T1 $T2))) (empty)))

; ── Negation ─────────────────────────────────────────────────────────
; negation elimination (upstream: |- ((Not $A) $T) => ($A (Truth_Negation $T)))
(= (|-thrml ((Not $A) $T))
   ($A (thrml-negation! ($A $T))))

; ── Inversion ────────────────────────────────────────────────────────
; Inheritance Inversion Rule (upstream: lib_pln.metta Truth_inversion)
(= (|-thrml ((Inheritance $A $B) $T))
   ((Inheritance $B $A) (thrml-inversion! ($A $B $T))))

; Implication Inversion Rule (same)
(= (|-thrml ((Implication $A $B) $T))
   ((Implication $B $A) (thrml-inversion! ($A $B $T))))

; ── Equivalence to Implication ───────────────────────────────────────
; A≡B ⊢ A→B (upstream: lib_pln.metta Truth_equivalenceToImplication)
(= (|-thrml ((Equivalence $A $B) $T))
   ((Implication $A $B) (thrml-equiv-to-impl! ($A $B $T))))

(= (|-thrml ((Equivalence $A $B) $T))
   ((Implication $B $A) (thrml-equiv-to-impl! ($B $A $T))))

; ── Transitive Similarity ────────────────────────────────────────────
; A~B, B~C ⊢ A~C (upstream: lib_pln.metta Truth_transitiveSimilarity)
(= (|-thrml ((Similarity $A $B) $T1)
            ((Similarity $B $C) $T2))
   ((Similarity $A $C) (thrml-transitive-sim! ($A $B $C $T1 $T2))))

; ── Evaluation Implication ───────────────────────────────────────────
; (Eval A B), (Impl A C) ⊢ (Eval C B)
; (upstream: lib_pln.metta Truth_evaluationImplication)
(= (|-thrml ((Evaluation $A $B) $T1)
            ((Implication $A $C) $T2))
   ((Evaluation $C $B) (thrml-eval-impl! ($A $B $C $T1 $T2))))

; ── Evaluation via Inheritance (upstream: Usage of inheritance for predicates) ──
; unary arg
(= (|-thrml ((Evaluation (Predicate $x)
                         (List (Concept $C))) $T1)
            ((Inheritance (Concept $S) (Concept $C)) $T2))
   ((Evaluation (Predicate $x)
                (List (Concept $S))) (thrml-modus-ponens! ((Evaluation (Predicate $x) (List (Concept $C))) (Evaluation (Predicate $x) (List (Concept $S))) $T1 $T2))))

; ── Member Deduction ─────────────────────────────────────────────────
; (Member A B), (Inheritance B C) ⊢ (Member A C)
; (upstream: extensional member deduction, uses Truth_Deduction)
(= (|-thrml ((Member $A $B) $T1)
            ((Inheritance $B $C) $T2))
   ((Member $A $C) (thrml-deduction! ($A $B $C $T1 $T2))))

; ── Full-graph compilation ─────────────────────────────────────────
; Compile entire knowledge space into one factor graph, sample once.
; Usage:
;   !(thrml-compile!)              ; compile & sample
;   !(thrml-query! (X))            ; P(X=1)        marginal
;   !(thrml-query! (Y X))          ; P(Y=1|X=1)    conditional
