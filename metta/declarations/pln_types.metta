; pln_types.metta — Type declarations for PLN thermodynamic inference
; ====================================================================
; Aligned with upstream lib_pln.metta (trueagi-io/PLN)

(: stv (-> Number Number Atom))

(: Implication (-> Atom Atom Atom))
(: Inheritance (-> Atom Atom Atom))
(: Similarity (-> Atom Atom Atom))
(: Equivalence (-> Atom Atom Atom))
(: Evaluation (-> Atom Atom Atom))
(: Member (-> Atom Atom Atom))

; Extension: background rate for thrml factor graphs
(: Background (-> Atom Atom Atom))

; STV query — upstream forward declaration + per-node overrides
(= (STV $stv) (empty))
; Fallback: query space for (node (stv s c)) assertions
(= (STV $x) (match &self ($x (stv $s $c)) (stv $s $c)))

; Guards (from upstream lib_pln.metta)
(= (SyllogisticRuleGuard Inheritance) True)
(= (SyllogisticRuleGuard Implication) True)
(= (SymmetricModusPonensRuleGuard Similarity) True)
(= (SymmetricModusPonensRuleGuard IntentionalSimilarity) True)
(= (SymmetricModusPonensRuleGuard ExtensionalSimilarity) True)

; Consistency conditions (from upstream lib_pln.metta)
(: smallest-intersection-probability (-> Number Number Number))
(= (smallest-intersection-probability $As $Bs)
   (clamp (/ (- (+ $As $Bs) 1) $As) 0 1))

(: largest-intersection-probability (-> Number Number Number))
(= (largest-intersection-probability $As $Bs)
   (clamp (/ $Bs $As) 0 1))

(: conditional-probability-consistency (-> Number Number Number Bool))
(= (conditional-probability-consistency $As $Bs $ABs)
   (and (< 0 $As)
        (and (<= (smallest-intersection-probability $As $Bs) $ABs)
             (<= $ABs (largest-intersection-probability $As $Bs)))))
