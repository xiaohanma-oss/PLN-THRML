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
