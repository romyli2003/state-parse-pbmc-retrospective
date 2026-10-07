# Experimental design

## Donor-zero-shot evaluation

The exploratory evaluation held out donors rather than randomly splitting cells. Eight donors were used for training, two for validation, and two for test. The retained analysis covered approximately 1.23 million test cells, 18 cell types, 90 perturbations plus PBS, and 1,617 donor-pooled cytokine × cell-type contexts.

This design reduces direct donor leakage but does not establish all forms of generalization. In particular, the current Cell-Eval grouping pooled the two held-out donors within contexts; it is not a substitute for donor-level uncertainty estimation or independent-study validation.

## Variant comparison

The four variants crossed intended control inclusion with batch-encoder status. They were planned as an exploratory engineering comparison, not a preregistered causal ablation. See `configs/model_variants.md` and the correction described in `limitations.md`.
