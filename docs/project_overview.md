# Project overview

This project explored cytokine-response prediction in PBMCs using a large public perturbation dataset and the upstream STATE framework. Its technical arc was: inspect the released H5AD without loading the full matrix; prepare a reproducible feature representation and donor-aware splits; run exploratory STATE variants; evaluate held-out donors; perform descriptive QC; and assess the metadata/readiness of candidate independent datasets.

The unit of biological interpretation is donor × cytokine × cell type. Cells are observations rather than independent biological replicates. This distinction shaped split construction, QC summaries, and the interpretation limits applied to aggregate evaluation outputs.

The package documents project-specific engineering and analysis decisions. It does not claim ownership of STATE, Cell-Eval, the Parse dataset, or any candidate external dataset.
