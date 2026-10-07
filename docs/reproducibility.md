# Reproducibility notes

This is a compact technical record, not a turnkey one-command reproduction package. Reproduction would require independently obtaining the original Parse release, upstream STATE and Cell-Eval dependencies, compatible compute resources, and any required data-use permissions.

Minimal conceptual dependencies are Python, HDF5 tooling, sparse-matrix support, tabular analysis, plotting, and the upstream STATE runtime. Large H5AD files should be inspected structurally before full-object loading. Preserve donor-aware grouping and matched-control semantics throughout preparation and evaluation.

The sanitized scripts use arguments and placeholders such as `<DATA_ROOT>` and `<STATE_REPO>`. Review each script and current upstream documentation before use; they are examples of project methods, not a supported production workflow.
