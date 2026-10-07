# Methods at a glance

## Large-data preparation

The released H5AD was inspected through HDF5 structure and selected metadata rather than loaded as a full AnnData object. Preparation emphasized stable feature ordering, a custom high-variable-gene representation, categorical metadata checks, and split manifests that could be validated before training.

## Training and evaluation

STATE/ST-HVG-Parse experiments used set-based inputs and a donor-zero-shot split. Project-specific work covered configuration preparation, preflight checks, campaign orchestration, prediction-output audits, and aggregate Cell-Eval analysis. Upstream STATE and Cell-Eval implementations are not included.

## QC and validation readiness

Parse QC was descriptive and read-only. It used released-cell metadata and reported donor, perturbation, cell-type, depth, PBS-control availability, and threshold sensitivity. Candidate external datasets were assessed through metadata, file structure, and provenance only where available; no external model validation is claimed.
