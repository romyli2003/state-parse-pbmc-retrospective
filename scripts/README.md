# Scripts

These are sanitized project-specific scripts, not a turnkey pipeline. They require separately obtained data and dependencies; no upstream STATE or Cell-Eval code is included.

| Script | Purpose | Example |
|---|---|---|
| `data_inspection/inspect_h5ad_structure.py` | Inspect H5AD layout and selected metadata without loading expression values or `uns`. | `python inspect_h5ad_structure.py <DATA_ROOT>/input.h5ad` |
| `data_inspection/preflight_processed_h5.py` | Inventory a candidate validation HDF5/H5AD file without sampling matrix payloads. | `python preflight_processed_h5.py <DATA_ROOT>/candidate.h5ad --output-dir report` |
| `data_preparation/validate_donor_zero_shot_split.py` | Validate prepared donor-zero-shot H5AD splits and their STATE TOML. | `python validate_donor_zero_shot_split.py --data-dir prepared --toml split.toml` |
| `training/audit_control_configuration.py` | Check whether differently labelled configuration files actually differ. | `python audit_control_configuration.py --controls-in in.toml --controls-out out.toml --output audit.json` |
| `evaluation/analyze_four_variant_aggregates.py` | Validate and aggregate four Cell-Eval output sets. | `python analyze_four_variant_aggregates.py --help` |
| `evaluation/generate_portfolio_model_figures.py` | Create the retained aggregate model figure from `summary_metrics.csv`. | `python generate_portfolio_model_figures.py --metrics results/summary_metrics.csv --output figure.png` |
| `qc/parse_metadata_qc.py` | Produce read-only Parse metadata QC and coverage-sensitivity outputs. | `python parse_metadata_qc.py --input <DATA_ROOT>/input.h5ad --output qc_report` |

The H5AD preflight and inspection scripts deliberately avoid AnnData loading. The QC and split-validator scripts require review before use on large inputs.

## Minimal dependencies

- Python 3.11 or later
- anndata
- h5py
- numpy
- pandas
- matplotlib
- seaborn

Verified package versions in the STATE Python environment are recorded in `requirements-minimal.txt`. This is a compact compatibility note, not a complete environment export.

## Four-variant aggregate manifest

`evaluation/analyze_four_variant_aggregates.py` accepts a manifest describing each model’s Cell-Eval directory and paired real/predicted H5AD outputs. A minimal placeholder structure is:

```json
{
  "current_benchmark": {
    "models": {
      "M1": {
        "cell_eval_dir": "<DATA_ROOT>/M1/cell_eval",
        "real_h5ad": "<DATA_ROOT>/M1/real.h5ad",
        "pred_h5ad": "<DATA_ROOT>/M1/pred.h5ad"
      }
    }
  }
}
```

The same structure is required for M2, M3, and M4b. The model paths must refer to matched held-out evaluation outputs; the script validates structural compatibility before aggregation.
