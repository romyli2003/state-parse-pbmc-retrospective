# External-validation readiness

This workstream was a readiness assessment, not completed independent model validation. It focused on whether candidate cytokine datasets had sufficient metadata, stable cell identifiers, interpretable control labels, usable annotation provenance, and safely inspectable file structures.

Wood, Dong, and Cui had varying levels of metadata-linked preparation evidence, but each retained study-specific limitations. HIRISA and GSE181897 remained pending source-file verification. No external QC threshold, normalization, cell-type crosswalk, eligibility rule, model scoring, or validation conclusion was established in this package.

The retained `scripts/data_inspection/preflight_processed_h5.py` script illustrates a reusable structure-only check that avoids loading expression payloads. It does not itself validate a dataset scientifically.

## Candidate-study identifiers

- **HIRISA:** [GSE306664](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE306664).
- **64-donor cytokine candidate:** [GSE181897](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE181897).
- **IFN-beta time-course candidate:** [GSE226572](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE226572).
- **Rigby type-I-interferon candidate:** [PRJEB60774](https://www.ebi.ac.uk/ena/browser/view/PRJEB60774).
- **Wood:** [preprint](https://doi.org/10.1101/2025.06.27.661918).
- **Dong:** [dataset record](https://doi.org/10.5061/dryad.4xgxd25g1) and [paper](https://doi.org/10.1038/s41592-023-02040-5).
- **Cui:** [paper](https://doi.org/10.1038/s41586-023-06816-9).
