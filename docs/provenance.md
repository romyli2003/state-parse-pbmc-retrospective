# Provenance

This file maps retained results to logical source artifacts without recording server locations. The private audit report outside this package contains the full original-to-sanitized mapping.

| Retained artifact | Generating artifact / source | Generated | Provenance status |
|---|---|---|---|
| `results/summary_metrics.csv` | Four-model aggregate alignment analysis | 2026-07-22 | Verified: four variants, common cell types/contexts, nonempty outputs, matching real-observation metadata. Exploratory only. |
| `results/figures/model_results/exploratory_heldout_donor_metrics.png` | Regenerated from retained `summary_metrics.csv` by `scripts/evaluation/generate_portfolio_model_figures.py` | 2026-08-21 | Verified; aggregate macro display with no new analysis. |
| `results/figures/parse_qc/released_cell_qc_distributions.png` | Parse descriptive QC v3 | 2026-08-04 | Verified; read-only metadata/QC output. |
| `results/figures/parse_qc/coverage_sensitivity.png` | Parse descriptive QC v3 | 2026-08-04 | Verified; coverage sensitivity, not a QC rule. |
| `results/figures/parse_qc/qc_profile_by_cell_type.png` | Parse descriptive QC v3 | 2026-08-04 | Verified; descriptive cell-type QC display. |

The model alignment analysis post-dates the configuration audit. Its underlying variant history is nevertheless affected by the controls correction; consult `limitations.md` before interpreting any retained model result.
