# Results summary

The four exploratory variants each had complete prediction and Cell-Eval output inventories for the retained held-out-donor evaluation. `results/summary_metrics.csv` contains macro, distributional, and cell-count-weighted aggregate metrics for 1,617 contexts. The retained three-panel model figure displays macro discrimination score, Pearson correlation of perturbation deltas, and overlap at N.

- **Discrimination score:** separates perturbed from control response structure; higher is better.
- **Pearson correlation of response deltas:** correlation between predicted and observed perturbation-associated expression changes; higher is better.
- **Top-N response overlap:** overlap between predicted and observed leading response features; higher is better.

These figures show that the variants differed in aggregate behavior. They do not establish a definitive ranking, clinical utility, or a causal effect of control inclusion or batch encoding. The controls-configuration correction and pooled-donor evaluation limit are material to their interpretation.

Parse QC found 9,697,974 released cells, 12 donors, 90 cytokines plus a PBS control, and 18 cell types. At the 30-cell coverage reference, perturbed-depth eligibility retained 14,583 candidate strata (75.0%) and 99.5% of perturbed cells; additionally requiring a matched donor × cell-type PBS pool removed six more strata. These are coverage scenarios, not recommended QC thresholds.

The retained QC figures show released-cell QC distributions, cell-type QC profiles, and coverage sensitivity. They do not diagnose doublets or ambient RNA because those fields were not available in the stored metadata.
