# Limitations

## Configuration correction

A later configuration audit found that an earlier pair of files labelled controls-in and controls-out was byte-identical. Therefore the original campaign labels do not support a clean causal statement about control inclusion. The aggregate four-variant results retained here were generated after that audit and have an established analysis provenance, but they include outputs from the affected configuration history. They are displayed only as exploratory engineering evidence.

## Evaluation scope

- The two held-out donors were pooled within Cell-Eval contexts.
- Aggregate metrics can conceal context-specific failure modes, including false positives and weak-response behavior.
- The project did not complete an independent-study validation.
- The comparison is not a direct paper replication and should not be presented as one.

## Data and QC scope

- Parse is a vendor-processed released H5AD, not unfiltered sequencing output.
- QC was descriptive and did not set new filtering thresholds.
- Stored metadata did not directly support doublet or ambient-RNA assessment.
- Candidate external datasets have heterogeneous platform, time, annotation, control, and provenance limitations.
