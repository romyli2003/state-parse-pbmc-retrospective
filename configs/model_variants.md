# Exploratory model variants

| Variant | Campaign label | Batch encoder | Intended control condition |
|---|---|---:|---|
| M1 | Exploratory baseline | No | Controls out |
| M2 | Exploratory baseline | No | Controls in |
| M3 | Exploratory batch-aware variant | Yes | Controls in |
| M4b | Exploratory batch-aware variant | Yes | Controls out |

All variants used the same strict donor-zero-shot evaluation design. The intended control labels should not be read as a validated causal ablation: a later audit found an earlier pair of controls-in/controls-out configuration files byte-identical. The retained aggregate comparisons are therefore descriptive engineering evidence only.
