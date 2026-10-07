#!/usr/bin/env python3
"""Validate full split AnnData files and strict donor-matching coverage.

Inputs are prepared train/validation/test H5AD files and the corresponding
STATE TOML. The script checks split-specific metadata, 2,000-feature HVG shape,
donor-disjoint design, and the availability of PBS controls for perturbed
donor × cell-type groups. It is a post-preparation validator, not a writer.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import pandas as pd


EXPECTED_DONORS = {
    "train": {
        "Donor1", "Donor11", "Donor12", "Donor2",
        "Donor5", "Donor6", "Donor8", "Donor9",
    },
    "val": {"Donor3", "Donor4"},
    "test": {"Donor10", "Donor7"},
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--toml", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    for split in ("train", "val", "test"):
        path = args.data_dir / f"full_split_v1_{split}.h5ad"
        if not path.exists():
            raise FileNotFoundError(path)

        adata = ad.read_h5ad(path, backed="r")
        required_obs = {
            "donor", "cytokine", "treatment", "sample",
            "cell_type", "cell_type_clean", "split",
            "source_row_index",
        }
        missing = required_obs - set(adata.obs.columns)
        if missing:
            raise ValueError(f"{split}: missing obs columns {sorted(missing)}")
        if "X_hvg" not in adata.obsm:
            raise ValueError(f"{split}: missing obsm['X_hvg']")
        if adata.obsm["X_hvg"].shape != (adata.n_obs, 2000):
            raise ValueError(
                f"{split}: unexpected X_hvg shape "
                f"{adata.obsm['X_hvg'].shape}"
            )
        if "feature" not in adata.var.columns:
            raise ValueError(f"{split}: var['feature'] missing")
        if adata.n_vars != 2000:
            raise ValueError(f"{split}: expected 2000 vars, got {adata.n_vars}")

        donors = set(adata.obs["donor"].astype(str).unique())
        if donors != EXPECTED_DONORS[split]:
            raise ValueError(
                f"{split}: donors {sorted(donors)} do not match "
                f"expected {sorted(EXPECTED_DONORS[split])}"
            )

        controls = (
            adata.obs.assign(is_pbs=adata.obs["cytokine"].astype(str).eq("PBS"))
            .groupby(["donor", "cell_type"], observed=True)
            .agg(
                n_total=("cytokine", "size"),
                n_PBS=("is_pbs", "sum"),
            )
            .reset_index()
        )
        controls["n_perturbed"] = controls["n_total"] - controls["n_PBS"]
        conflicts = controls.loc[
            (controls["n_perturbed"] > 0) & (controls["n_PBS"] == 0)
        ]
        if not conflicts.empty:
            raise ValueError(
                f"{split}: donor x cell-type groups still lack PBS:\n"
                f"{conflicts.to_string(index=False)}"
            )

        if split == "train":
            forbidden = (
                (adata.obs["donor"].astype(str) == "Donor5")
                & (adata.obs["cell_type"].astype(str) == "Plasmablast")
            )
            if forbidden.any():
                raise ValueError(
                    "train: Donor5 Plasmablast cells remain after exclusion"
                )

        rows.append(
            {
                "split": split,
                "n_obs": adata.n_obs,
                "n_vars": adata.n_vars,
                "n_donors": len(donors),
                "n_cell_types": adata.obs["cell_type"].nunique(),
                "n_cytokines": adata.obs["cytokine"].nunique(),
                "file_gib": path.stat().st_size / 1024**3,
                "strict_matching_conflicts": len(conflicts),
            }
        )
        adata.file.close()

    if not args.toml.exists():
        raise FileNotFoundError(args.toml)
    toml_text = args.toml.read_text()
    for split in ("train", "val", "test"):
        expected = str(args.data_dir / f"full_split_v1_{split}.h5ad")
        if expected not in toml_text:
            raise ValueError(f"TOML does not reference {expected}")

    summary = pd.DataFrame(rows)
    print("Full split validation passed.")
    print(summary.to_string(index=False))
    print(f"TOML: {args.toml}")


if __name__ == "__main__":
    main()
