#!/usr/bin/env python3
"""Audit whether two STATE TOMLs substantively encode different control policies.

Inputs
------
``--controls-in`` and ``--controls-out`` are STATE TOML files.  The optional
``--training-h5ad`` enables a metadata-only check of the donor and perturbation
fields; no expression values are read.

Outputs
-------
A JSON report documenting byte equality, parsed dataset/training sections, and,
when requested, held-out-control counts.  This check was used to prevent named
configuration labels from being mistaken for verified experimental differences.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import tomllib
from pathlib import Path
from typing import Any

import h5py
import numpy as np


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of a small configuration file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compact_toml(path: Path) -> dict[str, Any]:
    """Keep configuration fields relevant to data/split semantics."""
    parsed = tomllib.loads(path.read_text())
    return {
        "dataset_keys": sorted(parsed.get("datasets", {})),
        "training": parsed.get("training", {}),
        "zeroshot": parsed.get("zeroshot", {}),
        "fewshot": parsed.get("fewshot", {}),
    }


def decode(values: np.ndarray) -> list[str]:
    return [x.decode("utf-8", errors="replace") if isinstance(x, bytes) else str(x) for x in values]


def categorical_values(handle: h5py.File, key: str) -> np.ndarray:
    """Read one categorical observation field; never open X or layers."""
    node = handle[f"obs/{key}"]
    if isinstance(node, h5py.Group) and {"codes", "categories"} <= set(node):
        categories = np.asarray(decode(node["categories"][:]), dtype=object)
        return categories[node["codes"][:].astype(int)]
    return np.asarray(decode(node[:]), dtype=object)


def optional_metadata_check(path: Path, held_out: set[str], control: str) -> dict[str, int]:
    with h5py.File(path, "r") as handle:
        donors = categorical_values(handle, "donor")
        perturbations = categorical_values(handle, "cytokine")
    held = np.isin(donors, sorted(held_out))
    is_control = perturbations == control
    return {
        "held_out_total": int(held.sum()),
        "held_out_control": int((held & is_control).sum()),
        "held_out_perturbed": int((held & ~is_control).sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controls-in", type=Path, required=True)
    parser.add_argument("--controls-out", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--training-h5ad", type=Path)
    parser.add_argument("--held-out-donors", nargs="*", default=[])
    parser.add_argument("--control-label", default="PBS")
    args = parser.parse_args()

    for path in (args.controls_in, args.controls_out):
        if not path.is_file():
            raise FileNotFoundError(path)
    report: dict[str, Any] = {
        "controls_in_sha256": sha256(args.controls_in),
        "controls_out_sha256": sha256(args.controls_out),
        "byte_identical": args.controls_in.read_bytes() == args.controls_out.read_bytes(),
        "controls_in": compact_toml(args.controls_in),
        "controls_out": compact_toml(args.controls_out),
    }
    if args.training_h5ad is not None:
        report["metadata_only_training_check"] = optional_metadata_check(
            args.training_h5ad, set(args.held_out_donors), args.control_label
        )
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
