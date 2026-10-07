#!/usr/bin/env python3
"""Lightweight H5AD/HDF5 layout inspector.

Avoids anndata.read_h5ad so it can inspect very large H5ADs safely.
Prints dataset/group layout, chunking/compression, key obs/var metadata, and X_hvg layout.
It deliberately does not traverse ``uns`` because legacy payloads are not needed
for structural inspection and may be malformed.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable

import h5py

KEY_OBS = [
    "donor", "Donor", "cell_type", "cell_type_clean", "celltype", "cell_type_key",
    "cytokine", "condition", "perturbation", "pert", "batch", "split", "well", "time", "dose"
]

MAX_ITEMS = 80
MAX_CATS = 80


def fmt_size(n: int | None) -> str:
    if n is None:
        return "NA"
    units = ["B", "KB", "MB", "GB", "TB"]
    x = float(n)
    for u in units:
        if x < 1024 or u == units[-1]:
            return f"{x:.1f}{u}"
        x /= 1024
    return f"{x:.1f}TB"


def node_summary(path: str, node) -> str:
    if isinstance(node, h5py.Dataset):
        return (
            f"DATASET {path}: shape={node.shape} dtype={node.dtype} "
            f"chunks={node.chunks} compression={node.compression} compression_opts={node.compression_opts}"
        )
    if isinstance(node, h5py.Group):
        return f"GROUP   {path}: n_children={len(node.keys())} keys={list(node.keys())[:MAX_ITEMS]}"
    return f"NODE    {path}: {type(node)}"


def print_node(f: h5py.File, key: str) -> None:
    if key in f:
        print(node_summary(key, f[key]))
        if isinstance(f[key], h5py.Group):
            for sub in list(f[key].keys())[:MAX_ITEMS]:
                p = f"{key}/{sub}"
                print("  " + node_summary(p, f[p]))
    else:
        print(f"MISSING {key}")


def decode_arr(arr):
    out = []
    for x in arr:
        if isinstance(x, bytes):
            out.append(x.decode("utf-8", errors="replace"))
        else:
            out.append(str(x))
    return out


def print_categorical_or_dataset(group: h5py.Group, col: str) -> None:
    if col not in group:
        return
    node = group[col]
    print(f"OBS_COL {col}: {node_summary('/obs/' + col, node)}")
    try:
        if isinstance(node, h5py.Group):
            if "categories" in node:
                cats = decode_arr(node["categories"][:min(len(node["categories"]), MAX_CATS)])
                print(f"  categories[{min(len(node['categories']), MAX_CATS)} of {len(node['categories'])}]: {cats}")
            if "codes" in node:
                codes = node["codes"]
                print(f"  codes: shape={codes.shape} dtype={codes.dtype} chunks={codes.chunks} compression={codes.compression}")
        elif isinstance(node, h5py.Dataset):
            n = min(node.shape[0] if node.shape else 1, 2000)
            vals = node[:n]
            vals = decode_arr(vals) if getattr(vals, "ndim", 1) else [str(vals)]
            uniq = []
            seen = set()
            for v in vals:
                if v not in seen:
                    uniq.append(v); seen.add(v)
                if len(uniq) >= MAX_CATS:
                    break
            print(f"  sample_unique_first_{n}: {uniq}")
    except Exception as e:
        print(f"  WARN could not inspect values for {col}: {type(e).__name__}: {e}")


def inspect(path: Path) -> None:
    print("=" * 100)
    print(f"FILE {path}")
    if not path.exists():
        print("MISSING_FILE")
        return
    print(f"size={fmt_size(path.stat().st_size)} mtime={path.stat().st_mtime}")
    try:
        with h5py.File(path, "r") as f:
            print(f"root_keys={list(f.keys())[:MAX_ITEMS]}")
            for key in ["X", "raw", "raw/X", "layers", "obsm", "obsm/X_hvg", "obs", "var"]:
                print_node(f, key)
            if "obs" in f and isinstance(f["obs"], h5py.Group):
                obs = f["obs"]
                print(f"OBS_COLUMNS n={len(obs.keys())}: {list(obs.keys())[:MAX_ITEMS]}")
                for col in KEY_OBS:
                    print_categorical_or_dataset(obs, col)
            if "var" in f and isinstance(f["var"], h5py.Group):
                var = f["var"]
                print(f"VAR_COLUMNS n={len(var.keys())}: {list(var.keys())[:MAX_ITEMS]}")
                for col in ["_index", "gene_ids", "gene_id", "feature_name", "ensembl_id", "highly_variable", "means", "dispersions", "dispersions_norm"]:
                    if col in var:
                        print("VAR_" + node_summary('/var/' + col, var[col]))
    except Exception as e:
        print(f"ERROR opening {path}: {type(e).__name__}: {e}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    args = ap.parse_args()
    for p in args.paths:
        inspect(Path(p))

if __name__ == "__main__":
    main()
