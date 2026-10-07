#!/usr/bin/env python3
"""Aggregate Cell-Eval outputs from four exploratory model variants.

Inputs are four directories containing per-cell-type Cell-Eval CSV outputs plus
their prediction H5AD files. The script validates structural compatibility,
computes macro and cell-count-weighted summaries, and writes CSV/JSON/PNG
artifacts. It does not train models or generate predictions.

Use `<DATA_ROOT>` for inputs and write results to a new output directory.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import re
import sys
import traceback
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


SUFFIXES = [
    ("_agg_results.csv", "agg"),
    ("_pred_de.csv", "pred_de"),
    ("_real_de.csv", "real_de"),
    ("_results.csv", "results"),
]

LOWER_IS_BETTER = {"mae", "mse"}
COUNT_METRICS = {"de_nsig_counts_real", "de_nsig_counts_pred"}


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n")


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("_")[:160]


def classify(path: Path) -> tuple[str | None, str | None]:
    for suffix, kind in SUFFIXES:
        if path.name.endswith(suffix):
            return path.name[:-len(suffix)], kind
    return None, None


def read_csv(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
    except Exception:
        df = pd.read_csv(path, engine="python")
    unnamed = [c for c in df.columns if str(c).startswith("Unnamed:")]
    if unnamed:
        df = df.drop(columns=unnamed)
    return df


def normalize_results(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    lower = {str(c).lower(): c for c in out.columns}
    id_col = next(
        (
            lower[x]
            for x in ("perturbation", "cytokine", "target", "condition")
            if x in lower
        ),
        None,
    )
    if id_col is None:
        raise KeyError(f"Cannot identify perturbation column: {list(out.columns)}")
    if id_col != "perturbation":
        out = out.rename(columns={id_col: "perturbation"})
    return out


def numeric_metrics(df: pd.DataFrame) -> list[str]:
    exclude = {"model", "cell_type", "perturbation"}
    metrics = []
    for col in df.columns:
        if col in exclude:
            continue
        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.notna().sum() > 0:
            df[col] = converted
            metrics.append(col)
    return metrics


def inventory_model(model: str, directory: Path) -> tuple[pd.DataFrame, dict[str, dict[str, Path]]]:
    rows = []
    groups: dict[str, dict[str, Path]] = {}
    for path in sorted(directory.glob("*.csv")):
        cell_type, kind = classify(path)
        rows.append(
            {
                "model": model,
                "filename": path.name,
                "path": str(path),
                "cell_type": cell_type,
                "kind": kind or "unknown",
                "size_bytes": path.stat().st_size,
                "zero_byte": path.stat().st_size == 0,
            }
        )
        if cell_type is not None and kind is not None:
            if kind in groups.setdefault(cell_type, {}):
                raise RuntimeError(f"Duplicate {model}/{cell_type}/{kind}")
            groups[cell_type][kind] = path
    return pd.DataFrame(rows), groups


def read_h5_shape(path: Path) -> tuple[int, int, str]:
    with h5py.File(path, "r") as f:
        x = f["X"]
        return int(x.shape[0]), int(x.shape[1]), str(x.dtype)


def read_obs(path: Path) -> pd.DataFrame:
    import anndata as ad
    with h5py.File(path, "r") as f:
        return ad.io.read_elem(f["obs"])


def stable_obs_digest(obs: pd.DataFrame, columns: list[str]) -> str:
    h = hashlib.sha256()
    h.update(str(len(obs)).encode())
    for col in columns:
        h.update(col.encode())
        values = obs[col].astype(str).to_numpy()
        for value in values:
            h.update(value.encode("utf-8", errors="replace"))
            h.update(b"\0")
    return h.hexdigest()


def summarize_model(
    results: pd.DataFrame,
    coverage: pd.DataFrame,
    model: str,
    metrics: list[str],
) -> list[dict[str, Any]]:
    sub = results[results["model"] == model].copy()
    weights = coverage.groupby(["cell_type", "perturbation"], as_index=False)["n_cells"].sum()
    sub = sub.merge(weights, on=["cell_type", "perturbation"], how="left")
    sub["n_cells"] = sub["n_cells"].fillna(0)

    rows = []
    for metric in metrics:
        values = pd.to_numeric(sub[metric], errors="coerce")
        keep = np.isfinite(values.to_numpy(dtype=float, na_value=np.nan))
        valid = sub.loc[keep].copy()
        if valid.empty:
            continue
        v = pd.to_numeric(valid[metric], errors="coerce").to_numpy(float)
        w = valid["n_cells"].to_numpy(float)
        rows.append(
            {
                "model": model,
                "metric": metric,
                "n_contexts": len(v),
                "macro_mean": float(np.mean(v)),
                "macro_median": float(np.median(v)),
                "macro_sd": float(np.std(v, ddof=1)) if len(v) > 1 else math.nan,
                "q10": float(np.quantile(v, 0.10)),
                "q90": float(np.quantile(v, 0.90)),
                "cell_count_weighted_mean": (
                    float(np.average(v, weights=w)) if np.sum(w) > 0 else math.nan
                ),
            }
        )
    return rows


def paired_contrast(
    results: pd.DataFrame,
    model_a: str,
    model_b: str,
    metrics: list[str],
    label: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    key = ["cell_type", "perturbation"]
    a = results[results["model"] == model_a].drop(columns=["model"])
    b = results[results["model"] == model_b].drop(columns=["model"])
    paired = a.merge(b, on=key, suffixes=(f"_{model_a}", f"_{model_b}"), how="inner")
    for metric in metrics:
        paired[f"{metric}_raw_delta_{model_b}_minus_{model_a}"] = (
            paired[f"{metric}_{model_b}"] - paired[f"{metric}_{model_a}"]
        )
        direction = -1.0 if metric in LOWER_IS_BETTER else 1.0
        paired[f"{metric}_improvement_{model_b}_over_{model_a}"] = (
            direction * paired[f"{metric}_raw_delta_{model_b}_minus_{model_a}"]
        )
    paired.insert(0, "contrast", label)

    improvement_cols = [
        f"{m}_improvement_{model_b}_over_{model_a}" for m in metrics
    ]
    by_cell = paired.groupby("cell_type", as_index=False)[improvement_cols].mean()
    by_cell.insert(0, "contrast", label)
    by_pert = paired.groupby("perturbation", as_index=False)[improvement_cols].mean()
    by_pert.insert(0, "contrast", label)
    return paired, by_cell, by_pert


def save_bar_summary(summary: pd.DataFrame, output: Path) -> None:
    plot_dir = output / "plots"
    plot_dir.mkdir(exist_ok=True)
    for metric, sub in summary.groupby("metric"):
        sub = sub.sort_values("model")
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.bar(sub["model"], sub["macro_mean"])
        ax.set_title(metric)
        ax.set_ylabel("Macro mean")
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(plot_dir / f"overall_{safe_name(metric)}.png", dpi=160)
        plt.close(fig)


def official_reference_inventory(reference_dir: Path | None) -> dict[str, Any]:
    if reference_dir is None or not reference_dir.exists():
        return {
            "available": False,
            "reason": "Official reference directory not supplied or does not exist.",
        }
    files = sorted(reference_dir.glob("zeroshot/split_*/eval_best.ckpt/*_results.csv"))
    # Exclude agg files from this list.
    files = [f for f in files if not f.name.endswith("_agg_results.csv")]
    return {
        "available": bool(files),
        "reference_dir": str(reference_dir),
        "result_files": [str(f) for f in files],
        "n_result_files": len(files),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--official-reference-dir", type=Path)
    args = ap.parse_args()

    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    write_json(args.output_dir / "status.json", {"status": "RUNNING"})

    manifest = json.loads(args.manifest.read_text())
    current = manifest["current_benchmark"]
    models = current["models"]

    inventories = []
    groups_by_model = {}
    all_results = []
    all_agg = []
    shape_rows = []

    for model, spec in models.items():
        ce_dir = Path(spec["cell_eval_dir"])
        real_h5ad = Path(spec["real_h5ad"])
        pred_h5ad = Path(spec["pred_h5ad"])
        for path in (ce_dir, real_h5ad, pred_h5ad):
            if not path.exists():
                raise FileNotFoundError(path)

        inv, groups = inventory_model(model, ce_dir)
        inventories.append(inv)
        groups_by_model[model] = groups

        required = {"results", "agg", "pred_de", "real_de"}
        missing = {
            cell: sorted(required - set(found))
            for cell, found in groups.items()
            if required - set(found)
        }
        if len(groups) != 18 or missing:
            raise RuntimeError(
                f"{model}: incomplete Cell-Eval output; "
                f"cell_types={len(groups)}, missing={missing}"
            )

        for cell_type, found in sorted(groups.items()):
            res = normalize_results(read_csv(found["results"]))
            res.insert(0, "cell_type", cell_type)
            res.insert(0, "model", model)
            all_results.append(res)

            agg = read_csv(found["agg"])
            agg.insert(0, "cell_type", cell_type)
            agg.insert(0, "model", model)
            all_agg.append(agg)

        for role, path in (("real", real_h5ad), ("pred", pred_h5ad)):
            n_obs, n_vars, dtype = read_h5_shape(path)
            shape_rows.append(
                {
                    "model": model,
                    "role": role,
                    "path": str(path),
                    "n_obs": n_obs,
                    "n_vars": n_vars,
                    "dtype": dtype,
                }
            )

    inventory = pd.concat(inventories, ignore_index=True)
    inventory.to_csv(args.output_dir / "file_inventory.csv", index=False)
    pd.DataFrame(shape_rows).to_csv(args.output_dir / "h5ad_shapes.csv", index=False)

    results = pd.concat(all_results, ignore_index=True, sort=False)
    agg = pd.concat(all_agg, ignore_index=True, sort=False)
    metrics = numeric_metrics(results)
    common_metrics = [
        m
        for m in metrics
        if all(m in results[results["model"] == model].columns for model in models)
    ]

    results.to_csv(args.output_dir / "all_context_metrics.csv", index=False)
    agg.to_csv(args.output_dir / "all_aggregate_csvs.csv", index=False)

    # Read only M3 real obs for weights, then verify M1 real obs against it.
    m3_obs = read_obs(Path(models["M3"]["real_h5ad"]))
    required_obs_cols = ["cell_type_clean", "cytokine"]
    for col in required_obs_cols:
        if col not in m3_obs.columns:
            raise KeyError(f"M3 real obs missing {col}")
    donor_col = "donor" if "donor" in m3_obs.columns else None
    group_cols = ([donor_col] if donor_col else []) + required_obs_cols
    coverage = (
        m3_obs.groupby(group_cols, observed=True)
        .size()
        .rename("n_cells")
        .reset_index()
        .rename(
            columns={
                "cell_type_clean": "cell_type",
                "cytokine": "perturbation",
            }
        )
    )
    coverage.to_csv(args.output_dir / "test_coverage_counts.csv", index=False)

    digest_cols = [c for c in ["donor", "cell_type_clean", "cytokine"] if c in m3_obs.columns]
    m3_digest = stable_obs_digest(m3_obs, digest_cols)
    del m3_obs
    gc.collect()

    m1_obs = read_obs(Path(models["M1"]["real_h5ad"]))
    m1_digest = stable_obs_digest(m1_obs, digest_cols)
    del m1_obs
    gc.collect()

    summary_rows = []
    for model in models:
        summary_rows.extend(
            summarize_model(results, coverage, model, common_metrics)
        )
    model_summary = pd.DataFrame(summary_rows)
    model_summary.to_csv(args.output_dir / "model_metric_summary.csv", index=False)
    save_bar_summary(model_summary, args.output_dir)

    contrasts = [
        ("M1", "M2", "M2_minus_M1_controls_effect_without_batch_encoder"),
        ("M2", "M3", "M3_minus_M2_batch_encoder_effect_with_controls"),
        ("M1", "M4b", "M4b_minus_M1_batch_encoder_effect_without_controls"),
        ("M4b", "M3", "M3_minus_M4b_controls_effect_with_batch_encoder"),
    ]
    paired_frames = []
    cell_frames = []
    pert_frames = []
    for a, b, label in contrasts:
        paired, by_cell, by_pert = paired_contrast(
            results, a, b, common_metrics, label
        )
        paired_frames.append(paired)
        cell_frames.append(by_cell)
        pert_frames.append(by_pert)

    paired_all = pd.concat(paired_frames, ignore_index=True)
    cell_all = pd.concat(cell_frames, ignore_index=True)
    pert_all = pd.concat(pert_frames, ignore_index=True)
    paired_all.to_csv(args.output_dir / "paired_contrasts_context.csv", index=False)
    cell_all.to_csv(args.output_dir / "paired_contrasts_cell_type.csv", index=False)
    pert_all.to_csv(args.output_dir / "paired_contrasts_perturbation.csv", index=False)

    # Difference-in-differences: interaction of controls and batch encoder.
    key = ["cell_type", "perturbation"]
    wide = results.pivot_table(
        index=key,
        columns="model",
        values=common_metrics,
        aggfunc="first",
    )
    did_rows = []
    for metric in common_metrics:
        frame = wide[metric].dropna(subset=["M1", "M2", "M3", "M4b"]).copy()
        direction = -1.0 if metric in LOWER_IS_BETTER else 1.0
        raw_did = (frame["M3"] - frame["M2"]) - (frame["M4b"] - frame["M1"])
        improvement_did = direction * raw_did
        for idx, raw, imp in zip(frame.index, raw_did, improvement_did):
            did_rows.append(
                {
                    "cell_type": idx[0],
                    "perturbation": idx[1],
                    "metric": metric,
                    "raw_difference_in_differences": float(raw),
                    "improvement_scale_difference_in_differences": float(imp),
                }
            )
    did = pd.DataFrame(did_rows)
    did.to_csv(args.output_dir / "difference_in_differences_context.csv", index=False)
    did.groupby("metric", as_index=False).agg(
        n_contexts=("improvement_scale_difference_in_differences", "size"),
        macro_mean=("improvement_scale_difference_in_differences", "mean"),
        macro_median=("improvement_scale_difference_in_differences", "median"),
        positive_fraction=(
            "improvement_scale_difference_in_differences",
            lambda x: float((x > 0).mean()),
        ),
    ).to_csv(args.output_dir / "difference_in_differences_summary.csv", index=False)

    # M3 paper sanity table. This is deliberately gated and labeled approximate.
    paper = manifest["paper_benchmark"]
    aliases = manifest["metric_aliases"]
    m3_lookup = model_summary[model_summary["model"] == "M3"].set_index("metric")
    sanity_rows = []
    for paper_metric, current_metric in aliases.items():
        approx = paper["figure3h_st_hvg_approximate"][paper_metric]
        current_value = (
            float(m3_lookup.loc[current_metric, "macro_mean"])
            if current_metric in m3_lookup.index
            else math.nan
        )
        sanity_rows.append(
            {
                "paper_metric": paper_metric,
                "current_metric": current_metric,
                "paper_figure3h_st_hvg_approximate": approx,
                "M3_current_donor_zero_shot_macro": current_value,
                "raw_difference_M3_minus_paper_approx": current_value - approx,
                "directly_comparable": False,
                "reason": "Different held-out axis and non-identical model/training configuration.",
            }
        )
    for missing_metric in ("spearman_fold_change", "auprc"):
        sanity_rows.append(
            {
                "paper_metric": missing_metric,
                "current_metric": None,
                "paper_figure3h_st_hvg_approximate": paper[
                    "figure3h_st_hvg_approximate"
                ][missing_metric],
                "M3_current_donor_zero_shot_macro": math.nan,
                "raw_difference_M3_minus_paper_approx": math.nan,
                "directly_comparable": False,
                "reason": "Not produced by the completed minimal Cell-Eval profile.",
            }
        )
    pd.DataFrame(sanity_rows).to_csv(
        args.output_dir / "M3_paper_figure3H_sanity_check.csv",
        index=False,
    )

    official_ref = official_reference_inventory(args.official_reference_dir)

    alignment_items = [
        {
            "dimension": "dataset",
            "current": "Parse-PBMC",
            "paper": "Parse-PBMC",
            "match": True,
        },
        {
            "dimension": "held_out_axis",
            "current": "donor",
            "paper": "cell_type",
            "match": False,
        },
        {
            "dimension": "test_units",
            "current": "Donor7 and Donor10; 18 cell types",
            "paper": "five released zero-shot cell-type splits",
            "match": False,
        },
        {
            "dimension": "heldout_controls",
            "current": "available in M3",
            "paper": "available",
            "match": True,
        },
        {
            "dimension": "batch_encoder",
            "current": "M3 true",
            "paper_methods_table": "true",
            "official_released_ST_HVG_config": "false",
            "match": None,
            "note": "Source discrepancy; cannot assign a single unambiguous paper target.",
        },
        {
            "dimension": "hidden_dim",
            "current": 1440,
            "paper_methods_table": 1440,
            "official_released_ST_HVG_config": 384,
            "match": None,
            "note": "Source discrepancy; cannot assign a single unambiguous paper target.",
        },
        {
            "dimension": "metric_coverage",
            "current": sorted(common_metrics),
            "paper": paper["paper_metrics"],
            "match": False,
            "note": "Minimal profile lacks paper Figure 3H Spearman-fold-change and AUPRC metrics.",
        },
    ]

    direct_replication = all(
        item.get("match") is True
        for item in alignment_items
        if item.get("match") is not None
    )
    replication_status = (
        "DIRECTLY_COMPARABLE"
        if direct_replication
        else "NOT_DIRECTLY_COMPARABLE_CURRENT_BENCHMARK"
    )
    alignment = {
        "replication_status": replication_status,
        "current_benchmark_label": "strict donor zero-shot",
        "paper_benchmark_label": "cell-type zero-shot",
        "alignment_items": alignment_items,
        "official_reference": official_ref,
        "interpretation_gate": {
            "paper_replication_claim_allowed": direct_replication,
            "M3_paper_values_may_be_used_as": (
                "exact replication comparison"
                if direct_replication
                else "broad sanity check only"
            ),
            "zoomed_M3_analysis_allowed": True,
            "four_model_ablation_allowed": True,
            "required_label_for_zoomed_and_ablation_results": (
                "strict donor-zero-shot extension, not direct Figure 3H replication"
            ),
        },
    }
    write_json(args.output_dir / "paper_alignment_audit.json", alignment)

    checks = {
        "four_models_present": set(models) == {"M1", "M2", "M3", "M4b"},
        "all_models_have_18_complete_cell_types": all(
            len(groups_by_model[m]) == 18
            and all(
                set(found) == {"results", "agg", "pred_de", "real_de"}
                for found in groups_by_model[m].values()
            )
            for m in models
        ),
        "no_zero_byte_csvs": not bool(inventory["zero_byte"].any()),
        "same_cell_type_set": len(
            {tuple(sorted(groups_by_model[m])) for m in models}
        ) == 1,
        "all_h5ad_shapes_match": (
            len(
                {
                    (row["n_obs"], row["n_vars"])
                    for row in shape_rows
                }
            )
            == 1
        ),
        "M1_M3_real_obs_digest_match": m1_digest == m3_digest,
        "common_metric_count": len(common_metrics),
        "common_metrics": common_metrics,
    }
    structural_pass = all(
        value
        for key, value in checks.items()
        if key not in {"common_metric_count", "common_metrics"}
    ) and len(common_metrics) > 0

    compact = {
        "status": "PASS" if structural_pass else "CHECK_NEEDED",
        "structural_checks": checks,
        "replication_status": replication_status,
        "paper_replication_claim_allowed": direct_replication,
        "n_context_rows": int(
            len(results[results["model"] == "M3"])
        ),
        "n_test_cells": int(coverage["n_cells"].sum()),
        "n_cell_types": int(coverage["cell_type"].nunique()),
        "n_perturbations_including_control": int(
            coverage["perturbation"].nunique()
        ),
        "primary_outputs": [
            "paper_alignment_audit.json",
            "M3_paper_figure3H_sanity_check.csv",
            "model_metric_summary.csv",
            "paired_contrasts_context.csv",
            "paired_contrasts_cell_type.csv",
            "paired_contrasts_perturbation.csv",
            "difference_in_differences_summary.csv",
        ],
    }
    write_json(args.output_dir / "compact_summary.json", compact)
    write_json(
        args.output_dir / "status.json",
        {"status": compact["status"], "replication_status": replication_status},
    )

    lines = [
        f"STATUS={compact['status']}",
        f"REPLICATION_STATUS={replication_status}",
        f"PAPER_REPLICATION_CLAIM_ALLOWED={direct_replication}",
        f"N_TEST_CELLS={compact['n_test_cells']}",
        f"N_CELL_TYPES={compact['n_cell_types']}",
        f"N_PERTURBATIONS_INCLUDING_CONTROL={compact['n_perturbations_including_control']}",
        f"N_CONTEXT_ROWS={compact['n_context_rows']}",
        f"COMMON_METRICS={','.join(common_metrics)}",
        f"OFFICIAL_REFERENCE_AVAILABLE={official_ref.get('available', False)}",
    ]
    (args.output_dir / "SUMMARY.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0 if structural_pass else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        traceback.print_exc()
        raise
