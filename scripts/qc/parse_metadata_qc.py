#!/usr/bin/env python3
"""Read-only, presentation-oriented QC audit for the original Parse-PBMC H5AD.

The script deliberately reads only selected ``obs`` columns with h5py/anndata.
It does not load the expression matrix, normalize data, filter cells, modify the
input file, or touch ``uns`` (the released file has a legacy ``uns/log1p`` entry
that is unnecessary for this analysis).
"""

from __future__ import annotations

import argparse
import html
import json
import platform
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any, Iterable

import anndata as ad
import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


WORKFLOW_VERSION = "3.0"
CONTROL_NAMES = {"PBS", "CONTROL", "UNTREATED", "VEHICLE"}
BENCHMARK_THRESHOLDS = np.array([10, 20, 30, 50])

NAVY = "#183B56"
BLUE = "#2E86AB"
TEAL = "#2A9D8F"
ORANGE = "#F4A261"
RED = "#D1495B"
GREY = "#7A8793"
LIGHT = "#EEF3F6"

FIELD_CANDIDATES = {
    "donor": ["donor", "Donor"],
    "cytokine": ["cytokine", "perturbation", "condition"],
    "treatment": ["treatment", "condition_type"],
    "cell_type": ["cell_type_clean", "cell_type", "celltype"],
    "sample": ["sample", "sample_id"],
    "n_genes": ["gene_count", "n_genes_by_counts"],
    "log1p_n_genes": ["log1p_n_genes_by_counts"],
    "total_counts": ["tscp_count", "total_counts", "n_counts"],
    "log1p_total_counts": ["log1p_total_counts"],
    "pct_mito": ["pct_counts_MT", "pct_counts_mt", "percent_mito"],
    "matched_reads": ["mread_count", "matched_reads"],
}


@dataclass(frozen=True)
class ResolvedFields:
    donor: str
    cytokine: str
    cell_type: str
    n_genes: str | None
    log1p_n_genes: str | None
    total_counts: str | None
    log1p_total_counts: str | None
    pct_mito: str | None
    treatment: str | None
    sample: str | None
    matched_reads: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate read-only QC and coverage figures from the original Parse-PBMC H5AD."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input H5AD, for example <DATA_ROOT>/Parse_10M_PBMC_cytokines.h5ad",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New directory for compact reports and figures",
    )
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--scatter-sample", type=int, default=80_000)
    parser.add_argument("--violin-sample-per-group", type=int, default=2_500)
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=["png", "pdf", "svg"],
        default=["png", "pdf"],
        help="Figure formats to save. PNG is convenient for slides; PDF is editable/vector.",
    )
    return parser.parse_args()


def _decode_attr(value: Any) -> list[str]:
    if value is None:
        return []
    values = value.tolist() if hasattr(value, "tolist") else list(value)
    return [x.decode() if isinstance(x, bytes) else str(x) for x in values]


def obs_columns(obs_group: h5py.Group) -> list[str]:
    order = _decode_attr(obs_group.attrs.get("column-order"))
    return order or [key for key in obs_group.keys() if key != "_index"]


def first_present(columns: set[str], candidates: Iterable[str]) -> str | None:
    return next((candidate for candidate in candidates if candidate in columns), None)


def resolve_fields(columns: list[str]) -> ResolvedFields:
    available = set(columns)
    found = {
        key: first_present(available, candidates)
        for key, candidates in FIELD_CANDIDATES.items()
    }
    missing_dimensions = [
        key for key in ("donor", "cytokine", "cell_type") if found[key] is None
    ]
    if missing_dimensions:
        raise KeyError(
            "Missing required metadata dimensions: "
            f"{missing_dimensions}. Available obs columns: {columns}"
        )
    if found["n_genes"] is None and found["log1p_n_genes"] is None:
        raise KeyError("No detected-gene count field found in obs.")
    if found["total_counts"] is None and found["log1p_total_counts"] is None:
        raise KeyError("No transcript/UMI count field found in obs.")
    if found["pct_mito"] is None:
        raise KeyError("No mitochondrial-fraction field found in obs.")
    return ResolvedFields(**found)  # type: ignore[arg-type]


def read_obs_column(obs_group: h5py.Group, column: str) -> Any:
    """Read one H5AD dataframe column without constructing the full AnnData object."""
    return ad.io.read_elem(obs_group[column])


def as_category(values: Any) -> pd.Categorical:
    if isinstance(values, pd.Categorical):
        return values
    if isinstance(values, pd.Series) and isinstance(values.dtype, pd.CategoricalDtype):
        return values.array
    array = np.asarray(values)
    if array.dtype.kind == "S":
        array = array.astype(str)
    return pd.Categorical(array)


def as_float32(values: Any) -> np.ndarray:
    return np.asarray(values, dtype=np.float32)


def load_metadata(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    with h5py.File(path, "r") as handle:
        if "obs" not in handle or "var" not in handle:
            raise KeyError("Input is not a standard H5AD with obs and var groups.")
        columns = obs_columns(handle["obs"])
        fields = resolve_fields(columns)
        obs = handle["obs"]

        categories: dict[str, pd.Categorical] = {
            "donor": as_category(read_obs_column(obs, fields.donor)),
            "cytokine": as_category(read_obs_column(obs, fields.cytokine)),
            "cell_type": as_category(read_obs_column(obs, fields.cell_type)),
        }
        if fields.treatment:
            categories["treatment"] = as_category(read_obs_column(obs, fields.treatment))
        if fields.sample:
            categories["sample"] = as_category(read_obs_column(obs, fields.sample))

        if fields.n_genes:
            n_genes = as_float32(read_obs_column(obs, fields.n_genes))
        else:
            n_genes = np.expm1(
                as_float32(read_obs_column(obs, fields.log1p_n_genes))
            ).astype(np.float32)

        if fields.total_counts:
            total_counts = as_float32(read_obs_column(obs, fields.total_counts))
        else:
            total_counts = np.expm1(
                as_float32(read_obs_column(obs, fields.log1p_total_counts))
            ).astype(np.float32)

        numeric = {
            "n_genes": n_genes,
            "total_counts": total_counts,
            "pct_mito": as_float32(read_obs_column(obs, fields.pct_mito)),
        }
        if fields.matched_reads:
            numeric["matched_reads"] = as_float32(
                read_obs_column(obs, fields.matched_reads)
            )

        frame = pd.DataFrame({**categories, **numeric}, copy=False)
        if "X" in handle:
            x_element = handle["X"]
            if isinstance(x_element, h5py.Dataset):
                shape = x_element.shape
            else:
                encoded_shape = x_element.attrs.get("shape")
                shape = tuple(int(x) for x in encoded_shape) if encoded_shape is not None else (len(frame), len(handle["var"]))
            x_encoding = x_element.attrs.get("encoding-type", "unknown")
        else:
            shape = (len(frame), len(handle["var"]))
            x_encoding = "missing"
        if isinstance(x_encoding, bytes):
            x_encoding = x_encoding.decode()

        provenance = {
            "input_path": str(path),
            "n_obs": int(shape[0]),
            "n_vars": int(shape[1]),
            "x_encoding": str(x_encoding),
            "available_obs_columns": columns,
            "resolved_fields": fields.__dict__,
            "matrix_read": False,
            "input_modified": False,
        }
    return frame, provenance


def clean_categories(frame: pd.DataFrame) -> pd.DataFrame:
    for column in ["donor", "cytokine", "cell_type", "treatment", "sample"]:
        if column in frame:
            series = frame[column]
            if not isinstance(series.dtype, pd.CategoricalDtype):
                series = series.astype("category")
            old_categories = list(series.cat.categories)
            new_categories = [str(value).strip() for value in old_categories]
            # Parse labels are already unique; retain the originals if trimming would
            # collapse two labels so this cleanup can never corrupt categories.
            if len(set(new_categories)) == len(new_categories):
                series = series.cat.rename_categories(new_categories)
            frame[column] = series
    return frame


def configure_style() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    mpl.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.titleweight": "bold",
            "axes.titlecolor": NAVY,
            "axes.labelcolor": NAVY,
            "axes.edgecolor": "#CAD5DC",
            "grid.color": "#E7EDF1",
            "grid.linewidth": 0.8,
            "font.family": "DejaVu Sans",
            "legend.frameon": False,
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
        }
    )


def save_figure(fig: plt.Figure, stem: str, figures_dir: Path, formats: list[str], dpi: int) -> None:
    for extension in formats:
        fig.savefig(figures_dir / f"{stem}.{extension}", dpi=dpi)
    plt.close(fig)


def finite(values: pd.Series | np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return array[np.isfinite(array)]


def robust_upper(values: pd.Series | np.ndarray, quantile: float = 0.995) -> float:
    array = finite(values)
    return float(np.quantile(array, quantile)) if len(array) else 1.0


def format_count(value: float | int) -> str:
    value = float(value)
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if value >= 1_000:
        return f"{value / 1_000:.1f}K"
    return f"{value:.0f}"


def natural_order(values: Iterable[Any]) -> list[str]:
    def key(value: str) -> list[Any]:
        return [int(part) if part.isdigit() else part.casefold() for part in re.split(r"(\d+)", value)]

    return sorted({str(value) for value in values}, key=key)


def sample_rows(frame: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if len(frame) <= n:
        return frame.copy()
    return frame.sample(n=n, random_state=seed)


def stratified_sample(frame: pd.DataFrame, group: str, per_group: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    selected: list[np.ndarray] = []
    codes = frame[group].cat.codes.to_numpy()
    for code in range(len(frame[group].cat.categories)):
        idx = np.flatnonzero(codes == code)
        if len(idx) > per_group:
            idx = rng.choice(idx, size=per_group, replace=False)
        selected.append(idx)
    if not selected:
        return frame.iloc[[]].copy()
    return frame.iloc[np.concatenate(selected)].copy()


def metric_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric, label in [
        ("total_counts", "Transcripts per cell"),
        ("n_genes", "Detected genes per cell"),
        ("pct_mito", "Mitochondrial fraction (%)"),
    ]:
        values = finite(frame[metric])
        rows.append(
            {
                "metric": metric,
                "label": label,
                "nonmissing_n": int(len(values)),
                "missing_n": int(frame[metric].isna().sum()),
                "mean": float(np.mean(values)),
                "sd": float(np.std(values)),
                "min": float(np.min(values)),
                "p01": float(np.quantile(values, 0.01)),
                "p05": float(np.quantile(values, 0.05)),
                "median": float(np.median(values)),
                "p95": float(np.quantile(values, 0.95)),
                "p99": float(np.quantile(values, 0.99)),
                "max": float(np.max(values)),
            }
        )
    return pd.DataFrame(rows)


def qc_boundary_diagnostics(frame: pd.DataFrame) -> pd.DataFrame:
    """Describe observed metric boundaries without treating them as new cutoffs."""
    rows = []
    for metric, label in [
        ("total_counts", "Transcripts per cell"),
        ("n_genes", "Detected genes per cell"),
        ("pct_mito", "Mitochondrial fraction (%)"),
    ]:
        values = finite(frame[metric])
        observed_min = float(np.min(values))
        observed_max = float(np.max(values))
        rows.append(
            {
                "metric": metric,
                "label": label,
                "observed_min": observed_min,
                "observed_max": observed_max,
                "n_at_min": int(np.isclose(values, observed_min).sum()),
                "fraction_at_min": float(np.isclose(values, observed_min).mean()),
                "n_at_max": int(np.isclose(values, observed_max).sum()),
                "fraction_at_max": float(np.isclose(values, observed_max).mean()),
                "unique_values": int(np.unique(values).size),
            }
        )
    return pd.DataFrame(rows)


def figure_qc_distributions(frame: pd.DataFrame, out: Path, formats: list[str], dpi: int) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.333, 4.4))
    specs = [
        ("total_counts", "Transcripts per cell", TEAL, True),
        ("n_genes", "Detected genes per cell", BLUE, True),
        ("pct_mito", "Mitochondrial fraction (%)", ORANGE, False),
    ]
    for ax, (metric, label, color, log_x) in zip(axes, specs):
        values = finite(frame[metric])
        upper = robust_upper(values)
        plotted = values[(values >= 0) & (values <= upper)]
        if log_x:
            plotted = plotted[plotted > 0]
            bins = np.geomspace(max(1.0, plotted.min()), plotted.max(), 70)
            ax.set_xscale("log")
        else:
            bins = 70
        ax.hist(plotted, bins=bins, color=color, alpha=0.9, edgecolor="white", linewidth=0.25)
        median = float(np.median(values))
        ax.axvline(median, color=NAVY, linewidth=2, linestyle="--")
        ax.text(
            0.97,
            0.92,
            f"Median: {median:,.1f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            color=NAVY,
            fontsize=10,
        )
        ax.set_title(label, fontsize=14)
        ax.set_xlabel(label)
        ax.set_ylabel("Cells")
        ax.tick_params(labelsize=9)
    fig.suptitle("Released Parse-PBMC cells show broad single-cell QC distributions", fontsize=18, color=NAVY, y=1.04)
    fig.text(0.01, -0.03, "Displays up to the 99.5th percentile; dashed line marks the full-data median. No filtering applied.", fontsize=9, color=GREY)
    fig.tight_layout()
    save_figure(fig, "01_single_cell_qc_distributions", out, formats, dpi)


def figure_qc_by_donor(
    frame: pd.DataFrame,
    out: Path,
    formats: list[str],
    dpi: int,
    per_group: int,
    seed: int,
) -> None:
    sampled = stratified_sample(frame, "donor", per_group, seed)
    donor_order = natural_order(frame["donor"].dropna().astype(str).unique())
    fig, axes = plt.subplots(3, 1, figsize=(13.333, 8.2), sharex=True)
    specs = [
        ("total_counts", "Transcripts per cell", TEAL, True),
        ("n_genes", "Detected genes per cell", BLUE, True),
        ("pct_mito", "Mitochondrial fraction (%)", ORANGE, False),
    ]
    for ax, (metric, label, color, log_y) in zip(axes, specs):
        sns.boxplot(
            data=sampled,
            x="donor",
            y=metric,
            order=donor_order,
            ax=ax,
            color=color,
            width=0.72,
            showfliers=False,
            linewidth=0.9,
        )
        if log_y:
            ax.set_yscale("log")
        else:
            ax.set_ylim(0, robust_upper(frame[metric]))
        ax.set_ylabel(label, fontsize=11)
        ax.set_xlabel("")
        ax.tick_params(axis="both", labelsize=9)
    axes[-1].set_xlabel("Donor")
    fig.suptitle("Single-cell QC is compared across all 12 donors", fontsize=18, color=NAVY, y=1.01)
    fig.text(0.01, 0.005, f"Boxplots use a reproducible sample of up to {per_group:,} cells per donor; no cells are excluded from summaries.", fontsize=9, color=GREY)
    fig.tight_layout(rect=(0, 0.02, 1, 0.98))
    save_figure(fig, "02_single_cell_qc_by_donor", out, formats, dpi)


def figure_metric_relationships(
    frame: pd.DataFrame,
    out: Path,
    formats: list[str],
    dpi: int,
    n_sample: int,
    seed: int,
) -> None:
    sampled = sample_rows(frame[["total_counts", "n_genes", "pct_mito"]].dropna(), n_sample, seed)
    sampled = sampled[(sampled["total_counts"] > 0) & (sampled["n_genes"] > 0)]
    mt_cap = robust_upper(sampled["pct_mito"], 0.99)
    fig, axes = plt.subplots(1, 2, figsize=(13.333, 5.1))
    scatter = axes[0].scatter(
        sampled["total_counts"],
        sampled["n_genes"],
        c=np.clip(sampled["pct_mito"], 0, mt_cap),
        s=6,
        alpha=0.32,
        cmap="viridis",
        linewidths=0,
        rasterized=True,
    )
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Transcripts per cell")
    axes[0].set_ylabel("Detected genes per cell")
    axes[0].set_title("Library complexity")
    cbar = fig.colorbar(scatter, ax=axes[0], pad=0.02)
    cbar.set_label("Mitochondrial fraction (%)", fontsize=10)

    hexplot = axes[1].hexbin(
        sampled["total_counts"],
        sampled["pct_mito"],
        gridsize=65,
        xscale="log",
        bins="log",
        mincnt=1,
        cmap="mako",
    )
    axes[1].set_ylim(0, robust_upper(frame["pct_mito"]))
    axes[1].set_xlabel("Transcripts per cell")
    axes[1].set_ylabel("Mitochondrial fraction (%)")
    axes[1].set_title("Library size vs mitochondrial fraction")
    cbar2 = fig.colorbar(hexplot, ax=axes[1], pad=0.02)
    cbar2.set_label("log10(cell density)", fontsize=10)
    fig.suptitle("QC relationships identify low-complexity and high-mitochondrial tails", fontsize=18, color=NAVY, y=1.03)
    fig.text(0.01, -0.02, f"Reproducible display sample: {len(sampled):,} cells. Points are visualization-only; no QC cutoff is proposed.", fontsize=9, color=GREY)
    fig.tight_layout()
    save_figure(fig, "03_qc_metric_relationships", out, formats, dpi)


def figure_recovery_and_composition(frame: pd.DataFrame, out: Path, formats: list[str], dpi: int) -> None:
    donor_order = natural_order(frame["donor"].dropna().astype(str).unique())
    counts = frame.groupby("donor", observed=True).size().reindex(donor_order)
    composition = pd.crosstab(frame["donor"], frame["cell_type"], normalize="index").reindex(donor_order)
    composition = composition[composition.mean(axis=0).sort_values(ascending=False).index]

    fig, axes = plt.subplots(1, 2, figsize=(13.333, 5.2), gridspec_kw={"width_ratios": [0.9, 1.6]})
    bars = axes[0].bar(counts.index.astype(str), counts.values, color=BLUE)
    axes[0].set_title("Recovered cells by donor")
    axes[0].set_ylabel("Cells")
    axes[0].tick_params(axis="x", rotation=45, labelsize=9)
    for bar, value in zip(bars, counts.values):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height(), format_count(value), ha="center", va="bottom", fontsize=8, color=NAVY)

    palette = sns.color_palette("tab20", n_colors=max(3, composition.shape[1]))
    bottom = np.zeros(len(composition))
    for color, cell_type in zip(palette, composition.columns):
        values = composition[cell_type].to_numpy() * 100
        axes[1].bar(composition.index.astype(str), values, bottom=bottom, label=str(cell_type), color=color, width=0.78)
        bottom += values
    axes[1].set_title("Cell-type composition by donor")
    axes[1].set_ylabel("Cell share (%)")
    axes[1].set_ylim(0, 100)
    axes[1].tick_params(axis="x", rotation=45, labelsize=9)
    axes[1].legend(title="Cell type", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7, title_fontsize=8, ncol=1)
    fig.suptitle("Donor balance and immune composition determine benchmark representation", fontsize=18, color=NAVY, y=1.03)
    fig.tight_layout()
    save_figure(fig, "04_donor_recovery_and_cell_type_composition", out, formats, dpi)


def cytokine_order(values: pd.Series) -> list[str]:
    labels = sorted(values.dropna().astype(str).unique())
    controls = [x for x in labels if x.upper() in CONTROL_NAMES]
    return controls + [x for x in labels if x not in controls]


def control_labels(values: pd.Series) -> list[str]:
    return [label for label in cytokine_order(values) if label.upper() in CONTROL_NAMES]


def figure_condition_coverage(frame: pd.DataFrame, out: Path, formats: list[str], dpi: int) -> None:
    donors = natural_order(frame["donor"].dropna().astype(str).unique())
    controls = set(control_labels(frame["cytokine"]))
    counts = (
        frame.groupby(["donor", "cytokine"], observed=True)
        .size()
        .rename("n_cells")
        .reset_index()
        .assign(donor=lambda x: x["donor"].astype(str), cytokine=lambda x: x["cytokine"].astype(str))
    )
    treated = counts.loc[~counts["cytokine"].isin(controls)].copy()
    pbs = counts.loc[counts["cytokine"].isin(controls)].groupby("donor", as_index=False)["n_cells"].sum()
    expected = len(donors) * frame.loc[~frame["cytokine"].isin(controls), "cytokine"].nunique()

    fig, ax = plt.subplots(figsize=(13.333, 5.4))
    sns.boxplot(
        data=treated,
        x="donor",
        y="n_cells",
        order=donors,
        color=TEAL,
        width=0.66,
        showfliers=False,
        linewidth=0.9,
        ax=ax,
    )
    pbs_ordered = pbs.set_index("donor").reindex(donors)
    ax.scatter(
        np.arange(len(donors)),
        pbs_ordered["n_cells"],
        color=ORANGE,
        marker="D",
        s=58,
        label="PBS control",
        zorder=4,
    )
    ax.set_yscale("log")
    ax.set_xlabel("Donor")
    ax.set_ylabel("Cells per donor × condition (log scale)")
    ax.tick_params(axis="x", rotation=35, labelsize=9)
    ax.set_title(
        f"Perturbed-condition depth varies, while PBS controls are deliberately deeper\n"
        f"{len(treated):,}/{expected:,} donor × perturbed-cytokine conditions observed",
        fontsize=17,
        color=NAVY,
        pad=15,
    )
    ax.legend(loc="upper right", fontsize=9)
    fig.text(
        0.01,
        -0.01,
        "Boxes summarize perturbed cytokines within each donor; diamonds show pooled control depth. PBS is excluded from the perturbation denominator.",
        fontsize=9,
        color=GREY,
    )
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    save_figure(fig, "05_donor_by_cytokine_condition_coverage", out, formats, dpi)


def stratum_table(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.groupby(["donor", "cytokine", "cell_type"], observed=True)
        .size()
        .rename("n_cells")
        .reset_index()
    )


def build_benchmark_coverage(
    frame: pd.DataFrame,
    strata: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Compare nested benchmark-eligibility definitions without filtering data.

    Candidate strata are all donor × perturbed-cytokine × cell-type
    combinations, including zero-count combinations. At each threshold, the
    perturbed-sufficient cohort requires only the perturbed stratum to meet the
    count reference; the matched-control cohort additionally requires the
    donor × cell-type control pool to meet it. Control cells are summarized as
    unique donor × cell-type pools so one PBS pool is never double-counted for
    every cytokine it supports.
    """
    donors = natural_order(frame["donor"].dropna().astype(str).unique())
    cell_types = sorted(frame["cell_type"].dropna().astype(str).unique())
    controls = control_labels(frame["cytokine"])
    perturbations = [x for x in cytokine_order(frame["cytokine"]) if x not in controls]

    normalized = strata.assign(
        donor=strata["donor"].astype(str),
        cytokine=strata["cytokine"].astype(str),
        cell_type=strata["cell_type"].astype(str),
    )
    perturbed_index = pd.MultiIndex.from_product(
        [donors, perturbations, cell_types],
        names=["donor", "cytokine", "cell_type"],
    )
    perturbed = (
        normalized.loc[~normalized["cytokine"].isin(controls)]
        .set_index(["donor", "cytokine", "cell_type"])["n_cells"]
        .reindex(perturbed_index, fill_value=0)
        .rename("n_perturbed")
        .reset_index()
    )
    matched_control = (
        normalized.loc[normalized["cytokine"].isin(controls)]
        .groupby(["donor", "cell_type"], observed=True)["n_cells"]
        .sum()
        .rename("n_matched_control")
    )
    perturbed = perturbed.join(matched_control, on=["donor", "cell_type"])
    perturbed["n_matched_control"] = perturbed["n_matched_control"].fillna(0).astype(int)
    perturbed["n_perturbed"] = perturbed["n_perturbed"].astype(int)
    for threshold in BENCHMARK_THRESHOLDS:
        perturbed[f"perturbed_sufficient_ge_{threshold}"] = (
            perturbed["n_perturbed"] >= threshold
        )
        perturbed[f"matched_control_ge_{threshold}"] = (
            perturbed[f"perturbed_sufficient_ge_{threshold}"]
            & (perturbed["n_matched_control"] >= threshold)
        )

    total_candidate_strata = len(perturbed)
    total_perturbed_cells = int(perturbed["n_perturbed"].sum())
    cohort_rows: list[dict[str, Any]] = []
    impact_rows: list[dict[str, Any]] = []
    for threshold in BENCHMARK_THRESHOLDS:
        definitions = {
            "All candidate strata": pd.Series(True, index=perturbed.index),
            "Perturbed sufficient": perturbed[f"perturbed_sufficient_ge_{threshold}"],
            "Matched control": perturbed[f"matched_control_ge_{threshold}"],
        }
        for cohort, eligible in definitions.items():
            retained_cells = int(perturbed.loc[eligible, "n_perturbed"].sum())
            cohort_rows.append(
                {
                    "threshold": int(threshold),
                    "cohort": cohort,
                    "retained_candidate_strata": int(eligible.sum()),
                    "total_candidate_strata": total_candidate_strata,
                    "fraction_candidate_strata_retained": float(eligible.mean()),
                    "retained_perturbed_cells": retained_cells,
                    "total_perturbed_cells": total_perturbed_cells,
                    "fraction_perturbed_cells_retained": (
                        retained_cells / total_perturbed_cells if total_perturbed_cells else np.nan
                    ),
                }
            )
        pert_ok = definitions["Perturbed sufficient"]
        matched_ok = definitions["Matched control"]
        impact_rows.append(
            {
                "threshold": int(threshold),
                "strata_lost_due_to_perturbed_depth": int((~pert_ok).sum()),
                "strata_additionally_lost_due_to_matched_control": int((pert_ok & ~matched_ok).sum()),
                "perturbed_cells_lost_due_to_perturbed_depth": int(
                    perturbed.loc[~pert_ok, "n_perturbed"].sum()
                ),
                "perturbed_cells_additionally_lost_due_to_matched_control": int(
                    perturbed.loc[pert_ok & ~matched_ok, "n_perturbed"].sum()
                ),
            }
        )

    control_pools = (
        perturbed[["donor", "cell_type", "n_matched_control"]]
        .drop_duplicates(["donor", "cell_type"])
        .reset_index(drop=True)
    )
    total_control_cells = int(control_pools["n_matched_control"].sum())
    control_rows: list[dict[str, Any]] = []
    for threshold in BENCHMARK_THRESHOLDS:
        sufficient = control_pools["n_matched_control"] >= threshold
        retained_control_cells = int(control_pools.loc[sufficient, "n_matched_control"].sum())
        control_rows.append(
            {
                "threshold": int(threshold),
                "sufficient_control_pools": int(sufficient.sum()),
                "total_control_pools": int(len(control_pools)),
                "fraction_control_pools_sufficient": float(sufficient.mean()),
                "cells_in_sufficient_control_pools": retained_control_cells,
                "total_control_cells": total_control_cells,
                "fraction_control_cells_in_sufficient_pools": (
                    retained_control_cells / total_control_cells if total_control_cells else np.nan
                ),
            }
        )

    def summarize(group: str | None) -> pd.DataFrame:
        groups: list[tuple[str, pd.DataFrame]]
        if group is None:
            groups = [("all", perturbed)]
        else:
            groups = [(str(name), subset) for name, subset in perturbed.groupby(group, observed=True)]
        rows: list[dict[str, Any]] = []
        for name, subset in groups:
            for threshold in BENCHMARK_THRESHOLDS:
                perturbed_ok = subset[f"perturbed_sufficient_ge_{threshold}"]
                matched_ok = subset[f"matched_control_ge_{threshold}"]
                row: dict[str, Any] = {
                    "threshold": int(threshold),
                    "total_possible_strata": int(len(subset)),
                    "perturbed_sufficient_strata": int(perturbed_ok.sum()),
                    "fraction_perturbed_sufficient": float(perturbed_ok.mean()),
                    "matched_control_strata": int(matched_ok.sum()),
                    "fraction_matched_control": float(matched_ok.mean()),
                    "incremental_strata_lost_to_control_requirement": int(
                        (perturbed_ok & ~matched_ok).sum()
                    ),
                    "perturbed_cells_retained_perturbed_sufficient": int(
                        subset.loc[perturbed_ok, "n_perturbed"].sum()
                    ),
                    "perturbed_cells_retained_matched_control": int(
                        subset.loc[matched_ok, "n_perturbed"].sum()
                    ),
                    "total_perturbed_cells": int(subset["n_perturbed"].sum()),
                    "zero_perturbed_strata": int((subset["n_perturbed"] == 0).sum()),
                    "strata_below_perturbed_threshold": int((subset["n_perturbed"] < threshold).sum()),
                    "strata_below_control_threshold": int((subset["n_matched_control"] < threshold).sum()),
                }
                if group is not None:
                    row[group] = name
                rows.append(row)
        return pd.DataFrame(rows)

    return {
        "benchmark_stratum_eligibility": perturbed,
        "cohort_retention_by_threshold": pd.DataFrame(cohort_rows),
        "threshold_filtering_impact": pd.DataFrame(impact_rows),
        "matched_control_pool_sensitivity": pd.DataFrame(control_rows),
        "benchmark_coverage_overall": summarize(None),
        "benchmark_coverage_by_cell_type": summarize("cell_type"),
        "benchmark_coverage_by_cytokine": summarize("cytokine"),
        "benchmark_coverage_by_donor": summarize("donor"),
    }


def figure_stratum_coverage(
    coverage: dict[str, pd.DataFrame],
    out: Path,
    formats: list[str],
    dpi: int,
) -> None:
    strata = coverage["benchmark_stratum_eligibility"]
    retention = coverage["cohort_retention_by_threshold"]

    fig, axes = plt.subplots(1, 2, figsize=(13.333, 6.4), gridspec_kw={"width_ratios": [0.9, 1.35]})
    positive = strata.loc[strata["n_perturbed"] > 0, "n_perturbed"].to_numpy()
    max_display = max(500.0, float(np.quantile(positive, 0.995))) if len(positive) else 500.0
    bins = np.geomspace(1, max_display, 65)
    axes[0].hist(positive[positive <= max_display], bins=bins, color=TEAL, alpha=0.9, edgecolor="white", linewidth=0.25)
    axes[0].set_xscale("log")
    axes[0].axvline(30, color=RED, linestyle="--", linewidth=2, label="30-cell reference")
    axes[0].set_title("Perturbed cells per candidate stratum")
    axes[0].set_xlabel("Perturbed cells per candidate stratum")
    axes[0].set_ylabel("Strata")
    axes[0].legend(fontsize=9)
    zero_count = int((strata["n_perturbed"] == 0).sum())
    axes[0].text(
        0.98,
        0.93,
        f"{zero_count:,} zero-count strata\nomitted from log scale",
        transform=axes[0].transAxes,
        ha="right",
        va="top",
        fontsize=9,
        color=NAVY,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#DCE5EA"},
    )

    colors = {"Perturbed sufficient": TEAL, "Matched control": ORANGE}
    markers = {"Perturbed sufficient": "o", "Matched control": "s"}
    for cohort in ("Perturbed sufficient", "Matched control"):
        subset = retention.loc[retention["cohort"] == cohort].sort_values("threshold")
        axes[1].plot(
            subset["threshold"],
            subset["fraction_candidate_strata_retained"] * 100,
            color=colors[cohort],
            marker=markers[cohort],
            linewidth=2.3,
            markersize=7,
            label=f"{cohort}: strata",
        )
        axes[1].plot(
            subset["threshold"],
            subset["fraction_perturbed_cells_retained"] * 100,
            color=colors[cohort],
            marker=markers[cohort],
            linewidth=1.8,
            linestyle="--",
            alpha=0.9,
            label=f"{cohort}: cells",
        )
    axes[1].set_ylim(0, 103)
    axes[1].set_xticks(BENCHMARK_THRESHOLDS)
    axes[1].set_xlabel("Minimum cells per required stratum")
    axes[1].set_ylabel("Retained (%)")
    axes[1].set_title("Threshold sensitivity by cohort definition")
    axes[1].grid(axis="y", color="#DCE5EA", linewidth=0.7)
    axes[1].legend(fontsize=8, ncol=2, loc="lower left")
    axes[1].text(
        0.98,
        0.04,
        "Solid = candidate strata\nDashed = unique perturbed cells",
        transform=axes[1].transAxes,
        ha="right",
        va="bottom",
        fontsize=8.5,
        color=GREY,
    )
    fig.suptitle("Coverage conclusions depend on both cell threshold and control requirement", fontsize=18, color=NAVY, y=1.01)
    fig.text(0.01, -0.02, "PBS is excluded from candidate perturbation strata. Perturbed-cell retention counts each cell once; reused donor × cell-type PBS pools are summarized separately in the tables.", fontsize=9, color=GREY)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    save_figure(fig, "06_donor_cytokine_cell_type_stratum_coverage", out, formats, dpi)


def figure_cohort_coverage_breakdown(
    coverage: dict[str, pd.DataFrame],
    out: Path,
    formats: list[str],
    dpi: int,
) -> None:
    threshold = 30
    by_cell = coverage["benchmark_coverage_by_cell_type"].query("threshold == @threshold").sort_values("fraction_matched_control")
    by_cytokine = coverage["benchmark_coverage_by_cytokine"].query("threshold == @threshold")
    by_donor = coverage["benchmark_coverage_by_donor"].query("threshold == @threshold").copy()
    donor_order = natural_order(by_donor["donor"])
    by_donor["donor"] = pd.Categorical(by_donor["donor"], categories=donor_order, ordered=True)
    by_donor = by_donor.sort_values("donor")

    fig, axes = plt.subplots(1, 3, figsize=(15.4, 6.2), gridspec_kw={"width_ratios": [1.35, 0.9, 1.05]})
    y = np.arange(len(by_cell))
    axes[0].barh(y + 0.18, by_cell["fraction_perturbed_sufficient"] * 100, height=0.34, color=TEAL, label="Perturbed sufficient")
    axes[0].barh(y - 0.18, by_cell["fraction_matched_control"] * 100, height=0.34, color=ORANGE, label="Matched control")
    axes[0].set_yticks(y, by_cell["cell_type"])
    axes[0].set_xlim(0, 100)
    axes[0].set_xlabel("Candidate strata retained (%)")
    axes[0].set_title("Cell type")
    axes[0].tick_params(axis="y", labelsize=8)
    axes[0].legend(fontsize=8, loc="lower right")

    bins = np.linspace(0, 100, 18)
    axes[1].hist(by_cytokine["fraction_perturbed_sufficient"] * 100, bins=bins, color=TEAL, alpha=0.55, label="Perturbed sufficient")
    axes[1].hist(by_cytokine["fraction_matched_control"] * 100, bins=bins, color=ORANGE, alpha=0.55, label="Matched control")
    axes[1].set_xlabel("Retained strata per cytokine (%)")
    axes[1].set_ylabel("Cytokines")
    axes[1].set_title("Perturbation distribution")
    axes[1].legend(fontsize=8)

    x = np.arange(len(by_donor))
    axes[2].bar(x - 0.18, by_donor["fraction_perturbed_sufficient"] * 100, width=0.36, color=TEAL, label="Perturbed sufficient")
    axes[2].bar(x + 0.18, by_donor["fraction_matched_control"] * 100, width=0.36, color=ORANGE, label="Matched control")
    axes[2].set_ylim(0, 100)
    axes[2].set_xlabel("Donor")
    axes[2].set_ylabel("Candidate strata retained (%)")
    axes[2].set_title("Donor")
    axes[2].set_xticks(x, by_donor["donor"])
    axes[2].tick_params(axis="x", rotation=40, labelsize=8)

    overall = coverage["benchmark_coverage_overall"].query("threshold == @threshold").iloc[0]
    fig.suptitle(
        f"At ≥30 cells: {overall['perturbed_sufficient_strata']:,.0f} perturbed-sufficient vs "
        f"{overall['matched_control_strata']:,.0f} matched-control strata "
        f"({overall['incremental_strata_lost_to_control_requirement']:,.0f} additional lost)",
        fontsize=17,
        color=NAVY,
        y=1.02,
    )
    fig.text(0.01, -0.01, "The gap between teal and orange isolates the incremental coverage cost of requiring a donor × cell-type PBS control; PBS itself is excluded from every perturbation denominator.", fontsize=9, color=GREY)
    fig.tight_layout(rect=(0, 0.02, 1, 0.98))
    save_figure(fig, "09_cohort_coverage_breakdown_at_30_cells", out, formats, dpi)


def figure_qc_by_cell_type(frame: pd.DataFrame, out: Path, formats: list[str], dpi: int) -> pd.DataFrame:
    medians = (
        frame.groupby("cell_type", observed=True)[["total_counts", "n_genes", "pct_mito"]]
        .median()
        .rename(
            columns={
                "total_counts": "Median transcripts",
                "n_genes": "Median detected genes",
                "pct_mito": "Median mitochondrial %",
            }
        )
    )
    medians["Cells"] = frame.groupby("cell_type", observed=True).size()
    metric_columns = ["Median transcripts", "Median detected genes", "Median mitochondrial %"]
    def robust_zscore(values: pd.Series) -> pd.Series:
        center = values.median()
        mad = (values - center).abs().median() * 1.4826
        scale = mad if np.isfinite(mad) and mad > 1e-9 else values.std(ddof=0)
        return (values - center) / max(float(scale), 1e-9)

    z = medians[metric_columns].apply(robust_zscore, axis=0)
    z = z.loc[medians["Cells"].sort_values(ascending=False).index]
    z_display = z.clip(-2.5, 2.5)

    fig, ax = plt.subplots(figsize=(9.2, 7.2))
    sns.heatmap(
        z_display,
        cmap="vlag",
        center=0,
        vmin=-2.5,
        vmax=2.5,
        linewidths=0.5,
        linecolor="white",
        annot=True,
        fmt=".1f",
        annot_kws={"fontsize": 8},
        cbar_kws={"label": "Robust z-score (clipped at ±2.5)"},
        ax=ax,
    )
    ax.set_title("Cell types differ in expected RNA content and mitochondrial fraction", fontsize=17, color=NAVY, pad=16)
    ax.set_xlabel("")
    ax.set_ylabel("Cell type (ordered by abundance)")
    ax.tick_params(axis="x", labelsize=9, rotation=20)
    ax.tick_params(axis="y", labelsize=8, rotation=0)
    fig.text(0.01, -0.02, "Values are standardized medians across cell types, so differences are descriptive and should not be interpreted as universal QC failures.", fontsize=9, color=GREY)
    fig.tight_layout()
    save_figure(fig, "07_qc_profile_by_cell_type", out, formats, dpi)
    return medians.reset_index()


def figure_control_comparison(frame: pd.DataFrame, out: Path, formats: list[str], dpi: int) -> pd.DataFrame | None:
    control_labels = {
        str(label)
        for label in frame["cytokine"].cat.categories
        if str(label).upper() in {"PBS", "CONTROL", "UNTREATED", "VEHICLE"}
    }
    control_mask = frame["cytokine"].isin(control_labels)
    if not bool(control_mask.any()) or bool(control_mask.all()):
        return None
    temp = frame[["donor", "cell_type", "total_counts", "n_genes", "pct_mito"]].copy()
    temp["arm"] = pd.Categorical(np.where(control_mask, "PBS control", "Cytokine-treated"))
    grouped = (
        temp.groupby(["donor", "cell_type", "arm"], observed=True)[["total_counts", "n_genes", "pct_mito"]]
        .median()
        .unstack("arm")
    )
    required = [(metric, arm) for metric in ["total_counts", "n_genes", "pct_mito"] for arm in ["PBS control", "Cytokine-treated"]]
    if not all(column in grouped.columns for column in required):
        return None
    paired = pd.DataFrame(index=grouped.index)
    paired["Transcripts log2 ratio"] = np.log2(
        (grouped[("total_counts", "Cytokine-treated")] + 1)
        / (grouped[("total_counts", "PBS control")] + 1)
    )
    paired["Detected genes log2 ratio"] = np.log2(
        (grouped[("n_genes", "Cytokine-treated")] + 1)
        / (grouped[("n_genes", "PBS control")] + 1)
    )
    paired["Mitochondrial percentage-point difference"] = (
        grouped[("pct_mito", "Cytokine-treated")] - grouped[("pct_mito", "PBS control")]
    )
    long = paired.reset_index().melt(id_vars=["donor", "cell_type"], var_name="contrast", value_name="value")
    fig, axes = plt.subplots(1, 3, figsize=(13.333, 4.8))
    contrast_order = list(paired.columns)
    colors = [TEAL, BLUE, ORANGE]
    for ax, contrast, color in zip(axes, contrast_order, colors):
        values = paired[contrast].dropna()
        sns.boxplot(y=values, ax=ax, color=color, width=0.45, showfliers=True, fliersize=2)
        sns.stripplot(y=values, ax=ax, color=NAVY, alpha=0.35, size=3, jitter=0.16)
        ax.axhline(0, color=GREY, linewidth=1.2, linestyle="--")
        ax.set_title(contrast.replace(" log2 ratio", "").replace(" percentage-point difference", ""), fontsize=13)
        ax.set_xlabel("")
        ax.set_ylabel("log2 ratio" if "ratio" in contrast else "Percentage-point difference")
        ax.set_xticks([])
    fig.suptitle("Released QC profiles are checked for systematic control–treatment shifts", fontsize=18, color=NAVY, y=1.03)
    fig.text(0.01, -0.02, "Each point is one donor × cell-type pair. Cytokine biology can alter RNA content, so this is a diagnostic rather than a pass/fail test.", fontsize=9, color=GREY)
    fig.tight_layout()
    save_figure(fig, "08_pbs_vs_perturbed_qc_diagnostic", out, formats, dpi)
    return long


def write_tables(
    frame: pd.DataFrame,
    strata: pd.DataFrame,
    coverage: dict[str, pd.DataFrame],
    cell_type_qc: pd.DataFrame,
    control_comparison: pd.DataFrame | None,
    tables_dir: Path,
) -> dict[str, pd.DataFrame]:
    controls = set(control_labels(frame["cytokine"]))
    condition_counts = frame.groupby(["donor", "cytokine"], observed=True).size().rename("n_cells").reset_index()
    condition_counts["donor"] = condition_counts["donor"].astype(str)
    condition_counts["cytokine"] = condition_counts["cytokine"].astype(str)
    donor_cell_type_counts = (
        frame.groupby(["donor", "cell_type"], observed=True)
        .size()
        .rename("n_cells")
        .reset_index()
    )
    donor_cell_type_counts["fraction_within_donor"] = (
        donor_cell_type_counts["n_cells"]
        / donor_cell_type_counts.groupby("donor", observed=True)["n_cells"].transform("sum")
    )
    summaries = {
        "metric_summary": metric_summary(frame),
        "qc_boundary_diagnostics": qc_boundary_diagnostics(frame),
        "cell_counts_by_donor": frame.groupby("donor", observed=True).size().rename("n_cells").reset_index(),
        "cell_counts_by_cell_type": frame.groupby("cell_type", observed=True).size().rename("n_cells").reset_index(),
        "cell_counts_by_cytokine": frame.groupby("cytokine", observed=True).size().rename("n_cells").reset_index(),
        "condition_counts_donor_by_cytokine": condition_counts,
        "perturbed_condition_counts_donor_by_cytokine": condition_counts.loc[~condition_counts["cytokine"].isin(controls)].reset_index(drop=True),
        "control_counts_by_donor": condition_counts.loc[condition_counts["cytokine"].isin(controls)].groupby("donor", as_index=False)["n_cells"].sum(),
        "donor_cell_type_composition": donor_cell_type_counts,
        "qc_medians_by_donor": (
            frame.groupby("donor", observed=True)[["total_counts", "n_genes", "pct_mito"]]
            .median()
            .reset_index()
        ),
        "stratum_counts_donor_by_cytokine_by_cell_type": strata,
        "qc_medians_by_cell_type": cell_type_qc,
        **coverage,
    }
    if control_comparison is not None:
        summaries["pbs_vs_perturbed_qc_by_donor_cell_type"] = control_comparison
    for name, table in summaries.items():
        table.to_csv(tables_dir / f"{name}.csv", index=False)
    return summaries


def dataframe_html(table: pd.DataFrame, max_rows: int = 30) -> str:
    return table.head(max_rows).to_html(index=False, border=0, classes="dataframe", float_format=lambda x: f"{x:,.2f}")


def write_report(
    output: Path,
    frame: pd.DataFrame,
    provenance: dict[str, Any],
    tables: dict[str, pd.DataFrame],
) -> None:
    metric = tables["metric_summary"].copy()
    donor_counts = tables["cell_counts_by_donor"]
    cell_type_counts = tables["cell_counts_by_cell_type"].sort_values("n_cells", ascending=False)
    cytokine_counts = tables["cell_counts_by_cytokine"]
    perturbed_conditions = tables["perturbed_condition_counts_donor_by_cytokine"]
    control_counts = tables["control_counts_by_donor"]
    overall_30 = tables["benchmark_coverage_overall"].query("threshold == 30").iloc[0]
    cohort_30 = tables["cohort_retention_by_threshold"].query("threshold == 30")
    perturbed_30 = cohort_30.loc[cohort_30["cohort"] == "Perturbed sufficient"].iloc[0]
    matched_30 = cohort_30.loc[cohort_30["cohort"] == "Matched control"].iloc[0]
    impact_30 = tables["threshold_filtering_impact"].query("threshold == 30").iloc[0]
    zero_perturbed_30 = int(overall_30["zero_perturbed_strata"])
    missingness = frame.isna().sum().rename("missing_n").rename_axis("field").reset_index()

    key_findings = [
        f"The released H5AD contains {len(frame):,} cells × {provenance['n_vars']:,} features.",
        f"Coverage spans {frame['donor'].nunique(dropna=True)} donors, {frame['cytokine'].nunique(dropna=True)} cytokine labels (including control), and {frame['cell_type'].nunique(dropna=True)} annotated cell types.",
        f"Across perturbed conditions only, median donor × cytokine size is {perturbed_conditions['n_cells'].median():,.0f} cells (range {perturbed_conditions['n_cells'].min():,}–{perturbed_conditions['n_cells'].max():,}); median pooled control depth is {control_counts['n_cells'].median():,.0f} cells per donor.",
        f"At the 30-cell reference, perturbed-depth eligibility retains {int(perturbed_30['retained_candidate_strata']):,} candidate strata ({perturbed_30['fraction_candidate_strata_retained']:.1%}) and {perturbed_30['fraction_perturbed_cells_retained']:.1%} of perturbed cells.",
        f"Adding the matched donor × cell-type control requirement retains {int(matched_30['retained_candidate_strata']):,} strata ({matched_30['fraction_candidate_strata_retained']:.1%}) and {matched_30['fraction_perturbed_cells_retained']:.1%} of perturbed cells; this requirement additionally removes {int(impact_30['strata_additionally_lost_due_to_matched_control']):,} strata.",
        f"Control conditions are excluded from candidate perturbation strata; {zero_perturbed_30:,} candidate strata have zero recovered perturbed cells and remain visible in the denominator.",
        "This is the vendor-released, already low-quality-cell-filtered H5AD; it is original relative to STATE preprocessing, not unfiltered FASTQ-level data.",
        "The audit is descriptive: it proposes no cell-removal thresholds and makes no changes to the source file.",
    ]

    figure_items = sorted((output / "figures").glob("*.png"))
    cards = "\n".join(
        f'<section class="figure"><h3>{html.escape(path.stem.replace("_", " ").title())}</h3><img src="figures/{html.escape(path.name)}" alt="{html.escape(path.stem)}"></section>'
        for path in figure_items
    )
    html_text = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Parse-PBMC original dataset QC audit</title>
<style>
body{{font-family:Arial,Helvetica,sans-serif;color:#183B56;margin:0;background:#F5F8FA;line-height:1.45}}
main{{max-width:1320px;margin:0 auto;padding:42px}} h1{{font-size:34px;margin-bottom:4px}} h2{{margin-top:34px}}
.subtitle{{color:#657786;margin-top:0}} .callout{{background:#EAF4F4;border-left:6px solid #2A9D8F;padding:18px 24px;border-radius:8px}}
.figure{{background:white;padding:18px 22px;margin:24px 0;border-radius:10px;box-shadow:0 3px 14px rgba(24,59,86,.08)}}
.figure img{{width:100%;height:auto;display:block}} table{{border-collapse:collapse;width:100%;background:white;font-size:13px}}
th,td{{padding:8px 10px;border-bottom:1px solid #DFE7EC;text-align:right}} th:first-child,td:first-child{{text-align:left}} th{{background:#183B56;color:white}}
code{{background:#E9EFF3;padding:2px 5px;border-radius:4px}} footer{{color:#657786;font-size:12px;margin-top:40px}}
</style></head><body><main>
<h1>Parse-PBMC original dataset: descriptive QC audit</h1>
<p class="subtitle">Presentation-oriented figures for the vendor-released 10M-cell H5AD; generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}.</p>
<div class="callout"><strong>Scope:</strong> read-only assessment of released-cell QC and benchmark representation. No normalization, HVG selection, doublet removal, cell filtering, matrix rewrite, or STATE preprocessing was performed.</div>
<h2>Key takeaways</h2><ul>{''.join(f'<li>{html.escape(x)}</li>' for x in key_findings)}</ul>
<h2>Core QC metrics</h2>{dataframe_html(metric)}
<h2>Missing metadata/QC values</h2>{dataframe_html(missingness)}
<h2>Cell-type representation</h2>{dataframe_html(cell_type_counts)}
<h2>Cohort retention at 30 cells</h2>{dataframe_html(cohort_30)}
<h2>Threshold sensitivity</h2>{dataframe_html(tables['cohort_retention_by_threshold'], max_rows=20)}
<h2>Coverage by cell type at 30 cells</h2>{dataframe_html(tables['benchmark_coverage_by_cell_type'].query('threshold == 30').sort_values('fraction_matched_control', ascending=False))}
<h2>Presentation figures</h2>{cards}
<h2>Interpretation boundaries</h2>
<ul><li>The released file already underwent Parse Analysis Pipeline v1.4.0 processing and low-quality-cell filtering.</li>
<li>Mitochondrial fraction, detected genes, and transcript counts are standard descriptive QC indicators, but this audit does not define pass/fail cutoffs.</li>
<li>Cell types naturally differ in RNA content; cell-type-specific shifts should not be interpreted as technical failure without further evidence.</li>
<li>Cytokine treatment can cause real transcriptional shifts. The PBS comparison is a diagnostic for systematic differences, not a QC decision rule.</li>
<li>The supplied metadata do not include a doublet score or ambient-RNA score, so those failure modes cannot be directly quantified from stored obs columns.</li></ul>
<h2>Sources</h2>
<ul><li><a href="https://www.parsebiosciences.com/datasets/10-million-human-pbmcs-in-a-single-experiment/">Parse Biosciences: 10 Million Human PBMCs in a Single Experiment</a></li>
<li>STATE manuscript supplied with this project, Section 4.5.1 (dataset preprocessing).</li></ul>
<footer>Input: <code>{html.escape(provenance['input_path'])}</code>. Expression matrix read: {provenance['matrix_read']}. Input modified: {provenance['input_modified']}.</footer>
</main></body></html>"""
    (output / "parse_pbmc_original_qc_report.html").write_text(html_text, encoding="utf-8")

    markdown = "# Parse-PBMC original dataset: descriptive QC audit\n\n"
    markdown += "## Scope\n\nRead-only assessment of released-cell QC and benchmark representation. No normalization, HVG selection, doublet removal, cell filtering, matrix rewrite, or STATE preprocessing was performed.\n\n"
    markdown += "## Key takeaways\n\n" + "\n".join(f"- {item}" for item in key_findings) + "\n\n"
    markdown += "## Figure guide\n\n"
    markdown += "1. Single-cell QC distributions: overall transcript, detected-gene, and mitochondrial-fraction distributions.\n"
    markdown += "2. QC by donor: checks whether one donor has a systematically shifted library profile.\n"
    markdown += "3. QC metric relationships: highlights low-complexity and high-mitochondrial tails without setting cutoffs.\n"
    markdown += "4. Donor recovery and composition: shows total representation and cell-type balance.\n"
    markdown += "5. Donor × condition depth: separates perturbed-condition distributions from deliberately deeper PBS controls.\n"
    markdown += "6. Cohort and threshold sensitivity: compares perturbed-only and matched-control retention for both strata and unique perturbed cells.\n"
    markdown += "7. QC by cell type: separates expected biological differences from global QC shifts.\n"
    markdown += "8. PBS vs perturbed diagnostic: checks treatment-associated QC shifts while acknowledging biological effects.\n\n"
    markdown += "9. Cohort coverage breakdown at 30 cells: shows the incremental effect of matched controls by cell type, cytokine, and donor.\n\n"
    markdown += "## Interpretation boundaries\n\n"
    markdown += "- The vendor-released H5AD was already filtered for low-quality cells by Parse. It is original relative to STATE, not unfiltered sequencing output.\n"
    markdown += "- The script proposes no QC-removal thresholds.\n"
    markdown += "- Doublet and ambient-RNA scores are unavailable in the stored metadata and therefore not directly assessed.\n"
    (output / "qc_summary.md").write_text(markdown, encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    output = args.output.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Input H5AD not found: {input_path}")
    if input_path == output or str(output).startswith(str(input_path) + "/"):
        raise ValueError("Output must be a directory separate from the input file.")

    figures_dir = output / "figures"
    tables_dir = output / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    configure_style()

    print(f"[{datetime.now().isoformat(timespec='seconds')}] Reading selected obs columns from {input_path}", flush=True)
    frame, provenance = load_metadata(input_path)
    frame = clean_categories(frame)
    provenance.update(
        {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "python": sys.version,
            "platform": platform.platform(),
            "package_versions": {
                "anndata": version("anndata"),
                "h5py": h5py.__version__,
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "matplotlib": mpl.__version__,
                "seaborn": sns.__version__,
            },
            "arguments": {
                "seed": args.seed,
                "scatter_sample": args.scatter_sample,
                "violin_sample_per_group": args.violin_sample_per_group,
                "dpi": args.dpi,
                "formats": args.formats,
            },
            "workflow_version": WORKFLOW_VERSION,
            "benchmark_eligibility_definitions": {
                "all_candidate_strata": "all donor × perturbed-cytokine × cell-type combinations, including zero-count combinations",
                "perturbed_sufficient": "perturbed donor × cytokine × cell type has at least threshold cells",
                "matched_control": "perturbed-sufficient AND matched donor × cell-type control has at least threshold cells",
                "denominator_note": "control conditions excluded from candidate perturbation strata",
            },
        }
    )
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")

    print(f"[{datetime.now().isoformat(timespec='seconds')}] Creating presentation figures", flush=True)
    figure_qc_distributions(frame, figures_dir, args.formats, args.dpi)
    figure_qc_by_donor(frame, figures_dir, args.formats, args.dpi, args.violin_sample_per_group, args.seed)
    figure_metric_relationships(frame, figures_dir, args.formats, args.dpi, args.scatter_sample, args.seed)
    figure_recovery_and_composition(frame, figures_dir, args.formats, args.dpi)
    figure_condition_coverage(frame, figures_dir, args.formats, args.dpi)
    strata = stratum_table(frame)
    coverage = build_benchmark_coverage(frame, strata)
    figure_stratum_coverage(coverage, figures_dir, args.formats, args.dpi)
    cell_type_qc = figure_qc_by_cell_type(frame, figures_dir, args.formats, args.dpi)
    control_comparison = figure_control_comparison(frame, figures_dir, args.formats, args.dpi)
    figure_cohort_coverage_breakdown(coverage, figures_dir, args.formats, args.dpi)
    tables = write_tables(frame, strata, coverage, cell_type_qc, control_comparison, tables_dir)
    write_report(output, frame, provenance, tables)

    print(f"[{datetime.now().isoformat(timespec='seconds')}] Complete: {output}", flush=True)


if __name__ == "__main__":
    main()
