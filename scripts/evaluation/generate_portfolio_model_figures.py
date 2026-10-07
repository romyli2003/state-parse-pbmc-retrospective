#!/usr/bin/env python3
"""Create disclosure-safe aggregate model figures from summary metrics.

Input is the compact ``summary_metrics.csv`` produced by the four-variant
aggregate analysis.  The script reads no predictions, cell-level metrics, or
checkpoints.  It produces a single three-panel PNG for portfolio presentation.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


METRICS = [
    ("discrimination_score_l1", "Discrimination score", "Higher is better"),
    ("pearson_delta", "Correlation of response deltas", "Higher is better"),
    ("overlap_at_N", "Top-N response overlap", "Higher is better"),
]
MODELS = ["M1", "M2", "M3", "M4b"]
COLORS = ["#4C78A8", "#72B7B2", "#F2CF5B", "#E45756"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    frame = pd.read_csv(args.metrics)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.8))
    fig.subplots_adjust(top=0.76, bottom=0.18, wspace=0.18)
    for axis, (metric, title, direction) in zip(axes, METRICS, strict=True):
        values = frame.loc[frame["metric"].eq(metric), ["model", "macro_mean"]]
        values = values.set_index("model").reindex(MODELS)
        if values["macro_mean"].isna().any():
            raise ValueError(f"Missing {metric} for one or more variants")
        bars = axis.bar(MODELS, values["macro_mean"], color=COLORS, width=0.68)
        ymax = float(values["macro_mean"].max())
        axis.set_ylim(0, ymax * 1.22 if ymax else 1)
        axis.set_title(title, weight="bold")
        axis.set_ylabel("Macro mean")
        axis.text(0.5, 0.97, direction, transform=axis.transAxes, ha="center", va="top", fontsize=8.5, color="#4A5568")
        axis.grid(axis="y", alpha=0.25)
        axis.set_axisbelow(True)
        for bar, value in zip(bars, values["macro_mean"], strict=True):
            axis.text(bar.get_x() + bar.get_width() / 2, float(value) + ymax * 0.035, f"{value:.3f}", ha="center", va="bottom", fontsize=9)
    fig.suptitle("Exploratory held-out-donor evaluation", weight="bold", fontsize=15)
    fig.text(0.5, 0.045, "Macro mean across 1,617 held-out contexts; not a definitive ranking or causal comparison.", ha="center", fontsize=9, color="#4A5568")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200, metadata={})
    plt.close(fig)


if __name__ == "__main__":
    main()
