"""
Model comparison plots for Smart Power Forecasting.

Two groups of charts are produced:
  1. Within-interval comparisons  — for each time interval (30min, 1hr, 2hr, 4hr,
     6hr) show how all models stack up on every metric.
  2. Cross-interval comparisons   — for each model, show how its metrics change
     as the resampling interval grows (coarser vs finer granularity).

All charts are saved as PNG files inside  reports/plots/.
Run:
    python plot_model_comparisons.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
REPORTS_DIR = ROOT / "reports"
PLOTS_DIR = REPORTS_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

INTERVALS = ["30min", "1hr", "2hr", "4hr", "6hr"]

EXCLUDED_MODELS = {"StackingEnsemble"}

# Metrics to show (column name → display label)
METRICS: dict[str, str] = {
    "MAE": "MAE (kW)",
    "RMSE": "RMSE (kW)",
    "R2": "R²",
    "MAPE": "MAPE (%)",
    "RAE": "RAE",
}

# For "lower is better" vs "higher is better" annotations
LOWER_BETTER = {"MAE", "RMSE", "MAPE", "RAE"}
HIGHER_BETTER = {"R2"}

# Colour palette — one colour per model (consistent across all plots)
MODEL_PALETTE = {
    "LightGBM":         "#2196F3",   # blue
    "XGBoost":          "#FF9800",   # orange
    "Weighted Ensemble":"#4CAF50",   # green
    # "StackingEnsemble": "#9C27B0",   # purple
    "Ridge":            "#F44336",   # red
    "RandomForest":     "#795548",   # brown
}
DEFAULT_COLOUR = "#607D8B"          # grey for unknown models

# ── Helpers ────────────────────────────────────────────────────────────────────

def load_metrics() -> dict[str, pd.DataFrame]:
    """Load all per-interval metrics CSVs and return a dict keyed by interval."""
    dfs: dict[str, pd.DataFrame] = {}
    for interval in INTERVALS:
        csv = REPORTS_DIR / f"model_metrics_{interval}.csv"
        if not csv.exists():
            print(f"  [WARN] Missing {csv.name} — skipping {interval}")
            continue
        df = pd.read_csv(csv)
        df = df[~df["Model"].isin(EXCLUDED_MODELS)].reset_index(drop=True)
        df["interval"] = interval
        dfs[interval] = df
    return dfs


def model_colour(name: str) -> str:
    return MODEL_PALETTE.get(name, DEFAULT_COLOUR)


def _save(fig: plt.Figure, name: str) -> None:
    out = PLOTS_DIR / f"{name}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out.relative_to(ROOT)}")


# ── 1. Within-interval comparisons ────────────────────────────────────────────

def plot_within_interval(dfs: dict[str, pd.DataFrame]) -> None:
    """
    For each interval: one figure with a sub-plot per metric.
    Each sub-plot is a horizontal bar chart comparing models.
    """
    print("\n[1/3] Plotting within-interval comparisons …")

    for interval, df in dfs.items():
        n_metrics = len(METRICS)
        fig, axes = plt.subplots(
            1, n_metrics,
            figsize=(5 * n_metrics, max(4, 0.55 * len(df) + 2)),
        )
        fig.suptitle(
            f"Model Comparison — {interval} Interval",
            fontsize=15, fontweight="bold", y=1.02,
        )

        for ax, (col, label) in zip(axes, METRICS.items()):
            if col not in df.columns:
                ax.set_visible(False)
                continue

            # Sort: best (lowest/highest) at top
            ascending = col in LOWER_BETTER
            sorted_df = df.sort_values(col, ascending=not ascending)

            models = sorted_df["Model"].tolist()
            values = sorted_df[col].tolist()
            colours = [model_colour(m) for m in models]

            bars = ax.barh(models, values, color=colours, edgecolor="white", height=0.6)

            # Value labels on bars
            for bar, val in zip(bars, values):
                ax.text(
                    bar.get_width() + bar.get_width() * 0.02,
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:.4f}" if col != "MAPE" else f"{val:.2f}%",
                    va="center", ha="left", fontsize=8,
                )

            ax.set_title(label, fontsize=11)
            ax.set_xlabel(label, fontsize=9)
            ax.tick_params(axis="y", labelsize=9)

            # Subtle grid
            ax.xaxis.set_minor_locator(mticker.AutoMinorLocator())
            ax.grid(axis="x", linestyle="--", linewidth=0.5, alpha=0.5)
            ax.set_axisbelow(True)

            # Highlight the best bar with a star annotation
            best_bar = bars[0]
            ax.annotate(
                "★",
                xy=(best_bar.get_width(), best_bar.get_y() + best_bar.get_height() / 2),
                xytext=(3, 0),
                textcoords="offset points",
                fontsize=10, color="gold", va="center",
            )

        fig.tight_layout()
        _save(fig, f"within_interval_{interval}")


# ── 2. All models, all metrics for one interval (grouped bar) ─────────────────

def plot_grouped_bars_per_interval(dfs: dict[str, pd.DataFrame]) -> None:
    """
    For each interval: a single grouped bar chart where X = model,
    and each group contains one bar per metric (normalised to [0,1]).
    Useful for a quick at-a-glance quality picture.
    """
    print("\n[2/3] Plotting grouped metric bars per interval …")

    metric_cols = list(METRICS.keys())

    for interval, df in dfs.items():
        # Normalise metrics to [0, 1] so they fit on one axis.
        # For "lower is better" metrics, invert: score = 1 - norm.
        normed = df[["Model"] + metric_cols].copy()
        for col in metric_cols:
            mn, mx = normed[col].min(), normed[col].max()
            rng = mx - mn if mx != mn else 1.0
            if col in LOWER_BETTER:
                normed[col] = 1 - (normed[col] - mn) / rng    # higher = better
            else:
                normed[col] = (normed[col] - mn) / rng

        models = normed["Model"].tolist()
        n_models = len(models)
        n_metrics = len(metric_cols)
        x = np.arange(n_models)
        width = 0.8 / n_metrics

        fig, ax = plt.subplots(figsize=(max(10, n_models * 1.5), 5))

        metric_colours = plt.cm.tab10(np.linspace(0, 0.8, n_metrics))
        for i, (col, label) in enumerate(METRICS.items()):
            offsets = x - 0.4 + (i + 0.5) * width
            ax.bar(
                offsets,
                normed[col],
                width=width * 0.9,
                label=f"{label} ({'↓' if col in LOWER_BETTER else '↑'})",
                color=metric_colours[i],
                edgecolor="white",
            )

        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=20, ha="right", fontsize=10)
        ax.set_ylabel("Normalised Score  (higher = better)", fontsize=10)
        ax.set_title(
            f"Normalised Metric Scores — {interval} Interval\n"
            "(metrics inverted where lower is better, so taller bar = better)",
            fontsize=12, fontweight="bold",
        )
        ax.legend(loc="lower right", fontsize=9, ncol=3)
        ax.set_ylim(0, 1.15)
        ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5)
        ax.set_axisbelow(True)

        fig.tight_layout()
        _save(fig, f"grouped_metrics_{interval}")


# ── 3. Cross-interval comparisons ─────────────────────────────────────────────

def plot_cross_interval(dfs: dict[str, pd.DataFrame]) -> None:
    """
    For each metric: one figure with one line per model showing how the
    metric evolves across resampling intervals.
    """
    print("\n[3/3] Plotting cross-interval comparisons …")

    # Build a combined long-form frame
    frames = []
    for interval, df in dfs.items():
        tmp = df[["Model"] + list(METRICS.keys())].copy()
        tmp["interval"] = interval
        frames.append(tmp)
    combined = pd.concat(frames, ignore_index=True)

    # Canonical interval ordering (fine → coarse)
    interval_order = [i for i in INTERVALS if i in combined["interval"].unique()]
    combined["interval"] = pd.Categorical(combined["interval"], categories=interval_order, ordered=True)
    combined = combined.sort_values("interval")

    all_models = combined["Model"].unique().tolist()

    # ── 3a. One figure per metric (line chart) ──────────────────────────────
    for col, label in METRICS.items():
        fig, ax = plt.subplots(figsize=(9, 5))

        for model in all_models:
            sub = combined[combined["Model"] == model].sort_values("interval")
            ax.plot(
                sub["interval"].astype(str),
                sub[col],
                marker="o",
                linewidth=2,
                label=model,
                color=model_colour(model),
            )
            # Annotate last point
            last = sub.iloc[-1]
            ax.annotate(
                f"{float(last[col]):.3f}",
                xy=(len(interval_order) - 1, float(last[col])),
                xytext=(6, 0), textcoords="offset points",
                fontsize=7, color=model_colour(model), va="center",
            )

        direction = "↓ lower is better" if col in LOWER_BETTER else "↑ higher is better"
        ax.set_title(
            f"{label} Across Resampling Intervals\n({direction})",
            fontsize=13, fontweight="bold",
        )
        ax.set_xlabel("Resampling Interval (fine → coarse)", fontsize=11)
        ax.set_ylabel(label, fontsize=11)
        ax.legend(loc="upper right", fontsize=9, ncol=2)
        ax.grid(linestyle="--", linewidth=0.5, alpha=0.5)
        ax.set_axisbelow(True)

        fig.tight_layout()
        _save(fig, f"cross_interval_{col.lower()}")

    # ── 3b. Heatmaps: metric × interval for each model rank ────────────────
    for col, label in METRICS.items():
        pivot = combined.pivot_table(index="Model", columns="interval", values=col, aggfunc="first")
        pivot = pivot[interval_order]

        fig, ax = plt.subplots(figsize=(len(interval_order) * 1.5 + 3, len(pivot) * 0.7 + 1.5))

        # Choose colour direction
        cmap = "RdYlGn_r" if col in LOWER_BETTER else "RdYlGn"
        im = ax.imshow(pivot.values.astype(float), cmap=cmap, aspect="auto")
        plt.colorbar(im, ax=ax, shrink=0.8, label=label)

        ax.set_xticks(range(len(interval_order)))
        ax.set_xticklabels(interval_order, fontsize=11)
        ax.set_yticks(range(len(pivot)))
        ax.set_yticklabels(pivot.index.tolist(), fontsize=10)
        ax.set_title(f"{label} Heatmap — Models × Intervals", fontsize=12, fontweight="bold")

        # Annotate cells
        for i in range(len(pivot)):
            for j in range(len(interval_order)):
                val = pivot.values[i, j]
                if not np.isnan(val):
                    fmt = f"{val:.2f}%" if col == "MAPE" else f"{val:.4f}"
                    ax.text(j, i, fmt, ha="center", va="center", fontsize=9,
                            color="black")

        fig.tight_layout()
        _save(fig, f"heatmap_{col.lower()}")

    # ── 3c. Best-model win count per interval ───────────────────────────────
    win_data: dict[str, dict[str, int]] = {m: {k: 0 for k in METRICS} for m in all_models}
    for interval in interval_order:
        sub = combined[combined["interval"] == interval]
        for col in METRICS:
            if col in LOWER_BETTER:
                best_model = sub.loc[sub[col].idxmin(), "Model"]
            else:
                best_model = sub.loc[sub[col].idxmax(), "Model"]
            if best_model in win_data:
                win_data[best_model][col] += 1

    win_df = pd.DataFrame(win_data).T.fillna(0)
    win_df["Total"] = win_df.sum(axis=1)
    win_df = win_df.sort_values("Total", ascending=True)

    fig, ax = plt.subplots(figsize=(9, max(4, len(win_df) * 0.6 + 1.5)))
    bottom = np.zeros(len(win_df))
    metric_colours = plt.cm.tab10(np.linspace(0, 0.8, len(METRICS)))
    for (col, label), colour in zip(METRICS.items(), metric_colours):
        if col in win_df.columns:
            ax.barh(win_df.index, win_df[col], left=bottom, label=label,
                    color=colour, edgecolor="white")
            bottom += win_df[col].values

    ax.set_xlabel("Number of interval × metric wins", fontsize=11)
    ax.set_title("Best-Model Win Count\n(across all intervals and metrics)", fontsize=12,
                 fontweight="bold")
    ax.legend(loc="lower right", fontsize=9, ncol=2)
    ax.grid(axis="x", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.set_axisbelow(True)
    fig.tight_layout()
    _save(fig, "win_count_summary")


# ── 4. Radar / spider chart: best model per interval ─────────────────────────

def plot_radar_best_per_interval(dfs: dict[str, pd.DataFrame]) -> None:
    """
    One radar chart per interval showing the best model's normalised
    performance across all metrics.
    Also a combined radar overlaying one line per interval (best model each).
    """
    print("\n[4/4] Plotting radar charts …")

    metric_cols = list(METRICS.keys())
    n = len(metric_cols)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    labels = list(METRICS.values())

    # ── Per-interval radar ──────────────────────────────────────────────────
    combined_radar: list[dict] = []

    for interval, df in dfs.items():
        # Best model = row with lowest RMSE
        best_row = df.loc[df["RMSE"].idxmin()]
        best_name = best_row["Model"]

        # Normalise across all models for this interval
        norm_vals = []
        for col in metric_cols:
            mn, mx = df[col].min(), df[col].max()
            rng = mx - mn if mx != mn else 1.0
            if col in LOWER_BETTER:
                score = 1 - (best_row[col] - mn) / rng
            else:
                score = (best_row[col] - mn) / rng
            norm_vals.append(float(score))

        combined_radar.append({
            "interval": interval,
            "model": best_name,
            "values": norm_vals,
        })

        values = norm_vals + norm_vals[:1]
        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"polar": True})
        ax.plot(angles, values, linewidth=2, color=model_colour(best_name))
        ax.fill(angles, values, alpha=0.25, color=model_colour(best_name))
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=7)
        ax.set_title(
            f"Best Model ({best_name})\n{interval} Interval",
            size=12, fontweight="bold", pad=20,
        )
        fig.tight_layout()
        _save(fig, f"radar_{interval}")

    # ── Combined radar: best model per interval ─────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"polar": True})
    interval_colours = plt.cm.viridis(np.linspace(0, 0.85, len(combined_radar)))

    for entry, colour in zip(combined_radar, interval_colours):
        values = entry["values"] + entry["values"][:1]
        ax.plot(angles, values, linewidth=2, label=f"{entry['interval']} ({entry['model']})",
                color=colour)
        ax.fill(angles, values, alpha=0.08, color=colour)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=8)
    ax.set_title(
        "Best Model Normalised Performance\nAcross All Intervals",
        size=13, fontweight="bold", pad=20,
    )
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=9)
    fig.tight_layout()
    _save(fig, "radar_combined_intervals")


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    print("Loading metrics …")
    dfs = load_metrics()
    if not dfs:
        print("No metric CSVs found. Run the training pipeline first.")
        return

    print(f"Loaded data for intervals: {list(dfs.keys())}")

    plot_within_interval(dfs)
    plot_grouped_bars_per_interval(dfs)
    plot_cross_interval(dfs)
    plot_radar_best_per_interval(dfs)

    print(f"\nAll plots saved to  {PLOTS_DIR.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
