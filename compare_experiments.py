"""Compare experiment results across runs.

Scans output directories for experiment_summary.json files and prints a
comparison table showing hyperparameters alongside evaluation metrics.
Optionally generates visual charts for an overarching view.

Usage:
    # Compare all experiments under the default output directory
    python compare_experiments.py

    # Filter by dataset
    python compare_experiments.py --dataset goose

    # Sort by a specific metric (default: best_mIoU)
    python compare_experiments.py --sort best_mIoU

    # Export to CSV
    python compare_experiments.py --csv results.csv

    # Generate visualization dashboard
    python compare_experiments.py --plot

    # Generate and save plots to a directory
    python compare_experiments.py --plot --plot_dir figures/
"""

import argparse
import json
import csv
import sys
from glob import glob
from pathlib import Path

# Hyperparameters to show in the comparison table (order matters)
HPARAM_COLUMNS = [
    "backbone",
    "lr",
    "batch_size",
    "crop_size",
    "num_epochs",
    "optimizer",
    "weight_decay",
    "warmup_iter",
    "timesteps",
    "bit_scale",
    "accumulation",
    "aux_rate",
    "lr_power",
    "train_scale_array",
]

METRIC_COLUMNS = [
    "best_mIoU",
    "best_epoch",
    "final_train_loss",
    "training_time_s",
]


def find_summaries(output_dir, dataset=None):
    """Find all experiment_summary.json files using glob for speed."""
    pattern = str(Path(output_dir) / "**" / "experiment_summary.json")
    summaries = []
    for path in glob(pattern, recursive=True):
        try:
            with open(path) as f:
                data = json.load(f)
            data["_path"] = str(Path(path).parent)
            if dataset and data.get("experiment_dataset") != dataset:
                continue
            summaries.append(data)
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: skipping {path}: {e}", file=sys.stderr)
    return summaries


def format_value(v):
    """Format a value for display."""
    if v is None:
        return "-"
    if isinstance(v, float):
        if v < 0.01:
            return f"{v:.2e}"
        return f"{v:.4f}"
    if isinstance(v, list):
        return str(v)
    return str(v)


def build_table(summaries, sort_by="best_mIoU"):
    """Build rows for the comparison table."""
    rows = []
    for s in summaries:
        hparams = s.get("hparams", {})
        metrics = s.get("metrics", {})
        row = {
            "experiment": s.get("experiment_name", "?"),
            "dataset": s.get("experiment_dataset", "?"),
        }
        for col in HPARAM_COLUMNS:
            row[col] = hparams.get(col)
        for col in METRIC_COLUMNS:
            row[col] = metrics.get(col)
        rows.append(row)

    # Sort: descending for mIoU, ascending for loss/time
    reverse = sort_by in ("best_mIoU",)
    rows.sort(
        key=lambda r: r.get(sort_by) or (float("-inf") if reverse else float("inf")),
        reverse=reverse,
    )
    return rows


def detect_varying(rows):
    """Return set of hparam columns that differ across experiments."""
    if len(rows) <= 1:
        return set(HPARAM_COLUMNS)
    varying = set()
    for col in HPARAM_COLUMNS:
        values = {format_value(r.get(col)) for r in rows}
        if len(values) > 1:
            varying.add(col)
    return varying


def print_table(rows, show_all=False):
    """Print a formatted comparison table to stdout."""
    if not rows:
        print("No experiments found.")
        return

    varying = detect_varying(rows)

    # Always show: experiment name, dataset, varying hparams, all metrics
    columns = ["experiment", "dataset"]
    if show_all:
        columns += HPARAM_COLUMNS
    else:
        columns += [c for c in HPARAM_COLUMNS if c in varying]
    columns += METRIC_COLUMNS

    # Compute column widths
    header = {c: c for c in columns}
    all_rows = [header] + [{c: format_value(r.get(c)) for c in columns} for r in rows]
    widths = {c: max(len(str(row[c])) for row in all_rows) for c in columns}

    # Print
    sep = "  "
    header_line = sep.join(str(header[c]).ljust(widths[c]) for c in columns)
    print(header_line)
    print(sep.join("-" * widths[c] for c in columns))
    for row in all_rows[1:]:
        print(sep.join(str(row[c]).ljust(widths[c]) for c in columns))

    # Summary
    miou_values = [r.get("best_mIoU") for r in rows if r.get("best_mIoU") is not None]
    if miou_values:
        best_idx = max(range(len(rows)), key=lambda i: rows[i].get("best_mIoU") or 0)
        print(
            f"\nBest: {rows[best_idx]['experiment']} "
            f"(mIoU={rows[best_idx]['best_mIoU']:.4f})"
        )


def export_csv(rows, path):
    """Export comparison table to CSV."""
    columns = ["experiment", "dataset"] + HPARAM_COLUMNS + METRIC_COLUMNS
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({c: format_value(row.get(c)) for c in columns})
    print(f"Exported {len(rows)} experiments to {path}")


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def _short_name(name, max_len=30):
    """Truncate long experiment names for plot labels."""
    if len(name) <= max_len:
        return name
    return name[:max_len - 3] + "..."


def plot_miou_bar(rows, ax):
    """Horizontal bar chart of best_mIoU per experiment, color-coded."""
    filtered = [r for r in rows if r.get("best_mIoU") is not None]
    if not filtered:
        ax.text(0.5, 0.5, "No mIoU data", ha="center", va="center",
                transform=ax.transAxes)
        return
    filtered.sort(key=lambda r: r["best_mIoU"])
    names = [_short_name(r["experiment"]) for r in filtered]
    values = [r["best_mIoU"] for r in filtered]

    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    cmap = plt.cm.RdYlGn
    if len(values) > 1:
        vmin, vmax = min(values), max(values)
        norm = mcolors.Normalize(vmin=vmin - 0.001, vmax=vmax + 0.001)
    else:
        norm = mcolors.Normalize(vmin=0, vmax=1)
    colors = [cmap(norm(v)) for v in values]

    bars = ax.barh(names, values, color=colors, edgecolor="gray", linewidth=0.5)
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=8)
    ax.set_xlabel("best mIoU")
    ax.set_title("Experiment Comparison: best mIoU", fontweight="bold")
    ax.set_xlim(right=max(values) * 1.08 if values else 1)


def plot_hparam_impact(rows, ax):
    """Show which hyperparameters vary and their correlation with mIoU.

    For each varying numeric hyperparameter, compute rank correlation with mIoU
    and display as a horizontal bar chart.
    """
    filtered = [r for r in rows if r.get("best_mIoU") is not None]
    if len(filtered) < 3:
        ax.text(0.5, 0.5, "Need >= 3 experiments\nwith mIoU for correlation",
                ha="center", va="center", transform=ax.transAxes, fontsize=10)
        return

    varying = detect_varying(filtered)
    miou_vals = [r["best_mIoU"] for r in filtered]

    correlations = {}
    for col in HPARAM_COLUMNS:
        if col not in varying:
            continue
        vals = []
        valid = True
        for r in filtered:
            v = r.get(col)
            if isinstance(v, (int, float)) and v is not True and v is not False:
                vals.append(float(v))
            elif isinstance(v, bool):
                vals.append(1.0 if v else 0.0)
            else:
                valid = False
                break
        if not valid or len(set(vals)) <= 1:
            continue
        # Spearman rank correlation (manual to avoid scipy dependency)
        n = len(vals)
        rank_x = _rank(vals)
        rank_y = _rank(miou_vals)
        d_sq = sum((rx - ry) ** 2 for rx, ry in zip(rank_x, rank_y))
        rho = 1 - (6 * d_sq) / (n * (n ** 2 - 1))
        correlations[col] = rho

    if not correlations:
        ax.text(0.5, 0.5, "No numeric varying\nhyperparameters found",
                ha="center", va="center", transform=ax.transAxes, fontsize=10)
        return

    # Sort by absolute correlation
    sorted_cols = sorted(correlations, key=lambda c: abs(correlations[c]))
    names = sorted_cols
    vals = [correlations[c] for c in sorted_cols]
    colors = ["#2ecc71" if v > 0 else "#e74c3c" for v in vals]

    ax.barh(names, vals, color=colors, edgecolor="gray", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=0.5)
    for i, (name, val) in enumerate(zip(names, vals)):
        ax.text(val + (0.02 if val >= 0 else -0.02), i,
                f"{val:+.2f}", va="center", fontsize=8,
                ha="left" if val >= 0 else "right")
    ax.set_xlabel("Rank correlation with mIoU")
    ax.set_title("Hyperparameter Impact (Spearman rho)", fontweight="bold")
    ax.set_xlim(-1.15, 1.15)


def _rank(values):
    """Compute ranks for a list of values (average rank for ties)."""
    indexed = sorted(enumerate(values), key=lambda x: x[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i
        while j < len(indexed) and indexed[j][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + j + 1) / 2  # 1-based average
        for k in range(i, j):
            ranks[indexed[k][0]] = avg_rank
        i = j
    return ranks


def plot_scatter_grid(rows, axes_flat, hparam_cols):
    """Scatter plots of each varying numeric hparam vs mIoU."""
    filtered = [r for r in rows if r.get("best_mIoU") is not None]
    miou_vals = [r["best_mIoU"] for r in filtered]

    for idx, col in enumerate(hparam_cols):
        ax = axes_flat[idx]
        vals = []
        mious = []
        for r, m in zip(filtered, miou_vals):
            v = r.get(col)
            if isinstance(v, bool):
                v = int(v)
            if isinstance(v, (int, float)):
                vals.append(float(v))
                mious.append(m)
        if not vals:
            ax.set_visible(False)
            continue
        ax.scatter(vals, mious, c=mious, cmap="RdYlGn", edgecolors="gray",
                   linewidth=0.5, s=60, zorder=3)
        ax.set_xlabel(col, fontsize=9)
        ax.set_ylabel("mIoU", fontsize=9)
        ax.set_title(col, fontsize=10, fontweight="bold")
        ax.grid(True, alpha=0.3)

    # Hide unused axes
    for idx in range(len(hparam_cols), len(axes_flat)):
        axes_flat[idx].set_visible(False)


def plot_efficiency(rows, ax):
    """Scatter plot of training time vs mIoU (efficiency view)."""
    filtered = [r for r in rows
                if r.get("best_mIoU") is not None
                and r.get("training_time_s") is not None]
    if not filtered:
        ax.text(0.5, 0.5, "No time + mIoU data", ha="center", va="center",
                transform=ax.transAxes)
        return

    times_h = [r["training_time_s"] / 3600 for r in filtered]
    mious = [r["best_mIoU"] for r in filtered]
    names = [_short_name(r["experiment"], 20) for r in filtered]

    ax.scatter(times_h, mious, c=mious, cmap="RdYlGn", edgecolors="gray",
               linewidth=0.5, s=80, zorder=3)
    for x, y, name in zip(times_h, mious, names):
        ax.annotate(name, (x, y), fontsize=7, textcoords="offset points",
                    xytext=(5, 5), alpha=0.8)
    ax.set_xlabel("Training time (hours)")
    ax.set_ylabel("best mIoU")
    ax.set_title("Efficiency: mIoU vs Training Time", fontweight="bold")
    ax.grid(True, alpha=0.3)


def plot_loss_vs_miou(rows, ax):
    """Scatter plot of final training loss vs best mIoU."""
    filtered = [r for r in rows
                if r.get("best_mIoU") is not None
                and r.get("final_train_loss") is not None]
    if not filtered:
        ax.text(0.5, 0.5, "No loss + mIoU data", ha="center", va="center",
                transform=ax.transAxes)
        return

    losses = [r["final_train_loss"] for r in filtered]
    mious = [r["best_mIoU"] for r in filtered]
    names = [_short_name(r["experiment"], 20) for r in filtered]

    ax.scatter(losses, mious, c=mious, cmap="RdYlGn", edgecolors="gray",
               linewidth=0.5, s=80, zorder=3)
    for x, y, name in zip(losses, mious, names):
        ax.annotate(name, (x, y), fontsize=7, textcoords="offset points",
                    xytext=(5, 5), alpha=0.8)
    ax.set_xlabel("Final training loss")
    ax.set_ylabel("best mIoU")
    ax.set_title("Loss vs mIoU", fontweight="bold")
    ax.grid(True, alpha=0.3)


def generate_plots(rows, plot_dir=None):
    """Generate all visualization plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Determine varying numeric hparams for scatter grid
    varying = detect_varying(rows)
    numeric_varying = []
    for col in HPARAM_COLUMNS:
        if col not in varying:
            continue
        has_numeric = any(
            isinstance(r.get(col), (int, float, bool))
            for r in rows if r.get("best_mIoU") is not None
        )
        if has_numeric:
            numeric_varying.append(col)

    # --- Figure 1: Overview dashboard ---
    fig1, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig1.suptitle("Experiment Comparison Dashboard", fontsize=14, fontweight="bold",
                  y=0.98)

    plot_miou_bar(rows, axes[0, 0])
    plot_hparam_impact(rows, axes[0, 1])
    plot_efficiency(rows, axes[1, 0])
    plot_loss_vs_miou(rows, axes[1, 1])

    fig1.tight_layout(rect=[0, 0, 1, 0.96])

    if plot_dir:
        Path(plot_dir).mkdir(parents=True, exist_ok=True)
        fig1.savefig(Path(plot_dir) / "dashboard.png", dpi=150, bbox_inches="tight")
        print(f"Saved dashboard to {Path(plot_dir) / 'dashboard.png'}")

    # --- Figure 2: Hyperparameter scatter grid ---
    if numeric_varying:
        n = len(numeric_varying)
        ncols = min(3, n)
        nrows = (n + ncols - 1) // ncols
        fig2, axes2 = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
        if n == 1:
            axes2 = [axes2]
        else:
            axes2 = axes2.flatten() if hasattr(axes2, "flatten") else [axes2]
        fig2.suptitle("Hyperparameter vs mIoU", fontsize=14, fontweight="bold",
                      y=0.98)
        plot_scatter_grid(rows, axes2, numeric_varying)
        fig2.tight_layout(rect=[0, 0, 1, 0.96])

        if plot_dir:
            fig2.savefig(Path(plot_dir) / "hparam_scatter.png", dpi=150,
                         bbox_inches="tight")
            print(f"Saved scatter grid to {Path(plot_dir) / 'hparam_scatter.png'}")

    if not plot_dir:
        plt.show()
    else:
        plt.close("all")

    # --- HTML report ---
    if plot_dir:
        _generate_html_report(rows, plot_dir, numeric_varying)


def _generate_html_report(rows, plot_dir, numeric_varying):
    """Generate an HTML dashboard that embeds all charts and the data table."""
    varying = detect_varying(rows)
    columns = ["experiment", "dataset"]
    columns += [c for c in HPARAM_COLUMNS if c in varying]
    columns += METRIC_COLUMNS

    # Find best mIoU for highlighting
    best_miou = max((r.get("best_mIoU") or 0) for r in rows) if rows else 0

    table_rows = ""
    for r in rows:
        cells = ""
        for c in columns:
            val = format_value(r.get(c))
            style = ""
            if c == "best_mIoU" and r.get("best_mIoU") == best_miou and best_miou > 0:
                style = ' style="background:#2ecc71;color:white;font-weight:bold"'
            cells += f"<td{style}>{val}</td>"
        table_rows += f"<tr>{cells}</tr>\n"

    header_cells = "".join(f"<th>{c}</th>" for c in columns)

    scatter_img = ""
    if numeric_varying:
        scatter_img = '<img src="hparam_scatter.png" style="max-width:100%">'

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Experiment Comparison Dashboard</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
         margin: 20px; background: #f5f6fa; }}
  h1 {{ color: #2c3e50; }}
  h2 {{ color: #34495e; margin-top: 30px; }}
  table {{ border-collapse: collapse; width: 100%; margin-top: 10px;
           background: white; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
  th {{ background: #2c3e50; color: white; padding: 10px 12px; text-align: left;
       font-size: 13px; }}
  td {{ padding: 8px 12px; border-bottom: 1px solid #ecf0f1; font-size: 13px; }}
  tr:hover {{ background: #f0f3f5; }}
  .chart {{ margin: 20px 0; text-align: center; }}
  .chart img {{ max-width: 100%; box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                border-radius: 4px; }}
  .summary {{ background: #2ecc71; color: white; padding: 15px 20px;
              border-radius: 6px; display: inline-block; margin: 10px 0; }}
</style>
</head>
<body>
<h1>Experiment Comparison Dashboard</h1>
<p>Found <strong>{len(rows)}</strong> experiment(s)</p>

<div class="chart">
  <img src="dashboard.png" style="max-width:100%">
</div>

<h2>Results Table</h2>
<div style="overflow-x:auto">
<table>
<thead><tr>{header_cells}</tr></thead>
<tbody>
{table_rows}
</tbody>
</table>
</div>

<h2>Hyperparameter Scatter Plots</h2>
<div class="chart">
  {scatter_img}
</div>
</body>
</html>"""

    html_path = Path(plot_dir) / "dashboard.html"
    with open(html_path, "w") as f:
        f.write(html)
    print(f"Saved HTML dashboard to {html_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare experiment results")
    parser.add_argument(
        "--output_dir", default="output_dir/",
        help="Root directory containing experiment outputs",
    )
    parser.add_argument(
        "--dataset", default=None,
        help="Filter by dataset name (e.g. goose, nyuv2, sunrgbd)",
    )
    parser.add_argument(
        "--sort", default="best_mIoU",
        choices=METRIC_COLUMNS + ["experiment"],
        help="Column to sort by (default: best_mIoU)",
    )
    parser.add_argument(
        "--csv", default=None,
        help="Export results to CSV file",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Show all hparam columns, not just varying ones",
    )
    parser.add_argument(
        "--plot", action="store_true",
        help="Generate visualization charts",
    )
    parser.add_argument(
        "--plot_dir", default=None,
        help="Directory to save plots (if omitted with --plot, shows interactively)",
    )
    args = parser.parse_args()

    summaries = find_summaries(args.output_dir, dataset=args.dataset)
    if not summaries:
        msg = f"No experiment_summary.json found under {args.output_dir}"
        if args.dataset:
            msg += f" for dataset '{args.dataset}'"
        print(msg)
        return

    rows = build_table(summaries, sort_by=args.sort)
    print(f"Found {len(rows)} experiment(s)\n")
    print_table(rows, show_all=args.all)

    if args.csv:
        print()
        export_csv(rows, args.csv)

    if args.plot:
        plot_dir = args.plot_dir or "output_dir/comparison_plots"
        generate_plots(rows, plot_dir=plot_dir)


if __name__ == "__main__":
    main()
