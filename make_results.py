"""Final results table and comparison plots across all experiments (E1-E5).

Collects every results/<run>/metrics.json (+ config.json) and writes:
    results/summary.csv, results/summary.md     EX per difficulty, one row per run
    figures/ex_by_difficulty.png                grouped bars: EX per difficulty, per run
    figures/exec_error_rate.png                 share of predictions that fail to execute

Usage:
    python make_results.py                      # all runs (run names starting with "_" are skipped)
    python make_results.py --runs e1_zeroshot_Qwen2.5-Coder-1.5B-Instruct_balanced e2_fewshot_...
    python make_results.py --labels e1_zeroshot_...="E1 zero-shot" e2_fewshot_...="E2 few-shot"
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from evaluate import HARDNESS_LEVELS

LEVELS = HARDNESS_LEVELS + ["all"]
LEVEL_NAMES = {"easy": "Easy", "medium": "Medium", "hard": "Hard", "extra": "Extra hard", "all": "Overall"}
# Fixed categorical order (validated colorblind-safe palette); never cycled.
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def collect(results_dir: Path, runs: list[str] | None) -> list[dict]:
    names = runs or sorted(p.parent.name for p in results_dir.glob("*/metrics.json")
                           if not p.parent.name.startswith("_"))
    out = []
    for name in names:
        d = results_dir / name
        metrics = json.loads((d / "metrics.json").read_text())
        cfg = json.loads((d / "config.json").read_text()) if (d / "config.json").exists() else {}
        out.append({"run": name, "label": name, "experiment": cfg.get("experiment", ""),
                    "model": cfg.get("model", ""), "subset": cfg.get("subset", ""), "metrics": metrics})
    return out


def write_tables(rows: list[dict], results_dir: Path) -> None:
    header = ["run", "experiment", "model", "subset", "n"] + [f"EX_{l}" for l in LEVELS] + ["exec_error_rate"]
    with open(results_dir / "summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            m = r["metrics"]
            w.writerow([r["label"], r["experiment"], r["model"], r["subset"], m["all"]["n"]]
                       + [m[l]["ex"] for l in LEVELS] + [m["all"]["exec_error_rate"]])
    md = ["| Run | Model | n | " + " | ".join(LEVEL_NAMES[l] for l in LEVELS) + " | Exec errors |",
          "|---|---|---:|" + "---:|" * (len(LEVELS) + 1)]
    for r in rows:
        m = r["metrics"]
        cells = [f"{100 * m[l]['ex']:.1f}" if m[l]["n"] else "-" for l in LEVELS]
        cells[-1] = f"**{cells[-1]}**"
        md.append(f"| {r['label']} | {r['model'].split('/')[-1]} | {m['all']['n']} | " + " | ".join(cells)
                  + f" | {100 * m['all']['exec_error_rate']:.1f}% |")
    (results_dir / "summary.md").write_text("Execution accuracy (EX, %) by Spider difficulty\n\n" + "\n".join(md) + "\n")
    print((results_dir / "summary.md").read_text())


def _style(ax, ylabel: str) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(INK_2)
    ax.tick_params(colors=INK_2, length=0)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_ylabel(ylabel, color=INK_2)


def plot(rows: list[dict], fig_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if len(rows) > len(SERIES_COLORS):
        raise SystemExit(f"{len(rows)} runs is too many for one chart; pass --runs to pick at most "
                         f"{len(SERIES_COLORS)}")
    fig_dir.mkdir(parents=True, exist_ok=True)
    k = len(rows)
    group_w = 0.8
    bar_w = group_w / k

    # 1) EX by difficulty
    fig, ax = plt.subplots(figsize=(max(6.5, 1.6 * len(LEVELS) + 0.4 * k), 4), facecolor=SURFACE)
    for i, r in enumerate(rows):
        xs = [j - group_w / 2 + bar_w * (i + 0.5) for j in range(len(LEVELS))]
        ys = [100 * (r["metrics"][l]["ex"] or 0) for l in LEVELS]
        bars = ax.bar(xs, ys, width=bar_w * 0.9, color=SERIES_COLORS[i], label=r["label"],
                      edgecolor=SURFACE, linewidth=1)
        # direct labels only on the Overall group (selective labelling)
        b = bars[-1]
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.2, f"{ys[-1]:.0f}",
                ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_xticks(range(len(LEVELS)), [LEVEL_NAMES[l] for l in LEVELS], color=INK)
    ax.set_ylim(0, 100)
    _style(ax, "Execution accuracy (%)")
    ax.set_title("Execution accuracy by query difficulty", loc="left", color=INK, fontsize=12)
    ax.legend(frameon=False, fontsize=8, loc="upper right", labelcolor=INK)
    fig.tight_layout()
    fig.savefig(fig_dir / "ex_by_difficulty.png", dpi=200)
    plt.close(fig)

    # 2) Execution-error rate (overall), one bar per run
    fig, ax = plt.subplots(figsize=(max(5, 1.1 * k + 2), 3.2), facecolor=SURFACE)
    ys = [100 * r["metrics"]["all"]["exec_error_rate"] for r in rows]
    bars = ax.bar(range(k), ys, width=0.6, color=SERIES_COLORS[:k], edgecolor=SURFACE)
    for b, y in zip(bars, ys):
        ax.text(b.get_x() + b.get_width() / 2, y + 0.5, f"{y:.1f}%", ha="center", va="bottom",
                fontsize=8, color=INK)
    ax.set_xticks(range(k), [r["label"] for r in rows], color=INK, fontsize=8)
    _style(ax, "Queries that fail to execute (%)")
    ax.set_title("Execution errors per run", loc="left", color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(fig_dir / "exec_error_rate.png", dpi=200)
    plt.close(fig)
    print(f"wrote {fig_dir}/ex_by_difficulty.png and exec_error_rate.png")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    ap.add_argument("--fig-dir", type=Path, default=Path("figures"))
    ap.add_argument("--runs", nargs="*", default=None)
    ap.add_argument("--labels", nargs="*", default=[], metavar="RUN=LABEL",
                    help="display names for runs, e.g. e1_zeroshot_x='E1 zero-shot'")
    args = ap.parse_args()
    rows = collect(args.results_dir, args.runs)
    if not rows:
        raise SystemExit(f"no metrics.json found under {args.results_dir}")
    labels = dict(item.split("=", 1) for item in args.labels)
    unknown = set(labels) - {r["run"] for r in rows}
    if unknown:
        raise SystemExit(f"--labels refers to unknown runs: {sorted(unknown)}")
    for r in rows:
        r["label"] = labels.get(r["run"], r["run"])
    write_tables(rows, args.results_dir)
    plot(rows, args.fig_dir)


if __name__ == "__main__":
    main()
