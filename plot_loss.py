"""Plot the fine-tuning loss curves (Part 4) from loss_log.json.

    python plot_loss.py loss_log.json            # writes figures/e4_loss_curve.png

loss_log.json is the Hugging Face Trainer's log_history: entries with 'loss' are training steps,
entries with 'eval_loss' are validation checks. After a --resume the log may only start at the
resume step; the script prints the step range so this can be stated in the report.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log", help="path to loss_log.json")
    ap.add_argument("--out", default="figures/e4_loss_curve.png")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    log = json.loads(Path(args.log).read_text())
    train = [(e["step"], float(e["loss"])) for e in log if "loss" in e]
    evals = [(e["step"], float(e["eval_loss"])) for e in log if "eval_loss" in e]
    if not train:
        raise SystemExit("no training-loss entries in the log")

    print(f"training points: {len(train)}, steps {train[0][0]}..{train[-1][0]}")
    print(f"validation points: {len(evals)}", [f"{s}: {v:.4f}" for s, v in evals])

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(*zip(*train), label="training loss", linewidth=1.5)
    if evals:
        ax.plot(*zip(*evals), "o-", label="validation loss", linewidth=1.5)
    ax.set_xlabel("optimiser step")
    ax.set_ylabel("loss")
    ax.set_title("LoRA fine-tuning of Qwen2.5-Coder-1.5B-Instruct on Spider")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    print("saved", out)


if __name__ == "__main__":
    main()
