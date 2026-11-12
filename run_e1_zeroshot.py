"""E1: zero-shot Text-to-SQL with a small code LLM (schema prompt, greedy decoding).

Run on Google Colab with a T4 GPU (see notebooks/run_e1_colab.ipynb):
    python run_e1_zeroshot.py                                  # balanced subset (200)
    python run_e1_zeroshot.py --subset full                    # all 1034 dev questions
    python run_e1_zeroshot.py --model Qwen/Qwen2.5-Coder-7B-Instruct --load-in-4bit

Outputs in results/<run_name>/:
    config.json          everything needed to reproduce the run
    predictions.jsonl    one line per question (raw output + cleaned SQL)
    metrics.json         EX per difficulty + overall
    eval_details.jsonl   per-question correctness and execution errors
The run resumes automatically if predictions.jsonl is partly written.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from evaluate import clean_sql, load_dev_subset, read_jsonl, score_file
from prompts import ZERO_SHOT_VERSION, zero_shot_messages

DEFAULT_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"


def git_info() -> dict:
    def run(*cmd):
        try:
            return subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            return None
    status = run("git", "status", "--porcelain", "--untracked-files=no")
    return {"commit": run("git", "rev-parse", "HEAD"),
            "uncommitted_changes": bool(status) if status is not None else None}


def load_model(name: str, load_in_4bit: bool):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name)
    tok.padding_side = "left"  # required for batched generation with decoder-only models
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kwargs = {}
    if torch.cuda.is_available():
        kwargs["device_map"] = "auto"
        kwargs["dtype"] = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=kwargs["dtype"])
    model = AutoModelForCausalLM.from_pretrained(name, **kwargs)
    model.eval()
    return tok, model


def generate_batch(tok, model, batch_messages: list[list[dict]], max_new_tokens: int) -> list[str]:
    import torch

    texts = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
             for m in batch_messages]
    enc = tok(texts, return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tok.pad_token_id)
    new_tokens = out[:, enc["input_ids"].shape[1]:]
    return tok.batch_decode(new_tokens, skip_special_tokens=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--subset", choices=["balanced", "full"], default="balanced")
    ap.add_argument("--n-rows", type=int, default=3, help="sample rows per table in the prompt")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--load-in-4bit", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="only the first N questions (quick test)")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import torch
    import transformers
    torch.manual_seed(args.seed)

    examples = load_dev_subset(args.subset)
    if args.limit:
        examples = examples[: args.limit]
    run_name = args.run_name or f"e1_zeroshot_{args.model.split('/')[-1]}_{args.subset}"
    out_dir = Path("results") / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_file = out_dir / "predictions.jsonl"

    done = {r["idx"] for r in read_jsonl(pred_file)} if pred_file.exists() else set()
    todo = [e for e in examples if e["idx"] not in done]
    print(f"{run_name}: {len(examples)} questions, {len(done)} already done, {len(todo)} to go")

    config = {
        "experiment": "E1 zero-shot",
        "model": args.model,
        "load_in_4bit": args.load_in_4bit,
        "prompt_version": ZERO_SHOT_VERSION,
        "schema_format": "CREATE TABLE + sample rows",
        "n_sample_rows": args.n_rows,
        "decoding": {"do_sample": False, "max_new_tokens": args.max_new_tokens},
        "batch_size": args.batch_size,
        "subset": args.subset,
        "n_examples": len(examples),
        "seed": args.seed,
        "git": git_info(),
        "versions": {"python": platform.python_version(), "torch": torch.__version__,
                     "transformers": transformers.__version__},
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    config_file = out_dir / "config.json"
    if config_file.exists():  # resumed run: keep the original settings, only add time
        old = json.loads(config_file.read_text())
        for k in ("model", "prompt_version", "n_sample_rows", "decoding", "subset"):
            if old.get(k) != config[k]:
                raise SystemExit(f"{config_file} has {k}={old.get(k)!r}, this run has {config[k]!r}. "
                                 "Use a different --run-name.")
        old["n_resumes"] = old.get("n_resumes", 0) + (1 if todo else 0)
        old["n_examples"] = max(old["n_examples"], config["n_examples"])
        config = old

    t0 = time.time()
    if todo:
        tok, model = load_model(args.model, args.load_in_4bit)
        with open(pred_file, "a") as f:
            for i in range(0, len(todo), args.batch_size):
                batch = todo[i: i + args.batch_size]
                raws = generate_batch(tok, model, [zero_shot_messages(e, args.n_rows) for e in batch],
                                      args.max_new_tokens)
                for e, raw in zip(batch, raws):
                    f.write(json.dumps({"idx": e["idx"], "db_id": e["db_id"], "hardness": e["hardness"],
                                        "question": e["question"], "raw_output": raw,
                                        "pred_sql": clean_sql(raw)}, ensure_ascii=False) + "\n")
                f.flush()
                print(f"  {min(i + args.batch_size, len(todo))}/{len(todo)}", flush=True)

    if todo:
        config["generation_seconds"] = round(config.get("generation_seconds", 0) + time.time() - t0, 1)
        config["finished_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    config_file.write_text(json.dumps(config, indent=2))
    score_file(pred_file)


if __name__ == "__main__":
    main()
