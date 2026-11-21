"""E2: Few-shot Text-to-SQL using retrieved examples (random or similar).

Run on Google Colab with a T4 GPU (see notebooks/run_e1_colab.ipynb for setup):
    # Random retrieval, k=3 (balanced subset)
    python run_e2_fewshot.py --strategy random --k 3

    # Similarity-based retrieval, k=5, with embedding cache
    python run_e2_fewshot.py --strategy similar --k 5 --cache embeddings.npy

    # Ablation: sweep all strategies and k values
    for strategy in random similar; do
      for k in 1 3 5; do
        python run_e2_fewshot.py --strategy $strategy --k $k --cache embeddings.npy
      done
    done

    # 4-bit model, full dev set
    python run_e2_fewshot.py --strategy similar --k 3 --subset full --load-in-4bit

Outputs in results/<run_name>/:
    config.json          everything needed to reproduce the run
    predictions.jsonl    one line per question (retrieved examples + raw output + cleaned SQL)
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
from prompts import FEW_SHOT_VERSION, few_shot_messages
from retrieval import RandomRetriever, SimilarRetriever, load_spider_train

DEFAULT_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
DEFAULT_TRAIN_PATH = "data/spider/train_spider.json"


def git_info() -> dict:
    def run(*cmd):
        try:
            return subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            return None
    status = run("git", "status", "--porcelain", "--untracked-files=no")
    return {
        "commit": run("git", "rev-parse", "HEAD"),
        "uncommitted_changes": bool(status) if status is not None else None,
    }


def load_model(name: str, load_in_4bit: bool):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    kwargs = {}
    if torch.cuda.is_available():
        kwargs["device_map"] = "auto"
        kwargs["dtype"] = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=kwargs["dtype"],
            )
    model = AutoModelForCausalLM.from_pretrained(name, **kwargs)
    model.eval()
    return tok, model


def generate_batch(
    tok, model, batch_messages: list[list[dict]], max_new_tokens: int
) -> list[str]:
    import torch

    texts = [
        tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
        for m in batch_messages
    ]
    enc = tok(texts, return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tok.pad_token_id,
        )
    new_tokens = out[:, enc["input_ids"].shape[1]:]
    return tok.batch_decode(new_tokens, skip_special_tokens=True)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--train", default=DEFAULT_TRAIN_PATH, help="Path to train_spider.json")
    ap.add_argument(
        "--strategy",
        choices=["random", "similar"],
        default="similar",
        help="Retrieval strategy: random baseline or similarity-based",
    )
    ap.add_argument(
        "--k", type=int, default=3, help="Number of few-shot examples to retrieve (1, 3, or 5)"
    )
    ap.add_argument("--cache", default=None, help="Path to save/load sentence embeddings (.npy)")
    ap.add_argument("--subset", choices=["balanced", "full"], default="balanced")
    ap.add_argument("--n-rows", type=int, default=3, help="Sample rows per table in the schema")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--load-in-4bit", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="Only process the first N questions")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import torch
    import transformers
    torch.manual_seed(args.seed)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    examples = load_dev_subset(args.subset)
    if args.limit:
        examples = examples[: args.limit]

    train_examples = load_spider_train(args.train)

    # ------------------------------------------------------------------
    # Build retriever
    # ------------------------------------------------------------------
    if args.strategy == "random":
        retriever = RandomRetriever(train_examples, seed=args.seed)
    else:
        retriever = SimilarRetriever(train_examples, cache_path=args.cache)

    # ------------------------------------------------------------------
    # Output directory
    # ------------------------------------------------------------------
    model_short = args.model.split("/")[-1]
    run_name = (
        args.run_name
        or f"e2_fewshot_{args.strategy}_k{args.k}_{model_short}_{args.subset}"
    )
    out_dir = Path("results") / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_file = out_dir / "predictions.jsonl"

    done = {r["idx"] for r in read_jsonl(pred_file)} if pred_file.exists() else set()
    todo = [e for e in examples if e["idx"] not in done]
    print(f"{run_name}: {len(examples)} questions, {len(done)} already done, {len(todo)} to go")

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------
    config = {
        "experiment": "E2 few-shot",
        "model": args.model,
        "load_in_4bit": args.load_in_4bit,
        "prompt_version": FEW_SHOT_VERSION,
        "schema_format": "CREATE TABLE + sample rows",
        "n_sample_rows": args.n_rows,
        "retrieval": {
            "strategy": args.strategy,
            "k": args.k,
            "train_path": args.train,
            "cache_path": args.cache,
        },
        "decoding": {"do_sample": False, "max_new_tokens": args.max_new_tokens},
        "batch_size": args.batch_size,
        "subset": args.subset,
        "n_examples": len(examples),
        "seed": args.seed,
        "git": git_info(),
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    config_file = out_dir / "config.json"
    if config_file.exists():
        old = json.loads(config_file.read_text())
        for key in ("model", "load_in_4bit", "prompt_version", "n_sample_rows", "decoding", "subset", "retrieval", "seed"):
            if old.get(key) != config[key]:
                raise SystemExit(
                    f"{config_file} has {key}={old.get(key)!r}, this run has {config[key]!r}. "
                    "Use a different --run-name."
                )
        old["n_resumes"] = old.get("n_resumes", 0) + (1 if todo else 0)
        old["n_examples"] = max(old["n_examples"], config["n_examples"])
        config = old

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------
    t0 = time.time()
    if todo:
        tok, model = load_model(args.model, args.load_in_4bit)
        with open(pred_file, "a", encoding="utf-8") as f:
            for i in range(0, len(todo), args.batch_size):
                batch = todo[i : i + args.batch_size]

                # Retrieve few-shot examples for each question in the batch.
                batch_retrieved = [
                    retriever.retrieve(e["question"], k=args.k, db_id=e["db_id"])
                    for e in batch
                ]
                batch_messages = [
                    few_shot_messages(e, retrieved, n_rows=args.n_rows)
                    for e, retrieved in zip(batch, batch_retrieved)
                ]

                raws = generate_batch(tok, model, batch_messages, args.max_new_tokens)

                for e, raw, retrieved in zip(batch, raws, batch_retrieved):
                    record = {
                        "idx": e["idx"],
                        "db_id": e["db_id"],
                        "hardness": e["hardness"],
                        "question": e["question"],
                        "retrieved_questions": [ex.question for ex in retrieved],
                        "raw_output": raw,
                        "pred_sql": clean_sql(raw),
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
                print(f"  {min(i + args.batch_size, len(todo))}/{len(todo)}", flush=True)

    if todo:
        config["generation_seconds"] = round(
            config.get("generation_seconds", 0) + time.time() - t0, 1
        )
        config["finished_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    config_file.write_text(json.dumps(config, indent=2) + "\n")
    score_file(pred_file)


if __name__ == "__main__":
    main()
