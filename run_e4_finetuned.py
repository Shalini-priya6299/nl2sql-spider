"""E4: zero-shot Text-to-SQL with the LoRA fine-tuned model.

Same as E1 (run_e1_zeroshot.py): same prompt (zeroshot-v1), same dev subset, same greedy
decoding, same scoring. The ONLY difference is that the LoRA adapter from Part 4 is loaded
on top of the base model. So any change in accuracy comes from fine-tuning.

Run on Google Colab with a T4 GPU:
    python run_e4_finetuned.py --adapter /content/drive/MyDrive/ft_runs/ft_1p5b/adapter
    python run_e4_finetuned.py --adapter <adapter> --subset full     # all 1034 dev questions
    python run_e4_finetuned.py --adapter <adapter> --limit 8         # quick test (8 questions)

Outputs in results/<run_name>/ (same files as E1):
    config.json, predictions.jsonl, metrics.json, eval_details.jsonl
The run resumes automatically if predictions.jsonl is partly written.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

from evaluate import clean_sql, load_dev_subset, read_jsonl, score_file
from prompts import ZERO_SHOT_VERSION, zero_shot_messages
from run_e1_zeroshot import DEFAULT_MODEL, generate_batch, git_info


def load_finetuned(base_name: str, adapter_dir: str, load_in_4bit: bool):
    """Load the base model (same way as E1) and put the LoRA adapter on top."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(base_name)
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
    model = AutoModelForCausalLM.from_pretrained(base_name, **kwargs)
    model = PeftModel.from_pretrained(model, adapter_dir)
    if not load_in_4bit:
        model = model.merge_and_unload()  # fold the adapter into the weights: faster generation
    model.eval()
    return tok, model


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter", required=True, help="folder with adapter_model.safetensors")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="base model the adapter was trained on")
    ap.add_argument("--subset", choices=["balanced", "full"], default="balanced")
    ap.add_argument("--n-rows", type=int, default=3, help="sample rows per table in the prompt")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--load-in-4bit", action="store_true",
                    help="load the base in 4-bit like during training (E1 used fp16, so leave off to compare)")
    ap.add_argument("--limit", type=int, default=None, help="only the first N questions (quick test)")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import peft
    import torch
    import transformers
    torch.manual_seed(args.seed)

    if not (Path(args.adapter) / "adapter_config.json").exists():
        raise SystemExit(f"No adapter_config.json in {args.adapter}. Check the --adapter path.")

    examples = load_dev_subset(args.subset)
    if args.limit:
        examples = examples[: args.limit]
    run_name = args.run_name or f"e4_finetuned_{args.model.split('/')[-1]}_{args.subset}"
    if args.limit:
        run_name += f"_limit{args.limit}"  # a quick test never mixes into the real run
    out_dir = Path("results") / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_file = out_dir / "predictions.jsonl"

    done = {r["idx"] for r in read_jsonl(pred_file)} if pred_file.exists() else set()
    todo = [e for e in examples if e["idx"] not in done]
    print(f"{run_name}: {len(examples)} questions, {len(done)} already done, {len(todo)} to go")

    config = {
        "experiment": "E4 zero-shot, LoRA fine-tuned",
        "model": args.model,
        "adapter": args.adapter,
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
                     "transformers": transformers.__version__, "peft": peft.__version__},
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    config_file = out_dir / "config.json"
    if config_file.exists():  # resumed run: keep the original settings, only add time
        old = json.loads(config_file.read_text())
        for k in ("model", "adapter", "load_in_4bit", "prompt_version", "n_sample_rows", "decoding", "subset"):
            if old.get(k) != config[k]:
                raise SystemExit(f"{config_file} has {k}={old.get(k)!r}, this run has {config[k]!r}. "
                                 "Use a different --run-name.")
        old["n_resumes"] = old.get("n_resumes", 0) + (1 if todo else 0)
        old["n_examples"] = max(old["n_examples"], config["n_examples"])
        config = old

    t0 = time.time()
    if todo:
        tok, model = load_finetuned(args.model, args.adapter, args.load_in_4bit)
        with open(pred_file, "a", encoding="utf-8") as f:
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
    config_file.write_text(json.dumps(config, indent=2) + "\n")
    score_file(pred_file)


if __name__ == "__main__":
    main()
