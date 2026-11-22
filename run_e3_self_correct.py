
"""E3 / E5: Text-to-SQL with SQL execution and self-correction.

E3: few-shot prompt (retrieved examples, as E2) + self-correction, base model
    python run_e3_self_correct.py --strategy similar --k 3

E5: fine-tuned model + self-correction. Pass the Part 4 LoRA adapter. The adapter was
    trained on the zero-shot prompt (as E4), so E5 uses that prompt, not few-shot.
    python run_e3_self_correct.py --adapter /content/drive/MyDrive/ft_runs/ft_1p5b/adapter

Outputs in results/<run_name>/:
    config.json          settings + totals (extra LLM calls, generation/execution seconds, fixes)
    predictions.jsonl    final SQL, initial SQL and cost per question (git-ignored)
    execution_logs.jsonl every attempt with its SQLite error and result
    metrics.json, eval_details.jsonl   same format as E1/E2
The run resumes automatically if predictions.jsonl is partly written.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import transformers

from data_utils import db_path
from evaluate import (
    clean_sql,
    load_dev_subset,
    read_jsonl,
    score_file,
)
from prompts import FEW_SHOT_VERSION, ZERO_SHOT_VERSION, few_shot_messages, zero_shot_messages
from retrieval import RandomRetriever, SimilarRetriever, load_spider_train
from run_e2_fewshot import DEFAULT_MODEL, git_info, load_model
from self_correct import SELF_CORRECT_VERSION, self_correct


def load_with_adapter(base_name: str, adapter_dir: str, load_in_4bit: bool):
    """Base model (same loading as E2) with the Part 4 LoRA adapter on top (E5)."""
    from peft import PeftModel

    if not (Path(adapter_dir) / "adapter_config.json").exists():
        raise SystemExit(f"No adapter_config.json in {adapter_dir}. Check the --adapter path.")
    tokenizer, model = load_model(base_name, load_in_4bit)
    model = PeftModel.from_pretrained(model, adapter_dir)
    if not load_in_4bit:
        model = model.merge_and_unload()  # fold the adapter into the weights: faster generation
    model.eval()
    return tokenizer, model


def main():
    parser = argparse.ArgumentParser(
        description="E3 (few-shot) / E5 (fine-tuned adapter) + SQL self-correction"
    )
    parser.add_argument(
        "--adapter",
        default=None,
        help="LoRA adapter folder from Part 4: runs E5 instead of E3",
    )

    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--train",
        default="data/spider_data/train_spider.json",
    )
    parser.add_argument(
        "--strategy",
        choices=["random", "similar"],
        default="similar",
    )
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--cache", default=None)
    parser.add_argument(
        "--subset",
        choices=["balanced", "full"],
        default="balanced",
    )
    parser.add_argument("--n-rows", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if args.k < 1:
        parser.error("--k must be at least 1")

    if args.max_retries < 0:
        parser.error("--max-retries cannot be negative")

    if args.timeout <= 0:
        parser.error("--timeout must be positive")

    torch.manual_seed(args.seed)

    # 1. Load dev and training data
    examples = load_dev_subset(args.subset)

    if args.limit is not None:
        examples = examples[:args.limit]

    # 2. Retriever (E3 only: E5 uses the zero-shot prompt the adapter was trained on)
    retriever = None
    if args.adapter is None:
        train_examples = load_spider_train(args.train)
        if args.strategy == "random":
            retriever = RandomRetriever(train_examples, seed=args.seed)
        else:
            retriever = SimilarRetriever(train_examples, cache_path=args.cache)

    # 3. Create output directory
    model_short = args.model.split("/")[-1]

    if args.adapter is None:
        default_name = (
            f"e3_self_correct_{args.strategy}_k{args.k}_{model_short}_{args.subset}"
        )
    else:
        default_name = f"e5_self_correct_finetuned_{model_short}_{args.subset}"
    run_name = args.run_name or default_name
    if args.limit is not None:
        run_name += f"_limit{args.limit}"  # a quick test never mixes into the real run

    out_dir = Path("results") / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_file = out_dir / "predictions.jsonl"
    log_file = out_dir / "execution_logs.jsonl"
    config_file = out_dir / "config.json"

    # 4. Resume support
    existing = read_jsonl(pred_file) if pred_file.exists() else []
    done = {r["idx"] for r in existing}
    todo = [e for e in examples if e["idx"] not in done]

    print(
        f"{run_name}: {len(examples)} questions, "
        f"{len(done)} already completed, {len(todo)} remaining"
    )

    # 5. Experiment configuration
    config = {
        "experiment": (
            "E3 few-shot + self-correction"
            if args.adapter is None
            else "E5 fine-tuned + self-correction"
        ),
        "model": args.model,
        "adapter": args.adapter,
        "load_in_4bit": args.load_in_4bit,
        "prompt_version": (
            FEW_SHOT_VERSION if args.adapter is None else ZERO_SHOT_VERSION
        ),
        "self_correct_version": SELF_CORRECT_VERSION,
        "max_retries": args.max_retries,
        "execution_timeout": args.timeout,
        "retry_on_empty_result": True,
        "retrieval": None if args.adapter is not None else {
            "strategy": args.strategy,
            "k": args.k,
            "train_path": args.train,
            "cache_path": args.cache,
        },
        "schema_format": "CREATE TABLE + sample rows",
        "n_sample_rows": args.n_rows,
        "decoding": {
            "do_sample": False,
            "max_new_tokens": args.max_new_tokens,
        },
        "subset": args.subset,
        "n_examples": len(examples),
        "seed": args.seed,
        "git": git_info(),
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "gpu": (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else None
        ),
        "started_at": datetime.now(timezone.utc).isoformat(
            timespec="seconds"
        ),
    }

    # Why this check: runs take hours on a free Colab GPU and get disconnected, so we resume.
    # Mixing predictions made with different settings (model, retries, prompt...) would give
    # numbers nobody can reproduce, so a changed setting stops the run instead.
    if config_file.exists():
        old = json.loads(config_file.read_text(encoding="utf-8"))

        keys_to_check = [
            "model",
            "adapter",
            "load_in_4bit",
            "prompt_version",
            "n_sample_rows",
            "subset",
            "retrieval",
            "seed",
            "max_retries",
            "execution_timeout",
            "decoding",
        ]

        for key in keys_to_check:
            if old.get(key) != config[key]:
                raise SystemExit(
                    f"Existing config differs for {key}. "
                    "Choose a different --run-name."
                )

        config = old

    # 6. Load model only if there are questions to process
    if todo:
        if args.adapter is None:
            tokenizer, model = load_model(args.model, args.load_in_4bit)
        else:
            tokenizer, model = load_with_adapter(
                args.model, args.adapter, args.load_in_4bit
            )

        total_generation_time = 0.0
        total_execution_time = 0.0
        total_extra_calls = 0
        fixed_execution_errors = 0

        with (
            open(pred_file, "a", encoding="utf-8") as pred_out,
            open(log_file, "a", encoding="utf-8") as log_out,
        ):
            for position, example in enumerate(todo, start=1):

                # E3 uses the same few-shot prompt as E2, so "E3 minus E2" measures only the
                # self-correction. E5 uses the zero-shot prompt because the Part 4 adapter was
                # trained on exactly that prompt (same as E4); a few-shot prompt would be
                # unfamiliar to it, and "E5 minus E4" would then mix two changes.
                if retriever is not None:
                    retrieved = retriever.retrieve(
                        example["question"],
                        k=args.k,
                        db_id=example["db_id"],
                    )
                    messages = few_shot_messages(
                        example, retrieved, n_rows=args.n_rows
                    )
                else:
                    retrieved = []
                    messages = zero_shot_messages(example, args.n_rows)

                # Run generation, execution and correction
                start_time = time.perf_counter()

                result = self_correct(
                    messages=messages,
                    db_file=db_path(example["db_id"]),
                    model=model,
                    tokenizer=tokenizer,
                    max_retries=args.max_retries,
                    max_new_tokens=args.max_new_tokens,
                    timeout=args.timeout,
                )

                wall_time = time.perf_counter() - start_time

                attempts = result["attempts"]
                initial_sql = (
                    attempts[0]["sql"] if attempts else ""
                )
                final_sql = clean_sql(result["sql"])

                generation_time = result.get("generation_time", 0.0)
                extra_calls = result.get("extra_llm_calls", 0)

                execution_time = sum(
                    a.get("execution_time", 0.0)
                    for a in attempts
                )

                # "Fixed" = the first SQL crashed and the final SQL runs. We count this
                # separately from EX because it is what self-correction can directly do
                # (it sees errors, not the gold answer); EX also needs the right result.
                initial_failed = (
                    bool(attempts)
                    and not attempts[0]["success"]
                )
                finally_succeeded = result["success"]

                fixed = initial_failed and finally_succeeded

                if fixed:
                    fixed_execution_errors += 1

                total_generation_time += generation_time
                total_execution_time += execution_time
                total_extra_calls += extra_calls

                record = {
                    "idx": example["idx"],
                    "db_id": example["db_id"],
                    "hardness": example["hardness"],
                    "question": example["question"],
                    "retrieved_questions": [
                        ex.question for ex in retrieved
                    ],
                    "initial_sql": initial_sql,
                    "initial_empty": bool(attempts) and attempts[0]["empty"],
                    "pred_sql": final_sql,
                    "execution_success": result["success"],
                    "execution_error": result["error"],
                    "attempts": result["total_attempts"],
                    "llm_calls": result["llm_calls"],
                    "extra_llm_calls": extra_calls,
                    "generation_time": generation_time,
                    "execution_time": execution_time,
                    "wall_time": wall_time,
                    "execution_error_fixed": fixed,
                }

                log_record = {
                    "idx": example["idx"],
                    "question": example["question"],
                    "attempt_details": attempts,
                }

                pred_out.write(
                    json.dumps(record, ensure_ascii=False) + "\n"
                )
                log_out.write(
                    json.dumps(log_record, ensure_ascii=False) + "\n"
                )

                pred_out.flush()
                log_out.flush()

                print(
                    f"  {position}/{len(todo)} "
                    f"| attempts={result['total_attempts']} "
                    f"| extra calls={extra_calls} "
                    f"| execution success={result['success']}",
                    flush=True,
                )

        # 7. Save run-level metrics
        config["generation_seconds"] = round(
            config.get("generation_seconds", 0.0)
            + total_generation_time,
            2,
        )
        config["execution_seconds"] = round(
            config.get("execution_seconds", 0.0)
            + total_execution_time,
            2,
        )
        config["extra_llm_calls"] = (
            config.get("extra_llm_calls", 0)
            + total_extra_calls
        )
        config["execution_errors_fixed"] = (
            config.get("execution_errors_fixed", 0)
            + fixed_execution_errors
        )

    config["finished_at"] = datetime.now(
        timezone.utc
    ).isoformat(timespec="seconds")

    config_file.write_text(
        json.dumps(config, indent=2) + "\n",
        encoding="utf-8",
    )

    # 8. Evaluate final SQL using the existing Spider evaluator
    score_file(pred_file, timeout=args.timeout)

    print(f"\nE3 completed. Results saved in: {out_dir}")


if __name__ == "__main__":
    main()
