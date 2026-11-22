"""
prepare_ft_data.py  (Part 4: LoRA fine-tuning)

What this file does, in simple words:
    Fine-tuning needs many examples of "input -> correct output".
    Spider gives us: question + database + correct SQL.
    This script turns every Spider training example into:
        prompt     = instruction + database schema + question
        completion = the correct SQL
    and saves them in a .jsonl file (one example per line) for finetune.py.

IMPORTANT: the prompt is NOT written here any more. We reuse Part 1's
zero_shot_messages() from prompts.py (branch part1-data-eval-zeroshot), so the
fine-tuning prompt is exactly the E1 prompt. If Part 1 changes it, just rerun this.
Part 1's files (data_utils.py, prompts.py) must be in --team_code_dir.

Usage:
    python src/prepare_ft_data.py --spider_dir data/spider_data --out_dir data/ft --team_code_dir team_code
"""

import argparse
import json
import os
import random
import sys

# ---------------------------------------------------------------------------
# 1. Load Spider train files and build examples
# ---------------------------------------------------------------------------

def clean_sql(sql: str) -> str:
    """Make gold SQL one tidy line (collapse extra spaces/newlines, drop trailing ';')."""
    return " ".join(sql.strip().rstrip(";").split())


def load_examples(spider_dir: str, files, num_sample_rows: int, zero_shot_messages):
    examples, skipped = [], 0
    for fname in files:
        path = os.path.join(spider_dir, fname)
        if not os.path.exists(path):
            print(f"  (skipping {fname}: file not found)")
            continue
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            db_id = item["db_id"]
            db_path = os.path.join(spider_dir, "database", db_id, f"{db_id}.sqlite")
            if not os.path.exists(db_path):
                skipped += 1
                continue
            # Same function E1 uses -> same system message, schema text and question layout.
            # "messages" = [system, user]. finetune.py turns it into text with the model's chat template.
            messages = zero_shot_messages(item, num_sample_rows)
            examples.append({
                "db_id": db_id,
                "question": item["question"],
                "messages": messages,
                "completion": clean_sql(item["query"]),
            })
    return examples, skipped


def write_jsonl(path: str, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser(description="Make fine-tuning data from Spider train set.")
    ap.add_argument("--spider_dir", default="data/spider_data",
                    help="folder with train_spider.json and database/ (must be named spider_data, Part 1's code expects that)")
    ap.add_argument("--team_code_dir", default="team_code",
                    help="folder with Part 1's data_utils.py and prompts.py")
    ap.add_argument("--out_dir", default="data/ft", help="where to save train.jsonl / val.jsonl")
    ap.add_argument("--files", nargs="+", default=["train_spider.json", "train_others.json"],
                    help="Spider train files to use")
    ap.add_argument("--sample_rows", type=int, default=3, help="example rows per table in the prompt")
    ap.add_argument("--val_db_fraction", type=float, default=0.1,
                    help="fraction of DATABASES kept aside to watch validation loss (NOT the Spider dev set)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # Point Part 1's code at our Spider folder, then import it.
    spider_dir = os.path.abspath(args.spider_dir)
    if os.path.basename(spider_dir) != "spider_data":
        raise SystemExit("--spider_dir must end with 'spider_data' (Part 1's data_utils.py looks for that name).")
    os.environ["SPIDER_DATA_ROOT"] = os.path.dirname(spider_dir)
    sys.path.insert(0, os.path.abspath(args.team_code_dir))
    try:
        from prompts import zero_shot_messages
    except ImportError as e:
        raise SystemExit(f"Cannot import Part 1's prompts.py from '{args.team_code_dir}': {e}")

    examples, skipped = load_examples(spider_dir, args.files, args.sample_rows, zero_shot_messages)
    if not examples:
        raise SystemExit("No examples found. Check --spider_dir.")

    # Split by DATABASE, not by question, so validation uses databases the model never trained on.
    # (This copies the real task: the test databases are unseen.)
    rng = random.Random(args.seed)
    db_ids = sorted({e["db_id"] for e in examples})
    rng.shuffle(db_ids)
    n_val_dbs = max(1, round(len(db_ids) * args.val_db_fraction)) if args.val_db_fraction > 0 else 0
    val_dbs = set(db_ids[:n_val_dbs])
    train = [e for e in examples if e["db_id"] not in val_dbs]
    val = [e for e in examples if e["db_id"] in val_dbs]
    rng.shuffle(train)

    os.makedirs(args.out_dir, exist_ok=True)
    write_jsonl(os.path.join(args.out_dir, "train.jsonl"), train)
    write_jsonl(os.path.join(args.out_dir, "val.jsonl"), val)

    print(f"Total examples : {len(examples)}  (skipped, missing database: {skipped})")
    print(f"Train          : {len(train)}  from {len(db_ids) - len(val_dbs)} databases")
    print(f"Validation     : {len(val)}  from {len(val_dbs)} held-out databases")
    print(f"Saved to       : {args.out_dir}/train.jsonl and val.jsonl")
    print("\n--- one example ---")
    for m in train[0]["messages"]:
        print(f"[{m['role']}]\n{m['content']}\n")
    print("[assistant]\n" + train[0]["completion"])


if __name__ == "__main__":
    main()
