"""
finetune.py  (Part 4: LoRA / QLoRA fine-tuning)

What this file does, in simple words:
    1. Loads a small code model (Qwen2.5-Coder) in 4-bit  -> uses little GPU memory   (the "Q")
    2. Adds small trainable "sticky notes" (LoRA adapters) -> only these are trained   (the "LoRA")
    3. Trains on train.jsonl made by prepare_ft_data.py
    4. Saves ONLY the adapter (a few MB), plus the loss log for our loss-curve plot.

Run on Kaggle/Colab (needs a GPU). Quick test first with the 1.5B model and --max_steps 20:
    python src/finetune.py --model Qwen/Qwen2.5-Coder-1.5B-Instruct --max_steps 20 --out_dir outputs/ft_test
Final run with the 7B model:
    python src/finetune.py --model Qwen/Qwen2.5-Coder-7B-Instruct --out_dir outputs/ft_7b

Install:  pip install transformers peft bitsandbytes accelerate datasets
"""

import argparse
import json
import os
import time

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)


def parse_args():
    ap = argparse.ArgumentParser(description="QLoRA fine-tuning on Spider (text-to-SQL).")
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    ap.add_argument("--data_dir", default="data/ft", help="folder with train.jsonl and val.jsonl")
    ap.add_argument("--out_dir", default="outputs/ft_run", help="adapter + logs are saved here")
    # LoRA settings
    ap.add_argument("--lora_r", type=int, default=16, help="size of the sticky notes (bigger = more to learn)")
    ap.add_argument("--lora_alpha", type=int, default=32, help="strength of the sticky notes")
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    # Training settings
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch_size", type=int, default=2, help="examples per GPU step (keep small for memory)")
    ap.add_argument("--grad_accum", type=int, default=8, help="batch_size x grad_accum = real batch size")
    ap.add_argument("--max_len", type=int, default=1536, help="longer examples are dropped, not cut (Spider schemas can be long)")
    ap.add_argument("--max_steps", type=int, default=-1, help="set e.g. 20 for a quick test; -1 = train fully")
    ap.add_argument("--save_steps", type=int, default=200, help="checkpoint often: free GPU sessions disconnect")
    ap.add_argument("--resume", action="store_true", help="continue from the latest checkpoint in out_dir")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def make_tokenize_fn(tokenizer, max_len):
    """
    Turn one example into numbers the model understands.
    The model should learn to write the SQL, not to repeat the prompt, so we set the
    prompt part of 'labels' to -100 (this means: "ignore these tokens when computing loss").
    """
    def tokenize(example):
        # Same call E1 uses at test time (run_e1_zeroshot.py): chat template + generation prompt.
        prompt_text = tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=True
        )
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        answer_ids = tokenizer(example["completion"], add_special_tokens=False)["input_ids"]
        answer_ids = answer_ids + [tokenizer.eos_token_id]  # teach the model when to stop

        # No cutting here: cutting from the end would remove the SQL answer itself.
        # Examples that are too long are dropped later (see 'keep_short' in main()).
        input_ids = prompt_ids + answer_ids
        labels = [-100] * len(prompt_ids) + answer_ids
        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": labels,
        }
    return tokenize


def main():
    args = parse_args()
    assert torch.cuda.is_available(), "No GPU found. Run this on Kaggle/Colab with a GPU turned on."
    os.makedirs(args.out_dir, exist_ok=True)

    # ---- 1. Tokenizer ------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- 2. Model in 4-bit (QLoRA) ----------------------------------------
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",           # a 4-bit format that keeps accuracy well
        bnb_4bit_compute_dtype=torch.float16,  # T4/P100 GPUs do not support bfloat16
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model, quantization_config=bnb_config, device_map={"": 0}
    )
    model = prepare_model_for_kbit_training(model)  # gets the 4-bit model ready for training

    # ---- 3. Add LoRA adapters ---------------------------------------------
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()  # prints how small the trainable part is (good for viva)

    # ---- 4. Data -----------------------------------------------------------
    data = load_dataset(
        "json",
        data_files={
            "train": os.path.join(args.data_dir, "train.jsonl"),
            "validation": os.path.join(args.data_dir, "val.jsonl"),
        },
    )
    tok_fn = make_tokenize_fn(tokenizer, args.max_len)
    data = data.map(tok_fn, remove_columns=data["train"].column_names)

    # Drop (not cut) examples longer than max_len, so the SQL answer is never chopped off.
    for split in ("train", "validation"):
        before = len(data[split])
        data[split] = data[split].filter(lambda ex: len(ex["input_ids"]) <= args.max_len)
        print(f"{split}: kept {len(data[split])} of {before} (dropped {before - len(data[split])} too long)")

    # ---- 5. Train ----------------------------------------------------------
    training_args = TrainingArguments(
        output_dir=args.out_dir,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=25,                    # newer transformers removed warmup_ratio
        fp16=True,
        gradient_checkpointing=True,         # saves memory by recomputing instead of storing
        optim="paged_adamw_8bit",            # memory-friendly optimizer
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=args.save_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        report_to="none",
        seed=args.seed,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=data["train"],
        eval_dataset=data["validation"],
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100),
    )

    start = time.time()
    trainer.train(resume_from_checkpoint=True if args.resume else None)
    hours = (time.time() - start) / 3600

    # ---- 6. Save adapter + records ----------------------------------------
    adapter_dir = os.path.join(args.out_dir, "adapter")
    trainer.model.save_pretrained(adapter_dir)   # only the small adapter, not the whole model
    tokenizer.save_pretrained(adapter_dir)

    with open(os.path.join(args.out_dir, "loss_log.json"), "w") as f:
        json.dump(trainer.state.log_history, f, indent=2)  # for the loss-curve plot

    config = vars(args) | {
        "train_hours": round(hours, 3),
        "gpu": torch.cuda.get_device_name(0),
        "train_examples": len(data["train"]),
    }
    with open(os.path.join(args.out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)  # rules: save settings with every run

    print(f"\nDone in {hours:.2f} GPU-hours. Adapter saved to {adapter_dir}")


if __name__ == "__main__":
    main()
