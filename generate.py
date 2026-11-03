"""
generate.py — Shared inference engine for NL2SQL on Spider benchmark.

Supports:
  - fp16 (default) and 4-bit quantisation (bitsandbytes)
  - Batched generation
  - Clean SQL extraction from model output
"""

import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(model_name: str, load_in_4bit: bool = False, device: str = "auto"):
    """
    Load a causal LM and its tokenizer.

    Args:
        model_name:   HuggingFace model ID or local path.
        load_in_4bit: Use bitsandbytes 4-bit quantisation (saves ~4× VRAM).
                      Falls back to fp16 if bitsandbytes is unavailable.
        device:       "auto" lets accelerate place layers across available GPUs/CPU.

    Returns:
        (model, tokenizer)
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    # Many instruction-tuned models lack a pad token; reuse eos.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if load_in_4bit:
        try:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map=device,
                trust_remote_code=True,
            )
        except Exception as e:
            print(f"[generate] 4-bit load failed ({e}), falling back to fp16.")
            load_in_4bit = False

    if not load_in_4bit:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map=device,
            trust_remote_code=True,
        )

    model.eval()
    return model, tokenizer


# ---------------------------------------------------------------------------
# SQL extraction
# ---------------------------------------------------------------------------

# Patterns tried in order; first match wins.
_SQL_PATTERNS = [
    # Fenced code block: ```sql ... ``` or ``` ... ```
    re.compile(r"```(?:sql)?\s*(.*?)```", re.DOTALL | re.IGNORECASE),
    # Inline after "SQL:" / "Answer:" / "Query:" label
    re.compile(
        r"(?:sql|answer|query|output)\s*:\s*(select\b.*?)(?:\n\n|\Z)",
        re.DOTALL | re.IGNORECASE,
    ),
    # Bare SELECT … anywhere in the text
    re.compile(r"(select\b.*?)(?:;|\n\n|\Z)", re.DOTALL | re.IGNORECASE),
]


def extract_sql(raw: str) -> str:
    """
    Extract a clean SQL statement from raw model output.

    Tries fenced blocks, labelled lines, then bare SELECT.
    Returns the cleaned string, or the stripped raw text if nothing matches.
    """
    for pattern in _SQL_PATTERNS:
        m = pattern.search(raw)
        if m:
            sql = m.group(1).strip()
            # Collapse internal whitespace runs to single spaces.
            sql = re.sub(r"\s+", " ", sql)
            # Strip trailing semicolon for consistency.
            sql = sql.rstrip(";").strip()
            return sql

    # Last resort: return whatever the model produced, cleaned up.
    return re.sub(r"\s+", " ", raw.strip()).rstrip(";").strip()


# ---------------------------------------------------------------------------
# Batched generation
# ---------------------------------------------------------------------------

def generate_sql(
    prompts: list[str],
    model,
    tokenizer,
    max_new_tokens: int = 256,
    batch_size: int = 4,
    temperature: float = 0.0,
    do_sample: bool = False,
) -> list[str]:
    """
    Run batched generation and return clean SQL for each prompt.

    Args:
        prompts:        List of fully-formed prompt strings.
        model:          Loaded causal LM (from load_model).
        tokenizer:      Matching tokenizer.
        max_new_tokens: Maximum tokens the model may generate per prompt.
        batch_size:     How many prompts to process in one forward pass.
        temperature:    Sampling temperature (ignored when do_sample=False).
        do_sample:      Greedy decode by default (temperature=0 equivalent).

    Returns:
        List of extracted SQL strings, one per prompt.
    """
    results: list[str] = []
    device = next(model.parameters()).device

    for start in range(0, len(prompts), batch_size):
        batch = prompts[start : start + batch_size]

        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        ).to(device)

        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        if do_sample:
            gen_kwargs["temperature"] = temperature

        with torch.no_grad():
            output_ids = model.generate(**inputs, **gen_kwargs)

        # Decode only the newly generated tokens (strip the prompt).
        prompt_len = inputs["input_ids"].shape[1]
        for ids in output_ids:
            new_tokens = ids[prompt_len:]
            raw = tokenizer.decode(new_tokens, skip_special_tokens=True)
            results.append(extract_sql(raw))

    return results


# ---------------------------------------------------------------------------
# Quick smoke-test (run: python generate.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="NL2SQL inference smoke-test")
    parser.add_argument("--model", default="defog/sqlcoder-7b-2", help="HF model ID")
    parser.add_argument("--4bit", dest="use_4bit", action="store_true")
    args = parser.parse_args()

    print(f"Loading {args.model}  (4-bit={args.use_4bit}) …")
    model, tokenizer = load_model(args.model, load_in_4bit=args.use_4bit)

    test_prompts = [
        (
            "### Task\nConvert the question to SQL.\n\n"
            "### Schema\nCREATE TABLE singer (singer_id INT, name TEXT, age INT);\n\n"
            "### Question\nHow many singers are there?\n\n"
            "### SQL\n"
        ),
        (
            "### Task\nConvert the question to SQL.\n\n"
            "### Schema\nCREATE TABLE concert (concert_id INT, theme TEXT, year INT);\n\n"
            "### Question\nList all concert themes.\n\n"
            "### SQL\n"
        ),
    ]

    print("Running generation …")
    outputs = generate_sql(test_prompts, model, tokenizer, batch_size=2)
    for prompt, sql in zip(test_prompts, outputs):
        q = prompt.split("### Question\n")[1].split("\n")[0]
        print(f"  Q: {q}")
        print(f"  SQL: {sql}\n")
