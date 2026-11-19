"""Spider data utilities: download, loading, and schema-to-prompt conversion.

Shared by all parts of the project (E1-E5). Typical use:

    from data_utils import download_spider, load_split, load_schemas, schema_to_prompt, db_path

    download_spider("data")                     # once (use only_dev=True on low disk)
    dev = load_split("dev")                     # list of examples (with "hardness")
    schemas = load_schemas()                    # db_id -> schema dict
    text = schema_to_prompt(schemas["concert_singer"], db_path("concert_singer"))

Command line:
    python data_utils.py download [--only-dev]
    python data_utils.py stats
    python data_utils.py show concert_singer
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import zipfile
from collections import Counter
from functools import lru_cache
from pathlib import Path

from hardness import eval_hardness

# Official Spider 1.0 release (yale-lily.github.io/spider -> Google Drive).
SPIDER_GDRIVE_ID = "1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J"

DATA_ROOT = Path(os.environ.get("SPIDER_DATA_ROOT", Path(__file__).parent / "data"))
SPIDER_DIR = DATA_ROOT / "spider_data"

SPLIT_FILES = {
    # Spider's official training set = train_spider.json + train_others.json
    "train": ["train_spider.json", "train_others.json"],
    "train_spider": ["train_spider.json"],
    "dev": ["dev.json"],
}

HARDNESS_LEVELS = ["easy", "medium", "hard", "extra"]


# --------------------------------------------------------------------------- #
# Download
# --------------------------------------------------------------------------- #
def download_spider(dest: str | Path = DATA_ROOT, only_dev: bool = False,
                    zip_path: str | Path | None = None, keep_zip: bool = False) -> Path:
    """Download and extract Spider 1.0 into ``dest/spider_data``.

    only_dev=True extracts the JSON files plus only the 20 dev databases
    (~110 MB instead of ~1.8 GB). Pass ``zip_path`` to reuse an already
    downloaded ``spider_data.zip``.
    """
    dest = Path(dest)
    out_dir = dest / "spider_data"
    if (out_dir / "dev.json").exists() and (out_dir / "database").exists():
        print(f"Spider already present at {out_dir}")
        return out_dir

    dest.mkdir(parents=True, exist_ok=True)
    if zip_path is None:
        zip_path = dest / "spider_data.zip"
        if not Path(zip_path).exists():
            import gdown  # pip install gdown
            gdown.download(id=SPIDER_GDRIVE_ID, output=str(zip_path), quiet=False)
    zip_path = Path(zip_path)

    with zipfile.ZipFile(zip_path) as zf:
        dev = json.loads(zf.read("spider_data/dev.json"))
        dev_dbs = {e["db_id"] for e in dev}
        members = []
        for name in zf.namelist():
            if name.startswith("__MACOSX") or name.endswith(".DS_Store"):
                continue
            if not name.startswith("spider_data/"):
                continue
            parts = name.split("/")
            if parts[1] == "test_database":
                continue  # test databases are not needed (we never evaluate on test)
            if only_dev and parts[1] == "database" and len(parts) > 2 and parts[2] not in dev_dbs:
                continue
            members.append(name)
        zf.extractall(dest, members=members)

    if not keep_zip and zip_path.parent == dest:
        zip_path.unlink()
    print(f"Extracted Spider to {out_dir} ({'dev databases only' if only_dev else 'all databases'})")
    return out_dir


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def db_path(db_id: str, spider_dir: Path = SPIDER_DIR) -> Path:
    return Path(spider_dir) / "database" / db_id / f"{db_id}.sqlite"


def load_split(split: str = "dev", spider_dir: Path = SPIDER_DIR) -> list[dict]:
    """Load a split. Each example gets ``idx`` (position in the split) and ``hardness``."""
    examples = []
    for fname in SPLIT_FILES[split]:
        with open(Path(spider_dir) / fname) as f:
            examples.extend(json.load(f))
    for i, ex in enumerate(examples):
        ex["idx"] = i
        ex["hardness"] = eval_hardness(ex["sql"])
    return examples


@lru_cache(maxsize=None)
def _load_tables(spider_dir: str) -> dict:
    with open(Path(spider_dir) / "tables.json") as f:
        return {t["db_id"]: t for t in json.load(f)}


def load_schemas(spider_dir: Path = SPIDER_DIR) -> dict[str, dict]:
    """db_id -> raw Spider schema entry from tables.json."""
    return _load_tables(str(spider_dir))


# --------------------------------------------------------------------------- #
# Schema -> prompt
# --------------------------------------------------------------------------- #
def _quote(name: str) -> str:
    """Quote identifiers that are not plain words (spaces, keywords, etc.)."""
    if name.replace("_", "").isalnum() and not name[0].isdigit():
        return name
    return '"' + name.replace('"', '""') + '"'


def _sql_type(t: str) -> str:
    return {"text": "TEXT", "number": "NUMBER", "time": "TIME", "boolean": "BOOLEAN",
            "others": "TEXT"}.get(t, t.upper())


def create_table_statements(schema: dict) -> list[str]:
    """Build CREATE TABLE statements (columns, primary keys, foreign keys) from tables.json.

    Uses the *original* table/column names, i.e. the names the SQL must use.
    """
    tables = schema["table_names_original"]
    cols = schema["column_names_original"]  # [(table_idx, col_name)], index 0 is (-1, "*")
    types = schema["column_types"]
    pks = schema["primary_keys"]
    pk_set = set()
    for pk in pks:  # an entry may be an int or a list (composite key)
        pk_set.update(pk if isinstance(pk, list) else [pk])

    stmts = []
    for t_idx, t_name in enumerate(tables):
        lines = []
        t_pks = []
        for c_idx, (ct, c_name) in enumerate(cols):
            if ct != t_idx:
                continue
            lines.append(f"  {_quote(c_name)} {_sql_type(types[c_idx])}")
            if c_idx in pk_set:
                t_pks.append(_quote(c_name))
        if t_pks:
            lines.append(f"  PRIMARY KEY ({', '.join(t_pks)})")
        for src, tgt in schema["foreign_keys"]:
            if cols[src][0] == t_idx:
                lines.append(
                    f"  FOREIGN KEY ({_quote(cols[src][1])}) REFERENCES "
                    f"{_quote(tables[cols[tgt][0]])}({_quote(cols[tgt][1])})"
                )
        stmts.append(f"CREATE TABLE {_quote(t_name)} (\n" + ",\n".join(lines) + "\n);")
    return stmts


def _fmt_value(v, max_len: int = 40) -> str:
    if v is None:
        return "NULL"
    if isinstance(v, bytes):
        v = v.decode("utf-8", errors="replace")
    s = str(v).replace("\n", " ")
    return s if len(s) <= max_len else s[: max_len - 3] + "..."


def sample_rows(db_file: Path, table: str, n: int = 3) -> tuple[list[str], list[tuple]]:
    conn = sqlite3.connect(f"file:{db_file}?mode=ro", uri=True)
    conn.text_factory = lambda b: b.decode("utf-8", errors="replace")
    try:
        cur = conn.execute(f"SELECT * FROM {_quote(table)} LIMIT {int(n)}")
        header = [d[0] for d in cur.description]
        return header, cur.fetchall()
    except sqlite3.Error:
        return [], []
    finally:
        conn.close()


def schema_to_prompt(schema: dict, db_file: Path | None = None, n_rows: int = 3) -> str:
    """Schema text: CREATE TABLE statements, each followed by ``n_rows`` sample rows.

    Format follows Rajkumar et al. 2022 ("Evaluating the Text-to-SQL Capabilities
    of Large Language Models"): CREATE TABLE + ``SELECT * ... LIMIT 3`` comment.
    Set n_rows=0 (or db_file=None) for schema-only prompts.
    """
    blocks = []
    for t_name, stmt in zip(schema["table_names_original"], create_table_statements(schema)):
        block = stmt
        if db_file is not None and n_rows > 0:
            header, rows = sample_rows(db_file, t_name, n_rows)
            if header:
                body = "\n".join("\t".join(_fmt_value(v) for v in r) for r in rows)
                block += (f"\n/*\n{len(rows)} example rows:\n"
                          f"SELECT * FROM {_quote(t_name)} LIMIT {n_rows};\n"
                          + "\t".join(header) + ("\n" + body if body else "") + "\n*/")
        blocks.append(block)
    return "\n\n".join(blocks)


def get_schema_prompt(db_id: str, n_rows: int = 3, spider_dir: Path = SPIDER_DIR) -> str:
    """Convenience wrapper: schema prompt for a db_id (cached)."""
    return _schema_prompt_cached(db_id, n_rows, str(spider_dir))


@lru_cache(maxsize=None)
def _schema_prompt_cached(db_id: str, n_rows: int, spider_dir: str) -> str:
    schema = load_schemas(Path(spider_dir))[db_id]
    return schema_to_prompt(schema, db_path(db_id, Path(spider_dir)), n_rows)


# --------------------------------------------------------------------------- #
# Leakage check
# --------------------------------------------------------------------------- #
def check_no_leakage(train: list[dict], dev: list[dict]) -> list[dict]:
    """Spider is cross-database: no dev database may appear in training data
    (hard failure). Also returns dev examples whose question+SQL text is identical
    to a training example *on a different database* (e.g. "Count the number of
    documents." -> SELECT count(*) FROM Documents). These are generic questions,
    not leakage, but are reported for transparency."""
    train_dbs = {e["db_id"] for e in train}
    dev_dbs = {e["db_id"] for e in dev}
    overlap = train_dbs & dev_dbs
    assert not overlap, f"dev databases also in train: {sorted(overlap)}"
    key = lambda e: (e["question"].strip().lower(), e["query"].strip().lower())
    train_pairs = {key(e) for e in train}
    return [e for e in dev if key(e) in train_pairs]


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _stats() -> None:
    dev = load_split("dev")
    print(f"dev: {len(dev)} examples, {len({e['db_id'] for e in dev})} databases")
    print("  hardness:", dict(Counter(e["hardness"] for e in dev)))
    if (SPIDER_DIR / "train_spider.json").exists():
        train = load_split("train")
        print(f"train: {len(train)} examples, {len({e['db_id'] for e in train})} databases")
        print("  hardness:", dict(Counter(e["hardness"] for e in train)))
        same_text = check_no_leakage(train, dev)
        print("leakage check: OK (no database shared between train and dev)")
        print(f"  note: {len(same_text)} dev examples have question+SQL text identical to a "
              f"train example on a different database: {[e['idx'] for e in same_text]}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("download")
    d.add_argument("--dest", default=str(DATA_ROOT))
    d.add_argument("--only-dev", action="store_true", help="extract only dev databases (~110 MB)")
    d.add_argument("--zip", default=None, help="path to an existing spider_data.zip")
    sub.add_parser("stats")
    s = sub.add_parser("show")
    s.add_argument("db_id")
    s.add_argument("--rows", type=int, default=3)
    args = p.parse_args()

    if args.cmd == "download":
        download_spider(args.dest, only_dev=args.only_dev, zip_path=args.zip, keep_zip=args.zip is not None)
    elif args.cmd == "stats":
        _stats()
    elif args.cmd == "show":
        print(get_schema_prompt(args.db_id, args.rows))


if __name__ == "__main__":
    main()
