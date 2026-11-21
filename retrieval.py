"""
retrieval.py — Few-shot example retrieval for NL2SQL.

Two retrieval strategies:
  - RandomRetriever:  samples k examples uniformly at random (baseline).
  - SimilarRetriever: ranks by cosine similarity of sentence embeddings
                      (sentence-transformers, all-MiniLM-L6-v2 by default).

Both share the same interface so experiments can swap them with one line.

Spider train format expected in train_spider.json:
  [{"question": "...", "query": "...", "db_id": "..."}, ...]
"""

import json
import random
from pathlib import Path
from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Example:
    question: str
    query: str      # gold SQL
    db_id: str


# ---------------------------------------------------------------------------
# Shared loader
# ---------------------------------------------------------------------------

def load_spider_train(train_path: str | Path) -> list[Example]:
    """
    Load Spider train_spider.json and return a flat list of Examples.

    Args:
        train_path: Path to train_spider.json.

    Returns:
        List of Example objects.
    """
    with open(train_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    examples = [
        Example(
            question=item["question"],
            query=item["query"],
            db_id=item["db_id"],
        )
        for item in raw
    ]
    print(f"[retrieval] Loaded {len(examples)} training examples from {train_path}")
    return examples


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class BaseRetriever:
    """Common interface for all retrievers."""

    def __init__(self, examples: list[Example]):
        self.examples = examples

    def retrieve(self, question: str, k: int, db_id: str | None = None) -> list[Example]:
        """
        Return k examples to use as few-shot context for `question`.

        Args:
            question: The natural language query to answer.
            k:        Number of examples to return.
            db_id:    If provided, exclude examples from the same database
                      to prevent the model from seeing the exact schema.

        Returns:
            List of up to k Example objects.
        """
        raise NotImplementedError

    def _filter_pool(self, db_id: str | None) -> list[Example]:
        """Return the candidate pool, optionally excluding the target db."""
        if db_id is None:
            return self.examples
        return [e for e in self.examples if e.db_id != db_id]


# ---------------------------------------------------------------------------
# Random retriever
# ---------------------------------------------------------------------------

class RandomRetriever(BaseRetriever):
    """Uniform random sampling — baseline for few-shot experiments."""

    def __init__(self, examples: list[Example], seed: int = 42):
        super().__init__(examples)
        self._rng = random.Random(seed)

    def retrieve(self, question: str, k: int, db_id: str | None = None) -> list[Example]:
        if k <= 0:
            raise ValueError(f"k must be a positive integer, got {k}")
        pool = self._filter_pool(db_id)
        k = min(k, len(pool))
        return self._rng.sample(pool, k)


# ---------------------------------------------------------------------------
# Similarity-based retriever
# ---------------------------------------------------------------------------

class SimilarRetriever(BaseRetriever):
    """
    Retrieves examples whose questions are most similar to the query,
    measured by cosine similarity of sentence embeddings.

    Embeddings are computed once at construction time and cached as a
    normalised numpy matrix for fast dot-product search.

    Args:
        examples:    Training examples to index.
        model_name:  sentence-transformers model to use.
        batch_size:  Encoding batch size (tune for VRAM/RAM).
        cache_path:  Optional .npy file to save/load pre-computed embeddings.
    """

    DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(
        self,
        examples: list[Example],
        model_name: str = DEFAULT_MODEL,
        batch_size: int = 64,
        cache_path: str | Path | None = None,
    ):
        super().__init__(examples)
        self._model_name = model_name
        self._encoder = self._load_encoder(model_name)
        self._embeddings = self._build_index(batch_size, cache_path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_encoder(model_name: str):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "sentence-transformers is required for SimilarRetriever.\n"
                "Install it with:  pip install sentence-transformers"
            ) from e
        print(f"[retrieval] Loading encoder: {model_name}")
        return SentenceTransformer(model_name)

    def _build_index(
        self,
        batch_size: int,
        cache_path: str | Path | None,
    ) -> np.ndarray:
        """
        Encode all training questions into a normalised embedding matrix.
        Shape: (N, dim).  Loads from cache_path if it exists.
        """
        meta_path: Path | None = None
        if cache_path is not None:
            cache_path = Path(cache_path)
            meta_path = cache_path.with_name(cache_path.stem + ".meta.json")
            if cache_path.exists():
                print(f"[retrieval] Loading embeddings from cache: {cache_path}")
                emb = np.load(cache_path)
                valid = emb.shape[0] == len(self.examples)
                if valid and meta_path.exists():
                    meta = json.loads(meta_path.read_text())
                    if meta.get("model_name") != self._model_name:
                        print(
                            f"[retrieval] Cache model mismatch "
                            f"(cached={meta.get('model_name')!r}, "
                            f"current={self._model_name!r}) — recomputing."
                        )
                        valid = False
                if valid:
                    return emb
                if emb.shape[0] != len(self.examples):
                    print("[retrieval] Cache size mismatch — recomputing.")

        print(f"[retrieval] Encoding {len(self.examples)} questions …")
        questions = [e.question for e in self.examples]
        emb = self._encoder.encode(
            questions,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,   # unit-norm → cosine sim = dot product
        )

        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache_path, emb)
            meta_path.write_text(
                json.dumps({"model_name": self._model_name, "n_examples": len(self.examples)}, indent=2)
            )
            print(f"[retrieval] Embeddings saved to {cache_path}")

        return emb  # shape (N, dim), already normalised

    def _encode_query(self, question: str) -> np.ndarray:
        """Encode a single question; returns shape (dim,)."""
        return self._encoder.encode(
            [question],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )[0]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(self, question: str, k: int, db_id: str | None = None) -> list[Example]:
        """
        Return the k most similar training examples to `question`.

        If db_id is given, examples from that database are excluded first,
        then similarity is computed only over the remaining pool.
        """
        if k <= 0:
            raise ValueError(f"k must be a positive integer, got {k}")

        pool_indices = (
            [i for i, e in enumerate(self.examples) if e.db_id != db_id]
            if db_id is not None
            else list(range(len(self.examples)))
        )

        if not pool_indices:
            return []

        pool_emb = self._embeddings[pool_indices]          # (M, dim)
        query_emb = self._encode_query(question)            # (dim,)
        scores = pool_emb @ query_emb                       # cosine sim, (M,)

        k = min(k, len(pool_indices))
        top_local = np.argpartition(scores, -k)[-k:]
        top_local = top_local[np.argsort(scores[top_local])[::-1]]  # descending

        return [self.examples[pool_indices[i]] for i in top_local]


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------

def format_few_shot_context(examples: list[Example]) -> str:
    """
    Format retrieved examples as a few-shot block to prepend to a prompt.

    Each example is rendered as:
        -- Example N
        -- Question: <question>
        SELECT ...

    Returns an empty string when examples is empty (zero-shot path).
    """
    if not examples:
        return ""

    parts = []
    for i, ex in enumerate(examples, 1):
        parts.append(
            f"-- Example {i}\n"
            f"-- Question: {ex.question}\n"
            f"{ex.query.strip()}"
        )
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Quick smoke-test (run: python retrieval.py --train path/to/train_spider.json)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Retrieval smoke-test")
    parser.add_argument("--train", required=True, help="Path to train_spider.json")
    parser.add_argument("--cache", default=None, help="Path to save/load embeddings (.npy)")
    parser.add_argument("--k", type=int, default=3, help="Examples to retrieve")
    parser.add_argument("--strategy", choices=["random", "similar"], default="similar")
    args = parser.parse_args()

    examples = load_spider_train(args.train)

    if args.strategy == "random":
        retriever = RandomRetriever(examples)
    else:
        retriever = SimilarRetriever(examples, cache_path=args.cache)

    test_questions = [
        ("How many singers are there?", "concert_singer"),
        ("What is the average age of all students?", "student_transcripts_tracking"),
        ("List the names of all employees who work in department 5.", "hr_1"),
    ]

    for question, db_id in test_questions:
        retrieved = retriever.retrieve(question, k=args.k, db_id=db_id)
        ctx = format_few_shot_context(retrieved)
        print(f"\n{'='*60}")
        print(f"Q: {question}  (db: {db_id})")
        print(f"--- Retrieved {len(retrieved)} examples ---")
        print(ctx)
