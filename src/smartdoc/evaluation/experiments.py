
import json
import time
from pathlib import Path

from .dataset import EvaluationQuestion, validate_chunk_ids
from .metrics import hit_at_k, precision_at_k, recall_at_k
from ..retrieval.retriever import Retriever
from ..retrieval.vector import VectorStore


STRATEGIES = ("vector", "keyword", "hybrid")


def evaluate_retrieval(
    retriever: Retriever,
    dataset: list[EvaluationQuestion],
    k: int = 3,
) -> dict:
    """Calculate average retrieval performance."""

    hits = []
    precisions = []
    recalls = []
    latencies = []

    for item in dataset:
        start = time.perf_counter()
        results = retriever.search(item.question, k)
        latency = time.perf_counter() - start

        retrieved_ids = [
            document.metadata["chunk_id"]
            for document, _ in results
        ]

        hits.append(
            hit_at_k(retrieved_ids, item.relevant_chunk_ids, k)
        )
        precisions.append(
            precision_at_k(retrieved_ids, item.relevant_chunk_ids, k)
        )
        recalls.append(
            recall_at_k(retrieved_ids, item.relevant_chunk_ids, k)
        )
        latencies.append(latency)

    count = len(dataset)

    return {
        "hit_at_k": sum(hits) / count,
        "precision_at_k": sum(precisions) / count,
        "recall_at_k": sum(recalls) / count,
        "latency_ms": (sum(latencies) / count) * 1000,
    }


def run_experiments(
    vector_store: VectorStore,
    dataset: list[EvaluationQuestion],
    k: int = 3,
) -> dict:
    """Compare retrieval strategies using summary metrics only."""

    available_chunk_ids = [
        document.metadata["chunk_id"]
        for document in vector_store.get_all_documents()
    ]
    validate_chunk_ids(dataset, available_chunk_ids)

    comparison = {}

    for strategy in STRATEGIES:
        retriever = Retriever(
            strategy=strategy,
            vector_store=vector_store,
        )

        metrics = evaluate_retrieval(retriever, dataset, k)

        comparison[strategy] = {
            "hit_at_k": round(metrics["hit_at_k"], 4),
            "precision_at_k": round(metrics["precision_at_k"], 4),
            "recall_at_k": round(metrics["recall_at_k"], 4),
            "latency_ms": round(metrics["latency_ms"], 3),
        }

    return {
        "top_k": k,
        "total_questions": len(dataset),
        "comparison": comparison,
    }


def save_results(results: dict, path: Path) -> None:
    """Save the simple comparison report as JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )
