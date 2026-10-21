"""Teste-gate de regressão semântica — executado por `run_evals.py` no CI/CD.

Este teste roda o benchmark completo sobre o dataset real com o pipeline mockado
(determinístico) e **falha** se o score consolidado consolidado cair abaixo do
threshold (`RAG_EVAL_THRESHOLD`, padrão 0.80). É o que dá exit code 1 no CI.

Variáveis de ambiente suportadas:
    RAG_EVAL_DATASET_PATH   Caminho do dataset (default: data/eval_dataset.json)
    RAG_EVAL_THRESHOLD      Score mínimo (default: 0.80)
    RAG_EVAL_REPORT_PATH    Se definido, salva o relatório JSON neste caminho
    RAG_EVAL_MAX_CONCURRENCY Paralelismo (default: 8)
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from rag_evals.adapters.embedding_providers import LocalDeterministicEmbedder
from rag_evals.adapters.rag_client import MockRAGClient
from rag_evals.config import PROJECT_ROOT, Settings
from rag_evals.domain.models import GateStatus
from rag_evals.services.benchmark_runner import BenchmarkRunner, default_evaluators
from rag_evals.services.dataset import load_dataset


@pytest.mark.asyncio
async def test_semantic_regression_gate():
    dataset_path = Path(os.environ.get("RAG_EVAL_DATASET_PATH", PROJECT_ROOT / "data" / "eval_dataset.json"))
    threshold = float(os.environ.get("RAG_EVAL_THRESHOLD", "0.80"))
    max_concurrency = int(os.environ.get("RAG_EVAL_MAX_CONCURRENCY", "8"))

    dataset = load_dataset(dataset_path)
    assert len(dataset) >= 3, "Dataset deve ter ao menos 3 casos de teste."

    client = MockRAGClient.from_test_cases(dataset)
    runner = BenchmarkRunner(
        rag_client=client,
        evaluators=default_evaluators(embedder=LocalDeterministicEmbedder()),
        settings=Settings(threshold=threshold, max_concurrency=max_concurrency),
    )
    report = await runner.run(dataset, threshold=threshold, dataset_path=str(dataset_path))

    report_path = os.environ.get("RAG_EVAL_REPORT_PATH")
    if report_path:
        path = Path(report_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(report.model_dump_json(indent=2), encoding="utf-8")

    m = report.metrics
    print("\n--- Benchmark Report ---")
    print(f"overall_score={m.overall_score:.4f} (threshold={threshold:.2f})")
    print(f"accuracy={m.accuracy_mean:.4f} relevance={m.relevance_mean:.4f} faithfulness={m.faithfulness_mean:.4f}")
    print(f"latency_mean={m.latency_mean_ms:.2f}ms p95={m.latency_p95_ms:.2f}ms")

    assert report.gate_status == GateStatus.PASSED, (
        f"REGRESSION GATE FALHOU: overall_score={m.overall_score:.4f} < threshold={threshold:.2f}. "
        f"Accuracy={m.accuracy_mean:.4f}, Relevance={m.relevance_mean:.4f}, Faithfulness={m.faithfulness_mean:.4f}"
    )
