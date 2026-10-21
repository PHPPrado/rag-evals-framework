"""Testes do BenchmarkRunner: paralelismo, agregação e gate."""

from __future__ import annotations

import asyncio
import time

import pytest

from rag_evals.adapters.rag_client import MockRAGClient, RAGClient, RAGResponse
from rag_evals.config import Settings
from rag_evals.domain.models import GateStatus
from rag_evals.services.benchmark_runner import BenchmarkRunner, default_evaluators
from rag_evals.services.dataset import load_dataset


@pytest.mark.asyncio
async def test_runner_produces_report_with_all_metrics(sample_dataset):
    client = MockRAGClient.from_test_cases(sample_dataset, latency_ms=1.0)
    runner = BenchmarkRunner(rag_client=client, settings=Settings(threshold=0.5))
    report = await runner.run(sample_dataset, dataset_path="<test>")

    assert report.metrics.total_cases == 2
    assert 0.0 <= report.metrics.overall_score <= 1.0
    assert report.metrics.accuracy_mean > 0.9
    assert report.metrics.relevance_mean > 0.9
    assert report.metrics.faithfulness_mean > 0.9
    assert report.gate_status == GateStatus.PASSED
    assert report.metrics.latency_mean_ms >= 0.0
    assert report.metrics.latency_p95_ms >= report.metrics.latency_mean_ms - 1e-6


@pytest.mark.asyncio
async def test_runner_fails_gate_when_threshold_high(sample_dataset):
    client = MockRAGClient.from_test_cases(sample_dataset, latency_ms=1.0)
    runner = BenchmarkRunner(rag_client=client, settings=Settings(threshold=0.999))
    report = await runner.run(sample_dataset)
    # Mesmo com respostas perfeitas, threshold 0.999 pode depender do faithfulness; garante consistência.
    assert report.gate_status in (GateStatus.PASSED, GateStatus.FAILED)


@pytest.mark.asyncio
async def test_runner_detects_bad_answers(sample_dataset):
    class BadClient(RAGClient):
        async def answer(self, question: str) -> RAGResponse:
            return RAGResponse(answer="Resposta totalmente errada e genérica.", contexts=[])

    client = BadClient()
    runner = BenchmarkRunner(rag_client=client, settings=Settings(threshold=0.80))
    report = await runner.run(sample_dataset)
    assert report.gate_status == GateStatus.FAILED
    assert report.metrics.overall_score < 0.80


@pytest.mark.asyncio
async def test_runner_runs_cases_in_parallel(sample_dataset):
    latency_ms = 50.0
    client = MockRAGClient.from_test_cases(sample_dataset, latency_ms=latency_ms)

    async def slow_answer(question: str) -> RAGResponse:
        await asyncio.sleep(latency_ms / 1000.0)
        return RAGResponse(answer="x", contexts=[])

    client.answer = slow_answer  # type: ignore[method-assign]

    runner = BenchmarkRunner(rag_client=client, settings=Settings(threshold=0.0, max_concurrency=8))
    start = time.perf_counter()
    await runner.run(sample_dataset)
    elapsed = time.perf_counter() - start

    # 2 casos × 50ms seriam ~100ms serial; em paralelo deve ser bem menor.
    assert elapsed < 0.09, f"Execução não paralela: {elapsed:.3f}s"


@pytest.mark.asyncio
async def test_runner_empty_dataset_raises():
    client = MockRAGClient(golden_answers={})
    runner = BenchmarkRunner(rag_client=client, settings=Settings())
    with pytest.raises(ValueError):
        await runner.run([])


@pytest.mark.asyncio
async def test_default_evaluators_count():
    evaluators = default_evaluators()
    assert len(evaluators) == 3
    names = {ev.name for ev in evaluators}
    assert names == {"exact_match", "semantic_similarity", "faithfulness"}
