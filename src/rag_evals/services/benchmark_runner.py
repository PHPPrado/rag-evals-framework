"""BenchmarkRunner: motor de execução assíncrona do benchmark de regressão.

- Dispara a chamada ao RAG para cada caso em paralelo via `asyncio.gather`
  (com semáforo para respeitar `max_concurrency`).
- Executa os evaluators de cada caso também em paralelo.
- Consolida métricas (médias por evaluator, overall ponderado, latência média
  e p95) e avalia o gate de regressão contra o threshold.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import Sequence

from rag_evals.adapters.rag_client import RAGClient
from rag_evals.config import Settings, settings as default_settings
from rag_evals.domain.evaluators import BaseEvaluator
from rag_evals.domain.models import (
    BenchmarkMetrics,
    BenchmarkReport,
    CaseResult,
    GateStatus,
    TestCase,
)
from rag_evals.services.evaluators import (
    ExactMatchEvaluator,
    FaithfulnessEvaluator,
    SemanticSimilarityEvaluator,
)


def _percentile(values: list[float], pct: float) -> float:
    """Percentil por interpolação linear (pct em [0, 100]). Lista vazia → 0.0."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (pct / 100.0) * (len(ordered) - 1)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    frac = rank - low
    return ordered[low] * (1 - frac) + ordered[high] * frac


def default_evaluators(embedder=None) -> list[BaseEvaluator]:
    """Conjunto padrão de evaluators: exact_match, semantic_similarity, faithfulness."""
    semantic = SemanticSimilarityEvaluator(embedder=embedder) if embedder else SemanticSimilarityEvaluator()
    return [ExactMatchEvaluator(), semantic, FaithfulnessEvaluator()]


class BenchmarkRunner:
    """Motor que orquestra dataset → RAG → evaluators → relatório consolidado."""

    def __init__(
        self,
        rag_client: RAGClient,
        evaluators: Sequence[BaseEvaluator] | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.rag_client = rag_client
        self.settings = settings or default_settings
        self.evaluators = list(evaluators) if evaluators else default_evaluators()

    async def _evaluate_case(
        self,
        case: TestCase,
        semaphore: asyncio.Semaphore,
    ) -> CaseResult:
        async with semaphore:
            started = time.perf_counter()
            response = await self.rag_client.answer(case.question)
            latency_ms = (time.perf_counter() - started) * 1000.0

            scores = await asyncio.gather(
                *(
                    ev.evaluate(
                        question=case.question,
                        generated_answer=response.answer,
                        expected_answer=case.expected_answer,
                        context=case.reference_context,
                    )
                    for ev in self.evaluators
                )
            )

        passed = all(s.passed for s in scores)
        return CaseResult(
            case_id=case.id,
            question=case.question,
            expected_answer=case.expected_answer,
            generated_answer=response.answer,
            latency_ms=round(latency_ms, 3),
            scores=list(scores),
            passed=passed,
        )

    async def run(
        self,
        dataset: Sequence[TestCase],
        *,
        threshold: float | None = None,
        dataset_path: str = "<in-memory>",
        max_concurrency: int | None = None,
    ) -> BenchmarkReport:
        """Executa o benchmark completo e retorna o relatório com o gate avaliado."""
        if not dataset:
            raise ValueError("Dataset vazio: nada a avaliar.")

        effective_threshold = self.settings.threshold if threshold is None else threshold
        concurrency = max_concurrency or self.settings.max_concurrency
        semaphore = asyncio.Semaphore(concurrency)

        started_at = BenchmarkReport.utcnow()
        wall_start = time.perf_counter()
        results = await asyncio.gather(
            *(self._evaluate_case(case, semaphore) for case in dataset)
        )
        wall_ms = (time.perf_counter() - wall_start) * 1000.0
        finished_at = BenchmarkReport.utcnow()

        metrics = self._aggregate(list(results))
        gate = GateStatus.PASSED if metrics.overall_score >= effective_threshold else GateStatus.FAILED

        return BenchmarkReport(
            run_id=uuid.uuid4().hex[:12],
            started_at=started_at,
            finished_at=finished_at,
            duration_ms=round(wall_ms, 3),
            dataset_path=str(dataset_path),
            threshold=effective_threshold,
            gate_status=gate,
            metrics=metrics,
            results=list(results),
        )

    def _aggregate(self, results: list[CaseResult]) -> BenchmarkMetrics:
        """Consolida médias por evaluator, score ponderado e estatísticas de latência."""
        n = len(results)

        def mean_of(evaluator_name: str) -> float:
            values = [
                s.score
                for r in results
                for s in r.scores
                if s.evaluator == evaluator_name
            ]
            return sum(values) / len(values) if values else 0.0

        accuracy = mean_of("exact_match")
        relevance = mean_of("semantic_similarity")
        faithfulness = mean_of("faithfulness")

        w_acc, w_rel, w_fai = self.settings.weight_vector
        overall = w_acc * accuracy + w_rel * relevance + w_fai * faithfulness

        latencies = [r.latency_ms for r in results]
        latency_mean = sum(latencies) / n if n else 0.0

        return BenchmarkMetrics(
            total_cases=n,
            passed_cases=sum(1 for r in results if r.passed),
            accuracy_mean=round(accuracy, 4),
            relevance_mean=round(relevance, 4),
            faithfulness_mean=round(faithfulness, 4),
            overall_score=round(overall, 4),
            latency_mean_ms=round(latency_mean, 3),
            latency_p95_ms=round(_percentile(latencies, 95.0), 3),
        )
